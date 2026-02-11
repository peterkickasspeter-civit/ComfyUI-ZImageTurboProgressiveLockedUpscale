import torch
import torch.nn.functional as F
import comfy.samplers
import comfy.sample
import comfy.model_management
import comfy.utils
import latent_preview

# -------------------------
# Utilities: stage schedule
# -------------------------

def _progressive_scales(total_scale: float, max_step: float):
    if total_scale <= 1.0:
        return [1.0]
    s = 1.0
    out = []
    while s < total_scale - 1e-6:
        s = min(s * max_step, total_scale)
        out.append(s)
    return out

def _round_latent_hw(h, w):
    # Even latent sizes are safer for VAE pipelines
    h = max(2, (h // 2) * 2)
    w = max(2, (w // 2) * 2)
    return h, w

def _resize_image_bicubic(img_bhwc, target_h, target_w):
    x = img_bhwc.movedim(-1, 1)  # BCHW
    x = F.interpolate(x, size=(target_h, target_w), mode="bicubic", align_corners=False)
    x = x.movedim(1, -1)
    return torch.clamp(x, 0.0, 1.0)

# -------------------------
# Upscale model support
# -------------------------

def _upscale_image_with_model(img_bhwc, upscale_model, target_h, target_w, device):
    """Upscale image using a pixel-space upscale model (e.g. RealESRGAN), then resize to exact target."""
    upscale_model.to(device)
    try:
        in_img = img_bhwc.movedim(-1, 1).to(device)  # BCHW
        tile = 512
        overlap = 32
        oom = True
        while oom:
            try:
                steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(
                    in_img.shape[3], in_img.shape[2], tile_x=tile, tile_y=tile, overlap=overlap)
                pbar = comfy.utils.ProgressBar(steps)
                s = comfy.utils.tiled_scale(
                    in_img, lambda a: upscale_model(a),
                    tile_x=tile, tile_y=tile, overlap=overlap,
                    upscale_amount=upscale_model.scale, pbar=pbar)
                oom = False
            except comfy.model_management.OOM_EXCEPTION:
                tile //= 2
                if tile < 128:
                    raise
    finally:
        upscale_model.to(comfy.model_management.vae_offload_device())

    current_h, current_w = s.shape[2], s.shape[3]
    if current_h != target_h or current_w != target_w:
        s = F.interpolate(s, size=(target_h, target_w), mode="bicubic", align_corners=False)
    s = s.movedim(1, -1)  # BHWC
    return torch.clamp(s, 0.0, 1.0)

# -----------------------------------------
# Orthogonal subspace locking for arbitrary
# (H,W) -> (H',W') using separable partitions
# -----------------------------------------

_PARTITION_CACHE = {}

def _build_partition_map(low_n: int, high_n: int, device):
    """
    Build a deterministic contiguous partition of [0, high_n) into low_n bins.
    Returns:
      map_hi_to_lo: Long[high_n] where each high index maps to a coarse bin
      inv_sqrt_count_hi: Float[high_n] containing 1/sqrt(count_of_bin(map[i]))
      low_counts: Long[low_n] bin sizes
    """
    key = (low_n, high_n, device.type)
    if key in _PARTITION_CACHE:
        return _PARTITION_CACHE[key]

    if high_n < low_n:
        raise ValueError(f"Partition requires high_n >= low_n, got {high_n} < {low_n}")

    base = high_n // low_n
    rem = high_n % low_n

    counts = torch.full((low_n,), base, dtype=torch.long, device=device)
    if rem > 0:
        counts[:rem] += 1

    map_hi_to_lo = torch.repeat_interleave(torch.arange(low_n, device=device), counts)
    inv_sqrt = (counts.float().rsqrt())[map_hi_to_lo]  # length high_n

    _PARTITION_CACHE[key] = (map_hi_to_lo, inv_sqrt, counts)
    return map_hi_to_lo, inv_sqrt, counts

def _reduce_height(x, map_h, inv_sqrt_h, low_h):
    B, C, Hh, W = x.shape
    out = torch.zeros((B, C, low_h, W), device=x.device, dtype=x.dtype)
    weighted = x * inv_sqrt_h.view(1, 1, Hh, 1)
    out.index_add_(2, map_h, weighted)
    return out

def _expand_height(coeff, map_h, inv_sqrt_h):
    Hh = map_h.shape[0]
    out = coeff.index_select(2, map_h) * inv_sqrt_h.view(1, 1, Hh, 1)
    return out

def _reduce_width(x, map_w, inv_sqrt_w, low_w):
    B, C, H, Ww = x.shape
    out = torch.zeros((B, C, H, low_w), device=x.device, dtype=x.dtype)
    weighted = x * inv_sqrt_w.view(1, 1, 1, Ww)
    out.index_add_(3, map_w, weighted)
    return out

def _expand_width(coeff, map_w, inv_sqrt_w):
    Ww = map_w.shape[0]
    out = coeff.index_select(3, map_w) * inv_sqrt_w.view(1, 1, 1, Ww)
    return out

def _project_to_coarse_subspace(x, low_h, low_w, high_h, high_w, device):
    map_h, inv_h, _ = _build_partition_map(low_h, high_h, device)
    map_w, inv_w, _ = _build_partition_map(low_w, high_w, device)

    tmp = _reduce_width(x, map_w, inv_w, low_w)
    coeff = _reduce_height(tmp, map_h, inv_h, low_h)

    recon = _expand_height(coeff, map_h, inv_h)
    recon = _expand_width(recon, map_w, inv_w)
    return recon

def _lift_noise(eps_prev, high_h, high_w):
    device = eps_prev.device
    low_h, low_w = eps_prev.shape[-2], eps_prev.shape[-1]
    map_h, inv_h, _ = _build_partition_map(low_h, high_h, device)
    map_w, inv_w, _ = _build_partition_map(low_w, high_w, device)

    out = _expand_height(eps_prev, map_h, inv_h)
    out = _expand_width(out, map_w, inv_w)
    return out

def _locked_noise_from_prev(eps_prev, target_shape, seed_new):
    device = eps_prev.device
    dtype = eps_prev.dtype
    B, C, H1, W1 = target_shape
    H0, W0 = eps_prev.shape[-2], eps_prev.shape[-1]

    g = torch.Generator(device=device)
    g.manual_seed(seed_new)
    eta = torch.randn((B, C, H1, W1), generator=g, device=device, dtype=dtype)

    proj = _project_to_coarse_subspace(eta, H0, W0, H1, W1, device)
    eta_perp = eta - proj

    lifted = _lift_noise(eps_prev, H1, W1)

    eps_new = lifted + eta_perp
    return eps_new

# -------------------------
# Sigma schedule utilities
# -------------------------

def _get_sigma_schedule(model, sampler_name, scheduler, steps):
    """Compute the full sigma schedule for the given model/sampler/scheduler/steps."""
    try:
        model_sampling = model.get_model_object("model_sampling")
        discard = sampler_name in comfy.samplers.KSampler.DISCARD_PENULTIMATE_SIGMA_SAMPLERS
        sigmas = comfy.samplers.calculate_sigmas(model_sampling, scheduler, steps + (1 if discard else 0))
        if discard:
            sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])
        return sigmas
    except Exception:
        return None

def _slice_sigmas_at_entry(sigmas, enter_sigma):
    """
    Given a full sigma schedule, find the entry point closest to enter_sigma
    and return the tail from that point onward.

    Returns:
      sliced_sigmas: the tail sigma tensor starting at or below enter_sigma
      start_index: the index into the original schedule where we start
    """
    if sigmas is None or len(sigmas) < 2:
        return sigmas, 0

    # Find the first sigma value <= enter_sigma (schedule is descending)
    for i in range(len(sigmas) - 1):
        if sigmas[i].item() <= enter_sigma:
            return sigmas[i:], i

    # If enter_sigma is larger than all sigmas, return the full schedule
    return sigmas, 0

# -------------------------
# Sampling with step slicing
# -------------------------

def _sample_with_noise_and_slice(model, latent, noise, positive, negative,
                                 steps, cfg, sampler_name, scheduler,
                                 start_step, seed):
    device = comfy.model_management.get_torch_device()
    x0 = latent.to(device)
    eps = noise.to(device=device, dtype=x0.dtype)

    callback = latent_preview.prepare_callback(model, steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

    out = comfy.sample.sample(
        model, eps, steps, cfg,
        sampler_name, scheduler, positive, negative,
        x0,
        denoise=1.0,
        disable_noise=False,
        start_step=start_step,
        last_step=None,
        force_full_denoise=True,
        noise_mask=None,
        callback=callback,
        disable_pbar=disable_pbar,
        seed=seed
    )
    return out

def _sample_custom_with_sigmas(model, latent, noise, positive, negative,
                               sigmas, sampler_name, seed):
    """Sample using explicit sigma schedule via sample_custom."""
    device = comfy.model_management.get_torch_device()
    x0 = latent.to(device)
    eps = noise.to(device=device, dtype=x0.dtype)

    num_steps = len(sigmas) - 1  # sigmas has N+1 entries for N steps
    callback = latent_preview.prepare_callback(model, num_steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

    sampler_obj = comfy.samplers.sampler_object(sampler_name)

    out = comfy.sample.sample_custom(
        model, eps, 1.0, sampler_obj, sigmas,
        positive, negative, x0,
        noise_mask=None,
        callback=callback,
        disable_pbar=disable_pbar,
        seed=seed
    )
    return out

# -------------------------
# Node
# -------------------------

class ZImageTurboProgressiveLockedUpscale:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "latent": ("LATENT",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),

                "upscale_factor": ("FLOAT", {"default": 6.0, "min": 1.0, "max": 24.0, "step": 0.25}),
                "max_step_scale": ("FLOAT", {"default": 1.6, "min": 1.1, "max": 6.0, "step": 0.05}),

                "steps": ("INT", {"default": 9, "min": 4, "max": 99}),
                "sampler": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "beta57"}),

                "tail_steps_first_upscale": ("INT", {"default": 6, "min": 1, "max": 12}),
                "tail_steps_last_upscale": ("INT", {"default": 3, "min": 1, "max": 12}),

                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": {
                "upscale_model": ("UPSCALE_MODEL", {
                    "tooltip": "Optional pixel-space upscale model (e.g. RealESRGAN). "
                               "Used for resizing between upscale stages instead of bicubic. Requires VAE."
                }),
                "refine_model": ("MODEL", {
                    "tooltip": "Model for the final refinement stage after all upscale stages complete."
                }),
                "refine_sampler": (comfy.samplers.KSampler.SAMPLERS, {
                    "default": "euler",
                    "tooltip": "Sampler for the refinement stage."
                }),
                "refine_scheduler": (comfy.samplers.KSampler.SCHEDULERS, {
                    "default": "normal",
                    "tooltip": "Scheduler for the refinement stage."
                }),
                "refine_steps": ("INT", {
                    "default": 9, "min": 1, "max": 99,
                    "tooltip": "Steps to compute the sigma schedule for refinement. "
                               "Actual executed steps will be fewer based on refine_enter_sigma."
                }),
                "refine_enter_sigma": ("FLOAT", {
                    "default": 0.6, "min": 0.01, "max": 15.0, "step": 0.05,
                    "tooltip": "Sigma value at which to enter the refinement schedule. "
                               "Lower values = less denoising = more preservation of existing detail."
                }),
            }
        }

    RETURN_TYPES = ("LATENT", "IMAGE", "INT")
    RETURN_NAMES = ("latent", "image", "seed")
    FUNCTION = "process"
    CATEGORY = "Z-Image"

    def process(self, model, latent, positive, negative,
                upscale_factor, max_step_scale,
                steps, sampler, scheduler,
                tail_steps_first_upscale, tail_steps_last_upscale,
                seed, vae,
                upscale_model=None,
                refine_model=None,
                refine_sampler="euler",
                refine_scheduler="normal",
                refine_steps=9,
                refine_enter_sigma=0.6):

        cfg = 1.0
        base_full_steps_if_empty = True
        refine_base_if_not_empty = False
        device = comfy.model_management.get_torch_device()

        x = latent["samples"].to(device)
        B, C, H0, W0 = x.shape

        # "Empty latent" detection (keep it strict)
        empty = (x.abs().mean() < 1e-4) and (x.std() < 1e-4)

        scales = _progressive_scales(upscale_factor, max_step_scale)
        n_up = len(scales)

        # initialize base noise at current resolution (preserved DOFs start here)
        g0 = torch.Generator(device=device)
        g0.manual_seed(seed)
        eps = torch.randn(x.shape, generator=g0, device=device, dtype=x.dtype)

        # Optional base refinement / generation
        if empty and base_full_steps_if_empty:
            start_step = 0
            x = _sample_with_noise_and_slice(
                model, x, eps, positive, negative,
                steps, cfg, sampler, scheduler,
                start_step=start_step, seed=seed
            )
        elif (not empty) and refine_base_if_not_empty:
            tail = max(1, min(steps, tail_steps_first_upscale))
            start_step = max(0, steps - tail)
            x = _sample_with_noise_and_slice(
                model, x, eps, positive, negative,
                steps, cfg, sampler, scheduler,
                start_step=start_step, seed=seed
            )

        # Progressive upscales
        for i, s in enumerate(scales):
            Ht = int(round(H0 * s))
            Wt = int(round(W0 * s))
            Ht, Wt = _round_latent_hw(Ht, Wt)

            # Lift deterministic state (latent) via pixel-space if possible
            if vae is not None:
                img = vae.decode(x.to("cpu"))
                if upscale_model is not None:
                    img = _upscale_image_with_model(img, upscale_model,
                                                    Ht * 8, Wt * 8, device)
                else:
                    img = _resize_image_bicubic(img, Ht * 8, Wt * 8)
                x = vae.encode(img[:, :, :, :3]).to(device)
            else:
                x = F.interpolate(x, size=(Ht, Wt), mode="bicubic", align_corners=False)

            # Lift + refresh noise orthogonally
            stage_seed_new = seed + 10007 * (i + 1)
            eps = _locked_noise_from_prev(eps, (B, C, Ht, Wt), stage_seed_new)

            # Sigma/time slicing: later stages use fewer tail steps
            if n_up <= 1:
                tail = max(1, min(steps, tail_steps_first_upscale))
            else:
                t = i / (n_up - 1)
                tail_f = tail_steps_first_upscale + (tail_steps_last_upscale - tail_steps_first_upscale) * t
                tail = int(round(tail_f))
                tail = max(1, min(steps, tail))

            start_step = max(0, steps - tail)

            x = _sample_with_noise_and_slice(
                model, x, eps, positive, negative,
                steps, cfg, sampler, scheduler,
                start_step=start_step, seed=seed + 20011 * (i + 1)
            )

        # -------------------------
        # Refinement stage
        # -------------------------
        if refine_model is not None:
            refine_seed = seed + 99991

            # Compute sigma schedule for refine_steps using refine model/sampler/scheduler
            full_sigmas = _get_sigma_schedule(refine_model, refine_sampler,
                                              refine_scheduler, refine_steps)

            if full_sigmas is not None and len(full_sigmas) >= 2:
                # Slice at the entry sigma
                sliced_sigmas, start_idx = _slice_sigmas_at_entry(full_sigmas, refine_enter_sigma)

                if sliced_sigmas is not None and len(sliced_sigmas) >= 2:
                    # Generate fresh noise for the refine stage at current resolution
                    _, _, Hf, Wf = x.shape
                    g_refine = torch.Generator(device=device)
                    g_refine.manual_seed(refine_seed)
                    refine_noise = torch.randn(x.shape, generator=g_refine,
                                               device=device, dtype=x.dtype)

                    x = _sample_custom_with_sigmas(
                        refine_model, x, refine_noise, positive, negative,
                        sliced_sigmas, refine_sampler, seed=refine_seed
                    )

        out_latent = {"samples": x.to("cpu")}
        if vae is not None:
            out_img = vae.decode(out_latent["samples"])
        else:
            out_img = torch.zeros(1, 8, 8, 3)

        return (out_latent, out_img, seed)

NODE_CLASS_MAPPINGS = {
    "ZImageTurboProgressiveLockedUpscale": ZImageTurboProgressiveLockedUpscale,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZImageTurboProgressiveLockedUpscale": "Z-Image: Locked Noise + Sigma-Sliced Upscale",
}