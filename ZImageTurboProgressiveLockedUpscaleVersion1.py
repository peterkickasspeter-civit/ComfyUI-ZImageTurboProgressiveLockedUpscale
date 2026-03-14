import logging
from contextlib import contextmanager

import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview

LOGGER = logging.getLogger(__name__)
_MISSING = object()

try:
    import nvvfx  # type: ignore
    _HAS_NVVFX = True
except Exception:
    nvvfx = None
    _HAS_NVVFX = False


# -------------------------
# Utilities: stage schedule
# -------------------------

def _progressive_scales(total_scale: float, max_step: float):
    if total_scale <= 1.0:
        return []
    s = 1.0
    out = []
    while s < total_scale - 1e-6:
        s = min(s * max_step, total_scale)
        out.append(s)
    return out


def _stage_visibility_scale(stage_index: int, stage_count: int):
    """
    Keep the cumulative pre-sampler depth separation bounded regardless of how many
    progressive upscales we take. More stages should not silently mean more total
    depth bias and more compounding.
    """
    if stage_count <= 1:
        return 0.35

    weights = []
    for j in range(stage_count):
        t = (j + 1) / float(stage_count)
        weights.append(0.55 + 0.45 * (t ** 1.8))

    total_budget = 0.50
    return total_budget * weights[stage_index] / max(sum(weights), 1e-8)


def _round_latent_hw(h, w):
    # Even latent sizes are safer for VAE pipelines.
    h = max(2, (h // 2) * 2)
    w = max(2, (w // 2) * 2)
    return h, w


def _image_to_bchw(img_bhwc: torch.Tensor) -> torch.Tensor:
    if img_bhwc.dim() == 3:
        img_bhwc = img_bhwc.unsqueeze(0)
    return img_bhwc.movedim(-1, 1)


def _bchw_to_image(img_bchw: torch.Tensor) -> torch.Tensor:
    return img_bchw.movedim(1, -1)


def _interp_bchw(x: torch.Tensor, size, mode: str) -> torch.Tensor:
    if mode in ("nearest", "area"):
        return F.interpolate(x, size=size, mode=mode)
    return F.interpolate(x, size=size, mode=mode, align_corners=False)


def _resize_image_bicubic(img_bhwc: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    x = _image_to_bchw(img_bhwc.float())
    x = _interp_bchw(x, size=(target_h, target_w), mode="bicubic")
    return torch.clamp(_bchw_to_image(x), 0.0, 1.0)


def _match_batch(x, target_b: int):
    if x is None:
        return None
    if x.shape[0] == target_b:
        return x
    if x.shape[0] == 1:
        return x.expand(target_b, -1, -1, -1)
    return x[:1].expand(target_b, -1, -1, -1)


@contextmanager
def _temporary_model_transformer_options(model, updates: dict):
    if not hasattr(model, "model_options") or not isinstance(getattr(model, "model_options"), dict):
        yield
        return

    model_options = model.model_options
    had_transformer_options = "transformer_options" in model_options
    transformer_options = model_options.setdefault("transformer_options", {})
    prior = {}
    for key, value in updates.items():
        prior[key] = transformer_options.get(key, _MISSING)
        transformer_options[key] = value

    try:
        yield
    finally:
        for key, value in prior.items():
            if value is _MISSING:
                transformer_options.pop(key, None)
            else:
                transformer_options[key] = value

        if not had_transformer_options and len(transformer_options) == 0:
            model_options.pop("transformer_options", None)


# -------------------------
# Smooth optical kernels
# -------------------------

_GAUSS_CACHE = {}


def _get_gaussian_weights(channels: int, device, dtype):
    key = (channels, str(device), str(dtype))
    if key in _GAUSS_CACHE:
        return _GAUSS_CACHE[key]

    k = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], device=device, dtype=dtype)
    k = k / k.sum()
    kh = k.view(1, 1, 1, 5).repeat(channels, 1, 1, 1)
    kv = k.view(1, 1, 5, 1).repeat(channels, 1, 1, 1)
    _GAUSS_CACHE[key] = (kh, kv)
    return kh, kv


def _gaussian_blur_bchw(x: torch.Tensor, passes: int = 1) -> torch.Tensor:
    if passes <= 0:
        return x
    c = x.shape[1]
    kh, kv = _get_gaussian_weights(c, x.device, x.dtype)
    out = x
    for _ in range(passes):
        out = F.pad(out, (2, 2, 0, 0), mode="reflect")
        out = F.conv2d(out, kh, groups=c)
        out = F.pad(out, (0, 0, 2, 2), mode="reflect")
        out = F.conv2d(out, kv, groups=c)
    return out


def _rgb_luma(x_bchw: torch.Tensor) -> torch.Tensor:
    return 0.2126 * x_bchw[:, 0:1] + 0.7152 * x_bchw[:, 1:2] + 0.0722 * x_bchw[:, 2:3]


def _smoothstep(x: torch.Tensor, edge0: float, edge1: float) -> torch.Tensor:
    denom = max(edge1 - edge0, 1e-6)
    t = torch.clamp((x - edge0) / denom, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _edge_strength_bchw(x: torch.Tensor, gain: float = 6.0, blur_passes: int = 1) -> torch.Tensor:
    dx = F.pad(x[:, :, :, 1:] - x[:, :, :, :-1], (0, 1, 0, 0), mode="replicate")
    dy = F.pad(x[:, :, 1:, :] - x[:, :, :-1, :], (0, 0, 0, 1), mode="replicate")
    e = 0.5 * (dx.abs() + dy.abs())
    if blur_passes > 0:
        e = _gaussian_blur_bchw(e, passes=blur_passes)
    return torch.clamp(e * gain, 0.0, 1.0)


def _mild_depth_recontrast(near_bchw: torch.Tensor) -> torch.Tensor:
    """
    Depth maps that arrive with a very compressed dynamic range tend to produce weak,
    mushy subject/background separation. Expand them only a little, and only when the
    distribution is obviously flat.
    """
    if near_bchw is None:
        return None

    b, _, h, w = near_bchw.shape
    stat_h = max(8, min(128, h))
    stat_w = max(8, min(128, w))
    stats = _interp_bchw(near_bchw.float(), (stat_h, stat_w), mode="area")
    flat = stats.flatten(2)

    lo = torch.quantile(flat, 0.05, dim=2).view(b, 1, 1, 1)
    hi = torch.quantile(flat, 0.95, dim=2).view(b, 1, 1, 1)
    span = torch.clamp(hi - lo, min=1e-4)

    blend = torch.clamp((0.50 - span) / 0.30, 0.0, 1.0) * 0.30
    norm = torch.clamp((near_bchw - lo) / span, 0.0, 1.0)
    return torch.clamp(near_bchw * (1.0 - blend) + norm * blend, 0.0, 1.0)


def _refine_near_with_guidance(near_bchw: torch.Tensor, guidance_img_bhwc=None, transition_band=None):
    """
    Refine the near map with image guidance so depth smoothing follows actual scene
    boundaries instead of blindly blurring across them.
    """
    if near_bchw is None:
        return None

    near = torch.clamp(near_bchw.float(), 0.0, 1.0)
    near1 = _gaussian_blur_bchw(near, passes=1)
    near2 = _gaussian_blur_bchw(near1, passes=1)

    blend = torch.full_like(near, 0.34)

    if guidance_img_bhwc is not None:
        guide = guidance_img_bhwc.float()
        if guide.dim() == 3:
            guide = guide.unsqueeze(0)
        guide = guide[..., :3].to(device=near.device, dtype=near.dtype)
        guide = _image_to_bchw(guide)
        guide = _match_batch(guide, near.shape[0])
        if guide.shape[-2:] != near.shape[-2:]:
            guide = _interp_bchw(guide, near.shape[-2:], mode="bilinear")

        guide_y = _rgb_luma(guide)
        img_edge = _edge_strength_bchw(guide_y, gain=5.0, blur_passes=1)
        flat = 1.0 - img_edge
        blend = 0.18 + 0.42 * flat

    depth_edge = _edge_strength_bchw(near, gain=7.0, blur_passes=1)
    blend = blend * (1.0 - 0.50 * depth_edge)

    if transition_band is not None:
        edge = transition_band.to(device=near.device, dtype=near.dtype)
        edge = _match_batch(edge, near.shape[0])
        if edge.shape[-2:] != near.shape[-2:]:
            edge = _interp_bchw(edge, near.shape[-2:], mode="bilinear")
        blend = torch.maximum(blend, 0.24 * edge)

    near = near * (1.0 - 0.55 * blend) + near1 * (0.35 * blend) + near2 * (0.20 * blend)
    return torch.clamp(near, 0.0, 1.0)


def _local_depth_salience(near_bchw: torch.Tensor):
    """
    Absolute depth alone is not enough. A subject also needs to read as closer than its
    local neighborhood. This relative-depth signal is what creates pop without requiring
    a hard layer split.
    """
    if near_bchw is None:
        return None, None

    local = _gaussian_blur_bchw(near_bchw, passes=3)
    rel = near_bchw - local
    forward = torch.clamp(rel * 4.0, 0.0, 1.0)
    backward = torch.clamp(-rel * 4.0, 0.0, 1.0)

    forward = _gaussian_blur_bchw(forward, passes=1)
    backward = _gaussian_blur_bchw(backward, passes=1)
    return forward, backward


# -------------------------
# RTX / fallback resize
# -------------------------

def _upscale_image_with_rtx_vsr(img_bhwc: torch.Tensor, target_h: int, target_w: int, quality: str):
    if not _HAS_NVVFX or not torch.cuda.is_available():
        return None

    output_width = max(8, round(target_w / 8) * 8)
    output_height = max(8, round(target_h / 8) * 8)

    quality_mapping = {
        "LOW": nvvfx.effects.QualityLevel.LOW,
        "MEDIUM": nvvfx.effects.QualityLevel.MEDIUM,
        "HIGH": nvvfx.effects.QualityLevel.HIGH,
        "ULTRA": nvvfx.effects.QualityLevel.ULTRA,
    }
    selected_quality = quality_mapping.get(quality, nvvfx.effects.QualityLevel.ULTRA)

    max_pixels = 1024 * 1024 * 16
    out_pixels = max(1, output_width * output_height)
    batch_size = max(1, max_pixels // out_pixels)
    upscaled_batches = []

    try:
        with nvvfx.VideoSuperRes(selected_quality) as sr:
            sr.output_width = output_width
            sr.output_height = output_height
            sr.load()

            for i in range(0, img_bhwc.shape[0], batch_size):
                batch = img_bhwc[i:i + batch_size]
                batch_cuda = batch.cuda().permute(0, 3, 1, 2).contiguous()

                batch_outputs = []
                for j in range(batch_cuda.shape[0]):
                    input_frame = batch_cuda[j]
                    dlpack_out = sr.run(input_frame).image
                    output = torch.from_dlpack(dlpack_out).clone()
                    batch_outputs.append(output)

                batch_out_tensor = torch.stack(batch_outputs, dim=0)
                batch_out_tensor = batch_out_tensor.permute(0, 2, 3, 1).contiguous().cpu()
                upscaled_batches.append(batch_out_tensor)

        final_images = torch.cat(upscaled_batches, dim=0)
        if output_height != target_h or output_width != target_w:
            final_images = _resize_image_bicubic(final_images, target_h, target_w)
        return torch.clamp(final_images, 0.0, 1.0)
    except Exception as exc:
        LOGGER.warning("RTX VSR failed, falling back to bicubic resize: %s", exc)
        return None


def _resize_image_prefer_rtx(img_bhwc: torch.Tensor, target_h: int, target_w: int, rtx_quality: str) -> torch.Tensor:
    rtx = _upscale_image_with_rtx_vsr(img_bhwc, target_h, target_w, rtx_quality)
    if rtx is not None:
        return rtx
    return _resize_image_bicubic(img_bhwc, target_h, target_w)


# -----------------------------------------
# Orthogonal subspace locking for arbitrary
# (H,W) -> (H',W') using separable partitions
# -----------------------------------------

_PARTITION_CACHE = {}


def _build_partition_map(low_n: int, high_n: int, device):
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
    inv_sqrt = (counts.float().rsqrt())[map_hi_to_lo]

    _PARTITION_CACHE[key] = (map_hi_to_lo, inv_sqrt, counts)
    return map_hi_to_lo, inv_sqrt, counts


def _reduce_height(x, map_h, inv_sqrt_h, low_h):
    b, c, hh, w = x.shape
    out = torch.zeros((b, c, low_h, w), device=x.device, dtype=x.dtype)
    weighted = x * inv_sqrt_h.view(1, 1, hh, 1)
    out.index_add_(2, map_h, weighted)
    return out


def _expand_height(coeff, map_h, inv_sqrt_h):
    hh = map_h.shape[0]
    return coeff.index_select(2, map_h) * inv_sqrt_h.view(1, 1, hh, 1)


def _reduce_width(x, map_w, inv_sqrt_w, low_w):
    b, c, h, ww = x.shape
    out = torch.zeros((b, c, h, low_w), device=x.device, dtype=x.dtype)
    weighted = x * inv_sqrt_w.view(1, 1, 1, ww)
    out.index_add_(3, map_w, weighted)
    return out


def _expand_width(coeff, map_w, inv_sqrt_w):
    ww = map_w.shape[0]
    return coeff.index_select(3, map_w) * inv_sqrt_w.view(1, 1, 1, ww)


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
    # Keep the orthogonal Gaussian construction untouched.
    device = eps_prev.device
    dtype = eps_prev.dtype
    b, c, h1, w1 = target_shape
    h0, w0 = eps_prev.shape[-2], eps_prev.shape[-1]

    g = torch.Generator(device=device)
    g.manual_seed(seed_new)
    eta = torch.randn((b, c, h1, w1), generator=g, device=device, dtype=dtype)

    proj = _project_to_coarse_subspace(eta, h0, w0, h1, w1, device)
    eta_perp = eta - proj
    lifted = _lift_noise(eps_prev, h1, w1)
    eps_new = lifted + eta_perp
    return eps_new


# -------------------------
# Depth / mask preparation
# -------------------------

def _prepare_depth_map(depth_map, target_h: int, target_w: int):
    if depth_map is None:
        return None

    d = depth_map
    if d.dim() == 2:
        d = d.unsqueeze(0).unsqueeze(-1)
    elif d.dim() == 3:
        if d.shape[-1] in (1, 3, 4):
            d = d.unsqueeze(0)
        else:
            d = d.unsqueeze(1)
    elif d.dim() != 4:
        raise ValueError(f"Unsupported depth map shape: {tuple(d.shape)}")

    if d.dim() == 4 and d.shape[1] in (1, 3, 4) and d.shape[-1] not in (1, 3, 4):
        # Already BCHW.
        d = d[:, :1]
    else:
        # Treat as BHWC image. Use the first channel directly.
        d = d[..., :1]
        d = _image_to_bchw(d)

    d = _interp_bchw(d.float(), (target_h, target_w), mode="bicubic")
    d = torch.clamp(d, 0.0, 1.0)
    d = _gaussian_blur_bchw(d, passes=1)
    return d


def _prepare_subject_mask(subject_mask, target_h: int, target_w: int):
    if subject_mask is None:
        return None

    m = subject_mask
    if m.dim() == 2:
        m = m.unsqueeze(0).unsqueeze(0)
    elif m.dim() == 3:
        # Prefer MASK semantics (B, H, W); allow image-like fallback.
        if m.shape[-1] in (1, 3, 4):
            m = _image_to_bchw(m.unsqueeze(0)[..., :1])
        else:
            m = m.unsqueeze(1)
    elif m.dim() == 4:
        if m.shape[1] == 1:
            pass
        elif m.shape[-1] in (1, 3, 4):
            m = _image_to_bchw(m[..., :1])
        else:
            raise ValueError(f"Unsupported subject mask shape: {tuple(m.shape)}")
    else:
        raise ValueError(f"Unsupported subject mask shape: {tuple(m.shape)}")

    m = _interp_bchw(m.float(), (target_h, target_w), mode="bilinear")
    m = torch.clamp(m, 0.0, 1.0)
    m = _gaussian_blur_bchw(m, passes=2)
    return m


def _subject_edge_band(mask_bchw: torch.Tensor) -> torch.Tensor:
    """
    Turn a full subject mask into a soft edge-only guidance band.

    This keeps the mask from globally biasing the whole subject "near" and instead
    uses it only to feather depth transitions at the subject boundary.
    """
    if mask_bchw is None:
        return None

    dilated = F.max_pool2d(mask_bchw, kernel_size=7, stride=1, padding=3)
    eroded = -F.max_pool2d(-mask_bchw, kernel_size=7, stride=1, padding=3)
    edge = torch.clamp(dilated - eroded, 0.0, 1.0)
    edge = _gaussian_blur_bchw(edge, passes=2)
    return torch.clamp(edge * 2.0, 0.0, 1.0)


def _build_near_map(
    depth_map=None,
    subject_mask=None,
    target_h: int = 0,
    target_w: int = 0,
    return_edge: bool = False,
    guidance_img=None,
):
    depth_bchw = _prepare_depth_map(depth_map, target_h, target_w) if depth_map is not None else None
    if depth_bchw is None:
        return (None, None) if return_edge else None

    near = _mild_depth_recontrast(depth_bchw)
    edge_out = None
    if subject_mask is not None:
        mask_bchw = _prepare_subject_mask(subject_mask, target_h, target_w)
        if mask_bchw is not None:
            target_b = max(near.shape[0], mask_bchw.shape[0])
            near = _match_batch(near, target_b)
            mask_bchw = _match_batch(mask_bchw, target_b)

            # The mask is only a transition guide. Do not lift the entire subject "near",
            # which can make the subject read as a different processed layer.
            edge_out = _subject_edge_band(mask_bchw)

            # Use the subject edge to stabilize the handoff, but keep the interior governed
            # by depth rather than by a hard matte.
            near_smooth = _gaussian_blur_bchw(near, passes=2)
            mix = 0.50 * edge_out
            near = near * (1.0 - mix) + near_smooth * mix

    near = _refine_near_with_guidance(near, guidance_img_bhwc=guidance_img, transition_band=edge_out)
    near = _gaussian_blur_bchw(torch.clamp(near, 0.0, 1.0), passes=1)
    near = torch.clamp(near, 0.0, 1.0)

    if return_edge:
        if edge_out is not None:
            edge_out = _gaussian_blur_bchw(torch.clamp(edge_out, 0.0, 1.0), passes=1)
            edge_out = torch.clamp(edge_out, 0.0, 1.0)
        return near, edge_out
    return near


# -------------------------
# Depth-aware visibility transfer
# -------------------------

def _prepare_near_for_image(near_map, x_bchw):
    if near_map is None:
        return None
    near = near_map.to(device=x_bchw.device, dtype=x_bchw.dtype)
    near = _match_batch(near, x_bchw.shape[0])
    if near.shape[-2:] != x_bchw.shape[-2:]:
        near = _interp_bchw(near, x_bchw.shape[-2:], mode="bilinear")
    return torch.clamp(near, 0.0, 1.0)


def _apply_depth_visibility_transfer(
    img_bhwc: torch.Tensor,
    near_map,
    depth_strength: float,
    strength_scale: float,
    transition_band=None,
):
    """
    A depth-shaped visibility transfer that uses more than one signal:

    - absolute depth bands for atmospheric recession,
    - relative local depth salience for foreground pop,
    - image/depth edge protection so the transfer does not flatten or halo.

    The orthogonal Gaussian noise path is still untouched; only the decoded image state
    is adjusted.
    """
    if near_map is None or depth_strength <= 1e-6:
        return img_bhwc

    x = _image_to_bchw(img_bhwc.float())
    near = _prepare_near_for_image(near_map, x)
    edge = _prepare_near_for_image(transition_band, x) if transition_band is not None else None

    if near is None:
        return img_bhwc

    y = _rgb_luma(x)
    y1 = _gaussian_blur_bchw(y, passes=1)
    y2 = _gaussian_blur_bchw(y1, passes=1)

    low = y2
    mid = y1 - y2
    fine = y - y1

    far = 1.0 - near

    # Soft stratified depth bands are more expressive than a single linear far mask.
    near_focus = _smoothstep(near, 0.58, 0.90)
    far_field = _smoothstep(far, 0.28, 0.82)
    deep_far = _smoothstep(far, 0.60, 0.96)

    # Relative local depth lets a subject read as closer than its surroundings without
    # requiring us to hard-cut it into a separate plate.
    local_pop, local_sink = _local_depth_salience(near)

    depth_edge = _edge_strength_bchw(near, gain=7.0, blur_passes=1)
    img_edge = _edge_strength_bchw(y, gain=5.0, blur_passes=1)
    detail_guard = torch.clamp((fine.abs() / 0.055) + (mid.abs() / 0.10), 0.0, 1.0)

    protect = 0.38 * depth_edge + 0.22 * img_edge + 0.18 * detail_guard
    if edge is not None:
        protect = torch.maximum(protect, 0.55 * edge)
    protect = torch.clamp(protect, 0.0, 0.82)

    far_detail = far_field * (1.0 - 0.60 * protect)
    far_sink = torch.clamp(0.75 * far_field + 0.25 * local_sink, 0.0, 1.0)
    far_sink = far_sink * (1.0 - 0.55 * protect)
    far_deep = deep_far * (1.0 - 0.72 * protect)

    # Keep the subject-pop component mostly for the stronger final pass rather than
    # compounding it through all progressive stages.
    pop_scale = max(0.0, min(1.0, (strength_scale - 0.32) / 0.58))
    near_pop = near_focus * (0.35 + 0.65 * local_pop) * (1.0 - 0.30 * img_edge)

    mid_gain = 1.0 - (0.16 * depth_strength * strength_scale) * far_sink
    fine_gain = 1.0 - (0.28 * depth_strength * strength_scale) * far_detail

    if pop_scale > 0.0:
        mid_gain = mid_gain + (0.038 * depth_strength * pop_scale) * near_pop
        fine_gain = fine_gain + (0.022 * depth_strength * pop_scale) * near_pop

    mid_gain = torch.clamp(mid_gain, 0.70, 1.15)
    fine_gain = torch.clamp(fine_gain, 0.55, 1.12)

    w = torch.clamp(far_field + 0.05, 0.05, 1.05)
    air_y = (low * w).sum(dim=(-2, -1), keepdim=True) / torch.clamp(w.sum(dim=(-2, -1), keepdim=True), min=1e-6)
    air_amt = (0.040 * depth_strength * strength_scale) * far_deep
    low = low * (1.0 - air_amt) + air_y * air_amt

    y_new = low + mid * mid_gain + fine * fine_gain
    x = x + (y_new - y)

    # Slight chroma rolloff only in the deeper background. Keep it off protected edges.
    y_after = _rgb_luma(x)
    desat = (0.022 * depth_strength * strength_scale) * far_deep * (1.0 - 0.45 * protect)
    x = y_after + (x - y_after) * (1.0 - desat)

    return torch.clamp(_bchw_to_image(x), 0.0, 1.0)


# -------------------------
# Final ISP finish
# -------------------------

def _apply_isp_finish(img_bhwc: torch.Tensor, isp_strength: float, near_map=None):
    if isp_strength <= 1e-6:
        return img_bhwc

    x = _image_to_bchw(img_bhwc.float())
    near = None
    if near_map is not None:
        near = _prepare_near_for_image(near_map, x)

    if near is None:
        near = torch.full_like(_rgb_luma(x), 0.7)

    y = _rgb_luma(x)
    y1 = _gaussian_blur_bchw(y, passes=1)
    y2 = _gaussian_blur_bchw(y1, passes=1)

    low = y2
    mid = y1 - y2
    fine = y - y1

    # Smartphone-like tone shaping on low frequencies.
    high = torch.clamp((low - 0.72) / 0.28, 0.0, 1.0)
    low = low - (0.045 * isp_strength) * high * high

    shadow = torch.clamp((0.28 - low) / 0.28, 0.0, 1.0)
    low = low + (0.015 * isp_strength) * shadow * shadow

    # Local contrast / clarity on luminance only. Far regions still receive some boost,
    # but less than near regions so depth separation remains believable.
    mid_clarity = 0.55 + 0.45 * near
    fine_clarity = 0.60 + 0.40 * near

    mid_gain = 1.0 + (0.160 * isp_strength) * mid_clarity
    fine_gain = 1.0 + (0.070 * isp_strength) * fine_clarity

    y_new = low + mid * mid_gain + fine * fine_gain
    x = x + (y_new - y)

    # Tiny highlight chroma cleanup to keep the finish stable.
    y_after = _rgb_luma(x)
    hi = torch.clamp((y_after - 0.75) / 0.25, 0.0, 1.0)
    x = y_after + (x - y_after) * (1.0 - 0.030 * isp_strength * hi)

    return torch.clamp(_bchw_to_image(x), 0.0, 1.0)


# -------------------------
# Sampling with step slicing
# -------------------------

def _sample_with_noise_and_slice(model, latent, noise, positive, negative,
                                 steps, cfg, sampler_name, scheduler,
                                 start_step, seed):
    device = comfy.model_management.get_torch_device()
    x0 = latent.to(device)
    eps = noise.to(device=device, dtype=x0.dtype)

    callback = latent_preview.prepare_callback(model, max(1, steps))
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
        seed=seed,
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

                "steps": ("INT", {"default": 9, "min": 1, "max": 99}),
                "sampler": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "beta57"}),

                "tail_steps_first_upscale": ("INT", {"default": 6, "min": 1, "max": 24}),
                "tail_steps_last_upscale": ("INT", {"default": 5, "min": 1, "max": 24}),

                "depth_strength": ("FLOAT", {
                    "default": 0.35, "min": 0.0, "max": 1.5, "step": 0.01,
                    "tooltip": "Depth-driven visibility rolloff. Depth is used directly as white=near, black=far/background."
                }),
                "isp_strength": ("FLOAT", {
                    "default": 0.25, "min": 0.0, "max": 1.5, "step": 0.01,
                    "tooltip": "Final smartphone-like ISP finish: tone shoulder, shadow lift, and luminance-only local contrast shaping."
                }),
                "rtx_quality": (["LOW", "MEDIUM", "HIGH", "ULTRA"], {
                    "default": "ULTRA",
                    "tooltip": "Uses NVIDIA Video Super Resolution when nvvfx is installed; otherwise falls back to bicubic."
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": {
                "depth_map": ("IMAGE", {
                    "tooltip": "Optional depth map used directly as white=near, black=far/background. No inversion or auto-remapping is applied."
                }),
                "subject_mask": ("MASK", {
                    "tooltip": "Optional white-on-black subject boundary guide. It only smooths how the rolloff crosses the subject; it is not a hard isolation matte."
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
                depth_strength, isp_strength, rtx_quality, seed, vae,
                depth_map=None, subject_mask=None):

        cfg = 1.0
        device = comfy.model_management.get_torch_device()

        x = latent["samples"].to(device)
        b, c, h0, w0 = x.shape

        scales = _progressive_scales(upscale_factor, max_step_scale)
        n_up = len(scales)

        g0 = torch.Generator(device=device)
        g0.manual_seed(seed)
        eps = torch.randn(x.shape, generator=g0, device=device, dtype=x.dtype)

        did_upscale = False

        for i, s in enumerate(scales):
            did_upscale = True
            ht = int(round(h0 * s))
            wt = int(round(w0 * s))
            ht, wt = _round_latent_hw(ht, wt)

            target_px_h = ht * 8
            target_px_w = wt * 8
            stage_frac = 1.0 if n_up <= 1 else (i / float(max(n_up - 1, 1)))

            img = vae.decode(x.to("cpu"))
            img = _resize_image_prefer_rtx(img, target_px_h, target_px_w, rtx_quality)

            near_px = None
            edge_px = None
            if depth_map is not None:
                near_px, edge_px = _build_near_map(
                    depth_map=depth_map,
                    subject_mask=subject_mask,
                    target_h=target_px_h,
                    target_w=target_px_w,
                    return_edge=True,
                    guidance_img=img,
                )

            if near_px is not None and depth_strength > 1e-6:
                stage_visibility = _stage_visibility_scale(i, n_up)
                img = _apply_depth_visibility_transfer(
                    img,
                    near_px,
                    depth_strength,
                    strength_scale=stage_visibility,
                    transition_band=edge_px,
                )

            x = vae.encode(img[:, :, :, :3]).to(device)

            stage_seed_new = seed + 10007 * (i + 1)
            eps = _locked_noise_from_prev(eps, (b, c, ht, wt), stage_seed_new)

            if n_up <= 1:
                tail = max(1, min(steps, tail_steps_first_upscale))
            else:
                t = i / (n_up - 1)
                tail_f = tail_steps_first_upscale + (tail_steps_last_upscale - tail_steps_first_upscale) * t
                tail = int(round(tail_f))
                tail = max(1, min(steps, tail))

            start_step = max(0, steps - tail)
            stage_metadata = {
                "zimage_global_steps": int(steps),
                "zimage_start_step": int(start_step),
                "zimage_stage_index": int(i),
                "zimage_stage_count": int(max(n_up, 1)),
                "zimage_is_final_stage": bool(i == (n_up - 1)),
                "zimage_tail_steps": int(tail),
            }

            with _temporary_model_transformer_options(model, stage_metadata):
                x = _sample_with_noise_and_slice(
                    model, x, eps, positive, negative,
                    steps, cfg, sampler, scheduler,
                    start_step=start_step, seed=seed + 20011 * (i + 1),
                )

        final_img = None
        if did_upscale and (depth_strength > 1e-6 or isp_strength > 1e-6):
            final_img = vae.decode(x.to("cpu"))
            final_near = None
            final_edge = None
            if depth_map is not None:
                final_near, final_edge = _build_near_map(
                    depth_map=depth_map,
                    subject_mask=subject_mask,
                    target_h=final_img.shape[1],
                    target_w=final_img.shape[2],
                    return_edge=True,
                    guidance_img=final_img,
                )

            if final_near is not None and depth_strength > 1e-6:
                final_img = _apply_depth_visibility_transfer(
                    final_img,
                    final_near,
                    depth_strength,
                    strength_scale=0.90,
                    transition_band=final_edge,
                )

            if isp_strength > 1e-6:
                final_img = _apply_isp_finish(final_img, isp_strength, near_map=final_near)

            x = vae.encode(final_img[:, :, :, :3]).to(device)

        out_latent = {"samples": x.to("cpu")}
        if final_img is not None:
            # Return the finished pixels directly. Re-decoding the newly encoded latent
            # adds another VAE round trip and can visibly soften skin texture and eyes.
            out_img = torch.clamp(final_img[:, :, :, :3].to("cpu"), 0.0, 1.0)
        else:
            out_img = vae.decode(out_latent["samples"])
        return (out_latent, out_img, seed)


NODE_CLASS_MAPPINGS = {
    "ZImageTurboProgressiveLockedUpscale": ZImageTurboProgressiveLockedUpscale,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZImageTurboProgressiveLockedUpscale": "Z-Image: Locked Noise + Depth-Aware Upscale",
}
