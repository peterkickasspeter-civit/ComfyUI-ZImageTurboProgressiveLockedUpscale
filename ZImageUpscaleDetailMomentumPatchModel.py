
import logging
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

LOGGER = logging.getLogger(__name__)

_WEAK_FLAG = "zimage_token_detail_guidance_weak_pass"
_WEAK_STATE_KEY = f"{_WEAK_FLAG}_state"


@dataclass(frozen=True)
class DetailGuidanceSpec:
    start_layer: int          # inclusive
    end_layer: int            # exclusive
    step_start: float         # 0 = earliest/noisiest, 1 = latest/cleanest
    step_end: float
    guidance_scale: float
    weaken_strength: float    # 0 = no weakening, 1 = remove all local detail in selected band
    blur_passes: int
    output_rescale: float     # 0 = disabled, 1 = full std-match back to the normal pass
    adaptive_gate: bool


# -------------------------
# Core helpers
# -------------------------

def _identity_wrapper(apply_model, args):
    return apply_model(args["input"], args["timestep"], **args["c"])


def _get_diffusion_transformer(model_patcher):
    if not hasattr(model_patcher, "model"):
        raise ValueError("Expected a ComfyUI MODEL patcher object with a .model attribute.")
    base_model = model_patcher.model
    if not hasattr(base_model, "diffusion_model"):
        raise ValueError("This MODEL does not expose .model.diffusion_model and cannot be patched.")
    return base_model.diffusion_model


def _extract_transformer_options(args):
    c = args.get("c", {}) or {}
    return c.get("transformer_options", {}) or {}


def _extract_sample_sigmas(args) -> Optional[torch.Tensor]:
    transformer_options = _extract_transformer_options(args)
    sigmas = transformer_options.get("sample_sigmas")
    if sigmas is None:
        return None
    if torch.is_tensor(sigmas):
        return sigmas.flatten()
    try:
        return torch.as_tensor(sigmas).flatten()
    except Exception:
        return None


def _current_step_fraction(args) -> Tuple[Optional[float], Optional[int], Optional[int]]:
    """
    Returns a sampling progress fraction in [0, 1].

    When the sampler executes only a tail slice (start_step > 0), ComfyUI exposes
    only the sliced sigma schedule in transformer_options.sample_sigmas. In that case
    a naive local fraction is *not* the global denoise fraction. We first compute the
    local position inside the executed slice and then, when the caller supplied the
    original schedule metadata, lift it back into global schedule coordinates.
    """
    transformer_options = _extract_transformer_options(args)

    sigmas = _extract_sample_sigmas(args)
    if sigmas is not None and sigmas.numel() > 0:
        if sigmas.numel() == 1:
            local_frac = 0.0
            local_step = 0
            local_steps = 1
        else:
            timestep = args.get("timestep")
            if torch.is_tensor(timestep):
                t = timestep.detach().flatten()[0]
                t = t.to(device=sigmas.device, dtype=sigmas.dtype)
            else:
                t = torch.tensor(float(timestep), device=sigmas.device, dtype=sigmas.dtype)

            sigma_idx = int(torch.argmin(torch.abs(sigmas - t)).item())
            local_steps = max(int(sigmas.numel()) - 1, 1)
            local_step = min(sigma_idx, local_steps - 1)
            local_frac = local_step / float(max(local_steps - 1, 1))

        global_steps = transformer_options.get("zimage_global_steps", None)
        start_step = transformer_options.get("zimage_start_step", 0)
        try:
            global_steps = int(global_steps) if global_steps is not None else None
            start_step = int(start_step)
        except Exception:
            global_steps = None
            start_step = 0

        if global_steps is not None and global_steps > 1:
            global_step = max(0, min(start_step + int(local_step), global_steps - 1))
            global_frac = global_step / float(max(global_steps - 1, 1))
            return global_frac, global_step, global_steps

        return local_frac, int(local_step), int(local_steps)

    current_percent = transformer_options.get("current_percent", None)
    if current_percent is not None:
        try:
            frac = float(current_percent)
            frac = max(0.0, min(1.0, frac))
            return frac, None, None
        except Exception:
            return None, None, None

    return None, None, None


def _fit_layer_band(start_layer: int, end_layer: int, num_layers: int) -> Tuple[int, int]:
    if num_layers <= 0:
        raise ValueError(f"num_layers must be > 0, got {num_layers}")

    requested_start = int(start_layer)
    requested_end = int(end_layer)

    if num_layers <= 4 and (
        requested_start < 0
        or requested_end <= requested_start
        or requested_start >= num_layers
        or requested_end > num_layers
    ):
        LOGGER.warning(
            "Token detail guidance band %d:%d is not valid for tiny stack depth %d; using full stack 0:%d instead.",
            requested_start,
            requested_end,
            num_layers,
            num_layers,
        )
        return 0, num_layers

    start = max(0, min(requested_start, num_layers - 1))
    end = max(start + 1, min(requested_end, num_layers))

    if (start, end) != (requested_start, requested_end):
        LOGGER.warning(
            "Token detail guidance band %d:%d adjusted to %d:%d for stack depth %d.",
            requested_start,
            requested_end,
            start,
            end,
            num_layers,
        )

    return start, end


def _avg_blur_bchw(x: torch.Tensor, passes: int) -> torch.Tensor:
    if passes <= 0:
        return x

    out = x
    for _ in range(int(passes)):
        out = F.pad(out, (1, 1, 1, 1), mode="reflect")
        out = F.avg_pool2d(out, kernel_size=3, stride=1)
    return out


def _rescale_like(reference: torch.Tensor, guided: torch.Tensor, blend: float) -> torch.Tensor:
    if blend <= 1e-6:
        return guided

    ref_mean = reference.mean(dim=(1, 2, 3), keepdim=True)
    guided_mean = guided.mean(dim=(1, 2, 3), keepdim=True)

    ref_center = reference - ref_mean
    guided_center = guided - guided_mean

    ref_std = ref_center.float().flatten(1).std(dim=1, unbiased=False).view(-1, 1, 1, 1)
    guided_std = guided_center.float().flatten(1).std(dim=1, unbiased=False).view(-1, 1, 1, 1)

    scale = torch.ones_like(ref_std)
    valid = guided_std > 1e-6
    scale = torch.where(valid, (ref_std / guided_std).clamp(0.25, 4.0), scale)
    scale = scale.to(device=guided.device, dtype=guided.dtype)

    rescaled = ref_mean + guided_center * scale
    return guided * (1.0 - blend) + rescaled * blend


def _clone_args_with_weak_flag(args, flag_name: str):
    cloned = dict(args)
    c = dict(args.get("c", {}) or {})
    transformer_options = dict(c.get("transformer_options", {}) or {})
    transformer_options[flag_name] = True
    c["transformer_options"] = transformer_options
    cloned["c"] = c
    return cloned


def _limit_delta_like(reference: torch.Tensor, delta: torch.Tensor, max_ratio: float = 0.75) -> torch.Tensor:
    if delta.dim() != 4 or max_ratio <= 0.0:
        return delta

    ref_center = reference - reference.mean(dim=(2, 3), keepdim=True)
    delta_center = delta - delta.mean(dim=(2, 3), keepdim=True)

    ref_std = ref_center.float().flatten(1).std(dim=1, unbiased=False).view(-1, 1, 1, 1)
    delta_std = delta_center.float().flatten(1).std(dim=1, unbiased=False).view(-1, 1, 1, 1)

    allowed = ref_std * max_ratio
    scale = torch.ones_like(ref_std)
    valid = delta_std > 1e-6
    limited = torch.where(valid, torch.clamp(allowed / delta_std, max=1.0), scale)
    limited = limited.to(device=delta.device, dtype=delta.dtype)
    return delta * limited


def _late_detail_factor(step_frac: Optional[float]) -> float:
    if step_frac is None:
        return 0.0
    try:
        sf = float(step_frac)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, (sf - 0.55) / 0.45))


def _reference_texture_gate(reference: torch.Tensor, step_frac: Optional[float] = None) -> torch.Tensor:
    """
    Suppress self-guidance mainly on the strongest semantic edges where it tends to
    create outline halos, but keep much more energy in textured portrait regions.

    The original gate was too eager: in an upscale-tail setting the useful signal is
    already concentrated into a relatively small residual, so aggressively damping it
    on all strong contrast edges sterilizes the very microstructure we were trying to
    recover. This version attenuates only the hardest edges and eases off further in
    the late/clean steps where texture injection matters most.
    """
    if reference.dim() != 4:
        return torch.ones_like(reference)

    late = _late_detail_factor(step_frac)

    lum = reference.float().mean(dim=1, keepdim=True)
    lum_low = _avg_blur_bchw(lum, passes=1)
    edge = torch.abs(lum - lum_low)

    edge_norm = edge.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    edge = torch.clamp(edge / ((3.0 + 0.8 * late) * edge_norm), 0.0, 1.0)

    atten = 0.18 - 0.08 * late
    gate = 1.0 - atten * edge * edge
    return gate.to(device=reference.device, dtype=reference.dtype)


def _shape_guidance_delta(
    reference: torch.Tensor,
    delta: torch.Tensor,
    blur_passes: int = 1,
    step_frac: Optional[float] = None,
) -> torch.Tensor:
    """
    Shape the self-guidance delta for upscale refinement.

    The key correction here is that late-tail upscale refinement needs substantially
    more high-frequency retention than the earlier portrait-friendly variant. The old
    0.85*mid + 0.30*high mix was too conservative and effectively sanded off the very
    detail band the weak-vs-full pass had isolated.
    """
    if delta.dim() != 4:
        return delta

    delta = delta - delta.mean(dim=(2, 3), keepdim=True)

    passes = max(int(blur_passes), 1)
    low = _avg_blur_bchw(delta, passes=passes)
    lower = _avg_blur_bchw(low, passes=passes)

    mid = low - lower
    high = delta - low

    late = _late_detail_factor(step_frac)
    mid_w = 0.70 - 0.20 * late
    high_w = 0.60 + 0.55 * late

    shaped = mid_w * mid + high_w * high
    shaped = shaped * _reference_texture_gate(reference, step_frac=step_frac)
    shaped = _limit_delta_like(reference, shaped, max_ratio=0.60 + 0.10 * late)
    return shaped


# -------------------------
# Weak-pass patch
# -------------------------

class _TokenDetailWeakener:
    """
    Double-block patch that weakens local image-token detail on selected layers,
    but only when the wrapper marks the current pass as a weak/detail-starved pass.

    The second version keeps an anchor to the original image-token input and only
    suppresses the transformer's *added* local detail residual. That is much less
    likely to shift broad color/tonal structure than blurring the absolute token map.
    """

    __slots__ = (
        "start_layer",
        "end_layer",
        "weaken_strength",
        "blur_passes",
        "adaptive_gate",
        "patch_size",
        "flag_name",
    )

    def __init__(
        self,
        start_layer: int,
        end_layer: int,
        weaken_strength: float,
        blur_passes: int,
        adaptive_gate: bool,
        patch_size: int,
        flag_name: str,
    ):
        self.start_layer = int(start_layer)
        self.end_layer = int(end_layer)
        self.weaken_strength = float(weaken_strength)
        self.blur_passes = int(blur_passes)
        self.adaptive_gate = bool(adaptive_gate)
        self.patch_size = max(int(patch_size), 1)
        self.flag_name = str(flag_name)

    def __call__(self, args):
        transformer_options = args.get("transformer_options", {}) or {}
        if not bool(transformer_options.get(self.flag_name, False)):
            return {}

        block_index = args.get("block_index", transformer_options.get("block_index", None))
        if block_index is None:
            return {}
        block_index = int(block_index)

        img = args.get("img", None)
        img_input = args.get("img_input", None)
        x = args.get("x", None)
        if not torch.is_tensor(img) or img.dim() != 3 or img.shape[1] <= 0:
            return {}
        if not torch.is_tensor(x) or x.dim() < 4:
            return {}

        state = transformer_options.get(_WEAK_STATE_KEY, None)
        if not isinstance(state, dict):
            state = {}
            transformer_options[_WEAK_STATE_KEY] = state

        prev_img = state.get("prev_img", None)
        if torch.is_tensor(prev_img) and prev_img.shape == img.shape:
            anchor_tokens = prev_img
        elif torch.is_tensor(img_input) and img_input.shape == img.shape:
            anchor_tokens = img_input
        else:
            anchor_tokens = img

        # Always advance the local anchor state, even when the block is outside the
        # active band. That lets the first selected block anchor against the actual
        # input it received, instead of the stack input from many layers earlier.
        state["prev_img"] = img.detach()

        if not (self.start_layer <= block_index < self.end_layer):
            return {}

        latent_h = int(x.shape[-2])
        latent_w = int(x.shape[-1])
        grid_h = latent_h // self.patch_size
        grid_w = latent_w // self.patch_size
        num_img_tokens = int(img.shape[1])

        if grid_h <= 0 or grid_w <= 0 or grid_h * grid_w != num_img_tokens:
            return {}

        b, n, d = img.shape
        img_grid = img.transpose(1, 2).contiguous().view(b, d, grid_h, grid_w)
        anchor_grid = anchor_tokens.transpose(1, 2).contiguous().view(b, d, grid_h, grid_w)

        # Weaken only the *local* detail residual relative to the effective input of the
        # current block. The previous implementation anchored to the stack input, which
        # makes late-layer weakening regress broad accumulated state instead of isolating
        # genuine late detail contribution.
        residual = img_grid - anchor_grid
        low = _avg_blur_bchw(residual, self.blur_passes)
        detail = residual - low

        if self.adaptive_gate:
            energy = detail.float().pow(2).mean(dim=1, keepdim=True)
            energy_norm = energy.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
            gate = torch.clamp(torch.sqrt(energy / energy_norm), 0.0, 1.0)
            atten = self.weaken_strength * gate.to(device=img_grid.device, dtype=img_grid.dtype)
        else:
            atten = self.weaken_strength

        weakened = anchor_grid + low + detail * (1.0 - atten)
        weakened = weakened.view(b, d, n).transpose(1, 2).contiguous()
        state["prev_img"] = weakened.detach()
        return {"img": weakened}

    def to(self, device):
        return self

    def models(self):
        return []

    def cleanup(self, **kwargs):
        return None


# -------------------------
# Wrapper
# -------------------------

class _DetailGuidanceWrapper:
    """
    Two-pass self-guidance wrapper.

    Full pass:   eps_full = model(x_t)
    Weak pass:   eps_weak = model(x_t) with selected layer-band detail attenuation
    Final:       eps = eps_full + s * shaped(eps_full - eps_weak)

    The shaped() step is where we suppress low-frequency/color drift while keeping
    portrait-friendly texture bands rather than edge-only energy.
    """

    __slots__ = ("_spec", "_prior", "_flag_name", "_warned_missing_sigmas")

    def __init__(self, spec: DetailGuidanceSpec, prior_wrapper, flag_name: str):
        self._spec = spec
        self._prior = prior_wrapper
        self._flag_name = flag_name
        self._warned_missing_sigmas = False

    def __call__(self, apply_model, args):
        step_frac, _, _ = _current_step_fraction(args)
        if step_frac is None:
            if not self._warned_missing_sigmas:
                LOGGER.warning(
                    "ZImageTokenDetailGuidance: sample progress metadata missing; detail guidance is inactive for this sampling path."
                )
                self._warned_missing_sigmas = True
            return self._prior(apply_model, args)

        active = self._spec.step_start <= step_frac <= self._spec.step_end
        if (not active) or self._spec.guidance_scale <= 1e-6 or self._spec.weaken_strength <= 1e-6:
            return self._prior(apply_model, args)

        full = self._prior(apply_model, args)
        weak_args = _clone_args_with_weak_flag(args, self._flag_name)
        weak = self._prior(apply_model, weak_args)

        delta = full - weak
        delta = _shape_guidance_delta(
            full,
            delta,
            blur_passes=max(1, self._spec.blur_passes),
            step_frac=step_frac,
        )

        late = _late_detail_factor(step_frac)
        guided = full + self._spec.guidance_scale * delta

        # Late-tail upscale detail is the one place where aggressive stat matching is
        # actively counterproductive. Keep the stabilizer, but ease it off as sampling
        # approaches the clean end of the schedule so the shaped detail delta can remain
        # visible.
        rescale_blend = self._spec.output_rescale * (1.0 - 0.55 * late)
        guided = _rescale_like(full, guided, rescale_blend)
        return guided

    def to(self, device):
        if hasattr(self._prior, "to"):
            maybe_new_prior = self._prior.to(device)
            if maybe_new_prior is not None:
                self._prior = maybe_new_prior
        return self

    def models(self):
        if hasattr(self._prior, "models"):
            return self._prior.models()
        return []

    def cleanup(self, **kwargs):
        if hasattr(self._prior, "cleanup"):
            return self._prior.cleanup(**kwargs)
        return None

    def __getattr__(self, name):
        return getattr(self._prior, name)

    def __deepcopy__(self, memo):
        copied = type(self)(self._spec, self._prior, self._flag_name)
        copied._warned_missing_sigmas = self._warned_missing_sigmas
        memo[id(self)] = copied
        return copied


# -------------------------
# Node
# -------------------------

class ZImageUpscaleDetailMomentumPatchModel:
    """
    Separate MODEL -> MODEL patch node for upscale refinement.

    Intended use:
      base generation:   unpatched model (or replay-only if you want)
      upscale branch:    model -> this node -> upscale node

    It does not modify weights. It adds:
      1) a double-block patch that can weaken image-token detail on demand, and
      2) a wrapper that runs a normal pass + a weak pass and guides toward the
         normal-vs-weak *detail* difference during selected denoising steps.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enabled": ("BOOLEAN", {"default": True}),
                "start_layer": (
                    "INT",
                    {
                        "default": 18,
                        "min": 0,
                        "max": 255,
                        "step": 1,
                        "tooltip": "Start layer (inclusive) on the main `layers` stack.",
                    },
                ),
                "end_layer": (
                    "INT",
                    {
                        "default": 26,
                        "min": 1,
                        "max": 256,
                        "step": 1,
                        "tooltip": "End layer (exclusive) on the main `layers` stack.",
                    },
                ),
                "step_start": (
                    "FLOAT",
                    {
                        "default": 0.68,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Start step fraction. 0 = earliest/noisiest step, 1 = latest/cleanest step.",
                    },
                ),
                "step_end": (
                    "FLOAT",
                    {
                        "default": 0.98,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "End step fraction for detail guidance.",
                    },
                ),
                "guidance_scale": (
                    "FLOAT",
                    {
                        "default": 0.22,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": "Strength of the shaped detail guidance delta. Start around 0.15-0.30 for upscale refinement.",
                    },
                ),
                "weaken_strength": (
                    "FLOAT",
                    {
                        "default": 0.55,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "How strongly the weak pass suppresses local image-token detail inside the selected band.",
                    },
                ),
                "blur_passes": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 4,
                        "step": 1,
                        "tooltip": "Number of local blur passes used to estimate the weak/detail-starved token state.",
                    },
                ),
                "output_rescale": (
                    "FLOAT",
                    {
                        "default": 0.45,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Stabilizer that rescales the guided output back toward the normal pass statistics.",
                    },
                ),
                "adaptive_gate": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Only weaken detail strongly where the local hidden-state detail energy is high. Keeps flat regions more stable.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "Z-Image/experimental"

    def patch(
        self,
        model,
        enabled,
        start_layer,
        end_layer,
        step_start,
        step_end,
        guidance_scale,
        weaken_strength,
        blur_passes,
        output_rescale,
        adaptive_gate,
    ):
        if not enabled:
            return (model,)

        if not (0.0 <= float(step_start) <= 1.0 and 0.0 <= float(step_end) <= 1.0):
            raise ValueError(f"step_start/step_end must be within [0,1], got {step_start}..{step_end}")
        if float(step_start) > float(step_end):
            raise ValueError(f"step_start must be <= step_end, got {step_start} > {step_end}")
        if int(blur_passes) < 1:
            raise ValueError(f"blur_passes must be >= 1, got {blur_passes}")

        patched = model.clone()
        transformer = _get_diffusion_transformer(patched)

        if not hasattr(transformer, "layers"):
            raise ValueError(
                "ZImageTokenDetailGuidancePatchModel currently supports architectures exposing a main `layers` transformer stack."
            )

        num_layers = len(transformer.layers)
        resolved_start, resolved_end = _fit_layer_band(start_layer, end_layer, num_layers)

        spec = DetailGuidanceSpec(
            start_layer=resolved_start,
            end_layer=resolved_end,
            step_start=float(step_start),
            step_end=float(step_end),
            guidance_scale=float(guidance_scale),
            weaken_strength=float(weaken_strength),
            blur_passes=int(blur_passes),
            output_rescale=float(output_rescale),
            adaptive_gate=bool(adaptive_gate),
        )

        patch_size = getattr(transformer, "patch_size", 1)
        weakener = _TokenDetailWeakener(
            start_layer=spec.start_layer,
            end_layer=spec.end_layer,
            weaken_strength=spec.weaken_strength,
            blur_passes=spec.blur_passes,
            adaptive_gate=spec.adaptive_gate,
            patch_size=int(patch_size) if patch_size is not None else 1,
            flag_name=_WEAK_FLAG,
        )

        if hasattr(patched, "set_model_double_block_patch"):
            patched.set_model_double_block_patch(weakener)
        elif hasattr(patched, "set_model_patch"):
            patched.set_model_patch(weakener, "double_block")
        else:
            raise ValueError("This MODEL patcher does not expose a compatible double_block patch API.")

        prior_wrapper = patched.model_options.get("model_function_wrapper", _identity_wrapper)
        wrapper = _DetailGuidanceWrapper(spec=spec, prior_wrapper=prior_wrapper, flag_name=_WEAK_FLAG)
        patched.set_model_unet_function_wrapper(wrapper)

        patched.model_options["zimage_token_detail_guidance"] = {
            "num_layers": num_layers,
            "patch_size": int(patch_size) if patch_size is not None else 1,
            "spec": asdict(spec),
            "version": 4,
        }

        LOGGER.info(
            "Applied Z-Image token detail guidance v4: layers=%d band=%d:%d steps=%.2f..%.2f guidance=%.3f weaken=%.3f blur_passes=%d rescale=%.3f adaptive_gate=%s",
            num_layers,
            spec.start_layer,
            spec.end_layer,
            spec.step_start,
            spec.step_end,
            spec.guidance_scale,
            spec.weaken_strength,
            spec.blur_passes,
            spec.output_rescale,
            spec.adaptive_gate,
        )

        return (patched,)


NODE_CLASS_MAPPINGS = {
    "ZImageUpscaleDetailMomentumPatchModel": ZImageUpscaleDetailMomentumPatchModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZImageUpscaleDetailMomentumPatchModel": "Z-Image: Token Detail Guidance (UDG)",
}
