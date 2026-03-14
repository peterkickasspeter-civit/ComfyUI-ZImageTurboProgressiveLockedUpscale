import logging
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from typing import Optional, Sequence, Tuple

import torch

LOGGER = logging.getLogger(__name__)
_MISSING = object()


@dataclass(frozen=True)
class ReplaySpec:
    start_layer: int  # inclusive
    end_layer: int    # exclusive
    replays: int      # number of extra traversals of the band
    step_start: float
    step_end: float


class _ReplayModuleView:
    """
    Lightweight iterable/indexable view that replays a subset of modules
    without modifying weights or permanently altering module registration.
    """

    __slots__ = ("_modules", "_path")

    def __init__(self, modules: Sequence, path: Sequence[int]):
        self._modules = modules
        self._path = tuple(int(i) for i in path)

    def __len__(self):
        return len(self._path)

    def __iter__(self):
        for idx in self._path:
            yield self._modules[idx]

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self._modules[i] for i in self._path[idx]]
        return self._modules[self._path[idx]]

    def __repr__(self):
        return f"_ReplayModuleView(len={len(self._path)}, path={self._path})"


@dataclass(frozen=True)
class _ResolvedPatch:
    block_attr: str
    num_layers: int
    spec: ReplaySpec
    path: Tuple[int, ...]
    schedule_name: str


_PRESETS = {
    # one extra traversal, contiguous band, step-fraction windows measured on the
    # sampler trajectory: 0.0 = earliest/noisiest step, 1.0 = final/cleanest step.
    "semantic_v1": ReplaySpec(start_layer=6, end_layer=12, replays=1, step_start=0.00, step_end=0.35),
    "binding_v1": ReplaySpec(start_layer=10, end_layer=16, replays=1, step_start=0.10, step_end=0.45),
    "detail_v1": ReplaySpec(start_layer=18, end_layer=24, replays=1, step_start=0.65, step_end=0.90),
}


def _identity_wrapper(apply_model, args):
    return apply_model(args["input"], args["timestep"], **args["c"])


def _get_diffusion_transformer(model_patcher):
    if not hasattr(model_patcher, "model"):
        raise ValueError("Expected a ComfyUI MODEL patcher object with a .model attribute.")
    base_model = model_patcher.model
    if not hasattr(base_model, "diffusion_model"):
        raise ValueError("This MODEL does not expose .model.diffusion_model and cannot be replay-patched.")
    return base_model.diffusion_model


def _find_block_stack(transformer) -> Tuple[str, Sequence]:
    for name in ("layers", "transformer_blocks", "joint_blocks"):
        blocks = getattr(transformer, name, None)
        if blocks is not None and hasattr(blocks, "__len__") and hasattr(blocks, "__iter__"):
            return name, blocks
    available = [k for k in dir(transformer) if "layer" in k.lower() or "block" in k.lower()]
    raise ValueError(
        "Could not find a replayable transformer block stack on diffusion_model. "
        f"Tried: layers, transformer_blocks, joint_blocks. Related attrs: {available[:20]}"
    )


def _resolve_spec(
    schedule: str,
    start_layer: int,
    end_layer: int,
    replays: int,
    step_start: float,
    step_end: float,
    num_layers: int,
) -> ReplaySpec:
    if schedule != "custom":
        spec = _PRESETS[schedule]
    else:
        spec = ReplaySpec(
            start_layer=int(start_layer),
            end_layer=int(end_layer),
            replays=int(replays),
            step_start=float(step_start),
            step_end=float(step_end),
        )

    if spec.replays < 1:
        raise ValueError(f"replays must be >= 1, got {spec.replays}")
    if not (0.0 <= spec.step_start <= 1.0 and 0.0 <= spec.step_end <= 1.0):
        raise ValueError(
            f"step_start/step_end must be within [0,1], got {spec.step_start}..{spec.step_end}"
        )
    if spec.step_start > spec.step_end:
        raise ValueError(
            f"step_start must be <= step_end, got {spec.step_start} > {spec.step_end}"
        )
    if spec.start_layer < 0 or spec.end_layer < 0:
        raise ValueError(
            f"start_layer/end_layer must be non-negative, got {spec.start_layer}, {spec.end_layer}"
        )
    if spec.end_layer <= spec.start_layer:
        raise ValueError(
            f"end_layer must be > start_layer (end is exclusive), got {spec.start_layer}:{spec.end_layer}"
        )
    if spec.end_layer > num_layers:
        raise ValueError(
            f"Requested replay band {spec.start_layer}:{spec.end_layer} exceeds model depth {num_layers}."
        )
    return spec


def _build_path(num_layers: int, start_layer: int, end_layer: int, replays: int) -> Tuple[int, ...]:
    prefix = list(range(0, start_layer))
    band = list(range(start_layer, end_layer))
    suffix = list(range(end_layer, num_layers))
    return tuple(prefix + band * (replays + 1) + suffix)


@contextmanager
def _temporary_layer_path(transformer, block_attr: str, path: Sequence[int]):
    """
    Shadow the registered submodule attribute via __dict__ so forward() sees our
    replay view, while the original ModuleList remains intact in _modules.
    """
    prior_shadow = transformer.__dict__.get(block_attr, _MISSING)
    base_stack = transformer._modules.get(block_attr, getattr(transformer, block_attr))
    transformer.__dict__[block_attr] = _ReplayModuleView(base_stack, path)
    try:
        yield
    finally:
        if prior_shadow is _MISSING:
            transformer.__dict__.pop(block_attr, None)
        else:
            transformer.__dict__[block_attr] = prior_shadow


def _extract_sample_sigmas(args) -> Optional[torch.Tensor]:
    c = args.get("c", {}) or {}
    transformer_options = c.get("transformer_options", {}) or {}
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
    sigmas = _extract_sample_sigmas(args)
    if sigmas is None or sigmas.numel() == 0:
        return None, None, None

    if sigmas.numel() == 1:
        return 0.0, 0, 1

    timestep = args.get("timestep")
    if torch.is_tensor(timestep):
        t = timestep.detach().flatten()[0]
        t = t.to(device=sigmas.device, dtype=sigmas.dtype)
    else:
        t = torch.tensor(float(timestep), device=sigmas.device, dtype=sigmas.dtype)

    idx = int(torch.argmin(torch.abs(sigmas - t)).item())
    frac = idx / float(max(sigmas.numel() - 1, 1))
    return frac, idx, int(sigmas.numel())


class ZImageReplayPatchModel:
    """
    Inference-only MODEL -> MODEL patch node.

    It does not change weights. It temporarily replays a contiguous transformer
    layer band during selected denoising steps by patching the diffusion-model
    layer iterator at forward time.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enabled": ("BOOLEAN", {"default": True}),
                "schedule": (["custom", "semantic_v1", "binding_v1", "detail_v1"], {"default": "custom"}),
                "start_layer": (
                    "INT",
                    {
                        "default": 10,
                        "min": 0,
                        "max": 255,
                        "step": 1,
                        "tooltip": "Replay band start layer (inclusive). Used when schedule=custom.",
                    },
                ),
                "end_layer": (
                    "INT",
                    {
                        "default": 16,
                        "min": 1,
                        "max": 256,
                        "step": 1,
                        "tooltip": "Replay band end layer (exclusive). Used when schedule=custom.",
                    },
                ),
                "replays": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 4,
                        "step": 1,
                        "tooltip": "Number of extra traversals of the selected band.",
                    },
                ),
                "step_start": (
                    "FLOAT",
                    {
                        "default": 0.10,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Start step fraction. 0 = earliest/noisiest step, 1 = final/cleanest step. Used when schedule=custom.",
                    },
                ),
                "step_end": (
                    "FLOAT",
                    {
                        "default": 0.45,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "End step fraction. Used when schedule=custom.",
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
        schedule,
        start_layer,
        end_layer,
        replays,
        step_start,
        step_end,
    ):
        if not enabled:
            return (model,)

        patched = model.clone()
        transformer = _get_diffusion_transformer(patched)
        block_attr, block_stack = _find_block_stack(transformer)
        num_layers = len(block_stack)

        spec = _resolve_spec(
            schedule=schedule,
            start_layer=start_layer,
            end_layer=end_layer,
            replays=replays,
            step_start=step_start,
            step_end=step_end,
            num_layers=num_layers,
        )
        path = _build_path(num_layers, spec.start_layer, spec.end_layer, spec.replays)
        resolved = _ResolvedPatch(
            block_attr=block_attr,
            num_layers=num_layers,
            spec=spec,
            path=path,
            schedule_name=schedule,
        )

        prior_wrapper = patched.model_options.get("model_function_wrapper", _identity_wrapper)

        warned_missing_sigmas = {"value": False}

        def replay_wrapper(apply_model, args):
            step_frac, _, _ = _current_step_fraction(args)
            if step_frac is None:
                if not warned_missing_sigmas["value"]:
                    LOGGER.warning(
                        "ZImageReplayLayerBand: transformer_options.sample_sigmas missing; replay patch is inactive for this sampling path."
                    )
                    warned_missing_sigmas["value"] = True
                return prior_wrapper(apply_model, args)

            active = resolved.spec.step_start <= step_frac <= resolved.spec.step_end
            if not active:
                return prior_wrapper(apply_model, args)

            with _temporary_layer_path(transformer, resolved.block_attr, resolved.path):
                return prior_wrapper(apply_model, args)

        patched.set_model_unet_function_wrapper(replay_wrapper)

        patched.model_options["zimage_layer_replay"] = {
            "schedule": resolved.schedule_name,
            "block_attr": resolved.block_attr,
            "num_layers": resolved.num_layers,
            "path": list(resolved.path),
            "spec": asdict(resolved.spec),
        }

        LOGGER.info(
            "Applied Z-Image replay patch: schedule=%s attr=%s layers=%d band=%d:%d replays=%d steps=%.2f..%.2f",
            resolved.schedule_name,
            resolved.block_attr,
            resolved.num_layers,
            resolved.spec.start_layer,
            resolved.spec.end_layer,
            resolved.spec.replays,
            resolved.spec.step_start,
            resolved.spec.step_end,
        )

        return (patched,)


NODE_CLASS_MAPPINGS = {
    "ZImageReplayPatchModel": ZImageReplayPatchModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZImageReplayPatchModel": "Z-Image: Replay Layer Band (RYS)",
}
