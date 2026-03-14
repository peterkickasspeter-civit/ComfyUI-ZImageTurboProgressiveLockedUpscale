from .ZImageReplay import ZImageReplayPatchModel
from .ZImageTurboProgressiveLockedUpscaleVersion1 import ZImageTurboProgressiveLockedUpscale
from .ZImageUpscaleDetailMomentumPatchModel import ZImageUpscaleDetailMomentumPatchModel

NODE_CLASS_MAPPINGS = {
    "ZImageReplayPatchModel": ZImageReplayPatchModel,
    "ZImageUpscaleDetailMomentumPatchModel": ZImageUpscaleDetailMomentumPatchModel,
    "ZImageTurboProgressiveLockedUpscale": ZImageTurboProgressiveLockedUpscale
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZImageReplayPatchModel": "ZImageReplayPatchModel",
    "ZImageUpscaleDetailMomentumPatchModel": "ZImageUpscaleDetailMomentumPatchModel",
    "ZImageTurboProgressiveLockedUpscale": "ZImageTurboProgressiveLockedUpscale"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']