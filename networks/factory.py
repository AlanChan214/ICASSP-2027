"""Model construction shared by training and inference."""

from .anatomy_guided_unet2d import AnatomyGuidedDiffusionUNet2D
from .diffusion_unet2d import DiffusionUNet2D


def build_model(model_cfg: dict, inference: bool = False):
    architecture = model_cfg.get("architecture", "baseline")
    common = dict(
        in_channels=model_cfg["in_channels"],
        out_channels=model_cfg["out_channels"],
        base_ch=model_cfg["base_ch"],
        emb_dim=model_cfg["emb_dim"],
        dropout=0.0 if inference else model_cfg.get("dropout", 0.0),
    )
    if architecture == "baseline":
        return DiffusionUNet2D(**common)
    if architecture == "anatomy_guided":
        return AnatomyGuidedDiffusionUNet2D(
            **common,
            anatomy_channels=model_cfg.get("anatomy_channels", [32, 64, 128, 192, 256]),
            fpn_channels=model_cfg.get("fpn_channels", 64),
            prior_input=model_cfg.get("prior_input", False),
            enhancement_head=model_cfg.get("enhancement_head", False),
        )
    raise ValueError(f"Unknown model.architecture: {architecture!r}")
