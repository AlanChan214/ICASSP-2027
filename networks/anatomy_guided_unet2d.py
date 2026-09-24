"""Anatomy-guided 2.5D Flow U-Net with a CT-only vessel locator."""

from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .diffusion_unet2d import (
    DoubleResBlock,
    DownBlock,
    ScalarEmbedding,
    SEBlock,
    SelfAttention2D,
    UpBlock,
)


class CTAnatomyEncoder(nn.Module):
    """CT-only five-level encoder; z position is injected with FiLM."""

    def __init__(
        self,
        in_channels: int = 5,
        channels: Sequence[int] = (32, 64, 128, 192, 256),
        emb_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        if len(channels) != 5:
            raise ValueError("CTAnatomyEncoder requires exactly five channel levels")
        self.channels = tuple(int(c) for c in channels)
        self.z_emb = ScalarEmbedding(emb_dim)
        self.enc0 = DoubleResBlock(in_channels, self.channels[0], emb_dim, dropout)
        self.down = nn.ModuleList(
            DownBlock(self.channels[i], self.channels[i + 1], emb_dim, dropout)
            for i in range(4)
        )

    def forward(self, ct: Tensor, z_pos: Tensor) -> List[Tensor]:
        z_emb = self.z_emb(z_pos)
        features = [self.enc0(ct, z_emb)]
        for block in self.down:
            features.append(block(features[-1], z_emb))
        return features


class CTVesselLocator(nn.Module):
    """Top-down feature pyramid for multiscale vessel localization."""

    def __init__(
        self,
        anatomy_channels: Sequence[int],
        fpn_channels: int = 64,
        emb_dim: int = 256,
        enhancement_head: bool = False,
    ):
        super().__init__()
        if len(anatomy_channels) != 5:
            raise ValueError("CTVesselLocator requires exactly five feature levels")
        self.lateral = nn.ModuleList(nn.Conv2d(c, fpn_channels, 1) for c in anatomy_channels)
        self.smooth = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1),
                nn.GroupNorm(min(8, fpn_channels), fpn_channels),
                nn.GELU(),
            )
            for _ in anatomy_channels
        )
        self.z_emb = ScalarEmbedding(emb_dim)
        self.z_film = nn.ModuleList(nn.Linear(emb_dim, 2 * fpn_channels) for _ in anatomy_channels)
        self.heads = nn.ModuleList(nn.Conv2d(fpn_channels, 1, 1) for _ in anatomy_channels)
        self.enhancement_head = bool(enhancement_head)
        self.enh_heads = (
            nn.ModuleList(nn.Conv2d(fpn_channels, 1, 1) for _ in anatomy_channels)
            if self.enhancement_head
            else None
        )

    def forward(
        self,
        anatomy_features: Sequence[Tensor],
        z_pos: Tensor,
        return_enhancement: bool = False,
    ):
        if len(anatomy_features) != 5:
            raise ValueError("Expected five anatomy feature maps")
        z_emb = self.z_emb(z_pos)
        pyramid: List[Tensor] = [torch.empty(0)] * 5
        top = None
        for level in range(4, -1, -1):
            lateral = self.lateral[level](anatomy_features[level])
            if top is not None:
                lateral = lateral + F.interpolate(top, size=lateral.shape[2:], mode="bilinear", align_corners=False)
            top = self.smooth[level](lateral)
            scale, shift = self.z_film[level](z_emb)[:, :, None, None].chunk(2, dim=1)
            pyramid[level] = top * (1.0 + scale) + shift
        logits = [head(feature) for head, feature in zip(self.heads, pyramid)]
        if not return_enhancement:
            return logits
        if self.enh_heads is None:
            raise ValueError(
                "return_enhancement=True requires the locator to be built with "
                "enhancement_head=True (config model.enhancement_head)"
            )
        enhancement = [
            torch.sigmoid(head(feature)) for head, feature in zip(self.enh_heads, pyramid)
        ]
        return logits, enhancement


class AnatomyVesselFusion(nn.Module):
    """Soft vessel-gated residual injection of CT anatomy into flow features."""

    def __init__(self, flow_channels: Sequence[int], anatomy_channels: Sequence[int]):
        super().__init__()
        if len(flow_channels) != 5 or len(anatomy_channels) != 5:
            raise ValueError("AnatomyVesselFusion requires five feature levels")
        self.anatomy_proj = nn.ModuleList(
            nn.Conv2d(a_ch, h_ch, 1) for h_ch, a_ch in zip(flow_channels, anatomy_channels)
        )
        self.gates = nn.ModuleList(
            nn.Conv2d(h_ch * 2 + 1, h_ch, 3, padding=1) for h_ch in flow_channels
        )

    def forward(self, level: int, flow: Tensor, anatomy: Tensor, vessel_logits: Tensor) -> Tensor:
        conditioned = self.anatomy_proj[level](anatomy)
        prior = torch.sigmoid(vessel_logits)
        if conditioned.shape[2:] != flow.shape[2:]:
            conditioned = F.interpolate(conditioned, size=flow.shape[2:], mode="bilinear", align_corners=False)
        if prior.shape[2:] != flow.shape[2:]:
            prior = F.interpolate(prior, size=flow.shape[2:], mode="bilinear", align_corners=False)
        gate = torch.sigmoid(self.gates[level](torch.cat([flow, conditioned, prior], dim=1)))
        return flow + gate * conditioned


class AnatomyGuidedDiffusionUNet2D(nn.Module):
    """Flow U-Net augmented with CT anatomy features and soft vessel priors.

    The ordinary forward call remains sampler-compatible and returns velocity.
    ``return_aux=True`` additionally returns the five CT-only locator logits, and
    -- when the anatomy branch is evaluated rather than supplied via
    ``anatomy_cache`` -- the five dense enhancement maps under
    ``vessel_enhancement``.
    """

    def __init__(
        self,
        in_channels: int = 6,
        out_channels: int = 1,
        base_ch: int = 48,
        emb_dim: int = 256,
        dropout: float = 0.0,
        anatomy_channels: Sequence[int] = (32, 64, 128, 192, 256),
        fpn_channels: int = 64,
        prior_input: bool = False,
        enhancement_head: bool = False,
    ):
        super().__init__()
        if in_channels < 2:
            raise ValueError("Input must contain CT channels and one noisy CTA channel")
        self.ct_channels = in_channels - 1
        self.prior_input = bool(prior_input)
        flow_in_channels = in_channels + (1 if self.prior_input else 0)
        flow_ch = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8, base_ch * 8]
        anatomy_channels = tuple(int(c) for c in anatomy_channels)

        self.t_emb = ScalarEmbedding(emb_dim)
        self.z_emb = ScalarEmbedding(emb_dim)
        self.anatomy_encoder = CTAnatomyEncoder(
            self.ct_channels, anatomy_channels, emb_dim=emb_dim, dropout=dropout
        )
        self.vessel_locator = CTVesselLocator(
            anatomy_channels, fpn_channels, emb_dim, enhancement_head=enhancement_head
        )
        self.fusion = AnatomyVesselFusion(flow_ch, anatomy_channels)

        self.enc0 = DoubleResBlock(flow_in_channels, flow_ch[0], emb_dim, dropout)
        self.enc1 = DownBlock(flow_ch[0], flow_ch[1], emb_dim, dropout)
        self.enc2 = DownBlock(flow_ch[1], flow_ch[2], emb_dim, dropout)
        self.enc3 = DownBlock(flow_ch[2], flow_ch[3], emb_dim, dropout)
        self.enc3_attn = SelfAttention2D(flow_ch[3])
        self.enc4 = DownBlock(flow_ch[3], flow_ch[4], emb_dim, dropout)

        self.bottleneck = DoubleResBlock(flow_ch[4], flow_ch[4], emb_dim, dropout)
        self.bottleneck_attn = SelfAttention2D(flow_ch[4])
        self.dec3 = UpBlock(flow_ch[4], flow_ch[3], flow_ch[3], emb_dim, dropout)
        self.dec3_se = SEBlock(flow_ch[3])
        self.dec2 = UpBlock(flow_ch[3], flow_ch[2], flow_ch[2], emb_dim, dropout)
        self.dec2_se = SEBlock(flow_ch[2])
        self.dec1 = UpBlock(flow_ch[2], flow_ch[1], flow_ch[1], emb_dim, dropout)
        self.dec0 = UpBlock(flow_ch[1], flow_ch[0], flow_ch[0], emb_dim, dropout)
        self.out_conv = nn.Conv2d(flow_ch[0], out_channels, 1)

    def forward(
        self,
        x: Tensor,
        t_norm: Tensor,
        z_pos: Tensor,
        return_aux: bool = False,
        anatomy_cache=None,
    ):
        vessel_enhancement = None
        want_enhancement = return_aux and self.vessel_locator.enhancement_head
        if anatomy_cache is None:
            if want_enhancement:
                anatomy, vessel_logits, vessel_enhancement = self._encode_anatomy(
                    x[:, : self.ct_channels], z_pos, return_enhancement=True
                )
            else:
                anatomy, vessel_logits = self._encode_anatomy(x[:, : self.ct_channels], z_pos)
        else:
            anatomy, vessel_logits = anatomy_cache
        emb = self.t_emb(t_norm) + self.z_emb(z_pos)

        if self.prior_input:
            prior = torch.sigmoid(vessel_logits[0]).detach()
            if prior.shape[2:] != x.shape[2:]:
                prior = F.interpolate(prior, size=x.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, prior], dim=1)

        s0 = self.fusion(0, self.enc0(x, emb), anatomy[0], vessel_logits[0])
        s1 = self.fusion(1, self.enc1(s0, emb), anatomy[1], vessel_logits[1])
        s2 = self.fusion(2, self.enc2(s1, emb), anatomy[2], vessel_logits[2])
        s3 = self.fusion(3, self.enc3(s2, emb), anatomy[3], vessel_logits[3])
        s3 = self.enc3_attn(s3)
        s4 = self.fusion(4, self.enc4(s3, emb), anatomy[4], vessel_logits[4])

        h = self.bottleneck_attn(self.bottleneck(s4, emb))
        h = self.dec3_se(self.dec3(h, s3, emb))
        h = self.dec2_se(self.dec2(h, s2, emb))
        h = self.dec1(h, s1, emb)
        velocity = self.out_conv(self.dec0(h, s0, emb))
        if return_aux:
            aux = {"velocity": velocity, "vessel_logits": vessel_logits}
            if vessel_enhancement is not None:
                aux["vessel_enhancement"] = vessel_enhancement
            return aux
        return velocity

    def _encode_anatomy(self, ct: Tensor, z_pos: Tensor, return_enhancement: bool = False):
        anatomy = self.anatomy_encoder(ct, z_pos)
        if return_enhancement:
            logits, enhancement = self.vessel_locator(anatomy, z_pos, return_enhancement=True)
            return anatomy, logits, enhancement
        return anatomy, self.vessel_locator(anatomy, z_pos)

    def locate_vessels(self, ct: Tensor, z_pos: Tensor) -> List[Tensor]:
        """Run the CT-only branch without evaluating the flow U-Net."""
        return self._encode_anatomy(ct, z_pos)[1]

    def encode_anatomy_cache(self, ct: Tensor, z_pos: Tensor):
        """Precompute the CT-only branch for reuse across ODE steps."""
        return self._encode_anatomy(ct, z_pos)
