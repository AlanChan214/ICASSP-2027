"""2.5D diffusion U-Net with dual conditioning (timestep + z-position)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def sinusoidal_embedding(t: Tensor, dim: int) -> Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / max(half - 1, 1))
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class ScalarEmbedding(nn.Module):
    """Sinusoidal + MLP embedding for a scalar in [0, 1]."""

    def __init__(self, out_dim: int, sin_dim: int = 64, scale: float = 1000.0):
        super().__init__()
        self.sin_dim = sin_dim
        self.scale = scale
        hidden = max(out_dim // 2, 32)
        self.mlp = nn.Sequential(
            nn.Linear(sin_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.mlp(sinusoidal_embedding(value * self.scale, self.sin_dim))


class SelfAttention2D(nn.Module):
    """Multi-head spatial self-attention for feature maps."""

    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        assert channels % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv.unbind(1)
        q, k, v = (t.transpose(-1, -2).contiguous() for t in (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(-1, -2).reshape(B, C, H, W)
        return x + self.proj(out)


class SEBlock(nn.Module):
    """Squeeze-Excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden), nn.SiLU(),
            nn.Linear(hidden, channels), nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x * self.fc(self.pool(x).flatten(1)).view(x.shape[0], x.shape[1], 1, 1)


class ResBlock(nn.Module):
    """Conv block with FiLM conditioning from the combined timestep+z embedding."""

    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, dropout: float = 0.0):
        super().__init__()
        groups = min(8, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.act = nn.GELU()
        self.film = nn.Linear(emb_dim, out_ch * 2)
        self.residual = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        scale, shift = self.film(emb)[:, :, None, None].chunk(2, dim=1)
        h = self.norm2(self.conv2(h)) * (1 + scale) + shift
        h = self.act(h)
        return self.drop(h) + self.residual(x)


class DoubleResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, dropout: float = 0.0):
        super().__init__()
        self.b1 = ResBlock(in_ch, out_ch, emb_dim, dropout)
        self.b2 = ResBlock(out_ch, out_ch, emb_dim, dropout)

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        return self.b2(self.b1(x, emb), emb)


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, dropout: float = 0.0):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleResBlock(in_ch, out_ch, emb_dim, dropout)

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        return self.conv(self.pool(x), emb)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, emb_dim: int, dropout: float = 0.0):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleResBlock(in_ch + skip_ch, out_ch, emb_dim, dropout)

    def forward(self, x: Tensor, skip: Tensor, emb: Tensor) -> Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1), emb)


class DiffusionUNet2D(nn.Module):
    """U-Net noise predictor for 2.5D conditional diffusion (v-prediction).

    Input:  [B, 6, H, W]  — 5 brain-window CT slices + 1 noisy CTA channel (concatenated)
    Output: [B, 1, H, W]  — predicted v  (no output activation; unbounded)

    Conditioning injected via FiLM at every ResBlock:
        emb = ScalarEmbedding(t_norm) + ScalarEmbedding(z_pos)

    Attention at enc3 (64×64) and bottleneck (32×32) for vessel structure modeling.
    """

    def __init__(
        self,
        in_channels: int = 6,
        out_channels: int = 1,
        base_ch: int = 48,
        emb_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        ch = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8, base_ch * 8]

        self.t_emb = ScalarEmbedding(emb_dim)
        self.z_emb = ScalarEmbedding(emb_dim)

        self.enc0 = DoubleResBlock(in_channels, ch[0], emb_dim, dropout)
        self.enc1 = DownBlock(ch[0], ch[1], emb_dim, dropout)
        self.enc2 = DownBlock(ch[1], ch[2], emb_dim, dropout)
        self.enc3 = DownBlock(ch[2], ch[3], emb_dim, dropout)
        self.enc3_attn = SelfAttention2D(ch[3])            # 64×64
        self.enc4 = DownBlock(ch[3], ch[4], emb_dim, dropout)

        self.bottleneck = DoubleResBlock(ch[4], ch[4], emb_dim, dropout)
        self.bottleneck_attn = SelfAttention2D(ch[4])      # 32×32

        self.dec3 = UpBlock(ch[4], ch[3], ch[3], emb_dim, dropout)
        self.dec3_se = SEBlock(ch[3])
        self.dec2 = UpBlock(ch[3], ch[2], ch[2], emb_dim, dropout)
        self.dec2_se = SEBlock(ch[2])
        self.dec1 = UpBlock(ch[2], ch[1], ch[1], emb_dim, dropout)
        self.dec0 = UpBlock(ch[1], ch[0], ch[0], emb_dim, dropout)

        self.out_conv = nn.Conv2d(ch[0], out_channels, 1)

    def forward(self, x: Tensor, t_norm: Tensor, z_pos: Tensor) -> Tensor:
        """
        Args:
            x:      [B, 6, H, W]  concat of CT cond (5ch) + noisy CTA (1ch)
            t_norm: [B]            t / T, in [0, 1]
            z_pos:  [B]            slice depth in [0, 1]
        Returns:
            v_pred: [B, 1, H, W]
        """
        emb = self.t_emb(t_norm) + self.z_emb(z_pos)

        s0 = self.enc0(x, emb)
        s1 = self.enc1(s0, emb)
        s2 = self.enc2(s1, emb)
        s3 = self.enc3(s2, emb)
        s3 = self.enc3_attn(s3)
        s4 = self.enc4(s3, emb)

        h = self.bottleneck(s4, emb)
        h = self.bottleneck_attn(h)

        h = self.dec3(h, s3, emb)
        h = self.dec3_se(h)
        h = self.dec2(h, s2, emb)
        h = self.dec2_se(h)
        h = self.dec1(h, s1, emb)
        h = self.dec0(h, s0, emb)

        return self.out_conv(h)
