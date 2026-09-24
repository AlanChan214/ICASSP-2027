"""Euler ODE sampler for Flow Matching models."""
from typing import Callable, Optional

import torch
from torch import Tensor


class FlowMatchingSampler:
    """Euler ODE integration from t=0 (noise) to t=1 (data).

    Model interface:
        model(x_t, t_norm, z_pos) -> v_pred
          x_t:    [B, 6, H, W]  (5 CT channels + 1 interpolated CTA)
          t_norm: [B]            t in [0, 1]
          z_pos:  [B]            slice depth in [0, 1]
        -> v_pred: [B, 1, H, W]
    """

    @torch.no_grad()
    def sample(
        self,
        model: Callable,
        ct_cond: Tensor,
        z_pos: Tensor,
        num_steps: int = 8,
        x_T: Optional[Tensor] = None,
        clamp_x0: bool = True,
        cfg_scale: float = 1.0,
    ) -> Tensor:
        """Generate a CTA slice by Euler integration of the flow ODE.

        Args:
            x_T: starting noise [B, 1, H, W]; sampled from N(0,I) if None.
            cfg_scale: guidance scale. 1.0 = no guidance.

        Returns:
            Predicted CTA slice [B, 1, H, W] in [-1, 1].
        """
        device = ct_cond.device
        B, _, H, W = ct_cond.shape

        x = x_T if x_T is not None else torch.randn(B, 1, H, W, device=device, dtype=ct_cond.dtype)

        use_cfg = cfg_scale > 1.0
        ct_null = torch.zeros_like(ct_cond) if use_cfg else None

        dt = 1.0 / num_steps

        extra = {}
        if hasattr(model, "encode_anatomy_cache"):
            if use_cfg:
                extra["anatomy_cache"] = model.encode_anatomy_cache(
                    torch.cat([ct_cond, ct_null], dim=0), torch.cat([z_pos, z_pos], dim=0)
                )
            else:
                extra["anatomy_cache"] = model.encode_anatomy_cache(ct_cond, z_pos)

        for i in range(num_steps):
            t_val = i / num_steps
            t_batch = torch.full((B,), t_val, device=device, dtype=ct_cond.dtype)

            if use_cfg:
                cond2 = torch.cat([
                    torch.cat([ct_cond, x], dim=1),
                    torch.cat([ct_null, x], dim=1),
                ], dim=0)
                t2 = torch.cat([t_batch, t_batch], dim=0)
                z2 = torch.cat([z_pos, z_pos], dim=0)
                v_both = model(cond2, t2, z2, **extra)
                v_cond, v_uncond = v_both.chunk(2, dim=0)
                v_pred = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                x_input = torch.cat([ct_cond, x], dim=1)
                v_pred = model(x_input, t_batch, z_pos, **extra)

            x = x + v_pred * dt

        if clamp_x0:
            x = x.clamp(-1.0, 1.0)

        return x

    @torch.no_grad()
    def sample_mean(
        self,
        model: Callable,
        ct_cond: Tensor,
        z_pos: Tensor,
        num_steps: int = 8,
        k: int = 1,
        x_T_fn: Optional[Callable[[int], Optional[Tensor]]] = None,
        clamp_x0: bool = True,
        cfg_scale: float = 1.0,
    ) -> Tensor:
        """Return the mean of ``k`` independently sampled predictions."""
        if k < 1:
            raise ValueError(f"Ensemble size k must be >= 1, got {k}")

        total = None
        for draw in range(k):
            x_T = x_T_fn(draw) if x_T_fn is not None else None
            drawn = self.sample(
                model,
                ct_cond=ct_cond,
                z_pos=z_pos,
                num_steps=num_steps,
                x_T=x_T,
                clamp_x0=clamp_x0,
                cfg_scale=cfg_scale,
            ).float()
            total = drawn if total is None else total + drawn

        return (total / float(k)).to(ct_cond.dtype)
