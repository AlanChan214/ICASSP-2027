"""Flow Matching scheduler: straight-line interpolation between noise and data."""
import torch
from torch import Tensor


class FlowMatchingScheduler:
    """Straight-line conditional flow matching (Lipman et al. 2022).

    Forward process: x_t = (1-t)·x_0 + t·x_1
      x_0 ~ N(0, I)  noise  (t=0)
      x_1 = data            (t=1)
      t ∈ [0, 1]     uniformly sampled during training

    Target velocity: u_t = x_1 - x_0  (constant, independent of t)
    Loss: MSE(v_θ(x_t, t), x_1 - x_0)  — no SNR weighting needed
    """

    def forward(self, x_1: Tensor, x_0: Tensor, t: Tensor):
        """Interpolate and return (x_t, velocity target).

        Args:
            x_1: clean data  [B, C, H, W]
            x_0: noise       [B, C, H, W]
            t:   time        [B], uniform in [0, 1]

        Returns:
            x_t:      (1-t)·x_0 + t·x_1   [B, C, H, W]
            v_target: x_1 - x_0            [B, C, H, W]
        """
        t_ = t[:, None, None, None]
        x_t = (1.0 - t_) * x_0 + t_ * x_1
        v_target = x_1 - x_0
        return x_t, v_target

    def predict_x1(self, x_t: Tensor, v_pred: Tensor, t: Tensor) -> Tensor:
        """Recover data estimate from current state and predicted velocity.

        Derivation: x_1 = x_t + (1-t)·v
        (from x_t = (1-t)·x_0 + t·x_1 and v = x_1 - x_0)
        """
        t_ = t[:, None, None, None]
        return x_t + (1.0 - t_) * v_pred
