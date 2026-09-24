"""Losses for anatomy-guided flow matching and CT vessel localization."""

from typing import Dict, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


def _zero_with_grad(reference: Tensor) -> Tensor:
    return reference.sum() * 0.0


def weighted_flow_mse(
    prediction: Tensor,
    target: Tensor,
    valid_mask: Tensor,
    vessel_mask: Tensor,
    vessel_weight: Tensor,
    vessel_factor: float = 4.0,
    eps: float = 1e-6,
) -> Tensor:
    case_weight = vessel_weight.to(prediction.dtype).reshape(-1, 1, 1, 1)
    valid = valid_mask.to(prediction.dtype)
    vessel = vessel_mask.to(prediction.dtype)
    weight = valid * (1.0 + vessel_factor * case_weight * vessel)
    return (weight * (prediction - target).square()).sum() / weight.sum().clamp_min(eps)


def soft_dice_loss(logits: Tensor, target: Tensor, valid: Tensor, eps: float = 1e-6) -> Tensor:
    """Dice over vessel-containing slices only; BCE handles empty slices."""
    probs = torch.sigmoid(logits.float()) * valid.float()
    target = target.float() * valid.float()
    present = target.flatten(1).sum(1) > 0
    if not bool(present.any()):
        return _zero_with_grad(logits)
    probs, target = probs[present], target[present]
    intersection = (probs * target).flatten(1).sum(1)
    denom = probs.flatten(1).sum(1) + target.flatten(1).sum(1)
    return (1.0 - (2.0 * intersection + eps) / (denom + eps)).mean()


def soft_erode(image: Tensor) -> Tensor:
    eroded_h = -F.max_pool2d(-image, (3, 1), stride=1, padding=(1, 0))
    eroded_w = -F.max_pool2d(-image, (1, 3), stride=1, padding=(0, 1))
    return torch.minimum(eroded_h, eroded_w)


def soft_dilate(image: Tensor) -> Tensor:
    return F.max_pool2d(image, 3, stride=1, padding=1)


def soft_open(image: Tensor) -> Tensor:
    return soft_dilate(soft_erode(image))


def soft_skeletonize(image: Tensor, iterations: int = 5) -> Tensor:
    image = image.float()
    skeleton = F.relu(image - soft_open(image))
    for _ in range(iterations):
        image = soft_erode(image)
        delta = F.relu(image - soft_open(image))
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


def soft_cldice_loss(
    logits: Tensor,
    target: Tensor,
    valid: Tensor,
    iterations: int = 5,
    eps: float = 1e-6,
) -> Tensor:
    """Full-resolution soft-clDice over vessel-containing slices only."""
    target = target.float() * valid.float()
    present = target.flatten(1).sum(1) > 0
    if not bool(present.any()):
        return _zero_with_grad(logits)
    probs = torch.sigmoid(logits.float()) * valid.float()
    probs, target = probs[present], target[present]
    skel_pred = soft_skeletonize(probs, iterations)
    skel_true = soft_skeletonize(target, iterations)
    topology_precision = (skel_pred * target).flatten(1).sum(1) / skel_pred.flatten(1).sum(1).clamp_min(eps)
    topology_sensitivity = (skel_true * probs).flatten(1).sum(1) / skel_true.flatten(1).sum(1).clamp_min(eps)
    cldice = 2.0 * topology_precision * topology_sensitivity / (
        topology_precision + topology_sensitivity
    ).clamp_min(eps)
    return (1.0 - cldice).mean()


def multiscale_locator_losses(
    vessel_logits: Sequence[Tensor],
    vessel_mask: Tensor,
    valid_mask: Tensor,
    vessel_weight: Tensor,
    scale_weights: Sequence[float] = (1.0, 0.5, 0.25, 0.125, 0.0625),
    vessel_factor: float = 4.0,
    skeleton_iterations: int = 5,
    eps: float = 1e-6,
) -> Dict[str, Tensor]:
    if len(vessel_logits) != len(scale_weights):
        raise ValueError(f"Expected {len(scale_weights)} vessel logits, got {len(vessel_logits)}")
    case_weight = vessel_weight.float().reshape(-1, 1, 1, 1)
    bce_total = _zero_with_grad(vessel_logits[0])
    dice_total = _zero_with_grad(vessel_logits[0])
    weight_total = float(sum(scale_weights))

    for logits, scale_weight in zip(vessel_logits, scale_weights):
        size = logits.shape[2:]
        target = F.adaptive_max_pool2d(vessel_mask.float(), size)
        valid = F.interpolate(valid_mask.float(), size=size, mode="nearest")
        target = target * valid
        pixel_weight = valid * (1.0 + vessel_factor * case_weight * target)
        bce_map = F.binary_cross_entropy_with_logits(logits.float(), target, reduction="none")
        bce = (bce_map * pixel_weight).sum() / pixel_weight.sum().clamp_min(eps)
        dice = soft_dice_loss(logits, target, valid, eps)
        bce_total = bce_total + float(scale_weight) * bce
        dice_total = dice_total + float(scale_weight) * dice

    cldice = soft_cldice_loss(
        vessel_logits[0], vessel_mask, valid_mask, iterations=skeleton_iterations, eps=eps
    )
    return {"bce": bce_total / weight_total, "dice": dice_total / weight_total, "cldice": cldice}


def multiscale_enhancement_loss(
    enhancement_preds: Sequence[Tensor],
    enhancement_target: Tensor,
    valid_mask: Tensor,
    scale_weights: Sequence[float] = (1.0, 0.5, 0.25, 0.125, 0.0625),
    eps: float = 1e-6,
) -> Tensor:
    """Compute masked multiscale L1 enhancement loss."""
    if len(enhancement_preds) != len(scale_weights):
        raise ValueError(
            f"Expected {len(scale_weights)} enhancement maps, got {len(enhancement_preds)}"
        )
    total = _zero_with_grad(enhancement_preds[0])
    weight_total = float(sum(scale_weights))

    for pred, scale_weight in zip(enhancement_preds, scale_weights):
        size = pred.shape[2:]
        target = F.adaptive_avg_pool2d(enhancement_target.float(), size)
        valid = F.interpolate(valid_mask.float(), size=size, mode="nearest")
        l1 = ((pred.float() - target).abs() * valid).sum() / valid.sum().clamp_min(eps)
        total = total + float(scale_weight) * l1

    return total / weight_total


def vessel_intensity_l1(
    x1_pred: Tensor,
    x1_target: Tensor,
    valid_mask: Tensor,
    vessel_mask: Tensor,
    vessel_weight: Tensor,
    t: Tensor,
    vessel_factor: float = 4.0,
    eps: float = 1e-6,
) -> Tensor:
    """Compute vessel-weighted L1 loss on the reconstructed target."""
    case_weight = vessel_weight.to(x1_pred.dtype).reshape(-1, 1, 1, 1)
    weight = valid_mask.to(x1_pred.dtype) * (
        1.0 + vessel_factor * case_weight * vessel_mask.to(x1_pred.dtype)
    )
    weight = weight * t.to(x1_pred.dtype).reshape(-1, 1, 1, 1)
    return (weight * (x1_pred - x1_target).abs()).sum() / weight.sum().clamp_min(eps)


def anatomy_guided_loss(
    velocity: Tensor,
    velocity_target: Tensor,
    vessel_logits: Sequence[Tensor],
    valid_mask: Tensor,
    vessel_mask: Tensor,
    vessel_weight: Tensor,
    flow_vessel_factor: float = 4.0,
    scale_weights: Sequence[float] = (1.0, 0.5, 0.25, 0.125, 0.0625),
    bce_coefficient: float = 0.10,
    dice_coefficient: float = 0.10,
    cldice_coefficient: float = 0.015,
    skeleton_iterations: int = 5,
    x_t: Tensor = None,
    t: Tensor = None,
    x1_target: Tensor = None,
    intensity_coefficient: float = 0.0,
    vessel_enhancement: Sequence[Tensor] = None,
    enhancement_target: Tensor = None,
    enhancement_coefficient: float = 0.0,
) -> Dict[str, Tensor]:
    flow = weighted_flow_mse(
        velocity, velocity_target, valid_mask, vessel_mask, vessel_weight, flow_vessel_factor
    )
    locator = multiscale_locator_losses(
        vessel_logits,
        vessel_mask,
        valid_mask,
        vessel_weight,
        scale_weights=scale_weights,
        vessel_factor=flow_vessel_factor,
        skeleton_iterations=skeleton_iterations,
    )
    contributions = {
        "flow": flow,
        "bce": locator["bce"],
        "dice": locator["dice"],
        "cldice": locator["cldice"],
        "weighted_bce": bce_coefficient * locator["bce"],
        "weighted_dice": dice_coefficient * locator["dice"],
        "weighted_cldice": cldice_coefficient * locator["cldice"],
    }
    contributions["total"] = (
        flow + contributions["weighted_bce"] + contributions["weighted_dice"] + contributions["weighted_cldice"]
    )

    if intensity_coefficient > 0.0:
        if x_t is None or t is None or x1_target is None:
            raise ValueError(
                "intensity_coefficient > 0 requires x_t, t and x1_target so the "
                "clean image can be reconstructed as x_t + (1-t)*velocity"
            )
        x1_pred = x_t + (1.0 - t.to(x_t.dtype).reshape(-1, 1, 1, 1)) * velocity
        intensity = vessel_intensity_l1(
            x1_pred, x1_target, valid_mask, vessel_mask, vessel_weight, t,
            vessel_factor=flow_vessel_factor,
        )
        contributions["intensity"] = intensity
        contributions["weighted_intensity"] = intensity_coefficient * intensity
        contributions["total"] = contributions["total"] + contributions["weighted_intensity"]

    if enhancement_coefficient > 0.0:
        if vessel_enhancement is None or enhancement_target is None:
            raise ValueError(
                "enhancement_coefficient > 0 requires vessel_enhancement (from "
                "forward(..., return_aux=True)) and enhancement_target (from the "
                "dataset's 'enhancement' key)"
            )
        enhancement = multiscale_enhancement_loss(
            vessel_enhancement, enhancement_target, valid_mask, scale_weights=scale_weights
        )
        contributions["enhancement"] = enhancement
        contributions["weighted_enhancement"] = enhancement_coefficient * enhancement
        contributions["total"] = contributions["total"] + contributions["weighted_enhancement"]

    return contributions
