"""
CT->CTA Volume Inference: slice-by-slice Flow Matching synthesis
Usage:
  python inference_diffusion.py \
      --config config.yaml \
      --checkpoint outputs/run/checkpoints/latest.pt \
      --ct_path /path/to/ct.nii.gz \
      --output_dir ./predictions_diffusion
"""
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import yaml
from tqdm import tqdm

from diffusion import FlowMatchingSampler
from networks import build_model


def _autocast_context(device: torch.device, enabled: bool, dtype=torch.bfloat16):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=dtype, enabled=enabled)
    return torch.autocast(device_type=device.type, enabled=False)


def apply_window(hu: np.ndarray, center: float, width: float) -> np.ndarray:
    hu_min, hu_max = center - width / 2, center + width / 2
    return (np.clip(hu, hu_min, hu_max) - hu_min) / width * 2 - 1


def denormalize_cta(pred_norm: np.ndarray, center: float, width: float) -> np.ndarray:
    return (pred_norm + 1) / 2 * width + (center - width / 2)


def build_ct_condition(ct_vol: np.ndarray, z: int, brain_c: float = 35, brain_w: float = 90, n_ctx: int = 5) -> np.ndarray:
    depth = ct_vol.shape[2]
    half = n_ctx // 2
    indices = [min(max(z + dz, 0), depth - 1) for dz in range(-half, half + 1)]
    return np.stack([apply_window(ct_vol[:, :, i], brain_c, brain_w) for i in indices], axis=0).astype(np.float32)


def build_brain_mask(ct_vol: np.ndarray, z: int) -> np.ndarray:
    return (ct_vol[:, :, z] > -900).astype(np.float32)


def load_model(cfg: dict, checkpoint_path: str, device: torch.device, use_ema: bool = True):
    model_cfg = cfg["model"]
    model = build_model(model_cfg, inference=True).to(device)
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_key = "ema" if (use_ema and ck.get("ema") is not None) else "model"
    if state_key not in ck:
        state_key = next((k for k in ("model", "generator") if k in ck), None)
    if state_key is None:
        raise KeyError(f"Cannot find model weights in checkpoint keys: {list(ck.keys())}")
    checkpoint_class = ck.get("model_class")
    if checkpoint_class and checkpoint_class != model.__class__.__name__:
        raise ValueError(
            f"Checkpoint contains {checkpoint_class}, but config builds {model.__class__.__name__}"
        )
    missing, unexpected = model.load_state_dict(ck[state_key], strict=False)
    model.eval()
    if missing:
        print(f"Missing keys (random-init): {missing}")
    if unexpected:
        print(f"Unexpected keys (ignored): {unexpected}")
    print(f"Loaded '{state_key}' weights (epoch {ck.get('epoch', '?')})")
    return model


@torch.no_grad()
def infer_volume(
    model,
    sampler: FlowMatchingSampler,
    ct_vol: np.ndarray,
    cfg: dict,
    device: torch.device,
    batch_size: int = 4,
    num_steps: int = 50,
    amp_enabled: bool = False,
    amp_dtype=torch.bfloat16,
    return_vessel_prior: bool = False,
    ensemble_k: int = 1,
):
    model.eval()
    height, width, depth = ct_vol.shape
    pred_vol = np.zeros((height, width, depth), dtype=np.float32)
    vessel_prior_vol = np.zeros((height, width, depth), dtype=np.float32) if return_vessel_prior else None

    data_cfg = cfg["data"]
    brain_w, brain_c = data_cfg["ct_brain_window"]
    cta_w, cta_c = data_cfg["cta_window"]
    n_ctx = data_cfg["num_slices_context"]

    live_z = [z for z in range(depth) if bool((ct_vol[:, :, z] > -900).any())]

    for batch_start in tqdm(range(0, len(live_z), batch_size), desc="Synthesizing slices"):
        z_batch = live_z[batch_start : batch_start + batch_size]
        conds = [build_ct_condition(ct_vol, z, brain_c=brain_c, brain_w=brain_w, n_ctx=n_ctx) for z in z_batch]
        ct_cond = torch.tensor(np.stack(conds, axis=0), device=device)
        z_pos = torch.tensor([z / max(depth - 1, 1) for z in z_batch], device=device, dtype=torch.float32)
        with _autocast_context(device, amp_enabled, amp_dtype):
            pred_batch = sampler.sample_mean(
                model, ct_cond=ct_cond, z_pos=z_pos, num_steps=num_steps,
                k=ensemble_k,
            )
            if return_vessel_prior:
                if not hasattr(model, "locate_vessels"):
                    raise ValueError("Vessel-prior export requires model.architecture=anatomy_guided")
                prior_batch = torch.sigmoid(model.locate_vessels(ct_cond, z_pos)[0])

        pred_batch = pred_batch.clamp(-1.0, 1.0)
        if not torch.isfinite(pred_batch).all():
            raise RuntimeError("Inference produced non-finite values.")

        for i, z in enumerate(z_batch):
            pred_vol[:, :, z] = denormalize_cta(pred_batch[i, 0].cpu().numpy(), center=cta_c, width=cta_w)
            if return_vessel_prior:
                vessel_prior_vol[:, :, z] = prior_batch[i, 0].float().cpu().numpy()

    valid = ct_vol > -900
    pred_vol[~valid] = -1000.0
    if return_vessel_prior:
        vessel_prior_vol[~valid] = 0.0
        return pred_vol, vessel_prior_vol
    return pred_vol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default="outputs/run/checkpoints/latest.pt")
    parser.add_argument("--ct_path", required=True, help="Skull-stripped CT NIfTI file")
    parser.add_argument("--output_dir", default="./predictions_diffusion")
    parser.add_argument("--no_ema", action="store_true")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_steps", type=int, default=None, help="Flow Euler steps (overrides config)")
    parser.add_argument(
        "--ensemble_k", type=int, default=None,
        help="Average k independent draws (overrides config diffusion.ensemble_k).",
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"
    amp_dtype = torch.bfloat16 if cfg["training"].get("amp_dtype", "bfloat16") == "bfloat16" else torch.float16

    diff_cfg = cfg.get("diffusion", {})
    num_steps = args.num_steps or int(diff_cfg.get("sampler_steps", 50))
    ensemble_k = args.ensemble_k or int(diff_cfg.get("ensemble_k", 1))

    sampler = FlowMatchingSampler()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(cfg, args.checkpoint, device, use_ema=not args.no_ema)
    print(f"Flow Matching {num_steps} steps | ensemble k={ensemble_k} | "
          f"batch_size={args.batch_size} | device={device}")
    print("Input CT must be skull-stripped to match training distribution.")

    ct_nib = nib.load(args.ct_path)
    ct_vol = ct_nib.get_fdata().astype(np.float32)
    print(f"CT volume shape: {ct_vol.shape}")

    pred_hu = infer_volume(model, sampler, ct_vol, cfg, device,
                           batch_size=args.batch_size, num_steps=num_steps,
                           amp_enabled=amp_enabled, amp_dtype=amp_dtype,
                           ensemble_k=ensemble_k)
    print(f"Predicted CTA HU range: [{pred_hu.min():.1f}, {pred_hu.max():.1f}]")

    ct_name = Path(args.ct_path).name.replace(".nii.gz", "").replace(".nii", "")
    out_path = out_dir / f"{ct_name}_pred_cta.nii.gz"
    nib.save(nib.Nifti1Image(pred_hu, affine=ct_nib.affine, header=ct_nib.header), str(out_path))
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
