"""Train the anatomy-guided conditional flow-matching model."""

import argparse
import json
import os
import random
from copy import deepcopy
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.dataset import CT2CTADataset2D
from diffusion import FlowMatchingScheduler
from networks import build_model
from utils.losses import anatomy_guided_loss


def _autocast_context(device: torch.device, enabled: bool, dtype=torch.bfloat16):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=dtype, enabled=enabled)
    return torch.autocast(device_type=device.type, enabled=False)


class EMA:
    def __init__(self, model, decay: float = 0.9995):
        self.decay = decay
        self.shadow = deepcopy(model).eval()
        for parameter in self.shadow.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for shadow, current in zip(self.shadow.parameters(), model.parameters()):
            shadow.data.mul_(self.decay).add_(current.data, alpha=1 - self.decay)

    def state_dict(self):
        return self.shadow.state_dict()


class CaseSliceSampler(Sampler):
    def __init__(
        self,
        dataset,
        batch_size: int,
        rank: int = 0,
        world_size: int = 1,
        drop_last: bool = True,
    ):
        self.batch_size = int(batch_size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        case_to_indices = {}
        for sample_idx, (case_idx, _z) in enumerate(dataset._index):
            case_to_indices.setdefault(case_idx, []).append(sample_idx)
        self.case_to_indices = case_to_indices

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def _build_batches(self):
        rng = np.random.default_rng(self.epoch)
        case_ids = list(self.case_to_indices)
        rng.shuffle(case_ids)
        batches = []
        for case_id in case_ids:
            indices = list(self.case_to_indices[case_id])
            rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        usable = (len(batches) // self.world_size) * self.world_size
        return batches[:usable]

    def __iter__(self):
        batches = self._build_batches()
        indices = [idx for batch in batches[self.rank :: self.world_size] for idx in batch]
        return iter(indices)

    def __len__(self):
        return len(self._build_batches()[self.rank :: self.world_size]) * self.batch_size


def _save_checkpoint(
    path: Path,
    model,
    ema,
    optimizer,
    lr_scheduler,
    epoch: int,
    global_step: int,
    include_optimizer: bool = True,
):
    torch.save(
        {
            "model_class": model.__class__.__name__,
            "model": model.state_dict(),
            "ema": ema.state_dict() if ema else None,
            "optimizer": optimizer.state_dict() if include_optimizer else None,
            "lr_scheduler": (
                lr_scheduler.state_dict()
                if include_optimizer and lr_scheduler is not None
                else None
            ),
            "epoch": epoch,
            "global_step": global_step,
        },
        path,
    )


def train(cfg: dict, args):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    ddp = world_size > 1

    if ddp:
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", timeout=timedelta(hours=6))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    seed = int(cfg["training"]["seed"]) + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    with open(cfg["data"]["train_list"]) as stream:
        train_list = json.load(stream)

    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    diffusion_cfg = cfg.get("diffusion", {})

    dataset = CT2CTADataset2D(
        train_list,
        augment=train_cfg.get("augment", True),
        num_slices_context=data_cfg["num_slices_context"],
        ct_brain_window=data_cfg["ct_brain_window"],
        cta_window=data_cfg["cta_window"],
        z_weight_middle=data_cfg["z_weight_middle"],
        z_middle_range=data_cfg["z_middle_range"],
        vessel_density_weight=data_cfg.get("vessel_density_weight", 0.0),
        cache_size_cases=data_cfg.get("cache_size_cases", 4),
        rotation_prob=data_cfg.get("rotation_prob", 0.0),
    )
    sampler = CaseSliceSampler(
        dataset,
        train_cfg["batch_size"],
        rank=rank,
        world_size=world_size,
    )
    loader_kwargs = {
        "batch_size": train_cfg["batch_size"],
        "sampler": sampler,
        "shuffle": False,
        "num_workers": train_cfg["num_workers"],
        "pin_memory": True,
        "drop_last": True,
        "persistent_workers": train_cfg["num_workers"] > 0,
    }
    if train_cfg["num_workers"] > 0:
        loader_kwargs["prefetch_factor"] = train_cfg.get("prefetch_factor", 2)
    train_loader = DataLoader(dataset, **loader_kwargs)

    model_cfg = cfg["model"]
    anatomy_guided = model_cfg.get("architecture") == "anatomy_guided"
    model = build_model(model_cfg).to(device)

    enhancement_head = bool(model_cfg.get("enhancement_head", False))
    enhancement_coefficient = float(train_cfg.get("locator_enhancement_weight", 0.0))
    if enhancement_head != (enhancement_coefficient > 0.0):
        raise ValueError(
            "model.enhancement_head and training.locator_enhancement_weight must be enabled together"
        )

    if ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    raw_model = model.module if ddp else model
    ema = EMA(raw_model, decay=float(train_cfg["ema_decay"])) if rank == 0 else None

    flow_scheduler = FlowMatchingScheduler()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        betas=tuple(train_cfg.get("betas", [0.9, 0.999])),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    lr_scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(train_cfg["epochs"]),
            eta_min=float(train_cfg.get("lr_min", 1e-6)),
        )
        if train_cfg.get("lr_schedule", "cosine") == "cosine"
        else None
    )

    amp_enabled = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    amp_dtype = (
        torch.bfloat16
        if train_cfg.get("amp_dtype", "bfloat16") == "bfloat16"
        else torch.float16
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=amp_enabled and amp_dtype == torch.float16
    )

    output_dir = Path(train_cfg["output_dir"])
    checkpoint_dir = output_dir / "checkpoints"
    if rank == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(output_dir / "tensorboard")
    else:
        writer = None

    start_epoch = 0
    global_step = 0
    resume_path = Path(args.resume) if args.resume else checkpoint_dir / "latest.pt"
    if resume_path.exists():
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        raw_model.load_state_dict(checkpoint["model"], strict=True)
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        global_step = int(checkpoint.get("global_step", 0))
        if checkpoint.get("optimizer") is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if ema is not None and checkpoint.get("ema") is not None:
            ema.shadow.load_state_dict(checkpoint["ema"], strict=True)
        if lr_scheduler is not None and checkpoint.get("lr_scheduler") is not None:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    cfg_dropout = float(diffusion_cfg.get("cfg_dropout", 0.0))
    save_interval = int(train_cfg.get("save_interval", 1))

    for epoch in range(start_epoch, int(train_cfg["epochs"])):
        model.train()
        sampler.set_epoch(epoch)
        running_loss = 0.0
        num_batches = 0
        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{train_cfg['epochs']}",
            disable=rank != 0,
        )

        for batch in progress:
            ct_condition = batch["ct_cond"].to(device, non_blocking=True)
            cta_target = batch["cta_target"].to(device, non_blocking=True)
            z_position = batch["z_pos"].to(device, non_blocking=True)
            valid_mask = batch["valid_mask"].to(device, non_blocking=True)
            vessel_mask = batch["vessel_mask"].to(device, non_blocking=True)
            vessel_weight = batch["vessel_weight"].to(device, non_blocking=True)
            enhancement_target = batch["enhancement"].to(device, non_blocking=True)

            batch_size = ct_condition.shape[0]
            time_step = torch.rand(batch_size, device=device)
            noise = torch.randn_like(cta_target)

            with _autocast_context(device, amp_enabled, amp_dtype):
                noisy_cta, velocity_target = flow_scheduler.forward(
                    cta_target, noise, time_step
                )
                condition = ct_condition
                if cfg_dropout > 0:
                    drop = torch.rand(batch_size, 1, 1, 1, device=device) < cfg_dropout
                    condition = condition * (~drop).float()
                model_input = torch.cat([condition, noisy_cta], dim=1)

                if anatomy_guided:
                    outputs = model(model_input, time_step, z_position, return_aux=True)
                    loss_parts = anatomy_guided_loss(
                        outputs["velocity"],
                        velocity_target,
                        outputs["vessel_logits"],
                        valid_mask,
                        vessel_mask,
                        vessel_weight,
                        flow_vessel_factor=float(train_cfg.get("flow_vessel_factor", 4.0)),
                        scale_weights=train_cfg.get(
                            "locator_scale_weights", [1.0, 0.5, 0.25, 0.125, 0.0625]
                        ),
                        bce_coefficient=float(train_cfg.get("locator_bce_weight", 0.1)),
                        dice_coefficient=float(train_cfg.get("locator_dice_weight", 0.1)),
                        cldice_coefficient=float(train_cfg.get("locator_cldice_weight", 0.015)),
                        skeleton_iterations=int(train_cfg.get("cldice_iterations", 5)),
                        x_t=noisy_cta,
                        t=time_step,
                        x1_target=cta_target,
                        intensity_coefficient=float(train_cfg.get("intensity_coefficient", 0.0)),
                        vessel_enhancement=outputs.get("vessel_enhancement"),
                        enhancement_target=enhancement_target,
                        enhancement_coefficient=enhancement_coefficient,
                    )
                    loss = loss_parts["total"]
                else:
                    velocity = model(model_input, time_step, z_position)
                    loss = F.mse_loss(velocity, velocity_target)

            optimizer.zero_grad(set_to_none=True)
            if amp_enabled and amp_dtype == torch.float16:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip"]))
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip"]))
                optimizer.step()

            if ema is not None:
                ema.update(raw_model)

            running_loss += float(loss.item())
            num_batches += 1
            global_step += 1
            if rank == 0:
                progress.set_postfix(loss=f"{loss.item():.5f}")
                if writer is not None and global_step % 100 == 0:
                    writer.add_scalar("train/loss", loss.item(), global_step)
                    writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
                    if anatomy_guided:
                        for name, value in loss_parts.items():
                            writer.add_scalar(f"train/{name}", value.item(), global_step)

        if lr_scheduler is not None:
            lr_scheduler.step()

        if rank == 0:
            epoch_loss = running_loss / max(num_batches, 1)
            print(f"Epoch {epoch + 1} | loss: {epoch_loss:.5f}")
            if writer is not None:
                writer.add_scalar("train/epoch_loss", epoch_loss, epoch)
            if (epoch + 1) % save_interval == 0 or epoch + 1 == int(train_cfg["epochs"]):
                _save_checkpoint(
                    checkpoint_dir / "latest.pt",
                    raw_model,
                    ema,
                    optimizer,
                    lr_scheduler,
                    epoch,
                    global_step,
                )
                _save_checkpoint(
                    checkpoint_dir / f"epoch_{epoch + 1:03d}.pt",
                    raw_model,
                    ema,
                    optimizer,
                    lr_scheduler,
                    epoch,
                    global_step,
                    include_optimizer=False,
                )

        if ddp:
            dist.barrier(device_ids=[local_rank])

    if writer is not None:
        writer.close()
    if ddp:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    with open(args.config) as stream:
        config = yaml.safe_load(stream)
    train(config, args)


if __name__ == "__main__":
    main()
