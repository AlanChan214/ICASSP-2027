from collections import OrderedDict
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import rotate as ndimage_rotate
from torch.utils.data import Dataset


ENHANCEMENT_SCALE_HU = 150.0


def _window(hu: np.ndarray, center: float, width: float) -> np.ndarray:
    hu_min = center - width / 2
    hu_max = center + width / 2
    return (np.clip(hu, hu_min, hu_max) - hu_min) / width * 2 - 1


class CT2CTADataset2D(Dataset):
    """2.5D dataset: brain-window CT context → 1 CTA slice"""

    def __init__(
        self,
        data_list,
        augment=False,
        num_slices_context=5,
        ct_brain_window=(90, 35),
        cta_window=(400, 40),
        z_weight_middle=1.5,
        z_middle_range=(0.25, 0.75),
        vessel_density_weight: float = 0.0,
        cache_size_cases: int = 1,
        rotation_prob: float = 0.0,
        crop_size=None,
    ):
        self.data_list = data_list
        self.augment = augment
        self.num_slices_context = int(num_slices_context)
        if self.num_slices_context % 2 == 0:
            raise ValueError("num_slices_context must be odd for symmetric 2.5D context")
        self.ct_brain_window = tuple(ct_brain_window)
        self.cta_window = tuple(cta_window)
        self.z_weight_middle = float(z_weight_middle)
        self.z_middle_range = tuple(z_middle_range)
        self.vessel_density_weight = float(vessel_density_weight)
        self.cache_size_cases = max(int(cache_size_cases), 0)
        self.rotation_prob = float(rotation_prob)
        if crop_size is None:
            self.crop_size = None
        elif isinstance(crop_size, int):
            self.crop_size = (crop_size, crop_size)
        else:
            self.crop_size = tuple(int(v) for v in crop_size)
        self._cache = OrderedDict()
        self._index = []  # list of (list_idx, z)
        rng = np.random.default_rng(0)

        for i, entry in enumerate(data_list):
            missing = {"ct", "cta"} - set(entry)
            if missing:
                raise KeyError(f"Manifest entry {i} is missing required fields: {sorted(missing)}")
            ct_img = nib.load(entry["ct"])
            cta_img = nib.load(entry["cta"])
            if ct_img.shape != cta_img.shape:
                raise ValueError(
                    f"Shape mismatch for {entry.get('case_id', i)}: "
                    f"CT {ct_img.shape}, CTA {cta_img.shape}"
                )
            if entry.get("vessel_mask"):
                vessel_img = nib.load(entry["vessel_mask"])
                if ct_img.shape != vessel_img.shape:
                    raise ValueError(
                        f"Shape mismatch for {entry.get('case_id', i)}: "
                        f"CT {ct_img.shape}, vessel mask {vessel_img.shape}"
                    )
            D = int(ct_img.shape[2])
            mid_lo = int(self.z_middle_range[0] * D)
            mid_hi = int(self.z_middle_range[1] * D)
            if self.vessel_density_weight > 0:
                cta_proxy = nib.load(entry["cta"]).dataobj
            for z in range(D):
                weight = self.z_weight_middle if mid_lo <= z < mid_hi else 1.0
                if self.vessel_density_weight > 0:
                    cta_sl = np.asarray(cta_proxy[:, :, z], dtype=np.float32)
                    cta_norm = _window(cta_sl, center=self.cta_window[1], width=self.cta_window[0])
                    vessel_frac = float((cta_norm > 0.3).mean())
                    weight = weight * (1.0 + self.vessel_density_weight * vessel_frac * 10.0)
                repeats = max(1, int(np.floor(weight)))
                frac = float(weight - np.floor(weight))
                if frac > 0 and rng.random() < frac:
                    repeats += 1
                for _ in range(repeats):
                    self._index.append((i, z))

    def _load(self, idx: int):
        if idx in self._cache:
            self._cache.move_to_end(idx)
            return self._cache[idx]

        entry = self.data_list[idx]
        ct = nib.load(entry["ct"]).get_fdata(dtype=np.float32)
        cta = nib.load(entry["cta"]).get_fdata(dtype=np.float32)
        valid_mask_3d = ((ct > -900) & (cta > -900)).astype(np.uint8)
        if entry.get("vessel_mask"):
            vessel_mask_3d = (
                nib.load(entry["vessel_mask"]).get_fdata(dtype=np.float32) > 0.5
            ).astype(np.uint8)
        else:
            vessel_mask_3d = np.zeros_like(valid_mask_3d, dtype=np.uint8)
        vessel_mask_3d *= valid_mask_3d
        loaded = (ct, cta, valid_mask_3d, vessel_mask_3d)

        if self.cache_size_cases > 0:
            self._cache[idx] = loaded
            self._cache.move_to_end(idx)
            while len(self._cache) > self.cache_size_cases:
                self._cache.popitem(last=False)

        return loaded

    def _load_ct(self, idx: int) -> np.ndarray:
        return self._load(idx)[0]

    def __len__(self):
        return len(self._index)

    def __getitem__(self, item):
        list_idx, z = self._index[item]
        ct_vol, cta_vol, valid_mask_3d, vessel_mask_3d = self._load(list_idx)
        D = ct_vol.shape[2]
        entry = self.data_list[list_idx]
        case_id = entry.get("case_id") or Path(entry["ct"]).name.replace(".nii.gz", "").replace(".nii", "")

        half = self.num_slices_context // 2
        offsets = list(range(-half, half + 1))
        brain_slices = []
        for off in offsets:
            z_clamped = int(np.clip(z + off, 0, D - 1))
            sl = ct_vol[:, :, z_clamped]
            brain_slices.append(_window(sl, center=self.ct_brain_window[1], width=self.ct_brain_window[0]))

        ct_cond = np.stack(brain_slices, axis=0).astype(np.float32)

        valid_mask = valid_mask_3d[:, :, z][np.newaxis]
        vessel_mask = vessel_mask_3d[:, :, z][np.newaxis]
        ct_hu = ct_vol[:, :, z][np.newaxis].astype(np.float32)

        cta_sl = cta_vol[:, :, z]
        cta_target = _window(
            cta_sl,
            center=self.cta_window[1],
            width=self.cta_window[0],
        )[np.newaxis].astype(np.float32)
        enhancement = np.clip(
            (cta_sl[np.newaxis].astype(np.float32) - ct_hu) / ENHANCEMENT_SCALE_HU,
            0.0,
            1.0,
        ).astype(np.float32)
        z_pos = np.float32(z / max(D - 1, 1))

        if self.crop_size is not None:
            crop_h, crop_w = self.crop_size
            height, width = cta_target.shape[1:]
            if crop_h > height or crop_w > width:
                raise ValueError(f"crop_size {self.crop_size} exceeds slice shape {(height, width)}")
            y0, x0 = (height - crop_h) // 2, (width - crop_w) // 2
            spatial = (slice(y0, y0 + crop_h), slice(x0, x0 + crop_w))
            ct_cond = ct_cond[:, spatial[0], spatial[1]]
            cta_target = cta_target[:, spatial[0], spatial[1]]
            valid_mask = valid_mask[:, spatial[0], spatial[1]]
            vessel_mask = vessel_mask[:, spatial[0], spatial[1]]
            ct_hu = ct_hu[:, spatial[0], spatial[1]]
            enhancement = enhancement[:, spatial[0], spatial[1]]

        if self.augment:
            if np.random.random() < 0.5:
                ct_cond = ct_cond[:, :, ::-1].copy()
                cta_target = cta_target[:, :, ::-1].copy()
                valid_mask = valid_mask[:, :, ::-1].copy()
                vessel_mask = vessel_mask[:, :, ::-1].copy()
                ct_hu = ct_hu[:, :, ::-1].copy()
                enhancement = enhancement[:, :, ::-1].copy()
            if self.rotation_prob > 0 and np.random.random() < self.rotation_prob:
                angle = np.random.uniform(-10, 10)
                ct_cond = np.stack(
                    [ndimage_rotate(sl, angle, reshape=False, order=1, mode="nearest") for sl in ct_cond],
                    axis=0,
                )
                cta_target = ndimage_rotate(cta_target[0], angle, reshape=False, order=1, mode="nearest")[np.newaxis]
                valid_mask = ndimage_rotate(
                    valid_mask[0],
                    angle,
                    reshape=False,
                    order=0,
                    mode="nearest",
                )[np.newaxis]
                vessel_mask = ndimage_rotate(
                    vessel_mask[0], angle, reshape=False, order=0, mode="nearest"
                )[np.newaxis]
                ct_hu = ndimage_rotate(
                    ct_hu[0], angle, reshape=False, order=1, mode="constant", cval=-1000.0
                )[np.newaxis]
                enhancement = ndimage_rotate(
                    enhancement[0], angle, reshape=False, order=1, mode="constant", cval=0.0
                )[np.newaxis]
            if np.random.random() < 0.3:
                ct_cond = np.clip(ct_cond + np.random.uniform(-0.05, 0.05), -1.0, 1.0).astype(np.float32)
            if np.random.random() < 0.3:
                noise_std = np.random.uniform(0.005, 0.02)
                ct_cond = np.clip(
                    ct_cond + np.random.normal(0, noise_std, ct_cond.shape).astype(np.float32),
                    -1.0, 1.0,
                )
        valid_mask = (valid_mask > 0.5).astype(np.float32)
        vessel_mask = (vessel_mask > 0.5).astype(np.float32) * valid_mask
        enhancement = np.clip(enhancement, 0.0, 1.0) * valid_mask

        return {
            "enhancement": torch.from_numpy(np.ascontiguousarray(enhancement, dtype=np.float32)),
            "ct_cond": torch.from_numpy(ct_cond),
            "cta_target": torch.from_numpy(cta_target),
            "valid_mask": torch.from_numpy(valid_mask),
            "vessel_mask": torch.from_numpy(vessel_mask),
            "vessel_weight": torch.tensor(float(entry.get("vessel_weight", 1.0)), dtype=torch.float32),
            "brain_mask": torch.from_numpy(valid_mask),
            "z_pos": torch.tensor(z_pos, dtype=torch.float32),
            "z_idx": z,
            "case_id": case_id,
        }
