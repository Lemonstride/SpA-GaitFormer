from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.nn import functional as F
from torch.utils.data import Dataset


# A label-blind bank for the new from-scratch experiment. Each appearance profile
# has an unmirrored and mirrored view; geometry shrinks into padding and never crops.
_RGB32_PROFILES = (
    (1.00, 1.00, 1.00, 1.00, 1.00, 0.0, 0.0),
    (0.92, 1.08, 1.00, 1.00, 1.00, 0.0, 0.0),
    (1.08, 0.94, 1.00, 1.00, 1.00, 0.0, 0.0),
    (1.00, 1.00, 0.90, 1.00, 1.00, 0.0, 0.0),
    (1.00, 1.00, 1.08, 1.00, 1.00, 0.0, 0.0),
    (1.00, 1.00, 1.00, 0.94, 1.00, 0.0, 0.0),
    (1.00, 1.00, 1.00, 1.06, 1.00, 0.0, 0.0),
    (0.96, 1.04, 0.95, 1.00, 0.98, 0.0, 0.0),
    (1.04, 0.96, 1.04, 1.00, 0.98, -1.0, 0.0),
    (0.94, 1.02, 1.04, 1.00, 0.98, 1.0, 0.0),
    (1.06, 0.98, 0.96, 1.00, 0.96, 0.0, -1.0),
    (0.98, 1.06, 0.96, 1.00, 0.96, 0.0, 1.0),
    (1.02, 0.94, 1.02, 0.97, 0.96, -1.0, -1.0),
    (0.96, 1.02, 1.00, 1.03, 0.96, 1.0, 1.0),
    (1.04, 1.02, 0.94, 1.00, 0.94, -1.0, 1.0),
    (0.96, 0.98, 1.06, 1.00, 0.94, 1.0, -1.0),
)


class SpAWindowDataset(Dataset[dict[str, torch.Tensor | str]]):
    def __init__(
        self,
        manifest: str | Path,
        task: str,
        image_size: int,
        augmentation: dict[str, Any] | None = None,
        rd_normalization: str = "none",
        headturn_enabled: bool = False,
    ) -> None:
        self.manifest = Path(manifest).expanduser().resolve()
        with self.manifest.open(newline="", encoding="utf-8-sig") as handle:
            self.rows = list(csv.DictReader(handle))
        if not self.rows:
            raise ValueError(f"Empty manifest: {self.manifest}")
        if task not in {"binary", "severity"}:
            raise ValueError(f"Unknown task: {task}")
        self.label_column = f"{task}_label"
        self.image_size = int(image_size)
        self.mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        self.std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        self.augmentation = augmentation if augmentation and augmentation.get("enabled", True) else None
        self.rd_normalization = rd_normalization
        self.headturn_enabled = bool(headturn_enabled)
        self.headturn_mean = 0.0
        self.headturn_std = 1.0
        if self.headturn_enabled:
            self.headturn_values_by_subject()
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def headturn_values_by_subject(self) -> dict[str, float]:
        values: dict[str, float] = {}
        for row in self.rows:
            subject = row["subject_id"]
            raw_value = row.get("headturn_span_deg", "").strip()
            if not raw_value:
                raise ValueError(f"Missing headturn_span_deg for {subject} in {self.manifest}")
            value = float(raw_value)
            if not math.isfinite(value):
                raise ValueError(f"Non-finite headturn_span_deg for {subject}: {raw_value}")
            if subject in values and not math.isclose(values[subject], value, abs_tol=1e-6):
                raise ValueError(f"Inconsistent headturn_span_deg values for {subject}")
            values[subject] = value
        return values

    def set_headturn_normalization(self, mean: float, std: float) -> None:
        if not math.isfinite(mean) or not math.isfinite(std) or std <= 0.0:
            raise ValueError("Head-turn normalization requires a finite mean and positive std")
        self.headturn_mean = float(mean)
        self.headturn_std = float(std)

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def _image_paths(path: Path) -> list[Path]:
        suffixes = {".png", ".jpg", ".jpeg", ".bmp"}
        return sorted(item for item in path.iterdir() if item.suffix.lower() in suffixes)

    @staticmethod
    def _uniform(low: float, high: float) -> float:
        if low == high:
            return low
        return float(torch.empty(1).uniform_(low, high).item())

    def _sample_rgb_transform(self, subject_id: str = "") -> dict[str, float | bool | str]:
        if self.augmentation is None:
            return {}
        policy = str(self.augmentation.get("policy", "random_per_window"))
        if policy == "subject_cycle_rgb32_v1":
            seed = int(self.augmentation.get("seed", 2026))
            digest = hashlib.sha256(f"{seed}:{subject_id}".encode()).digest()
            offset = int.from_bytes(digest[:8], "big")
            view_index = (self.epoch + offset) % (len(_RGB32_PROFILES) * 2)
            profile = _RGB32_PROFILES[view_index // 2]
            brightness, contrast, saturation, gamma, scale, shift_x, shift_y = profile
            return {
                "policy": policy,
                "view": f"view_{view_index:02d}",
                "flip": bool(view_index % 2),
                "brightness": brightness,
                "contrast": contrast,
                "saturation": saturation,
                "gamma": gamma,
                "scale": scale,
                "shift_x": shift_x,
                "shift_y": shift_y,
            }
        if policy != "random_per_window":
            raise ValueError(f"Unknown augmentation policy: {policy}")
        cfg = self.augmentation.get("rgb", {})
        crop_scale_min = float(cfg.get("crop_scale_min", 1.0))
        brightness = float(cfg.get("brightness", 0.0))
        contrast = float(cfg.get("contrast", 0.0))
        saturation = float(cfg.get("saturation", 0.0))
        return {
            "crop_scale": self._uniform(crop_scale_min, 1.0),
            "crop_x": self._uniform(0.0, 1.0),
            "crop_y": self._uniform(0.0, 1.0),
            "flip": bool(torch.rand(()) < float(cfg.get("horizontal_flip_p", 0.0))),
            "brightness": self._uniform(1.0 - brightness, 1.0 + brightness),
            "contrast": self._uniform(1.0 - contrast, 1.0 + contrast),
            "saturation": self._uniform(1.0 - saturation, 1.0 + saturation),
        }

    @staticmethod
    def _placement(
        width: int, height: int, scale: float, shift_x: float, shift_y: float
    ) -> tuple[int, int, int, int]:
        if not 0 < scale <= 1 or abs(shift_x) > 1 or abs(shift_y) > 1:
            raise ValueError("Non-cropping geometry requires scale in (0, 1] and bounded shifts")
        target_width = max(1, round(width * scale))
        target_height = max(1, round(height * scale))
        left = round((width - target_width) * (1 + shift_x) / 2)
        top = round((height - target_height) * (1 + shift_y) / 2)
        return left, top, target_width, target_height

    def _transform_rgb_image(
        self, image: Image.Image, parameters: dict[str, float | bool | str]
    ) -> Image.Image:
        if parameters and parameters.get("policy") == "subject_cycle_rgb32_v1":
            if bool(parameters["flip"]):
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            image = ImageEnhance.Brightness(image).enhance(float(parameters["brightness"]))
            image = ImageEnhance.Contrast(image).enhance(float(parameters["contrast"]))
            image = ImageEnhance.Color(image).enhance(float(parameters["saturation"]))
            gamma = float(parameters["gamma"])
            if gamma != 1.0:
                lut = [round(255 * (value / 255) ** gamma) for value in range(256)]
                image = image.point(lut * 3)
            width, height = image.size
            left, top, target_width, target_height = self._placement(
                width,
                height,
                float(parameters["scale"]),
                float(parameters["shift_x"]),
                float(parameters["shift_y"]),
            )
            if (target_width, target_height) != image.size:
                resized = image.resize((target_width, target_height), Image.Resampling.BICUBIC)
                canvas = Image.new("RGB", image.size, (0, 0, 0))
                canvas.paste(resized, (left, top))
                image = canvas
        elif parameters:
            width, height = image.size
            side_scale = math.sqrt(float(parameters["crop_scale"]))
            crop_width = max(1, round(width * side_scale))
            crop_height = max(1, round(height * side_scale))
            left = round((width - crop_width) * float(parameters["crop_x"]))
            top = round((height - crop_height) * float(parameters["crop_y"]))
            image = image.crop((left, top, left + crop_width, top + crop_height))
            if bool(parameters["flip"]):
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            image = ImageEnhance.Brightness(image).enhance(float(parameters["brightness"]))
            image = ImageEnhance.Contrast(image).enhance(float(parameters["contrast"]))
            image = ImageEnhance.Color(image).enhance(float(parameters["saturation"]))
        return image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)

    def _load_rgb(
        self,
        directory: Path,
        start: int,
        end: int,
        parameters: dict[str, float | bool | str] | None = None,
    ) -> torch.Tensor:
        paths = self._image_paths(directory)[start:end]
        if len(paths) != end - start:
            raise ValueError(f"RGB window [{start}:{end}] is incomplete in {directory}")
        images = []
        parameters = parameters if parameters is not None else self._sample_rgb_transform()
        for path in paths:
            with Image.open(path) as image:
                image = self._transform_rgb_image(image.convert("RGB"), parameters)
                array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
            images.append((torch.from_numpy(array) - self.mean) / self.std)
        return torch.stack(images)

    def _augment_skeleton(
        self,
        features: torch.Tensor,
        parameters: dict[str, float | bool | str],
    ) -> torch.Tensor:
        if self.augmentation is None:
            return features
        if bool(parameters.get("flip", False)) and features.ndim == 4:
            features = torch.flip(features, dims=(-1,))
        if parameters.get("policy") == "subject_cycle_rgb32_v1" and features.ndim == 4:
            scale = float(parameters["scale"])
            if scale != 1.0:
                height, width = features.shape[-2:]
                left, top, target_width, target_height = self._placement(
                    width,
                    height,
                    scale,
                    float(parameters["shift_x"]),
                    float(parameters["shift_y"]),
                )
                resized = F.interpolate(
                    features,
                    size=(target_height, target_width),
                    mode="bilinear",
                    align_corners=False,
                )
                canvas = features.new_zeros(features.shape)
                canvas[..., top : top + target_height, left : left + target_width] = resized
                features = canvas
        cfg = self.augmentation.get("skeleton", {})
        noise_std = float(cfg.get("noise_std", 0.0))
        if noise_std > 0:
            scale = features.float().std(unbiased=False).clamp_min(1e-6)
            features = features + torch.randn_like(features) * (noise_std * scale)
        dropout_p = float(cfg.get("feature_dropout_p", 0.0))
        if dropout_p > 0:
            keep = torch.rand_like(features) >= dropout_p
            features = features * keep / (1.0 - dropout_p)
        return features

    def _augment_rd(self, maps: torch.Tensor) -> torch.Tensor:
        if self.augmentation is None:
            return maps
        cfg = self.augmentation.get("radar", {})
        scale_range = cfg.get("amplitude_scale", [1.0, 1.0])
        maps = maps * self._uniform(float(scale_range[0]), float(scale_range[1]))
        noise_std = float(cfg.get("noise_std", 0.0))
        if noise_std > 0:
            scale = maps.float().std(unbiased=False).clamp_min(1e-6)
            maps = maps + torch.randn_like(maps) * (noise_std * scale)
        return maps

    def _normalize_rd(self, maps: torch.Tensor) -> torch.Tensor:
        if self.rd_normalization == "none":
            return maps
        if self.rd_normalization == "per_window":
            return (maps - maps.mean()) / maps.std(unbiased=False).clamp_min(1e-6)
        raise ValueError(f"Unknown RD normalization: {self.rd_normalization}")

    def _apply_shared_temporal_mask(
        self,
        rgb: torch.Tensor,
        skeleton: torch.Tensor,
        rd: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.augmentation is None:
            return rgb, skeleton, rd
        cfg = self.augmentation.get("temporal", {})
        probability = float(cfg.get("mask_p", 0.0))
        if rd.size(0) == 0 or torch.rand(()) >= probability:
            return rgb, skeleton, rd
        max_fraction = float(cfg.get("max_mask_fraction", 0.1))
        count = max(1, min(rd.size(0), math.ceil(rd.size(0) * max_fraction)))
        indices = torch.randperm(rd.size(0))[:count]
        rgb = rgb.clone()
        skeleton = skeleton.clone()
        rd = rd.clone()
        rd[indices] = 0
        for index in indices.tolist():
            rgb[index * 3 : (index + 1) * 3] = 0
            skeleton[index * 3 : (index + 1) * 3] = 0
        return rgb, skeleton, rd

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.rows[index]
        rgb_start, rgb_end = int(row["rgb_start"]), int(row["rgb_end"])
        rd_start, rd_end = int(row["rd_start"]), int(row["rd_end"])
        skeleton = np.load(row["skeleton_path"], mmap_mode="r")[rgb_start:rgb_end]
        rd = np.load(row["rd_path"], mmap_mode="r")[rd_start:rd_end]
        if skeleton.shape[0] != rgb_end - rgb_start or rd.shape[0] != rd_end - rd_start:
            raise ValueError(f"Incomplete array window in row {index} of {self.manifest}")
        if rd.ndim == 3:
            rd = rd[:, None, :, :]
        if rd.ndim != 4 or rd.shape[1] != 1:
            raise ValueError(f"RD map must be [T,H,W] or [T,1,H,W], got {rd.shape}")
        rgb_parameters = self._sample_rgb_transform(row["subject_id"])
        rgb_tensor = self._load_rgb(
            Path(row["rgb_dir"]), rgb_start, rgb_end, rgb_parameters
        )
        skeleton_tensor = self._augment_skeleton(
            torch.from_numpy(np.asarray(skeleton, dtype=np.float32).copy()),
            rgb_parameters,
        )
        rd_tensor = self._normalize_rd(
            torch.from_numpy(np.asarray(rd, dtype=np.float32).copy())
        )
        rd_tensor = self._augment_rd(rd_tensor)
        rgb_tensor, skeleton_tensor, rd_tensor = self._apply_shared_temporal_mask(
            rgb_tensor, skeleton_tensor, rd_tensor
        )
        sample: dict[str, torch.Tensor | str] = {
            "rgb": rgb_tensor,
            "skeleton_features": skeleton_tensor,
            "rd_maps": rd_tensor,
            "label": torch.tensor(int(row[self.label_column]), dtype=torch.long),
            "subject_id": row["subject_id"],
            "session": row["session"],
            "augmentation_view": str(rgb_parameters.get("view", "original")),
        }
        if self.headturn_enabled:
            value = (float(row["headturn_span_deg"]) - self.headturn_mean) / self.headturn_std
            sample["headturn"] = torch.tensor([value], dtype=torch.float32)
        return sample

