from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class SpAWindowDataset(Dataset[dict[str, torch.Tensor | str]]):
    def __init__(self, manifest: str | Path, task: str, image_size: int) -> None:
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

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def _image_paths(path: Path) -> list[Path]:
        suffixes = {".png", ".jpg", ".jpeg", ".bmp"}
        return sorted(item for item in path.iterdir() if item.suffix.lower() in suffixes)

    def _load_rgb(self, directory: Path, start: int, end: int) -> torch.Tensor:
        paths = self._image_paths(directory)[start:end]
        if len(paths) != end - start:
            raise ValueError(f"RGB window [{start}:{end}] is incomplete in {directory}")
        images = []
        for path in paths:
            with Image.open(path) as image:
                image = image.convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
                array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
            images.append((torch.from_numpy(array) - self.mean) / self.std)
        return torch.stack(images)

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
        return {
            "rgb": self._load_rgb(Path(row["rgb_dir"]), rgb_start, rgb_end),
            "skeleton_features": torch.from_numpy(np.asarray(skeleton, dtype=np.float32).copy()),
            "rd_maps": torch.from_numpy(np.asarray(rd, dtype=np.float32).copy()),
            "label": torch.tensor(int(row[self.label_column]), dtype=torch.long),
            "subject_id": row["subject_id"],
            "session": row["session"],
        }

