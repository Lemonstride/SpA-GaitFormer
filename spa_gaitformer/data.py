from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from spa_gaitformer.config import DataConfig, TaskConfig


@dataclass
class SampleRecord:
    sample_id: str
    subject_id: str
    session: str
    session_dir: str
    rgb_path: str
    skeleton_path: str
    disease_label: int
    severity_label: float


def load_records_for_split(
    split: str,
    data_config: DataConfig,
) -> list[SampleRecord]:
    manifest_path = getattr(data_config, f"{split}_manifest", None)
    if manifest_path:
        return read_manifest(manifest_path)

    if not data_config.dataset_root:
        raise ValueError(
            "Either data.dataset_root or split-specific manifest paths must be configured."
        )

    split_path = getattr(data_config, f"{split}_split", None)
    if not split_path:
        raise ValueError(
            f"Missing data.{split}_split. Provide a subject split file for '{split}'."
        )

    subject_entries = read_split_file(split_path)
    return build_records_from_processed_root(
        dataset_root=Path(data_config.dataset_root),
        split_entries=subject_entries,
        data_config=data_config,
    )


def read_manifest(path: str | Path) -> list[SampleRecord]:
    records: list[SampleRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "sample_id",
            "rgb_path",
            "skeleton_path",
            "disease_label",
            "severity_label",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            missing_columns = ", ".join(sorted(missing))
            raise ValueError(f"Manifest {path} is missing columns: {missing_columns}")

        for row in reader:
            severity_raw = (row.get("severity_label") or "").strip()
            sample_id = row["sample_id"].strip()
            subject_id, session = _split_sample_id(sample_id)
            records.append(
                SampleRecord(
                    sample_id=sample_id,
                    subject_id=subject_id,
                    session=session,
                    session_dir=row.get("session_dir", ""),
                    rgb_path=row["rgb_path"],
                    skeleton_path=row["skeleton_path"],
                    disease_label=parse_binary_label(row["disease_label"]),
                    severity_label=parse_severity_label(severity_raw),
                )
            )
    return records


def read_split_file(path: str | Path) -> list[str]:
    entries: list[str] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            entries.append(value)
    if not entries:
        raise ValueError(f"Split file is empty: {path}")
    return entries


def build_records_from_processed_root(
    dataset_root: Path,
    split_entries: list[str],
    data_config: DataConfig,
) -> list[SampleRecord]:
    records: list[SampleRecord] = []
    for entry in split_entries:
        subject_id, session = _parse_split_entry(entry, data_config.session)
        session_dir = dataset_root / subject_id / session
        rgb_path = session_dir / data_config.rgb_dir
        skeleton_path = session_dir / data_config.skeleton_file
        binary_label_path = session_dir / data_config.binary_label_file
        severity_label_path = session_dir / data_config.severity_label_file

        required_paths = {
            "session": session_dir,
            "rgb": rgb_path,
            "skeleton": skeleton_path,
            "binary_label": binary_label_path,
        }
        missing = [name for name, current in required_paths.items() if not current.exists()]
        if missing:
            message = (
                f"Missing required files for {subject_id}/{session}: {', '.join(missing)}"
            )
            if data_config.strict_missing_files:
                raise FileNotFoundError(message)
            print(f"[skip] {message}")
            continue

        severity_label = -1.0
        if severity_label_path.exists():
            severity_label = parse_severity_label(severity_label_path.read_text(encoding="utf-8"))

        sample_id = f"{subject_id}/{session}"
        records.append(
            SampleRecord(
                sample_id=sample_id,
                subject_id=subject_id,
                session=session,
                session_dir=str(session_dir),
                rgb_path=str(rgb_path),
                skeleton_path=str(skeleton_path),
                disease_label=parse_binary_label(
                    binary_label_path.read_text(encoding="utf-8")
                ),
                severity_label=severity_label,
            )
        )

    if not records:
        raise ValueError("No valid samples were found for the requested split.")
    return records


def parse_binary_label(raw: str) -> int:
    normalized = raw.strip().lower()
    if normalized in {"0", "healthy", "normal", "control", "negative"}:
        return 0
    if normalized in {"1", "spa", "patient", "positive", "diseased"}:
        return 1

    try:
        numeric = int(float(normalized))
    except ValueError as exc:
        raise ValueError(f"Unsupported binary label value: {raw!r}") from exc
    if numeric not in {0, 1}:
        raise ValueError(f"Binary label must be 0 or 1, got: {raw!r}")
    return numeric


def parse_severity_label(raw: str) -> float:
    normalized = raw.strip().lower()
    if normalized in {"", "na", "nan", "none", "unknown"}:
        return -1.0
    try:
        return float(normalized)
    except ValueError as exc:
        raise ValueError(f"Unsupported severity label value: {raw!r}") from exc


def _split_sample_id(sample_id: str) -> tuple[str, str]:
    normalized = sample_id.replace("\\", "/")
    parts = [part for part in normalized.split("/") if part]
    if len(parts) >= 2:
        return parts[0], parts[1]
    return parts[0], "walk"


def _parse_split_entry(entry: str, default_session: str) -> tuple[str, str]:
    normalized = entry.replace("\\", "/")
    parts = [part for part in normalized.split("/") if part]
    if len(parts) == 1:
        return parts[0], default_session
    return parts[0], parts[1]


def _uniform_frame_indices(length: int, num_frames: int) -> torch.Tensor:
    if length <= 0:
        raise ValueError("Sequence length must be positive.")
    if length == num_frames:
        return torch.arange(length)
    positions = torch.linspace(0, length - 1, steps=num_frames)
    return positions.round().long()


def _resize_rgb_tensor(rgb: torch.Tensor, image_size: int) -> torch.Tensor:
    return F.interpolate(
        rgb,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )


def _to_channel_first_video(video: torch.Tensor) -> torch.Tensor:
    if video.ndim != 4:
        raise ValueError(f"Expected 4D RGB tensor, got shape {tuple(video.shape)}.")
    if video.shape[1] == 3:
        return video.float()
    if video.shape[-1] == 3:
        return video.permute(0, 3, 1, 2).float()
    raise ValueError(
        "RGB tensor must be [T, C, H, W] or [T, H, W, C] with 3 channels."
    )


def _load_optional_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "numpy is required to read .npy files. Install dependencies first."
        ) from exc
    return np


def load_rgb_sequence(path: str | Path, num_frames: int, image_size: int) -> torch.Tensor:
    rgb_path = Path(path)
    if rgb_path.is_dir():
        try:
            from PIL import Image
            import numpy as np
        except ImportError as exc:
            raise ImportError(
                "Pillow and numpy are required to read RGB frame directories."
            ) from exc

        frame_paths = sorted(
            candidate
            for candidate in rgb_path.iterdir()
            if candidate.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
        )
        if not frame_paths:
            raise ValueError(f"No image frames found in {rgb_path}.")
        indices = _uniform_frame_indices(len(frame_paths), num_frames)
        frames = []
        for index in indices.tolist():
            image = Image.open(frame_paths[index]).convert("RGB")
            frame = torch.from_numpy(np.asarray(image)).permute(2, 0, 1).float()
            frames.append(frame)
        rgb = torch.stack(frames, dim=0)
    elif rgb_path.suffix.lower() in {".pt", ".pth"}:
        rgb = torch.as_tensor(torch.load(rgb_path, map_location="cpu"))
        rgb = _to_channel_first_video(rgb)
        indices = _uniform_frame_indices(rgb.shape[0], num_frames)
        rgb = rgb.index_select(0, indices)
    elif rgb_path.suffix.lower() == ".npy":
        np = _load_optional_numpy()
        rgb = torch.from_numpy(np.load(rgb_path))
        rgb = _to_channel_first_video(rgb)
        indices = _uniform_frame_indices(rgb.shape[0], num_frames)
        rgb = rgb.index_select(0, indices)
    else:
        raise ValueError(
            "Unsupported RGB input. Use a frame directory, .pt, .pth, or .npy file."
        )

    rgb = rgb / 255.0 if rgb.max() > 1.0 else rgb
    return _resize_rgb_tensor(rgb, image_size)


def _to_temporal_joints(tensor: torch.Tensor, joint_dim: int) -> torch.Tensor:
    if tensor.ndim == 4:
        tensor = tensor[0]
    if tensor.ndim != 3:
        raise ValueError(
            f"Expected 3D skeleton tensor, got shape {tuple(tensor.shape)}."
        )
    if tensor.shape[-1] == joint_dim:
        return tensor.float()
    if tensor.shape[1] == joint_dim:
        return tensor.permute(0, 2, 1).float()
    raise ValueError(
        "Skeleton tensor must be [T, J, C] or [T, C, J] with matching coordinate dim."
    )


def load_skeleton_sequence(
    path: str | Path,
    num_frames: int,
    max_joints: int,
    joint_dim: int,
    normalize: bool,
) -> torch.Tensor:
    skeleton_path = Path(path)
    if skeleton_path.suffix.lower() in {".pt", ".pth"}:
        skeleton = torch.as_tensor(torch.load(skeleton_path, map_location="cpu"))
    elif skeleton_path.suffix.lower() == ".npy":
        np = _load_optional_numpy()
        skeleton = torch.from_numpy(np.load(skeleton_path))
    elif skeleton_path.suffix.lower() == ".json":
        with skeleton_path.open("r", encoding="utf-8") as handle:
            skeleton = torch.tensor(json.load(handle), dtype=torch.float32)
    else:
        raise ValueError(
            "Unsupported skeleton input. Use .pt, .pth, .npy, or .json."
        )

    skeleton = _to_temporal_joints(torch.as_tensor(skeleton), joint_dim)
    frame_indices = _uniform_frame_indices(skeleton.shape[0], num_frames)
    skeleton = skeleton.index_select(0, frame_indices)

    if skeleton.shape[1] > max_joints:
        skeleton = skeleton[:, :max_joints]
    elif skeleton.shape[1] < max_joints:
        pad_joints = max_joints - skeleton.shape[1]
        skeleton = F.pad(skeleton, (0, 0, 0, pad_joints))

    if normalize:
        root = skeleton[:, :1, :]
        skeleton = skeleton - root
        scale = (
            skeleton.flatten(0, 1)
            .std(dim=0, unbiased=False)
            .mean()
            .clamp(min=1e-6)
        )
        skeleton = skeleton / scale

    return skeleton.float()


class SpAMMDDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        split: str,
        data_config: DataConfig,
        task_config: TaskConfig,
    ) -> None:
        self.records = load_records_for_split(split, data_config)
        self.data_config = data_config
        self.task_config = task_config

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        rgb = load_rgb_sequence(
            record.rgb_path,
            num_frames=self.data_config.num_frames,
            image_size=self.data_config.image_size,
        )
        skeleton = load_skeleton_sequence(
            record.skeleton_path,
            num_frames=self.data_config.num_frames,
            max_joints=self.data_config.max_joints,
            joint_dim=self.data_config.joint_dim,
            normalize=self.data_config.normalize_skeleton,
        )

        rgb_mean = torch.tensor(self.data_config.rgb_mean).view(1, 3, 1, 1)
        rgb_std = torch.tensor(self.data_config.rgb_std).view(1, 3, 1, 1)
        rgb = (rgb - rgb_mean) / rgb_std

        severity_valid = float(record.severity_label >= 0.0)
        if self.task_config.severity_mode == "classification":
            severity_target = int(record.severity_label) if severity_valid else -1
            severity_tensor = torch.tensor(severity_target, dtype=torch.long)
        elif self.task_config.severity_mode == "regression":
            severity_value = record.severity_label if severity_valid else 0.0
            severity_tensor = torch.tensor(severity_value, dtype=torch.float32)
        else:
            raise ValueError(
                "task.severity_mode must be 'classification' or 'regression'."
            )

        return {
            "sample_id": record.sample_id,
            "rgb": rgb,
            "skeleton": skeleton,
            "disease_label": torch.tensor(record.disease_label, dtype=torch.float32),
            "severity_label": severity_tensor,
            "severity_mask": torch.tensor(severity_valid, dtype=torch.float32),
        }
