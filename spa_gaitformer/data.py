from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
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
    window_start: int | None = None
    window_end: int | None = None


def load_records_for_split(
    split: str,
    data_config: DataConfig,
) -> list[SampleRecord]:
    window_manifest_path = getattr(data_config, f"{split}_window_manifest", None)
    if window_manifest_path:
        return read_window_manifest(window_manifest_path, data_config)

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


def create_stratified_folds(
    records: list[SampleRecord],
    num_folds: int,
    seed: int,
) -> list[tuple[list[SampleRecord], list[SampleRecord]]]:
    if num_folds < 2:
        raise ValueError("num_folds must be at least 2.")
    subject_groups = _group_records_by_subject(records)
    if len(subject_groups) < num_folds:
        raise ValueError("Number of subjects must be >= num_folds.")

    grouped: dict[int, list[list[SampleRecord]]] = defaultdict(list)
    for subject_id in sorted(subject_groups):
        subject_records = sorted(
            subject_groups[subject_id],
            key=lambda record: record.sample_id,
        )
        grouped[_subject_stratify_label(subject_records)].append(subject_records)

    rng = random.Random(seed)
    fold_buckets: list[list[list[SampleRecord]]] = [[] for _ in range(num_folds)]
    for label in sorted(grouped):
        label_groups = grouped[label][:]
        rng.shuffle(label_groups)
        for index, subject_records in enumerate(label_groups):
            fold_buckets[index % num_folds].append(subject_records)

    folds: list[tuple[list[SampleRecord], list[SampleRecord]]] = []
    for fold_index in range(num_folds):
        val_records = sorted(
            [
                record
                for subject_records in fold_buckets[fold_index]
                for record in subject_records
            ],
            key=lambda record: record.sample_id,
        )
        train_records = sorted(
            [
                record
                for bucket_index, bucket in enumerate(fold_buckets)
                if bucket_index != fold_index
                for subject_records in bucket
                for record in subject_records
            ],
            key=lambda record: record.sample_id,
        )
        folds.append((train_records, val_records))
    return folds


def summarize_records(records: list[SampleRecord]) -> dict[str, dict[str, int]]:
    binary = {"healthy": 0, "spa": 0}
    severity: dict[str, int] = defaultdict(int)
    sessions: dict[str, int] = defaultdict(int)
    subjects = {record.subject_id for record in records}
    for record in records:
        if record.disease_label == 0:
            binary["healthy"] += 1
        else:
            binary["spa"] += 1
        sessions[record.session] += 1
        severity_key = (
            str(int(record.severity_label))
            if record.severity_label >= 0
            else "unknown"
        )
        severity[severity_key] += 1
    return {
        "subjects": {"count": len(subjects)},
        "sessions": dict(sorted(sessions.items(), key=lambda item: item[0])),
        "binary": binary,
        "severity": dict(sorted(severity.items(), key=lambda item: item[0])),
    }


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


def read_window_manifest(path: str | Path, data_config: DataConfig) -> list[SampleRecord]:
    records: list[SampleRecord] = []
    base_root = Path(data_config.dataset_root) if data_config.dataset_root else None
    with Path(path).open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"subject_id", "session", "start_frame", "end_frame"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            missing_columns = ", ".join(sorted(missing))
            raise ValueError(
                f"Window manifest {path} is missing columns: {missing_columns}"
            )

        for row in reader:
            subject_id = row["subject_id"].strip()
            session = row["session"].strip()
            session_dir = (
                base_root / subject_id / session if base_root else Path(row.get("session_dir", ""))
            )
            if not session_dir:
                raise ValueError("dataset_root is required for window manifests.")
            rgb_path = session_dir / data_config.rgb_dir
            skeleton_path = session_dir / data_config.skeleton_file
            disease_label, severity_label = _read_session_labels(session_dir, data_config)
            start_frame = int(float(row["start_frame"]))
            end_frame = int(float(row["end_frame"]))
            sample_id = f"{subject_id}/{session}/{start_frame}-{end_frame}"
            records.append(
                SampleRecord(
                    sample_id=sample_id,
                    subject_id=subject_id,
                    session=session,
                    session_dir=str(session_dir),
                    rgb_path=str(rgb_path),
                    skeleton_path=str(skeleton_path),
                    disease_label=disease_label,
                    severity_label=severity_label,
                    window_start=start_frame,
                    window_end=end_frame,
                )
            )
    if not records:
        raise ValueError("Window manifest produced no samples.")
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
        subject_id, sessions = _parse_split_entry(entry, data_config)
        for session in sessions:
            session_dir = dataset_root / subject_id / session
            rgb_path = session_dir / data_config.rgb_dir
            skeleton_path = session_dir / data_config.skeleton_file

            required_paths = {
                "session": session_dir,
                "rgb": rgb_path,
                "skeleton": skeleton_path,
            }
            missing = [
                name for name, current in required_paths.items() if not current.exists()
            ]
            if missing:
                message = (
                    f"Missing required files for {subject_id}/{session}: "
                    f"{', '.join(missing)}"
                )
                if data_config.strict_missing_files:
                    raise FileNotFoundError(message)
                print(f"[skip] {message}")
                continue

            disease_label, severity_label = _read_session_labels(session_dir, data_config)

            sample_id = f"{subject_id}/{session}"
            records.append(
                SampleRecord(
                    sample_id=sample_id,
                    subject_id=subject_id,
                    session=session,
                    session_dir=str(session_dir),
                    rgb_path=str(rgb_path),
                    skeleton_path=str(skeleton_path),
                    disease_label=disease_label,
                    severity_label=severity_label,
                )
            )

    if not records:
        raise ValueError("No valid samples were found for the requested split.")
    return sorted(records, key=lambda record: record.sample_id)


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


def _parse_split_entry(entry: str, data_config: DataConfig) -> tuple[str, list[str]]:
    normalized = entry.replace("\\", "/")
    parts = [part for part in normalized.split("/") if part]
    if len(parts) == 1:
        return parts[0], _resolve_sessions(data_config)
    return parts[0], [parts[1]]


def _stratify_label(record: SampleRecord) -> int:
    if record.severity_label >= 0:
        return int(record.severity_label)
    return int(record.disease_label)


def _subject_stratify_label(records: list[SampleRecord]) -> int:
    labels = {_stratify_label(record) for record in records}
    if len(labels) != 1:
        raise ValueError(
            f"Inconsistent stratification labels across sessions for subject "
            f"{records[0].subject_id}: {sorted(labels)}"
        )
    return next(iter(labels))


def _group_records_by_subject(
    records: list[SampleRecord],
) -> dict[str, list[SampleRecord]]:
    grouped: dict[str, list[SampleRecord]] = defaultdict(list)
    for record in records:
        grouped[record.subject_id].append(record)
    return grouped


def _resolve_sessions(data_config: DataConfig) -> list[str]:
    if data_config.sessions:
        return list(dict.fromkeys(data_config.sessions))
    return [data_config.session]


def _read_session_labels(
    session_dir: Path,
    data_config: DataConfig,
) -> tuple[int, float]:
    annotation = _load_annotation_json(session_dir / data_config.annotation_file)

    binary_label = _read_required_label(
        label_path=session_dir / data_config.binary_label_file,
        annotation=annotation,
        annotation_key="binary_label",
        parser=parse_binary_label,
        label_name="binary label",
    )
    severity_label = _read_optional_label(
        label_path=session_dir / data_config.severity_label_file,
        annotation=annotation,
        annotation_key="severity_label",
        parser=parse_severity_label,
    )
    return binary_label, severity_label


def _load_annotation_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_required_label(
    label_path: Path,
    annotation: dict[str, Any] | None,
    annotation_key: str,
    parser,
    label_name: str,
) -> int:
    if label_path.exists():
        return parser(label_path.read_text(encoding="utf-8"))
    if annotation is not None and annotation_key in annotation:
        return parser(str(annotation[annotation_key]))
    raise FileNotFoundError(
        f"Missing {label_name}: {label_path} and annotation key {annotation_key!r}."
    )


def _read_optional_label(
    label_path: Path,
    annotation: dict[str, Any] | None,
    annotation_key: str,
    parser,
) -> float:
    if label_path.exists():
        return parser(label_path.read_text(encoding="utf-8"))
    if annotation is not None and annotation_key in annotation:
        return parser(str(annotation[annotation_key]))
    return -1.0


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


def load_rgb_sequence(
    path: str | Path,
    num_frames: int,
    image_size: int,
    window_start: int | None = None,
    window_end: int | None = None,
) -> torch.Tensor:
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
        if window_start is not None and window_end is not None:
            frame_paths = frame_paths[window_start : window_end + 1]
        if not frame_paths:
            raise ValueError(f"No image frames found in {rgb_path}.")
        indices = _uniform_frame_indices(len(frame_paths), num_frames)
        frames = []
        for index in indices.tolist():
            image = Image.open(frame_paths[index]).convert("RGB")
            frame = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float()
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
    if tensor.shape[-1] >= joint_dim:
        return tensor[..., :joint_dim].float()
    if tensor.shape[1] >= joint_dim:
        return tensor.permute(0, 2, 1)[..., :joint_dim].float()
    raise ValueError(
        "Skeleton tensor must be [T, J, C] or [T, C, J] with coordinate dim >= joint_dim."
    )


def load_skeleton_sequence(
    path: str | Path,
    num_frames: int,
    max_joints: int,
    joint_dim: int,
    normalize: bool,
    window_start: int | None = None,
    window_end: int | None = None,
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
    if window_start is not None and window_end is not None:
        skeleton = skeleton[window_start : window_end + 1]
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
        data_config: DataConfig,
        task_config: TaskConfig,
        split: str | None = None,
        records: list[SampleRecord] | None = None,
    ) -> None:
        if records is None and split is None:
            raise ValueError("Either split or records must be provided.")
        self.records = (
            records if records is not None else load_records_for_split(split, data_config)
        )
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
            window_start=record.window_start,
            window_end=record.window_end,
        )
        skeleton = load_skeleton_sequence(
            record.skeleton_path,
            num_frames=self.data_config.num_frames,
            max_joints=self.data_config.max_joints,
            joint_dim=self.data_config.joint_dim,
            normalize=self.data_config.normalize_skeleton,
            window_start=record.window_start,
            window_end=record.window_end,
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
