from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from typing import Any


_DATA_PATH_FIELDS = {
    "dataset_root",
    "train_manifest",
    "val_manifest",
    "test_manifest",
    "train_window_manifest",
    "val_window_manifest",
    "test_window_manifest",
    "train_split",
    "val_split",
    "test_split",
}


@dataclass
class DataConfig:
    dataset_root: str | None = None
    train_manifest: str | None = None
    val_manifest: str | None = None
    test_manifest: str | None = None
    train_window_manifest: str | None = None
    val_window_manifest: str | None = None
    test_window_manifest: str | None = None
    train_split: str | None = None
    val_split: str | None = None
    test_split: str | None = None
    session: str = "walk"
    sessions: list[str] | None = None
    rgb_dir: str = "rgb"
    skeleton_file: str = "skeleton/kpt3d/kpt3d.npy"
    binary_label_file: str = "labels/binary_label.txt"
    severity_label_file: str = "labels/severity_label.txt"
    annotation_file: str = "labels/disease_annotations.json"
    strict_missing_files: bool = True
    num_frames: int = 24
    image_size: int = 224
    rgb_mean: list[float] = field(
        default_factory=lambda: [0.485, 0.456, 0.406]
    )
    rgb_std: list[float] = field(
        default_factory=lambda: [0.229, 0.224, 0.225]
    )
    max_joints: int = 33
    joint_dim: int = 3
    normalize_skeleton: bool = True


@dataclass
class ModelConfig:
    input_mode: str = "fusion"
    embed_dim: int = 256
    rgb_patch_size: int = 16
    rgb_spatial_depth: int = 2
    rgb_temporal_depth: int = 2
    skeleton_joint_depth: int = 2
    skeleton_temporal_depth: int = 2
    fusion_depth: int = 4
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1


@dataclass
class TaskConfig:
    severity_mode: str = "classification"
    severity_num_classes: int = 4
    disease_loss_weight: float = 1.0
    severity_loss_weight: float = 1.0
    disease_pos_weight: float | None = None


@dataclass
class TrainConfig:
    batch_size: int = 4
    epochs: int = 50
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 0
    device: str = "cpu"
    checkpoint_dir: str = "checkpoints"
    log_interval: int = 10
    amp: bool = False
    seed: int = 42
    balance_disease: bool = False


@dataclass
class CVConfig:
    enabled: bool = True
    num_folds: int = 2
    seed: int = 42
    selection_metric: str = "loss"
    use_mean_best_epoch: bool = True
    final_epochs: int | None = None


@dataclass
class ExperimentConfig:
    data: DataConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    cv: CVConfig = field(default_factory=CVConfig)

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)

        data_raw = cls._resolve_data_paths(
            raw_data=raw["data"],
            config_dir=config_path.parent,
        )
        train_raw = cls._resolve_train_paths(
            raw_train=raw.get("train", {}),
            config_dir=config_path.parent,
        )

        return cls(
            data=DataConfig(**data_raw),
            model=ModelConfig(**raw.get("model", {})),
            task=TaskConfig(**raw.get("task", {})),
            train=TrainConfig(**train_raw),
            cv=CVConfig(**raw.get("cv", {})),
        )

    @staticmethod
    def _resolve_data_paths(
        raw_data: dict[str, Any],
        config_dir: Path,
    ) -> dict[str, Any]:
        resolved = dict(raw_data)
        for key in _DATA_PATH_FIELDS:
            value = resolved.get(key)
            if value:
                resolved[key] = str(_resolve_path_like(value, config_dir))
        return resolved

    @staticmethod
    def _resolve_train_paths(
        raw_train: dict[str, Any],
        config_dir: Path,
    ) -> dict[str, Any]:
        resolved = dict(raw_train)
        checkpoint_dir = resolved.get("checkpoint_dir")
        if checkpoint_dir:
            resolved["checkpoint_dir"] = str(
                _resolve_path_like(checkpoint_dir, config_dir)
            )
        return resolved

    def to_dict(self) -> dict[str, Any]:
        return {
            "data": vars(self.data),
            "model": vars(self.model),
            "task": vars(self.task),
            "train": vars(self.train),
            "cv": vars(self.cv),
        }


def _resolve_path_like(value: str, config_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    windows_path = PureWindowsPath(value)
    if windows_path.is_absolute():
        return Path(value)
    return (config_dir / path).resolve()
