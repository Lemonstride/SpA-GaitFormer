from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from spa_gaitformer.dataset import SpAWindowDataset


def make_manifest(tmp_path: Path) -> Path:
    rgb_dir = tmp_path / "rgb"
    rgb_dir.mkdir()
    for index in range(6):
        pixels = np.full((12, 16, 3), 32 + index * 10, dtype=np.uint8)
        Image.fromarray(pixels).save(rgb_dir / f"{index:04d}.png")

    skeleton_path = tmp_path / "skeleton.npy"
    rd_path = tmp_path / "rd.npy"
    np.save(skeleton_path, np.arange(6 * 8, dtype=np.float32).reshape(6, 8))
    np.save(rd_path, np.arange(2 * 5 * 7, dtype=np.float32).reshape(2, 5, 7))

    manifest = tmp_path / "manifest.csv"
    row = {
        "subject_id": "S01",
        "session": "walk",
        "window_id": "0",
        "rgb_dir": str(rgb_dir),
        "skeleton_path": str(skeleton_path),
        "rd_path": str(rd_path),
        "rgb_start": "0",
        "rgb_end": "6",
        "rd_start": "0",
        "rd_end": "2",
        "binary_label": "1",
        "severity_label": "2",
    }
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    return manifest


def test_evaluation_dataset_is_deterministic(tmp_path: Path) -> None:
    dataset = SpAWindowDataset(make_manifest(tmp_path), "binary", image_size=10)
    first = dataset[0]
    second = dataset[0]
    assert torch.equal(first["rgb"], second["rgb"])
    assert torch.equal(first["skeleton_features"], second["skeleton_features"])
    assert torch.equal(first["rd_maps"], second["rd_maps"])


def test_training_augmentation_preserves_shapes_and_three_to_one_ratio(tmp_path: Path) -> None:
    augmentation = {
        "enabled": True,
        "rgb": {
            "crop_scale_min": 0.9,
            "horizontal_flip_p": 1.0,
            "brightness": 0.1,
            "contrast": 0.1,
            "saturation": 0.1,
        },
        "skeleton": {"noise_std": 0.01, "feature_dropout_p": 0.01},
        "radar": {"amplitude_scale": [0.9, 1.1], "noise_std": 0.01},
        "temporal": {"mask_p": 1.0, "max_mask_fraction": 0.1},
    }
    torch.manual_seed(2026)
    dataset = SpAWindowDataset(
        make_manifest(tmp_path), "severity", image_size=10, augmentation=augmentation
    )
    sample = dataset[0]
    assert sample["rgb"].shape == (6, 3, 10, 10)
    assert sample["skeleton_features"].shape == (6, 8)
    assert sample["rd_maps"].shape == (2, 1, 5, 7)
    assert sample["rgb"].shape[0] == sample["rd_maps"].shape[0] * 3
    assert sample["skeleton_features"].shape[0] == sample["rd_maps"].shape[0] * 3
    masked_rd = torch.where(sample["rd_maps"].abs().sum(dim=(1, 2, 3)) == 0)[0]
    assert masked_rd.numel() == 1
    index = int(masked_rd.item())
    assert torch.count_nonzero(sample["rgb"][index * 3 : (index + 1) * 3]) == 0
    assert torch.count_nonzero(sample["skeleton_features"][index * 3 : (index + 1) * 3]) == 0


def test_subject_cycle_is_deterministic_and_advances_by_epoch(tmp_path: Path) -> None:
    augmentation = {
        "enabled": True,
        "policy": "subject_cycle_rgb32_v1",
        "seed": 2026,
        "skeleton": {"noise_std": 0.0, "feature_dropout_p": 0.0},
        "radar": {"amplitude_scale": [1.0, 1.0], "noise_std": 0.0},
        "temporal": {"mask_p": 0.0},
    }
    dataset = SpAWindowDataset(
        make_manifest(tmp_path), "binary", image_size=10, augmentation=augmentation
    )

    first = dataset._sample_rgb_transform("S01")
    repeated = dataset._sample_rgb_transform("S01")
    dataset.set_epoch(1)
    second = dataset._sample_rgb_transform("S01")

    assert first == repeated
    assert first["view"] != second["view"]
    assert int(str(second["view"]).split("_")[-1]) == (
        int(str(first["view"]).split("_")[-1]) + 1
    ) % 32


def test_subject_cycle_keeps_modal_shapes_and_rd_unchanged(tmp_path: Path) -> None:
    augmentation = {
        "enabled": True,
        "policy": "subject_cycle_rgb32_v1",
        "seed": 2026,
        "skeleton": {"noise_std": 0.0, "feature_dropout_p": 0.0},
        "radar": {"amplitude_scale": [1.0, 1.0], "noise_std": 0.0},
        "temporal": {"mask_p": 0.0},
    }
    plain = SpAWindowDataset(make_manifest(tmp_path), "binary", image_size=10)
    augmented = SpAWindowDataset(
        plain.manifest, "binary", image_size=10, augmentation=augmentation
    )
    sample = augmented[0]
    reference = plain[0]

    assert sample["rgb"].shape == (6, 3, 10, 10)
    assert sample["skeleton_features"].shape == (6, 8)
    assert sample["rd_maps"].shape == (2, 1, 5, 7)
    assert torch.equal(sample["rd_maps"], reference["rd_maps"])
    assert sample["augmentation_view"].startswith("view_")
