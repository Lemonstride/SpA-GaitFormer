from pathlib import Path

import numpy as np
import pytest

from spa_gaitformer.manifest import build_manifest, count_windows, load_labels


def test_window_count_uses_total_windows() -> None:
    assert count_windows(total_steps=10, window=4, stride=3) == 3
    assert count_windows(total_steps=3, window=4, stride=1) == 0


def test_unknown_clinical_label_is_rejected(tmp_path: Path) -> None:
    labels = tmp_path / "labels.csv"
    labels.write_text(
        "subject_id,binary_label,severity_label\nS01,unknown,unknown\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="unknown"):
        load_labels(labels)


def test_manifest_rejects_extra_rgb_and_skeleton_frames(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    skeleton_root = tmp_path / "skeleton"
    rd_root = tmp_path / "rd"
    rgb_dir = processed / "S01" / "walk" / "rgb"
    rgb_dir.mkdir(parents=True)
    for frame in range(9):
        (rgb_dir / f"frame_{frame:03d}.png").touch()

    skeleton_path = skeleton_root / "S01" / "walk" / "frame_features.npy"
    skeleton_path.parent.mkdir(parents=True)
    np.save(skeleton_path, np.zeros((9, 4), dtype=np.float32))
    rd_path = rd_root / "S01" / "walk" / "rd.npy"
    rd_path.parent.mkdir(parents=True)
    np.save(rd_path, np.zeros((2, 4, 4), dtype=np.float32))

    labels = tmp_path / "labels.csv"
    labels.write_text(
        "subject_id,binary_label,severity_label\nS01,1,2\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Exact 3:1"):
        build_manifest(processed, labels, rd_root, skeleton_root, rd_window=2, rd_stride=1)

