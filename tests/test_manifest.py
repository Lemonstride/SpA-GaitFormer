from pathlib import Path

import pytest

from spa_gaitformer.manifest import count_windows, load_frame_quality, load_labels


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


def test_frame_quality_combines_roi_silhouette_and_timestamps(tmp_path: Path) -> None:
    rgb = tmp_path / "rgb_session"
    skeleton = tmp_path / "skeleton_session" / "silhouette"
    rgb.mkdir()
    skeleton.mkdir(parents=True)
    (rgb / "roi_frames.csv").write_text(
        "file_name,timestamp_ms,valid_for_training\n"
        "frame_1.png,0,1\n"
        "frame_2.png,33,0\n"
        "frame_3.png,66,1\n"
        "frame_4.png,166,1\n",
        encoding="utf-8",
    )
    (skeleton / "silhouette_frames.csv").write_text(
        "file_name,area\n"
        "frame_1.png,100\n"
        "frame_2.png,100\n"
        "frame_3.png,0\n"
        "frame_4.png,100\n",
        encoding="utf-8",
    )

    eligible, gaps = load_frame_quality(rgb, skeleton.parent, 4)

    assert eligible == [True, False, False, True]
    assert gaps.tolist() == [False, False, True]
