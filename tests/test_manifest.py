from pathlib import Path

import pytest

from spa_gaitformer.manifest import count_windows, load_labels


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

