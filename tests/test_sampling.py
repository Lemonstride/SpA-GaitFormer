from __future__ import annotations

import pytest

from spa_gaitformer.sampling import subject_class_balanced_weights


def test_subject_class_balanced_weights_equalize_classes_and_subjects() -> None:
    rows = [
        {"subject_id": "S01", "binary_label": "0"},
        {"subject_id": "S01", "binary_label": "0"},
        {"subject_id": "S02", "binary_label": "0"},
        {"subject_id": "S03", "binary_label": "1"},
        {"subject_id": "S03", "binary_label": "1"},
        {"subject_id": "S03", "binary_label": "1"},
    ]

    weights = subject_class_balanced_weights(rows, "binary_label")
    totals = {
        subject: sum(weight for row, weight in zip(rows, weights) if row["subject_id"] == subject)
        for subject in ("S01", "S02", "S03")
    }

    assert totals["S01"] == pytest.approx(totals["S02"])
    assert totals["S01"] + totals["S02"] == pytest.approx(totals["S03"])


def test_subject_class_balanced_weights_reject_inconsistent_subject_labels() -> None:
    with pytest.raises(ValueError, match="Inconsistent"):
        subject_class_balanced_weights(
            [
                {"subject_id": "S01", "binary_label": "0"},
                {"subject_id": "S01", "binary_label": "1"},
            ],
            "binary_label",
        )
