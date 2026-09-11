from __future__ import annotations

import pytest

from spa_gaitformer.metrics import aggregate_subject_probabilities


def test_subject_probabilities_are_averaged_before_prediction() -> None:
    metrics, records = aggregate_subject_probabilities(
        ["S02", "S01", "S01", "S02"],
        [1, 0, 0, 1],
        [[0.2, 0.8], [0.9, 0.1], [0.4, 0.6], [0.7, 0.3]],
    )

    assert metrics["accuracy"] == 1.0
    assert [record["subject_id"] for record in records] == ["S01", "S02"]
    assert records[0]["prediction"] == 0
    assert records[0]["window_count"] == 2
    assert records[0]["mean_probabilities"] == pytest.approx([0.65, 0.35])


def test_subject_aggregation_rejects_inconsistent_labels() -> None:
    with pytest.raises(ValueError, match="Inconsistent labels"):
        aggregate_subject_probabilities(
            ["S01", "S01"],
            [0, 1],
            [[0.8, 0.2], [0.1, 0.9]],
        )
