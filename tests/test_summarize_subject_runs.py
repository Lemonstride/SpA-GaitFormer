from __future__ import annotations

import json
from pathlib import Path

from spa_gaitformer.summarize_subject_runs import summarize_task


def test_summarize_requires_and_combines_subject_results(tmp_path: Path) -> None:
    for split, accuracy in enumerate((0.5, 1.0)):
        output = tmp_path / "outputs" / f"binary_split_{split}"
        output.mkdir(parents=True)
        payload = {
            "primary_evaluation_unit": "subject",
            "subject_level": {
                "accuracy": accuracy,
                "macro_precision": accuracy,
                "macro_recall": accuracy,
                "macro_f1": accuracy,
                "count": 1,
            },
            "window_level": {"accuracy": accuracy},
            "subject_predictions": [
                {
                    "subject_id": f"S0{split + 1}",
                    "label": 0,
                    "prediction": 0,
                    "mean_probabilities": [0.8, 0.2],
                    "window_count": 3,
                }
            ],
        }
        (output / "test_metrics.json").write_text(json.dumps(payload), encoding="utf-8")

    result = summarize_task(tmp_path, "binary", repeats=2)

    assert result["summary"]["accuracy"]["mean"] == 0.75
    assert result["summary"]["accuracy"]["sample_std"] > 0
    assert result["test_subject_occurrences"] == {"S01": 1, "S02": 1}
