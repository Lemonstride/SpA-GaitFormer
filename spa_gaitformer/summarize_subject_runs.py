from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def summarize_task(run_dir: Path, task: str, repeats: int) -> dict[str, object]:
    runs = []
    test_occurrences: Counter[str] = Counter()
    for split in range(repeats):
        path = run_dir / "outputs" / f"{task}_split_{split}" / "test_metrics.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing completed test result: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("primary_evaluation_unit") != "subject":
            raise ValueError(f"Result is not subject-level: {path}")
        predictions = payload.get("subject_predictions", [])
        if not predictions:
            raise ValueError(f"No subject predictions in {path}")
        test_occurrences.update(record["subject_id"] for record in predictions)
        runs.append(
            {
                "split": split,
                "path": str(path.resolve()),
                "subject_level": payload["subject_level"],
                "window_level": payload["window_level"],
            }
        )

    metric_names = ("accuracy", "macro_precision", "macro_recall", "macro_f1")
    summary = {}
    for metric in metric_names:
        values = [float(run["subject_level"][metric]) for run in runs]
        summary[metric] = {
            "mean": float(np.mean(values)),
            "sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "values": values,
        }
    return {
        "task": task,
        "repeats": repeats,
        "primary_evaluation_unit": "subject",
        "aggregation": "mean_softmax_probability_over_all_valid_subject_windows",
        "summary": summary,
        "test_subject_occurrences": dict(sorted(test_occurrences.items())),
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize repeated subject-level runs")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    result = {
        "experiment": "independent_from_scratch_baseline",
        "tasks": {
            task: summarize_task(run_dir, task, args.repeats)
            for task in ("binary", "severity")
        },
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
