from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METRICS = ("accuracy", "macro_precision", "macro_recall", "macro_f1")
VARIANTS = ("walk_only", "walk_headturn")
TASKS = ("binary", "severity")


def load_result(run_dir: Path, variant: str, task: str, split: int) -> dict[str, object]:
    path = run_dir / "outputs" / variant / f"{task}_split_{split}" / "test_metrics.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("primary_evaluation_unit") != "subject":
        raise ValueError(f"Not a subject-level result: {path}")
    return payload


def metric_summary(values: list[float]) -> dict[str, object]:
    return {
        "mean": float(np.mean(values)),
        "sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "values": values,
    }


def summarize(run_dir: Path, repeats: int) -> dict[str, object]:
    result: dict[str, object] = {
        "experiment": "paired_walk_only_vs_walk_plus_reviewed_headturn",
        "primary_evaluation_unit": "subject",
        "repeats": repeats,
        "tasks": {},
    }
    tasks: dict[str, object] = {}
    for task in TASKS:
        loaded = {
            variant: [load_result(run_dir, variant, task, split) for split in range(repeats)]
            for variant in VARIANTS
        }
        variants = {
            variant: {
                metric: metric_summary(
                    [float(payload["subject_level"][metric]) for payload in loaded[variant]]
                )
                for metric in METRICS
            }
            for variant in VARIANTS
        }
        paired_delta = {
            metric: metric_summary(
                [
                    float(loaded["walk_headturn"][split]["subject_level"][metric])
                    - float(loaded["walk_only"][split]["subject_level"][metric])
                    for split in range(repeats)
                ]
            )
            for metric in METRICS
        }
        tasks[task] = {
            "variants": variants,
            "paired_delta_headturn_minus_walk": paired_delta,
        }
    result["tasks"] = tasks
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize paired head-turn experiments")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.run_dir.resolve(), args.repeats)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

