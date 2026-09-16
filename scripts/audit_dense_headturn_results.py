from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path


COHORTS = ("primary20", "all25_sensitivity")
TASKS = ("binary", "severity")
VARIANTS = ("walk_only", "walk_headturn")
SPLITS = range(5)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def subject_ids(rows: list[dict[str, str]]) -> set[str]:
    return {row["subject_id"] for row in rows}


def close(left: float, right: float, tolerance: float = 1e-9) -> bool:
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)


def audit(run: Path) -> dict[str, object]:
    failures: list[str] = []
    collapsed: list[dict[str, object]] = []
    summaries: dict[str, object] = {}

    status = (run / "status.txt").read_text(encoding="utf-8").strip()
    if status != "DONE":
        failures.append(f"status is {status!r}, expected 'DONE'")

    for cohort in COHORTS:
        summary_path = run / cohort / "subject_headturn_comparison_summary.json"
        if not summary_path.is_file():
            failures.append(f"missing {summary_path.relative_to(run)}")
        else:
            summaries[cohort] = json.loads(summary_path.read_text(encoding="utf-8"))

        for task in TASKS:
            manifest_dir = run / cohort / "manifests" / task
            for split in SPLITS:
                partitions = {
                    name: read_rows(manifest_dir / f"split_{split}_{name}.csv")
                    for name in ("train", "val", "test")
                }
                subjects = {name: subject_ids(rows) for name, rows in partitions.items()}
                for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
                    overlap = subjects[left] & subjects[right]
                    if overlap:
                        failures.append(
                            f"{cohort}/{task}/split_{split}: {left}-{right} leakage {sorted(overlap)}"
                        )

                test_counts = Counter(row["subject_id"] for row in partitions["test"])
                paired_predictions: dict[str, list[dict[str, object]]] = {}
                for variant in VARIANTS:
                    output = run / cohort / "outputs" / variant / f"{task}_split_{split}"
                    metrics_path = output / "test_metrics.json"
                    checkpoint_path = output / "best.pt"
                    if not metrics_path.is_file():
                        failures.append(f"missing {metrics_path.relative_to(run)}")
                        continue
                    if not checkpoint_path.is_file() or checkpoint_path.stat().st_size == 0:
                        failures.append(f"missing or empty {checkpoint_path.relative_to(run)}")
                    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                    if metrics.get("primary_evaluation_unit") != "subject":
                        failures.append(f"{output.relative_to(run)}: primary unit is not subject")
                    predictions = metrics.get("subject_predictions", [])
                    paired_predictions[variant] = predictions
                    predicted_subjects = {str(item["subject_id"]) for item in predictions}
                    if predicted_subjects != subjects["test"]:
                        failures.append(f"{output.relative_to(run)}: test subjects do not match manifest")
                    for item in predictions:
                        sid = str(item["subject_id"])
                        if int(item["window_count"]) != test_counts[sid]:
                            failures.append(f"{output.relative_to(run)}: window count mismatch for {sid}")
                        probabilities = [float(value) for value in item["mean_probabilities"]]
                        if not all(math.isfinite(value) for value in probabilities):
                            failures.append(f"{output.relative_to(run)}: non-finite probability for {sid}")
                        if not close(sum(probabilities), 1.0, tolerance=1e-5):
                            failures.append(f"{output.relative_to(run)}: probabilities do not sum to one for {sid}")
                    unique_predictions = {int(item["prediction"]) for item in predictions}
                    if len(unique_predictions) == 1:
                        collapsed.append(
                            {
                                "cohort": cohort,
                                "task": task,
                                "variant": variant,
                                "split": split,
                                "prediction": next(iter(unique_predictions)),
                                "test_subjects": len(predictions),
                            }
                        )

                    config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
                    if Path(config["train_manifest"]).name != f"split_{split}_train.csv":
                        failures.append(f"{output.relative_to(run)}: wrong train manifest")
                    if Path(config["val_manifest"]).name != f"split_{split}_val.csv":
                        failures.append(f"{output.relative_to(run)}: wrong validation manifest")
                    normalization = config.get("headturn_normalization")
                    if variant == "walk_only" and normalization is not None:
                        failures.append(f"{output.relative_to(run)}: walk-only run has head-turn normalization")
                    if variant == "walk_headturn":
                        values_by_subject = {
                            row["subject_id"]: float(row["headturn_span_deg"])
                            for row in partitions["train"]
                        }
                        values = list(values_by_subject.values())
                        mean = sum(values) / len(values)
                        std = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
                        if normalization is None:
                            failures.append(f"{output.relative_to(run)}: missing head-turn normalization")
                        elif (
                            normalization.get("fit_unit") != "unique_training_subject"
                            or int(normalization.get("subject_count", -1)) != len(values)
                            or not close(float(normalization["mean_deg"]), mean)
                            or not close(float(normalization["std_deg"]), std)
                        ):
                            failures.append(f"{output.relative_to(run)}: head-turn normalization mismatch")

                if set(paired_predictions) == set(VARIANTS):
                    for left, right in zip(
                        paired_predictions["walk_only"], paired_predictions["walk_headturn"]
                    ):
                        left_identity = (left["subject_id"], left["label"], left["window_count"])
                        right_identity = (right["subject_id"], right["label"], right["window_count"])
                        if left_identity != right_identity:
                            failures.append(f"{cohort}/{task}/split_{split}: paired test records differ")
                            break

    exit_files = sorted((run / "jobs").glob("*.exit.json"))
    bad_exits = []
    for path in exit_files:
        record = json.loads(path.read_text(encoding="utf-8"))
        if int(record.get("exit_code", -1)) != 0:
            bad_exits.append(path.name)
    if len(exit_files) != 40:
        failures.append(f"found {len(exit_files)} exit records, expected 40")
    if bad_exits:
        failures.append(f"nonzero exits: {bad_exits}")

    result = {
        "experiment": "from_scratch_walk_headturn_dense_8gpu_v2",
        "status": status,
        "execution_jobs": len(exit_files),
        "metrics": len(list(run.glob("*/outputs/*/*/test_metrics.json"))),
        "checkpoints": len(list(run.glob("*/outputs/*/*/best.pt"))),
        "integrity_failures": failures,
        "single_class_prediction_jobs": collapsed,
        "single_class_prediction_count": len(collapsed),
        "summaries": summaries,
        "accepted": not failures,
        "interpretation": (
            "Execution and partition integrity accepted; participant-level estimates remain "
            "descriptive because test partitions contain only three or four participants."
        ),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the dense head-turn experiment matrix")
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.run.resolve())
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.resolve().write_text(payload, encoding="utf-8")
    print(payload, end="")
    raise SystemExit(0 if result["accepted"] else 1)


if __name__ == "__main__":
    main()

