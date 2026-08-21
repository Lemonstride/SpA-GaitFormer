from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    return rows


def subject_labels(rows: list[dict[str, str]], label_column: str) -> dict[str, str]:
    labels: dict[str, str] = {}
    for row in rows:
        subject, label = row["subject_id"], row[label_column]
        if subject in labels and labels[subject] != label:
            raise ValueError(f"Inconsistent {label_column} for {subject}")
        labels[subject] = label
    return labels


def stratified_subject_split(
    labels: dict[str, str], train_ratio: float, val_ratio: float, seed: int
) -> dict[str, set[str]]:
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Ratios must satisfy train>0, val>0, and train+val<1")
    if len(labels) < 3:
        raise ValueError("At least three independent subjects are needed for train/val/test splitting")
    groups: dict[str, list[str]] = defaultdict(list)
    for subject, label in labels.items():
        groups[label].append(subject)
    rng = random.Random(seed)
    splits = {"train": set(), "val": set(), "test": set()}
    for subjects in groups.values():
        rng.shuffle(subjects)
        count = len(subjects)
        train_count = max(1, round(count * train_ratio))
        val_count = max(1, round(count * val_ratio)) if count >= 3 else 0
        if count >= 3:
            train_count = min(train_count, count - val_count - 1)
        elif train_count + val_count >= count:
            train_count = max(1, count - max(1, val_count))
        splits["train"].update(subjects[:train_count])
        splits["val"].update(subjects[train_count : train_count + val_count])
        splits["test"].update(subjects[train_count + val_count :])
    if any(not split for split in splits.values()):
        raise ValueError(
            "The subject/class inventory cannot produce non-empty stratified train, val, and test sets"
        )
    return splits


def write_rows(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create repeated subject-independent splits")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", choices=["binary", "severity"], default="binary")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--train-ratio", type=float, required=True)
    parser.add_argument("--val-ratio", type=float, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.manifest.resolve())
    labels = subject_labels(rows, f"{args.task}_label")
    for repeat in range(args.repeats):
        split_subjects = stratified_subject_split(labels, args.train_ratio, args.val_ratio, args.seed + repeat)
        for split, subjects in split_subjects.items():
            split_rows = [row for row in rows if row["subject_id"] in subjects]
            write_rows(split_rows, args.output_dir.resolve() / f"split_{repeat}_{split}.csv")
    print(f"Wrote {args.repeats} subject-independent split sets to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
