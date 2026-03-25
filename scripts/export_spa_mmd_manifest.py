from __future__ import annotations

import argparse
import csv
from pathlib import Path

from spa_gaitformer.config import DataConfig
from spa_gaitformer.data import build_records_from_processed_root, read_split_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a SpA-MMD split into a CSV manifest."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--session", default="walk")
    parser.add_argument("--rgb-dir", default="rgb")
    parser.add_argument("--skeleton-file", default="skeleton/kpt3d/kpt3d.npy")
    parser.add_argument("--binary-label-file", default="labels/binary_label.txt")
    parser.add_argument("--severity-label-file", default="labels/severity_label.txt")
    parser.add_argument(
        "--strict-missing-files",
        action="store_true",
        help="Raise an error instead of skipping incomplete sessions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_config = DataConfig(
        dataset_root=str(args.dataset_root),
        session=args.session,
        rgb_dir=args.rgb_dir,
        skeleton_file=args.skeleton_file,
        binary_label_file=args.binary_label_file,
        severity_label_file=args.severity_label_file,
        strict_missing_files=args.strict_missing_files,
    )
    records = build_records_from_processed_root(
        dataset_root=args.dataset_root,
        split_entries=read_split_file(args.split_file),
        data_config=data_config,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "subject_id",
                "session",
                "session_dir",
                "rgb_path",
                "skeleton_path",
                "disease_label",
                "severity_label",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(record.__dict__)


if __name__ == "__main__":
    main()
