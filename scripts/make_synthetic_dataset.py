from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from spa_gaitformer.manifest import build_manifest, write_manifest
from spa_gaitformer.splits import stratified_subject_split, write_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create software-only SpA-Gaitformer smoke data")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--subjects", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.subjects < 6 or args.subjects % 2:
        raise ValueError("Smoke data needs an even subject count of at least 6")
    root = args.output_root.resolve()
    processed, skeleton_root, rd_root = root / "processed", root / "skeleton", root / "rd"
    rng = np.random.default_rng(2026)
    labels = []
    for index in range(args.subjects):
        subject = f"S{index + 1:02d}"
        binary = index % 2
        severity = index % 4
        labels.append({"subject_id": subject, "binary_label": binary, "severity_label": severity})
        for session in ("walk", "head_turn"):
            rgb_dir = processed / subject / session / "rgb"
            rgb_dir.mkdir(parents=True, exist_ok=True)
            for frame in range(6):
                image = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
                Image.fromarray(image).save(rgb_dir / f"frame_{frame + 1:06d}.png")
            skeleton_path = skeleton_root / subject / session / "frame_features.npy"
            skeleton_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(skeleton_path, rng.normal(size=(6, 16)).astype(np.float32))
            rd_path = rd_root / subject / session / "rd.npy"
            rd_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(rd_path, rng.normal(size=(2, 16, 16)).astype(np.float32))

    labels_path = root / "labels.csv"
    with labels_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(labels[0]))
        writer.writeheader()
        writer.writerows(labels)
    rows = build_manifest(processed, labels_path, rd_root, skeleton_root, rd_window=2, rd_stride=1)
    manifest = root / "manifests" / "all.csv"
    write_manifest(rows, manifest)

    subject_to_label = {row["subject_id"]: str(row["binary_label"]) for row in labels}
    splits = stratified_subject_split(subject_to_label, train_ratio=0.5, val_ratio=0.25, seed=2026)
    for split, subjects in splits.items():
        write_rows(
            [row for row in rows if str(row["subject_id"]) in subjects],
            root / "manifests" / f"split_0_{split}.csv",
        )
    print(f"Created synthetic smoke dataset with {len(rows)} windows at {root}")


if __name__ == "__main__":
    main()
