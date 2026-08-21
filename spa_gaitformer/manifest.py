from __future__ import annotations

import argparse
import csv
from pathlib import Path


SESSIONS = ("walk", "head_turn")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def count_images(path: Path) -> int:
    return sum(1 for item in path.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES)


def count_windows(total_steps: int, window: int, stride: int) -> int:
    if window <= 0 or stride <= 0:
        raise ValueError("Window and stride must be positive")
    return 0 if total_steps < window else 1 + (total_steps - window) // stride


def load_labels(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"subject_id", "binary_label", "severity_label"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Labels CSV must contain {sorted(required)}")
    labels: dict[str, dict[str, str]] = {}
    for row in rows:
        subject = row["subject_id"].strip()
        if not subject or subject in labels:
            raise ValueError(f"Missing or duplicate subject_id in labels: {subject!r}")
        if row["binary_label"].strip().lower() == "unknown" or row["severity_label"].strip().lower() == "unknown":
            raise ValueError(f"Clinical label is unknown for {subject}; training manifest was not generated")
        labels[subject] = row
    return labels


def npy_steps(path: Path) -> int:
    import numpy as np

    array = np.load(path, mmap_mode="r")
    if array.ndim < 2:
        raise ValueError(f"Expected a temporal array at {path}, got {array.shape}")
    return int(array.shape[0])


def build_manifest(
    processed_root: Path,
    labels_csv: Path,
    rd_root: Path,
    skeleton_root: Path,
    rd_window: int,
    rd_stride: int,
) -> list[dict[str, object]]:
    labels = load_labels(labels_csv)
    rows: list[dict[str, object]] = []
    for subject_dir in sorted(path for path in processed_root.iterdir() if path.is_dir()):
        subject = subject_dir.name
        if subject not in labels:
            raise ValueError(f"No clinical label for subject {subject}")
        for session in SESSIONS:
            rgb_dir = subject_dir / session / "rgb"
            rd_path = rd_root / subject / session / "rd.npy"
            skeleton_path = skeleton_root / subject / session / "frame_features.npy"
            missing = [str(path) for path in (rgb_dir, rd_path, skeleton_path) if not path.exists()]
            if missing:
                raise FileNotFoundError(
                    f"Missing synchronized assets for {subject}/{session}: " + ", ".join(missing)
                )

            rgb_steps = count_images(rgb_dir)
            skeleton_steps = npy_steps(skeleton_path)
            rd_steps = npy_steps(rd_path)
            if rgb_steps != skeleton_steps:
                raise ValueError(
                    f"RGB/skeleton frame mismatch for {subject}/{session}: {rgb_steps} != {skeleton_steps}"
                )
            synchronized_rd_steps = min(rd_steps, rgb_steps // 3)
            if rgb_steps < rd_steps * 3:
                raise ValueError(
                    f"Not enough RGB/skeleton frames for exact 3:1 alignment in {subject}/{session}: "
                    f"RGB={rgb_steps}, RD={rd_steps}"
                )

            for window_id in range(count_windows(synchronized_rd_steps, rd_window, rd_stride)):
                rd_start = window_id * rd_stride
                rd_end = rd_start + rd_window
                rgb_start = rd_start * 3
                rgb_end = rd_end * 3
                rows.append(
                    {
                        "subject_id": subject,
                        "session": session,
                        "window_id": window_id,
                        "rgb_dir": str(rgb_dir.resolve()),
                        "skeleton_path": str(skeleton_path.resolve()),
                        "rd_path": str(rd_path.resolve()),
                        "rgb_start": rgb_start,
                        "rgb_end": rgb_end,
                        "rd_start": rd_start,
                        "rd_end": rd_end,
                        "binary_label": labels[subject]["binary_label"].strip(),
                        "severity_label": labels[subject]["severity_label"].strip(),
                    }
                )
    if not rows:
        raise ValueError("No complete synchronized windows were generated")
    return rows


def write_manifest(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build strict 3:1 SpA-MMD window manifest")
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--rd-root", type=Path, required=True)
    parser.add_argument("--skeleton-root", type=Path, required=True)
    parser.add_argument("--rd-window", type=int, required=True)
    parser.add_argument("--rd-stride", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_manifest(
        args.processed_root.resolve(),
        args.labels_csv.resolve(),
        args.rd_root.resolve(),
        args.skeleton_root.resolve(),
        args.rd_window,
        args.rd_stride,
    )
    write_manifest(rows, args.output.resolve())
    print(f"Wrote {len(rows)} strict synchronized windows to {args.output.resolve()}")


if __name__ == "__main__":
    main()

