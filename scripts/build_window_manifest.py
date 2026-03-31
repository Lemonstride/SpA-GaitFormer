from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build window manifest CSV from a subject split."
    )
    parser.add_argument(
        "--processed_root",
        type=Path,
        required=True,
        help="SpA-MMD processed root (contains SXX).",
    )
    parser.add_argument(
        "--split_file",
        type=Path,
        required=True,
        help="Subject split file (one subject per line).",
    )
    parser.add_argument(
        "--sessions",
        type=str,
        default="walk,head_turn",
        help="Comma-separated sessions to include.",
    )
    parser.add_argument(
        "--window",
        type=int,
        required=True,
        help="Window length in frames.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        required=True,
        help="Stride in frames.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        required=True,
        help="Output window manifest CSV path.",
    )
    return parser.parse_args()


def read_split(path: Path) -> list[str]:
    subjects = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            subjects.append(value.split("/")[0])
    return subjects


def count_frames(rgb_dir: Path) -> int:
    if not rgb_dir.exists():
        return 0
    return len(
        [p for p in rgb_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}]
    )


def main() -> None:
    args = parse_args()
    sessions = [s.strip() for s in args.sessions.split(",") if s.strip()]
    subjects = read_split(args.split_file)
    rows: list[dict[str, int | str]] = []

    for subject_id in subjects:
        subject_dir = args.processed_root / subject_id
        for session in sessions:
            session_dir = subject_dir / session
            rgb_dir = session_dir / "rgb"
            total_frames = count_frames(rgb_dir)
            if total_frames < args.window:
                continue
            start = 0
            window_id = 0
            while start + args.window <= total_frames:
                rows.append(
                    {
                        "subject_id": subject_id,
                        "session": session,
                        "window_id": window_id,
                        "start_frame": start,
                        "end_frame": start + args.window - 1,
                        "total_frames": total_frames,
                    }
                )
                window_id += 1
                start += args.stride

    if not rows:
        raise SystemExit("No windows generated. Check inputs.")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} windows to {args.output_csv}")


if __name__ == "__main__":
    main()
