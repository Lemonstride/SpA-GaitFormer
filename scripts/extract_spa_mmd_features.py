from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


HEAD_TURN_KEYS = [
    "left_max_angle_deg",
    "right_max_angle_deg",
    "total_rom_deg",
    "asymmetry_deg",
    "relative_rom_score",
    "asymmetry_score",
]


PREFERRED_ROOT_NAMES = [
    "pelvis",
    "mid_hip",
    "hip_center",
    "root",
    "spine_base",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract hand-crafted features from SpA-MMD for baseline models."
    )
    parser.add_argument(
        "--dataset_root",
        type=Path,
        required=True,
        help="Root path to SpA-MMD processed directory.",
    )
    parser.add_argument(
        "--sessions",
        type=str,
        default="walk,head_turn",
        help="Comma-separated list of sessions to include.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        required=True,
        help="Output CSV file path.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any required file is missing.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_root_index(meta: dict[str, Any] | None, num_joints: int) -> int:
    if not meta:
        return 0
    names = meta.get("joint_names")
    if not isinstance(names, list):
        return 0
    normalized = [str(name).lower() for name in names]
    for target in PREFERRED_ROOT_NAMES:
        if target in normalized:
            return normalized.index(target)
    return 0


def _pair_left_right(names: list[str]) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    name_to_index = {name: index for index, name in enumerate(names)}
    for name, index in name_to_index.items():
        if "left" in name:
            partner = name.replace("left", "right")
        elif name.startswith("l_"):
            partner = "r_" + name[2:]
        else:
            continue
        if partner in name_to_index:
            pairs.append((index, name_to_index[partner]))
    return pairs


def _safe_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(values.mean())


def _safe_std(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(values.std())


def _extract_features_from_kpt3d(
    kpt3d: np.ndarray,
    meta: dict[str, Any] | None,
) -> dict[str, float]:
    if kpt3d.ndim != 3:
        raise ValueError(f"Expected kpt3d with shape [T,J,C], got {kpt3d.shape}")
    if kpt3d.shape[-1] < 3:
        raise ValueError("kpt3d must have at least 3 coordinate dims.")

    coords = kpt3d[..., :3]
    conf = kpt3d[..., 3] if kpt3d.shape[-1] >= 4 else None

    root_idx = _resolve_root_index(meta, coords.shape[1])
    root = coords[:, root_idx, :]
    root_vel = np.diff(root, axis=0)
    root_speed = np.linalg.norm(root_vel, axis=-1)

    joint_vel = np.diff(coords, axis=0)
    joint_speed = np.linalg.norm(joint_vel, axis=-1)

    pose_centered = coords - root[:, None, :]
    pose_spread = np.linalg.norm(pose_centered, axis=-1)

    root_speed_median = float(np.median(root_speed)) if root_speed.size else 0.0
    root_speed_iqr = (
        float(np.percentile(root_speed, 75) - np.percentile(root_speed, 25))
        if root_speed.size
        else 0.0
    )

    accel = np.diff(root_speed) if root_speed.size > 1 else np.asarray([])
    accel_mean = _safe_mean(accel)
    accel_std = _safe_std(accel)

    step_count, step_interval_mean, step_interval_std = _estimate_step_stats(root_speed)

    features = {
        "root_speed_mean": _safe_mean(root_speed),
        "root_speed_std": _safe_std(root_speed),
        "root_range_x": float(root[:, 0].max() - root[:, 0].min()),
        "root_range_y": float(root[:, 1].max() - root[:, 1].min()),
        "root_range_z": float(root[:, 2].max() - root[:, 2].min()),
        "joint_speed_mean": _safe_mean(joint_speed),
        "joint_speed_std": _safe_std(joint_speed),
        "pose_spread_mean": _safe_mean(pose_spread),
        "pose_spread_std": _safe_std(pose_spread),
        "root_speed_median": root_speed_median,
        "root_speed_iqr": root_speed_iqr,
        "root_accel_mean": accel_mean,
        "root_accel_std": accel_std,
        "step_count": float(step_count),
        "step_interval_mean": step_interval_mean,
        "step_interval_std": step_interval_std,
        "confidence_mean": _safe_mean(conf) if conf is not None else 0.0,
    }

    if meta and isinstance(meta.get("joint_names"), list):
        names = [str(name).lower() for name in meta["joint_names"]]
        pairs = _pair_left_right(names)
        if pairs:
            pair_distances = []
            pair_xdiff = []
            for left, right in pairs:
                delta = coords[:, left, :] - coords[:, right, :]
                pair_distances.append(np.linalg.norm(delta, axis=-1))
                pair_xdiff.append(np.abs(delta[:, 0]))
            pair_distances = np.concatenate(pair_distances, axis=0)
            pair_xdiff = np.concatenate(pair_xdiff, axis=0)
            features["symmetry_dist_mean"] = _safe_mean(pair_distances)
            features["symmetry_xdiff_mean"] = _safe_mean(pair_xdiff)
        else:
            features["symmetry_dist_mean"] = 0.0
            features["symmetry_xdiff_mean"] = 0.0
    else:
        features["symmetry_dist_mean"] = 0.0
        features["symmetry_xdiff_mean"] = 0.0

    return features


def _estimate_step_stats(root_speed: np.ndarray) -> tuple[int, float, float]:
    if root_speed.size < 3:
        return 0, 0.0, 0.0
    threshold = 0.5 * float(root_speed.max())
    peaks = []
    for index in range(1, root_speed.size - 1):
        if (
            root_speed[index] > threshold
            and root_speed[index] >= root_speed[index - 1]
            and root_speed[index] >= root_speed[index + 1]
        ):
            peaks.append(index)
    if len(peaks) < 2:
        return len(peaks), 0.0, 0.0
    intervals = np.diff(peaks).astype(np.float32)
    return len(peaks), float(intervals.mean()), float(intervals.std())


def _extract_head_turn_features(session_dir: Path) -> dict[str, float]:
    summary_path = session_dir / "labels" / "head_turn_state" / "summary.json"
    if not summary_path.exists():
        return {key: 0.0 for key in HEAD_TURN_KEYS}
    summary = _load_json(summary_path)
    values = {}
    for key in HEAD_TURN_KEYS:
        raw = summary.get(key, 0.0)
        try:
            values[key] = float(raw)
        except (TypeError, ValueError):
            values[key] = 0.0
    return values


def _read_label(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _iter_subjects(dataset_root: Path) -> list[Path]:
    return sorted([p for p in dataset_root.iterdir() if p.is_dir()])


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root
    sessions = [s.strip() for s in args.sessions.split(",") if s.strip()]
    rows: list[dict[str, Any]] = []

    for subject_dir in _iter_subjects(dataset_root):
        subject_id = subject_dir.name
        for session in sessions:
            session_dir = subject_dir / session
            if not session_dir.exists():
                if args.strict:
                    raise FileNotFoundError(f"Missing session dir: {session_dir}")
                continue

            kpt_path = session_dir / "skeleton" / "kpt3d" / "kpt3d.npy"
            label_path = session_dir / "labels" / "binary_label.txt"
            severity_path = session_dir / "labels" / "severity_label.txt"
            meta_path = session_dir / "skeleton" / "kpt3d" / "pose_meta.json"

            if not kpt_path.exists() or not label_path.exists():
                if args.strict:
                    raise FileNotFoundError(
                        f"Missing required files for {subject_id}/{session}"
                    )
                continue

            kpt3d = np.load(kpt_path)
            meta = _load_json(meta_path) if meta_path.exists() else None
            features = _extract_features_from_kpt3d(kpt3d, meta)
            head_turn_features = (
                _extract_head_turn_features(session_dir)
                if session == "head_turn"
                else {key: 0.0 for key in HEAD_TURN_KEYS}
            )

            row: dict[str, Any] = {
                "subject_id": subject_id,
                "session": session,
                "disease_label": int(_read_label(label_path)),
                "severity_label": int(_read_label(severity_path))
                if severity_path.exists()
                else -1,
            }
            row.update(features)
            row.update(head_turn_features)
            rows.append(row)

    if not rows:
        raise SystemExit("No samples found. Check dataset_root and sessions.")

    fieldnames = list(rows[0].keys())
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
