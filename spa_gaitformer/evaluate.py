from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import load_config, task_num_classes
from .dataset import SpAWindowDataset
from .engine import run_epoch
from .losses import ClassificationObjective
from .model import SpAGaitformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SpA-Gaitformer")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task", choices=["binary", "severity"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    data_cfg = config["data"]
    headturn_enabled = bool(config["model"].get("headturn", {}).get("enabled", False))
    dataset = SpAWindowDataset(
        args.manifest,
        args.task,
        int(data_cfg["image_size"]),
        rd_normalization=data_cfg.get("rd_normalization", "none"),
        headturn_enabled=headturn_enabled,
    )
    if headturn_enabled:
        normalization = checkpoint.get("headturn_normalization")
        if not normalization:
            raise ValueError("Head-turn checkpoint lacks training-only normalization metadata")
        dataset.set_headturn_normalization(
            float(normalization["mean_deg"]), float(normalization["std_deg"])
        )
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["data"].get("num_workers", 4)),
    )
    model = SpAGaitformer(config, task_num_classes(config, args.task)).to(device)
    model.load_state_dict(checkpoint["model"])
    metrics = run_epoch(
        model,
        loader,
        ClassificationObjective(),
        device,
        mixed_precision=bool(config["training"].get("mixed_precision", False)),
        include_subject_records=True,
    )
    subject_predictions = metrics.pop("subject_predictions")
    window_level = {
        key: metrics[key]
        for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "loss", "window_count")
    }
    subject_level = {
        key.removeprefix("subject_"): value
        for key, value in metrics.items()
        if key.startswith("subject_")
    }
    result = {
        "primary_evaluation_unit": "subject",
        "aggregation": "mean_softmax_probability_over_all_valid_subject_windows",
        "subject_level": subject_level,
        "window_level": window_level,
        "subject_predictions": subject_predictions,
    }
    payload = json.dumps(result, indent=2, ensure_ascii=False)
    print(payload)
    if args.output:
        args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
        args.output.resolve().write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()

