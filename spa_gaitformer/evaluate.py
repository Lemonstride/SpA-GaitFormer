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
    dataset = SpAWindowDataset(args.manifest, args.task, int(config["data"]["image_size"]))
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["data"].get("num_workers", 4)),
    )
    model = SpAGaitformer(config, task_num_classes(config, args.task)).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    metrics = run_epoch(model, loader, ClassificationObjective(), device)
    payload = json.dumps(metrics, indent=2, ensure_ascii=False)
    print(payload)
    if args.output:
        args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
        args.output.resolve().write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()

