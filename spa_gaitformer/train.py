from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import load_config, task_num_classes
from .dataset import SpAWindowDataset
from .engine import run_epoch
from .losses import ClassificationObjective
from .model import SpAGaitformer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SpA-Gaitformer")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--task", choices=["binary", "severity"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = int(config.get("seed", 2026))
    set_seed(seed)
    device = torch.device(args.device)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = config["data"]
    train_set = SpAWindowDataset(args.train_manifest, args.task, int(data_cfg["image_size"]))
    val_set = SpAWindowDataset(args.val_manifest, args.task, int(data_cfg["image_size"]))
    training_cfg = config["training"]
    loader_args = {
        "batch_size": int(training_cfg["batch_size"]),
        "num_workers": int(data_cfg.get("num_workers", 4)),
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)

    model = SpAGaitformer(config, task_num_classes(config, args.task)).to(device)
    objective = ClassificationObjective(training_cfg.get("class_weights"))
    if training_cfg.get("optimizer", "adam").lower() != "adam":
        raise ValueError("The paper-aligned optimizer is Adam")
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg.get("weight_decay", 0.0)),
    )

    history = []
    best_f1 = -1.0
    for epoch in range(1, int(training_cfg["epochs"]) + 1):
        train_metrics = run_epoch(model, train_loader, objective, device, optimizer)
        val_metrics = run_epoch(model, val_loader, objective, device)
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": config,
                    "task": args.task,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                output_dir / "best.pt",
            )
    (output_dir / "history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

