from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .config import load_config, task_num_classes
from .dataset import SpAWindowDataset
from .engine import run_epoch
from .losses import ClassificationObjective
from .model import SpAGaitformer
from .sampling import subject_class_balanced_weights


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


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
    headturn_enabled = bool(config["model"].get("headturn", {}).get("enabled", False))
    train_set = SpAWindowDataset(
        args.train_manifest,
        args.task,
        int(data_cfg["image_size"]),
        augmentation=data_cfg.get("augmentation"),
        rd_normalization=data_cfg.get("rd_normalization", "none"),
        headturn_enabled=headturn_enabled,
    )
    val_set = SpAWindowDataset(
        args.val_manifest,
        args.task,
        int(data_cfg["image_size"]),
        rd_normalization=data_cfg.get("rd_normalization", "none"),
        headturn_enabled=headturn_enabled,
    )
    headturn_normalization = None
    if headturn_enabled:
        values = np.asarray(list(train_set.headturn_values_by_subject().values()), dtype=np.float64)
        mean = float(values.mean())
        std = float(values.std())
        if not np.isfinite(std) or std <= 1e-6:
            raise ValueError("Training subjects do not provide variable head-turn spans")
        train_set.set_headturn_normalization(mean, std)
        val_set.set_headturn_normalization(mean, std)
        headturn_normalization = {
            "mean_deg": mean,
            "std_deg": std,
            "fit_unit": "unique_training_subject",
            "subject_count": int(values.size),
        }
    training_cfg = config["training"]
    loader_args = {
        "batch_size": int(training_cfg["batch_size"]),
        "num_workers": int(data_cfg.get("num_workers", 4)),
        "pin_memory": device.type == "cuda",
    }
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampling = str(training_cfg.get("sampling", "window_uniform"))
    sampler = None
    if sampling == "subject_class_balanced":
        sampler_generator = torch.Generator()
        sampler_generator.manual_seed(seed)
        sampler = WeightedRandomSampler(
            subject_class_balanced_weights(train_set.rows, train_set.label_column),
            num_samples=len(train_set),
            replacement=True,
            generator=sampler_generator,
        )
    elif sampling != "window_uniform":
        raise ValueError(f"Unknown training sampling policy: {sampling}")
    train_loader = DataLoader(
        train_set,
        shuffle=sampler is None,
        sampler=sampler,
        worker_init_fn=seed_worker,
        generator=generator,
        **loader_args,
    )
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
    mixed_precision = bool(training_cfg.get("mixed_precision", False))
    accumulation_steps = int(training_cfg.get("gradient_accumulation_steps", 1))
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=mixed_precision and device.type == "cuda",
    )
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "config": config,
                "task": args.task,
                "seed": seed,
                "train_manifest": str(args.train_manifest.resolve()),
                "val_manifest": str(args.val_manifest.resolve()),
                "device": str(device),
                "mixed_precision": mixed_precision,
                "gradient_accumulation_steps": accumulation_steps,
                "effective_batch_size": int(training_cfg["batch_size"]) * accumulation_steps,
                "sampling": sampling,
                "headturn_normalization": headturn_normalization,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    history = []
    primary_metric = str(training_cfg.get("primary_metric", "subject_macro_f1"))
    best_score = -1.0
    for epoch in range(1, int(training_cfg["epochs"]) + 1):
        train_set.set_epoch(epoch - 1)
        train_metrics = run_epoch(
            model,
            train_loader,
            objective,
            device,
            optimizer,
            scaler=scaler,
            mixed_precision=mixed_precision,
            accumulation_steps=accumulation_steps,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            objective,
            device,
            mixed_precision=mixed_precision,
        )
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))
        if primary_metric not in val_metrics:
            raise KeyError(f"Unknown primary metric: {primary_metric}")
        score = float(val_metrics[primary_metric])
        if score > best_score:
            best_score = score
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": config,
                    "task": args.task,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "primary_metric": primary_metric,
                    "headturn_normalization": headturn_normalization,
                },
                output_dir / "best.pt",
            )
    (output_dir / "history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

