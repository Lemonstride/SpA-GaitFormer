from __future__ import annotations

import json
import random
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from spa_gaitformer.config import ExperimentConfig
from spa_gaitformer.data import SpAMMDDataset
from spa_gaitformer.losses import MultiTaskCriterion
from spa_gaitformer.metrics import MetricTracker
from spa_gaitformer.model import DualStreamSpAGaitFormer


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device_name: str) -> torch.device:
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def _build_loader(
    split: str,
    config: ExperimentConfig,
    shuffle: bool,
) -> DataLoader:
    dataset = SpAMMDDataset(
        split=split,
        data_config=config.data,
        task_config=config.task,
    )
    return DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        shuffle=shuffle,
        num_workers=config.train.num_workers,
        pin_memory=config.train.device.startswith("cuda"),
        drop_last=False,
    )


def _create_amp_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.cuda.amp.autocast()
    return nullcontext()


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    config: ExperimentConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler | None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    criterion = MultiTaskCriterion(config.task, device=device)
    tracker = MetricTracker(config.task)
    grad_context = nullcontext() if is_train else torch.no_grad()

    with grad_context:
        for step, batch in enumerate(loader, start=1):
            rgb = batch["rgb"].to(device)
            skeleton = batch["skeleton"].to(device)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with _create_amp_context(device, config.train.amp):
                outputs = model(rgb, skeleton)
                loss, loss_logs = criterion(outputs, batch)

            if is_train:
                if scaler is not None and config.train.amp and device.type == "cuda":
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            metric_logs = tracker.update(outputs, batch, loss_logs)
            if is_train and step % config.train.log_interval == 0:
                print(
                    f"step={step} "
                    f"loss={metric_logs['loss']:.4f} "
                    f"disease_acc={metric_logs['disease_acc']:.4f} "
                    f"{tracker.severity_metric_name}="
                    f"{metric_logs[tracker.severity_metric_name]:.4f}"
                )

    return tracker.compute()


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
    epoch: int,
    best_metric: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_metric": best_metric,
            "config": config.to_dict(),
        },
        path,
    )


def _load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, object]:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint


def _severity_metric_name(config: ExperimentConfig) -> str:
    if config.task.severity_mode == "classification":
        return "severity_acc"
    return "severity_mae"


def train(
    config: ExperimentConfig,
    resume_checkpoint: str | Path | None = None,
) -> None:
    set_seed(config.train.seed)
    device = _resolve_device(config.train.device)
    train_loader = _build_loader("train", config, shuffle=True)
    val_loader = _build_loader("val", config, shuffle=False)

    model = DualStreamSpAGaitFormer(config.data, config.model, config.task).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=config.train.amp and device.type == "cuda"
    )

    start_epoch = 1
    best_loss = float("inf")
    if resume_checkpoint is not None:
        checkpoint = _load_checkpoint(Path(resume_checkpoint), model, optimizer)
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_loss = float(checkpoint.get("best_metric", float("inf")))

    checkpoint_dir = Path(config.train.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config_path = checkpoint_dir / "resolved_config.json"
    config_path.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")

    severity_metric = _severity_metric_name(config)
    for epoch in range(start_epoch, config.train.epochs + 1):
        train_logs = _run_epoch(model, train_loader, config, device, optimizer, scaler)
        val_logs = _run_epoch(model, val_loader, config, device, None, None)
        print(
            f"epoch={epoch} "
            f"train_loss={train_logs['loss']:.4f} "
            f"val_loss={val_logs['loss']:.4f} "
            f"val_disease_acc={val_logs['disease_acc']:.4f} "
            f"val_{severity_metric}={val_logs[severity_metric]:.4f}"
        )

        if val_logs["loss"] < best_loss:
            best_loss = val_logs["loss"]
            _save_checkpoint(
                checkpoint_dir / "best.pt",
                model,
                optimizer,
                config,
                epoch,
                best_loss,
            )
        _save_checkpoint(
            checkpoint_dir / "latest.pt",
            model,
            optimizer,
            config,
            epoch,
            best_loss,
        )


def evaluate(
    config: ExperimentConfig,
    split: str = "val",
    checkpoint_path: str | Path | None = None,
) -> dict[str, float]:
    set_seed(config.train.seed)
    device = _resolve_device(config.train.device)
    loader = _build_loader(split, config, shuffle=False)
    model = DualStreamSpAGaitFormer(config.data, config.model, config.task).to(device)

    resolved_checkpoint = checkpoint_path
    if resolved_checkpoint is None:
        resolved_checkpoint = Path(config.train.checkpoint_dir) / "best.pt"
    checkpoint = Path(resolved_checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    _load_checkpoint(checkpoint, model)

    return _run_epoch(model, loader, config, device, None, None)
