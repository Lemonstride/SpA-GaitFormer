from __future__ import annotations

import json
import math
import random
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from spa_gaitformer.config import ExperimentConfig
from spa_gaitformer.data import (
    SampleRecord,
    SpAMMDDataset,
    create_stratified_folds,
    load_records_for_split,
    summarize_records,
)
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


def _build_loader_from_records(
    records: list[SampleRecord],
    config: ExperimentConfig,
    shuffle: bool,
) -> DataLoader:
    dataset = SpAMMDDataset(
        data_config=config.data,
        task_config=config.task,
        records=records,
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


def _log_primary_metric(metrics: dict[str, float], config: ExperimentConfig) -> str:
    if config.task.severity_mode == "classification":
        return (
            f"disease_f1={metrics['disease_f1']:.4f} "
            f"severity_macro_f1={metrics['severity_macro_f1']:.4f}"
        )
    return (
        f"disease_f1={metrics['disease_f1']:.4f} "
        f"severity_mae={metrics['severity_mae']:.4f}"
    )


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
                    f"disease_acc={metric_logs['disease_acc']:.4f}"
                )

    return tracker.compute()


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    config: ExperimentConfig,
    epoch: int,
    best_metric: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
        "config": config.to_dict(),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)


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


def _selection_metric_direction(metric_name: str) -> str:
    if metric_name in {"loss", "severity_mae"}:
        return "min"
    return "max"


def _is_better(candidate: float, best: float, metric_name: str) -> bool:
    direction = _selection_metric_direction(metric_name)
    if direction == "min":
        return candidate < best
    return candidate > best


def _initial_best(metric_name: str) -> float:
    if _selection_metric_direction(metric_name) == "min":
        return float("inf")
    return float("-inf")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _train_with_validation(
    config: ExperimentConfig,
    train_records: list[SampleRecord],
    val_records: list[SampleRecord],
    output_dir: Path,
) -> dict[str, object]:
    set_seed(config.train.seed)
    device = _resolve_device(config.train.device)
    train_loader = _build_loader_from_records(train_records, config, shuffle=True)
    val_loader = _build_loader_from_records(val_records, config, shuffle=False)
    model = DualStreamSpAGaitFormer(config.data, config.model, config.task).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=config.train.amp and device.type == "cuda"
    )
    selection_metric = config.cv.selection_metric
    best_metric = _initial_best(selection_metric)
    best_epoch = 0
    best_logs: dict[str, float] = {}
    history: list[dict[str, object]] = []

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_dir / "split_summary.json",
        {
            "train_summary": summarize_records(train_records),
            "val_summary": summarize_records(val_records),
            "train_ids": [record.subject_id for record in train_records],
            "val_ids": [record.subject_id for record in val_records],
        },
    )

    for epoch in range(1, config.train.epochs + 1):
        train_logs = _run_epoch(model, train_loader, config, device, optimizer, scaler)
        val_logs = _run_epoch(model, val_loader, config, device, None, None)
        epoch_log = {
            "epoch": epoch,
            "train": train_logs,
            "val": val_logs,
        }
        history.append(epoch_log)
        print(
            f"epoch={epoch} "
            f"train_loss={train_logs['loss']:.4f} "
            f"val_loss={val_logs['loss']:.4f} "
            f"{_log_primary_metric(val_logs, config)}"
        )

        candidate = float(val_logs[selection_metric])
        if _is_better(candidate, best_metric, selection_metric):
            best_metric = candidate
            best_epoch = epoch
            best_logs = dict(val_logs)
            _save_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                config,
                epoch,
                best_metric,
            )

        _save_checkpoint(
            output_dir / "latest.pt",
            model,
            optimizer,
            config,
            epoch,
            best_metric,
        )

    summary = {
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "selection_metric": selection_metric,
        "best_val_metrics": best_logs,
        "history": history,
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def cross_validate(config: ExperimentConfig) -> dict[str, object]:
    train_records = load_records_for_split("train", config.data)
    folds = create_stratified_folds(
        records=train_records,
        num_folds=config.cv.num_folds,
        seed=config.cv.seed,
    )
    cv_root = Path(config.train.checkpoint_dir) / "cv"
    fold_results = []
    best_epochs = []

    print(f"cross_validation_records={len(train_records)}")
    print(f"cross_validation_summary={summarize_records(train_records)}")

    for fold_index, (fold_train, fold_val) in enumerate(folds, start=1):
        print(
            f"fold={fold_index} "
            f"train_size={len(fold_train)} "
            f"val_size={len(fold_val)}"
        )
        summary = _train_with_validation(
            config=config,
            train_records=fold_train,
            val_records=fold_val,
            output_dir=cv_root / f"fold_{fold_index}",
        )
        best_epochs.append(int(summary["best_epoch"]))
        fold_results.append(
            {
                "fold": fold_index,
                "best_epoch": int(summary["best_epoch"]),
                "best_metric": float(summary["best_metric"]),
                "best_val_metrics": summary["best_val_metrics"],
            }
        )

    aggregate = _aggregate_fold_metrics(
        [result["best_val_metrics"] for result in fold_results]
    )
    recommended_epoch = config.train.epochs
    if config.cv.use_mean_best_epoch and best_epochs:
        recommended_epoch = max(1, int(round(sum(best_epochs) / len(best_epochs))))
    if config.cv.final_epochs is not None:
        recommended_epoch = config.cv.final_epochs

    summary = {
        "num_folds": config.cv.num_folds,
        "selection_metric": config.cv.selection_metric,
        "folds": fold_results,
        "aggregate": aggregate,
        "recommended_final_epoch": recommended_epoch,
    }
    _write_json(cv_root / "summary.json", summary)
    print(f"cv_aggregate={aggregate}")
    print(f"recommended_final_epoch={recommended_epoch}")
    return summary


def train_final(config: ExperimentConfig) -> dict[str, object]:
    set_seed(config.train.seed)
    device = _resolve_device(config.train.device)
    train_records = load_records_for_split("train", config.data)
    train_loader = _build_loader_from_records(train_records, config, shuffle=True)
    model = DualStreamSpAGaitFormer(config.data, config.model, config.task).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=config.train.amp and device.type == "cuda"
    )

    final_dir = Path(config.train.checkpoint_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        final_dir / "split_summary.json",
        {
            "train_summary": summarize_records(train_records),
            "train_ids": [record.subject_id for record in train_records],
        },
    )

    final_epochs = _resolve_final_epochs(config)
    history = []
    for epoch in range(1, final_epochs + 1):
        train_logs = _run_epoch(model, train_loader, config, device, optimizer, scaler)
        history.append({"epoch": epoch, "train": train_logs})
        print(
            f"epoch={epoch} "
            f"train_loss={train_logs['loss']:.4f} "
            f"{_log_primary_metric(train_logs, config)}"
        )
        _save_checkpoint(
            final_dir / "latest.pt",
            model,
            optimizer,
            config,
            epoch,
            float(train_logs["loss"]),
        )

    _save_checkpoint(
        final_dir / "final.pt",
        model,
        optimizer,
        config,
        final_epochs,
        float(history[-1]["train"]["loss"]),
    )
    summary = {
        "epochs": final_epochs,
        "train_summary": summarize_records(train_records),
        "history": history,
    }
    _write_json(final_dir / "summary.json", summary)
    return summary


def _resolve_final_epochs(config: ExperimentConfig) -> int:
    if config.cv.final_epochs is not None:
        return config.cv.final_epochs
    summary_path = Path(config.train.checkpoint_dir) / "cv" / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if "recommended_final_epoch" in summary:
            return int(summary["recommended_final_epoch"])
    return config.train.epochs


def _aggregate_fold_metrics(metrics_list: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    if not metrics_list:
        return {}
    keys = sorted(metrics_list[0].keys())
    aggregated: dict[str, dict[str, float]] = {}
    for key in keys:
        values = [float(item[key]) for item in metrics_list]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        aggregated[key] = {
            "mean": mean,
            "std": math.sqrt(variance),
        }
    return aggregated


def evaluate(
    config: ExperimentConfig,
    split: str = "test",
    checkpoint_path: str | Path | None = None,
) -> dict[str, float]:
    set_seed(config.train.seed)
    device = _resolve_device(config.train.device)
    records = load_records_for_split(split, config.data)
    loader = _build_loader_from_records(records, config, shuffle=False)
    model = DualStreamSpAGaitFormer(config.data, config.model, config.task).to(device)

    checkpoint = Path(
        checkpoint_path if checkpoint_path is not None else _default_eval_checkpoint(config)
    )
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    _load_checkpoint(checkpoint, model)
    metrics = _run_epoch(model, loader, config, device, None, None)
    print(f"evaluate_summary={summarize_records(records)}")
    return metrics


def _default_eval_checkpoint(config: ExperimentConfig) -> Path:
    final_checkpoint = Path(config.train.checkpoint_dir) / "final" / "final.pt"
    if final_checkpoint.exists():
        return final_checkpoint
    return Path(config.train.checkpoint_dir) / "best.pt"
