from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from .metrics import aggregate_subject_probabilities, classification_metrics


def move_batch(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def run_epoch(
    model: nn.Module,
    batches: Iterable[dict[str, object]],
    objective: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    mixed_precision: bool = False,
    accumulation_steps: int = 1,
    include_subject_records: bool = False,
) -> dict[str, object]:
    training = optimizer is not None
    if accumulation_steps <= 0:
        raise ValueError("accumulation_steps must be positive")
    model.train(training)
    total_loss = 0.0
    total_examples = 0
    labels: list[int] = []
    predictions: list[int] = []
    probabilities: list[list[float]] = []
    subject_ids: list[str] = []
    context = torch.enable_grad if training else torch.no_grad
    if training:
        optimizer.zero_grad(set_to_none=True)
    pending_steps = 0
    with context():
        for raw_batch in batches:
            batch = move_batch(raw_batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=mixed_precision and device.type == "cuda",
            ):
                outputs = model(
                    batch["rgb"],
                    batch["skeleton_features"],
                    batch["rd_maps"],
                )
                target = batch["label"]
                loss = objective(outputs["logits"], target)
            if training:
                scaled_loss = loss / accumulation_steps
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                pending_steps += 1
                if pending_steps == accumulation_steps:
                    if scaler is not None and scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    pending_steps = 0
            size = int(target.size(0))
            total_loss += float(loss.detach()) * size
            total_examples += size
            labels.extend(target.detach().cpu().tolist())
            logits = outputs["logits"].detach()
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            probabilities.extend(torch.softmax(logits.float(), dim=1).cpu().tolist())
            subject_ids.extend(str(subject) for subject in raw_batch["subject_id"])
    if training and pending_steps:
        if scaler is not None and scaler.is_enabled():
            scaler.unscale_(optimizer)
            correction = accumulation_steps / pending_steps
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    if parameter.grad is not None:
                        parameter.grad.mul_(correction)
            scaler.step(optimizer)
            scaler.update()
        else:
            correction = accumulation_steps / pending_steps
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    if parameter.grad is not None:
                        parameter.grad.mul_(correction)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    metrics = classification_metrics(labels, predictions)
    metrics["loss"] = total_loss / max(total_examples, 1)
    metrics["window_count"] = total_examples
    subject_metrics, subject_records = aggregate_subject_probabilities(
        subject_ids, labels, probabilities
    )
    metrics.update({f"subject_{key}": value for key, value in subject_metrics.items()})
    metrics["subject_count"] = len(subject_records)
    if include_subject_records:
        metrics["subject_predictions"] = subject_records
    return metrics
