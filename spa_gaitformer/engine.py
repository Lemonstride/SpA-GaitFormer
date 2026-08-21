from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from .metrics import classification_metrics


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
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_examples = 0
    labels: list[int] = []
    predictions: list[int] = []
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for raw_batch in batches:
            batch = move_batch(raw_batch, device)
            outputs = model(
                batch["rgb"],
                batch["skeleton_features"],
                batch["rd_maps"],
            )
            target = batch["label"]
            loss = objective(outputs["logits"], target)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            size = int(target.size(0))
            total_loss += float(loss.detach()) * size
            total_examples += size
            labels.extend(target.detach().cpu().tolist())
            predictions.extend(outputs["logits"].argmax(dim=1).detach().cpu().tolist())
    metrics = classification_metrics(labels, predictions)
    metrics["loss"] = total_loss / max(total_examples, 1)
    return metrics

