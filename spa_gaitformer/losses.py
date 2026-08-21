from __future__ import annotations

import torch
from torch import nn


class ClassificationObjective(nn.Module):
    """Paper-aligned unweighted batch-mean supervised cross-entropy."""

    def __init__(self, class_weights: list[float] | None = None) -> None:
        super().__init__()
        if class_weights is not None:
            raise ValueError(
                "The reported SpA-MMD objective is unweighted cross-entropy; "
                "class weights require a separately documented experiment."
            )
        self.cross_entropy = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.cross_entropy(logits, labels)

