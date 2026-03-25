from __future__ import annotations

import torch
import torch.nn.functional as F

from spa_gaitformer.config import TaskConfig


class MultiTaskCriterion:
    def __init__(self, task_config: TaskConfig, device: torch.device) -> None:
        self.task_config = task_config
        self.device = device
        self.pos_weight = None
        if task_config.disease_pos_weight is not None:
            self.pos_weight = torch.tensor(
                [task_config.disease_pos_weight],
                dtype=torch.float32,
                device=device,
            )

    def __call__(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        disease_target = batch["disease_label"].to(self.device)
        disease_loss = F.binary_cross_entropy_with_logits(
            outputs["disease_logits"],
            disease_target,
            pos_weight=self.pos_weight,
        )

        severity_mask = batch["severity_mask"].to(self.device) > 0
        severity_loss = torch.zeros((), device=self.device)
        if severity_mask.any():
            if self.task_config.severity_mode == "classification":
                severity_target = batch["severity_label"].to(self.device).long()
                severity_loss = F.cross_entropy(
                    outputs["severity_logits"][severity_mask],
                    severity_target[severity_mask],
                )
            else:
                severity_target = batch["severity_label"].to(self.device).float()
                severity_loss = F.mse_loss(
                    outputs["severity_logits"][severity_mask],
                    severity_target[severity_mask],
                )

        total_loss = (
            self.task_config.disease_loss_weight * disease_loss
            + self.task_config.severity_loss_weight * severity_loss
        )
        return total_loss, {
            "loss": float(total_loss.detach().cpu()),
            "disease_loss": float(disease_loss.detach().cpu()),
            "severity_loss": float(severity_loss.detach().cpu()),
        }
