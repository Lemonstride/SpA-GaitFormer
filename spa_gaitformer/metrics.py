from __future__ import annotations

import torch

from spa_gaitformer.config import TaskConfig


class MetricTracker:
    def __init__(self, task_config: TaskConfig) -> None:
        self.task_config = task_config
        self.severity_metric_name = (
            "severity_acc"
            if task_config.severity_mode == "classification"
            else "severity_mae"
        )
        self.batch_count = 0
        self.loss_sum = 0.0
        self.disease_loss_sum = 0.0
        self.severity_loss_sum = 0.0
        self.tp = 0.0
        self.tn = 0.0
        self.fp = 0.0
        self.fn = 0.0
        self.severity_correct = 0.0
        self.severity_abs_error = 0.0
        self.severity_total = 0.0

    def update(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        loss_logs: dict[str, float],
    ) -> dict[str, float]:
        self.batch_count += 1
        self.loss_sum += loss_logs["loss"]
        self.disease_loss_sum += loss_logs["disease_loss"]
        self.severity_loss_sum += loss_logs["severity_loss"]

        disease_target = batch["disease_label"].float()
        disease_pred = (
            torch.sigmoid(outputs["disease_logits"].detach().cpu()) >= 0.5
        ).float()
        self.tp += float(((disease_pred == 1) & (disease_target == 1)).sum())
        self.tn += float(((disease_pred == 0) & (disease_target == 0)).sum())
        self.fp += float(((disease_pred == 1) & (disease_target == 0)).sum())
        self.fn += float(((disease_pred == 0) & (disease_target == 1)).sum())

        severity_mask = batch["severity_mask"] > 0
        if severity_mask.any():
            if self.task_config.severity_mode == "classification":
                severity_target = batch["severity_label"].long()
                severity_pred = outputs["severity_logits"].detach().cpu().argmax(dim=-1)
                self.severity_correct += float(
                    (
                        severity_pred[severity_mask]
                        == severity_target[severity_mask]
                    ).sum()
                )
            else:
                severity_target = batch["severity_label"].float()
                severity_pred = outputs["severity_logits"].detach().cpu()
                self.severity_abs_error += float(
                    (
                        severity_pred[severity_mask]
                        - severity_target[severity_mask]
                    ).abs().sum()
                )
            self.severity_total += float(severity_mask.sum())

        return self.current()

    def current(self) -> dict[str, float]:
        disease_total = self.tp + self.tn + self.fp + self.fn
        disease_acc = (self.tp + self.tn) / max(disease_total, 1.0)
        if self.task_config.severity_mode == "classification":
            severity_value = self.severity_correct / max(self.severity_total, 1.0)
        else:
            severity_value = self.severity_abs_error / max(self.severity_total, 1.0)
        return {
            "loss": self.loss_sum / max(self.batch_count, 1),
            "disease_loss": self.disease_loss_sum / max(self.batch_count, 1),
            "severity_loss": self.severity_loss_sum / max(self.batch_count, 1),
            "disease_acc": disease_acc,
            self.severity_metric_name: severity_value,
        }

    def compute(self) -> dict[str, float]:
        disease_total = self.tp + self.tn + self.fp + self.fn
        disease_acc = (self.tp + self.tn) / max(disease_total, 1.0)
        disease_precision = self.tp / max(self.tp + self.fp, 1.0)
        disease_recall = self.tp / max(self.tp + self.fn, 1.0)
        disease_f1 = (
            2.0 * disease_precision * disease_recall
            / max(disease_precision + disease_recall, 1e-8)
        )
        metrics = {
            "loss": self.loss_sum / max(self.batch_count, 1),
            "disease_loss": self.disease_loss_sum / max(self.batch_count, 1),
            "severity_loss": self.severity_loss_sum / max(self.batch_count, 1),
            "disease_acc": disease_acc,
            "disease_precision": disease_precision,
            "disease_recall": disease_recall,
            "disease_f1": disease_f1,
        }
        if self.task_config.severity_mode == "classification":
            metrics["severity_acc"] = self.severity_correct / max(
                self.severity_total, 1.0
            )
        else:
            metrics["severity_mae"] = self.severity_abs_error / max(
                self.severity_total, 1.0
            )
        return metrics
