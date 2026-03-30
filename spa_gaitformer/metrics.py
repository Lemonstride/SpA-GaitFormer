from __future__ import annotations

import torch

from spa_gaitformer.config import TaskConfig


class MetricTracker:
    def __init__(self, task_config: TaskConfig) -> None:
        self.task_config = task_config
        self.batch_count = 0
        self.loss_sum = 0.0
        self.disease_loss_sum = 0.0
        self.severity_loss_sum = 0.0
        self.tp = 0.0
        self.tn = 0.0
        self.fp = 0.0
        self.fn = 0.0
        self.disease_prob_sum = 0.0
        self.disease_pred_sum = 0.0
        self.disease_count = 0.0
        self.disease_probs: list[float] = []
        self.disease_targets: list[float] = []
        self.severity_correct = 0.0
        self.severity_abs_error = 0.0
        self.severity_total = 0.0
        if task_config.severity_mode == "classification":
            self.severity_confusion = torch.zeros(
                task_config.severity_num_classes,
                task_config.severity_num_classes,
                dtype=torch.float64,
            )
        else:
            self.severity_confusion = None

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
        disease_prob = torch.sigmoid(outputs["disease_logits"].detach().cpu())
        disease_pred = (disease_prob >= 0.5).float()
        self.tp += float(((disease_pred == 1) & (disease_target == 1)).sum())
        self.tn += float(((disease_pred == 0) & (disease_target == 0)).sum())
        self.fp += float(((disease_pred == 1) & (disease_target == 0)).sum())
        self.fn += float(((disease_pred == 0) & (disease_target == 1)).sum())
        self.disease_prob_sum += float(disease_prob.sum())
        self.disease_pred_sum += float(disease_pred.sum())
        self.disease_count += float(disease_pred.numel())
        self.disease_probs.extend(disease_prob.flatten().tolist())
        self.disease_targets.extend(disease_target.flatten().tolist())

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
                for predicted, target in zip(
                    severity_pred[severity_mask].tolist(),
                    severity_target[severity_mask].tolist(),
                ):
                    self.severity_confusion[target, predicted] += 1
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
        return self._build_metrics()

    def compute(self) -> dict[str, float]:
        return self._build_metrics()

    def _build_metrics(self) -> dict[str, float]:
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
            "disease_pos_rate": self.disease_pred_sum / max(self.disease_count, 1.0),
            "disease_prob_mean": self.disease_prob_sum / max(self.disease_count, 1.0),
        }
        best_f1, best_threshold = self._best_disease_f1()
        metrics["disease_f1_best"] = best_f1
        metrics["disease_best_threshold"] = best_threshold
        if self.task_config.severity_mode == "classification":
            metrics["severity_acc"] = self.severity_correct / max(
                self.severity_total, 1.0
            )
            metrics["severity_macro_f1"] = self._severity_macro_f1()
        else:
            metrics["severity_mae"] = self.severity_abs_error / max(
                self.severity_total, 1.0
            )
        return metrics

    def _severity_macro_f1(self) -> float:
        if self.severity_confusion is None:
            return 0.0
        f1_scores = []
        for index in range(self.severity_confusion.shape[0]):
            tp = float(self.severity_confusion[index, index])
            fp = float(self.severity_confusion[:, index].sum() - tp)
            fn = float(self.severity_confusion[index, :].sum() - tp)
            precision = tp / max(tp + fp, 1.0)
            recall = tp / max(tp + fn, 1.0)
            if precision + recall == 0:
                f1_scores.append(0.0)
            else:
                f1_scores.append(2.0 * precision * recall / (precision + recall))
        return sum(f1_scores) / max(len(f1_scores), 1)

    def _best_disease_f1(self) -> tuple[float, float]:
        if not self.disease_probs:
            return 0.0, 0.5
        probs = torch.tensor(self.disease_probs, dtype=torch.float32)
        targets = torch.tensor(self.disease_targets, dtype=torch.float32)
        thresholds = torch.linspace(0.0, 1.0, steps=101)
        best_f1 = 0.0
        best_threshold = 0.5
        for threshold in thresholds:
            preds = (probs >= threshold).float()
            tp = float(((preds == 1) & (targets == 1)).sum())
            fp = float(((preds == 1) & (targets == 0)).sum())
            fn = float(((preds == 0) & (targets == 1)).sum())
            precision = tp / max(tp + fp, 1.0)
            recall = tp / max(tp + fn, 1.0)
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2.0 * precision * recall / (precision + recall)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = float(threshold)
        return best_f1, best_threshold
