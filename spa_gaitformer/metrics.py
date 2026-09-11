from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def classification_metrics(labels: list[int], predictions: list[int]) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }


def aggregate_subject_probabilities(
    subject_ids: list[str],
    labels: list[int],
    probabilities: list[list[float]],
) -> tuple[dict[str, float], list[dict[str, object]]]:
    if not subject_ids or len(subject_ids) != len(labels) or len(labels) != len(probabilities):
        raise ValueError("Subject IDs, labels, and probabilities must have equal non-zero length")
    grouped: dict[str, dict[str, object]] = {}
    for subject_id, label, probability in zip(subject_ids, labels, probabilities):
        values = np.asarray(probability, dtype=np.float64)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError(f"Invalid probability vector for {subject_id}: {probability}")
        if subject_id not in grouped:
            grouped[subject_id] = {"label": int(label), "probabilities": []}
        elif grouped[subject_id]["label"] != int(label):
            raise ValueError(f"Inconsistent labels for subject {subject_id}")
        grouped[subject_id]["probabilities"].append(values)

    records: list[dict[str, object]] = []
    subject_labels: list[int] = []
    subject_predictions: list[int] = []
    for subject_id in sorted(grouped):
        group = grouped[subject_id]
        mean_probability = np.mean(np.stack(group["probabilities"]), axis=0)
        label = int(group["label"])
        prediction = int(mean_probability.argmax())
        subject_labels.append(label)
        subject_predictions.append(prediction)
        records.append(
            {
                "subject_id": subject_id,
                "label": label,
                "prediction": prediction,
                "mean_probabilities": mean_probability.tolist(),
                "window_count": len(group["probabilities"]),
            }
        )
    return classification_metrics(subject_labels, subject_predictions), records


def summarize_repeats(values: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    if not values:
        raise ValueError("No repeat metrics to summarize")
    return {
        key: {
            "mean": float(np.mean([item[key] for item in values])),
            "sample_std": float(np.std([item[key] for item in values], ddof=1)) if len(values) > 1 else 0.0,
        }
        for key in values[0]
    }
