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

