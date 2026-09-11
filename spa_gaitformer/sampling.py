from __future__ import annotations

from collections import Counter


def subject_class_balanced_weights(
    rows: list[dict[str, str]], label_column: str
) -> list[float]:
    if not rows:
        raise ValueError("Cannot build sampling weights for an empty dataset")
    windows_per_subject = Counter(row["subject_id"] for row in rows)
    label_by_subject: dict[str, str] = {}
    for row in rows:
        subject = row["subject_id"]
        label = row[label_column]
        if subject in label_by_subject and label_by_subject[subject] != label:
            raise ValueError(f"Inconsistent {label_column} for {subject}")
        label_by_subject[subject] = label
    subjects_per_class = Counter(label_by_subject.values())
    return [
        1.0
        / (
            subjects_per_class[label_by_subject[row["subject_id"]]]
            * windows_per_subject[row["subject_id"]]
        )
        for row in rows
    ]
