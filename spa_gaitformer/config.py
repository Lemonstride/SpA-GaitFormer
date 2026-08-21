from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    validate_config(config)
    return config


def validate_config(config: dict[str, Any], *, formal: bool = False) -> None:
    ratio = config.get("data", {}).get("ratio_rgb_to_rd")
    if ratio != 3:
        raise ValueError(f"SpA-Gaitformer requires an exact RGB/skeleton-to-RD ratio of 3, got {ratio!r}")

    model = config.get("model", {})
    shared_dim = int(model.get("shared_dim", 0))
    if shared_dim <= 0:
        raise ValueError("model.shared_dim must be positive")
    for section in ("radar", "fusion"):
        heads = int(model.get(section, {}).get("transformer_heads" if section == "radar" else "heads", 0))
        if heads <= 0 or shared_dim % heads:
            raise ValueError(f"model.shared_dim={shared_dim} must be divisible by {section} heads={heads}")

    if formal:
        missing = [
            key
            for key in ("rd_window", "rd_stride")
            if config.get("data", {}).get(key) is None
        ]
        missing.extend(
            key
            for key in ("train_ratio", "val_ratio")
            if config.get("evaluation", {}).get(key) is None
        )
        if missing:
            raise ValueError(
                "Formal reproduction needs explicit values not recoverable from the manuscript: "
                + ", ".join(missing)
            )


def task_num_classes(config: dict[str, Any], task: str) -> int:
    classes = config["model"]["num_classes"]
    if task not in classes:
        raise ValueError(f"Unknown task {task!r}; expected one of {sorted(classes)}")
    return int(classes[task])

