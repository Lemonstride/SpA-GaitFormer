from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _probability(value: Any, name: str, *, allow_one: bool = True) -> float:
    number = float(value)
    upper_valid = number <= 1.0 if allow_one else number < 1.0
    if number < 0.0 or not upper_valid:
        upper = "1" if allow_one else "1 (exclusive)"
        raise ValueError(f"{name} must be between 0 and {upper}, got {number}")
    return number


def _validate_augmentation(data: dict[str, Any]) -> None:
    augmentation = data.get("augmentation")
    if not augmentation or not augmentation.get("enabled", True):
        return
    rgb = augmentation.get("rgb", {})
    crop_scale_min = float(rgb.get("crop_scale_min", 1.0))
    if not 0.0 < crop_scale_min <= 1.0:
        raise ValueError("data.augmentation.rgb.crop_scale_min must be in (0, 1]")
    _probability(rgb.get("horizontal_flip_p", 0.0), "horizontal_flip_p")
    for name in ("brightness", "contrast", "saturation"):
        _probability(rgb.get(name, 0.0), name, allow_one=False)

    skeleton = augmentation.get("skeleton", {})
    if float(skeleton.get("noise_std", 0.0)) < 0.0:
        raise ValueError("skeleton.noise_std must be non-negative")
    _probability(
        skeleton.get("feature_dropout_p", 0.0),
        "skeleton.feature_dropout_p",
        allow_one=False,
    )

    radar = augmentation.get("radar", {})
    amplitude = radar.get("amplitude_scale", [1.0, 1.0])
    if not isinstance(amplitude, list) or len(amplitude) != 2:
        raise ValueError("radar.amplitude_scale must be [minimum, maximum]")
    if float(amplitude[0]) <= 0.0 or float(amplitude[0]) > float(amplitude[1]):
        raise ValueError("radar.amplitude_scale must be positive and ordered")
    if float(radar.get("noise_std", 0.0)) < 0.0:
        raise ValueError("radar.noise_std must be non-negative")

    temporal = augmentation.get("temporal", {})
    _probability(temporal.get("mask_p", 0.0), "temporal.mask_p")
    max_fraction = float(temporal.get("max_mask_fraction", 0.1))
    if not 0.0 < max_fraction <= 1.0:
        raise ValueError("temporal.max_mask_fraction must be in (0, 1]")


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    validate_config(config)
    return config


def validate_config(config: dict[str, Any], *, formal: bool = False) -> None:
    data = config.get("data", {})
    ratio = data.get("ratio_rgb_to_rd")
    if ratio != 3:
        raise ValueError(f"SpA-Gaitformer requires an exact RGB/skeleton-to-RD ratio of 3, got {ratio!r}")
    if data.get("rd_normalization", "none") not in {"none", "per_window"}:
        raise ValueError("data.rd_normalization must be 'none' or 'per_window'")
    _validate_augmentation(data)

    model = config.get("model", {})
    shared_dim = int(model.get("shared_dim", 0))
    if shared_dim <= 0:
        raise ValueError("model.shared_dim must be positive")
    for section in ("radar", "fusion"):
        heads = int(model.get(section, {}).get("transformer_heads" if section == "radar" else "heads", 0))
        if heads <= 0 or shared_dim % heads:
            raise ValueError(f"model.shared_dim={shared_dim} must be divisible by {section} heads={heads}")
    headturn = model.get("headturn", {})
    if "enabled" in headturn and not isinstance(headturn["enabled"], bool):
        raise ValueError("model.headturn.enabled must be a boolean")

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

