from __future__ import annotations

import torch


def validate_three_to_one_lengths(rgb_steps: int, skeleton_steps: int, rd_steps: int) -> None:
    expected = rd_steps * 3
    if rgb_steps != expected or skeleton_steps != expected:
        raise ValueError(
            "Strict frame-feature alignment failed: expected "
            f"RGB={expected} and skeleton={expected} for RD={rd_steps}, "
            f"got RGB={rgb_steps}, skeleton={skeleton_steps}."
        )


def pool_three_frame_features(features: torch.Tensor) -> torch.Tensor:
    """Mean-pool exact non-overlapping groups of three frame-level features."""
    if features.ndim != 3:
        raise ValueError(f"Expected [batch, frames, dim], got {tuple(features.shape)}")
    batch, frames, dim = features.shape
    if frames == 0 or frames % 3:
        raise ValueError(f"Frame count must be a positive multiple of 3, got {frames}")
    return features.reshape(batch, frames // 3, 3, dim).mean(dim=2)

