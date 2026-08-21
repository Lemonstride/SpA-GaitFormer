from pathlib import Path

import torch

from spa_gaitformer.config import load_config
from spa_gaitformer.losses import ClassificationObjective
from spa_gaitformer.model import SpAGaitformer


ROOT = Path(__file__).resolve().parents[1]


def test_multimodal_forward_and_cross_entropy_backward() -> None:
    torch.manual_seed(7)
    config = load_config(ROOT / "configs" / "smoke.yaml")
    model = SpAGaitformer(config, num_classes=2)
    outputs = model(
        rgb=torch.randn(2, 6, 3, 32, 32),
        skeleton_features=torch.randn(2, 6, 16),
        rd_maps=torch.randn(2, 2, 1, 16, 16),
    )
    assert outputs["logits"].shape == (2, 2)
    assert outputs["rgb_tokens"].shape == (2, 2, 32)
    assert outputs["skeleton_tokens"].shape == (2, 2, 32)
    assert outputs["radar_tokens"].shape == (2, 2, 32)
    loss = ClassificationObjective()(outputs["logits"], torch.tensor([0, 1]))
    loss.backward()
    assert torch.isfinite(loss)

