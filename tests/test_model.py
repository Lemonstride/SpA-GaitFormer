from copy import deepcopy
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


def test_from_scratch_skeletongait_pp_forward() -> None:
    torch.manual_seed(11)
    config = deepcopy(load_config(ROOT / "configs" / "smoke.yaml"))
    config["model"]["skeleton"] = {
        "backend": "skeletongait_pp",
        "feature_dim": 4096,
        "checkpoint": None,
        "trainable": True,
        "opengait_root": str(ROOT / "third_party" / "OpenGait"),
        "blocks": [1, 1, 1, 1],
        "channel_multiplier": 2,
    }
    model = SpAGaitformer(config, num_classes=2)
    outputs = model(
        rgb=torch.randn(1, 6, 3, 32, 32),
        skeleton_features=torch.randn(1, 6, 3, 64, 44),
        rd_maps=torch.randn(1, 2, 1, 16, 16),
    )
    assert outputs["logits"].shape == (1, 2)
    assert outputs["skeleton_tokens"].shape == (1, 2, 32)


def test_headturn_token_forward_and_backward() -> None:
    torch.manual_seed(13)
    config = deepcopy(load_config(ROOT / "configs" / "smoke.yaml"))
    config["model"]["headturn"] = {"enabled": True}
    model = SpAGaitformer(config, num_classes=2)
    headturn = torch.tensor([[-1.0], [1.0]])
    outputs = model(
        rgb=torch.randn(2, 6, 3, 32, 32),
        skeleton_features=torch.randn(2, 6, 16),
        rd_maps=torch.randn(2, 2, 1, 16, 16),
        headturn=headturn,
    )
    assert outputs["logits"].shape == (2, 2)
    assert outputs["headturn_token"].shape == (2, 32)
    outputs["logits"].sum().backward()
    assert model.headturn_branch is not None
    assert any(parameter.grad is not None for parameter in model.headturn_branch.parameters())

