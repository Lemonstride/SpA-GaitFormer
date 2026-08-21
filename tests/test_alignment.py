import pytest
import torch

from spa_gaitformer.alignment import pool_three_frame_features, validate_three_to_one_lengths


def test_three_frame_pooling_is_exact_mean() -> None:
    features = torch.arange(12, dtype=torch.float32).reshape(1, 6, 2)
    pooled = pool_three_frame_features(features)
    expected = torch.tensor([[[2.0, 3.0], [8.0, 9.0]]])
    torch.testing.assert_close(pooled, expected)


def test_three_frame_pooling_rejects_remainder() -> None:
    with pytest.raises(ValueError, match="multiple of 3"):
        pool_three_frame_features(torch.zeros(1, 5, 4))


def test_length_validator_rejects_nominal_but_nonexact_ratio() -> None:
    with pytest.raises(ValueError, match="Strict frame-feature alignment failed"):
        validate_three_to_one_lengths(rgb_steps=7, skeleton_steps=6, rd_steps=2)

