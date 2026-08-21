from pathlib import Path

import pytest

from spa_gaitformer.config import load_config, validate_config


ROOT = Path(__file__).resolve().parents[1]


def test_smoke_config_is_valid() -> None:
    config = load_config(ROOT / "configs" / "smoke.yaml")
    validate_config(config, formal=True)


def test_formal_config_requires_unreported_values() -> None:
    config = load_config(ROOT / "configs" / "spa_mmd.yaml")
    with pytest.raises(ValueError, match="not recoverable"):
        validate_config(config, formal=True)

