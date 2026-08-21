from pathlib import Path

import pytest

from spa_gaitformer.radar import RadarConfig


def test_radar_config_does_not_guess_missing_acquisition_values(tmp_path: Path) -> None:
    config = tmp_path / "radar.yaml"
    config.write_text("num_adc_samples: null\n", encoding="utf-8")
    with pytest.raises(ValueError, match="acquisition evidence is missing"):
        RadarConfig.from_yaml(config)

