from __future__ import annotations

import numpy as np

from scripts.build_rd_maps import aggregate_to_target, decode_frames, rd_power


CONFIG = {
    "num_adc_samples": 8,
    "chirps_per_hardware_frame": 6,
    "num_rx": 2,
    "num_tx": 3,
    "iq_order": "sample_rx_iq",
    "remove_static_clutter": True,
}


def test_decode_and_rd_shapes() -> None:
    values = np.arange(2 * 8 * 6 * 2 * 2, dtype=np.float32)
    adc = decode_frames(values, CONFIG)
    assert adc.shape == (2, 6, 2, 8)
    power = rd_power(adc, CONFIG)
    assert power.shape == (2, 4, 2)
    assert np.isfinite(power).all()


def test_temporal_aggregation_uses_every_hardware_frame() -> None:
    power = np.arange(12, dtype=np.float32).reshape(6, 2, 1) + 1
    result = aggregate_to_target(power, 3)
    expected = 10 * np.log10(np.array([[[2.0], [3.0]], [[6.0], [7.0]], [[10.0], [11.0]]]))
    assert np.allclose(result, expected.astype(np.float32))
