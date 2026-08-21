from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


REQUIRED_FIELDS = (
    "num_adc_samples",
    "chirps_per_frame",
    "num_rx",
    "num_tx",
    "num_frames",
    "adc_bits",
    "sample_rate_ksps",
    "frequency_slope_mhz_us",
    "frame_period_ms",
    "iq_order",
    "lane_interleave",
    "range_fft_size",
    "doppler_fft_size",
)


@dataclass(frozen=True)
class RadarConfig:
    num_adc_samples: int
    chirps_per_frame: int
    num_rx: int
    num_tx: int
    num_frames: int
    adc_bits: int
    sample_rate_ksps: float
    frequency_slope_mhz_us: float
    frame_period_ms: float
    iq_order: str
    lane_interleave: str
    range_fft_size: int
    doppler_fft_size: int
    window: str
    remove_static_clutter: bool

    @classmethod
    def from_yaml(cls, path: Path) -> "RadarConfig":
        with path.open("r", encoding="utf-8") as handle:
            payload: dict[str, Any] = yaml.safe_load(handle) or {}
        missing = [name for name in REQUIRED_FIELDS if payload.get(name) is None]
        if missing:
            raise ValueError(
                "ADC-to-RD conversion is blocked because acquisition evidence is missing: "
                + ", ".join(missing)
            )
        config = cls(**payload)
        config.validate()
        return config

    def validate(self) -> None:
        if self.adc_bits != 16:
            raise ValueError("This audited converter currently supports 16-bit ADC captures only")
        if self.iq_order not in {"iq", "qi"}:
            raise ValueError("iq_order must be 'iq' or 'qi'")
        if self.lane_interleave != "sample_rx_iq":
            raise ValueError(
                "lane_interleave must be 'sample_rx_iq'. Other DCA1000 lane layouts need an "
                "acquisition-specific decoder and must not be guessed."
            )
        if self.range_fft_size < self.num_adc_samples:
            raise ValueError("range_fft_size cannot be smaller than num_adc_samples")
        if self.doppler_fft_size < self.chirps_per_frame // self.num_tx:
            raise ValueError("doppler_fft_size is too small for chirps per TX")


def adc_to_rd(raw_path: Path, config: RadarConfig) -> np.ndarray:
    raw = np.fromfile(raw_path, dtype="<i2")
    complex_samples = (
        config.num_frames
        * config.chirps_per_frame
        * config.num_rx
        * config.num_adc_samples
    )
    expected_values = complex_samples * 2
    if raw.size != expected_values:
        raise ValueError(
            f"Raw ADC length mismatch: expected {expected_values} int16 values from config, got {raw.size}"
        )
    raw = raw.reshape(
        config.num_frames,
        config.chirps_per_frame,
        config.num_adc_samples,
        config.num_rx,
        2,
    )
    first, second = raw[..., 0].astype(np.float32), raw[..., 1].astype(np.float32)
    adc = first + 1j * second if config.iq_order == "iq" else second + 1j * first
    adc = adc.transpose(0, 1, 3, 2)

    range_window = np.hanning(config.num_adc_samples).astype(np.float32)
    range_fft = np.fft.fft(adc * range_window, n=config.range_fft_size, axis=-1)
    chirps_per_tx = config.chirps_per_frame // config.num_tx
    if chirps_per_tx * config.num_tx != config.chirps_per_frame:
        raise ValueError("chirps_per_frame must be divisible by num_tx")
    range_fft = range_fft.reshape(
        config.num_frames,
        chirps_per_tx,
        config.num_tx,
        config.num_rx,
        config.range_fft_size,
    )
    if config.remove_static_clutter:
        range_fft = range_fft - range_fft.mean(axis=1, keepdims=True)
    doppler_window = np.hanning(chirps_per_tx).astype(np.float32)
    doppler_fft = np.fft.fftshift(
        np.fft.fft(
            range_fft * doppler_window[None, :, None, None, None],
            n=config.doppler_fft_size,
            axis=1,
        ),
        axes=1,
    )
    power = np.abs(doppler_fft) ** 2
    rd = 10.0 * np.log10(power.sum(axis=(2, 3)) + 1e-6)
    return rd.transpose(0, 2, 1).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Configuration-gated ADC to RD conversion")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RadarConfig.from_yaml(args.config.resolve())
    rd = adc_to_rd(args.input.resolve(), config)
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output.resolve(), rd)
    print(f"Saved RD maps {rd.shape} to {args.output.resolve()}")


if __name__ == "__main__":
    main()

