"""Build temporally aligned RD maps from provisional DCA1000 frame blocks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    required = {
        "num_adc_samples",
        "chirps_per_hardware_frame",
        "num_rx",
        "num_tx",
        "iq_order",
    }
    missing = required.difference(config or {})
    if missing:
        raise ValueError(f"Radar config is missing: {sorted(missing)}")
    chirps = int(config["chirps_per_hardware_frame"])
    tx = int(config["num_tx"])
    if chirps % tx:
        raise ValueError(f"chirps_per_hardware_frame={chirps} is not divisible by num_tx={tx}")
    if config["iq_order"] not in {
        "sample_rx_iq",
        "lane2_rx_sample",
        "lane2_sample_rx",
    }:
        raise ValueError(f"Unsupported iq_order: {config['iq_order']}")
    return config


def values_per_hardware_frame(config: dict[str, Any]) -> int:
    return (
        int(config["num_adc_samples"])
        * int(config["chirps_per_hardware_frame"])
        * int(config["num_rx"])
        * 2
    )


def decode_frames(values: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    samples = int(config["num_adc_samples"])
    chirps = int(config["chirps_per_hardware_frame"])
    rx = int(config["num_rx"])
    per_frame = values_per_hardware_frame(config)
    if values.size % per_frame:
        raise ValueError(f"ADC chunk has {values.size} values, not a whole number of frames")
    frames = values.size // per_frame
    order = config["iq_order"]
    if order == "sample_rx_iq":
        iq = values.reshape(frames, chirps, samples, rx, 2)
        adc = iq[..., 0] + 1j * iq[..., 1]
        return adc.transpose(0, 1, 3, 2)

    groups = values.reshape(-1, 4)
    lanes = np.stack(
        (groups[:, 0] + 1j * groups[:, 2], groups[:, 1] + 1j * groups[:, 3]),
        axis=1,
    ).reshape(frames, -1)
    if order == "lane2_rx_sample":
        return lanes.reshape(frames, chirps, rx, samples)
    return lanes.reshape(frames, chirps, samples, rx).transpose(0, 1, 3, 2)


def rd_power(adc: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    tx = int(config["num_tx"])
    loops = adc.shape[1] // tx
    range_window = np.hanning(adc.shape[-1]).astype(np.float32)
    range_fft = np.fft.fft(adc * range_window, axis=-1)
    range_fft = range_fft.reshape(adc.shape[0], loops, tx, adc.shape[2], adc.shape[3])
    if bool(config.get("remove_static_clutter", True)):
        range_fft = range_fft - range_fft.mean(axis=1, keepdims=True)
    doppler_window = np.hanning(loops).astype(np.float32)
    doppler_fft = np.fft.fftshift(
        np.fft.fft(range_fft * doppler_window[None, :, None, None, None], axis=1),
        axes=1,
    )
    power = (np.abs(doppler_fft) ** 2).sum(axis=(2, 3))
    return power[:, :, : adc.shape[-1] // 2].transpose(0, 2, 1).astype(np.float32)


def compute_base_power(
    source: Path, config: dict[str, Any], chunk_frames: int
) -> tuple[np.ndarray, int]:
    raw = np.memmap(source, dtype="<i2", mode="r")
    per_frame = values_per_hardware_frame(config)
    complete_frames, trailing_values = divmod(raw.size, per_frame)
    if complete_frames == 0:
        raise ValueError(f"No complete radar frames in {source}")
    outputs = []
    for start in range(0, complete_frames, chunk_frames):
        count = min(chunk_frames, complete_frames - start)
        chunk = np.asarray(raw[start * per_frame : (start + count) * per_frame], dtype=np.float32)
        outputs.append(rd_power(decode_frames(chunk, config), config))
    return np.concatenate(outputs, axis=0), trailing_values


def aggregate_to_target(power: np.ndarray, target_frames: int) -> np.ndarray:
    if target_frames <= 0:
        raise ValueError("target_frames must be positive")
    if target_frames > power.shape[0]:
        raise ValueError(
            f"Cannot aggregate {power.shape[0]} hardware frames into {target_frames} RD frames"
        )
    edges = np.linspace(0, power.shape[0], target_frames + 1)
    boundaries = np.rint(edges).astype(int)
    boundaries[0], boundaries[-1] = 0, power.shape[0]
    if np.any(np.diff(boundaries) <= 0):
        raise ValueError("Temporal aggregation produced an empty interval")
    aggregated = np.stack(
        [power[boundaries[i] : boundaries[i + 1]].mean(axis=0) for i in range(target_frames)]
    )
    return (10.0 * np.log10(aggregated + 1e-6)).astype(np.float32)


def count_rgb_frames(path: Path) -> int:
    return sum(1 for item in path.iterdir() if item.suffix.lower() in IMAGE_SUFFIXES)


def convert_session(
    session_dir: Path,
    config: dict[str, Any],
    chunk_frames: int,
    overwrite: bool,
) -> Path:
    sources = sorted(
        path
        for path in (session_dir / "mmwave" / "raw").rglob("*.bin")
        if path.is_file() and not path.name.startswith("._")
    )
    if len(sources) != 1:
        raise ValueError(f"Expected one ADC bin in {session_dir}, found {len(sources)}")
    output_dir = session_dir / "mmwave" / "rdmap"
    output = output_dir / "rd.npy"
    metadata = output_dir / "rd_meta.json"
    if output.exists() and metadata.exists() and not overwrite:
        return output

    rgb_frames = count_rgb_frames(session_dir / "rgb")
    target_frames = rgb_frames // 3
    power, trailing_values = compute_base_power(sources[0], config, chunk_frames)
    rd = aggregate_to_target(power, target_frames)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output, rd)
    metadata.write_text(
        json.dumps(
            {
                "source": str(sources[0]),
                "source_bytes": sources[0].stat().st_size,
                "candidate_config": config,
                "hardware_frame_count": int(power.shape[0]),
                "trailing_int16_values": int(trailing_values),
                "rgb_frame_count": rgb_frames,
                "target_rd_frame_count": target_frames,
                "output_shape": list(rd.shape),
                "temporal_alignment": "uniform aggregation to floor(rgb_frames / 3)",
                "axis_units": "range-bin and Doppler-bin only; physical calibration unavailable",
                "status": "provisional reconstruction pending archived acquisition profile",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return output


def session_dirs(root: Path) -> list[Path]:
    return [
        path
        for subject in sorted(root.glob("S[0-9][0-9]"))
        for path in (subject / "walk", subject / "head_turn")
        if (path / "meta.json").is_file()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--session-dir", type=Path)
    parser.add_argument("--chunk-frames", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config.resolve())
    sessions = [args.session_dir.resolve()] if args.session_dir else session_dirs(args.processed_root.resolve())
    if not sessions:
        raise SystemExit("No completed processed sessions found")
    for session in sessions:
        output = convert_session(session, config, args.chunk_frames, args.overwrite)
        print(output, flush=True)


if __name__ == "__main__":
    main()
