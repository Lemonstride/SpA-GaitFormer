from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import torch
from torch import nn


def build_model(opengait_root: Path, checkpoint: Path) -> nn.Module:
    opengait_package = opengait_root / "opengait"
    sys.path.insert(0, str(opengait_package))
    import modeling.base_model  # noqa: F401

    models_package = types.ModuleType("modeling.models")
    models_package.__path__ = [str(opengait_package / "modeling" / "models")]
    sys.modules["modeling.models"] = models_package
    source = opengait_package / "modeling" / "models" / "skeletongait++.py"
    spec = importlib.util.spec_from_file_location("modeling.models.skeletongait_pp", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load official SkeletonGait++ source: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    model = module.SkeletonGaitPP.__new__(module.SkeletonGaitPP)
    nn.Module.__init__(model)
    model.build_network(
        {
            "Backbone": {"in_channels": 3, "blocks": [1, 1, 1, 1], "C": 2},
            "SeparateBNNecks": {"class_num": 250},
            "use_emb2": False,
        }
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = state.get("model", state.get("state_dict", state)) if isinstance(state, dict) else state
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise ValueError(f"Unexpected SkeletonGait++ checkpoint keys: {unexpected[:10]}")
    if len(missing) > 8:
        raise ValueError(f"Too many missing SkeletonGait++ checkpoint keys: {missing[:10]}")
    return model


def extract_chunk(model: nn.Module, chunk: torch.Tensor) -> torch.Tensor:
    captured: list[torch.Tensor] = []
    handle = model.layer4.register_forward_hook(lambda _module, _inputs, output: captured.append(output))
    labels = torch.zeros(chunk.size(0), dtype=torch.long, device=chunk.device)
    try:
        model([[chunk], labels, None, None, None])
    finally:
        handle.remove()
    stage = captured[0]
    batch, channels, frames, height, width = stage.shape
    frame_maps = stage.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
    parts = model.HPP(frame_maps)
    features = model.FCs(parts).reshape(batch, frames, -1)
    return features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tap official SkeletonGait++ frame features")
    parser.add_argument("--input", type=Path, required=True, help="[T,3,64,44] uint8 npy")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--opengait-root", type=Path, default=Path("third_party/OpenGait"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=96)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pose_sil = np.load(args.input.resolve(), mmap_mode="r")
    if pose_sil.ndim != 4 or pose_sil.shape[1:] != (3, 64, 44):
        raise ValueError(f"Expected [T,3,64,44], got {pose_sil.shape}")
    device = torch.device(args.device)
    model = build_model(args.opengait_root.resolve(), args.checkpoint.resolve()).to(device).eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, pose_sil.shape[0], args.chunk_size):
            chunk = torch.from_numpy(np.asarray(pose_sil[start : start + args.chunk_size], dtype=np.float32))
            chunk = (chunk / 255.0).unsqueeze(0).to(device)
            outputs.append(extract_chunk(model, chunk).squeeze(0).cpu().numpy())
    features = np.concatenate(outputs, axis=0).astype(np.float32)
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output.resolve(), features)
    print(f"Saved SkeletonGait++ frame features {features.shape} to {args.output.resolve()}")


if __name__ == "__main__":
    main()
