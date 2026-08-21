from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import cv2
import numpy as np


def load_heatmap_factory(opengait_root: Path):
    source = opengait_root / "datasets" / "pretreatment_heatmap.py"
    spec = importlib.util.spec_from_file_location("opengait_heatmap", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load official OpenGait heatmap code: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GenerateHeatmapTransform


def load_silhouettes(directory: Path, frames: int) -> np.ndarray:
    paths = sorted(directory.glob("*.png"))
    if len(paths) != frames:
        raise ValueError(f"Expected {frames} silhouettes in {directory}, got {len(paths)}")
    result = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"Cannot read silhouette: {path}")
        result.append(cv2.resize(image, (64, 64), interpolation=cv2.INTER_NEAREST))
    return np.asarray(result, dtype=np.uint8)[:, None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build official SkeletonGait++ pose/silhouette tensor")
    parser.add_argument("--keypoints", type=Path, required=True, help="COCO-17 [T,17,3] kpt2d.npy")
    parser.add_argument("--silhouette-dir", type=Path, required=True)
    parser.add_argument("--opengait-root", type=Path, default=Path("third_party/OpenGait"))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    keypoints = np.load(args.keypoints.resolve()).astype(np.float32)
    if keypoints.ndim != 3 or keypoints.shape[1:] != (17, 3):
        raise ValueError(f"Expected COCO-17 keypoints [T,17,3], got {keypoints.shape}")
    factory = load_heatmap_factory(args.opengait_root.resolve())
    transform = factory(
        {"transfer_to_coco17": False},
        {"pad_method": "knn", "use_conf": True},
        {"pose_format": "coco", "use_conf": True, "heatmap_image_height": 128},
        {"sigma": 8.0, "use_score": True, "img_h": 128, "img_w": 128, "with_limb": None, "with_kp": None},
        {"align": True, "final_img_size": 64, "offset": 0, "heatmap_image_size": 128},
    )
    heatmaps = transform(keypoints)
    silhouettes = load_silhouettes(args.silhouette_dir.resolve(), keypoints.shape[0])
    pose_sil = np.concatenate([heatmaps, silhouettes], axis=1)
    pose_sil = pose_sil[..., 10:-10]
    if pose_sil.shape[1:] != (3, 64, 44):
        raise RuntimeError(f"Unexpected SkeletonGait++ tensor shape: {pose_sil.shape}")
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output.resolve(), pose_sil)
    print(f"Saved SkeletonGait++ input {pose_sil.shape} to {args.output.resolve()}")


if __name__ == "__main__":
    main()

