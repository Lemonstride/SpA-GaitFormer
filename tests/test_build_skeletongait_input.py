from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_skeletongait_input.py"
SPEC = importlib.util.spec_from_file_location("build_skeletongait_input", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_mediapipe_33_maps_to_coco_17_order() -> None:
    source = np.zeros((2, 33, 3), dtype=np.float32)
    source[:, :, 0] = np.arange(33, dtype=np.float32)

    result = MODULE.to_coco17(source, "mediapipe33")

    assert result.shape == (2, 17, 3)
    np.testing.assert_array_equal(result[0, :, 0], MODULE.MEDIAPIPE_33_TO_COCO_17)


def test_auto_keeps_coco_17_input() -> None:
    source = np.arange(3 * 17 * 3, dtype=np.float32).reshape(3, 17, 3)
    result = MODULE.to_coco17(source, "auto")
    np.testing.assert_array_equal(result, source)


def test_rejects_unknown_joint_layout() -> None:
    with pytest.raises(ValueError, match="expected COCO-17 or MediaPipe-33"):
        MODULE.to_coco17(np.zeros((1, 25, 3), dtype=np.float32), "auto")
