from __future__ import annotations

import argparse
from pathlib import Path

from spa_gaitformer.config import ExperimentConfig
from spa_gaitformer.train import cross_validate, evaluate, train_final


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SpA-MMD dual-modal Transformer training pipeline"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to an experiment JSON config.",
    )
    parser.add_argument(
        "--mode",
        choices=("cross_validate", "train_final", "evaluate"),
        default="cross_validate",
        help="Run cross-validation, final training, or evaluation.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint used for evaluation.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="test",
        help="Dataset split for evaluation mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig.from_json(args.config)

    if args.mode == "cross_validate":
        summary = cross_validate(config)
        print("Cross-validation summary:")
        print(f"  recommended_final_epoch: {summary['recommended_final_epoch']}")
        return

    if args.mode == "train_final":
        summary = train_final(config)
        print("Final training summary:")
        print(f"  epochs: {summary['epochs']}")
        return

    metrics = evaluate(config, split=args.split, checkpoint_path=args.checkpoint)
    print(f"Evaluation on {args.split}:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
