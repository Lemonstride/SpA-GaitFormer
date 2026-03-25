from __future__ import annotations

import argparse
from pathlib import Path

from spa_gaitformer.config import ExperimentConfig
from spa_gaitformer.train import evaluate, train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SpA-MMD dual-modal Transformer baseline"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to an experiment JSON config.",
    )
    parser.add_argument(
        "--mode",
        choices=("train", "evaluate"),
        default="train",
        help="Run training or evaluation.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint used for evaluation or resumed training.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="val",
        help="Dataset split for evaluation mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig.from_json(args.config)

    if args.mode == "train":
        train(config, resume_checkpoint=args.checkpoint)
        return

    metrics = evaluate(config, split=args.split, checkpoint_path=args.checkpoint)
    print(f"Evaluation on {args.split}:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
