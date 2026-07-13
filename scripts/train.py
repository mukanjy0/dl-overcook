"""Thin local/Kaggle entry point for Stage A training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation.checkpoint_selection import evaluate_training_checkpoints
from src.experiment_config import load_experiment_config
from src.training.trainer import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Stage A training YAML")
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional runtime output-root override (used by Kaggle)",
    )
    parser.add_argument(
        "--evaluate-checkpoints",
        action="store_true",
        help="Evaluate and select every saved checkpoint after training",
    )
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    training_result = train(config, output_root_override=args.output_root)
    if args.evaluate_checkpoints:
        output_root = (
            Path(args.output_root).expanduser().resolve()
            if args.output_root is not None
            else config.outputs.root
        )
        checkpoint_evaluation = evaluate_training_checkpoints(
            config,
            checkpoint_dir=output_root / "checkpoints",
            output_root=output_root / "checkpoint_evaluation",
        )
        result = {
            "status": "complete",
            "training": training_result,
            "checkpoint_evaluation": checkpoint_evaluation,
        }
        (output_root / "experiment_summary.json").write_text(
            json.dumps(result, indent=2),
            encoding="utf-8",
        )
        selected = checkpoint_evaluation["selected"]
        printed_result = {
            "status": "complete",
            "training": training_result,
            "checkpoint_evaluation": {
                "status": checkpoint_evaluation["status"],
                "num_checkpoints": checkpoint_evaluation["num_checkpoints"],
                "report": str(
                    output_root
                    / "checkpoint_evaluation"
                    / "checkpoint_evaluation.json"
                ),
                "selected_training_checkpoint": selected[
                    "selected_training_checkpoint"
                ],
                "selected_inference_artifact": selected[
                    "selected_inference_artifact"
                ],
                "deterministic": selected["evaluation"]["modes"][
                    "deterministic"
                ],
                "stochastic": selected["evaluation"]["modes"]["stochastic"],
            },
        }
    else:
        result = training_result
        printed_result = result
    print(json.dumps(printed_result, indent=2))


if __name__ == "__main__":
    main()
