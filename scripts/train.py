"""Thin local/Kaggle entry point for Stage A training."""

from __future__ import annotations

import argparse
import json

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
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    result = train(config, output_root_override=args.output_root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
