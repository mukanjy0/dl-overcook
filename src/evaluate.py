"""Evaluation runner.

Usage:
    python -m src.evaluate --config configs/evaluate.yaml
"""

from __future__ import annotations

import argparse
import json

from src.evaluation.evaluator import evaluate_from_config
from src.experiment_config import load_runtime_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/evaluate.yaml")
    args = parser.parse_args()

    config = load_runtime_config(args.config)
    result = evaluate_from_config(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
