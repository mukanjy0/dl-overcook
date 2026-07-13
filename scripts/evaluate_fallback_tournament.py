"""Evaluate one generic fallback profile across layouts, partners, and positions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation.evaluator import evaluate_from_config
from src.experiment_config import load_runtime_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--profile", default="balanced")
    parser.add_argument("--policy-name", default="generic_task_planner")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    config = load_runtime_config(args.config)
    output_dir = Path(args.output_dir).expanduser().resolve()
    config["policies"]["agent_0"] = {
        "type": "builtin",
        "name": args.policy_name,
        "profile": args.profile,
        "max_action_time_ms": 100,
        "invalid_action": "stay",
        "timeout_action": "stay",
    }
    config["logging"] = {
        **(config.get("logging", {}) or {}),
        "output_dir": str(output_dir),
    }
    report = evaluate_from_config(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tournament.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "policy_name": args.policy_name,
                "profile": args.profile,
                "report": report,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "profile": args.profile,
                "mean_official_score": report["mean_official_score"],
                "num_rollouts": report["num_rollouts"],
            }
        )
    )


if __name__ == "__main__":
    main()
