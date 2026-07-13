"""Run one short teacher-compatible rollout for every known Stage D route."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.runner import run_from_config


SCENARIOS = (
    ("asymmetric_advantages", 67, {"name": "greedy_full_task"}),
    (
        "coordination_ring",
        0,
        {"name": "greedy_full_task", "sticky_action_prob": 0.10},
    ),
    (
        "counter_circuit",
        0,
        {
            "name": "greedy_full_task",
            "sticky_action_prob": 0.10,
            "random_action_prob": 0.10,
        },
    ),
)


def _policy(config: dict) -> dict:
    return {
        "type": "builtin",
        "max_action_time_ms": 100,
        "invalid_action": "stay",
        "timeout_action": "stay",
        **config,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mapping",
        default="configs/stage_d/specialists.yaml",
        help="Central Stage D specialist mapping",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/stage_d_smoke",
        help="Fresh directory for machine-readable smoke results",
    )
    parser.add_argument("--horizon", type=int, default=20)
    args = parser.parse_args()

    mapping = Path(args.mapping).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for layout_name, seed, partner_config in SCENARIOS:
        for ego_position in (0, 1):
            result = run_from_config(
                {
                    "seed": seed,
                    "environment": {
                        "layout_name": layout_name,
                        "horizon": args.horizon,
                        "old_dynamics": True,
                    },
                    "observation": {
                        "type": "featurized",
                        "include_agent_index": True,
                    },
                    "policies": {
                        "agent_0": {
                            "type": "stage_d_router",
                            "specialist_mapping": str(mapping),
                            "max_action_time_ms": 100,
                            "invalid_action": "stay",
                            "timeout_action": "stay",
                        },
                        "agent_1": _policy(partner_config),
                    },
                    "execution": {
                        "num_episodes": 1,
                        "episode_seeds": [seed],
                        "ego_player_positions": [ego_position],
                    },
                    "rendering": {"mode": "none"},
                    "logging": {
                        "output_dir": str(output_dir / layout_name / str(ego_position)),
                        "save_step_log": False,
                        "save_episode_summary": True,
                    },
                    "data_collection": {"enabled": False},
                }
            )
            episode = result["episode_results"][0]
            results.append(
                {
                    "layout": layout_name,
                    "ego_position": ego_position,
                    "official_score": episode["official_score"],
                    "timeouts": episode["timeout_count_total"],
                    "invalid_actions": episode["invalid_action_replacements_by_agent"],
                }
            )

    payload = {"status": "complete", "horizon": args.horizon, "routes": results}
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
