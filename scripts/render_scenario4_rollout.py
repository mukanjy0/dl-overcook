"""Render one teacher-compatible Scenario 4 router rollout to GIF."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.runner import run_from_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--position", type=int, choices=(0, 1), required=True)
    parser.add_argument("--gif", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = run_from_config(
        {
            "seed": args.seed,
            "environment": {
                "layout_name": None,
                "layout_file": str(root / "configs/layouts/scenario_4.layout"),
                "horizon": 400,
                "old_dynamics": True,
            },
            "observation": {"type": "featurized", "include_agent_index": True},
            "policies": {
                "agent_0": {
                    "type": "stage_d_router",
                    "specialist_mapping": str(root / "configs/stage_d/specialists.yaml"),
                    "max_action_time_ms": 100,
                    "invalid_action": "stay",
                    "timeout_action": "stay",
                },
                "agent_1": {"type": "builtin", "name": "random_motion"},
            },
            "execution": {
                "num_episodes": 1,
                "episode_seeds": [args.seed],
                "ego_player_positions": [args.position],
            },
            "rendering": {"mode": "gif", "gif_path": args.gif, "fps": 15},
            "logging": {"output_dir": str(Path(args.gif).with_suffix("")), "save_step_log": False},
            "data_collection": {"enabled": False},
        }
    )
    print(result["episode_results"][0])


if __name__ == "__main__":
    main()
