from __future__ import annotations

import pickle
from pathlib import Path

from src.runner import run_from_config


def test_template_policy_and_demonstration_v2_remain_compatible(
    tmp_path: Path,
    project_root: Path,
) -> None:
    demonstration_path = tmp_path / "demo.pkl"
    result = run_from_config(
        {
            "seed": 5,
            "mode": "demonstration_collection",
            "environment": {
                "layout_name": "cramped_room",
                "horizon": 2,
                "old_dynamics": True,
            },
            "observation": {"type": "featurized", "include_agent_index": True},
            "policies": {
                "agent_0": {
                    "type": "python_class",
                    "path": str(project_root / "policies" / "template.py"),
                    "class_name": "StudentAgent",
                    "config": {"action": "stay"},
                    "max_action_time_ms": 0,
                },
                "agent_1": {
                    "type": "builtin",
                    "name": "stay",
                    "max_action_time_ms": 0,
                },
            },
            "execution": {"num_episodes": 1, "episode_seeds": [5]},
            "rendering": {"mode": "none"},
            "logging": {
                "output_dir": str(tmp_path / "logs"),
                "save_step_log": False,
                "save_episode_summary": False,
            },
            "data_collection": {
                "enabled": True,
                "record_agent_indices": [0, 1],
                "include_next_obs": True,
                "auto_name": False,
                "overwrite": True,
                "output_path": str(demonstration_path),
            },
        }
    )
    assert result["num_rollouts"] == 1
    with demonstration_path.open("rb") as stream:
        payload = pickle.load(stream)
    assert payload["metadata"]["format_version"] == "overcooked_demonstrations_v2"
    assert len(payload["records"]) == 4
