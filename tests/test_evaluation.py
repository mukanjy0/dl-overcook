from __future__ import annotations

from pathlib import Path

from src.environment import build_env
from src.evaluation.evaluator import evaluate_from_config
from src.observations import ObservationBuilder
from src.policy_loader import build_policy


def _runtime_config(output_dir: Path) -> dict:
    return {
        "seed": 23,
        "environment": {
            "layout_name": "cramped_room",
            "horizon": 5,
            "old_dynamics": True,
        },
        "observation": {"type": "featurized", "include_agent_index": True},
        "policies": {
            "agent_0": {"type": "builtin", "name": "stay", "max_action_time_ms": 0},
            "agent_1": {
                "type": "builtin",
                "name": "random_motion",
                "max_action_time_ms": 0,
            },
        },
        "execution": {"num_episodes": 2, "episode_seeds": [23, 24]},
        "evaluation": {
            "layouts": ["cramped_room"],
            "partners": [
                {
                    "name": "random_motion",
                    "policy": {
                        "type": "builtin",
                        "name": "random_motion",
                        "max_action_time_ms": 0,
                    },
                }
            ],
            "seeds": [23, 24],
            "player_positions": [0, 1],
        },
        "rendering": {"mode": "none"},
        "logging": {
            "output_dir": str(output_dir),
            "save_step_log": False,
            "save_episode_summary": False,
        },
        "data_collection": {"enabled": False},
    }


def test_fixed_seed_suite_is_deterministic_and_covers_role_swaps(tmp_path: Path) -> None:
    first = evaluate_from_config(_runtime_config(tmp_path / "first"))
    second = evaluate_from_config(_runtime_config(tmp_path / "second"))
    assert first["num_rollouts"] == 4
    assert first["mean_official_score"] == second["mean_official_score"]
    first_episodes = first["cases"][0]["aggregate"]["episode_results"]
    second_episodes = second["cases"][0]["aggregate"]["episode_results"]
    assert first_episodes == second_episodes
    assert {episode["ego_player_index"] for episode in first_episodes} == {0, 1}


def test_builtin_random_policy_receives_effective_seed() -> None:
    env = build_env({"layout_name": "cramped_room", "horizon": 2})
    env.reset(regen_mdp=False)
    builder = ObservationBuilder(env, {"type": "state"})
    config = {"type": "builtin", "name": "random_motion", "max_action_time_ms": 0}
    first = build_policy(config, env, builder, seed=99)
    second = build_policy(config, env, builder, seed=99)
    for agent in (first, second):
        agent.set_agent_index(0)
        agent.set_mdp(env.mdp)
    first_actions = [first.action(env.state)[0] for _ in range(5)]
    second_actions = [second.action(env.state)[0] for _ in range(5)]
    assert first_actions == second_actions
