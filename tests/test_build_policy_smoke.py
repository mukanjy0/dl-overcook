from __future__ import annotations

from pathlib import Path

import yaml

from src.checkpointing import save_inference_artifact
from src.environment import build_env
from src.experiment_config import load_experiment_config
from src.models.actor_critic import ActorCritic, ActorCriticConfig
from src.models.interfaces import ObservationSpec
from src.observations import ObservationBuilder
from src.runner import run_from_config
from src.training.trainer import train


def test_rl_policy_loads_through_build_policy_in_both_positions(
    tmp_path: Path,
    project_root: Path,
) -> None:
    environment_config = {
        "layout_name": "cramped_room",
        "horizon": 5,
        "old_dynamics": True,
    }
    observation_config = {"type": "featurized", "include_agent_index": True}
    env = build_env(environment_config)
    env.reset(regen_mdp=False)
    observation_builder = ObservationBuilder(env, observation_config)
    observation_spec = ObservationSpec.from_observation(
        observation_builder(env.state, 0),
        obs_type="featurized",
    )
    model_config = ActorCriticConfig(hidden_sizes=(8,), activation="tanh")
    model = ActorCritic(observation_spec.encoded_size, 6, model_config)
    artifact = tmp_path / "inference.pt"
    save_inference_artifact(
        artifact,
        model=model,
        model_config=model_config,
        observation_spec=observation_spec,
        environment_metadata=environment_config,
    )

    config = {
        "seed": 13,
        "environment": environment_config,
        "observation": observation_config,
        "policies": {
            "agent_0": {
                "type": "python_class",
                "path": str(project_root / "policies" / "rl_policy.py"),
                "class_name": "StudentAgent",
                "config": {
                    "checkpoint_path": str(artifact),
                    "device": "cpu",
                    "deterministic": True,
                },
                "max_action_time_ms": 0,
            },
            "agent_1": {
                "type": "builtin",
                "name": "stay",
                "max_action_time_ms": 0,
            },
        },
        "execution": {
            "num_episodes": 1,
            "episode_seeds": [13],
            "ego_player_positions": [0, 1],
        },
        "rendering": {"mode": "none"},
        "logging": {
            "output_dir": str(tmp_path / "logs"),
            "save_step_log": False,
            "save_episode_summary": False,
        },
        "data_collection": {"enabled": False},
    }
    result = run_from_config(config)
    assert result["num_rollouts"] == 2
    assert {episode["ego_player_index"] for episode in result["episode_results"]} == {0, 1}
    assert all(episode["timeout_count_total"] == 0 for episode in result["episode_results"])
    assert all(
        episode["invalid_action_replacements_by_agent"] == [0, 0]
        for episode in result["episode_results"]
    )


def _tiny_training_config(root: Path, *, total_steps: int, resume_from=None) -> dict:
    return {
        "experiment": {"name": "vertical", "seed": 31, "device": "cpu"},
        "environment": {
            "layout_name": "cramped_room",
            "layout_file": None,
            "horizon": 3,
            "old_dynamics": True,
            "mdp_overrides": {},
        },
        "observation": {"type": "featurized", "include_agent_index": True},
        "model": {
            "architecture": "mlp_actor_critic",
            "parameters": {"hidden_sizes": [8], "activation": "tanh"},
        },
        "training": {
            "algorithm": "ppo",
            "total_steps": total_steps,
            "num_environments": 1,
            "rollout_steps": 1,
            "reward_shaping": 0.0,
            "ppo": {
                "learning_rate": 0.001,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "update_epochs": 1,
                "minibatch_size": 2,
            },
        },
        "partner": {"sampler": "self_play", "policies": []},
        "evaluation": {},
        "checkpoint": {
            "resume_from": resume_from,
            "save_interval": 0,
            "export_path": str(root / "configured_inference.pt"),
        },
        "outputs": {
            "root": str(root),
            "logs": str(root / "logs"),
            "checkpoints": str(root / "checkpoints"),
            "metrics": str(root / "metrics"),
        },
    }


def test_short_train_resume_export_and_build_policy_vertical(
    tmp_path: Path,
    project_root: Path,
) -> None:
    first_yaml = tmp_path / "first.yaml"
    first_yaml.write_text(
        yaml.safe_dump(_tiny_training_config(tmp_path / "first", total_steps=1)),
        encoding="utf-8",
    )
    first = train(load_experiment_config(first_yaml))
    assert Path(first["training_checkpoint"]).exists()

    second_yaml = tmp_path / "second.yaml"
    second_yaml.write_text(
        yaml.safe_dump(
            _tiny_training_config(
                tmp_path / "second",
                total_steps=2,
                resume_from=first["training_checkpoint"],
            )
        ),
        encoding="utf-8",
    )
    resumed = train(load_experiment_config(second_yaml))
    assert resumed["updates"] == 2
    artifact = Path(resumed["inference_artifact"])
    assert artifact.exists()

    result = run_from_config(
        {
            "environment": {
                "layout_name": "cramped_room",
                "horizon": 3,
                "old_dynamics": True,
            },
            "observation": {"type": "featurized", "include_agent_index": True},
            "policies": {
                "agent_0": {
                    "type": "python_class",
                    "path": str(project_root / "policies" / "rl_policy.py"),
                    "config": {
                        "checkpoint_path": str(artifact),
                        "device": "cpu",
                    },
                    "max_action_time_ms": 0,
                },
                "agent_1": {
                    "type": "builtin",
                    "name": "stay",
                    "max_action_time_ms": 0,
                },
            },
            "execution": {
                "num_episodes": 1,
                "episode_seeds": [31],
                "ego_player_positions": [0, 1],
            },
            "rendering": {"mode": "none"},
            "logging": {
                "output_dir": str(tmp_path / "vertical_evaluation"),
                "save_step_log": False,
                "save_episode_summary": False,
            },
            "data_collection": {"enabled": False},
        }
    )
    assert result["num_rollouts"] == 2
    assert all(episode["timeout_count_total"] == 0 for episode in result["episode_results"])
    assert all(
        episode["invalid_action_replacements_by_agent"] == [0, 0]
        for episode in result["episode_results"]
    )
