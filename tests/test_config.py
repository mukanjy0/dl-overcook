from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.config import ConfigError
from src.experiment_config import load_experiment_config, load_runtime_config


def _training_config() -> dict:
    return {
        "experiment": {"name": "test", "seed": 7, "device": "cpu"},
        "environment": {
            "layout_name": "cramped_room",
            "layout_file": None,
            "horizon": 4,
            "old_dynamics": True,
        },
        "observation": {"type": "featurized", "include_agent_index": True},
        "model": {
            "architecture": "mlp_actor_critic",
            "parameters": {"hidden_sizes": [8]},
        },
        "training": {
            "algorithm": "ppo",
            "total_steps": 2,
            "num_environments": 1,
            "rollout_steps": 2,
            "ppo": {},
            "reward_shaping": 0.0,
        },
        "partner": {"sampler": "self_play", "policies": []},
        "evaluation": {},
        "checkpoint": {
            "resume_from": None,
            "save_interval": 0,
            "export_path": "artifacts/inference.pt",
        },
        "outputs": {
            "root": "run",
            "logs": "run/logs",
            "checkpoints": "run/checkpoints",
            "metrics": "run/metrics",
        },
    }


def test_training_paths_are_resolved_relative_to_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "nested" / "train.yaml"
    config_path.parent.mkdir()
    config_path.write_text(yaml.safe_dump(_training_config()), encoding="utf-8")
    config = load_experiment_config(config_path)
    assert config.outputs.root == (config_path.parent / "run").resolve()
    assert config.checkpoint.export_path == (
        config_path.parent / "artifacts" / "inference.pt"
    ).resolve()
    assert config.checkpoint.restore_rng_state is True


def test_invalid_layout_selection_fails_early(tmp_path: Path) -> None:
    raw = _training_config()
    raw["environment"]["layout_file"] = "also.layout"
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="exactly one"):
        load_experiment_config(path)


def test_legacy_runtime_paths_are_unchanged_without_opt_in(tmp_path: Path) -> None:
    path = tmp_path / "runtime.yaml"
    path.write_text(
        yaml.safe_dump({"environment": {"layout_file": "relative.layout"}}),
        encoding="utf-8",
    )
    assert load_runtime_config(path)["environment"]["layout_file"] == "relative.layout"


def test_runtime_relative_path_opt_in_resolves_policy_assets(tmp_path: Path) -> None:
    path = tmp_path / "runtime.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "paths_relative_to_config": True,
                "policies": {
                    "agent_0": {
                        "path": "policy.py",
                        "config": {"checkpoint_path": "model.pt"},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    loaded = load_runtime_config(path)
    assert loaded["policies"]["agent_0"]["path"] == str((tmp_path / "policy.py").resolve())
    assert loaded["policies"]["agent_0"]["config"]["checkpoint_path"] == str(
        (tmp_path / "model.pt").resolve()
    )


def test_agent_index_observation_is_configurable(tmp_path: Path) -> None:
    raw = _training_config()
    raw["observation"]["include_agent_index"] = False
    path = tmp_path / "no_index.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    assert load_experiment_config(path).observation.include_agent_index is False


def test_finetuning_rng_and_reward_schedule_are_configurable(tmp_path: Path) -> None:
    raw = _training_config()
    raw["training"].update(
        {
            "reward_shaping": 1.0,
            "reward_shaping_final": 0.1,
            "reward_shaping_anneal_steps": 50_176,
        }
    )
    raw["checkpoint"]["restore_rng_state"] = False
    path = tmp_path / "fine_tune.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    loaded = load_experiment_config(path)
    assert loaded.training.reward_shaping == 1.0
    assert loaded.training.reward_shaping_final == 0.1
    assert loaded.training.reward_shaping_anneal_steps == 50_176
    assert loaded.checkpoint.restore_rng_state is False


def test_invalid_reward_schedule_fails_early(tmp_path: Path) -> None:
    raw = _training_config()
    raw["training"]["reward_shaping_anneal_steps"] = 0
    path = tmp_path / "invalid_schedule.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ConfigError, match="reward_shaping_anneal_steps"):
        load_experiment_config(path)


def test_stage_c_exact_partner_paths_are_resolved_and_validated(tmp_path: Path) -> None:
    raw = _training_config()
    raw["checkpoint"]["load_optimizer_state"] = False
    raw["partner"] = {
        "sampler": "exact",
        "position_sampler": "balanced",
        "exact_partner": "frozen_sp",
        "policies": [
            {
                "name": "frozen_sp",
                "weight": 1.0,
                "source": "frozen_checkpoint",
                "observation": {
                    "type": "featurized",
                    "include_agent_index": False,
                },
                "policy": {
                    "type": "python_class",
                    "path": "policy.py",
                    "config": {"checkpoint_path": "partner.pt"},
                },
            }
        ],
    }
    path = tmp_path / "stage_c.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    loaded = load_experiment_config(path)
    assert loaded.checkpoint.load_optimizer_state is False
    policy = loaded.partner["policies"][0]["policy"]
    assert policy["path"] == str((tmp_path / "policy.py").resolve())
    assert policy["config"]["checkpoint_path"] == str(
        (tmp_path / "partner.pt").resolve()
    )


def test_stage_c_exact_partner_must_name_a_pool_member(tmp_path: Path) -> None:
    raw = _training_config()
    raw["partner"] = {
        "sampler": "exact",
        "exact_partner": "missing",
        "policies": [
            {
                "name": "stay",
                "policy": {"type": "builtin", "name": "stay"},
            }
        ],
    }
    path = tmp_path / "invalid_stage_c.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ConfigError, match="not present"):
        load_experiment_config(path)
