from __future__ import annotations

from pathlib import Path

from src.experiment_config import load_experiment_config


def test_counter_circuit_continuations_have_exact_relative_budget(
    project_root: Path,
) -> None:
    control = load_experiment_config(
        project_root / "configs/stage_a/counter_circuit_control_continuation_50k.yaml"
    )
    consolidation = load_experiment_config(
        project_root
        / "configs/stage_a/counter_circuit_consolidation_continuation_50k.yaml"
    )

    assert control.training.total_steps - 900096 == 50176
    assert consolidation.training.total_steps - 900096 == 50176
    assert control.checkpoint.restore_rng_state is True
    assert control.checkpoint.load_optimizer_state is True
    assert consolidation.checkpoint.restore_rng_state is False
    assert consolidation.checkpoint.load_optimizer_state is False
    assert consolidation.training.reward_shaping_final == 0.1
    assert consolidation.training.reward_shaping_anneal_steps == 50176
    assert consolidation.training.ppo["entropy_anneal_steps"] == 50176
    assert len(control.evaluation["seeds"]) == 20


def test_coordination_ring_stage_b_matrix_changes_only_seed_and_reset_distribution(
    project_root: Path,
) -> None:
    configs = []
    for variant in ("standard", "mixed025", "mixed050", "augmented"):
        for seed in range(3):
            configs.append(
                load_experiment_config(
                    project_root
                    / "configs/stage_b"
                    / f"coordination_ring_{variant}_seed{seed}_200k.yaml"
                )
            )

    reference = configs[0]
    for config in configs:
        assert config.training.total_steps == 200704
        assert config.training.num_environments == reference.training.num_environments
        assert config.training.rollout_steps == reference.training.rollout_steps
        assert config.training.reward_shaping == reference.training.reward_shaping
        assert config.training.ppo == reference.training.ppo
        assert config.model == reference.model
        assert config.environment == reference.environment
        assert config.experiment.device == "cpu"
        assert config.experiment.seed in {0, 1, 2}

    reset_modes = [config.state_augmentation.reset_mode for config in configs]
    assert reset_modes.count("standard") == 3
    assert reset_modes.count("mixed") == 6
    assert reset_modes.count("augmented") == 3
    mixed_probabilities = sorted(
        {
            config.state_augmentation.augmented_probability
            for config in configs
            if config.state_augmentation.reset_mode == "mixed"
        }
    )
    assert mixed_probabilities == [0.25, 0.5]


def test_asymmetric_stage_c_pair_uses_fresh_streams_and_exact_budget(
    project_root: Path,
) -> None:
    exact = load_experiment_config(
        project_root / "configs/stage_c/asymmetric_exact_seed67_300k.yaml"
    )
    pool = load_experiment_config(
        project_root / "configs/stage_c/asymmetric_weighted_pool_seed68_300k.yaml"
    )

    for config in (exact, pool):
        assert config.training.total_steps - 900096 == 300032
        assert config.checkpoint.load_optimizer_state is False
        assert config.checkpoint.restore_rng_state is False
        assert config.checkpoint.save_interval == 50176
        assert config.experiment.device == "cpu"
        assert set(config.evaluation["player_positions"]) == {0, 1}
        assert set(config.evaluation["inference_modes"]) == {
            "deterministic",
            "stochastic",
        }
        assert len(config.evaluation["partners"]) == 5

    assert exact.partner["sampler"] == "exact"
    assert pool.partner["sampler"] == "weighted_pool"
