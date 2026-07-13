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


def test_counter_circuit_stage_c_pilot_uses_the_disclosed_partner(
    project_root: Path,
) -> None:
    config = load_experiment_config(
        project_root / "configs/stage_c/counter_circuit_exact_partner_seed2_150k.yaml"
    )

    assert config.training.total_steps - 900096 == 150528
    assert config.experiment.device == "cpu"
    assert config.checkpoint.load_optimizer_state is False
    assert config.checkpoint.restore_rng_state is False
    assert config.partner["sampler"] == "exact"
    assert config.partner["position_sampler"] == "balanced"
    policy = config.partner["policies"][0]["policy"]
    assert policy["sticky_action_prob"] == 0.10
    assert policy["random_action_prob"] == 0.10
    assert config.training.ppo["entropy_coefficient"] == 0.01
    assert set(config.evaluation["player_positions"]) == {0, 1}
    assert set(config.evaluation["inference_modes"]) == {
        "deterministic",
        "stochastic",
    }


def test_counter_circuit_stage_c_extension_preserves_the_pilot_stream(
    project_root: Path,
) -> None:
    pilot = load_experiment_config(
        project_root / "configs/stage_c/counter_circuit_exact_partner_seed2_150k.yaml"
    )
    extension = load_experiment_config(
        project_root / "configs/stage_c/counter_circuit_exact_partner_seed2_300k.yaml"
    )

    assert extension.training.total_steps - 900096 == 300032
    assert extension.training.total_steps - pilot.training.total_steps == 149504
    assert extension.training.ppo == pilot.training.ppo
    assert extension.partner == pilot.partner
    assert extension.checkpoint.load_optimizer_state is True
    assert extension.checkpoint.restore_rng_state is True


def test_counter_circuit_stage_b_matrix_changes_only_reset_distribution(
    project_root: Path,
) -> None:
    control = load_experiment_config(
        project_root / "configs/stage_b/counter_circuit_exact_standard_seed2_200k.yaml"
    )
    mixed = load_experiment_config(
        project_root / "configs/stage_b/counter_circuit_exact_mixed050_seed2_200k.yaml"
    )

    assert control.training.total_steps - 1200128 == 200704
    assert control.experiment.seed == mixed.experiment.seed == 2
    assert control.environment == mixed.environment
    assert control.model == mixed.model
    assert control.training == mixed.training
    assert control.partner == mixed.partner
    assert control.evaluation == mixed.evaluation
    assert control.state_augmentation.reset_mode == "standard"
    assert mixed.state_augmentation.reset_mode == "mixed"
    assert mixed.state_augmentation.augmented_probability == 0.5
    assert mixed.state_augmentation.buffer_path is not None
    for config in (control, mixed):
        assert config.checkpoint.load_optimizer_state is False
        assert config.checkpoint.restore_rng_state is False
        assert config.experiment.device == "cpu"


def test_counter_circuit_long_stage_c_branches_change_only_seed(
    project_root: Path,
) -> None:
    branches = [
        load_experiment_config(
            project_root / f"configs/stage_c/counter_circuit_exact_long_seed{seed}_1m.yaml"
        )
        for seed in (2, 3)
    ]
    reference = branches[0]

    for branch in branches:
        assert branch.training.total_steps - 1400832 == 1000448
        assert branch.training.total_steps == 2401280
        assert branch.training == reference.training
        assert branch.environment == reference.environment
        assert branch.model == reference.model
        assert branch.partner == reference.partner
        assert branch.evaluation == reference.evaluation
        assert branch.checkpoint.load_optimizer_state is False
        assert branch.checkpoint.restore_rng_state is False
        assert branch.checkpoint.save_interval == 100352
        assert branch.experiment.device == "cpu"

    assert {branch.experiment.seed for branch in branches} == {2, 3}
