from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import torch

from src.checkpointing import save_inference_artifact
from src.environment import build_env
from src.models.actor_critic import ActorCritic, ActorCriticConfig
from src.models.interfaces import ObservationSpec
from src.observations import ObservationBuilder
from src.partners.interfaces import ConfiguredPartnerFactory, PartnerSpec
from src.partners.samplers import (
    BalancedEgoPositionSampler,
    ExactPartnerSampler,
    WeightedPartner,
    WeightedPartnerSampler,
)
from src.training.ppo import PPOConfig, PPOUpdater
from src.training.rollouts import FrozenPartnerRolloutCollector


def test_weighted_and_exact_partner_sampling_are_seeded() -> None:
    partners = (
        WeightedPartner(PartnerSpec("greedy", {"type": "builtin", "name": "stay"}), 3.0),
        WeightedPartner(PartnerSpec("random", {"type": "builtin", "name": "stay"}), 1.0),
    )
    first_sampler = WeightedPartnerSampler(partners)
    second_sampler = WeightedPartnerSampler(partners)
    first_rng = np.random.default_rng(17)
    second_rng = np.random.default_rng(17)
    first = [first_sampler.sample(first_rng, {}).name for _ in range(200)]
    second = [second_sampler.sample(second_rng, {}).name for _ in range(200)]

    assert first == second
    counts = Counter(first)
    assert counts["greedy"] > counts["random"]
    exact = ExactPartnerSampler(partners[1].spec)
    assert {exact.sample(first_rng, {}).name for _ in range(20)} == {"random"}


def test_balanced_position_sampling_alternates_with_seeded_start() -> None:
    first = BalancedEgoPositionSampler()
    second = BalancedEgoPositionSampler()
    first_rng = np.random.default_rng(23)
    second_rng = np.random.default_rng(23)
    first_positions = [first.sample(first_rng, {}) for _ in range(7)]
    second_positions = [second.sample(second_rng, {}) for _ in range(7)]

    assert first_positions == second_positions
    assert all(left != right for left, right in zip(first_positions, first_positions[1:]))
    assert abs(first_positions.count(0) - first_positions.count(1)) == 1


def test_frozen_checkpoint_partner_can_use_its_own_observation_contract(
    tmp_path: Path,
    project_root: Path,
) -> None:
    env = build_env({"layout_name": "cramped_room", "horizon": 2})
    env.reset(regen_mdp=False)
    ego_builder = ObservationBuilder(
        env,
        {"type": "featurized", "include_agent_index": True},
    )
    partner_builder = ObservationBuilder(
        env,
        {"type": "featurized", "include_agent_index": False},
    )
    partner_spec = ObservationSpec.from_observation(
        partner_builder(env.state, 1),
        obs_type="featurized",
    )
    model_config = ActorCriticConfig(hidden_sizes=(8,), activation="tanh")
    model = ActorCritic(partner_spec.encoded_size, 6, model_config)
    artifact = tmp_path / "partner_inference.pt"
    save_inference_artifact(
        artifact,
        model=model,
        model_config=model_config,
        observation_spec=partner_spec,
        environment_metadata={"layout_name": "cramped_room"},
    )
    spec = PartnerSpec(
        name="frozen_sp",
        source="frozen_checkpoint",
        observation_config={
            "type": "featurized",
            "include_agent_index": False,
        },
        policy_config={
            "type": "python_class",
            "name": "frozen_sp",
            "path": str(project_root / "policies" / "rl_policy.py"),
            "class_name": "StudentAgent",
            "config": {
                "checkpoint_path": str(artifact),
                "device": "cpu",
                "deterministic": True,
            },
            "max_action_time_ms": 0,
        },
    )
    partner = ConfiguredPartnerFactory().build(
        spec,
        env=env,
        observation_builder=ego_builder,
        player_position=1,
        seed=29,
    )
    partner.reset()
    partner.set_agent_index(1)
    partner.set_mdp(env.mdp)

    action, info = partner.action(env.state)
    assert action is not None
    assert info["invalid_action_replaced"] is False, info


def test_frozen_partner_rollout_trains_only_ego_and_balances_positions() -> None:
    environment_config = {
        "layout_name": "cramped_room",
        "horizon": 2,
        "old_dynamics": True,
    }
    observation_config = {"type": "featurized", "include_agent_index": True}
    probe_env = build_env(environment_config)
    probe_env.reset(regen_mdp=False)
    probe_builder = ObservationBuilder(probe_env, observation_config)
    observation_spec = ObservationSpec.from_observation(
        probe_builder(probe_env.state, 0),
        obs_type="featurized",
    )
    model_config = ActorCriticConfig(hidden_sizes=(8,), activation="tanh")
    model = ActorCritic(observation_spec.encoded_size, 6, model_config)
    collector = FrozenPartnerRolloutCollector(
        environment_config=environment_config,
        observation_config=observation_config,
        observation_spec=observation_spec,
        num_environments=2,
        base_seed=31,
        device=torch.device("cpu"),
        reward_shaping=0.0,
        partner_sampler=ExactPartnerSampler(
            PartnerSpec(
                name="stay",
                policy_config={
                    "type": "builtin",
                    "name": "stay",
                    "max_action_time_ms": 0,
                },
            )
        ),
        position_sampler=BalancedEgoPositionSampler(),
    )
    rollout = collector.collect(model, rollout_steps=2, gamma=0.99, gae_lambda=0.95)

    assert rollout.observations.shape == (2, 2, observation_spec.encoded_size)
    assert rollout.flattened()["actions"].shape == (4,)
    assert rollout.partner_step_counts == {"stay": 4}
    assert rollout.ego_position_step_counts == {"0": 2, "1": 2}
    assert len(rollout.completed_episode_results) == 2
    assert {item["ego_player_index"] for item in rollout.completed_episode_metadata} == {0, 1}
    assert all(result.timeout_count_total == 0 for result in rollout.completed_episode_results)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    metrics = PPOUpdater(
        model,
        optimizer,
        PPOConfig(update_epochs=1, minibatch_size=4),
    ).update(rollout)
    assert set(metrics) >= {"policy_loss", "value_loss", "entropy"}
