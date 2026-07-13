"""Sequential vector rollout collection for shared-policy self-play."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from src.constants import action_index_to_overcooked_action, overcooked_action_to_index
from src.environment import build_env
from src.episode import EpisodeResult, EpisodeStep, build_episode_result
from src.models.actor_critic import ActorCritic
from src.models.interfaces import ObservationSpec
from src.observations import ObservationBuilder
from src.partners.interfaces import ConfiguredPartnerFactory, PartnerFactory, PartnerSpec
from src.partners.samplers import (
    EgoPositionSampler,
    PartnerSampler,
    SelfPlayPartnerSampler,
)
from src.seed_utils import derive_seed
from src.state_initialization import StandardStateSource, StateSource


@dataclass(frozen=True)
class RolloutBatch:
    """PPO-ready transitions from both self-play agents or one frozen-partner ego."""

    observations: torch.Tensor
    actions: torch.Tensor
    old_log_probabilities: torch.Tensor
    old_values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    completed_episode_results: tuple[EpisodeResult, ...]
    completed_episode_metadata: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    partner_step_counts: dict[str, int] = field(default_factory=dict)
    ego_position_step_counts: dict[str, int] = field(default_factory=dict)

    @property
    def completed_sparse_returns(self) -> tuple[float, ...]:
        return tuple(result.sparse_return for result in self.completed_episode_results)

    @property
    def num_environment_steps(self) -> int:
        return int(self.observations.shape[0] * self.observations.shape[1])

    def flattened(self) -> dict[str, torch.Tensor]:
        feature_size = self.observations.shape[-1]
        return {
            "observations": self.observations.reshape(-1, feature_size),
            "actions": self.actions.reshape(-1),
            "old_log_probabilities": self.old_log_probabilities.reshape(-1),
            "old_values": self.old_values.reshape(-1),
            "advantages": self.advantages.reshape(-1),
            "returns": self.returns.reshape(-1),
        }


def _generalized_advantages(
    *,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate GAE for either self-play or ego-only rollout tensors."""
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros_like(next_values)
    for rollout_index in reversed(range(rewards.shape[0])):
        next_non_terminal = 1.0 - dones[rollout_index]
        following_values = (
            next_values if rollout_index == rewards.shape[0] - 1 else values[rollout_index + 1]
        )
        delta = (
            rewards[rollout_index]
            + gamma * following_values * next_non_terminal
            - values[rollout_index]
        )
        last_advantage = (
            delta + gamma * gae_lambda * next_non_terminal * last_advantage
        )
        advantages[rollout_index] = last_advantage
    return advantages, advantages + values


class SelfPlayRolloutCollector:
    """Collect rollouts where the current model controls both player positions."""

    def __init__(
        self,
        *,
        environment_config: dict[str, Any],
        observation_config: dict[str, Any],
        observation_spec: ObservationSpec,
        num_environments: int,
        base_seed: int,
        device: torch.device,
        reward_shaping: float,
        state_source: StateSource | None = None,
    ):
        if num_environments <= 0:
            raise ValueError("num_environments must be positive")
        self.observation_spec = observation_spec
        self.device = device
        self.reward_shaping = float(reward_shaping)
        self.environments = []
        self.observation_builders = []
        self.episode_sparse_returns = [0.0 for _ in range(num_environments)]
        self.episode_shaped_returns = [0.0 for _ in range(num_environments)]
        self.environment_seeds: list[int] = []
        self.environment_episode_ids = [0 for _ in range(num_environments)]
        self.episode_start_timesteps = [0 for _ in range(num_environments)]
        source = StandardStateSource() if state_source is None else state_source
        self.partner_sampler = SelfPlayPartnerSampler()

        for env_index in range(num_environments):
            env_seed = derive_seed(base_seed, "environment", env_index)
            self.environment_seeds.append(env_seed)
            rng = np.random.default_rng(env_seed)
            env = build_env(
                environment_config,
                state_source=source,
                rng=rng,
            )
            env.reset(regen_mdp=False)
            self.episode_start_timesteps[env_index] = int(env.state.timestep)
            self.environments.append(env)
            self.observation_builders.append(
                ObservationBuilder(env, observation_config)
            )

        # Sampling remains explicit even though Stage A has one possible partner.
        sampled = self.partner_sampler.sample(
            np.random.default_rng(derive_seed(base_seed, "partner")),
            {"phase": "training"},
        )
        if sampled.source != "current_policy":
            raise ValueError("Stage A rollout collector requires current-policy self-play")

    def _encoded_observations(self) -> np.ndarray:
        encoded = []
        for env, builder in zip(self.environments, self.observation_builders):
            encoded.append(
                [
                    self.observation_spec.encode(builder(env.state, agent_index))
                    for agent_index in (0, 1)
                ]
            )
        return np.asarray(encoded, dtype=np.float32)

    def collect(
        self,
        model: ActorCritic,
        *,
        rollout_steps: int,
        gamma: float,
        gae_lambda: float,
    ) -> RolloutBatch:
        """Collect one fixed-length rollout and calculate generalized advantages."""
        if rollout_steps <= 0:
            raise ValueError("rollout_steps must be positive")
        num_environments = len(self.environments)
        observations = torch.empty(
            (rollout_steps, num_environments, 2, self.observation_spec.encoded_size),
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.empty(
            (rollout_steps, num_environments, 2),
            dtype=torch.long,
            device=self.device,
        )
        log_probabilities = torch.empty_like(actions, dtype=torch.float32)
        values = torch.empty_like(actions, dtype=torch.float32)
        rewards = torch.empty_like(actions, dtype=torch.float32)
        dones = torch.empty_like(actions, dtype=torch.float32)
        completed_results: list[EpisodeResult] = []

        model.eval()
        for rollout_index in range(rollout_steps):
            encoded = self._encoded_observations()
            obs_tensor = torch.as_tensor(encoded, dtype=torch.float32, device=self.device)
            flat_observations = obs_tensor.reshape(-1, self.observation_spec.encoded_size)
            with torch.no_grad():
                policy_step = model.act_batch(flat_observations, deterministic=False)

            observations[rollout_index] = obs_tensor
            actions[rollout_index] = policy_step.actions.reshape(num_environments, 2)
            log_probabilities[rollout_index] = policy_step.log_probabilities.reshape(
                num_environments, 2
            )
            values[rollout_index] = policy_step.values.reshape(num_environments, 2)

            for env_index, env in enumerate(self.environments):
                state = env.state
                action_indices = actions[rollout_index, env_index].detach().cpu().tolist()
                joint_action = tuple(
                    action_index_to_overcooked_action(int(action_index))
                    for action_index in action_indices
                )
                next_state, sparse_reward, done, info = env.step(joint_action)
                transition = EpisodeStep(
                    episode_id=self.environment_episode_ids[env_index],
                    timestep=int(state.timestep),
                    state=state,
                    next_state=next_state,
                    joint_action=joint_action,
                    joint_action_indices=(int(action_indices[0]), int(action_indices[1])),
                    joint_infos=(
                        {"policy_name": "trainable_self_play"},
                        {"policy_name": "trainable_self_play"},
                    ),
                    reward=float(sparse_reward),
                    done=bool(done),
                    info=info,
                )
                shaped = transition.info.get("shaped_r_by_agent", (0.0, 0.0))
                per_agent_rewards = [
                    transition.reward + self.reward_shaping * float(shaped[index])
                    for index in (0, 1)
                ]
                rewards[rollout_index, env_index] = torch.as_tensor(
                    per_agent_rewards,
                    dtype=torch.float32,
                    device=self.device,
                )
                dones[rollout_index, env_index] = float(done)
                self.episode_sparse_returns[env_index] += transition.reward
                self.episode_shaped_returns[env_index] += float(sum(shaped))
                if done:
                    completed_results.append(
                        build_episode_result(
                            env=env,
                            episode_id=self.environment_episode_ids[env_index],
                            seed=derive_seed(
                                self.environment_seeds[env_index],
                                "episode",
                                self.environment_episode_ids[env_index],
                            ),
                            ego_player_index=0,
                            role_swap=False,
                            sparse_return=self.episode_sparse_returns[env_index],
                            shaped_return=self.episode_shaped_returns[env_index],
                            episode_start_timestep=self.episode_start_timesteps[
                                env_index
                            ],
                        )
                    )
                    self.episode_sparse_returns[env_index] = 0.0
                    self.episode_shaped_returns[env_index] = 0.0
                    self.environment_episode_ids[env_index] += 1
                    env.reset(regen_mdp=False)
                    self.episode_start_timesteps[env_index] = int(
                        env.state.timestep
                    )

        next_encoded = torch.as_tensor(
            self._encoded_observations(),
            dtype=torch.float32,
            device=self.device,
        ).reshape(-1, self.observation_spec.encoded_size)
        with torch.no_grad():
            _, next_values = model(next_encoded)
        next_values = next_values.reshape(num_environments, 2)

        advantages, returns = _generalized_advantages(
            rewards=rewards,
            dones=dones,
            values=values,
            next_values=next_values,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

        return RolloutBatch(
            observations=observations,
            actions=actions,
            old_log_probabilities=log_probabilities,
            old_values=values,
            advantages=advantages,
            returns=returns,
            rewards=rewards,
            dones=dones,
            completed_episode_results=tuple(completed_results),
            partner_step_counts={
                "self_play": rollout_steps * num_environments,
            },
            ego_position_step_counts={
                "0": rollout_steps * num_environments,
                "1": rollout_steps * num_environments,
            },
        )


@dataclass
class _FrozenPartnerSession:
    spec: PartnerSpec
    agent: Any
    ego_player_index: int
    start_timestep: int


class FrozenPartnerRolloutCollector:
    """Collect ego-only PPO transitions against episode-frozen partners."""

    def __init__(
        self,
        *,
        environment_config: dict[str, Any],
        observation_config: dict[str, Any],
        observation_spec: ObservationSpec,
        num_environments: int,
        base_seed: int,
        device: torch.device,
        reward_shaping: float,
        partner_sampler: PartnerSampler,
        position_sampler: EgoPositionSampler,
        partner_factory: PartnerFactory | None = None,
        state_source: StateSource | None = None,
    ):
        if num_environments <= 0:
            raise ValueError("num_environments must be positive")
        self.observation_spec = observation_spec
        self.device = device
        self.reward_shaping = float(reward_shaping)
        self.partner_sampler = partner_sampler
        self.position_sampler = position_sampler
        self.partner_factory = (
            ConfiguredPartnerFactory() if partner_factory is None else partner_factory
        )
        self.partner_rng = np.random.default_rng(derive_seed(base_seed, "partner"))
        self.position_rng = np.random.default_rng(derive_seed(base_seed, "ego_position"))
        self.environments = []
        self.observation_builders = []
        self.environment_seeds: list[int] = []
        self.environment_episode_ids = [0 for _ in range(num_environments)]
        self.episode_sparse_returns = [0.0 for _ in range(num_environments)]
        self.episode_shaped_returns = [0.0 for _ in range(num_environments)]
        self.timeout_counts = [[0, 0] for _ in range(num_environments)]
        self.invalid_counts = [[0, 0] for _ in range(num_environments)]
        self.sessions: list[_FrozenPartnerSession] = []
        source = StandardStateSource() if state_source is None else state_source

        for env_index in range(num_environments):
            env_seed = derive_seed(base_seed, "environment", env_index)
            self.environment_seeds.append(env_seed)
            env = build_env(
                environment_config,
                state_source=source,
                rng=np.random.default_rng(env_seed),
            )
            self.environments.append(env)
            self.observation_builders.append(
                ObservationBuilder(env, observation_config)
            )
        for env_index in range(num_environments):
            self.sessions.append(self._start_episode(env_index))

    def _start_episode(self, env_index: int) -> _FrozenPartnerSession:
        env = self.environments[env_index]
        episode_id = self.environment_episode_ids[env_index]
        env.reset(regen_mdp=False)
        episode_context = {
            "phase": "training",
            "environment_index": env_index,
            "episode_id": episode_id,
        }
        spec = self.partner_sampler.sample(self.partner_rng, episode_context)
        ego_player_index = self.position_sampler.sample(
            self.position_rng,
            {**episode_context, "partner": spec.name},
        )
        partner_position = 1 - ego_player_index
        partner = self.partner_factory.build(
            spec,
            env=env,
            observation_builder=self.observation_builders[env_index],
            player_position=partner_position,
            seed=derive_seed(
                self.environment_seeds[env_index],
                "partner_episode",
                episode_id,
            ),
        )
        partner.reset()
        partner.set_agent_index(partner_position)
        partner.set_mdp(env.mdp)
        return _FrozenPartnerSession(
            spec=spec,
            agent=partner,
            ego_player_index=ego_player_index,
            start_timestep=int(env.state.timestep),
        )

    def _encoded_observations(self) -> np.ndarray:
        encoded = []
        for env, builder, session in zip(
            self.environments,
            self.observation_builders,
            self.sessions,
        ):
            encoded.append(
                self.observation_spec.encode(
                    builder(env.state, session.ego_player_index)
                )
            )
        return np.asarray(encoded, dtype=np.float32)

    def collect(
        self,
        model: ActorCritic,
        *,
        rollout_steps: int,
        gamma: float,
        gae_lambda: float,
    ) -> RolloutBatch:
        """Collect one rollout containing only trainable ego transitions."""
        if rollout_steps <= 0:
            raise ValueError("rollout_steps must be positive")
        num_environments = len(self.environments)
        observations = torch.empty(
            (rollout_steps, num_environments, self.observation_spec.encoded_size),
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.empty(
            (rollout_steps, num_environments),
            dtype=torch.long,
            device=self.device,
        )
        log_probabilities = torch.empty_like(actions, dtype=torch.float32)
        values = torch.empty_like(actions, dtype=torch.float32)
        rewards = torch.empty_like(actions, dtype=torch.float32)
        dones = torch.empty_like(actions, dtype=torch.float32)
        completed_results: list[EpisodeResult] = []
        completed_metadata: list[dict[str, Any]] = []
        partner_step_counts: Counter[str] = Counter()
        position_step_counts: Counter[str] = Counter()

        model.eval()
        for rollout_index in range(rollout_steps):
            encoded = self._encoded_observations()
            obs_tensor = torch.as_tensor(encoded, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                policy_step = model.act_batch(obs_tensor, deterministic=False)

            observations[rollout_index] = obs_tensor
            actions[rollout_index] = policy_step.actions
            log_probabilities[rollout_index] = policy_step.log_probabilities
            values[rollout_index] = policy_step.values

            for env_index, env in enumerate(self.environments):
                session = self.sessions[env_index]
                ego_position = session.ego_player_index
                partner_position = 1 - ego_position
                state = env.state
                ego_action_index = int(actions[rollout_index, env_index].item())
                ego_action = action_index_to_overcooked_action(ego_action_index)
                partner_action, raw_partner_info = session.agent.action(state)
                partner_info = dict(raw_partner_info or {})
                joint_actions = [None, None]
                joint_actions[ego_position] = ego_action
                joint_actions[partner_position] = partner_action
                joint_infos: list[dict[str, Any]] = [{}, {}]
                joint_infos[ego_position] = {
                    "policy_name": "trainable_ego",
                    "action_index": ego_action_index,
                }
                joint_infos[partner_position] = partner_info
                next_state, sparse_reward, done, info = env.step(
                    tuple(joint_actions),
                    tuple(joint_infos),
                )
                transition = EpisodeStep(
                    episode_id=self.environment_episode_ids[env_index],
                    timestep=int(state.timestep),
                    state=state,
                    next_state=next_state,
                    joint_action=tuple(joint_actions),
                    joint_action_indices=tuple(
                        overcooked_action_to_index(action) for action in joint_actions
                    ),
                    joint_infos=tuple(joint_infos),
                    reward=float(sparse_reward),
                    done=bool(done),
                    info=info,
                )
                shaped = transition.info.get("shaped_r_by_agent", (0.0, 0.0))
                rewards[rollout_index, env_index] = (
                    transition.reward
                    + self.reward_shaping * float(shaped[ego_position])
                )
                dones[rollout_index, env_index] = float(done)
                self.episode_sparse_returns[env_index] += transition.reward
                self.episode_shaped_returns[env_index] += float(sum(shaped))
                self.timeout_counts[env_index][partner_position] += int(
                    bool(partner_info.get("timeout_action_replaced", False))
                )
                self.invalid_counts[env_index][partner_position] += int(
                    bool(partner_info.get("invalid_action_replaced", False))
                )
                partner_step_counts[session.spec.name] += 1
                position_step_counts[str(ego_position)] += 1

                if done:
                    completed_results.append(
                        build_episode_result(
                            env=env,
                            episode_id=self.environment_episode_ids[env_index],
                            seed=derive_seed(
                                self.environment_seeds[env_index],
                                "episode",
                                self.environment_episode_ids[env_index],
                            ),
                            ego_player_index=ego_position,
                            role_swap=ego_position == 1,
                            sparse_return=self.episode_sparse_returns[env_index],
                            shaped_return=self.episode_shaped_returns[env_index],
                            episode_start_timestep=session.start_timestep,
                            timeout_counts_by_agent=tuple(
                                self.timeout_counts[env_index]
                            ),
                            invalid_counts_by_agent=tuple(
                                self.invalid_counts[env_index]
                            ),
                        )
                    )
                    completed_metadata.append(
                        {
                            "partner": session.spec.name,
                            "partner_source": session.spec.source,
                            "ego_player_index": ego_position,
                            "environment_index": env_index,
                            "episode_id": self.environment_episode_ids[env_index],
                        }
                    )
                    self.episode_sparse_returns[env_index] = 0.0
                    self.episode_shaped_returns[env_index] = 0.0
                    self.timeout_counts[env_index] = [0, 0]
                    self.invalid_counts[env_index] = [0, 0]
                    self.environment_episode_ids[env_index] += 1
                    self.sessions[env_index] = self._start_episode(env_index)

        next_encoded = torch.as_tensor(
            self._encoded_observations(),
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            _, next_values = model(next_encoded)
        advantages, returns = _generalized_advantages(
            rewards=rewards,
            dones=dones,
            values=values,
            next_values=next_values,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        return RolloutBatch(
            observations=observations,
            actions=actions,
            old_log_probabilities=log_probabilities,
            old_values=values,
            advantages=advantages,
            returns=returns,
            rewards=rewards,
            dones=dones,
            completed_episode_results=tuple(completed_results),
            completed_episode_metadata=tuple(completed_metadata),
            partner_step_counts=dict(partner_step_counts),
            ego_position_step_counts=dict(position_step_counts),
        )
