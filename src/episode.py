"""Reusable single-episode execution using the teacher's environment semantics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

from overcooked_ai_py.agents.agent import Agent, AgentPair

from src.constants import overcooked_action_to_index
from src.evaluation.scoring import calculate_official_score


@dataclass(frozen=True)
class EpisodeStep:
    """One environment transition shared by logging and data collection hooks."""

    episode_id: int
    timestep: int
    state: Any
    next_state: Any
    joint_action: tuple[Any, Any]
    joint_action_indices: tuple[int, int]
    joint_infos: tuple[dict[str, Any], dict[str, Any]]
    reward: float
    done: bool
    info: dict[str, Any]


@dataclass(frozen=True)
class EpisodeResult:
    """Canonical immutable result for one rollout."""

    episode_id: int
    layout: str
    seed: int
    ego_player_index: int
    role_swap: bool
    sparse_return: float
    shaped_return: float
    start_timestep: int
    episode_length: int
    delivery_timesteps_by_agent: tuple[tuple[int, ...], tuple[int, ...]]
    delivery_timesteps: tuple[int, ...]
    timeout_count_ego: int
    timeout_count_partner: int
    timeout_count_total: int
    invalid_action_replacements_by_agent: tuple[int, int]
    official_score: int
    stopped_by_user: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        result = asdict(self)
        result["delivery_timesteps_by_agent"] = [
            list(values) for values in self.delivery_timesteps_by_agent
        ]
        result["delivery_timesteps"] = list(self.delivery_timesteps)
        result["invalid_action_replacements_by_agent"] = list(
            self.invalid_action_replacements_by_agent
        )
        return result

    def legacy_summary(self) -> dict[str, Any]:
        """Expose original episode keys together with training diagnostics."""
        return {
            "episode_id": self.episode_id,
            "layout_name": self.layout,
            "seed": self.seed,
            "role_swap": self.role_swap,
            "ego_player_index": self.ego_player_index,
            "return_sparse": self.sparse_return,
            "ep_sparse_r": self.sparse_return,
            "ep_shaped_r": self.shaped_return,
            "start_timestep": self.start_timestep,
            "ep_length": self.episode_length,
            "delivery_timesteps_by_agent": [
                list(values) for values in self.delivery_timesteps_by_agent
            ],
            "delivery_timesteps": list(self.delivery_timesteps),
            "timeout_count_ego": self.timeout_count_ego,
            "timeout_count_partner": self.timeout_count_partner,
            "timeout_count_total": self.timeout_count_total,
            "invalid_action_replacements_by_agent": list(
                self.invalid_action_replacements_by_agent
            ),
            "official_score": self.official_score,
        }


ResetHook = Callable[[Any], bool | None]
StepHook = Callable[[EpisodeStep, Any], bool | None]


def build_episode_result(
    *,
    env: Any,
    episode_id: int,
    seed: int,
    ego_player_index: int,
    role_swap: bool,
    sparse_return: float,
    shaped_return: float,
    episode_start_timestep: int = 0,
    timeout_counts_by_agent: tuple[int, int] = (0, 0),
    invalid_counts_by_agent: tuple[int, int] = (0, 0),
    stopped_by_user: bool = False,
) -> EpisodeResult:
    """Build the canonical result from upstream episode statistics."""
    deliveries_raw = env.game_stats.get("soup_delivery", ((), ()))
    deliveries_by_agent = tuple(
        tuple(int(timestep) for timestep in deliveries_raw[index])
        if index < len(deliveries_raw)
        else tuple()
        for index in range(2)
    )
    delivery_timesteps = tuple(
        sorted(timestep for agent_events in deliveries_by_agent for timestep in agent_events)
    )
    partner_index = 1 - ego_player_index
    total_timeouts = sum(timeout_counts_by_agent)
    return EpisodeResult(
        episode_id=episode_id,
        layout=str(env.mdp.layout_name),
        seed=int(seed),
        ego_player_index=ego_player_index,
        role_swap=bool(role_swap),
        sparse_return=float(sparse_return),
        shaped_return=float(shaped_return),
        start_timestep=int(episode_start_timestep),
        episode_length=int(env.state.timestep) - int(episode_start_timestep),
        delivery_timesteps_by_agent=deliveries_by_agent,
        delivery_timesteps=delivery_timesteps,
        timeout_count_ego=int(timeout_counts_by_agent[ego_player_index]),
        timeout_count_partner=int(timeout_counts_by_agent[partner_index]),
        timeout_count_total=int(total_timeouts),
        invalid_action_replacements_by_agent=tuple(
            int(value) for value in invalid_counts_by_agent
        ),
        official_score=calculate_official_score(
            delivery_timesteps,
            horizon=int(env.horizon),
            total_team_timeouts=total_timeouts,
        ),
        stopped_by_user=bool(stopped_by_user),
    )


def run_episode(
    *,
    env: Any,
    agents: tuple[Agent, Agent],
    episode_id: int,
    seed: int,
    role_swap: bool = False,
    ego_player_index: int = 0,
    on_reset: ResetHook | None = None,
    on_step: StepHook | None = None,
) -> EpisodeResult:
    """Run one episode without rebuilding environment or policy semantics."""
    if ego_player_index not in (0, 1):
        raise ValueError("ego_player_index must be 0 or 1")

    agent_pair = AgentPair(*agents)
    env.reset(regen_mdp=False)
    episode_start_timestep = int(env.state.timestep)
    agent_pair.reset()
    agent_pair.set_mdp(env.mdp)

    stopped_by_user = bool(on_reset(env)) if on_reset is not None else False
    done = stopped_by_user
    sparse_return = 0.0
    shaped_return = 0.0
    timeout_counts = [0, 0]
    invalid_counts = [0, 0]

    while not done:
        state = env.state
        joint_action_and_infos = agent_pair.joint_action(state)
        joint_action, raw_joint_infos = zip(*joint_action_and_infos)
        joint_infos = tuple(dict(info or {}) for info in raw_joint_infos)
        joint_action_indices = tuple(
            overcooked_action_to_index(action) for action in joint_action
        )

        for agent_index, action_info in enumerate(joint_infos):
            timeout_counts[agent_index] += int(
                bool(action_info.get("timeout_action_replaced", False))
            )
            invalid_counts[agent_index] += int(
                bool(action_info.get("invalid_action_replaced", False))
            )

        next_state, reward, done, info = env.step(joint_action, joint_infos)
        sparse_return += float(reward)
        shaped_rewards = info.get("shaped_r_by_agent", (0.0, 0.0))
        shaped_return += float(sum(shaped_rewards or (0.0, 0.0)))

        transition = EpisodeStep(
            episode_id=episode_id,
            timestep=int(state.timestep),
            state=state,
            next_state=next_state,
            joint_action=tuple(joint_action),
            joint_action_indices=joint_action_indices,
            joint_infos=joint_infos,
            reward=float(reward),
            done=bool(done),
            info=info,
        )
        if on_step is not None and bool(on_step(transition, env)):
            stopped_by_user = True
            done = True

    return build_episode_result(
        env=env,
        episode_id=episode_id,
        seed=int(seed),
        ego_player_index=ego_player_index,
        role_swap=bool(role_swap),
        sparse_return=sparse_return,
        shaped_return=shaped_return,
        episode_start_timestep=episode_start_timestep,
        timeout_counts_by_agent=tuple(timeout_counts),
        invalid_counts_by_agent=tuple(invalid_counts),
        stopped_by_user=stopped_by_user,
    )
