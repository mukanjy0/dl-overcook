"""Main execution loop for the Overcooked competition base."""

from __future__ import annotations

from typing import Any

import numpy as np
from overcooked_ai_py.mdp.actions import Action

from src.dataset_metadata import build_dataset_metadata
from src.demonstrations import DemonstrationRecorder
from src.environment import build_env
from src.episode import EpisodeStep, run_episode
from src.logging_utils import CompetitionLogger, StepRecord
from src.observations import ObservationBuilder
from src.policy_loader import build_two_policies
from src.rendering import Renderer
from src.seed_utils import set_global_seed


def _action_to_name(action: Any) -> str:
    return Action.ACTION_TO_CHAR.get(action, str(action))


def _uses_human_keyboard(config: dict[str, Any]) -> bool:
    policies = config.get("policies", {}) or {}
    for policy_cfg in policies.values():
        if (
            str(policy_cfg.get("type", "builtin")).lower() == "builtin"
            and str(policy_cfg.get("name", "")).lower() == "human_keyboard"
        ):
            return True
    return False


def _prepare_rendering_config(config: dict[str, Any]) -> dict[str, Any]:
    rendering_config = dict(config.get("rendering", {}) or {})
    if _uses_human_keyboard(config) and rendering_config.get("mode", "none") != "window":
        rendering_config["mode"] = "window"
        rendering_config.setdefault("fps", 5)
        print("Human keyboard policy detected: forcing rendering.mode='window'.")
    return rendering_config


def _ego_player_positions(execution_config: dict[str, Any]) -> list[int]:
    configured = execution_config.get("ego_player_positions")
    if configured is None:
        return [0, 1] if bool(execution_config.get("swap_agent_positions", False)) else [0]
    positions = [int(value) for value in configured]
    if not positions or any(value not in (0, 1) for value in positions):
        raise ValueError("execution.ego_player_positions must contain only 0 and/or 1")
    return positions


def run_from_config(config: dict[str, Any]) -> dict[str, Any]:
    """Run one configured evaluation/debug session and return aggregate results."""
    seed = config.get("seed")
    set_global_seed(None if seed is None else int(seed))

    env = build_env(config["environment"])
    obs_builder = ObservationBuilder(env, config.get("observation", {}))
    logger = CompetitionLogger(config.get("logging", {}))
    renderer = Renderer(_prepare_rendering_config(config))
    demo_recorder = DemonstrationRecorder(config.get("data_collection", {}))

    execution_config = config.get("execution", {}) or {}
    num_episodes = int(execution_config.get("num_episodes", 1))
    episode_seeds = execution_config.get("episode_seeds")
    if episode_seeds is None:
        episode_seeds = list(range(num_episodes))
    if len(episode_seeds) < num_episodes:
        raise ValueError("execution.episode_seeds must have at least num_episodes elements")

    ego_player_positions = _ego_player_positions(execution_config)
    role_swaps = [position == 1 for position in ego_player_positions]
    episode_results = []
    episode_id = 0
    stop_requested = False

    try:
        for base_episode_idx in range(num_episodes):
            if stop_requested:
                break
            for ego_player_index in ego_player_positions:
                if stop_requested:
                    break
                ep_seed = int(episode_seeds[base_episode_idx])
                role_swap = ego_player_index == 1
                set_global_seed(ep_seed)

                # Rebuild policies per rollout to avoid state leakage across role swaps.
                ego_agent, partner_agent = build_two_policies(
                    config,
                    env,
                    obs_builder,
                    seed=ep_seed,
                )
                agents = (
                    (ego_agent, partner_agent)
                    if ego_player_index == 0
                    else (partner_agent, ego_agent)
                )
                trajectory: list[Any] = []

                def on_reset(current_env: Any) -> bool:
                    renderer.reset()
                    renderer.maybe_render(current_env, timestep=0)
                    return renderer.closed_by_user

                def on_step(step: EpisodeStep, current_env: Any) -> bool:
                    demo_recorder.record_step(
                        obs_builder=obs_builder,
                        episode_id=step.episode_id,
                        timestep=step.timestep,
                        layout_name=current_env.mdp.layout_name,
                        role_swap=role_swap,
                        episode_seed=ep_seed,
                        state=step.state,
                        next_state=step.next_state,
                        joint_action_indices=list(step.joint_action_indices),
                        reward=step.reward,
                        done=step.done,
                        info=step.info,
                    )
                    logger.log_step(
                        StepRecord(
                            episode_id=step.episode_id,
                            timestep=step.timestep,
                            layout_name=current_env.mdp.layout_name,
                            role_swap=role_swap,
                            action_0=_action_to_name(step.joint_action[0]),
                            action_1=_action_to_name(step.joint_action[1]),
                            reward=step.reward,
                            done=step.done,
                            info=step.info,
                        )
                    )
                    trajectory.append(
                        (step.state, step.joint_action, step.reward, step.done, step.info)
                    )
                    renderer.maybe_render(
                        current_env,
                        timestep=step.next_state.timestep,
                        joint_action=tuple(_action_to_name(a) for a in step.joint_action),
                        reward=step.reward,
                    )
                    return renderer.closed_by_user

                result = run_episode(
                    env=env,
                    agents=agents,
                    episode_id=episode_id,
                    seed=ep_seed,
                    role_swap=role_swap,
                    ego_player_index=ego_player_index,
                    on_reset=on_reset,
                    on_step=on_step,
                )
                summary = result.legacy_summary()
                logger.log_episode(summary)
                demo_recorder.record_episode(summary)
                logger.save_trajectory(episode_id, trajectory)
                episode_results.append(result)
                episode_id += 1
                stop_requested = result.stopped_by_user

        logger.flush()
        demo_metadata = build_dataset_metadata(
            config=config,
            env=env,
            obs_builder=obs_builder,
            num_episodes=num_episodes,
            episode_seeds=[int(value) for value in episode_seeds],
            role_swaps=role_swaps,
        )
        demo_recorder.flush(metadata=demo_metadata)
    finally:
        renderer.close()

    returns = [result.sparse_return for result in episode_results]
    scores = [result.official_score for result in episode_results]
    aggregate = {
        "num_rollouts": len(returns),
        "mean_return_sparse": float(np.mean(returns)) if returns else 0.0,
        "std_return_sparse": float(np.std(returns)) if returns else 0.0,
        "returns_sparse": returns,
        "mean_official_score": float(np.mean(scores)) if scores else 0.0,
        "std_official_score": float(np.std(scores)) if scores else 0.0,
        "official_scores": scores,
        "episode_results": [result.to_dict() for result in episode_results],
        "output_dir": str(logger.output_dir),
    }
    return aggregate
