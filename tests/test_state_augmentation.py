from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from itertools import product
from pathlib import Path

import numpy as np
import pytest
import yaml
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.agents.agent import Agent, AgentPair
from overcooked_ai_py.mdp.overcooked_mdp import ObjectState

from src.environment import build_env
from src.experiment_config import load_experiment_config
from src.evaluation.scoring import calculate_official_score
from src.episode import run_episode
from src.observations import ObservationBuilder
from src.policy_loader import build_policy
from src.state_augmentation.buffer import (
    SourcePolicyMetadata,
    StateBuffer,
    StateBufferCompatibilityError,
    StateRecord,
    TrajectoryMetadata,
    load_state_buffer,
    save_state_buffer,
    validate_state_buffer,
)
from src.state_augmentation.collection import (
    collect_state_buffer,
    load_state_collection_config,
)
from src.state_augmentation.sampling import StateBufferSampler
from src.state_augmentation.serialization import (
    build_environment_metadata,
    restore_state,
    serialize_state,
)
from src.state_augmentation.sources import BufferedStateSource
from src.training.trainer import train


ENVIRONMENT_CONFIG = {
    "layout_name": "cramped_room",
    "layout_file": None,
    "horizon": 6,
    "old_dynamics": True,
    "mdp_overrides": {},
}


@pytest.fixture(scope="module")
def coordination_ring_progress_states():
    environment_config = {
        **ENVIRONMENT_CONFIG,
        "layout_name": "coordination_ring",
        "horizon": 100,
    }
    env = build_env(environment_config)
    observation_builder = ObservationBuilder(
        env,
        {"type": "featurized", "include_agent_index": True},
    )
    agents = tuple(
        build_policy(
            {
                "type": "builtin",
                "name": "greedy_full_task",
                "max_action_time_ms": 0,
            },
            env,
            observation_builder,
            seed=100 + index,
        )
        for index in range(2)
    )
    pair = AgentPair(*agents)
    env.reset(regen_mdp=False)
    pair.reset()
    pair.set_mdp(env.mdp)

    states = {}
    delivery_transition = None
    while not env.is_done():
        state = env.state.deepcopy()
        serialized = serialize_state(state)
        held_objects = [
            player["held_object"]
            for player in serialized["players"]
            if player["held_object"] is not None
        ]
        soups = [
            obj
            for obj in held_objects + serialized["objects"]
            if obj["name"] == "soup"
        ]
        if held_objects and "held_object" not in states:
            states["held_object"] = state
        if any(obj.get("is_cooking", False) for obj in soups):
            states.setdefault("cooking_timer", state)
        joint_action = tuple(action for action, _ in pair.joint_action(env.state))
        _, sparse_reward, _, _ = env.step(joint_action)
        if sparse_reward > 0:
            states["near_delivery"] = state
            delivery_transition = (state, joint_action)
            break

    counter_state = env.mdp.get_standard_start_state()
    counter_position = next(
        position
        for position in env.mdp.terrain_pos_dict["X"]
        if position not in env.mdp.start_player_positions
    )
    counter_state.objects[counter_position] = ObjectState("onion", counter_position)
    states["counter_object"] = counter_state

    assert set(states) == {
        "held_object",
        "cooking_timer",
        "counter_object",
        "near_delivery",
    }
    assert delivery_transition is not None
    return env, states, delivery_transition


def _state_with_held_onion(env):
    env.reset(regen_mdp=False)
    env.step((Direction.NORTH, Action.STAY))
    env.step((Direction.WEST, Action.STAY))
    state, _, _, _ = env.step((Action.INTERACT, Action.STAY))
    assert state.players[0].has_object()
    return state.deepcopy()


def _buffer_for_states(env, states) -> StateBuffer:
    policy = SourcePolicyMetadata(
        identifier="self_play_checkpoint",
        source="frozen_checkpoint",
        policy_type="python_class",
        policy_name="frozen_self_play",
        checkpoint_path="checkpoints/example.pt",
        checkpoint_sha256="a" * 64,
        checkpoint_identity=f"sha256:{'a' * 64}",
        policy_config={"type": "python_class", "name": "frozen_self_play"},
    )
    records = tuple(
        StateRecord(
            record_id=f"state_{index:03d}",
            layout=str(env.mdp.layout_name),
            physical_player_assignment={
                "0": policy.identifier,
                "1": policy.identifier,
            },
            episode_id=0,
            trajectory_id="self_play:episode=0:seed=41",
            timestep=int(state.timestep),
            seed=41,
            serialized_state=serialize_state(state),
        )
        for index, state in enumerate(states)
    )
    return StateBuffer.create(
        environment=build_environment_metadata(env, ENVIRONMENT_CONFIG),
        source_policies=(policy,),
        collection_config={
            "collection": {"every_k": 1, "num_episodes": 1},
            "seed": 41,
        },
        trajectories=(
            TrajectoryMetadata(
                trajectory_id="self_play:episode=0:seed=41",
                pairing_id="self_play",
                physical_player_assignment={
                    "0": policy.identifier,
                    "1": policy.identifier,
                },
                episode_id=0,
                seed=41,
                episode_length=int(env.horizon),
                sparse_return=0.0,
                shaped_return=0.0,
                delivery_timesteps=(),
                official_score=0,
            ),
        ),
        records=records,
    )


def _tiny_training_config(root: Path) -> dict:
    return {
        "experiment": {"name": "stage_b_test", "seed": 17, "device": "cpu"},
        "environment": deepcopy(ENVIRONMENT_CONFIG),
        "observation": {"type": "featurized", "include_agent_index": True},
        "model": {
            "architecture": "mlp_actor_critic",
            "parameters": {"hidden_sizes": [8], "activation": "tanh"},
        },
        "training": {
            "algorithm": "ppo",
            "total_steps": 12,
            "num_environments": 2,
            "rollout_steps": 3,
            "reward_shaping": 0.0,
            "ppo": {
                "learning_rate": 0.001,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "update_epochs": 1,
                "minibatch_size": 4,
            },
        },
        "partner": {"sampler": "self_play", "policies": []},
        "evaluation": {},
        "checkpoint": {
            "resume_from": None,
            "save_interval": 0,
            "export_path": str(root / "inference.pt"),
        },
        "outputs": {
            "root": str(root),
            "logs": str(root / "logs"),
            "checkpoints": str(root / "checkpoints"),
            "metrics": str(root / "metrics"),
        },
    }


def test_serialize_restore_round_trip_fidelity() -> None:
    env = build_env(ENVIRONMENT_CONFIG)
    state = _state_with_held_onion(env)

    restored = restore_state(
        serialize_state(state),
        env.mdp,
        horizon=int(env.horizon),
    )

    assert restored == state
    assert serialize_state(restored) == serialize_state(state)
    assert restored is not state


def test_augmented_only_source_restores_through_environment_reset() -> None:
    source_env = build_env(ENVIRONMENT_CONFIG)
    state = _state_with_held_onion(source_env)
    source = BufferedStateSource(
        _buffer_for_states(source_env, [state]),
        augmented_probability=1.0,
    )
    restored_env = build_env(
        ENVIRONMENT_CONFIG,
        state_source=source,
        rng=np.random.default_rng(5),
    )

    restored_env.reset(regen_mdp=False)

    assert serialize_state(restored_env.state) == serialize_state(state)
    assert source.metrics()["standard_resets"] == 0
    assert source.metrics()["augmented_resets"] >= 1


def test_restored_state_has_equivalent_next_transition() -> None:
    env = build_env(ENVIRONMENT_CONFIG)
    state = _state_with_held_onion(env)
    restored = restore_state(serialize_state(state), env.mdp, horizon=int(env.horizon))
    joint_action = (Direction.SOUTH, Direction.WEST)

    expected_state, expected_info = env.mdp.get_state_transition(
        state.deepcopy(),
        joint_action,
    )
    actual_state, actual_info = env.mdp.get_state_transition(
        restored.deepcopy(),
        joint_action,
    )

    assert serialize_state(actual_state) == serialize_state(expected_state)
    assert actual_info == expected_info


def test_all_joint_actions_are_equivalent_for_task_progress_states(
    coordination_ring_progress_states,
) -> None:
    env, states, _ = coordination_ring_progress_states
    for label, state in states.items():
        restored = restore_state(
            serialize_state(state),
            env.mdp,
            horizon=int(env.horizon),
        )
        for joint_action in product(Action.ALL_ACTIONS, repeat=2):
            expected_state, expected_info = env.mdp.get_state_transition(
                state.deepcopy(),
                joint_action,
            )
            actual_state, actual_info = env.mdp.get_state_transition(
                restored.deepcopy(),
                joint_action,
            )
            assert serialize_state(actual_state) == serialize_state(expected_state), label
            assert actual_info == expected_info, label


def test_augmented_episode_resets_stats_timestamps_counters_and_policy_state(
    coordination_ring_progress_states,
) -> None:
    _, _, (state, joint_action) = coordination_ring_progress_states
    horizon = int(state.timestep) + 1
    env = build_env(
        {
            "layout_name": "coordination_ring",
            "horizon": horizon,
            "old_dynamics": True,
            "mdp_overrides": {},
        },
        start_state_fn=lambda: state.deepcopy(),
    )

    class OneActionAgent(Agent):
        def __init__(self, action):
            self.next_action = action
            self.reset_calls = 0
            self.action_calls = 0
            super().__init__()

        def reset(self):
            super().reset()
            self.reset_calls += 1
            self.action_calls = 0

        def action(self, current_state):
            del current_state
            self.action_calls += 1
            return self.next_action, {}

    agents = tuple(OneActionAgent(action) for action in joint_action)
    result = run_episode(
        env=env,
        agents=agents,
        episode_id=99,
        seed=123,
    )

    assert result.episode_id == 99
    assert result.start_timestep == int(state.timestep)
    assert result.episode_length == 1
    assert result.delivery_timesteps == (int(state.timestep),)
    assert result.official_score == calculate_official_score(
        result.delivery_timesteps,
        horizon=horizon,
        total_team_timeouts=0,
    )
    assert all(agent.reset_calls >= 2 for agent in agents)
    assert all(agent.action_calls == 1 for agent in agents)
    assert sum(env.game_stats["cumulative_sparse_rewards_by_agent"]) > 0

    env.reset(regen_mdp=False)
    assert sum(env.game_stats["cumulative_sparse_rewards_by_agent"]) == 0
    assert sum(env.game_stats["cumulative_shaped_rewards_by_agent"]) == 0
    assert all(
        not events
        for key, events_by_agent in env.game_stats.items()
        if key not in {
            "cumulative_sparse_rewards_by_agent",
            "cumulative_shaped_rewards_by_agent",
        }
        for events in events_by_agent
    )


def test_buffer_sampling_is_deterministic_under_fixed_seed() -> None:
    env = build_env(ENVIRONMENT_CONFIG)
    env.reset(regen_mdp=False)
    states = [env.state.deepcopy()]
    for _ in range(4):
        state, _, _, _ = env.step((Action.STAY, Action.STAY))
        states.append(state.deepcopy())
    sampler = StateBufferSampler(_buffer_for_states(env, states))

    first_rng = np.random.default_rng(2026)
    second_rng = np.random.default_rng(2026)
    first = [sampler.sample(first_rng).record_id for _ in range(20)]
    second = [sampler.sample(second_rng).record_id for _ in range(20)]

    assert first == second
    assert len(set(first)) > 1


def test_schema_and_environment_version_validation() -> None:
    env = build_env(ENVIRONMENT_CONFIG)
    env.reset(regen_mdp=False)
    buffer = _buffer_for_states(env, [env.state])

    unsupported_schema = buffer.to_dict()
    unsupported_schema["schema_version"] = 999
    with pytest.raises(StateBufferCompatibilityError, match="Unsupported"):
        StateBuffer.from_dict(unsupported_schema)

    wrong_runtime = replace(
        buffer,
        environment={**buffer.environment, "overcooked_ai_version": "0.0.invalid"},
    )
    with pytest.raises(StateBufferCompatibilityError, match="version mismatch"):
        validate_state_buffer(wrong_runtime, env=env)

    wrong_metadata = replace(
        buffer,
        environment={**buffer.environment, "metadata_version": 999},
    )
    with pytest.raises(StateBufferCompatibilityError, match="metadata version"):
        validate_state_buffer(wrong_metadata, env=env)


def test_checkpoint_identity_is_hash_based_and_path_portable() -> None:
    env = build_env(ENVIRONMENT_CONFIG)
    env.reset(regen_mdp=False)
    buffer = _buffer_for_states(env, [env.state])
    original = buffer.source_policies[0]
    moved = replace(
        original,
        checkpoint_path="/different/runtime/mount/inference.pt",
        policy_config={
            **original.policy_config,
            "config": {"checkpoint_path": "/another/mount/inference.pt"},
        },
    )
    moved_buffer = replace(buffer, source_policies=(moved,))

    validate_state_buffer(moved_buffer, env=env)

    assert moved.checkpoint_identity == original.checkpoint_identity
    assert moved.checkpoint_identity == f"sha256:{'a' * 64}"


def test_horizon_mismatch_is_allowed_only_for_nonterminal_records() -> None:
    source_env = build_env(ENVIRONMENT_CONFIG)
    source_env.reset(regen_mdp=False)
    state_at_one, _, _, _ = source_env.step((Action.STAY, Action.STAY))
    buffer = _buffer_for_states(source_env, [state_at_one])

    longer_env = build_env({**ENVIRONMENT_CONFIG, "horizon": 10})
    validate_state_buffer(
        buffer,
        env=longer_env,
        environment_config={**ENVIRONMENT_CONFIG, "horizon": 10},
    )

    terminal_env = build_env({**ENVIRONMENT_CONFIG, "horizon": 1})
    with pytest.raises(StateBufferCompatibilityError, match="below horizon"):
        validate_state_buffer(
            buffer,
            env=terminal_env,
            environment_config={**ENVIRONMENT_CONFIG, "horizon": 1},
        )


def test_incompatible_layout_is_rejected() -> None:
    source_env = build_env(ENVIRONMENT_CONFIG)
    source_env.reset(regen_mdp=False)
    buffer = _buffer_for_states(source_env, [source_env.state])
    target_env = build_env(
        {**ENVIRONMENT_CONFIG, "layout_name": "coordination_ring"}
    )

    with pytest.raises(StateBufferCompatibilityError, match="does not match"):
        validate_state_buffer(buffer, env=target_env)


def test_malformed_state_is_rejected() -> None:
    env = build_env(ENVIRONMENT_CONFIG)
    env.reset(regen_mdp=False)
    buffer = _buffer_for_states(env, [env.state])
    malformed = deepcopy(buffer.records[0].serialized_state)
    malformed["players"][0]["position"] = [0, 0]
    malformed_record = replace(buffer.records[0], serialized_state=malformed)

    with pytest.raises(StateBufferCompatibilityError, match="malformed"):
        validate_state_buffer(replace(buffer, records=(malformed_record,)), env=env)


def test_collection_records_every_kth_state_and_pairing_metadata(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "collect.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "paths_relative_to_config": True,
                "seed": 29,
                "environment": deepcopy(ENVIRONMENT_CONFIG),
                "observation": {
                    "type": "featurized",
                    "include_agent_index": True,
                },
                "collection": {
                    "output_path": "buffer.json.gz",
                    "every_k": 2,
                    "include_initial_state": True,
                    "num_episodes": 1,
                    "episode_seeds": [101],
                    "pairings": [
                        {
                            "id": "self_play_stay",
                            "player_0": {
                                "identifier": "stay",
                                "source": "scripted",
                                "policy": {"type": "builtin", "name": "stay"},
                            },
                            "player_1": {
                                "identifier": "stay",
                                "source": "scripted",
                                "policy": {"type": "builtin", "name": "stay"},
                            },
                        },
                        {
                            "id": "scripted_cross_play",
                            "player_0": {
                                "identifier": "stay",
                                "source": "scripted",
                                "policy": {"type": "builtin", "name": "stay"},
                            },
                            "player_1": {
                                "identifier": "random_motion",
                                "source": "scripted",
                                "policy": {
                                    "type": "builtin",
                                    "name": "random_motion",
                                },
                            },
                        },
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    result = collect_state_buffer(load_state_collection_config(config_path))
    buffer = load_state_buffer(result["output_path"])

    assert result["summary"]["num_states"] == 6
    assert {record.timestep for record in buffer.records} == {0, 2, 4}
    assert {record.seed for record in buffer.records} == {101}
    assert {policy.identifier for policy in buffer.source_policies} == {
        "stay",
        "random_motion",
    }
    assert {
        tuple(sorted(record.physical_player_assignment.items()))
        for record in buffer.records
    } == {
        (("0", "stay"), ("1", "stay")),
        (("0", "stay"), ("1", "random_motion")),
    }
    summary = result["summary"]
    assert summary["duplicate_statistics"]["exact_unique_states"] <= 6
    assert summary["source_pairing_balance"][
        "state_counts_by_physical_assignment"
    ] == {"stay|stay": 3, "stay|random_motion": 3}
    assert summary["timestep_distribution"]["regions"] == {
        "early_0_25pct": 2,
        "middle_25_75pct": 4,
        "late_75_100pct": 0,
    }
    assert summary["trajectory_outcomes"] == {
        "successful": 0,
        "failed": 2,
        "stopped_by_user": 0,
        "success_rate": 0.0,
        "deliveries": 0,
        "mean_sparse_return": 0.0,
        "by_pairing_id": {
            "self_play_stay": {
                "trajectories": 1,
                "successful": 0,
                "failed": 1,
                "deliveries": 0,
            },
            "scripted_cross_play": {
                "trajectories": 1,
                "successful": 0,
                "failed": 1,
                "deliveries": 0,
            },
        },
    }


def test_stage_a_training_is_unchanged_when_augmentation_is_absent(
    tmp_path: Path,
) -> None:
    raw = _tiny_training_config(tmp_path / "standard")
    raw["training"]["total_steps"] = 6
    config_path = tmp_path / "standard.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    loaded = load_experiment_config(config_path)
    result = train(loaded)

    assert loaded.state_augmentation.reset_mode == "standard"
    assert result["state_initialization"] == {
        "reset_mode": "standard",
        "standard_resets": 0,
        "augmented_resets": 0,
    }


def test_short_ppo_smoke_uses_mixed_reset_distribution(tmp_path: Path) -> None:
    env = build_env(ENVIRONMENT_CONFIG)
    env.reset(regen_mdp=False)
    augmented_state, _, _, _ = env.step((Action.STAY, Action.STAY))
    buffer_path = tmp_path / "mixed_buffer.json.gz"
    save_state_buffer(_buffer_for_states(env, [augmented_state]), buffer_path)

    raw = _tiny_training_config(tmp_path / "mixed")
    raw["training"]["total_steps"] = 48
    raw["state_augmentation"] = {
        "reset_mode": "mixed",
        "buffer_path": str(buffer_path),
        "augmented_probability": 0.5,
    }
    config_path = tmp_path / "mixed.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    result = train(load_experiment_config(config_path))

    initialization = result["state_initialization"]
    assert initialization["reset_mode"] == "mixed"
    assert initialization["standard_resets"] > 0
    assert initialization["augmented_resets"] > 0
    assert Path(result["training_checkpoint"]).exists()
