"""Trajectory-state collection through existing environments and partner factories."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config import ConfigError
from src.environment import build_env
from src.episode import EpisodeStep, run_episode
from src.experiment_config import load_runtime_config
from src.observations import ObservationBuilder
from src.partners.interfaces import ConfiguredPartnerFactory, PartnerSpec
from src.seed_utils import derive_seed, set_global_seed
from src.state_augmentation.buffer import (
    SourcePolicyMetadata,
    StateBuffer,
    StateRecord,
    TrajectoryMetadata,
    inspect_state_buffer,
    save_state_buffer,
)
from src.state_augmentation.serialization import (
    build_environment_metadata,
    serialize_state,
)


@dataclass(frozen=True)
class CollectionPolicy:
    identifier: str
    spec: PartnerSpec


@dataclass(frozen=True)
class PolicyPairing:
    identifier: str
    player_0: CollectionPolicy
    player_1: CollectionPolicy


@dataclass(frozen=True)
class StateCollectionConfig:
    source_path: Path
    seed: int
    environment: dict[str, Any]
    observation: dict[str, Any]
    output_path: Path
    every_k: int
    include_initial_state: bool
    num_episodes: int
    episode_seeds: tuple[int, ...]
    pairings: tuple[PolicyPairing, ...]
    effective: dict[str, Any]


def _collection_policy(raw: Any, path: str) -> CollectionPolicy:
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must be a mapping")
    identifier = str(raw.get("identifier", "")).strip()
    if not identifier:
        raise ConfigError(f"{path}.identifier is required")
    policy = raw.get("policy")
    if not isinstance(policy, dict):
        raise ConfigError(f"{path}.policy must be a mapping")
    observation = raw.get("observation")
    if observation is not None and not isinstance(observation, dict):
        raise ConfigError(f"{path}.observation must be a mapping")
    return CollectionPolicy(
        identifier=identifier,
        spec=PartnerSpec(
            name=identifier,
            policy_config=deepcopy(policy),
            observation_config=None if observation is None else deepcopy(observation),
            source=str(raw.get("source", "state_collection")),
        ),
    )


def load_state_collection_config(path: str | Path) -> StateCollectionConfig:
    """Load and validate one trajectory-state collection YAML."""
    source_path = Path(path).expanduser().resolve()
    config = load_runtime_config(source_path)
    environment = config.get("environment")
    observation = config.get("observation", {}) or {}
    collection = config.get("collection")
    if not isinstance(environment, dict):
        raise ConfigError("Missing or invalid section 'environment'")
    if not isinstance(observation, dict):
        raise ConfigError("observation must be a mapping")
    if not isinstance(collection, dict):
        raise ConfigError("Missing or invalid section 'collection'")
    layout_name = environment.get("layout_name")
    layout_file = environment.get("layout_file")
    if bool(layout_name) == bool(layout_file):
        raise ConfigError(
            "Set exactly one of environment.layout_name or environment.layout_file"
        )
    if int(environment.get("horizon", 0)) <= 0:
        raise ConfigError("environment.horizon must be positive")
    output_path = collection.get("output_path")
    if not output_path:
        raise ConfigError("collection.output_path is required")
    try:
        every_k = int(collection.get("every_k", 10))
        num_episodes = int(collection.get("num_episodes", 1))
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            "collection.every_k and collection.num_episodes must be integers"
        ) from exc
    if every_k <= 0:
        raise ConfigError("collection.every_k must be positive")
    if num_episodes <= 0:
        raise ConfigError("collection.num_episodes must be positive")

    seed = int(config.get("seed", 0))
    raw_seeds = collection.get("episode_seeds")
    episode_seeds = (
        tuple(
            derive_seed(seed, "state_collection_episode", index)
            for index in range(num_episodes)
        )
        if raw_seeds is None
        else tuple(int(value) for value in raw_seeds)
    )
    if len(episode_seeds) < num_episodes:
        raise ConfigError(
            "collection.episode_seeds must contain at least num_episodes values"
        )

    raw_pairings = collection.get("pairings")
    if not isinstance(raw_pairings, list) or not raw_pairings:
        raise ConfigError("collection.pairings must be a non-empty list")
    pairings: list[PolicyPairing] = []
    pairing_ids: set[str] = set()
    policy_configs: dict[str, dict[str, Any]] = {}
    for index, raw_pairing in enumerate(raw_pairings):
        if not isinstance(raw_pairing, dict):
            raise ConfigError(f"collection.pairings[{index}] must be a mapping")
        pairing_id = str(raw_pairing.get("id", "")).strip()
        if not pairing_id or pairing_id in pairing_ids:
            raise ConfigError("Collection pairing ids must be non-empty and unique")
        player_0 = _collection_policy(
            raw_pairing.get("player_0"),
            f"collection.pairings[{index}].player_0",
        )
        player_1 = _collection_policy(
            raw_pairing.get("player_1"),
            f"collection.pairings[{index}].player_1",
        )
        for policy in (player_0, player_1):
            comparable = {
                "source": policy.spec.source,
                "policy": policy.spec.policy_config,
                "observation": policy.spec.observation_config,
            }
            previous = policy_configs.setdefault(policy.identifier, comparable)
            if previous != comparable:
                raise ConfigError(
                    f"Policy identifier '{policy.identifier}' has conflicting configurations"
                )
        pairings.append(
            PolicyPairing(
                identifier=pairing_id,
                player_0=player_0,
                player_1=player_1,
            )
        )
        pairing_ids.add(pairing_id)

    return StateCollectionConfig(
        source_path=source_path,
        seed=seed,
        environment=deepcopy(environment),
        observation=deepcopy(observation),
        output_path=Path(str(output_path)),
        every_k=every_k,
        include_initial_state=bool(collection.get("include_initial_state", True)),
        num_episodes=num_episodes,
        episode_seeds=episode_seeds[:num_episodes],
        pairings=tuple(pairings),
        effective=deepcopy(config),
    )


def _checkpoint_metadata(policy_config: dict[str, Any]) -> tuple[str | None, str | None]:
    runtime = policy_config.get("config", {}) or {}
    checkpoint = runtime.get("checkpoint_path") or runtime.get("model_path")
    if checkpoint in (None, ""):
        return None, None
    path = Path(str(checkpoint))
    if not path.exists() or not path.is_file():
        return str(path), None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return str(path), digest.hexdigest()


def _source_metadata(policy: CollectionPolicy) -> SourcePolicyMetadata:
    config = deepcopy(policy.spec.policy_config or {})
    checkpoint_path, checkpoint_hash = _checkpoint_metadata(config)
    return SourcePolicyMetadata(
        identifier=policy.identifier,
        source=policy.spec.source,
        policy_type=str(config.get("type", "builtin")),
        policy_name=str(config.get("name", config.get("type", "builtin"))),
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_hash,
        checkpoint_identity=(
            None if checkpoint_hash is None else f"sha256:{checkpoint_hash}"
        ),
        policy_config=config,
    )


def collect_state_buffer(config: StateCollectionConfig) -> dict[str, Any]:
    """Collect every configured k-th non-terminal state and save one buffer."""
    set_global_seed(config.seed)
    env = build_env(config.environment)
    observation_builder = ObservationBuilder(env, config.observation)
    factory = ConfiguredPartnerFactory()
    records: list[StateRecord] = []
    source_policies: dict[str, SourcePolicyMetadata] = {}
    trajectory_summaries: list[TrajectoryMetadata] = []
    global_episode_id = 0

    for pairing in config.pairings:
        for policy in (pairing.player_0, pairing.player_1):
            if policy.identifier not in source_policies:
                source_policies[policy.identifier] = _source_metadata(policy)
        assignment = {
            "0": pairing.player_0.identifier,
            "1": pairing.player_1.identifier,
        }

        for episode_index, episode_seed in enumerate(config.episode_seeds):
            trajectory_id = (
                f"{pairing.identifier}:episode={episode_index}:seed={episode_seed}"
            )
            agents = []
            for player_position, policy in enumerate(
                (pairing.player_0, pairing.player_1)
            ):
                agent = factory.build(
                    policy.spec,
                    env=env,
                    observation_builder=observation_builder,
                    player_position=player_position,
                    seed=derive_seed(
                        episode_seed,
                        f"state_collection_{pairing.identifier}",
                        player_position,
                    ),
                )
                agents.append(agent)

            def capture(state: Any) -> None:
                records.append(
                    StateRecord(
                        record_id=f"state_{len(records):09d}",
                        layout=str(env.mdp.layout_name),
                        physical_player_assignment=dict(assignment),
                        episode_id=global_episode_id,
                        trajectory_id=trajectory_id,
                        timestep=int(state.timestep),
                        seed=int(episode_seed),
                        serialized_state=serialize_state(state),
                    )
                )

            def on_reset(current_env: Any) -> bool:
                if config.include_initial_state:
                    capture(current_env.state)
                return False

            def on_step(step: EpisodeStep, current_env: Any) -> bool:
                del current_env
                if not step.done and int(step.next_state.timestep) % config.every_k == 0:
                    capture(step.next_state)
                return False

            result = run_episode(
                env=env,
                agents=(agents[0], agents[1]),
                episode_id=global_episode_id,
                seed=episode_seed,
                ego_player_index=0,
                role_swap=False,
                on_reset=on_reset,
                on_step=on_step,
            )
            trajectory_summaries.append(
                TrajectoryMetadata(
                    trajectory_id=trajectory_id,
                    pairing_id=pairing.identifier,
                    physical_player_assignment=dict(assignment),
                    episode_id=global_episode_id,
                    seed=int(episode_seed),
                    episode_length=result.episode_length,
                    sparse_return=result.sparse_return,
                    shaped_return=result.shaped_return,
                    delivery_timesteps=result.delivery_timesteps,
                    official_score=result.official_score,
                    stopped_by_user=result.stopped_by_user,
                )
            )
            global_episode_id += 1

    buffer = StateBuffer.create(
        environment=build_environment_metadata(env, config.environment),
        source_policies=tuple(source_policies.values()),
        collection_config=deepcopy(config.effective),
        trajectories=tuple(trajectory_summaries),
        records=tuple(records),
    )
    output_path = save_state_buffer(buffer, config.output_path)
    return {
        "status": "complete",
        "output_path": str(output_path),
        "summary": inspect_state_buffer(buffer),
        "trajectories": [trajectory.to_dict() for trajectory in trajectory_summaries],
    }
