"""Versioned state-buffer storage, inspection, and compatibility validation."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.environment import build_env
from src.state_augmentation.serialization import (
    EnvironmentCompatibilityError,
    StateSerializationError,
    restore_state,
    validate_environment_metadata,
)

STATE_BUFFER_SCHEMA_VERSION = 2
STATE_BUFFER_PROFILE = "overcooked_state_buffer"


class StateBufferCompatibilityError(ValueError):
    """Raised when a buffer cannot safely initialize the requested environment."""


@dataclass(frozen=True)
class SourcePolicyMetadata:
    """Reproducibility metadata for one trajectory-producing policy."""

    identifier: str
    source: str
    policy_type: str
    policy_name: str
    checkpoint_path: str | None
    checkpoint_sha256: str | None
    checkpoint_identity: str | None
    policy_config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "identifier": self.identifier,
            "source": self.source,
            "policy_type": self.policy_type,
            "policy_name": self.policy_name,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_identity": self.checkpoint_identity,
            "policy_config": self.policy_config,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "SourcePolicyMetadata":
        if not isinstance(values, dict):
            raise StateBufferCompatibilityError("Source policy metadata must be a mapping")
        identifier = str(values.get("identifier", "")).strip()
        if not identifier:
            raise StateBufferCompatibilityError("Source policy identifier is required")
        policy_config = values.get("policy_config")
        if not isinstance(policy_config, dict):
            raise StateBufferCompatibilityError(
                f"Source policy '{identifier}' policy_config must be a mapping"
            )
        return cls(
            identifier=identifier,
            source=str(values.get("source", "configured")),
            policy_type=str(values.get("policy_type", "")),
            policy_name=str(values.get("policy_name", "")),
            checkpoint_path=(
                None
                if values.get("checkpoint_path") in (None, "")
                else str(values["checkpoint_path"])
            ),
            checkpoint_sha256=(
                None
                if values.get("checkpoint_sha256") in (None, "")
                else str(values["checkpoint_sha256"])
            ),
            checkpoint_identity=(
                None
                if values.get("checkpoint_identity") in (None, "")
                else str(values["checkpoint_identity"])
            ),
            policy_config=policy_config,
        )


@dataclass(frozen=True)
class StateRecord:
    """One restorable trajectory state and its physical collection context."""

    record_id: str
    layout: str
    physical_player_assignment: dict[str, str]
    episode_id: int
    trajectory_id: str
    timestep: int
    seed: int
    serialized_state: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "layout": self.layout,
            "physical_player_assignment": self.physical_player_assignment,
            "episode_id": self.episode_id,
            "trajectory_id": self.trajectory_id,
            "timestep": self.timestep,
            "seed": self.seed,
            "serialized_state": self.serialized_state,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "StateRecord":
        if not isinstance(values, dict):
            raise StateBufferCompatibilityError("Each state record must be a mapping")
        assignment = values.get("physical_player_assignment")
        if not isinstance(assignment, dict):
            raise StateBufferCompatibilityError(
                "physical_player_assignment must be a mapping"
            )
        serialized_state = values.get("serialized_state")
        if not isinstance(serialized_state, dict):
            raise StateBufferCompatibilityError("serialized_state must be a mapping")
        try:
            return cls(
                record_id=str(values["record_id"]),
                layout=str(values["layout"]),
                physical_player_assignment={
                    str(key): str(value) for key, value in assignment.items()
                },
                episode_id=int(values["episode_id"]),
                trajectory_id=str(values["trajectory_id"]),
                timestep=int(values["timestep"]),
                seed=int(values["seed"]),
                serialized_state=serialized_state,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise StateBufferCompatibilityError(
                f"Malformed state-record metadata: {exc!r}"
            ) from exc


@dataclass(frozen=True)
class TrajectoryMetadata:
    """Outcome and provenance for one completed collection trajectory."""

    trajectory_id: str
    pairing_id: str
    physical_player_assignment: dict[str, str]
    episode_id: int
    seed: int
    episode_length: int
    sparse_return: float
    shaped_return: float
    delivery_timesteps: tuple[int, ...]
    official_score: int
    stopped_by_user: bool = False

    @property
    def successful(self) -> bool:
        return bool(self.delivery_timesteps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "pairing_id": self.pairing_id,
            "physical_player_assignment": self.physical_player_assignment,
            "episode_id": self.episode_id,
            "seed": self.seed,
            "episode_length": self.episode_length,
            "sparse_return": self.sparse_return,
            "shaped_return": self.shaped_return,
            "delivery_timesteps": list(self.delivery_timesteps),
            "official_score": self.official_score,
            "stopped_by_user": self.stopped_by_user,
            "successful": self.successful,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "TrajectoryMetadata":
        if not isinstance(values, dict):
            raise StateBufferCompatibilityError("Trajectory metadata must be a mapping")
        assignment = values.get("physical_player_assignment")
        deliveries = values.get("delivery_timesteps")
        if not isinstance(assignment, dict):
            raise StateBufferCompatibilityError(
                "Trajectory physical_player_assignment must be a mapping"
            )
        if not isinstance(deliveries, list):
            raise StateBufferCompatibilityError(
                "Trajectory delivery_timesteps must be a list"
            )
        try:
            trajectory = cls(
                trajectory_id=str(values["trajectory_id"]),
                pairing_id=str(values["pairing_id"]),
                physical_player_assignment={
                    str(key): str(value) for key, value in assignment.items()
                },
                episode_id=int(values["episode_id"]),
                seed=int(values["seed"]),
                episode_length=int(values["episode_length"]),
                sparse_return=float(values["sparse_return"]),
                shaped_return=float(values["shaped_return"]),
                delivery_timesteps=tuple(int(value) for value in deliveries),
                official_score=int(values["official_score"]),
                stopped_by_user=bool(values.get("stopped_by_user", False)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise StateBufferCompatibilityError(
                f"Malformed trajectory metadata: {exc!r}"
            ) from exc
        if "successful" in values and bool(values["successful"]) != trajectory.successful:
            raise StateBufferCompatibilityError(
                f"Trajectory {trajectory.trajectory_id!r} has inconsistent success metadata"
            )
        return trajectory


@dataclass(frozen=True)
class StateBuffer:
    """In-memory representation of one versioned state-buffer artifact."""

    environment: dict[str, Any]
    source_policies: tuple[SourcePolicyMetadata, ...]
    collection_config: dict[str, Any]
    trajectories: tuple[TrajectoryMetadata, ...]
    records: tuple[StateRecord, ...]
    created_at_utc: str
    schema_version: int = STATE_BUFFER_SCHEMA_VERSION
    profile: str = STATE_BUFFER_PROFILE

    @classmethod
    def create(
        cls,
        *,
        environment: dict[str, Any],
        source_policies: tuple[SourcePolicyMetadata, ...],
        collection_config: dict[str, Any],
        trajectories: tuple[TrajectoryMetadata, ...],
        records: tuple[StateRecord, ...],
    ) -> "StateBuffer":
        return cls(
            environment=environment,
            source_policies=source_policies,
            collection_config=collection_config,
            trajectories=trajectories,
            records=records,
            created_at_utc=datetime.now(timezone.utc).isoformat(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "profile": self.profile,
            "created_at_utc": self.created_at_utc,
            "environment": self.environment,
            "source_policies": [policy.to_dict() for policy in self.source_policies],
            "collection_config": self.collection_config,
            "trajectories": [trajectory.to_dict() for trajectory in self.trajectories],
            "records": [record.to_dict() for record in self.records],
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "StateBuffer":
        if not isinstance(values, dict):
            raise StateBufferCompatibilityError("State buffer payload must be a mapping")
        if values.get("schema_version") != STATE_BUFFER_SCHEMA_VERSION:
            raise StateBufferCompatibilityError(
                f"Unsupported state-buffer schema {values.get('schema_version')!r}; "
                f"expected {STATE_BUFFER_SCHEMA_VERSION}"
            )
        if values.get("profile") != STATE_BUFFER_PROFILE:
            raise StateBufferCompatibilityError(
                f"Expected profile {STATE_BUFFER_PROFILE!r}, got {values.get('profile')!r}"
            )
        environment = values.get("environment")
        collection_config = values.get("collection_config")
        source_policies = values.get("source_policies")
        trajectories = values.get("trajectories")
        records = values.get("records")
        if not isinstance(environment, dict):
            raise StateBufferCompatibilityError("Buffer environment must be a mapping")
        if not isinstance(collection_config, dict):
            raise StateBufferCompatibilityError("collection_config must be a mapping")
        if not isinstance(source_policies, list):
            raise StateBufferCompatibilityError("source_policies must be a list")
        if not isinstance(trajectories, list):
            raise StateBufferCompatibilityError("trajectories must be a list")
        if not isinstance(records, list):
            raise StateBufferCompatibilityError("records must be a list")
        return cls(
            environment=environment,
            source_policies=tuple(
                SourcePolicyMetadata.from_dict(item) for item in source_policies
            ),
            collection_config=collection_config,
            trajectories=tuple(
                TrajectoryMetadata.from_dict(item) for item in trajectories
            ),
            records=tuple(StateRecord.from_dict(item) for item in records),
            created_at_utc=str(values.get("created_at_utc", "")),
            schema_version=int(values["schema_version"]),
            profile=str(values["profile"]),
        )


def _open_json(path: Path) -> dict[str, Any]:
    try:
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                payload = json.load(stream)
        else:
            with path.open("r", encoding="utf-8") as stream:
                payload = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise StateBufferCompatibilityError(
            f"Could not read state buffer {path}: {exc!r}"
        ) from exc
    if not isinstance(payload, dict):
        raise StateBufferCompatibilityError("State buffer root must be a mapping")
    return payload


def validate_state_buffer(
    buffer: StateBuffer,
    *,
    env: Any | None = None,
    environment_config: dict[str, Any] | None = None,
) -> None:
    """Validate metadata, source references, and every serialized state."""
    try:
        if env is None:
            config = environment_config or buffer.environment.get("environment_config")
            if not isinstance(config, dict):
                raise StateBufferCompatibilityError(
                    "An environment configuration is required to validate states"
                )
            env = build_env(config)
        validate_environment_metadata(
            buffer.environment,
            env=env,
            environment_config=environment_config,
        )
    except (EnvironmentCompatibilityError, OSError, ValueError) as exc:
        if isinstance(exc, StateBufferCompatibilityError):
            raise
        raise StateBufferCompatibilityError(str(exc)) from exc

    policy_ids = [policy.identifier for policy in buffer.source_policies]
    if not policy_ids or len(set(policy_ids)) != len(policy_ids):
        raise StateBufferCompatibilityError(
            "source_policies must contain unique policy identifiers"
        )
    if not buffer.records:
        raise StateBufferCompatibilityError("State buffer must contain at least one state")

    for policy in buffer.source_policies:
        if policy.checkpoint_sha256 is not None:
            digest = policy.checkpoint_sha256.lower()
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise StateBufferCompatibilityError(
                    f"Source policy {policy.identifier!r} has an invalid checkpoint SHA-256"
                )
            expected_identity = f"sha256:{digest}"
            if policy.checkpoint_identity != expected_identity:
                raise StateBufferCompatibilityError(
                    f"Source policy {policy.identifier!r} checkpoint identity must be "
                    f"{expected_identity!r}"
                )
        elif policy.checkpoint_identity is not None:
            raise StateBufferCompatibilityError(
                f"Source policy {policy.identifier!r} has a checkpoint identity without a hash"
            )
        if policy.source == "frozen_checkpoint" and policy.checkpoint_sha256 is None:
            raise StateBufferCompatibilityError(
                f"Frozen checkpoint source {policy.identifier!r} requires a readable "
                "checkpoint and SHA-256 identity during collection"
            )

    trajectories_by_id: dict[str, TrajectoryMetadata] = {}
    for trajectory in buffer.trajectories:
        if not trajectory.trajectory_id or trajectory.trajectory_id in trajectories_by_id:
            raise StateBufferCompatibilityError(
                "Trajectory identifiers must be non-empty and unique"
            )
        if not trajectory.pairing_id:
            raise StateBufferCompatibilityError(
                f"Trajectory {trajectory.trajectory_id!r} has no pairing id"
            )
        if set(trajectory.physical_player_assignment) != {"0", "1"}:
            raise StateBufferCompatibilityError(
                f"Trajectory {trajectory.trajectory_id!r} must assign players 0 and 1"
            )
        unknown = set(trajectory.physical_player_assignment.values()) - set(policy_ids)
        if unknown:
            raise StateBufferCompatibilityError(
                f"Trajectory {trajectory.trajectory_id!r} references unknown policies "
                f"{sorted(unknown)}"
            )
        if trajectory.episode_length < 0:
            raise StateBufferCompatibilityError("Trajectory episode_length cannot be negative")
        if any(timestep < 0 for timestep in trajectory.delivery_timesteps):
            raise StateBufferCompatibilityError("Delivery timesteps cannot be negative")
        trajectories_by_id[trajectory.trajectory_id] = trajectory
    if not trajectories_by_id:
        raise StateBufferCompatibilityError(
            "State buffer must contain completed trajectory metadata"
        )

    record_ids: set[str] = set()
    expected_layout = str(buffer.environment["layout"])
    for record in buffer.records:
        if not record.record_id or record.record_id in record_ids:
            raise StateBufferCompatibilityError(
                f"State record identifiers must be non-empty and unique: {record.record_id!r}"
            )
        record_ids.add(record.record_id)
        if record.layout != expected_layout:
            raise StateBufferCompatibilityError(
                f"Record {record.record_id} layout {record.layout!r} does not match "
                f"buffer layout {expected_layout!r}"
            )
        if set(record.physical_player_assignment) != {"0", "1"}:
            raise StateBufferCompatibilityError(
                f"Record {record.record_id} must assign policies to players 0 and 1"
            )
        unknown = set(record.physical_player_assignment.values()) - set(policy_ids)
        if unknown:
            raise StateBufferCompatibilityError(
                f"Record {record.record_id} references unknown policies {sorted(unknown)}"
            )
        trajectory = trajectories_by_id.get(record.trajectory_id)
        if trajectory is None:
            raise StateBufferCompatibilityError(
                f"Record {record.record_id} references unknown trajectory "
                f"{record.trajectory_id!r}"
            )
        if (
            record.episode_id != trajectory.episode_id
            or record.seed != trajectory.seed
            or record.physical_player_assignment
            != trajectory.physical_player_assignment
        ):
            raise StateBufferCompatibilityError(
                f"Record {record.record_id} disagrees with its trajectory metadata"
            )
        if record.timestep != record.serialized_state.get("timestep"):
            raise StateBufferCompatibilityError(
                f"Record {record.record_id} timestep metadata disagrees with its state"
            )
        try:
            restore_state(
                record.serialized_state,
                env.mdp,
                horizon=int(env.horizon),
            )
        except StateSerializationError as exc:
            raise StateBufferCompatibilityError(
                f"Record {record.record_id} is malformed: {exc}"
            ) from exc


def save_state_buffer(buffer: StateBuffer, path: str | Path) -> Path:
    """Atomically write a validated JSON or JSON.GZ state buffer."""
    validate_state_buffer(buffer)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        if destination.suffix == ".gz":
            with gzip.open(temporary_path, "wt", encoding="utf-8") as stream:
                json.dump(buffer.to_dict(), stream, sort_keys=True, separators=(",", ":"))
        else:
            with temporary_path.open("w", encoding="utf-8") as stream:
                json.dump(buffer.to_dict(), stream, sort_keys=True, separators=(",", ":"))
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return destination


def load_state_buffer(
    path: str | Path,
    *,
    env: Any | None = None,
    environment_config: dict[str, Any] | None = None,
    validate: bool = True,
) -> StateBuffer:
    """Read a state buffer and reject unsupported or incompatible data."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"State buffer not found: {source}")
    buffer = StateBuffer.from_dict(_open_json(source))
    if validate:
        validate_state_buffer(
            buffer,
            env=env,
            environment_config=environment_config,
        )
    return buffer


def _digest_state(state: dict[str, Any], *, include_timestep: bool) -> str:
    normalized = dict(state)
    if not include_timestep:
        normalized.pop("timestep", None)
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _percentile(values: list[int], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _task_progress_statistics(buffer: StateBuffer) -> dict[str, Any]:
    environment_config = buffer.environment.get("environment_config")
    terrain = None
    if isinstance(environment_config, dict):
        terrain = build_env(environment_config).mdp.terrain_mtx
    counts: Counter[str] = Counter()
    soup_ingredient_counts: list[int] = []
    stage_counts: Counter[str] = Counter()
    for record in buffer.records:
        state = record.serialized_state
        held_objects = [
            player.get("held_object")
            for player in state.get("players", [])
            if player.get("held_object") is not None
        ]
        unowned_objects = list(state.get("objects", []))
        all_objects = held_objects + unowned_objects
        soups = [obj for obj in all_objects if obj.get("name") == "soup"]
        counter_objects = []
        if terrain is not None:
            for obj in unowned_objects:
                x, y = obj.get("position", (-1, -1))
                if 0 <= y < len(terrain) and 0 <= x < len(terrain[y]):
                    if terrain[y][x] == "X":
                        counter_objects.append(obj)
        cooking_soups = [obj for obj in soups if bool(obj.get("is_cooking", False))]
        ready_soups = [obj for obj in soups if bool(obj.get("is_ready", False))]
        held_ready_soups = [
            obj
            for obj in held_objects
            if obj.get("name") == "soup" and bool(obj.get("is_ready", False))
        ]
        for soup in soups:
            soup_ingredient_counts.append(len(soup.get("_ingredients", [])))
        counts["states_with_held_object"] += int(bool(held_objects))
        counts["states_with_counter_object"] += int(bool(counter_objects))
        counts["states_with_soup"] += int(bool(soups))
        counts["states_with_cooking_soup"] += int(bool(cooking_soups))
        counts["states_with_ready_soup"] += int(bool(ready_soups))
        counts["states_with_held_ready_soup"] += int(bool(held_ready_soups))
        counts["states_with_dish"] += int(
            any(obj.get("name") == "dish" for obj in all_objects)
        )
        if held_ready_soups:
            stage = "near_delivery"
        elif ready_soups:
            stage = "ready"
        elif cooking_soups:
            stage = "cooking"
        elif soups or held_objects or counter_objects:
            stage = "assembly"
        else:
            stage = "empty"
        stage_counts[stage] += 1
    total = len(buffer.records)
    return {
        **dict(counts),
        "state_stage_counts": dict(stage_counts),
        "state_stage_fractions": {
            key: value / total for key, value in sorted(stage_counts.items())
        },
        "mean_soup_ingredients": (
            sum(soup_ingredient_counts) / len(soup_ingredient_counts)
            if soup_ingredient_counts
            else None
        ),
    }


def inspect_state_buffer(buffer: StateBuffer) -> dict[str, Any]:
    """Return detailed JSON-compatible composition and outcome statistics."""
    timesteps = [record.timestep for record in buffer.records]
    trajectories = {record.trajectory_id for record in buffer.records}
    assignments: Counter[str] = Counter()
    trajectory_assignments: Counter[str] = Counter()
    trajectory_by_id = {
        trajectory.trajectory_id: trajectory for trajectory in buffer.trajectories
    }
    pairing_state_counts: Counter[str] = Counter()
    for record in buffer.records:
        key = f"{record.physical_player_assignment['0']}|{record.physical_player_assignment['1']}"
        assignments[key] += 1
        pairing_state_counts[trajectory_by_id[record.trajectory_id].pairing_id] += 1
    for trajectory in buffer.trajectories:
        key = (
            f"{trajectory.physical_player_assignment['0']}|"
            f"{trajectory.physical_player_assignment['1']}"
        )
        trajectory_assignments[key] += 1
    exact_hashes = Counter(
        _digest_state(record.serialized_state, include_timestep=True)
        for record in buffer.records
    )
    physical_hashes = Counter(
        _digest_state(record.serialized_state, include_timestep=False)
        for record in buffer.records
    )
    successful = [trajectory for trajectory in buffer.trajectories if trajectory.successful]
    failed = [trajectory for trajectory in buffer.trajectories if not trajectory.successful]
    outcome_by_pairing: dict[str, dict[str, int]] = {}
    for trajectory in buffer.trajectories:
        outcome = outcome_by_pairing.setdefault(
            trajectory.pairing_id,
            {"trajectories": 0, "successful": 0, "failed": 0, "deliveries": 0},
        )
        outcome["trajectories"] += 1
        outcome["successful" if trajectory.successful else "failed"] += 1
        outcome["deliveries"] += len(trajectory.delivery_timesteps)
    source_horizon = int(buffer.environment["horizon"])
    early_end = source_horizon / 4
    late_start = 3 * source_horizon / 4
    timestep_regions = {
        "early_0_25pct": sum(timestep < early_end for timestep in timesteps),
        "middle_25_75pct": sum(
            early_end <= timestep < late_start for timestep in timesteps
        ),
        "late_75_100pct": sum(timestep >= late_start for timestep in timesteps),
    }
    assignment_values = list(assignments.values())
    return {
        "schema_version": buffer.schema_version,
        "profile": buffer.profile,
        "created_at_utc": buffer.created_at_utc,
        "layout": buffer.environment.get("layout"),
        "layout_fingerprint": buffer.environment.get("layout_fingerprint"),
        "overcooked_ai_version": buffer.environment.get("overcooked_ai_version"),
        "num_states": len(buffer.records),
        "num_trajectories": len(trajectories),
        "duplicate_statistics": {
            "exact_unique_states": len(exact_hashes),
            "exact_duplicate_records": len(buffer.records) - len(exact_hashes),
            "exact_duplicate_groups": sum(count > 1 for count in exact_hashes.values()),
            "physical_unique_states_ignoring_timestep": len(physical_hashes),
            "physical_duplicate_records_ignoring_timestep": (
                len(buffer.records) - len(physical_hashes)
            ),
        },
        "timestep_distribution": {
            "min": min(timesteps) if timesteps else None,
            "max": max(timesteps) if timesteps else None,
            "mean": sum(timesteps) / len(timesteps) if timesteps else None,
            "p25": _percentile(timesteps, 0.25),
            "median": _percentile(timesteps, 0.5),
            "p75": _percentile(timesteps, 0.75),
            "regions": timestep_regions,
        },
        "source_policies": [policy.to_dict() for policy in buffer.source_policies],
        "source_pairing_balance": {
            "state_counts_by_physical_assignment": dict(assignments),
            "trajectory_counts_by_physical_assignment": dict(trajectory_assignments),
            "state_counts_by_pairing_id": dict(pairing_state_counts),
            "max_to_min_state_count_ratio": (
                max(assignment_values) / min(assignment_values)
                if assignment_values and min(assignment_values) > 0
                else None
            ),
        },
        "task_progress": _task_progress_statistics(buffer),
        "trajectory_outcomes": {
            "successful": len(successful),
            "failed": len(failed),
            "stopped_by_user": sum(
                trajectory.stopped_by_user for trajectory in buffer.trajectories
            ),
            "success_rate": len(successful) / len(buffer.trajectories),
            "deliveries": sum(
                len(trajectory.delivery_timesteps)
                for trajectory in buffer.trajectories
            ),
            "mean_sparse_return": sum(
                trajectory.sparse_return for trajectory in buffer.trajectories
            )
            / len(buffer.trajectories),
            "by_pairing_id": outcome_by_pairing,
        },
        "collection": buffer.collection_config.get("collection", {}),
    }
