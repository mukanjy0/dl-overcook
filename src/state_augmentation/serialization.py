"""Exact Overcooked state serialization and environment compatibility checks."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from copy import deepcopy
from typing import Any

from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState

STATE_SERIALIZATION_VERSION = 1
ENVIRONMENT_METADATA_VERSION = 2


class StateSerializationError(ValueError):
    """Raised when serialized state data cannot be restored safely."""


class EnvironmentCompatibilityError(ValueError):
    """Raised when a buffer and target environment are incompatible."""


def _json_normalize(value: Any) -> Any:
    """Return the JSON representation used for stable hashes and storage."""
    try:
        return json.loads(json.dumps(value, sort_keys=True))
    except (TypeError, ValueError) as exc:
        raise StateSerializationError(
            f"Value is not JSON serializable: {type(value).__name__}"
        ) from exc


def _overcooked_version() -> str:
    try:
        return importlib.metadata.version("overcooked-ai")
    except importlib.metadata.PackageNotFoundError as exc:
        raise EnvironmentCompatibilityError(
            "The overcooked-ai package is not installed"
        ) from exc


def _environment_definition(env: Any) -> dict[str, Any]:
    mdp = env.mdp
    return _json_normalize(
        {
            "layout": str(mdp.layout_name),
            "terrain": mdp.terrain_mtx,
            "num_players": int(mdp.num_players),
            "start_player_positions": mdp.start_player_positions,
            "start_bonus_orders": mdp.start_bonus_orders,
            "start_all_orders": mdp.start_all_orders,
            "recipe_config": mdp.recipe_config,
            "reward_shaping_params": mdp.reward_shaping_params,
            "order_bonus": mdp.order_bonus,
        }
    )


def environment_fingerprint(env: Any) -> str:
    """Hash the effective layout and transition-relevant environment settings."""
    payload = json.dumps(
        _environment_definition(env),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _dynamics_config(environment_config: dict[str, Any]) -> dict[str, Any]:
    """Keep only configuration that can change transitions for the same layout."""
    return _json_normalize(
        {
            "old_dynamics": bool(environment_config.get("old_dynamics", True)),
            "mdp_overrides": deepcopy(environment_config.get("mdp_overrides", {}) or {}),
        }
    )


def _fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_environment_metadata(
    env: Any,
    environment_config: dict[str, Any],
) -> dict[str, Any]:
    """Describe the exact environment contract used by a state buffer."""
    dynamics_config = _dynamics_config(environment_config)
    return {
        "metadata_version": ENVIRONMENT_METADATA_VERSION,
        "layout": str(env.mdp.layout_name),
        "layout_fingerprint": environment_fingerprint(env),
        "horizon": int(env.horizon),
        "overcooked_ai_version": _overcooked_version(),
        "state_serialization_version": STATE_SERIALIZATION_VERSION,
        "dynamics_config": dynamics_config,
        "dynamics_fingerprint": _fingerprint(dynamics_config),
        "environment_config": _json_normalize(deepcopy(environment_config)),
    }


def validate_environment_metadata(
    metadata: dict[str, Any],
    *,
    env: Any | None = None,
    environment_config: dict[str, Any] | None = None,
) -> None:
    """Validate runtime and, when supplied, exact environment compatibility."""
    if not isinstance(metadata, dict):
        raise EnvironmentCompatibilityError("Buffer environment metadata must be a mapping")
    if metadata.get("metadata_version") != ENVIRONMENT_METADATA_VERSION:
        raise EnvironmentCompatibilityError(
            "Unsupported environment metadata version "
            f"{metadata.get('metadata_version')!r}; expected {ENVIRONMENT_METADATA_VERSION}"
        )
    if metadata.get("state_serialization_version") != STATE_SERIALIZATION_VERSION:
        raise EnvironmentCompatibilityError(
            "Unsupported state serialization version "
            f"{metadata.get('state_serialization_version')!r}; "
            f"expected {STATE_SERIALIZATION_VERSION}"
        )
    recorded_version = str(metadata.get("overcooked_ai_version", ""))
    current_version = _overcooked_version()
    if recorded_version != current_version:
        raise EnvironmentCompatibilityError(
            "Overcooked-AI version mismatch: "
            f"buffer={recorded_version!r}, runtime={current_version!r}"
        )
    if (
        not metadata.get("layout")
        or not metadata.get("layout_fingerprint")
        or not metadata.get("dynamics_fingerprint")
    ):
        raise EnvironmentCompatibilityError(
            "Buffer environment metadata is missing layout or dynamics information"
        )
    if int(metadata.get("horizon", 0)) <= 0:
        raise EnvironmentCompatibilityError("Buffer collection horizon must be positive")
    recorded_dynamics = metadata.get("dynamics_config")
    if not isinstance(recorded_dynamics, dict):
        raise EnvironmentCompatibilityError("Buffer dynamics_config must be a mapping")
    if _fingerprint(recorded_dynamics) != str(metadata["dynamics_fingerprint"]):
        raise EnvironmentCompatibilityError("Buffer dynamics metadata is internally inconsistent")
    if environment_config is not None:
        expected_dynamics = _dynamics_config(environment_config)
        if _fingerprint(expected_dynamics) != str(metadata["dynamics_fingerprint"]):
            raise EnvironmentCompatibilityError(
                "State buffer dynamics configuration does not match the target environment"
            )

    if env is None:
        return
    expected_layout = str(env.mdp.layout_name)
    if str(metadata["layout"]) != expected_layout:
        raise EnvironmentCompatibilityError(
            f"State buffer layout {metadata['layout']!r} does not match "
            f"environment layout {expected_layout!r}"
        )
    expected_fingerprint = environment_fingerprint(env)
    if str(metadata["layout_fingerprint"]) != expected_fingerprint:
        raise EnvironmentCompatibilityError(
            "State buffer environment fingerprint does not match the target environment"
        )


def serialize_state(state: OvercookedState) -> dict[str, Any]:
    """Serialize an upstream state through its canonical public dictionary API."""
    if not isinstance(state, OvercookedState):
        raise StateSerializationError(
            f"Expected OvercookedState, got {type(state).__name__}"
        )
    return _json_normalize(state.to_dict())


def _validate_state_structure(state: OvercookedState, mdp: Any, horizon: int | None) -> None:
    if len(state.players) != int(mdp.num_players):
        raise StateSerializationError(
            f"State has {len(state.players)} players; expected {int(mdp.num_players)}"
        )
    if not isinstance(state.timestep, int) or state.timestep < 0:
        raise StateSerializationError("State timestep must be a non-negative integer")
    if horizon is not None and state.timestep >= int(horizon):
        raise StateSerializationError(
            f"State timestep {state.timestep} must be below horizon {int(horizon)}"
        )

    valid_positions = set(mdp.get_valid_player_positions())
    player_positions = [tuple(player.position) for player in state.players]
    if len(set(player_positions)) != len(player_positions):
        raise StateSerializationError("Players cannot occupy the same position")
    for index, player in enumerate(state.players):
        position = tuple(player.position)
        orientation = tuple(player.orientation)
        if position not in valid_positions:
            raise StateSerializationError(
                f"Player {index} position {position} is invalid for layout {mdp.layout_name}"
            )
        if orientation not in Direction.ALL_DIRECTIONS:
            raise StateSerializationError(
                f"Player {index} orientation {orientation} is invalid"
            )
        if player.has_object() and tuple(player.get_object().position) != position:
            raise StateSerializationError(
                f"Player {index} held-object position does not match the player"
            )

    width = len(mdp.terrain_mtx[0])
    height = len(mdp.terrain_mtx)
    for position, obj in state.objects.items():
        x, y = tuple(position)
        if not (0 <= x < width and 0 <= y < height):
            raise StateSerializationError(f"Object position {position} is outside the layout")
        if tuple(obj.position) != tuple(position):
            raise StateSerializationError("Object mapping key and object position disagree")
        if mdp.terrain_mtx[y][x] == " ":
            raise StateSerializationError(
                f"Unowned object at {position} cannot occupy a walkable tile"
            )

    try:
        mdp.get_state_transition(
            state.deepcopy(),
            tuple(Action.STAY for _ in range(int(mdp.num_players))),
        )
    except Exception as exc:
        raise StateSerializationError(
            f"State is not accepted by the configured environment: {exc!r}"
        ) from exc


def restore_state(
    serialized_state: dict[str, Any],
    mdp: Any,
    *,
    horizon: int | None = None,
) -> OvercookedState:
    """Restore and validate an exact upstream state from JSON-compatible data."""
    if not isinstance(serialized_state, dict):
        raise StateSerializationError("Serialized state must be a mapping")
    normalized = _json_normalize(serialized_state)
    try:
        state = OvercookedState.from_dict(normalized)
    except Exception as exc:
        raise StateSerializationError(f"Malformed serialized state: {exc!r}") from exc
    if serialize_state(state) != normalized:
        raise StateSerializationError(
            "Serialized state does not round-trip through OvercookedState.from_dict"
        )
    _validate_state_structure(state, mdp, horizon)
    return state
