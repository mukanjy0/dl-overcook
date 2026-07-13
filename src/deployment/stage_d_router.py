"""Layout and physical-position dispatch for existing deployment specialists."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from overcooked_ai_py.agents.agent import Agent


class StageDRoutingError(ValueError):
    """Raised when a Stage D deployment route cannot be resolved safely."""


@dataclass(frozen=True)
class SpecialistSelection:
    """One selected specialist and the route that chose it."""

    specialist_id: str
    route_layout: str
    physical_position: int
    policy: dict[str, Any]
    artifact_sha256: str | None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_policy_paths(policy: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    resolved = deepcopy(policy)
    if resolved.get("path"):
        path = Path(str(resolved["path"])).expanduser()
        resolved["path"] = str(path if path.is_absolute() else base_dir / path)
    runtime = resolved.get("config", {}) or {}
    if not isinstance(runtime, dict):
        raise StageDRoutingError("Specialist policy.config must be a mapping")
    for key in ("checkpoint_path", "model_path"):
        if runtime.get(key):
            path = Path(str(runtime[key])).expanduser()
            runtime[key] = str(path if path.is_absolute() else base_dir / path)
    return resolved


@lru_cache(maxsize=16)
def _load_mapping_cached(mapping_path: str) -> dict[str, Any]:
    path = Path(mapping_path).expanduser().resolve()
    if not path.is_file():
        raise StageDRoutingError(f"Stage D specialist mapping is missing: {path}")
    with path.open("r", encoding="utf-8") as stream:
        mapping = yaml.safe_load(stream)
    if not isinstance(mapping, dict):
        raise StageDRoutingError("Stage D specialist mapping must be a YAML mapping")
    if not isinstance(mapping.get("fallback"), dict):
        raise StageDRoutingError("Stage D specialist mapping requires a fallback")
    if not isinstance(mapping.get("layouts"), dict):
        raise StageDRoutingError("Stage D specialist mapping requires layouts")
    return mapping


def _entry_to_selection(
    entry: dict[str, Any],
    *,
    route_layout: str,
    physical_position: int,
    base_dir: Path,
) -> SpecialistSelection:
    specialist_id = entry.get("id")
    policy = entry.get("policy")
    if not isinstance(specialist_id, str) or not specialist_id:
        raise StageDRoutingError("Each Stage D specialist entry requires a non-empty id")
    if not isinstance(policy, dict):
        raise StageDRoutingError(f"Stage D specialist '{specialist_id}' requires policy")
    artifact_sha256 = entry.get("artifact_sha256")
    if artifact_sha256 is not None and (
        not isinstance(artifact_sha256, str) or len(artifact_sha256) != 64
    ):
        raise StageDRoutingError(
            f"Stage D specialist '{specialist_id}' has an invalid artifact SHA-256"
        )
    return SpecialistSelection(
        specialist_id=specialist_id,
        route_layout=route_layout,
        physical_position=physical_position,
        policy=_resolve_policy_paths(policy, base_dir),
        artifact_sha256=artifact_sha256,
    )


def select_specialist(
    mapping_path: str | Path,
    layout_name: str,
    physical_position: int,
) -> SpecialistSelection:
    """Resolve a configured specialist or the explicit unknown-layout fallback."""
    if physical_position not in (0, 1):
        raise StageDRoutingError("Stage D physical position must be 0 or 1")
    path = Path(mapping_path).expanduser().resolve()
    mapping = _load_mapping_cached(str(path))
    layout_entry = mapping["layouts"].get(layout_name)
    if layout_entry is None:
        return _entry_to_selection(
            mapping["fallback"],
            route_layout="fallback",
            physical_position=physical_position,
            base_dir=path.parent,
        )
    if not isinstance(layout_entry, dict):
        raise StageDRoutingError(f"Stage D layout entry must be a mapping: {layout_name}")
    positions = layout_entry.get("positions")
    if not isinstance(positions, dict):
        raise StageDRoutingError(f"Stage D layout '{layout_name}' requires positions")
    entry = positions.get(str(physical_position))
    if not isinstance(entry, dict):
        raise StageDRoutingError(
            f"Stage D layout '{layout_name}' has no policy for position {physical_position}"
        )
    return _entry_to_selection(
        entry,
        route_layout=layout_name,
        physical_position=physical_position,
        base_dir=path.parent,
    )


class StageDDeploymentRouter(Agent):
    """Lazily construct and reuse the specialist selected for this Agent instance."""

    def __init__(
        self,
        *,
        layout_name: str,
        mapping_path: str | Path,
        policy_builder: Callable[[dict[str, Any]], Agent],
    ):
        self.layout_name = str(layout_name)
        self.mapping_path = Path(mapping_path).expanduser().resolve()
        self.policy_builder = policy_builder
        self._cached_policies: dict[str, Agent] = {}
        super().__init__()

    def _selection(self) -> SpecialistSelection:
        return select_specialist(
            self.mapping_path,
            self.layout_name,
            int(self.agent_index),
        )

    @property
    def selected_specialist(self) -> SpecialistSelection:
        """Expose the current route without constructing its policy."""
        return self._selection()

    def _validate_artifact(self, selection: SpecialistSelection) -> None:
        checkpoint_path = (selection.policy.get("config", {}) or {}).get(
            "checkpoint_path"
        )
        if checkpoint_path is None:
            return
        path = Path(str(checkpoint_path))
        if not path.is_file():
            raise StageDRoutingError(
                f"Stage D specialist '{selection.specialist_id}' checkpoint is missing: {path}"
            )
        if selection.artifact_sha256 is None:
            raise StageDRoutingError(
                f"Stage D specialist '{selection.specialist_id}' lacks artifact_sha256"
            )
        actual_sha256 = _sha256(path)
        if actual_sha256 != selection.artifact_sha256:
            raise StageDRoutingError(
                f"Stage D specialist '{selection.specialist_id}' checkpoint hash mismatch: "
                f"expected {selection.artifact_sha256}, got {actual_sha256}"
            )

    def _policy_for_selection(self, selection: SpecialistSelection) -> Agent:
        policy = self._cached_policies.get(selection.specialist_id)
        if policy is None:
            self._validate_artifact(selection)
            policy = self.policy_builder(deepcopy(selection.policy))
            self._cached_policies[selection.specialist_id] = policy
        policy.set_agent_index(self.agent_index)
        policy.set_mdp(self.mdp)
        return policy

    def reset(self) -> None:
        super().reset()
        for policy in self._cached_policies.values():
            policy.reset()

    def set_agent_index(self, agent_index):
        super().set_agent_index(agent_index)
        for policy in self._cached_policies.values():
            policy.set_agent_index(agent_index)

    def set_mdp(self, mdp):
        super().set_mdp(mdp)
        for policy in self._cached_policies.values():
            policy.set_mdp(mdp)

    def action(self, state):
        selection = self._selection()
        action, info = self._policy_for_selection(selection).action(state)
        routed_info = dict(info or {})
        routed_info.update(
            {
                "stage_d_layout": self.layout_name,
                "stage_d_route_layout": selection.route_layout,
                "stage_d_ego_index": int(self.agent_index),
                "stage_d_specialist": selection.specialist_id,
            }
        )
        return action, routed_info
