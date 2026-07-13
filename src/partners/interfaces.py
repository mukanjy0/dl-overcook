"""Stable partner-policy construction contracts."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Protocol

from src.policy_loader import build_policy


@dataclass(frozen=True)
class PartnerSpec:
    """A sampled partner description independent of an episode session."""

    name: str
    policy_config: dict[str, Any] | None = None
    source: str = "configured"


class PartnerFactory(Protocol):
    def build(
        self,
        spec: PartnerSpec,
        *,
        env: Any,
        observation_builder: Any,
        player_position: int,
        seed: int,
    ) -> Any:
        """Build a fresh partner session for one rollout."""


class ConfiguredPartnerFactory:
    """Build partners through the existing policy-loading path."""

    def build(
        self,
        spec: PartnerSpec,
        *,
        env: Any,
        observation_builder: Any,
        player_position: int,
        seed: int,
    ) -> Any:
        del player_position
        if spec.policy_config is None:
            raise ValueError(f"Partner '{spec.name}' has no policy configuration")
        return build_policy(
            deepcopy(spec.policy_config),
            env,
            observation_builder,
            seed=seed,
        )
