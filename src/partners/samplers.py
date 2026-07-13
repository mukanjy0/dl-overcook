"""Partner and physical-position sampling implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from src.partners.interfaces import PartnerSpec


class PartnerSampler(Protocol):
    def sample(
        self,
        rng: np.random.Generator,
        episode_context: dict[str, Any],
    ) -> PartnerSpec:
        """Select a partner for one episode."""


class EgoPositionSampler(Protocol):
    def sample(
        self,
        rng: np.random.Generator,
        episode_context: dict[str, Any],
    ) -> int:
        """Select the ego policy's physical player position for one episode."""


class SelfPlayPartnerSampler:
    """Stage A sampler selecting the current trainable policy."""

    def sample(
        self,
        rng: np.random.Generator,
        episode_context: dict[str, Any],
    ) -> PartnerSpec:
        del rng, episode_context
        return PartnerSpec(name="self_play", source="current_policy")


@dataclass(frozen=True)
class WeightedPartner:
    """One immutable partner-pool entry and its positive sampling weight."""

    spec: PartnerSpec
    weight: float


class WeightedPartnerSampler:
    """Sample frozen configured partners according to normalized weights."""

    def __init__(self, partners: tuple[WeightedPartner, ...]):
        if not partners:
            raise ValueError("A weighted partner pool must not be empty")
        weights = np.asarray([entry.weight for entry in partners], dtype=np.float64)
        if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
            raise ValueError("Partner weights must be finite and positive")
        self.partners = tuple(partners)
        self.probabilities = weights / weights.sum()

    def sample(
        self,
        rng: np.random.Generator,
        episode_context: dict[str, Any],
    ) -> PartnerSpec:
        del episode_context
        index = int(rng.choice(len(self.partners), p=self.probabilities))
        return self.partners[index].spec


class ExactPartnerSampler:
    """Always select one configured partner for best-response fine-tuning."""

    def __init__(self, partner: PartnerSpec):
        self.partner = partner

    def sample(
        self,
        rng: np.random.Generator,
        episode_context: dict[str, Any],
    ) -> PartnerSpec:
        del rng, episode_context
        return self.partner


class BalancedEgoPositionSampler:
    """Alternate positions with a seeded random first position."""

    def __init__(self) -> None:
        self._next_position: int | None = None

    def sample(
        self,
        rng: np.random.Generator,
        episode_context: dict[str, Any],
    ) -> int:
        del episode_context
        if self._next_position is None:
            self._next_position = int(rng.integers(0, 2))
        position = self._next_position
        self._next_position = 1 - position
        return position


def _configured_partners(
    partner_config: dict[str, Any],
) -> tuple[WeightedPartner, ...]:
    raw_entries = partner_config.get("policies", [])
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError(
            "Configured partner sampling requires a non-empty partner.policies list"
        )

    entries: list[WeightedPartner] = []
    names: set[str] = set()
    for index, raw_entry in enumerate(raw_entries):
        if not isinstance(raw_entry, dict):
            raise ValueError(f"partner.policies[{index}] must be a mapping")
        name = str(raw_entry.get("name", "")).strip()
        if not name:
            raise ValueError(f"partner.policies[{index}].name is required")
        if name in names:
            raise ValueError(f"Partner names must be unique; duplicate '{name}'")
        policy_config = raw_entry.get("policy")
        if not isinstance(policy_config, dict):
            raise ValueError(f"Partner '{name}' requires a policy mapping")
        observation_config = raw_entry.get("observation")
        if observation_config is not None and not isinstance(observation_config, dict):
            raise ValueError(f"Partner '{name}' observation must be a mapping")
        for probability_key in ("sticky_action_prob", "random_action_prob"):
            probability = float(policy_config.get(probability_key, 0.0) or 0.0)
            if not 0.0 <= probability <= 1.0:
                raise ValueError(
                    f"Partner '{name}' {probability_key} must be in [0, 1]"
                )
        weight = float(raw_entry.get("weight", 1.0))
        entries.append(
            WeightedPartner(
                spec=PartnerSpec(
                    name=name,
                    policy_config=dict(policy_config),
                    observation_config=(
                        None if observation_config is None else dict(observation_config)
                    ),
                    source=str(raw_entry.get("source", "configured")),
                ),
                weight=weight,
            )
        )
        names.add(name)
    return tuple(entries)


def build_partner_sampler(partner_config: dict[str, Any]) -> PartnerSampler:
    """Validate partner configuration and build its episode sampler."""
    sampler_name = str(partner_config.get("sampler", "self_play")).lower()
    if sampler_name == "self_play":
        return SelfPlayPartnerSampler()

    entries = _configured_partners(partner_config)
    if sampler_name == "weighted_pool":
        return WeightedPartnerSampler(entries)
    if sampler_name == "exact":
        exact_name = str(partner_config.get("exact_partner", "")).strip()
        if not exact_name:
            raise ValueError("partner.exact_partner is required for exact sampling")
        for entry in entries:
            if entry.spec.name == exact_name:
                return ExactPartnerSampler(entry.spec)
        raise ValueError(
            f"partner.exact_partner '{exact_name}' is not present in partner.policies"
        )
    raise ValueError("partner.sampler must be self_play, weighted_pool, or exact")


def build_ego_position_sampler(partner_config: dict[str, Any]) -> EgoPositionSampler:
    """Build the Stage C position sampler after validating its configured mode."""
    sampler_name = str(partner_config.get("position_sampler", "balanced")).lower()
    if sampler_name != "balanced":
        raise ValueError("Stage C partner.position_sampler must be 'balanced'")
    return BalancedEgoPositionSampler()
