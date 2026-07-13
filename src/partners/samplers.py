"""Partner-distribution implementations."""

from __future__ import annotations

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


class SelfPlayPartnerSampler:
    """Stage A sampler selecting the current trainable policy."""

    def sample(
        self,
        rng: np.random.Generator,
        episode_context: dict[str, Any],
    ) -> PartnerSpec:
        del rng, episode_context
        return PartnerSpec(name="self_play", source="current_policy")
