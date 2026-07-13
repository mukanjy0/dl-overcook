"""Stable state-initialization extension point for training environments."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class StateSource(Protocol):
    def sample(self, mdp: Any, rng: np.random.Generator) -> Any | None:
        """Return an initial OvercookedState or None for the standard state."""


class StandardStateSource:
    """Stage A source that always requests the upstream standard start state."""

    def sample(self, mdp: Any, rng: np.random.Generator) -> None:
        del mdp, rng
        return None


def build_start_state_fn(source: StateSource | None, mdp: Any, rng: np.random.Generator):
    """Adapt a StateSource to OvercookedEnv's start_state_fn contract."""
    if source is None or isinstance(source, StandardStateSource):
        return None

    def start_state_fn():
        state = source.sample(mdp, rng)
        if state is None:
            return mdp.get_standard_start_state()
        return state.deepcopy() if hasattr(state, "deepcopy") else state

    return start_state_fn
