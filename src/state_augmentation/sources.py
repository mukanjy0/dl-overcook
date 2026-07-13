"""StateSource implementations backed by validated state buffers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.state_augmentation.buffer import StateBuffer, load_state_buffer
from src.state_augmentation.sampling import StateBufferSampler
from src.state_augmentation.serialization import restore_state
from src.state_initialization import StandardStateSource, StateSource


class BufferedStateSource:
    """Choose standard or uniformly sampled augmented starts at reset time."""

    def __init__(self, buffer: StateBuffer, *, augmented_probability: float):
        probability = float(augmented_probability)
        if not 0.0 <= probability <= 1.0:
            raise ValueError("augmented_probability must be in [0, 1]")
        self.buffer = buffer
        self.augmented_probability = probability
        self.sampler = StateBufferSampler(buffer)
        self.standard_resets = 0
        self.augmented_resets = 0

    def sample(self, mdp: Any, rng: np.random.Generator) -> Any | None:
        if self.augmented_probability < 1.0 and rng.random() >= self.augmented_probability:
            self.standard_resets += 1
            return None
        record = self.sampler.sample(rng)
        if record.layout != str(mdp.layout_name):
            raise ValueError(
                f"Sampled state layout {record.layout!r} does not match {mdp.layout_name!r}"
            )
        self.augmented_resets += 1
        return restore_state(
            record.serialized_state,
            mdp,
            horizon=int(self.buffer.environment["horizon"]),
        )

    def metrics(self) -> dict[str, int]:
        return {
            "standard_resets": self.standard_resets,
            "augmented_resets": self.augmented_resets,
        }


def build_training_state_source(
    *,
    reset_mode: str,
    buffer_path: str | Path | None,
    augmented_probability: float,
    env: Any,
    environment_config: dict[str, Any],
) -> StateSource:
    """Build the configured default-off training state source."""
    normalized = str(reset_mode).lower()
    if normalized not in {"standard", "augmented", "mixed"}:
        raise ValueError("reset_mode must be standard, augmented, or mixed")
    if normalized == "standard":
        return StandardStateSource()
    if buffer_path is None:
        raise ValueError(f"state_augmentation.buffer_path is required for {normalized} mode")
    buffer = load_state_buffer(
        buffer_path,
        env=env,
        environment_config=environment_config,
    )
    probability = 1.0 if normalized == "augmented" else float(augmented_probability)
    return BufferedStateSource(buffer, augmented_probability=probability)


def state_source_metrics(source: StateSource) -> dict[str, int]:
    metrics = getattr(source, "metrics", None)
    if callable(metrics):
        return dict(metrics())
    return {"standard_resets": 0, "augmented_resets": 0}
