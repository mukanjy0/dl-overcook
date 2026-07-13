"""Deterministic state-buffer record sampling."""

from __future__ import annotations

import numpy as np

from src.state_augmentation.buffer import StateBuffer, StateRecord


class StateBufferSampler:
    """Uniformly sample records using only a caller-owned random generator."""

    def __init__(self, buffer: StateBuffer):
        if not buffer.records:
            raise ValueError("Cannot sample from an empty state buffer")
        self.buffer = buffer

    def sample(self, rng: np.random.Generator) -> StateRecord:
        index = int(rng.integers(0, len(self.buffer.records)))
        return self.buffer.records[index]
