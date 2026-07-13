"""Stable interfaces and data structures shared by policies and trainers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch


@dataclass(frozen=True)
class ObservationSpec:
    """Description of the exact observation contract stored in a checkpoint."""

    obs_type: str
    shape: tuple[int, ...]
    dtype: str = "float32"
    include_agent_index: bool = True
    num_agents: int = 2

    @classmethod
    def from_observation(cls, observation: Any, *, obs_type: str) -> "ObservationSpec":
        include_agent_index = isinstance(observation, dict) and "agent_index" in observation
        raw_observation = observation.get("obs") if isinstance(observation, dict) else observation
        array = np.asarray(raw_observation)
        if not np.issubdtype(array.dtype, np.number):
            raise TypeError("Trainable observations must contain a numeric array")
        return cls(
            obs_type=str(obs_type),
            shape=tuple(int(value) for value in array.shape),
            dtype="float32",
            include_agent_index=include_agent_index,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ObservationSpec":
        return cls(
            obs_type=str(data["obs_type"]),
            shape=tuple(int(value) for value in data["shape"]),
            dtype=str(data.get("dtype", "float32")),
            include_agent_index=bool(data.get("include_agent_index", True)),
            num_agents=int(data.get("num_agents", 2)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def encoded_size(self) -> int:
        base_size = int(np.prod(self.shape, dtype=np.int64))
        return base_size + (self.num_agents if self.include_agent_index else 0)

    def encode(self, observation: Any) -> np.ndarray:
        """Validate and flatten one observation for the neural policy."""
        if isinstance(observation, dict):
            if "obs" not in observation:
                raise ValueError("Numeric policy observations must contain an 'obs' field")
            raw_observation = observation["obs"]
            agent_index = observation.get("agent_index")
        else:
            raw_observation = observation
            agent_index = None

        array = np.asarray(raw_observation, dtype=np.float32)
        if tuple(array.shape) != self.shape:
            raise ValueError(f"Observation shape mismatch: expected {self.shape}, got {tuple(array.shape)}")
        encoded = array.reshape(-1)

        if self.include_agent_index:
            if agent_index is None:
                raise ValueError("Checkpoint expects observations with agent_index")
            index = int(agent_index)
            if index < 0 or index >= self.num_agents:
                raise ValueError(f"agent_index must be in [0, {self.num_agents - 1}], got {index}")
            one_hot = np.zeros(self.num_agents, dtype=np.float32)
            one_hot[index] = 1.0
            encoded = np.concatenate([encoded, one_hot])
        return encoded.astype(np.float32, copy=False)


@dataclass(frozen=True)
class PolicyStep:
    """Batched actions and actor-critic statistics."""

    actions: torch.Tensor
    log_probabilities: torch.Tensor
    values: torch.Tensor


@runtime_checkable
class InferencePolicy(Protocol):
    def reset(self) -> None:
        """Clear episode-local state."""

    def act(self, observation: Any, deterministic: bool = True) -> int:
        """Return one public action index."""


@runtime_checkable
class TrainablePolicy(Protocol):
    def act_batch(self, observations: torch.Tensor, deterministic: bool = False) -> PolicyStep:
        """Sample or choose actions for a batch of encoded observations."""

    def evaluate_actions(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return log probabilities, entropy, and values for PPO."""
