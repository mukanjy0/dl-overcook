"""Small shared-backbone actor-critic used by Stage A PPO."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from src.models.interfaces import ObservationSpec, PolicyStep


@dataclass(frozen=True)
class ActorCriticConfig:
    hidden_sizes: tuple[int, ...] = (128, 128)
    activation: str = "tanh"

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ActorCriticConfig":
        data = data or {}
        hidden_sizes = tuple(int(size) for size in data.get("hidden_sizes", (128, 128)))
        if not hidden_sizes or any(size <= 0 for size in hidden_sizes):
            raise ValueError("model.parameters.hidden_sizes must contain positive integers")
        activation = str(data.get("activation", "tanh")).lower()
        if activation not in {"tanh", "relu"}:
            raise ValueError("model.parameters.activation must be 'tanh' or 'relu'")
        return cls(hidden_sizes=hidden_sizes, activation=activation)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ActorCritic(nn.Module):
    """Feed-forward categorical actor with a shared scalar value function."""

    def __init__(self, input_size: int, num_actions: int, config: ActorCriticConfig):
        super().__init__()
        if input_size <= 0:
            raise ValueError("input_size must be positive")
        if num_actions <= 1:
            raise ValueError("num_actions must be greater than one")

        activation_type: type[nn.Module] = nn.Tanh if config.activation == "tanh" else nn.ReLU
        layers: list[nn.Module] = []
        previous_size = int(input_size)
        for hidden_size in config.hidden_sizes:
            layers.extend([nn.Linear(previous_size, hidden_size), activation_type()])
            previous_size = hidden_size

        self.input_size = int(input_size)
        self.num_actions = int(num_actions)
        self.config = config
        self.backbone = nn.Sequential(*layers)
        self.actor = nn.Linear(previous_size, self.num_actions)
        self.critic = nn.Linear(previous_size, 1)
        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2.0))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(observations)
        return self.actor(features), self.critic(features).squeeze(-1)

    def act_batch(self, observations: torch.Tensor, deterministic: bool = False) -> PolicyStep:
        logits, values = self(observations)
        distribution = Categorical(logits=logits)
        actions = torch.argmax(logits, dim=-1) if deterministic else distribution.sample()
        return PolicyStep(
            actions=actions,
            log_probabilities=distribution.log_prob(actions),
            values=values,
        )

    def evaluate_actions(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = self(observations)
        distribution = Categorical(logits=logits)
        return distribution.log_prob(actions), distribution.entropy(), values


class ActorCriticInferencePolicy:
    """Episode session that exposes an ActorCritic through the public policy API."""

    def __init__(self, model: ActorCritic, observation_spec: ObservationSpec, device: torch.device):
        self.model = model.to(device)
        self.observation_spec = observation_spec
        self.device = device
        self.model.eval()

    def reset(self) -> None:
        """The Stage A feed-forward model has no recurrent episode state."""

    def act(self, observation: Any, deterministic: bool = True) -> int:
        encoded = self.observation_spec.encode(observation)
        tensor = torch.as_tensor(encoded, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.inference_mode():
            step = self.model.act_batch(tensor, deterministic=deterministic)
        return int(step.actions.item())
