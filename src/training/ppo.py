"""Minimal, explicit Proximal Policy Optimization update."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from src.models.actor_critic import ActorCritic
from src.training.rollouts import RolloutBatch
from src.training.schedules import linear_schedule


@dataclass(frozen=True)
class PPOConfig:
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coefficient: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    entropy_final_coefficient: float | None = None
    entropy_anneal_steps: int | None = None
    update_epochs: int = 4
    minibatch_size: int = 256
    max_grad_norm: float = 0.5

    @classmethod
    def from_dict(cls, values: dict[str, Any] | None) -> "PPOConfig":
        values = values or {}
        config = cls(
            learning_rate=float(values.get("learning_rate", 3e-4)),
            gamma=float(values.get("gamma", 0.99)),
            gae_lambda=float(values.get("gae_lambda", 0.95)),
            clip_coefficient=float(values.get("clip_coefficient", 0.2)),
            value_coefficient=float(values.get("value_coefficient", 0.5)),
            entropy_coefficient=float(values.get("entropy_coefficient", 0.01)),
            entropy_final_coefficient=(
                None
                if values.get("entropy_final_coefficient") is None
                else float(values["entropy_final_coefficient"])
            ),
            entropy_anneal_steps=(
                None
                if values.get("entropy_anneal_steps") is None
                else int(values["entropy_anneal_steps"])
            ),
            update_epochs=int(values.get("update_epochs", 4)),
            minibatch_size=int(values.get("minibatch_size", 256)),
            max_grad_norm=float(values.get("max_grad_norm", 0.5)),
        )
        if config.learning_rate <= 0 or config.update_epochs <= 0 or config.minibatch_size <= 0:
            raise ValueError("PPO learning_rate, update_epochs, and minibatch_size must be positive")
        if not (0.0 <= config.gamma <= 1.0 and 0.0 <= config.gae_lambda <= 1.0):
            raise ValueError("PPO gamma and gae_lambda must be in [0, 1]")
        if config.entropy_coefficient < 0:
            raise ValueError("PPO entropy_coefficient must be non-negative")
        if (
            config.entropy_final_coefficient is not None
            and config.entropy_final_coefficient < 0
        ):
            raise ValueError("PPO entropy_final_coefficient must be non-negative")
        if config.entropy_anneal_steps is not None and config.entropy_anneal_steps <= 0:
            raise ValueError("PPO entropy_anneal_steps must be positive")
        return config

    def entropy_coefficient_at(self, environment_steps: int, total_steps: int) -> float:
        """Return the coefficient for steps completed in the current run."""
        anneal_steps = self.entropy_anneal_steps or total_steps
        return linear_schedule(
            self.entropy_coefficient,
            self.entropy_final_coefficient,
            completed_steps=environment_steps,
            anneal_steps=anneal_steps,
        )


class PPOUpdater:
    """Own the optimizer-facing PPO update, separate from rollout collection."""

    def __init__(
        self,
        model: ActorCritic,
        optimizer: torch.optim.Optimizer,
        config: PPOConfig,
    ):
        self.model = model
        self.optimizer = optimizer
        self.config = config

    def update(
        self,
        rollout: RolloutBatch,
        *,
        entropy_coefficient: float | None = None,
    ) -> dict[str, float]:
        effective_entropy_coefficient = (
            self.config.entropy_coefficient
            if entropy_coefficient is None
            else float(entropy_coefficient)
        )
        data = rollout.flattened()
        advantages = data["advantages"]
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        batch_size = int(data["actions"].shape[0])
        minibatch_size = min(self.config.minibatch_size, batch_size)
        metrics: dict[str, list[float]] = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "approx_kl": [],
            "clip_fraction": [],
        }

        self.model.train()
        for _ in range(self.config.update_epochs):
            permutation = torch.randperm(batch_size, device=data["actions"].device)
            for start in range(0, batch_size, minibatch_size):
                indices = permutation[start : start + minibatch_size]
                new_log_probabilities, entropy, new_values = self.model.evaluate_actions(
                    data["observations"][indices],
                    data["actions"][indices],
                )
                log_ratio = new_log_probabilities - data["old_log_probabilities"][indices]
                ratio = log_ratio.exp()
                minibatch_advantages = advantages[indices]
                unclipped = -minibatch_advantages * ratio
                clipped = -minibatch_advantages * torch.clamp(
                    ratio,
                    1.0 - self.config.clip_coefficient,
                    1.0 + self.config.clip_coefficient,
                )
                policy_loss = torch.maximum(unclipped, clipped).mean()
                value_loss = 0.5 * (
                    new_values - data["returns"][indices]
                ).pow(2).mean()
                entropy_mean = entropy.mean()
                loss = (
                    policy_loss
                    + self.config.value_coefficient * value_loss
                    - effective_entropy_coefficient * entropy_mean
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )
                self.optimizer.step()

                with torch.no_grad():
                    metrics["policy_loss"].append(float(policy_loss.item()))
                    metrics["value_loss"].append(float(value_loss.item()))
                    metrics["entropy"].append(float(entropy_mean.item()))
                    metrics["approx_kl"].append(float(((ratio - 1.0) - log_ratio).mean().item()))
                    metrics["clip_fraction"].append(
                        float(
                            ((ratio - 1.0).abs() > self.config.clip_coefficient)
                            .float()
                            .mean()
                            .item()
                        )
                    )

        result = {
            name: sum(values) / len(values) if values else 0.0
            for name, values in metrics.items()
        }
        result["entropy_coefficient"] = effective_entropy_coefficient
        return result
