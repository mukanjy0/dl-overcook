"""Thin teacher-facing adapter for exported Stage A inference artifacts."""

from __future__ import annotations

from typing import Any

from src.checkpointing import CheckpointLoader
from src.constants import NUM_ACTIONS


class StudentAgent:
    """Policy interface loaded by the teacher's existing build_policy workflow."""

    def __init__(self, config: dict[str, Any]):
        checkpoint_path = config.get("checkpoint_path")
        if not checkpoint_path:
            raise ValueError("rl_policy.StudentAgent requires config.checkpoint_path")
        loaded = CheckpointLoader.load_inference(
            checkpoint_path,
            device=str(config.get("device", "auto")),
        )
        self.policy = loaded.policy
        self.device = loaded.device
        self.deterministic = bool(config.get("deterministic", True))

    def reset(self) -> None:
        self.policy.reset()

    def act(self, obs: Any) -> int:
        action = int(self.policy.act(obs, deterministic=self.deterministic))
        if action < 0 or action >= NUM_ACTIONS:
            raise RuntimeError(f"Inference policy returned invalid action index {action}")
        return action
