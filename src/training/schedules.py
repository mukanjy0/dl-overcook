"""Small schedule helpers shared by staged PPO training."""

from __future__ import annotations


def linear_schedule(
    initial: float,
    final: float | None,
    *,
    completed_steps: int,
    anneal_steps: int,
) -> float:
    """Linearly interpolate over steps completed in the current training run."""
    if final is None:
        return float(initial)
    if anneal_steps <= 0:
        raise ValueError("anneal_steps must be positive")
    progress = min(max(float(completed_steps) / float(anneal_steps), 0.0), 1.0)
    return float(initial) + progress * (float(final) - float(initial))
