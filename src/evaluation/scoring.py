"""The single canonical implementation of the official competition score."""

from __future__ import annotations

from collections.abc import Iterable


class OfficialScoreCalculator:
    """Stable callable interface delegating to the canonical pure function."""

    def __call__(
        self,
        delivery_timesteps: Iterable[int],
        *,
        horizon: int,
        total_team_timeouts: int,
    ) -> int:
        return calculate_official_score(
            delivery_timesteps,
            horizon=horizon,
            total_team_timeouts=total_team_timeouts,
        )


def calculate_official_score(
    delivery_timesteps: Iterable[int],
    *,
    horizon: int,
    total_team_timeouts: int,
) -> int:
    """Calculate the official score from zero-based upstream event timestamps."""
    deliveries = [int(timestep) for timestep in delivery_timesteps]
    if not deliveries:
        return 0
    return int(
        10_000 * len(deliveries)
        + 10 * (int(horizon) - max(deliveries))
        + (int(horizon) - min(deliveries))
        - min(100 * int(total_team_timeouts), 5_000)
    )


def mean_official_score(scores: Iterable[int | float]) -> float:
    """Return the arithmetic mean, using zero for an empty collection."""
    values = [float(score) for score in scores]
    return sum(values) / len(values) if values else 0.0
