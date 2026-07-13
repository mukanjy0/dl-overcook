from __future__ import annotations

import pytest

from src.training.ppo import PPOConfig


def test_entropy_coefficient_anneals_linearly_and_clamps() -> None:
    config = PPOConfig.from_dict(
        {
            "entropy_coefficient": 0.01,
            "entropy_final_coefficient": 0.001,
            "entropy_anneal_steps": 100,
        }
    )
    assert config.entropy_coefficient_at(0, 200) == pytest.approx(0.01)
    assert config.entropy_coefficient_at(50, 200) == pytest.approx(0.0055)
    assert config.entropy_coefficient_at(100, 200) == pytest.approx(0.001)
    assert config.entropy_coefficient_at(200, 200) == pytest.approx(0.001)


def test_constant_entropy_remains_backward_compatible() -> None:
    config = PPOConfig.from_dict({"entropy_coefficient": 0.02})
    assert config.entropy_coefficient_at(100, 100) == pytest.approx(0.02)


def test_invalid_entropy_schedule_fails_early() -> None:
    with pytest.raises(ValueError, match="entropy_anneal_steps"):
        PPOConfig.from_dict(
            {
                "entropy_coefficient": 0.01,
                "entropy_final_coefficient": 0.001,
                "entropy_anneal_steps": 0,
            }
        )
