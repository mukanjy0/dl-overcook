from __future__ import annotations

import pytest

from src.training.ppo import PPOConfig
from src.training.schedules import linear_schedule


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


def test_continuation_schedules_use_steps_since_resume() -> None:
    entropy = PPOConfig.from_dict(
        {
            "entropy_coefficient": 0.01,
            "entropy_final_coefficient": 0.001,
            "entropy_anneal_steps": 50_176,
        }
    )
    assert entropy.entropy_coefficient_at(0, 50_176) == pytest.approx(0.01)
    assert entropy.entropy_coefficient_at(25_088, 50_176) == pytest.approx(0.0055)
    assert entropy.entropy_coefficient_at(50_176, 50_176) == pytest.approx(0.001)
    assert linear_schedule(
        1.0,
        0.1,
        completed_steps=25_088,
        anneal_steps=50_176,
    ) == pytest.approx(0.55)
