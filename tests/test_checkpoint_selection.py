from __future__ import annotations

from src.evaluation.checkpoint_selection import _selection_key


def _record(min_score: float, mean_score: float, steps: int = 1) -> dict:
    return {
        "training_checkpoint": "checkpoint.pt",
        "environment_steps": steps,
        "evaluation": {
            "modes": {
                "deterministic": {
                    "min_position_score": min_score,
                    "mean_position_score": mean_score,
                    "mean_official_score": mean_score,
                }
            }
        },
    }


def test_selection_prioritizes_minimum_position_before_mean() -> None:
    balanced = _record(10.0, 20.0)
    one_sided = _record(0.0, 100.0)
    assert _selection_key(balanced) > _selection_key(one_sided)


def test_selection_uses_mean_official_score_as_second_criterion() -> None:
    higher_mean = _record(10.0, 30.0)
    lower_mean = _record(10.0, 20.0)
    assert _selection_key(higher_mean) > _selection_key(lower_mean)
