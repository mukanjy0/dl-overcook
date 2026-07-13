from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from src.evaluation.checkpoint_selection import _runtime_config, _selection_key


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


def test_checkpoint_runtime_resolves_self_play_to_current_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "checkpoint.pt"
    config = SimpleNamespace(
        experiment=SimpleNamespace(seed=3, device="cpu"),
        environment=SimpleNamespace(config={"layout_name": "cramped_room"}),
        observation=SimpleNamespace(type="featurized", include_agent_index=True),
        evaluation={"partners": ["self_play"]},
    )
    runtime = _runtime_config(config, artifact_path=artifact, output_dir=tmp_path)
    partner = runtime["evaluation"]["partners"][0]
    assert partner["name"] == "self_play"
    assert partner["match_ego_inference_mode"] is True
    assert partner["policy"]["config"]["checkpoint_path"] == str(artifact)
