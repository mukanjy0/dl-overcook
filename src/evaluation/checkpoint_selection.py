"""Evaluate and select deployable artifacts from Stage A training checkpoints."""

from __future__ import annotations

import json
import logging
import os
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any

from src.checkpointing import (
    export_inference_artifact,
    inspect_training_checkpoint,
)
from src.evaluation.evaluator import evaluate_from_config
from src.experiment_config import StageAConfig

LOGGER = logging.getLogger(__name__)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary_path, path)


def _training_checkpoints(checkpoint_dir: Path) -> list[Path]:
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_step_*.pt"))
    final_checkpoint = checkpoint_dir / "training_final.pt"
    if final_checkpoint.exists():
        checkpoints.append(final_checkpoint)
    if not checkpoints:
        raise FileNotFoundError(f"No training checkpoints found in {checkpoint_dir}")
    return checkpoints


def _validate_selection_suite(evaluation: dict[str, Any]) -> None:
    modes = {str(value).lower() for value in evaluation.get("inference_modes", [])}
    if modes != {"deterministic", "stochastic"}:
        raise ValueError(
            "Checkpoint selection requires evaluation.inference_modes to contain "
            "deterministic and stochastic"
        )
    positions = {int(value) for value in evaluation.get("player_positions", [])}
    if positions != {0, 1}:
        raise ValueError(
            "Checkpoint selection requires evaluation.player_positions to contain 0 and 1"
        )
    if not evaluation.get("partners"):
        raise ValueError("Checkpoint selection requires at least one evaluation partner")


def _runtime_config(
    config: StageAConfig,
    *,
    artifact_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    policy_path = Path(__file__).resolve().parents[2] / "policies" / "rl_policy.py"
    ego_policy = {
        "type": "python_class",
        "name": "stage_a_checkpoint",
        "path": str(policy_path),
        "class_name": "StudentAgent",
        "config": {
            "checkpoint_path": str(artifact_path),
            "device": config.experiment.device,
            "deterministic": True,
        },
        "max_action_time_ms": 100,
        "invalid_action": "stay",
        "timeout_action": "stay",
    }
    evaluation = deepcopy(config.evaluation)
    resolved_partners: list[dict[str, Any]] = []
    for entry in evaluation.get("partners", []):
        if entry == "self_play" or (
            isinstance(entry, dict)
            and entry.get("name") == "self_play"
            and "policy" not in entry
        ):
            resolved_partners.append(
                {
                    "name": "self_play",
                    "policy": deepcopy(ego_policy),
                    "match_ego_inference_mode": True,
                }
            )
        else:
            resolved_partners.append(deepcopy(entry))
    evaluation["partners"] = resolved_partners
    return {
        "seed": config.experiment.seed,
        "environment": deepcopy(config.environment.config),
        "observation": {
            "type": config.observation.type,
            "include_agent_index": config.observation.include_agent_index,
        },
        "policies": {
            "agent_0": ego_policy,
        },
        "evaluation": evaluation,
        "rendering": {"mode": "none"},
        "logging": {
            "output_dir": str(output_dir),
            "save_step_log": False,
            "save_episode_summary": True,
            "save_trajectory_pickle": False,
        },
        "data_collection": {"enabled": False},
    }


def _selection_key(record: dict[str, Any]) -> tuple[float, float, int, int]:
    deterministic = record["evaluation"]["modes"]["deterministic"]
    return (
        float(deterministic["min_position_score"]),
        float(deterministic["mean_official_score"]),
        int(record["environment_steps"]),
        int(Path(record["training_checkpoint"]).name == "training_final.pt"),
    )


def evaluate_training_checkpoints(
    config: StageAConfig,
    *,
    checkpoint_dir: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Evaluate every checkpoint and copy the best deployment pair to ``selected``."""
    evaluation = deepcopy(config.evaluation)
    _validate_selection_suite(evaluation)
    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    artifacts_dir = output_root / "artifacts"
    reports_dir = output_root / "reports"
    selected_dir = output_root / "selected"
    for directory in (artifacts_dir, reports_dir, selected_dir):
        directory.mkdir(parents=True, exist_ok=True)

    checkpoints = _training_checkpoints(checkpoint_dir)
    records: list[dict[str, Any]] = []
    report_path = output_root / "checkpoint_evaluation.json"
    for index, training_checkpoint in enumerate(checkpoints, start=1):
        LOGGER.info(
            "Evaluating checkpoint %d/%d: %s",
            index,
            len(checkpoints),
            training_checkpoint.name,
        )
        metadata = inspect_training_checkpoint(training_checkpoint)
        artifact_path = artifacts_dir / f"{training_checkpoint.stem}_inference.pt"
        export_inference_artifact(training_checkpoint, artifact_path)
        report = evaluate_from_config(
            _runtime_config(
                config,
                artifact_path=artifact_path,
                output_dir=reports_dir / training_checkpoint.stem,
            )
        )
        record = {
            "training_checkpoint": str(training_checkpoint),
            "inference_artifact": str(artifact_path),
            "environment_steps": metadata["environment_steps"],
            "update": metadata["update"],
            "evaluation": report,
        }
        records.append(record)
        _write_json(
            report_path,
            {
                "status": "running",
                "evaluated_checkpoints": len(records),
                "total_checkpoints": len(checkpoints),
                "checkpoints": records,
            },
        )

    selected = max(records, key=_selection_key)
    selected_training = selected_dir / "training.pt"
    selected_inference = selected_dir / "inference.pt"
    shutil.copy2(selected["training_checkpoint"], selected_training)
    shutil.copy2(selected["inference_artifact"], selected_inference)
    result = {
        "status": "complete",
        "selection_criterion": [
            "deterministic_min_position_score",
            "deterministic_mean_official_score",
            "environment_steps",
        ],
        "num_checkpoints": len(records),
        "checkpoints": records,
        "selected": {
            **selected,
            "selected_training_checkpoint": str(selected_training),
            "selected_inference_artifact": str(selected_inference),
        },
    }
    _write_json(report_path, result)
    LOGGER.info(
        "Selected %s with deterministic min-position/mean score %.1f/%.1f",
        Path(selected["training_checkpoint"]).name,
        selected["evaluation"]["modes"]["deterministic"]["min_position_score"],
        selected["evaluation"]["modes"]["deterministic"]["mean_official_score"],
    )
    return result
