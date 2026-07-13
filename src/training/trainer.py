"""Reusable orchestration for Stage A PPO self-play training."""

from __future__ import annotations

import json
import logging
import os
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import yaml

from src.checkpointing import (
    export_inference_artifact,
    load_training_checkpoint,
    resolve_device,
    save_training_checkpoint,
)
from src.constants import NUM_ACTIONS
from src.environment import build_env
from src.experiment_config import StageAConfig
from src.models.actor_critic import ActorCritic, ActorCriticConfig
from src.models.interfaces import ObservationSpec
from src.observations import ObservationBuilder
from src.seed_utils import derive_seed, set_global_seed
from src.training.ppo import PPOConfig, PPOUpdater
from src.training.rollouts import SelfPlayRolloutCollector

LOGGER = logging.getLogger(__name__)


def _output_paths(
    config: StageAConfig,
    output_root_override: str | Path | None,
) -> tuple[Path, Path, Path, Path, Path]:
    if output_root_override is None:
        return (
            config.outputs.root,
            config.outputs.logs,
            config.outputs.checkpoints,
            config.outputs.metrics,
            config.checkpoint.export_path,
        )
    root = Path(output_root_override).expanduser().resolve()
    return root, root / "logs", root / "checkpoints", root / "metrics", root / "checkpoints" / "inference.pt"


def _write_effective_config(
    config: StageAConfig,
    destination: Path,
    *,
    output_root: Path,
    logs_dir: Path,
    checkpoints_dir: Path,
    metrics_dir: Path,
    export_path: Path,
) -> dict[str, Any]:
    effective = deepcopy(config.effective)
    effective["outputs"] = {
        "root": str(output_root),
        "logs": str(logs_dir),
        "checkpoints": str(checkpoints_dir),
        "metrics": str(metrics_dir),
    }
    effective.setdefault("checkpoint", {})["export_path"] = str(export_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(effective, stream, sort_keys=False)
    return effective


def _append_json_line(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically publish machine-readable run progress."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
    os.replace(temporary_path, path)


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _checkpoint(
    path: Path,
    *,
    model: ActorCritic,
    model_config: ActorCriticConfig,
    observation_spec: ObservationSpec,
    optimizer: torch.optim.Optimizer,
    trainer_state: dict[str, Any],
    effective_config: dict[str, Any],
    environment_metadata: dict[str, Any],
) -> Path:
    LOGGER.info("Saving training checkpoint to %s", path)
    return save_training_checkpoint(
        path,
        model=model,
        model_config=model_config,
        observation_spec=observation_spec,
        optimizer=optimizer,
        scheduler=None,
        trainer_state=trainer_state,
        effective_config=effective_config,
        environment_metadata=environment_metadata,
    )


def train(
    config: StageAConfig,
    *,
    output_root_override: str | Path | None = None,
) -> dict[str, Any]:
    """Train, checkpoint, and export one Stage A self-play actor-critic."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    output_root, logs_dir, checkpoints_dir, metrics_dir, export_path = _output_paths(
        config,
        output_root_override,
    )
    for directory in (output_root, logs_dir, checkpoints_dir, metrics_dir):
        directory.mkdir(parents=True, exist_ok=True)
    effective_config = _write_effective_config(
        config,
        output_root / "effective_config.yaml",
        output_root=output_root,
        logs_dir=logs_dir,
        checkpoints_dir=checkpoints_dir,
        metrics_dir=metrics_dir,
        export_path=export_path,
    )

    base_seed = config.experiment.seed
    set_global_seed(derive_seed(base_seed, "model"))
    requested_device = config.experiment.device
    device = resolve_device(requested_device)
    if requested_device == "cuda" and device.type != "cuda":
        LOGGER.warning("CUDA was requested but is unavailable; training on CPU")
    LOGGER.info("Configuration loaded from %s", config.source_path)
    LOGGER.info("Training started on %s", device)

    observation_config = {
        "type": config.observation.type,
        "include_agent_index": config.observation.include_agent_index,
    }
    probe_env = build_env(config.environment.config)
    probe_env.reset(regen_mdp=False)
    probe_builder = ObservationBuilder(probe_env, observation_config)
    observation_spec = ObservationSpec.from_observation(
        probe_builder(probe_env.state, 0),
        obs_type=config.observation.type,
    )
    model_config = ActorCriticConfig.from_dict(config.model.parameters)
    model = ActorCritic(
        input_size=observation_spec.encoded_size,
        num_actions=NUM_ACTIONS,
        config=model_config,
    ).to(device)
    ppo_config = PPOConfig.from_dict(config.training.ppo)
    optimizer = torch.optim.Adam(model.parameters(), lr=ppo_config.learning_rate, eps=1e-5)
    updater = PPOUpdater(model, optimizer, ppo_config)
    collector = SelfPlayRolloutCollector(
        environment_config=config.environment.config,
        observation_config=observation_config,
        observation_spec=observation_spec,
        num_environments=config.training.num_environments,
        base_seed=base_seed,
        device=device,
        reward_shaping=config.training.reward_shaping,
    )

    trainer_state: dict[str, Any] = {"update": 0, "environment_steps": 0}
    if config.checkpoint.resume_from is not None:
        LOGGER.info("Resuming checkpoint %s", config.checkpoint.resume_from)
        resumed = load_training_checkpoint(
            config.checkpoint.resume_from,
            model=model,
            optimizer=optimizer,
            scheduler=None,
        )
        trainer_state.update(resumed.get("trainer_state", {}) or {})

    environment_metadata = {
        "layout_name": config.environment.config.get("layout_name"),
        "layout_file": config.environment.config.get("layout_file"),
        "horizon": int(config.environment.config.get("horizon", 400)),
        "old_dynamics": bool(config.environment.config.get("old_dynamics", True)),
        "mdp_overrides": deepcopy(config.environment.config.get("mdp_overrides", {}) or {}),
    }
    metrics_path = metrics_dir / "training.jsonl"
    progress_path = output_root / "progress.json"
    summary_path = output_root / "run_summary.json"
    next_save_step = (
        int(trainer_state["environment_steps"]) + config.checkpoint.save_interval
        if config.checkpoint.save_interval > 0
        else None
    )
    recent_episode_returns: list[float] = []
    starting_environment_steps = int(trainer_state["environment_steps"])
    training_started_at = time.perf_counter()
    latest_checkpoint: Path | None = None
    _write_json(
        progress_path,
        {
            "status": "running",
            "experiment": config.experiment.name,
            "device": str(device),
            "update": int(trainer_state["update"]),
            "environment_steps": starting_environment_steps,
            "total_steps": config.training.total_steps,
            "progress_percent": 100.0 * starting_environment_steps / config.training.total_steps,
            "steps_per_second": None,
            "eta_seconds": None,
        },
    )

    while int(trainer_state["environment_steps"]) < config.training.total_steps:
        rollout = collector.collect(
            model,
            rollout_steps=config.training.rollout_steps,
            gamma=ppo_config.gamma,
            gae_lambda=ppo_config.gae_lambda,
        )
        update_metrics = updater.update(rollout)
        trainer_state["update"] = int(trainer_state["update"]) + 1
        trainer_state["environment_steps"] = (
            int(trainer_state["environment_steps"]) + rollout.num_environment_steps
        )
        recent_episode_returns.extend(rollout.completed_sparse_returns)
        record = {
            "update": int(trainer_state["update"]),
            "environment_steps": int(trainer_state["environment_steps"]),
            "total_steps": config.training.total_steps,
            "mean_completed_sparse_return": (
                sum(rollout.completed_sparse_returns) / len(rollout.completed_sparse_returns)
                if rollout.completed_sparse_returns
                else None
            ),
            **update_metrics,
        }
        if next_save_step is not None and int(trainer_state["environment_steps"]) >= next_save_step:
            latest_checkpoint = _checkpoint(
                checkpoints_dir / f"checkpoint_step_{int(trainer_state['environment_steps']):09d}.pt",
                model=model,
                model_config=model_config,
                observation_spec=observation_spec,
                optimizer=optimizer,
                trainer_state=trainer_state,
                effective_config=effective_config,
                environment_metadata=environment_metadata,
            )
            while next_save_step <= int(trainer_state["environment_steps"]):
                next_save_step += config.checkpoint.save_interval

        elapsed_seconds = max(time.perf_counter() - training_started_at, 1e-9)
        completed_this_run = int(trainer_state["environment_steps"]) - starting_environment_steps
        steps_per_second = completed_this_run / elapsed_seconds
        remaining_steps = max(
            config.training.total_steps - int(trainer_state["environment_steps"]),
            0,
        )
        eta_seconds = remaining_steps / steps_per_second if steps_per_second > 0 else None
        progress_percent = min(
            100.0,
            100.0 * int(trainer_state["environment_steps"]) / config.training.total_steps,
        )
        record.update(
            {
                "progress_percent": progress_percent,
                "elapsed_seconds": elapsed_seconds,
                "steps_per_second": steps_per_second,
                "eta_seconds": eta_seconds,
            }
        )
        _append_json_line(metrics_path, record)
        progress = {
            "status": "running",
            "experiment": config.experiment.name,
            "device": str(device),
            **record,
            "eta_hhmmss": _format_duration(eta_seconds),
            "latest_checkpoint": None if latest_checkpoint is None else str(latest_checkpoint),
            "metrics": str(metrics_path),
            "effective_config": str(output_root / "effective_config.yaml"),
        }
        _write_json(progress_path, progress)
        _write_json(summary_path, progress)
        mean_return = record["mean_completed_sparse_return"]
        LOGGER.info(
            "progress=%6.2f%% update=%d steps=%d/%d rate=%.1f env_steps/s "
            "eta=%s sparse_return=%s policy_loss=%.4f value_loss=%.4f",
            progress_percent,
            record["update"],
            record["environment_steps"],
            record["total_steps"],
            steps_per_second,
            _format_duration(eta_seconds),
            "n/a" if mean_return is None else f"{mean_return:.2f}",
            record["policy_loss"],
            record["value_loss"],
        )

    final_checkpoint = _checkpoint(
        checkpoints_dir / "training_final.pt",
        model=model,
        model_config=model_config,
        observation_spec=observation_spec,
        optimizer=optimizer,
        trainer_state=trainer_state,
        effective_config=effective_config,
        environment_metadata=environment_metadata,
    )
    export_inference_artifact(final_checkpoint, export_path)
    LOGGER.info("Exported inference artifact to %s", export_path)

    summary = {
        "status": "complete",
        "experiment": config.experiment.name,
        "device": str(device),
        "environment_steps": int(trainer_state["environment_steps"]),
        "updates": int(trainer_state["update"]),
        "mean_completed_sparse_return": (
            sum(recent_episode_returns) / len(recent_episode_returns)
            if recent_episode_returns
            else None
        ),
        "training_checkpoint": str(final_checkpoint),
        "inference_artifact": str(export_path),
        "metrics": str(metrics_path),
        "effective_config": str(output_root / "effective_config.yaml"),
    }
    _write_json(summary_path, summary)
    _write_json(progress_path, summary)
    LOGGER.info("Training completed")
    return summary
