"""Typed training configuration loading with early validation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.config import ConfigError, load_yaml
from src.partners.samplers import build_ego_position_sampler, build_partner_sampler


@dataclass(frozen=True)
class ExperimentSettings:
    name: str
    seed: int
    device: str


@dataclass(frozen=True)
class EnvironmentSettings:
    config: dict[str, Any]


@dataclass(frozen=True)
class ObservationSettings:
    type: str
    include_agent_index: bool


@dataclass(frozen=True)
class ModelSettings:
    architecture: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class TrainingSettings:
    algorithm: str
    total_steps: int
    num_environments: int
    rollout_steps: int
    reward_shaping: float
    ppo: dict[str, Any]


@dataclass(frozen=True)
class CheckpointSettings:
    resume_from: Path | None
    load_optimizer_state: bool
    save_interval: int
    export_path: Path


@dataclass(frozen=True)
class OutputSettings:
    root: Path
    logs: Path
    checkpoints: Path
    metrics: Path


@dataclass(frozen=True)
class StateAugmentationSettings:
    """Default-off reset distribution used by the PPO environment factory."""

    reset_mode: str
    buffer_path: Path | None
    augmented_probability: float


@dataclass(frozen=True)
class StageAConfig:
    """Validated training configuration shared by the staged PPO workflows."""

    source_path: Path
    experiment: ExperimentSettings
    environment: EnvironmentSettings
    observation: ObservationSettings
    model: ModelSettings
    training: TrainingSettings
    partner: dict[str, Any]
    evaluation: dict[str, Any]
    state_augmentation: StateAugmentationSettings
    checkpoint: CheckpointSettings
    outputs: OutputSettings
    effective: dict[str, Any]


def _mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"Missing or invalid section '{key}'")
    return value


def _resolve_path(base_dir: Path, value: Any) -> Any:
    if value in (None, ""):
        return value
    path = Path(str(value)).expanduser()
    return str(path if path.is_absolute() else (base_dir / path).resolve())


def _resolve_known_paths(config: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    resolved = deepcopy(config)
    environment = resolved.get("environment", {}) or {}
    if environment.get("layout_file"):
        environment["layout_file"] = _resolve_path(base_dir, environment["layout_file"])

    def resolve_policy(policy: dict[str, Any]) -> None:
        if policy.get("path"):
            policy["path"] = _resolve_path(base_dir, policy["path"])
        policy_runtime = policy.get("config", {}) or {}
        for key in ("checkpoint_path", "model_path"):
            if policy_runtime.get(key):
                policy_runtime[key] = _resolve_path(base_dir, policy_runtime[key])

    for policy in (resolved.get("policies", {}) or {}).values():
        resolve_policy(policy)

    for policy in (resolved.get("partner", {}) or {}).get("policies", []) or []:
        if isinstance(policy, dict):
            resolve_policy(policy.get("policy", policy))

    collection = resolved.get("collection", {}) or {}
    if collection.get("output_path"):
        collection["output_path"] = _resolve_path(base_dir, collection["output_path"])
    for pairing in collection.get("pairings", []) or []:
        if not isinstance(pairing, dict):
            continue
        for player_key in ("player_0", "player_1"):
            player = pairing.get(player_key)
            if isinstance(player, dict) and isinstance(player.get("policy"), dict):
                resolve_policy(player["policy"])

    evaluation = resolved.get("evaluation", {}) or {}
    for policy in evaluation.get("partners", []) or []:
        if isinstance(policy, dict):
            resolve_policy(policy.get("policy", policy))
    for layout in evaluation.get("layouts", []) or []:
        if isinstance(layout, dict) and layout.get("layout_file"):
            layout["layout_file"] = _resolve_path(base_dir, layout["layout_file"])

    for section, keys in {
        "logging": ("output_dir",),
        "rendering": ("gif_path",),
        "data_collection": (
            "output_dir",
            "output_path",
            "npz_path",
            "metadata_json_path",
        ),
        "checkpoint": ("resume_from", "export_path"),
        "outputs": ("root", "logs", "checkpoints", "metrics"),
        "state_augmentation": ("buffer_path",),
    }.items():
        values = resolved.get(section, {}) or {}
        for key in keys:
            if values.get(key):
                values[key] = _resolve_path(base_dir, values[key])
    return resolved


def load_runtime_config(path: str | Path) -> dict[str, Any]:
    """Load a runner config, opting into config-relative paths when requested."""
    source_path = Path(path).expanduser().resolve()
    config = load_yaml(source_path)
    if bool(config.get("paths_relative_to_config", False)):
        config = _resolve_known_paths(config, source_path.parent)
    return config


def load_experiment_config(path: str | Path) -> StageAConfig:
    """Load, resolve, and validate a staged PPO training configuration."""
    source_path = Path(path).expanduser().resolve()
    config = _resolve_known_paths(load_yaml(source_path), source_path.parent)

    experiment_cfg = _mapping(config, "experiment")
    environment_cfg = _mapping(config, "environment")
    observation_cfg = _mapping(config, "observation")
    model_cfg = _mapping(config, "model")
    training_cfg = _mapping(config, "training")
    partner_cfg = _mapping(config, "partner")
    state_augmentation_cfg = config.get("state_augmentation", {}) or {}
    if not isinstance(state_augmentation_cfg, dict):
        raise ConfigError("state_augmentation must be a mapping")
    checkpoint_cfg = _mapping(config, "checkpoint")
    outputs_cfg = _mapping(config, "outputs")

    layout_name = environment_cfg.get("layout_name")
    layout_file = environment_cfg.get("layout_file")
    if bool(layout_name) == bool(layout_file):
        raise ConfigError("Set exactly one of environment.layout_name or environment.layout_file")
    if int(environment_cfg.get("horizon", 0)) <= 0:
        raise ConfigError("environment.horizon must be positive")

    observation_type = str(observation_cfg.get("type", ""))
    if observation_type not in {"featurized", "lossless_grid"}:
        raise ConfigError("Training requires featurized or lossless_grid observations")
    architecture = str(model_cfg.get("architecture", ""))
    if architecture != "mlp_actor_critic":
        raise ConfigError("model.architecture must be 'mlp_actor_critic'")
    if str(training_cfg.get("algorithm", "")).lower() != "ppo":
        raise ConfigError("training.algorithm must be 'ppo'")

    reset_mode = str(
        state_augmentation_cfg.get("reset_mode", "standard")
    ).lower()
    if reset_mode not in {"standard", "augmented", "mixed"}:
        raise ConfigError(
            "state_augmentation.reset_mode must be standard, augmented, or mixed"
        )
    buffer_value = state_augmentation_cfg.get("buffer_path")
    buffer_path = None if buffer_value in (None, "") else Path(str(buffer_value))
    try:
        augmented_probability = float(
            state_augmentation_cfg.get(
                "augmented_probability",
                0.0 if reset_mode == "standard" else 1.0,
            )
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            "state_augmentation.augmented_probability must be numeric"
        ) from exc
    if reset_mode in {"augmented", "mixed"} and buffer_path is None:
        raise ConfigError(
            f"state_augmentation.buffer_path is required for {reset_mode} mode"
        )
    if reset_mode == "mixed" and not 0.0 < augmented_probability < 1.0:
        raise ConfigError(
            "state_augmentation.augmented_probability must be between 0 and 1 "
            "for mixed mode"
        )
    if reset_mode == "augmented":
        augmented_probability = 1.0
    elif reset_mode == "standard":
        augmented_probability = 0.0
    partner_sampler_name = str(partner_cfg.get("sampler", "self_play")).lower()
    try:
        build_partner_sampler(partner_cfg)
        if partner_sampler_name != "self_play":
            build_ego_position_sampler(partner_cfg)
    except (TypeError, ValueError) as exc:
        raise ConfigError(str(exc)) from exc

    positive_training_fields = ("total_steps", "num_environments", "rollout_steps")
    for key in positive_training_fields:
        if int(training_cfg.get(key, 0)) <= 0:
            raise ConfigError(f"training.{key} must be positive")

    device = str(experiment_cfg.get("device", "auto")).lower()
    if device not in {"auto", "cpu", "cuda"}:
        raise ConfigError("experiment.device must be auto, cpu, or cuda")

    root_value = outputs_cfg.get("root")
    if not root_value:
        raise ConfigError("outputs.root is required")
    root = Path(str(root_value))
    logs = Path(str(outputs_cfg.get("logs", root / "logs")))
    checkpoints = Path(str(outputs_cfg.get("checkpoints", root / "checkpoints")))
    metrics = Path(str(outputs_cfg.get("metrics", root / "metrics")))
    export_value = checkpoint_cfg.get("export_path")
    if not export_value:
        raise ConfigError("checkpoint.export_path is required")

    effective = deepcopy(config)
    effective["config_source"] = str(source_path)
    return StageAConfig(
        source_path=source_path,
        experiment=ExperimentSettings(
            name=str(experiment_cfg.get("name", "stage_a")),
            seed=int(experiment_cfg.get("seed", 0)),
            device=device,
        ),
        environment=EnvironmentSettings(config=deepcopy(environment_cfg)),
        observation=ObservationSettings(
            type=observation_type,
            include_agent_index=bool(observation_cfg.get("include_agent_index", True)),
        ),
        model=ModelSettings(
            architecture=architecture,
            parameters=deepcopy(model_cfg.get("parameters", {}) or {}),
        ),
        training=TrainingSettings(
            algorithm="ppo",
            total_steps=int(training_cfg["total_steps"]),
            num_environments=int(training_cfg["num_environments"]),
            rollout_steps=int(training_cfg["rollout_steps"]),
            reward_shaping=float(training_cfg.get("reward_shaping", 0.0)),
            ppo=deepcopy(training_cfg.get("ppo", {}) or {}),
        ),
        partner=deepcopy(partner_cfg),
        evaluation=deepcopy(config.get("evaluation", {}) or {}),
        state_augmentation=StateAugmentationSettings(
            reset_mode=reset_mode,
            buffer_path=buffer_path,
            augmented_probability=augmented_probability,
        ),
        checkpoint=CheckpointSettings(
            resume_from=(
                None
                if not checkpoint_cfg.get("resume_from")
                else Path(str(checkpoint_cfg["resume_from"]))
            ),
            load_optimizer_state=bool(
                checkpoint_cfg.get("load_optimizer_state", True)
            ),
            save_interval=int(checkpoint_cfg.get("save_interval", 0)),
            export_path=Path(str(export_value)),
        ),
        outputs=OutputSettings(
            root=root,
            logs=logs,
            checkpoints=checkpoints,
            metrics=metrics,
        ),
        effective=effective,
    )


def write_effective_config(config: StageAConfig, path: str | Path) -> None:
    """Write the fully resolved configuration used by a run."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config.effective, stream, sort_keys=False)
