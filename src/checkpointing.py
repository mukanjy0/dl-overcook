"""Versioned training checkpoints and compact inference artifacts."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.constants import ACTION_INDEX_TO_NAME, NUM_ACTIONS
from src.models.actor_critic import (
    ActorCritic,
    ActorCriticConfig,
    ActorCriticInferencePolicy,
)
from src.models.interfaces import ObservationSpec
from src.seed_utils import capture_rng_state, restore_rng_state

SCHEMA_VERSION = 1
TRAINING_PROFILE = "training"
INFERENCE_PROFILE = "inference"
MODEL_ARCHITECTURE = "mlp_actor_critic"


class CheckpointCompatibilityError(ValueError):
    """Raised when a checkpoint cannot satisfy the runtime contract."""


@dataclass(frozen=True)
class LoadedInferenceArtifact:
    """Loaded inference session and its validated artifact metadata."""

    policy: ActorCriticInferencePolicy
    metadata: dict[str, Any]
    device: torch.device


class CheckpointLoader:
    """Stable facade for loading resumable and deployable checkpoint profiles."""

    @staticmethod
    def load_training(path: str | Path, **kwargs: Any) -> dict[str, Any]:
        return load_training_checkpoint(path, **kwargs)

    @staticmethod
    def load_inference(
        path: str | Path,
        *,
        device: str | torch.device = "auto",
    ) -> LoadedInferenceArtifact:
        return load_inference_artifact(path, device=device)


def inspect_training_checkpoint(path: str | Path) -> dict[str, Any]:
    """Return selection metadata from a trusted resumable checkpoint."""
    payload = _load_payload(path, weights_only=False)
    _validate_envelope(payload, TRAINING_PROFILE)
    trainer_state = payload.get("trainer_state", {}) or {}
    return {
        "schema_version": int(payload["schema_version"]),
        "environment_steps": int(trainer_state.get("environment_steps", 0)),
        "update": int(trainer_state.get("update", 0)),
        "observation": payload.get("observation", {}),
        "environment": payload.get("environment", {}),
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _code_version() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def runtime_metadata() -> dict[str, Any]:
    """Describe the dependency/runtime versions needed for reproducibility."""
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "overcooked-ai": _package_version("overcooked-ai"),
        "numpy": str(np.__version__),
        "torch": str(torch.__version__),
        "cuda_runtime": None if torch.version.cuda is None else str(torch.version.cuda),
        "cuda_available": bool(torch.cuda.is_available()),
        "platform": sys.platform,
        "code_version": _code_version(),
    }


def resolve_device(requested: str | torch.device = "auto") -> torch.device:
    """Resolve auto/cpu/cuda, falling back to CPU when CUDA is unavailable."""
    if isinstance(requested, torch.device):
        requested = requested.type
    normalized = str(requested).lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cpu":
        return torch.device("cpu")
    raise ValueError("device must be auto, cpu, or cuda")


def _action_spec() -> dict[str, Any]:
    return {
        "num_actions": NUM_ACTIONS,
        "index_to_name": {
            int(index): ACTION_INDEX_TO_NAME[int(index)] for index in range(NUM_ACTIONS)
        },
    }


def _atomic_torch_save(payload: dict[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return destination


def _model_spec(
    model: ActorCritic,
    model_config: ActorCriticConfig,
    observation_spec: ObservationSpec,
) -> dict[str, Any]:
    return {
        "architecture": MODEL_ARCHITECTURE,
        "parameters": model_config.to_dict(),
        "input_size": observation_spec.encoded_size,
        "num_actions": model.num_actions,
    }


def save_training_checkpoint(
    path: str | Path,
    *,
    model: ActorCritic,
    model_config: ActorCriticConfig,
    observation_spec: ObservationSpec,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    trainer_state: dict[str, Any],
    effective_config: dict[str, Any],
    environment_metadata: dict[str, Any],
    rng_state: dict[str, Any] | None = None,
) -> Path:
    """Save the complete state required to resume training."""
    payload = {
        "schema_version": SCHEMA_VERSION,
        "profile": TRAINING_PROFILE,
        "model": _model_spec(model, model_config, observation_spec),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
        "trainer_state": dict(trainer_state),
        "rng_state": capture_rng_state() if rng_state is None else rng_state,
        "effective_config": effective_config,
        "observation": observation_spec.to_dict(),
        "actions": _action_spec(),
        "environment": environment_metadata,
        "dependencies": runtime_metadata(),
    }
    return _atomic_torch_save(payload, path)


def _load_payload(path: str | Path, *, weights_only: bool) -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Checkpoint not found: {source}")
    try:
        payload = torch.load(source, map_location="cpu", weights_only=weights_only)
    except TypeError:  # PyTorch versions predating the weights_only argument.
        payload = torch.load(source, map_location="cpu")
    if not isinstance(payload, dict):
        raise CheckpointCompatibilityError("Checkpoint payload must be a mapping")
    return payload


def _validate_envelope(payload: dict[str, Any], expected_profile: str) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise CheckpointCompatibilityError(
            f"Unsupported checkpoint schema {payload.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION}"
        )
    if payload.get("profile") != expected_profile:
        raise CheckpointCompatibilityError(
            f"Expected a {expected_profile} checkpoint, got {payload.get('profile')!r}"
        )
    model_spec = payload.get("model", {}) or {}
    if model_spec.get("architecture") != MODEL_ARCHITECTURE:
        raise CheckpointCompatibilityError(
            f"Unsupported model architecture {model_spec.get('architecture')!r}"
        )
    actions = payload.get("actions", {}) or {}
    expected_actions = _action_spec()
    normalized_names = {
        int(index): str(name)
        for index, name in (actions.get("index_to_name", {}) or {}).items()
    }
    if actions.get("num_actions") != NUM_ACTIONS or normalized_names != expected_actions["index_to_name"]:
        raise CheckpointCompatibilityError("Checkpoint action mapping is incompatible")
    if "observation" not in payload or "model_state_dict" not in payload:
        raise CheckpointCompatibilityError(
            "Checkpoint is missing observation metadata or model weights"
        )


def _validate_dependencies(payload: dict[str, Any]) -> None:
    recorded = payload.get("dependencies", {}) or {}
    current_overcooked = _package_version("overcooked-ai")
    recorded_overcooked = recorded.get("overcooked-ai")
    if recorded_overcooked and current_overcooked and recorded_overcooked != current_overcooked:
        raise CheckpointCompatibilityError(
            "Overcooked-AI version mismatch: "
            f"artifact={recorded_overcooked}, runtime={current_overcooked}"
        )
    if int(str(np.__version__).split(".")[0]) >= 2:
        raise CheckpointCompatibilityError("This project requires NumPy major version <2")
    recorded_torch = recorded.get("torch")
    if recorded_torch and str(recorded_torch).split(".")[0] != torch.__version__.split(".")[0]:
        raise CheckpointCompatibilityError(
            f"PyTorch major version mismatch: artifact={recorded_torch}, runtime={torch.__version__}"
        )


def load_training_checkpoint(
    path: str | Path,
    *,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    restore_random_state: bool = True,
) -> dict[str, Any]:
    """Load a training snapshot into caller-owned model and optimizer objects."""
    payload = _load_payload(path, weights_only=False)
    _validate_envelope(payload, TRAINING_PROFILE)
    _validate_dependencies(payload)
    observation_spec = ObservationSpec.from_dict(payload["observation"])
    if model.input_size != observation_spec.encoded_size:
        raise CheckpointCompatibilityError(
            f"Model input mismatch: checkpoint={observation_spec.encoded_size}, model={model.input_size}"
        )
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None and payload.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    if restore_random_state:
        restore_rng_state(payload.get("rng_state"))
    return payload


def save_inference_artifact(
    path: str | Path,
    *,
    model: ActorCritic,
    model_config: ActorCriticConfig,
    observation_spec: ObservationSpec,
    environment_metadata: dict[str, Any],
    dependencies: dict[str, Any] | None = None,
) -> Path:
    """Save the minimal device-neutral artifact used by build_policy."""
    payload = {
        "schema_version": SCHEMA_VERSION,
        "profile": INFERENCE_PROFILE,
        "model": _model_spec(model, model_config, observation_spec),
        "model_state_dict": {
            name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
        },
        "observation": observation_spec.to_dict(),
        "actions": _action_spec(),
        "environment": environment_metadata,
        "dependencies": runtime_metadata() if dependencies is None else dependencies,
    }
    return _atomic_torch_save(payload, path)


def export_inference_artifact(
    training_checkpoint_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Export a compact inference artifact without requiring trainer objects."""
    training_payload = _load_payload(training_checkpoint_path, weights_only=False)
    _validate_envelope(training_payload, TRAINING_PROFILE)
    _validate_dependencies(training_payload)
    inference_payload = {
        key: training_payload[key]
        for key in (
            "schema_version",
            "model",
            "model_state_dict",
            "observation",
            "actions",
            "environment",
            "dependencies",
        )
    }
    inference_payload["profile"] = INFERENCE_PROFILE
    inference_payload["model_state_dict"] = {
        name: tensor.detach().cpu()
        for name, tensor in inference_payload["model_state_dict"].items()
    }
    return _atomic_torch_save(inference_payload, output_path)


def load_inference_artifact(
    path: str | Path,
    *,
    device: str | torch.device = "auto",
) -> LoadedInferenceArtifact:
    """Validate and instantiate a deployable inference session."""
    payload = _load_payload(path, weights_only=True)
    _validate_envelope(payload, INFERENCE_PROFILE)
    _validate_dependencies(payload)

    observation_spec = ObservationSpec.from_dict(payload["observation"])
    model_spec = payload["model"]
    if int(model_spec.get("input_size", -1)) != observation_spec.encoded_size:
        raise CheckpointCompatibilityError("Checkpoint observation and model shapes disagree")
    if int(model_spec.get("num_actions", -1)) != NUM_ACTIONS:
        raise CheckpointCompatibilityError("Checkpoint model does not emit exactly six actions")

    model_config = ActorCriticConfig.from_dict(model_spec.get("parameters"))
    model = ActorCritic(
        input_size=observation_spec.encoded_size,
        num_actions=NUM_ACTIONS,
        config=model_config,
    )
    model.load_state_dict(payload["model_state_dict"])
    resolved_device = resolve_device(device)
    policy = ActorCriticInferencePolicy(model, observation_spec, resolved_device)

    if resolved_device.type == "cuda":
        warmup = torch.zeros(
            (1, observation_spec.encoded_size),
            dtype=torch.float32,
            device=resolved_device,
        )
        with torch.inference_mode():
            policy.model.act_batch(warmup, deterministic=True)
        torch.cuda.synchronize(resolved_device)

    return LoadedInferenceArtifact(
        policy=policy,
        metadata=payload,
        device=resolved_device,
    )
