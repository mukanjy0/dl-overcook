from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from src.checkpointing import (
    CheckpointCompatibilityError,
    export_inference_artifact,
    load_inference_artifact,
    load_training_checkpoint,
    save_inference_artifact,
    save_training_checkpoint,
)
from src.models.actor_critic import ActorCritic, ActorCriticConfig
from src.models.interfaces import ObservationSpec


def _model_and_spec() -> tuple[ActorCritic, ActorCriticConfig, ObservationSpec]:
    spec = ObservationSpec(
        obs_type="featurized",
        shape=(4,),
        include_agent_index=True,
    )
    config = ActorCriticConfig(hidden_sizes=(8,), activation="tanh")
    return ActorCritic(spec.encoded_size, 6, config), config, spec


def test_training_save_resume_and_inference_export(tmp_path: Path) -> None:
    model, model_config, observation_spec = _model_and_spec()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = model(torch.ones(2, observation_spec.encoded_size))[0].sum()
    loss.backward()
    optimizer.step()

    training_path = tmp_path / "training.pt"
    save_training_checkpoint(
        training_path,
        model=model,
        model_config=model_config,
        observation_spec=observation_spec,
        optimizer=optimizer,
        scheduler=None,
        trainer_state={"update": 2, "environment_steps": 16},
        effective_config={"experiment": {"seed": 3}},
        environment_metadata={"layout_name": "cramped_room"},
    )

    restored, _, _ = _model_and_spec()
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=1e-3)
    payload = load_training_checkpoint(
        training_path,
        model=restored,
        optimizer=restored_optimizer,
        restore_random_state=False,
    )
    assert payload["trainer_state"]["environment_steps"] == 16
    for expected, actual in zip(model.parameters(), restored.parameters()):
        assert torch.equal(expected, actual)

    inference_path = tmp_path / "inference.pt"
    export_inference_artifact(training_path, inference_path)
    loaded = load_inference_artifact(inference_path, device="cpu")
    observation = {
        "obs": np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "agent_index": 1,
    }
    expected_action = int(
        model.act_batch(
            torch.as_tensor(
                observation_spec.encode(observation), dtype=torch.float32
            ).unsqueeze(0),
            deterministic=True,
        ).actions.item()
    )
    assert loaded.policy.act(observation, deterministic=True) == expected_action


def test_training_checkpoint_can_restore_or_ignore_rng_state(tmp_path: Path) -> None:
    model, model_config, observation_spec = _model_and_spec()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    path = tmp_path / "rng.pt"
    save_training_checkpoint(
        path,
        model=model,
        model_config=model_config,
        observation_spec=observation_spec,
        optimizer=optimizer,
        scheduler=None,
        trainer_state={"update": 1, "environment_steps": 8},
        effective_config={},
        environment_metadata={},
    )
    expected = (random.random(), float(np.random.random()), float(torch.rand(())))

    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    load_training_checkpoint(path, model=model, restore_random_state=True)
    restored = (random.random(), float(np.random.random()), float(torch.rand(())))
    assert restored == expected

    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    expected_without_restore = (
        random.random(),
        float(np.random.random()),
        float(torch.rand(())),
    )
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    load_training_checkpoint(path, model=model, restore_random_state=False)
    actual_without_restore = (
        random.random(),
        float(np.random.random()),
        float(torch.rand(())),
    )
    assert actual_without_restore == expected_without_restore


def test_cpu_fallback_when_cuda_is_unavailable(tmp_path: Path, monkeypatch) -> None:
    model, model_config, observation_spec = _model_and_spec()
    artifact = tmp_path / "inference.pt"
    save_inference_artifact(
        artifact,
        model=model,
        model_config=model_config,
        observation_spec=observation_spec,
        environment_metadata={},
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert load_inference_artifact(artifact, device="cuda").device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_loading_path(tmp_path: Path) -> None:
    model, model_config, observation_spec = _model_and_spec()
    artifact = tmp_path / "inference.pt"
    save_inference_artifact(
        artifact,
        model=model,
        model_config=model_config,
        observation_spec=observation_spec,
        environment_metadata={},
    )
    assert load_inference_artifact(artifact, device="cuda").device.type == "cuda"


def test_incompatible_schema_action_shape_and_dependencies_fail(tmp_path: Path) -> None:
    model, model_config, observation_spec = _model_and_spec()
    base_path = tmp_path / "base.pt"
    save_inference_artifact(
        base_path,
        model=model,
        model_config=model_config,
        observation_spec=observation_spec,
        environment_metadata={},
    )
    payload = torch.load(base_path, map_location="cpu", weights_only=True)

    cases = []
    schema = dict(payload)
    schema["schema_version"] = 999
    cases.append(schema)

    actions = dict(payload)
    actions["actions"] = {"num_actions": 5, "index_to_name": {}}
    cases.append(actions)

    shape = dict(payload)
    shape["model"] = {**payload["model"], "input_size": 999}
    cases.append(shape)

    dependencies = dict(payload)
    dependencies["dependencies"] = {
        **payload["dependencies"],
        "overcooked-ai": "999.0.0",
    }
    cases.append(dependencies)

    for index, invalid_payload in enumerate(cases):
        path = tmp_path / f"invalid_{index}.pt"
        torch.save(invalid_payload, path)
        with pytest.raises(CheckpointCompatibilityError):
            load_inference_artifact(path, device="cpu")


def test_observation_validation_rejects_wrong_shape(tmp_path: Path) -> None:
    model, model_config, observation_spec = _model_and_spec()
    artifact = tmp_path / "inference.pt"
    save_inference_artifact(
        artifact,
        model=model,
        model_config=model_config,
        observation_spec=observation_spec,
        environment_metadata={},
    )
    loaded = load_inference_artifact(artifact, device="cpu")
    with pytest.raises(ValueError, match="shape mismatch"):
        loaded.policy.act(
            {"obs": np.zeros(3, dtype=np.float32), "agent_index": 0}
        )
