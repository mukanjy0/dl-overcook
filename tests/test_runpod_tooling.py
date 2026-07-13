"""Offline coverage for the RunPod skill's matrix validation and dry-run guard."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / ".codex/skills/runpod/scripts/runpod_matrix.py"
SPEC = importlib.util.spec_from_file_location("runpod_matrix", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RUNPOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNPOD)


def test_runpod_dry_run_creates_manifest_without_provisioning(tmp_path: Path) -> None:
    matrix = {
        "repository": {
            "url": "https://example.invalid/owner/project.git",
            "commit": "0123456789abcdef0123456789abcdef01234567",
            "setup_command": "uv sync --frozen",
        },
        "pod": {
            "image": "example/image:tag",
            "compute_type": "GPU",
            "gpu_type_ids": ["example-gpu"],
        },
        "jobs": [
            {
                "name": "smoke",
                "config": "configs/smoke.yaml",
                "command": ".venv/bin/python scripts/train.py --config {config}",
                "artifact_paths": ["outputs/smoke"],
            }
        ],
    }
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")
    output_dir = tmp_path / "output"

    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "launch", "--matrix", str(matrix_path), "--output-dir", str(output_dir), "--dry-run"],
        text=True,
        capture_output=True,
        check=True,
    )

    assert '"provisioned": false' in completed.stdout
    manifest = json.loads((output_dir / "runpod_state.json").read_text(encoding="utf-8"))
    assert manifest["dry_run"] is True
    assert manifest["jobs"][0]["status"] == "planned"
    assert manifest["jobs"][0].get("pod_id") is None


def test_runpod_cpu_payload_and_validation() -> None:
    matrix = RUNPOD.validate(
        {
            "repository": {
                "source": "git",
                "url": "https://example.invalid/repository.git",
                "commit": "0123456789abcdef0123456789abcdef01234567",
                "setup_command": "uv sync --frozen",
            },
            "pod": {
                "image": "example/image:tag",
                "compute_type": "CPU",
                "cpu_flavor_ids": ["CPU5C"],
                "vcpu_count": 4,
                "interruptible": False,
            },
            "jobs": [
                {
                    "name": "cpu-smoke",
                    "command": "true",
                    "artifact_paths": ["outputs/smoke"],
                }
            ],
        }
    )

    payload = RUNPOD.pod_payload(matrix, "cpu-smoke", matrix["jobs"][0], "test")
    assert payload["computeType"] == "CPU"
    assert payload["cpuFlavorIds"] == ["CPU5C"]
    assert payload["vcpuCount"] == 4
    assert payload["interruptible"] is False
    assert "gpuTypeIds" not in payload


def test_runpod_job_inputs_are_hashed_and_keep_safe_destinations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    checkpoint = project / "outputs" / "checkpoint.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint-fixture")
    monkeypatch.setattr(RUNPOD, "current_checkout", lambda: (project, "deadbeef"))

    bundle = RUNPOD.prepare_job_inputs(
        {
            "name": "smoke",
            "input_paths": [
                {
                    "source": "outputs/checkpoint.pt",
                    "destination": "outputs/checkpoint.pt",
                }
            ],
        },
        "smoke",
        tmp_path / "launcher",
    )

    assert bundle is not None
    assert bundle["inputs"][0]["size_bytes"] == len(b"checkpoint-fixture")
    assert len(bundle["inputs"][0]["sha256"]) == 64
    assert len(bundle["sha256"]) == 64
    with tarfile.open(bundle["path"], "r:gz") as archive:
        assert sorted(archive.getnames()) == [
            ".runpod-input-manifest.json",
            "outputs/checkpoint.pt",
        ]


def test_runpod_source_archive_is_pinned_and_hashed(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=project_root, text=True
    ).strip()
    monkeypatch.setattr(RUNPOD, "current_checkout", lambda: (project_root, commit))

    bundle = RUNPOD.prepare_source_archive(
        {"source": "archive", "commit": commit},
        tmp_path,
    )

    assert bundle is not None
    assert bundle["commit"] == commit
    assert len(bundle["sha256"]) == 64
    assert bundle["size_bytes"] > 0
    with tarfile.open(bundle["path"], "r:gz") as archive:
        assert "AGENTS.md" in archive.getnames()


@pytest.mark.parametrize("path", ["../secret", "/absolute/file", ".", "bad\npath"])
def test_runpod_rejects_path_traversal(path: str) -> None:
    with pytest.raises(RUNPOD.LauncherError):
        RUNPOD.safe_relative_path(path, "input")


def test_runpod_provisioned_rate_cost_cap() -> None:
    assert RUNPOD.provisioned_rate_within_cap({"costPerHr": 0.25}, 0.30) == 0.25
    with pytest.raises(RUNPOD.LauncherError, match="exceeds"):
        RUNPOD.provisioned_rate_within_cap({"adjustedCostPerHr": 0.31}, 0.30)
    with pytest.raises(RUNPOD.LauncherError, match="did not return"):
        RUNPOD.provisioned_rate_within_cap({}, 0.30)


def test_runpod_artifact_archive_rejects_traversal(tmp_path: Path) -> None:
    unsafe = tmp_path / "unsafe.tar.gz"
    payload = tmp_path / "payload"
    payload.write_text("unsafe", encoding="utf-8")
    with tarfile.open(unsafe, "w:gz") as archive:
        archive.add(payload, arcname="../payload")

    with pytest.raises(RUNPOD.LauncherError, match="unsafe path"):
        RUNPOD.verify_artifact_archive(unsafe)
