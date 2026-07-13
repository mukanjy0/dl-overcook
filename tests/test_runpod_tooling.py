"""Offline coverage for the RunPod skill's matrix validation and dry-run guard."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / ".codex/skills/runpod/scripts/runpod_matrix.py"


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
