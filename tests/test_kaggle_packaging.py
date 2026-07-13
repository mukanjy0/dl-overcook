from __future__ import annotations

import ast
import importlib.util
import json
import shutil
from pathlib import Path

import yaml


_PACKAGER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "package_kaggle_run.py"
_SPEC = importlib.util.spec_from_file_location("stage_a_kaggle_packager", _PACKAGER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
package_run = _MODULE.package_run


def test_kaggle_package_uses_skill_layout_and_shared_entrypoint(
    tmp_path: Path,
    project_root: Path,
) -> None:
    packaged_project = tmp_path / "project"
    for directory in ("src", "policies", "configs"):
        shutil.copytree(project_root / directory, packaged_project / directory)
    (packaged_project / "scripts").mkdir()
    shutil.copy2(
        project_root / "scripts" / "train.py",
        packaged_project / "scripts" / "train.py",
    )
    for filename in ("pyproject.toml", "uv.lock", "README.md"):
        shutil.copy2(project_root / filename, packaged_project / filename)

    package_root = package_run(
        project_root=packaged_project,
        version="v1",
        kernel_id="test-user/stage-a",
        title="Stage A test",
        config_path=packaged_project / "configs" / "stage_a" / "train_self_play.yaml",
        force=False,
    )
    input_dir = package_root / "input"
    assert (package_root / "outputs").is_dir()
    assert (input_dir / "project" / "scripts" / "train.py").is_file()
    assert (input_dir / "project.zip").is_file()
    generated_main = (input_dir / "main.py").read_text(encoding="utf-8")
    ast.parse(generated_main)
    assert "run_summary.json" in generated_main
    assert "scripts\" / \"train.py" in generated_main
    assert "PROJECT_ARCHIVE_B64" in generated_main
    assert "base64.b64decode(PROJECT_ARCHIVE_B64)" in generated_main
    assert "shutil.unpack_archive(PROJECT_ARCHIVE, PROJECT)" in generated_main

    metadata = json.loads(
        (input_dir / "kernel-metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["accelerator"] == "nvidiaTeslaT4"
    assert metadata["enable_internet"] is True


def test_kaggle_package_copies_configured_stage_b_buffer(tmp_path: Path) -> None:
    project = tmp_path / "project"
    for directory in ("src", "policies", "configs"):
        (project / directory).mkdir(parents=True, exist_ok=True)
    (project / "scripts").mkdir()
    (project / "scripts" / "train.py").write_text("pass\n", encoding="utf-8")
    (project / "outputs" / "stage_b" / "buffers").mkdir(parents=True)
    buffer_path = project / "outputs" / "stage_b" / "buffers" / "states.json.gz"
    buffer_path.write_bytes(b"fixture")
    config_path = project / "configs" / "stage_b" / "train.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.safe_dump(
            {
                "state_augmentation": {
                    "reset_mode": "mixed",
                    "buffer_path": "../../outputs/stage_b/buffers/states.json.gz",
                }
            }
        ),
        encoding="utf-8",
    )

    package_root = package_run(
        project_root=project,
        version="v1",
        kernel_id="test-user/stage-b",
        title="Stage B test",
        config_path=config_path,
        force=False,
    )

    packaged_buffer = (
        package_root
        / "input"
        / "project"
        / "outputs"
        / "stage_b"
        / "buffers"
        / "states.json.gz"
    )
    assert packaged_buffer.read_bytes() == b"fixture"
