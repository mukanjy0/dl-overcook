from __future__ import annotations

import ast
import importlib.util
import json
import shutil
from pathlib import Path


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
    generated_main = (input_dir / "main.py").read_text(encoding="utf-8")
    ast.parse(generated_main)
    assert "run_summary.json" in generated_main
    assert "scripts\" / \"train.py" in generated_main

    metadata = json.loads(
        (input_dir / "kernel-metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["accelerator"] == "nvidiaTeslaT4"
    assert metadata["enable_internet"] is True
