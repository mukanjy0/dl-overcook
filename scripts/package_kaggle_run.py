"""Package the committed training entry point into the Kaggle skill layout."""

from __future__ import annotations

import argparse
import base64
import json
import re
import shutil
from pathlib import Path

import yaml


KAGGLE_MAIN = '''"""Generated Kaggle orchestration; core logic remains in project/scripts/train.py."""
from __future__ import annotations

import base64
import json
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

PROJECT_ARCHIVE = Path("/kaggle/working/project.zip")
PROJECT_ARCHIVE_B64 = {project_archive_b64!r}
PROJECT = Path("/kaggle/working/project")
OUTPUT_ROOT = Path("/kaggle/working/stage_a")
SUMMARY_PATH = Path("/kaggle/working/run_summary.json")
CONFIG_PATH = PROJECT / {config_relative!r}


def write_summary(status: str, **values) -> None:
    payload = {{"status": status, **values}}
    SUMMARY_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


write_summary("running", config=str(CONFIG_PATH), output_root=str(OUTPUT_ROOT))
try:
    if PROJECT.exists():
        shutil.rmtree(PROJECT)
    PROJECT_ARCHIVE.write_bytes(base64.b64decode(PROJECT_ARCHIVE_B64))
    shutil.unpack_archive(PROJECT_ARCHIVE, PROJECT)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "overcooked-ai==1.1.0",
            "numpy>=1.24,<2",
            "scipy>=1.10,<2",
            "PyYAML>=6.0",
            "Pillow>=10.0",
            "imageio>=2.31",
        ],
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "-e", str(PROJECT)],
        check=True,
    )
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle GPU was requested but torch.cuda.is_available() is false")
    command = [
        sys.executable,
        str(PROJECT / "scripts" / "train.py"),
        "--config",
        str(CONFIG_PATH),
        "--output-root",
        str(OUTPUT_ROOT),
        "--evaluate-checkpoints",
    ]
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Training command exited with code {{result.returncode}}")
    experiment_summary = json.loads(
        (OUTPUT_ROOT / "experiment_summary.json").read_text(encoding="utf-8")
    )
    write_summary(
        "complete",
        config=str(CONFIG_PATH),
        output_root=str(OUTPUT_ROOT),
        cuda_device=torch.cuda.get_device_name(0),
        torch_version=str(torch.__version__),
        experiment=experiment_summary,
    )
except Exception as exc:
    write_summary(
        "failed",
        error=repr(exc),
        traceback=traceback.format_exc(),
        config=str(CONFIG_PATH),
        output_root=str(OUTPUT_ROOT),
    )
    print(traceback.format_exc(), file=sys.stderr)
'''


def _copy_state_buffer_asset(
    project_root: Path,
    destination: Path,
    config_path: Path,
) -> None:
    """Copy an opt-in Stage B buffer while preserving its config-relative path."""
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    state_augmentation = raw.get("state_augmentation", {}) or {}
    if not isinstance(state_augmentation, dict):
        return
    if str(state_augmentation.get("reset_mode", "standard")).lower() == "standard":
        return
    buffer_value = state_augmentation.get("buffer_path")
    if buffer_value in (None, ""):
        return
    configured_path = Path(str(buffer_value)).expanduser()
    if configured_path.is_absolute():
        raise ValueError(
            "Kaggle packaging requires state_augmentation.buffer_path to be "
            "relative to its training YAML"
        )
    source = (config_path.parent / configured_path).resolve()
    if not source.is_relative_to(project_root):
        raise ValueError("The Stage B state buffer must be inside the project root")
    if not source.is_file():
        raise FileNotFoundError(f"Stage B state buffer not found: {source}")
    target = destination / source.relative_to(project_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _copy_project(project_root: Path, destination: Path, config_path: Path) -> str:
    for directory in ("src", "policies"):
        shutil.copytree(
            project_root / directory,
            destination / directory,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
        )
    layouts_directory = project_root / "layouts"
    if layouts_directory.exists():
        shutil.copytree(layouts_directory, destination / "layouts")
    (destination / "scripts").mkdir(parents=True, exist_ok=True)
    shutil.copy2(project_root / "scripts" / "train.py", destination / "scripts" / "train.py")
    shutil.copytree(project_root / "configs", destination / "configs")
    for filename in ("pyproject.toml", "uv.lock", "README.md"):
        source = project_root / filename
        if source.exists():
            shutil.copy2(source, destination / filename)
    _copy_state_buffer_asset(project_root, destination, config_path)
    return str(config_path.relative_to(project_root))


def package_run(
    *,
    project_root: Path,
    version: str,
    kernel_id: str,
    title: str,
    config_path: Path,
    force: bool,
) -> Path:
    """Create kaggle/<version>/input and outputs without contacting Kaggle."""
    if not re.fullmatch(r"v[0-9]+", version):
        raise ValueError("version must look like v1, v2, ...")
    if "/" not in kernel_id:
        raise ValueError("kernel-id must be '<kaggle-username>/<kernel-slug>'")
    package_root = project_root / "kaggle" / version
    if package_root.exists():
        if not force:
            raise FileExistsError(f"Package already exists: {package_root}; pass --force to replace it")
        shutil.rmtree(package_root)
    input_dir = package_root / "input"
    project_dir = input_dir / "project"
    outputs_dir = package_root / "outputs"
    project_dir.mkdir(parents=True)
    outputs_dir.mkdir(parents=True)

    config_relative = _copy_project(project_root, project_dir, config_path)
    shutil.make_archive(
        str(input_dir / "project"),
        "zip",
        root_dir=project_dir,
    )
    project_archive_b64 = base64.b64encode(
        (input_dir / "project.zip").read_bytes()
    ).decode("ascii")
    (input_dir / "main.py").write_text(
        KAGGLE_MAIN.format(
            config_relative=config_relative,
            project_archive_b64=project_archive_b64,
        ),
        encoding="utf-8",
    )
    metadata = {
        "id": kernel_id,
        "title": title,
        "code_file": "main.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "accelerator": "nvidiaTeslaT4",
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (input_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return package_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True, help="Version directory, for example v1")
    parser.add_argument("--kernel-id", required=True, help="Kaggle username/kernel-slug")
    parser.add_argument("--title", required=True)
    parser.add_argument("--config", default="configs/stage_a/train_self_play.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    config_path = (project_root / args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    result = package_run(
        project_root=project_root,
        version=args.version,
        kernel_id=args.kernel_id,
        title=args.title,
        config_path=config_path,
        force=args.force,
    )
    print(result)


if __name__ == "__main__":
    main()
