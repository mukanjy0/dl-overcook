"""Package the committed training entry point into the Kaggle skill layout."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import shutil
import subprocess
import zipfile
from io import BytesIO
from pathlib import Path

import yaml


KAGGLE_MAIN = '''"""Generated Kaggle orchestration; core logic remains in project/scripts/train.py."""
from __future__ import annotations

import base64
import hashlib
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
SOURCE_COMMIT = {source_commit!r}
ACCELERATOR = {accelerator!r}
INPUT_MANIFEST_PATH = PROJECT / "remote_input_manifest.json"
ARTIFACT_MANIFEST_PATH = Path("/kaggle/working/artifact_manifest.json")


def write_summary(status: str, **values) -> None:
    payload = {{"status": status, **values}}
    SUMMARY_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def hash_artifacts(root: Path) -> list[dict[str, object]]:
    artifacts = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        artifacts.append(
            {{
                "path": str(path.relative_to(root)),
                "size_bytes": path.stat().st_size,
                "sha256": digest.hexdigest(),
            }}
        )
    return artifacts


write_summary(
    "running",
    config=str(CONFIG_PATH),
    output_root=str(OUTPUT_ROOT),
    source_commit=SOURCE_COMMIT,
    accelerator=ACCELERATOR,
)
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

    if ACCELERATOR == "t4" and not torch.cuda.is_available():
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
    artifact_manifest = hash_artifacts(OUTPUT_ROOT)
    ARTIFACT_MANIFEST_PATH.write_text(
        json.dumps(
            {{"source_commit": SOURCE_COMMIT, "artifacts": artifact_manifest}},
            indent=2,
        ),
        encoding="utf-8",
    )
    write_summary(
        "complete",
        config=str(CONFIG_PATH),
        output_root=str(OUTPUT_ROOT),
        source_commit=SOURCE_COMMIT,
        accelerator=ACCELERATOR,
        cuda_available=bool(torch.cuda.is_available()),
        cuda_device=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        torch_version=str(torch.__version__),
        input_manifest=json.loads(INPUT_MANIFEST_PATH.read_text(encoding="utf-8")),
        artifact_manifest=str(ARTIFACT_MANIFEST_PATH),
        artifact_count=len(artifact_manifest),
        experiment=experiment_summary,
    )
except Exception as exc:
    write_summary(
        "failed",
        error=repr(exc),
        traceback=traceback.format_exc(),
        config=str(CONFIG_PATH),
        output_root=str(OUTPUT_ROOT),
        source_commit=SOURCE_COMMIT,
        accelerator=ACCELERATOR,
    )
    print(traceback.format_exc(), file=sys.stderr)
'''


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _configured_input_assets(
    project_root: Path,
    config_path: Path,
) -> list[tuple[str, Path]]:
    """Return config-declared input files that a remote run must receive."""
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    candidates: list[tuple[str, object]] = []
    checkpoint = raw.get("checkpoint", {}) or {}
    candidates.append(("checkpoint.resume_from", checkpoint.get("resume_from")))
    augmentation = raw.get("state_augmentation", {}) or {}
    if str(augmentation.get("reset_mode", "standard")).lower() != "standard":
        candidates.append(("state_augmentation.buffer_path", augmentation.get("buffer_path")))

    def visit(value: object, prefix: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                child_prefix = f"{prefix}.{key}" if prefix else str(key)
                if key in {"checkpoint_path", "model_path"}:
                    candidates.append((child_prefix, child))
                else:
                    visit(child, child_prefix)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{prefix}[{index}]")

    for section in ("policies", "partner", "evaluation", "collection"):
        visit(raw.get(section, {}), section)

    assets: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    for key, value in candidates:
        if value in (None, ""):
            continue
        configured = Path(str(value)).expanduser()
        if configured.is_absolute():
            raise ValueError(f"Kaggle input {key} must be relative to its YAML")
        source = (config_path.parent / configured).resolve()
        if not source.is_relative_to(project_root):
            raise ValueError(f"Kaggle input {key} must stay inside the project root")
        if not source.is_file():
            raise FileNotFoundError(f"Kaggle input {key} does not exist: {source}")
        if source not in seen:
            seen.add(source)
            assets.append((key, source))
    return assets


def _archive_commit(project_root: Path, commit: str) -> tuple[str, bytes]:
    resolved = subprocess.run(
        ["git", "rev-parse", "--verify", f"{commit}^{{commit}}"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    runtime_paths = (
        "src",
        "policies",
        "scripts/train.py",
        "configs",
        "pyproject.toml",
        "uv.lock",
        "README.md",
    )
    result = subprocess.run(
        ["git", "archive", "--format=zip", resolved, "--", *runtime_paths],
        cwd=project_root,
        check=True,
        capture_output=True,
    )
    return resolved, result.stdout


def _copy_working_project(project_root: Path, destination: Path) -> None:
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


def _prepare_project(
    project_root: Path,
    destination: Path,
    config_path: Path,
    commit: str | None,
) -> tuple[str, str, list[dict[str, object]]]:
    config_relative = str(config_path.relative_to(project_root))
    if commit is None:
        _copy_working_project(project_root, destination)
        source_commit = "working-tree"
    else:
        source_commit, archive = _archive_commit(project_root, commit)
        with zipfile.ZipFile(BytesIO(archive)) as snapshot:
            if config_relative not in snapshot.namelist():
                raise FileNotFoundError(
                    f"Config {config_relative!r} is absent from commit {source_commit}"
                )
            snapshot.extractall(destination)
        archived_config = destination / config_relative
        if archived_config.read_bytes() != config_path.read_bytes():
            raise ValueError(
                f"Config {config_relative!r} differs from immutable commit {source_commit}"
            )

    manifest: list[dict[str, object]] = []
    for config_key, source in _configured_input_assets(project_root, config_path):
        relative = source.relative_to(project_root)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise ValueError(
                f"Tracked input {relative} differs from immutable commit {source_commit}"
            )
        if not target.exists():
            shutil.copy2(source, target)
        manifest.append(
            {
                "config_key": config_key,
                "path": str(relative),
                "size_bytes": source.stat().st_size,
                "sha256": _sha256(source),
            }
        )
    (destination / "remote_input_manifest.json").write_text(
        json.dumps(
            {"source_commit": source_commit, "inputs": manifest},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return config_relative, source_commit, manifest


def package_run(
    *,
    project_root: Path,
    version: str,
    kernel_id: str,
    title: str,
    config_path: Path,
    force: bool,
    commit: str | None = None,
    accelerator: str = "t4",
) -> Path:
    """Create kaggle/<version>/input and outputs without contacting Kaggle."""
    if not re.fullmatch(r"v[0-9]+", version):
        raise ValueError("version must look like v1, v2, ...")
    if "/" not in kernel_id:
        raise ValueError("kernel-id must be '<kaggle-username>/<kernel-slug>'")
    if accelerator not in {"cpu", "t4"}:
        raise ValueError("accelerator must be cpu or t4")
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

    config_relative, source_commit, _ = _prepare_project(
        project_root,
        project_dir,
        config_path,
        commit,
    )
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
            source_commit=source_commit,
            accelerator=accelerator,
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
        "enable_gpu": accelerator == "t4",
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    if accelerator == "t4":
        metadata["machine_shape"] = "NvidiaTeslaT4"
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
    parser.add_argument("--commit", default="HEAD")
    parser.add_argument("--accelerator", choices=("cpu", "t4"), default="t4")
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
        commit=args.commit,
        accelerator=args.accelerator,
    )
    print(result)


if __name__ == "__main__":
    main()
