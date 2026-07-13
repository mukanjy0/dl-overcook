"""Package a CPU/GPU Kaggle throughput benchmark from an immutable Git commit."""

from __future__ import annotations

import argparse
import base64
import json
import shutil
import subprocess
import zipfile
from io import BytesIO
from pathlib import Path


KAGGLE_MAIN_TEMPLATE = r'''"""Generated paired PPO throughput benchmark harness."""
from __future__ import annotations

import base64
import gc
import hashlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import replace
from pathlib import Path


PROJECT_ARCHIVE = Path("/kaggle/working/project.zip")
PROJECT_ARCHIVE_B64 = __PROJECT_ARCHIVE_B64__
PROJECT = Path("/kaggle/working/project")
CONFIG_PATH = PROJECT / __CONFIG_RELATIVE__
OUTPUT_ROOT = Path("/kaggle/working/benchmark")
TRAINING_ROOT = OUTPUT_ROOT / "training"
WARMUP_ROOT = OUTPUT_ROOT / "warmup"
SUMMARY_PATH = Path("/kaggle/working/run_summary.json")
RESULT_PATH = Path("/kaggle/working/benchmark_result.json")
COMMIT = __COMMIT__
WARMUP_STEPS = 1024


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def process_cpu_ticks() -> int:
    values = Path("/proc/self/stat").read_text(encoding="utf-8").split()
    return int(values[13]) + int(values[14])


def system_cpu_ticks() -> tuple[int, int]:
    values = [int(value) for value in Path("/proc/stat").read_text().splitlines()[0].split()[1:]]
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    return sum(values), idle


def resident_memory_mib() -> float:
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            return float(line.split()[1]) / 1024.0
    return 0.0


def gpu_sample() -> tuple[float, float, float] | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    first_gpu = result.stdout.strip().splitlines()[0]
    utilization, used, total = [float(value.strip()) for value in first_gpu.split(",")]
    return utilization, used, total


class ResourceSampler:
    """Sample process RAM and optional GPU counters during timed training."""

    def __init__(self, *, has_cuda: bool, interval_seconds: float = 2.0):
        self.has_cuda = has_cuda
        self.interval_seconds = interval_seconds
        self.rss_mib: list[float] = []
        self.gpu_utilization_percent: list[float] = []
        self.gpu_memory_used_mib: list[float] = []
        self.gpu_memory_total_mib: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="resource-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_seconds + 6.0)
        self._sample()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def _sample(self) -> None:
        self.rss_mib.append(resident_memory_mib())
        if not self.has_cuda:
            return
        sample = gpu_sample()
        if sample is None:
            return
        utilization, used, total = sample
        self.gpu_utilization_percent.append(utilization)
        self.gpu_memory_used_mib.append(used)
        self.gpu_memory_total_mib = total

    def summary(self) -> dict:
        return {
            "sample_interval_seconds": self.interval_seconds,
            "sample_count": len(self.rss_mib),
            "peak_rss_mib": max(self.rss_mib, default=0.0),
            "mean_gpu_utilization_percent": (
                statistics.fmean(self.gpu_utilization_percent)
                if self.gpu_utilization_percent
                else None
            ),
            "peak_gpu_utilization_percent": max(self.gpu_utilization_percent, default=None),
            "peak_gpu_memory_used_mib": max(self.gpu_memory_used_mib, default=None),
            "gpu_memory_total_mib": self.gpu_memory_total_mib,
            "gpu_sample_count": len(self.gpu_utilization_percent),
        }


job_started_at = time.perf_counter()
progress = {"status": "running", "stage": "bootstrap", "commit": COMMIT}
write_json(SUMMARY_PATH, progress)

try:
    phase_started_at = time.perf_counter()
    if PROJECT.exists():
        shutil.rmtree(PROJECT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    PROJECT_ARCHIVE.write_bytes(base64.b64decode(PROJECT_ARCHIVE_B64))
    shutil.unpack_archive(PROJECT_ARCHIVE, PROJECT)
    archive_setup_seconds = time.perf_counter() - phase_started_at

    phase_started_at = time.perf_counter()
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
    dependency_setup_seconds = time.perf_counter() - phase_started_at

    phase_started_at = time.perf_counter()
    import numpy
    import torch
    import yaml
    from src.checkpointing import CheckpointLoader, inspect_training_checkpoint
    from src.experiment_config import load_experiment_config
    from src.training.ppo import PPOUpdater
    from src.training.rollouts import SelfPlayRolloutCollector
    from src.training.trainer import train
    import_setup_seconds = time.perf_counter() - phase_started_at

    config_bytes = CONFIG_PATH.read_bytes()
    raw_config = yaml.safe_load(config_bytes)
    config = load_experiment_config(CONFIG_PATH)
    if config.training.total_steps != 200000:
        raise ValueError(f"Benchmark config must use exactly 200000 steps, got {config.training.total_steps}")
    if config.training.num_environments * config.training.rollout_steps != WARMUP_STEPS:
        raise ValueError("Warm-up must be exactly one unchanged rollout/update")

    progress.update({"stage": "warmup", "config": str(CONFIG_PATH)})
    write_json(SUMMARY_PATH, progress)
    warmup_config = replace(
        config,
        training=replace(config.training, total_steps=WARMUP_STEPS),
    )
    warmup_started_at = time.perf_counter()
    warmup_summary = train(warmup_config, output_root_override=WARMUP_ROOT)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    warmup_seconds = time.perf_counter() - warmup_started_at
    shutil.rmtree(WARMUP_ROOT)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    phase_times = {"rollout_collection_seconds": 0.0, "ppo_update_seconds": 0.0}
    phase_counts = {"rollout_collections": 0, "ppo_updates": 0}
    original_collect = SelfPlayRolloutCollector.collect
    original_update = PPOUpdater.update

    def synchronize() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def timed_collect(self, *args, **kwargs):
        synchronize()
        started_at = time.perf_counter()
        result = original_collect(self, *args, **kwargs)
        synchronize()
        phase_times["rollout_collection_seconds"] += time.perf_counter() - started_at
        phase_counts["rollout_collections"] += 1
        return result

    def timed_update(self, *args, **kwargs):
        synchronize()
        started_at = time.perf_counter()
        result = original_update(self, *args, **kwargs)
        synchronize()
        phase_times["ppo_update_seconds"] += time.perf_counter() - started_at
        phase_counts["ppo_updates"] += 1
        return result

    SelfPlayRolloutCollector.collect = timed_collect
    PPOUpdater.update = timed_update

    progress["stage"] = "timed_training"
    write_json(SUMMARY_PATH, progress)
    clock_ticks = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
    process_ticks_before = process_cpu_ticks()
    system_total_before, system_idle_before = system_cpu_ticks()
    sampler = ResourceSampler(has_cuda=torch.cuda.is_available())
    sampler.start()
    training_started_at = time.perf_counter()
    try:
        training_summary = train(config, output_root_override=TRAINING_ROOT)
        synchronize()
    finally:
        training_wall_seconds = time.perf_counter() - training_started_at
        sampler.stop()
        SelfPlayRolloutCollector.collect = original_collect
        PPOUpdater.update = original_update
    process_ticks_after = process_cpu_ticks()
    system_total_after, system_idle_after = system_cpu_ticks()

    progress["stage"] = "artifact_validation"
    write_json(SUMMARY_PATH, progress)
    required_paths = {
        "training_summary": TRAINING_ROOT / "run_summary.json",
        "training_checkpoint": TRAINING_ROOT / "checkpoints" / "training_final.pt",
        "inference_artifact": TRAINING_ROOT / "checkpoints" / "inference.pt",
        "effective_config": TRAINING_ROOT / "effective_config.yaml",
        "metrics": TRAINING_ROOT / "metrics" / "training.jsonl",
    }
    missing = [name for name, path in required_paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing benchmark artifacts: {missing}")
    checkpoint_metadata = inspect_training_checkpoint(required_paths["training_checkpoint"])
    loaded_inference = CheckpointLoader.load_inference(
        required_paths["inference_artifact"], device="cpu"
    )
    metrics_lines = sum(1 for line in required_paths["metrics"].read_text().splitlines() if line)
    actual_steps = int(training_summary["environment_steps"])
    if actual_steps < config.training.total_steps:
        raise RuntimeError(f"Training stopped early at {actual_steps} steps")
    if checkpoint_metadata["environment_steps"] != actual_steps:
        raise RuntimeError("Training checkpoint step metadata does not match the summary")
    if metrics_lines != int(training_summary["updates"]):
        raise RuntimeError("Metrics line count does not match the PPO update count")

    process_cpu_seconds = (process_ticks_after - process_ticks_before) / float(clock_ticks)
    system_total_delta = system_total_after - system_total_before
    system_busy_delta = system_total_delta - (system_idle_after - system_idle_before)
    cpu_count = os.cpu_count() or 1
    resources = sampler.summary()
    resources.update(
        {
            "logical_cpu_count": cpu_count,
            "process_cpu_seconds": process_cpu_seconds,
            "process_cpu_percent_one_core_equivalent": 100.0 * process_cpu_seconds / training_wall_seconds,
            "process_cpu_percent_of_machine": 100.0 * process_cpu_seconds / (training_wall_seconds * cpu_count),
            "system_cpu_utilization_percent": (
                100.0 * system_busy_delta / system_total_delta if system_total_delta > 0 else None
            ),
            "torch_peak_cuda_allocated_mib": (
                torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                if torch.cuda.is_available()
                else None
            ),
            "torch_peak_cuda_reserved_mib": (
                torch.cuda.max_memory_reserved() / (1024.0 * 1024.0)
                if torch.cuda.is_available()
                else None
            ),
        }
    )

    setup_seconds = archive_setup_seconds + dependency_setup_seconds + import_setup_seconds
    phase_accounted = phase_times["rollout_collection_seconds"] + phase_times["ppo_update_seconds"]
    result = {
        "status": "complete",
        "commit": COMMIT,
        "config_path": str(CONFIG_PATH),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "configuration": {
            "experiment_name": config.experiment.name,
            "seed": config.experiment.seed,
            "requested_device": config.experiment.device,
            "resolved_device": training_summary["device"],
            "layout_name": config.environment.config.get("layout_name"),
            "horizon": config.environment.config.get("horizon"),
            "total_steps_requested": config.training.total_steps,
            "environment_steps_completed": actual_steps,
            "num_environments": config.training.num_environments,
            "rollout_steps": config.training.rollout_steps,
            "ppo": raw_config["training"]["ppo"],
        },
        "timing": {
            "archive_setup_seconds": archive_setup_seconds,
            "dependency_setup_seconds": dependency_setup_seconds,
            "import_setup_seconds": import_setup_seconds,
            "startup_setup_seconds": setup_seconds,
            "warmup_steps": int(warmup_summary["environment_steps"]),
            "warmup_seconds_excluded": warmup_seconds,
            "timed_training_wall_seconds": training_wall_seconds,
            "environment_steps_per_second": actual_steps / training_wall_seconds,
            **phase_times,
            **phase_counts,
            "unattributed_training_seconds": max(training_wall_seconds - phase_accounted, 0.0),
        },
        "resources": resources,
        "runtime": {
            "python": platform.python_version(),
            "numpy": numpy.__version__,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "platform": platform.platform(),
        },
        "artifact_integrity": {
            "ok": True,
            "metrics_lines": metrics_lines,
            "checkpoint_environment_steps": checkpoint_metadata["environment_steps"],
            "inference_schema_version": loaded_inference.metadata["schema_version"],
            "files": {
                name: {
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
                for name, path in required_paths.items()
            },
        },
    }
    result["timing"]["post_training_validation_seconds"] = (
        time.perf_counter() - training_started_at - training_wall_seconds
    )
    result["timing"]["total_kernel_wall_seconds"] = time.perf_counter() - job_started_at
    write_json(RESULT_PATH, result)
    write_json(SUMMARY_PATH, result)
    print(json.dumps(result, indent=2, sort_keys=True))
except Exception as exc:
    progress.update(
        {
            "status": "failed",
            "stage": progress.get("stage"),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "total_kernel_wall_seconds": time.perf_counter() - job_started_at,
        }
    )
    write_json(SUMMARY_PATH, progress)
    write_json(RESULT_PATH, progress)
    print(traceback.format_exc(), file=sys.stderr)
'''


def archive_commit(project_root: Path, commit: str) -> bytes:
    """Return one immutable repository snapshot as a zip archive."""
    result = subprocess.run(
        ["git", "archive", "--format=zip", commit],
        cwd=project_root,
        check=True,
        capture_output=True,
    )
    return result.stdout


def _benchmark_main(archive: bytes, *, commit: str, config_relative: str) -> str:
    encoded = base64.b64encode(archive).decode("ascii")
    return (
        KAGGLE_MAIN_TEMPLATE.replace("__PROJECT_ARCHIVE_B64__", repr(encoded))
        .replace("__CONFIG_RELATIVE__", repr(config_relative))
        .replace("__COMMIT__", repr(commit))
    )


def _write_package(
    *,
    project_root: Path,
    version: str,
    kernel_id: str,
    title: str,
    main_source: str,
    use_gpu: bool,
    force: bool,
) -> Path:
    package_root = project_root / "kaggle" / version
    if package_root.exists():
        if not force:
            raise FileExistsError(f"Package already exists: {package_root}")
        shutil.rmtree(package_root)
    input_dir = package_root / "input"
    (package_root / "outputs").mkdir(parents=True)
    input_dir.mkdir(parents=True)
    (input_dir / "main.py").write_text(main_source, encoding="utf-8")
    metadata = {
        "id": kernel_id,
        "title": title,
        "code_file": "main.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": use_gpu,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    if use_gpu:
        metadata["accelerator"] = "nvidiaTeslaT4"
    (input_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return package_root


def package_pair(
    *,
    project_root: Path,
    commit: str,
    config_relative: str,
    owner: str,
    cpu_version: str,
    gpu_version: str,
    force: bool = False,
) -> tuple[Path, Path]:
    """Create byte-identical harnesses whose metadata differ only by accelerator and identity."""
    archive = archive_commit(project_root, commit)
    with zipfile.ZipFile(BytesIO(archive)) as source:
        if config_relative not in source.namelist():
            raise FileNotFoundError(f"Config {config_relative!r} is absent from commit {commit}")
    main_source = _benchmark_main(
        archive,
        commit=commit,
        config_relative=config_relative,
    )
    cpu = _write_package(
        project_root=project_root,
        version=cpu_version,
        kernel_id=f"{owner}/overcook-ppo-throughput-cpu-{commit[:7]}",
        title=f"overcook-ppo-throughput-cpu-{commit[:7]}",
        main_source=main_source,
        use_gpu=False,
        force=force,
    )
    gpu = _write_package(
        project_root=project_root,
        version=gpu_version,
        kernel_id=f"{owner}/overcook-ppo-throughput-t4-{commit[:7]}",
        title=f"overcook-ppo-throughput-t4-{commit[:7]}",
        main_source=main_source,
        use_gpu=True,
        force=force,
    )
    return cpu, gpu


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--cpu-version", required=True)
    parser.add_argument("--gpu-version", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    cpu, gpu = package_pair(
        project_root=project_root,
        commit=args.commit,
        config_relative=args.config,
        owner=args.owner,
        cpu_version=args.cpu_version,
        gpu_version=args.gpu_version,
        force=args.force,
    )
    print(cpu)
    print(gpu)


if __name__ == "__main__":
    main()
