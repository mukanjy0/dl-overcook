from __future__ import annotations

import importlib.util
import json
from pathlib import Path


_PACKAGER_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "package_kaggle_throughput_benchmark.py"
)
_SPEC = importlib.util.spec_from_file_location("kaggle_throughput_packager", _PACKAGER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
benchmark = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(benchmark)


def test_generated_pair_from_repository(tmp_path: Path, monkeypatch) -> None:
    repository = Path(__file__).resolve().parents[1]
    real_archive_commit = benchmark.archive_commit
    monkeypatch.setattr(
        benchmark,
        "archive_commit",
        lambda project_root, commit: real_archive_commit(repository, commit),
    )
    cpu, gpu = benchmark.package_pair(
        project_root=tmp_path,
        commit="ae29971",
        config_relative="configs/stage_a/ablation_baseline_200k.yaml",
        owner="test-owner",
        cpu_version="v1",
        gpu_version="v2",
    )

    assert (cpu / "input/main.py").read_bytes() == (gpu / "input/main.py").read_bytes()
    compile((cpu / "input/main.py").read_text(), "main.py", "exec")
    cpu_metadata = json.loads((cpu / "input/kernel-metadata.json").read_text())
    gpu_metadata = json.loads((gpu / "input/kernel-metadata.json").read_text())
    assert cpu_metadata["enable_gpu"] is False
    assert "accelerator" not in cpu_metadata
    assert gpu_metadata["enable_gpu"] is True
    assert gpu_metadata["accelerator"] == "nvidiaTeslaT4"
