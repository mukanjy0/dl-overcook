#!/usr/bin/env python3
"""Safely launch independent RunPod experiment jobs from a YAML or JSON matrix."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


REST_URL = "https://rest.runpod.io/v1"
GRAPHQL_URL = "https://api.runpod.io/graphql"
NAME_RE = re.compile(r"[^a-zA-Z0-9-]+")
STATE_LOCK = threading.Lock()


class LauncherError(RuntimeError):
    """Report a safe, actionable launcher error."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def slug(value: str) -> str:
    return NAME_RE.sub("-", value).strip("-")[:70] or "job"


def api_key(required: bool = True) -> str | None:
    value = os.environ.get("RUNPOD_API_KEY")
    if not value:
        config_path = Path(os.environ.get("RUNPOD_CONFIG_PATH", "~/.runpod/config.toml")).expanduser()
        if config_path.is_file():
            try:
                import tomllib

                profile = tomllib.loads(config_path.read_text(encoding="utf-8"))
                value = profile.get("apiKey") or profile.get("api_key") or profile.get("apikey")
            except (OSError, ValueError):
                value = None
    if required and not value:
        raise LauncherError("Set RUNPOD_API_KEY or run `runpodctl doctor`; never put the key in a matrix file.")
    return value


def graphql_url() -> str:
    """Read the non-secret API URL selected by runpodctl doctor when available."""
    config_path = Path(os.environ.get("RUNPOD_CONFIG_PATH", "~/.runpod/config.toml")).expanduser()
    if config_path.is_file():
        try:
            import tomllib

            profile = tomllib.loads(config_path.read_text(encoding="utf-8"))
            return str(profile.get("apiUrl") or profile.get("apiurl") or GRAPHQL_URL)
        except (OSError, ValueError):
            pass
    return GRAPHQL_URL


def request_json(method: str, url: str, *, body: dict[str, Any] | None = None) -> Any:
    key = api_key()
    payload = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        url,
        data=payload,
        method=method,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            raw = response.read().decode()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:500]
        raise LauncherError(f"RunPod API {method} failed ({exc.code}): {detail}") from exc
    except urllib.error.URLError as exc:
        raise LauncherError(f"Could not reach RunPod API: {exc.reason}") from exc
    return json.loads(raw) if raw else None


def rest(method: str, path: str, body: dict[str, Any] | None = None) -> Any:
    return request_json(method, f"{REST_URL}{path}", body=body)


def graphql(query: str) -> dict[str, Any]:
    key = api_key()
    encoded_key = urllib.parse.quote(key, safe="")
    payload = json.dumps({"query": query}).encode()
    request = urllib.request.Request(
        f"{graphql_url()}?api_key={encoded_key}",
        data=payload,
        method="POST",
        # Match the official CLI's non-browser request class; some accounts
        # reject Python's default user agent at the GraphQL edge.
        headers={"Content-Type": "application/json", "User-Agent": "runpodctl"},
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as result:
            response = json.loads(result.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:500]
        raise LauncherError(f"RunPod GraphQL estimate failed ({exc.code}): {detail}") from exc
    except urllib.error.URLError as exc:
        raise LauncherError(f"Could not reach RunPod GraphQL API: {exc.reason}") from exc
    if response.get("errors"):
        raise LauncherError("RunPod GraphQL estimate failed: " + json.dumps(response["errors"])[:500])
    return response["data"]


def run_local(args: list[str], *, cwd: Path | None = None) -> str:
    try:
        return subprocess.check_output(args, cwd=cwd, text=True, stderr=subprocess.PIPE).strip()
    except subprocess.CalledProcessError as exc:
        raise LauncherError(exc.stderr.strip() or "Command failed: " + " ".join(args)) from exc


def load_matrix(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise LauncherError(f"Matrix file does not exist: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        try:
            import yaml  # type: ignore[import-not-found]
        except ImportError as exc:
            raise LauncherError("YAML matrices require PyYAML; run the repository setup command first.") from exc
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise LauncherError("The matrix root must be a mapping.")
    return data


def current_repository() -> tuple[str, str]:
    return run_local(["git", "remote", "get-url", "origin"]), run_local(["git", "rev-parse", "HEAD"])


def current_checkout() -> tuple[Path, str]:
    root = Path(run_local(["git", "rev-parse", "--show-toplevel"])).resolve()
    return root, run_local(["git", "rev-parse", "HEAD"], cwd=root)


def resolved_matrix(matrix: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(matrix))
    repository = result.setdefault("repository", {})
    if not isinstance(repository, dict):
        raise LauncherError("repository must be a mapping.")
    source = str(repository.get("source", "git")).lower()
    repository["source"] = source
    if source == "git":
        if not repository.get("url") or not repository.get("commit"):
            url, commit = current_repository()
            repository.setdefault("url", url)
            repository.setdefault("commit", commit)
    elif source == "archive":
        if not repository.get("commit"):
            _, commit = current_checkout()
            repository["commit"] = commit
    else:
        raise LauncherError("repository.source must be git or archive.")
    return result


def safe_relative_path(value: object, label: str) -> Path:
    raw = str(value)
    if any(character in raw for character in ("\n", "\r", "\x00")):
        raise LauncherError(f"{label} contains an unsupported control character.")
    path = Path(raw)
    if path.is_absolute() or ".." in path.parts or path in {Path(""), Path(".")}:
        raise LauncherError(f"{label} must be a non-empty relative path without traversal.")
    return path


def require(value: Any, label: str) -> Any:
    if value is None or value == "" or value == []:
        raise LauncherError(f"Missing required matrix field: {label}")
    return value


def validate(matrix: dict[str, Any]) -> dict[str, Any]:
    matrix = resolved_matrix(matrix)
    repo = require(matrix.get("repository"), "repository")
    if repo["source"] == "git":
        require(repo.get("url"), "repository.url")
    require(repo.get("commit"), "repository.commit")
    require(repo.get("setup_command"), "repository.setup_command")
    pod = require(matrix.get("pod"), "pod")
    compute = str(pod.get("compute_type", "GPU")).upper()
    if compute not in {"GPU", "CPU"}:
        raise LauncherError("pod.compute_type must be GPU or CPU.")
    require(pod.get("image"), "pod.image")
    if compute == "GPU":
        require(pod.get("gpu_type_ids"), "pod.gpu_type_ids")
        if not isinstance(pod["gpu_type_ids"], list) or not all(isinstance(v, str) for v in pod["gpu_type_ids"]):
            raise LauncherError("pod.gpu_type_ids must be a list of RunPod GPU IDs.")
    else:
        require(pod.get("cpu_flavor_ids"), "pod.cpu_flavor_ids")
        if not isinstance(pod["cpu_flavor_ids"], list) or not all(
            isinstance(value, str) for value in pod["cpu_flavor_ids"]
        ):
            raise LauncherError("pod.cpu_flavor_ids must be a list of RunPod CPU flavor IDs.")
        try:
            vcpu_count = int(require(pod.get("vcpu_count"), "pod.vcpu_count"))
        except (TypeError, ValueError) as exc:
            raise LauncherError("pod.vcpu_count must be a positive integer.") from exc
        if vcpu_count <= 0:
            raise LauncherError("pod.vcpu_count must be a positive integer.")
    jobs = require(matrix.get("jobs"), "jobs")
    if not isinstance(jobs, list) or not jobs:
        raise LauncherError("jobs must be a non-empty list.")
    seen: set[str] = set()
    for index, job in enumerate(jobs):
        if not isinstance(job, dict):
            raise LauncherError(f"jobs[{index}] must be a mapping.")
        name = str(require(job.get("name"), f"jobs[{index}].name"))
        if name in seen:
            raise LauncherError(f"Job names must be unique: {name}")
        seen.add(name)
        require(job.get("command"), f"jobs[{index}].command")
        paths = require(job.get("artifact_paths"), f"jobs[{index}].artifact_paths")
        if not isinstance(paths, list) or not all(isinstance(v, str) for v in paths):
            raise LauncherError(f"jobs[{index}].artifact_paths must be a list of relative paths.")
        for path_index, item in enumerate(paths):
            safe_relative_path(item, f"jobs[{index}].artifact_paths[{path_index}]")
        inputs = job.get("input_paths", [])
        if not isinstance(inputs, list):
            raise LauncherError(f"jobs[{index}].input_paths must be a list.")
        for input_index, item in enumerate(inputs):
            if not isinstance(item, dict):
                raise LauncherError(f"jobs[{index}].input_paths[{input_index}] must be a mapping.")
            safe_relative_path(
                require(item.get("source"), f"jobs[{index}].input_paths[{input_index}].source"),
                f"jobs[{index}].input_paths[{input_index}].source",
            )
            safe_relative_path(
                require(item.get("destination"), f"jobs[{index}].input_paths[{input_index}].destination"),
                f"jobs[{index}].input_paths[{input_index}].destination",
            )
    return matrix


def state_path(output_dir: Path) -> Path:
    return output_dir / "runpod_state.json"


def write_state(output_dir: Path, state: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = state_path(output_dir)
    with STATE_LOCK:
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, destination)


def read_state(manifest: Path) -> dict[str, Any]:
    if not manifest.is_file():
        raise LauncherError(f"Manifest does not exist: {manifest}")
    return json.loads(manifest.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_source_archive(
    repository: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any] | None:
    """Create a byte-stable archive of the requested local commit."""
    if repository["source"] != "archive":
        return None
    project_root, _ = current_checkout()
    commit = run_local(
        ["git", "rev-parse", "--verify", f"{repository['commit']}^{{commit}}"],
        cwd=project_root,
    )
    repository["commit"] = commit
    transfer_dir = output_dir / "transfer_inputs"
    transfer_dir.mkdir(parents=True, exist_ok=True)
    archive = transfer_dir / f"source-{commit}.tar.gz"
    if not archive.is_file():
        try:
            subprocess.run(
                ["git", "archive", "--format=tar.gz", "--output", str(archive), commit],
                cwd=project_root,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            raise LauncherError(exc.stderr.strip() or "Could not create immutable source archive.") from exc
    return {
        "path": str(archive),
        "commit": commit,
        "size_bytes": archive.stat().st_size,
        "sha256": sha256_file(archive),
    }


def prepare_job_inputs(
    job: dict[str, Any],
    job_slug: str,
    output_dir: Path,
) -> dict[str, Any] | None:
    """Archive explicitly declared local files at safe repository destinations."""
    inputs = job.get("input_paths", [])
    if not inputs:
        return None
    project_root, _ = current_checkout()
    transfer_dir = output_dir / "transfer_inputs"
    transfer_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = transfer_dir / f"{job_slug}-input-manifest.json"
    archive_path = transfer_dir / f"{job_slug}-inputs.tar.gz"
    manifest: list[dict[str, Any]] = []
    resolved: list[tuple[Path, Path]] = []
    for index, item in enumerate(inputs):
        source_relative = safe_relative_path(item["source"], f"input_paths[{index}].source")
        destination = safe_relative_path(
            item["destination"], f"input_paths[{index}].destination"
        )
        source = (project_root / source_relative).resolve()
        if not source.is_relative_to(project_root):
            raise LauncherError(f"Input source escapes the checkout: {source_relative}")
        if not source.is_file():
            raise LauncherError(f"Input source is not a file: {source_relative}")
        resolved.append((source, destination))
        manifest.append(
            {
                "source": str(source_relative),
                "destination": str(destination),
                "size_bytes": source.stat().st_size,
                "sha256": sha256_file(source),
            }
        )
    manifest_path.write_text(
        json.dumps({"job": job["name"], "inputs": manifest}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with tarfile.open(archive_path, "w:gz") as archive:
        for source, destination in resolved:
            archive.add(source, arcname=str(destination), recursive=False)
        archive.add(manifest_path, arcname=".runpod-input-manifest.json", recursive=False)
    return {
        "path": str(archive_path),
        "size_bytes": archive_path.stat().st_size,
        "sha256": sha256_file(archive_path),
        "inputs": manifest,
    }


def prepare_transfers(
    state: dict[str, Any],
    matrix: dict[str, Any],
    output_dir: Path,
) -> None:
    source = prepare_source_archive(matrix["repository"], output_dir)
    state["source_archive"] = source
    for job_state in state["jobs"]:
        job_state["input_archive"] = prepare_job_inputs(
            job_state["matrix_job"], job_state["slug"], output_dir
        )
    write_state(output_dir, state)


def default_output(matrix_path: Path) -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    return Path("outputs/runpod") / f"{slug(matrix_path.stem)}-{stamp}"


def ssh_key_path(value: str | None) -> Path:
    if value or os.environ.get("RUNPOD_SSH_KEY_PATH"):
        candidates = [Path(value or os.environ["RUNPOD_SSH_KEY_PATH"]).expanduser()]
    else:
        candidates = [Path("~/.runpod/ssh/runpodctl-ssh-key").expanduser(), Path("~/.ssh/id_ed25519").expanduser()]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise LauncherError("SSH private key not found. Set RUNPOD_SSH_KEY_PATH or --ssh-key.")


def connection(pod: dict[str, Any], key: Path) -> list[str]:
    public_ip = pod.get("publicIp")
    port = (pod.get("portMappings") or {}).get("22")
    if not public_ip or not port:
        raise LauncherError("Pod is not ready for SSH yet.")
    return ["ssh", "-i", str(key), "-p", str(port), "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", f"root@{public_ip}"]


def wait_for_ssh(pod_id: str, key: Path, timeout_minutes: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_minutes * 60
    last_error = ""
    while time.monotonic() < deadline:
        pod = rest("GET", f"/pods/{pod_id}")
        try:
            ssh = connection(pod, key)
            completed = subprocess.run(ssh + ["true"], text=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=20)
            if completed.returncode == 0:
                return pod
            last_error = completed.stderr[-240:]
        except (LauncherError, subprocess.SubprocessError) as exc:
            last_error = str(exc)
        time.sleep(10)
    raise LauncherError(f"Timed out waiting for Pod {pod_id} SSH readiness: {last_error}")


def pod_payload(matrix: dict[str, Any], name: str, job: dict[str, Any], run_id: str) -> dict[str, Any]:
    pod = matrix["pod"]
    compute = str(pod.get("compute_type", "GPU")).upper()
    mount = str(pod.get("volume_mount_path", "/workspace"))
    payload: dict[str, Any] = {
        "name": name,
        "imageName": pod["image"],
        "computeType": compute,
        "cloudType": str(pod.get("cloud_type", "SECURE")).upper(),
        "containerDiskInGb": int(pod.get("container_disk_gb", 30)),
        "volumeInGb": int(pod.get("volume_gb", 20)),
        "volumeMountPath": mount,
        "ports": ["22/tcp"],
        "env": {"RUNPOD_RUN_ID": run_id},
    }
    if compute == "GPU":
        payload.update({"gpuTypeIds": pod["gpu_type_ids"], "gpuCount": int(pod.get("gpu_count", 1)), "gpuTypePriority": pod.get("gpu_type_priority", "custom"), "interruptible": bool(pod.get("interruptible", False))})
    else:
        payload.update(
            {
                "cpuFlavorIds": pod["cpu_flavor_ids"],
                "vcpuCount": int(pod["vcpu_count"]),
                "cpuFlavorPriority": pod.get("cpu_flavor_priority", "custom"),
                "interruptible": bool(pod.get("interruptible", False)),
            }
        )
    if pod.get("network_volume_id"):
        payload["networkVolumeId"] = pod["network_volume_id"]
    if pod.get("data_center_ids"):
        payload["dataCenterIds"] = pod["data_center_ids"]
    if pod.get("extra_env"):
        payload["env"].update(pod["extra_env"])
    if os.environ.get("RUNPOD_GIT_TOKEN"):
        payload["env"]["RUNPOD_GIT_TOKEN"] = os.environ["RUNPOD_GIT_TOKEN"]
    return {key: value for key, value in payload.items() if value is not None}


BOOTSTRAP = r'''#!/usr/bin/env bash
set -euo pipefail
workspace="$1"
source_mode="$2"
repo_url="$3"
commit="$4"
setup_command="$5"
job_command="$6"
job_name="$7"
artifact_paths="$8"
source_archive="$9"
source_sha="${10}"
input_archive="${11}"
input_sha="${12}"
mkdir -p "$workspace"
output="$workspace/result"
repo="$workspace/repository"
mkdir -p "$output"
started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [[ "$source_mode" == "archive" ]]; then
  [[ "$(sha256sum "$source_archive" | awk '{print $1}')" == "$source_sha" ]]
  mkdir -p "$repo"
  tar -xzf "$source_archive" -C "$repo"
else
  if [[ -n "${RUNPOD_GIT_TOKEN:-}" ]]; then
    GIT_TERMINAL_PROMPT=0 git -c http.extraHeader="Authorization: Bearer ${RUNPOD_GIT_TOKEN}" clone --no-checkout "$repo_url" "$repo"
  else
    GIT_TERMINAL_PROMPT=0 git clone --no-checkout "$repo_url" "$repo"
  fi
  git -C "$repo" checkout --detach "$commit"
fi
if [[ -n "$input_archive" ]]; then
  [[ "$(sha256sum "$input_archive" | awk '{print $1}')" == "$input_sha" ]]
  tar -xzf "$input_archive" -C "$repo"
fi
cd "$repo"
if ! command -v uv >/dev/null 2>&1; then
  python3 -m pip install --no-cache-dir uv >"$output/uv-install.log" 2>&1
fi
bash -lc "$setup_command" >"$output/setup.log" 2>&1
set +e
bash -lc "$job_command" >"$output/job.log" 2>&1
status=$?
set -e
python3 - "$output/metadata.json" "$job_name" "$commit" "$started_at" "$status" <<'PY'
import json, sys
from datetime import datetime, timezone
Path = __import__('pathlib').Path
Path(sys.argv[1]).write_text(json.dumps({"job": sys.argv[2], "commit": sys.argv[3], "started_at": sys.argv[4], "finished_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(), "exit_code": int(sys.argv[5])}, indent=2) + "\n")
PY
mkdir -p "$repo/.runpod-launcher-metadata"
cp "$output/job.log" "$output/setup.log" "$output/metadata.json" "$repo/.runpod-launcher-metadata/"
[[ -f "$output/uv-install.log" ]] && cp "$output/uv-install.log" "$repo/.runpod-launcher-metadata/"
printf '%s\n' "$artifact_paths" >"$output/artifact-paths.txt"
tar -C "$repo" --ignore-failed-read --verbatim-files-from -czf "$output/artifacts.tar.gz" -T "$output/artifact-paths.txt" .runpod-launcher-metadata
rm -rf "$repo/.runpod-launcher-metadata"
exit "$status"
'''


def run_remote(ssh: list[str], script: str, args: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # ssh sends its remote command through a shell. Quote every positional
    # argument so setup and job commands retain their boundaries remotely.
    remote_command = "bash -s -- " + " ".join(shlex.quote(value) for value in args)
    command = ssh + [remote_command]
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, input=script, text=True, stdout=log, stderr=subprocess.STDOUT)
    return completed.returncode


def upload_file(ssh: list[str], pod: dict[str, Any], source: Path, remote: str) -> None:
    remote_parent = str(Path(remote).parent)
    subprocess.run(ssh + ["mkdir", "-p", remote_parent], check=True)
    scp = [
        "scp",
        "-i",
        ssh[ssh.index("-i") + 1],
        "-P",
        ssh[ssh.index("-p") + 1],
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        str(source),
        f"root@{pod['publicIp']}:{remote}",
    ]
    subprocess.run(scp, check=True)


def verify_artifact_archive(path: Path) -> dict[str, Any]:
    """Validate archive readability/traversal and return integrity metadata."""
    try:
        with tarfile.open(path, "r:gz") as archive:
            members = archive.getmembers()
    except (tarfile.TarError, OSError) as exc:
        raise LauncherError(f"Fetched artifact archive is invalid: {path}") from exc
    for member in members:
        member_path = Path(member.name)
        if member_path.is_absolute() or ".." in member_path.parts:
            raise LauncherError(f"Fetched artifact archive contains unsafe path: {member.name}")
    return {
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "member_count": len(members),
    }


def fetch_job(job_state: dict[str, Any], key: Path, output_dir: Path) -> None:
    pod = rest("GET", f"/pods/{job_state['pod_id']}")
    ssh = connection(pod, key)
    remote = job_state["remote_result"] + "/artifacts.tar.gz"
    local = output_dir / job_state["slug"] / "artifacts.tar.gz"
    local.parent.mkdir(parents=True, exist_ok=True)
    scp = ["scp", "-i", str(key), "-P", ssh[ssh.index("-p") + 1], "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", f"root@{pod['publicIp']}:{remote}", str(local)]
    subprocess.run(scp, check=True)
    job_state["artifact_integrity"] = verify_artifact_archive(local)
    job_state["artifacts_fetched_at"] = utc_now()


def terminate_pod(pod_id: str) -> str:
    try:
        rest("DELETE", f"/pods/{pod_id}")
        return "terminated"
    except LauncherError as exc:
        if "(404)" in str(exc):
            return "already_absent"
        raise


def cleanup(state: dict[str, Any], output_dir: Path) -> int:
    ids = {job.get("pod_id") for job in state["jobs"] if job.get("pod_id")}
    try:
        for pod in rest("GET", "/pods"):
            if str(pod.get("name", "")).startswith(state["pod_name_prefix"]):
                ids.add(pod["id"])
    except LauncherError:
        pass
    errors = []
    for pod_id in sorted(ids):
        try:
            result = terminate_pod(pod_id)
            for job in state["jobs"]:
                if job.get("pod_id") == pod_id:
                    job["termination"] = result
        except LauncherError as exc:
            errors.append(f"{pod_id}: {exc}")
    state["cleanup_at"] = utc_now()
    write_state(output_dir, state)
    if errors:
        raise LauncherError("Some Pods could not be terminated: " + "; ".join(errors))
    return len(ids)


def resolve_command(job: dict[str, Any], remote_result: str) -> str:
    try:
        return str(job["command"]).format(config=job.get("config", ""), artifact_dir=remote_result)
    except (KeyError, ValueError) as exc:
        raise LauncherError(f"Job {job['name']} command has invalid template: {exc}") from exc


def provisioned_rate_within_cap(pod: dict[str, Any], maximum_rate: float) -> float:
    raw_rate = pod.get("adjustedCostPerHr", pod.get("costPerHr"))
    if raw_rate is None:
        raise LauncherError("RunPod did not return the provisioned hourly rate.")
    rate = float(raw_rate)
    if rate > maximum_rate:
        raise LauncherError(
            f"Provisioned rate ${rate:.4f}/hour exceeds the ${maximum_rate:.4f}/hour cap."
        )
    return rate


def launch_one(state: dict[str, Any], job_state: dict[str, Any], matrix: dict[str, Any], output_dir: Path, key: Path) -> None:
    job = job_state["matrix_job"]
    if job_state.get("pod_id"):
        return
    pod = rest("POST", "/pods", pod_payload(matrix, job_state["pod_name"], job, state["run_id"]))
    raw_rate = pod.get("adjustedCostPerHr", pod.get("costPerHr"))
    rate = float(raw_rate) if raw_rate is not None else None
    cap_error: LauncherError | None = None
    try:
        provisioned_rate_within_cap(pod, float(state["max_hourly_rate"]))
    except LauncherError as exc:
        cap_error = exc
    job_state.update({"pod_id": pod["id"], "provisioned_at": utc_now(), "provisioned_cost_per_hour": rate, "status": "provisioned"})
    write_state(output_dir, state)
    if cap_error is not None:
        job_state["status"] = "cost_cap_rejected"
        job_state["termination"] = terminate_pod(pod["id"])
        write_state(output_dir, state)
        raise LauncherError(
            f"Job {job['name']} failed its cost cap: {cap_error} Pod was terminated."
        )
    timeout = float(matrix["pod"].get("readiness_timeout_minutes", 15))
    ready = wait_for_ssh(pod["id"], key, timeout)
    remote_root = str(matrix["pod"].get("volume_mount_path", "/workspace")).rstrip("/")
    remote_workspace = f"{remote_root}/runpod/{state['run_id']}/{job_state['slug']}"
    remote_result = remote_workspace + "/result"
    job_state.update({"status": "running", "remote_result": remote_result, "ssh_ready_at": utc_now()})
    write_state(output_dir, state)
    ssh = connection(ready, key)
    source_archive = state.get("source_archive")
    remote_source = ""
    source_sha = ""
    if source_archive:
        remote_source = remote_workspace + "/uploads/source.tar.gz"
        upload_file(ssh, ready, Path(source_archive["path"]), remote_source)
        source_sha = str(source_archive["sha256"])
    input_archive = job_state.get("input_archive")
    remote_inputs = ""
    input_sha = ""
    if input_archive:
        remote_inputs = remote_workspace + "/uploads/inputs.tar.gz"
        upload_file(ssh, ready, Path(input_archive["path"]), remote_inputs)
        input_sha = str(input_archive["sha256"])
    job_state["inputs_uploaded_at"] = utc_now()
    write_state(output_dir, state)
    artifact_list = "\n".join(job["artifact_paths"]) + "\n"
    exit_code = run_remote(
        ssh,
        BOOTSTRAP,
        [
            remote_workspace,
            matrix["repository"]["source"],
            str(matrix["repository"].get("url", "")),
            matrix["repository"]["commit"],
            matrix["repository"]["setup_command"],
            resolve_command(job, remote_result),
            job["name"],
            artifact_list,
            remote_source,
            source_sha,
            remote_inputs,
            input_sha,
        ],
        output_dir / job_state["slug"] / "job.log",
    )
    job_state.update({"status": "completed" if exit_code == 0 else "failed", "exit_code": exit_code, "finished_at": utc_now()})
    write_state(output_dir, state)
    fetch_job(job_state, key, output_dir)
    write_state(output_dir, state)
    if exit_code:
        raise LauncherError(f"Job {job['name']} failed with exit code {exit_code}; its logs and artifacts were fetched.")


def command_check(args: argparse.Namespace) -> int:
    key = api_key(required=False)
    try:
        ssh = ssh_key_path(args.ssh_key)
    except LauncherError:
        ssh = Path(args.ssh_key or os.environ.get("RUNPOD_SSH_KEY_PATH", "~/.ssh/id_ed25519")).expanduser()
    report = {"runpod_api_key": "available" if key else "missing", "ssh_private_key": "present" if ssh.is_file() else "missing", "ssh_public_key": "present" if ssh.with_suffix(ssh.suffix + ".pub").is_file() else "not_checked", "git": shutil.which("git") or "missing", "ssh": shutil.which("ssh") or "missing", "scp": shutil.which("scp") or "missing", "runpodctl": shutil.which("runpodctl") or "optional_not_installed"}
    print(json.dumps(report, indent=2))
    if args.require_api and not key:
        raise LauncherError("RUNPOD_API_KEY is missing.")
    return 0


def command_validate(args: argparse.Namespace) -> int:
    matrix = validate(load_matrix(Path(args.matrix)))
    print(json.dumps({"status": "valid", "repository": matrix["repository"], "jobs": [job["name"] for job in matrix["jobs"]]}, indent=2))
    return 0


def command_estimate(args: argparse.Namespace) -> int:
    matrix = validate(load_matrix(Path(args.matrix)))
    pod = matrix["pod"]
    if str(pod.get("compute_type", "GPU")).upper() != "GPU":
        raise LauncherError("GPU price estimation currently requires pod.compute_type: GPU.")
    count, secure = int(pod.get("gpu_count", 1)), str(pod.get("cloud_type", "SECURE")).upper() == "SECURE"
    prices = []
    for gpu_id in pod["gpu_type_ids"]:
        query = "query { gpuTypes(input: { id: " + json.dumps(gpu_id) + " }) { id displayName lowestPrice(input: { gpuCount: " + str(count) + ", secureCloud: " + str(secure).lower() + " }) { stockStatus uninterruptablePrice availableGpuCounts } } }"
        details = graphql(query)["gpuTypes"]
        if details and details[0].get("lowestPrice"):
            item = details[0]
            rate = item["lowestPrice"].get("uninterruptablePrice")
            if rate is not None:
                prices.append({"gpu_id": item["id"], "stock": item["lowestPrice"].get("stockStatus"), "hourly_rate": float(rate)})
    if not prices:
        raise LauncherError("RunPod returned no price for the requested GPU preferences.")
    hours, jobs = float(args.hours), len(matrix["jobs"])
    hourly = [item["hourly_rate"] for item in prices]
    print(json.dumps({"jobs": jobs, "hours_per_job": hours, "candidate_rates": prices, "estimated_hourly_range": [min(hourly) * jobs, max(hourly) * jobs], "estimated_total_range": [min(hourly) * jobs * hours, max(hourly) * jobs * hours], "billing_note": "RunPod bills by the second; actual allocation price is recorded after provisioning."}, indent=2))
    return 0


def command_launch(args: argparse.Namespace) -> int:
    matrix_path = Path(args.matrix).resolve()
    matrix = validate(load_matrix(matrix_path))
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (Path.cwd() / default_output(matrix_path)).resolve()
    run_id = slug(args.run_id or output_dir.name)
    prefix = f"runpod-{run_id}-"
    if not args.dry_run and (args.max_hourly_rate is None or args.max_hourly_rate <= 0):
        raise LauncherError(
            "Paid launch requires a positive --max-hourly-rate cost cap for each Pod."
        )
    if state_path(output_dir).exists() and not args.resume:
        raise LauncherError(f"State already exists at {state_path(output_dir)}; use --resume or a new --output-dir.")
    state = read_state(state_path(output_dir)) if args.resume else {"run_id": run_id, "pod_name_prefix": prefix, "matrix": str(matrix_path), "created_at": utc_now(), "dry_run": bool(args.dry_run), "max_hourly_rate": args.max_hourly_rate, "jobs": []}
    state["max_hourly_rate"] = args.max_hourly_rate
    if not state["jobs"]:
        state["jobs"] = [{"name": job["name"], "slug": slug(job["name"]), "pod_name": prefix + slug(job["name"]), "matrix_job": job, "status": "planned"} for job in matrix["jobs"]]
    prepare_transfers(state, matrix, output_dir)
    if args.dry_run:
        print(json.dumps({"status": "dry_run_validated", "manifest": str(state_path(output_dir)), "jobs": [job["pod_name"] for job in state["jobs"]], "source_archive": state.get("source_archive"), "provisioned": False}, indent=2))
        return 0
    api_key()
    if str(matrix["pod"].get("compute_type", "GPU")).upper() == "GPU":
        command_estimate(argparse.Namespace(matrix=args.matrix, hours=args.estimate_hours))
    else:
        print(
            json.dumps(
                {
                    "cost_guard": "provisioned-rate-cap",
                    "max_hourly_rate_per_pod": args.max_hourly_rate,
                    "note": "RunPod's GPU estimator does not quote CPU Pods; each Pod is terminated before setup if its returned rate exceeds this cap.",
                },
                indent=2,
            )
        )
    key = ssh_key_path(args.ssh_key)
    failures: list[str] = []
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.max_parallel)) as pool:
            futures = [pool.submit(launch_one, state, job, matrix, output_dir, key) for job in state["jobs"] if not job.get("pod_id")]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as exc:  # Collect all worker failures before cleanup.
                    failures.append(str(exc))
    finally:
        if not args.keep_pods:
            cleanup(state, output_dir)
    if failures:
        raise LauncherError("; ".join(failures))
    print(json.dumps({"status": "complete", "manifest": str(state_path(output_dir)), "artifacts": str(output_dir), "pods_terminated": not args.keep_pods}, indent=2))
    return 0


def command_status(args: argparse.Namespace) -> int:
    manifest = Path(args.manifest).resolve()
    while True:
        state = read_state(manifest)
        rows = []
        for job in state["jobs"]:
            row = {"name": job["name"], "job_status": job["status"], "pod_id": job.get("pod_id"), "exit_code": job.get("exit_code")}
            if job.get("pod_id"):
                try:
                    pod = rest("GET", f"/pods/{job['pod_id']}")
                    row["pod_status"] = pod.get("desiredStatus")
                    row["cost_per_hour"] = pod.get("adjustedCostPerHr", pod.get("costPerHr"))
                except LauncherError as exc:
                    row["pod_status"] = "unavailable"
                    row["detail"] = str(exc)
            rows.append(row)
        print(json.dumps({"checked_at": utc_now(), "jobs": rows}, indent=2))
        if not args.follow:
            return 0
        time.sleep(args.poll_seconds)


def command_fetch(args: argparse.Namespace) -> int:
    manifest = Path(args.manifest).resolve()
    state = read_state(manifest)
    output_dir = manifest.parent
    key = ssh_key_path(args.ssh_key)
    for job in state["jobs"]:
        if job.get("pod_id") and job.get("remote_result"):
            fetch_job(job, key, output_dir)
    write_state(output_dir, state)
    print(json.dumps({"status": "fetched", "manifest": str(manifest)}, indent=2))
    return 0


def command_cleanup(args: argparse.Namespace) -> int:
    manifest = Path(args.manifest).resolve()
    state = read_state(manifest)
    count = cleanup(state, manifest.parent)
    print(json.dumps({"status": "cleanup_complete", "pods_considered": count, "manifest": str(manifest)}, indent=2))
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="operation", required=True)
    check = commands.add_parser("check", help="Check local credentials and optional RunPod CLI.")
    check.add_argument("--ssh-key")
    check.add_argument("--require-api", action="store_true")
    check.set_defaults(func=command_check)
    validate_cmd = commands.add_parser("validate", help="Validate and resolve a matrix without contacting RunPod.")
    validate_cmd.add_argument("--matrix", required=True)
    validate_cmd.set_defaults(func=command_validate)
    estimate = commands.add_parser("estimate", help="Fetch current RunPod GPU prices before provisioning.")
    estimate.add_argument("--matrix", required=True)
    estimate.add_argument("--hours", required=True, type=float)
    estimate.set_defaults(func=command_estimate)
    launch = commands.add_parser("launch", help="Launch independent jobs, fetch artifacts, then terminate Pods by default.")
    launch.add_argument("--matrix", required=True)
    launch.add_argument("--output-dir")
    launch.add_argument("--run-id")
    launch.add_argument("--ssh-key")
    launch.add_argument("--max-parallel", type=int, default=1)
    launch.add_argument("--estimate-hours", type=float, default=1.0, help="Expected hours per job for the mandatory pre-provision cost estimate.")
    launch.add_argument(
        "--max-hourly-rate",
        type=float,
        help="Required paid-run cap in USD/hour for each provisioned Pod.",
    )
    launch.add_argument("--dry-run", action="store_true")
    launch.add_argument("--keep-pods", action="store_true")
    launch.add_argument("--resume", action="store_true")
    launch.set_defaults(func=command_launch)
    status = commands.add_parser("status", help="Fetch or repeatedly stream Pod and job status.")
    status.add_argument("--manifest", required=True)
    status.add_argument("--follow", action="store_true")
    status.add_argument("--poll-seconds", type=float, default=20)
    status.set_defaults(func=command_status)
    fetch = commands.add_parser("fetch", help="Fetch declared artifacts again from live Pods.")
    fetch.add_argument("--manifest", required=True)
    fetch.add_argument("--ssh-key")
    fetch.set_defaults(func=command_fetch)
    cleanup_cmd = commands.add_parser("cleanup", help="Terminate every Pod belonging to a manifest run.")
    cleanup_cmd.add_argument("--manifest", required=True)
    cleanup_cmd.set_defaults(func=command_cleanup)
    return root


def main() -> int:
    try:
        args = parser().parse_args()
        return args.func(args)
    except LauncherError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
