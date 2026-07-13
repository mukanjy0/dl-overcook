# RunPod matrix format

Use YAML (requires the repository's `PyYAML`) or JSON. Keep secrets only in
environment variables or the user-level RunPod CLI profile. `RUNPOD_API_KEY` is
the required non-interactive/CI variable; locally, a key saved by
`runpodctl doctor` is used without printing or copying it. Omit `repository.url` and `repository.commit` to resolve
the current checkout's `origin` and `HEAD` at launch time.

The launcher automatically prefers the SSH key created by `runpodctl doctor`;
override it with `RUNPOD_SSH_KEY_PATH` or `--ssh-key` when needed.

```yaml
repository:
  url: https://HOST/OWNER/REPOSITORY.git # omit to use origin
  commit: GIT_COMMIT_SHA                 # omit to use HEAD
  setup_command: uv sync --frozen
pod:
  image: CONTAINER_IMAGE
  compute_type: GPU                      # GPU or CPU
  gpu_type_ids: [GPU_TYPE_ID]            # GPU only; ordered preferences
  gpu_count: 1
  cloud_type: SECURE                     # SECURE or COMMUNITY
  interruptible: false
  container_disk_gb: 30
  volume_gb: 20
  volume_mount_path: /workspace
  readiness_timeout_minutes: 15
jobs:
  - name: short-smoke
    config: PATH/TO/CONFIG.yaml
    command: .venv/bin/python PATH/TO/TRAIN.py --config {config}
    artifact_paths:
      - PATH/TO/OUTPUTS
```

`command` is expanded only for `{config}` and `{artifact_dir}`. `config` is
recorded in job metadata; the launcher does not modify it. Each job gets an
independent Pod and a checkout pinned to the recorded commit. The setup command
runs from the clone root before the job command.

The launcher preserves the selected image or template's default entrypoint so
RunPod can initialize its SSH service. Do not replace it with an idle command.

The local output directory defaults to `outputs/runpod/<matrix>-<timestamp>/`.
It contains `runpod_state.json`, one `job.log` and `metadata.json` per job, and
each job's `artifacts.tar.gz`. The archive includes the declared artifact paths
plus the job log and metadata. Fetching is safe to repeat.

Cost estimation queries RunPod's current `lowestPrice` and stock status for
every selected GPU type before provisioning. It reports an hourly range and an
`--hours` total. `launch` also performs this request automatically using
`--estimate-hours` (default: one hour per job) before it creates any Pod. This
is an estimate: allocation can select another allowed GPU
when `gpu_type_priority: availability` is used, and actual billing is measured
from the provisioned Pod's returned hourly rate and timestamps in the manifest.

`cleanup` deletes all Pod IDs recorded in the manifest and also searches for
the current run's deterministic name prefix, covering an API response that
arrived after a local write failed. It never deletes Pods outside that prefix.
