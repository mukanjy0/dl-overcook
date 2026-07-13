---
name: runpod
description: Launch, monitor, retrieve, and safely terminate independent CPU or GPU experiment Pods on RunPod using the official REST/GraphQL APIs. Use when Codex needs to validate RunPod access, estimate GPU cost, run a one-Pod smoke test, or launch a YAML/JSON experiment matrix while preserving artifacts and avoiding leaked credentials.
---

# RunPod experiments

Use `.codex/skills/runpod/scripts/runpod_matrix.py` from the target repository with its local
`.venv/bin/python`. It uses RunPod's official REST API for Pod lifecycle
operations and official GraphQL API for price/availability estimates; it does
not require a third-party SDK.

## Required inputs

Read [references/matrix-format.md](references/matrix-format.md). Before any
paid launch, require:

- A RunPod API key through `RUNPOD_API_KEY` (required for CI) or the local
  `runpodctl doctor` profile (never a matrix file).
- An SSH private key, by `RUNPOD_SSH_KEY_PATH` or `--ssh-key`, whose public key
  has been added to the RunPod account.
- A repository source mode and commit. Prefer `source: archive` when the remote
  cannot authenticate to the repository; the launcher uploads an immutable
  `git archive`. With `source: git`, omit URL or commit only when the current
  checkout's `origin` and `HEAD` are the intended source.
- A matrix-selected image, GPU/CPU choice, setup command, per-job command, and
  artifact paths. Do not invent these values.

For private HTTPS repositories, set the narrowly scoped `RUNPOD_GIT_TOKEN` in
the environment; the launcher injects it only into the Pod and never writes it
to the matrix, state, logs, or terminal output.

## Safe workflow

1. Run `check` and `validate`. For GPU jobs, request an API-backed `estimate`.
   RunPod does not expose the same estimator for CPU Pods, so use an explicitly
   approved `--max-hourly-rate` cap instead.
2. Run exactly one job with `launch --dry-run`; this writes a planned manifest
   and makes no API requests or Pods.
3. With explicit authorization for spend, launch one short smoke job. The
   launcher verifies the returned provisioned rate before setup, uploads and
   verifies hashed inputs, retrieves and validates the artifact archive, and
   terminates every recorded Pod even on a job, transfer, or
   partial-provisioning failure.
4. Inspect the local manifest and `job.log`, calculate throughput from the
   training summary, then launch the parallel matrix only after the smoke is
   complete.

Use `--keep-pods` only for debugging. If any process is interrupted, recover
with `cleanup --manifest OUTPUT_DIR/runpod_state.json`; cleanup is idempotent.

## Commands

```bash
.venv/bin/python .codex/skills/runpod/scripts/runpod_matrix.py check
.venv/bin/python .codex/skills/runpod/scripts/runpod_matrix.py validate --matrix PATH
.venv/bin/python .codex/skills/runpod/scripts/runpod_matrix.py estimate --matrix PATH --hours HOURS
.venv/bin/python .codex/skills/runpod/scripts/runpod_matrix.py launch --matrix PATH --dry-run
.venv/bin/python .codex/skills/runpod/scripts/runpod_matrix.py launch --matrix PATH --estimate-hours HOURS --max-hourly-rate USD --max-parallel N
.venv/bin/python .codex/skills/runpod/scripts/runpod_matrix.py status --manifest OUTPUT_DIR/runpod_state.json --follow
.venv/bin/python .codex/skills/runpod/scripts/runpod_matrix.py fetch --manifest OUTPUT_DIR/runpod_state.json
.venv/bin/python .codex/skills/runpod/scripts/runpod_matrix.py cleanup --manifest OUTPUT_DIR/runpod_state.json
```

Do not launch more than one paid job until the requested one-Pod lifecycle has
succeeded. Read [references/matrix-format.md](references/matrix-format.md) for
the schema, artifact layout, and cleanup behavior.
