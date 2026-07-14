# Reproducibility guide

> **Status:** stable commands are marked below. Large historical artifacts live
> outside Git; the final submission bundle is clone-runnable.

## Environment

```bash
uv sync
.venv/bin/pytest -q
```

Use the repository-local `.venv`; do not install dependencies globally. The
declared runtime pins Overcooked-AI 1.1.0, NumPy `<2`, PyTorch `<3`, PyYAML,
SciPy, Pillow, and imageio. Python 3.10–3.12 is supported.

## Stable: final teacher benchmark

```bash
cd final
../.venv/bin/python -m src.evaluate_competition \
  --config configs/competition.yaml --all-scenarios
```

Expected generated outputs:

- `final/results/competition_eval/per_attempt.csv`;
- `final/results/competition_eval/per_scenario.csv`;
- `final/results/competition_eval/competition_config_used.yaml`.

The active model hashes are recorded in
[`../final/README_STUDENT_AGENT.md`](../final/README_STUDENT_AGENT.md). The
bundle resolves all artifacts relative to `final/`; it does not require ignored
research checkpoints, remote storage, or external model downloads.

## Stable: vertical PPO smoke

```bash
.venv/bin/pytest -q tests/test_build_policy_smoke.py tests/test_checkpointing.py
```

This validates short train/resume/export/load behavior and preserves the
teacher-compatible `build_policy` contract. The full test suite is the stronger
local gate:

```bash
.venv/bin/pytest -q
```

## Experimental: short training and checkpoint evaluation

Run a 64-step local smoke before a larger experiment:

```bash
.venv/bin/python scripts/train.py \
  --config configs/examples/ppo_smoke.yaml
```

The Stage A config is a reproducible baseline; it is not the final submission
recipe:

```bash
.venv/bin/python scripts/train.py \
  --config configs/stage_a/ablation_baseline_200k.yaml \
  --evaluate-checkpoints
```

Each meaningful run writes, below its configured `outputs/` root:

- `effective_config.yaml`;
- `metrics/training.jsonl`;
- resumable training checkpoints and exported inference artifacts;
- `progress.json` and `run_summary.json`;
- `checkpoint_evaluation/checkpoint_evaluation.json` when requested.

The output root is ignored intentionally. Keep accepted artifacts locally or
in a durable experiment store and record hashes before using them as inputs to
a later continuation.

## Seeds, positions, and devices

- The experiment seed derives model, environment, partner, and position RNG
  streams through `src.seed_utils`.
- Stage C balances physical ego positions; final Scenario 4 evaluates both.
- Inference artifacts load on CPU first and can resolve `auto`, `cpu`, or
  `cuda`; final submission routes are CPU-compatible.
- Checkpoint continuation can intentionally use a fresh optimizer/RNG stream
  with `load_optimizer_state: false` and `restore_rng_state: false`.

## Kaggle workflow

The reusable Kaggle skill is operational tooling, not an inference dependency.
It packages a committed source/configuration, validates manifests, launches
through the Kaggle CLI, monitors status, and retrieves outputs. The canonical
workflow is in [`kaggle_workflow.md`](kaggle_workflow.md):

```bash
.venv/bin/python scripts/package_kaggle_run.py \
  --version vN --kernel-id OWNER/SLUG --title TITLE \
  --config configs/stage_a/train_self_play.yaml \
  --commit HEAD --accelerator cpu

.venv/bin/kaggle kernels push -p kaggle/vN/input
```

The project includes a controlled CPU-versus-T4 throughput benchmark. Its
conclusion for this PPO workload was operational: rollout/environment work was
CPU-bound enough that CPU Kaggle sessions were preferred for parallel sweeps;
GPU was not assumed beneficial merely because PPO uses PyTorch. See
[`kaggle_workflow.md`](kaggle_workflow.md) for the immutable-config and
artifact-integrity checks.

## RunPod workflow

RunPod was used for independent matrix jobs when isolated CPU/GPU capacity was
more appropriate than Kaggle concurrency. The repository skill packages a
`git archive` (or pinned commit), hashes declared job inputs, enforces a cost
cap before setup, retrieves and verifies archives, and terminates pods in
cleanup. This separates remote orchestration from learning code.

```bash
.venv/bin/python .codex/skills/runpod/scripts/runpod_matrix.py check --require-api
.venv/bin/python .codex/skills/runpod/scripts/runpod_matrix.py launch \
  --matrix PATH/TO/matrix.yaml --dry-run
```

Never commit credentials, tokens, SSH data, or cloud-specific absolute paths.
The repository's `.gitignore` excludes `.runpod/`, generated Kaggle packages,
and remote outputs.

## Artifact manifest

For pre-distillation research artifacts retained locally, consult
[`RESULTS.md`](RESULTS.md) and
[`workstreams/bootstrap_artifacts.json`](workstreams/bootstrap_artifacts.json).
For final clone-runnable model hashes, consult the final bundle README. These
two registries intentionally refer to different artifact classes.
