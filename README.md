# Overcooked-AI modular RL extension

This repository preserves the teacher-provided Overcooked-AI runner and its
`build_policy` contract while adding a small, configuration-driven Stage A PPO
self-play implementation. The inference submission remains a normal
`python_class` policy loaded by `src.policy_loader.build_policy`.

## Setup

Python 3.10 or 3.11 is supported. The repository pins Overcooked-AI 1.1.0,
keeps NumPy below 2, and uses PyTorch for both learning and inference.
SciPy is declared explicitly because Overcooked-AI 1.1.0 imports it without
including it in its published dependency metadata.

```bash
uv sync
.venv/bin/pytest
```

Do not install project packages globally. The generated `.venv` and all run
artifacts are ignored by git.

## Existing teacher workflow

Existing play, evaluate, demonstration, observation, wrapper, and policy YAML
files remain valid:

```bash
.venv/bin/python -m src.evaluate --config configs/evaluate.yaml
.venv/bin/python -m src.run_game --config configs/play.yaml
.venv/bin/python -m src.collect_demonstrations --config configs/collect_human_demonstrations.yaml
```

`policies/rl_policy.py` implements the same three-method student interface as
the supplied template. Its YAML configuration only needs an exported artifact,
a device selection (`auto`, `cpu`, or `cuda`), and deterministic inference.

## Stage A training and evaluation

```bash
.venv/bin/python scripts/train.py --config configs/stage_a/train_self_play.yaml
.venv/bin/python -m src.evaluate --config configs/stage_a/evaluate_checkpoint.yaml
```

Training writes the effective configuration, JSONL metrics, a resumable
training checkpoint, a compact device-neutral inference artifact, and a run
summary below the configured output root. To resume, set
`checkpoint.resume_from` to a training checkpoint. Evaluation loads the ego
through `build_policy`, tests both positions, and reports sparse return,
delivery events, timeout/invalid-action replacements, and the canonical
official score.

## Kaggle GPU execution

Local smoke tests should pass before packaging. The packager follows the
repository Kaggle skill's versioned input/output convention and runs the same
training script and YAML remotely:

```bash
.venv/bin/python scripts/package_kaggle_run.py \
  --version v1 \
  --kernel-id USER/stage-a-self-play \
  --title "Stage A self-play"
kaggle kernels push -p kaggle/v1/input
```

Monitor the kernel with the Kaggle CLI, then download `run_summary.json`,
checkpoints, metrics, logs, and the effective configuration into
`kaggle/v1/outputs` and copy important artifacts into the configured local
output structure. Kaggle-specific paths and APIs do not enter core modules.

See [docs/architecture.md](docs/architecture.md) for module responsibilities,
interfaces, checkpoint profiles, and compatibility guarantees.
