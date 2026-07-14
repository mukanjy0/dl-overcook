# Overcooked-AI modular RL extension

This repository preserves the teacher-provided Overcooked-AI runner and its
`build_policy` contract while adding a small, configuration-driven Stage A PPO
self-play implementation. The inference submission remains a normal
`python_class` policy loaded by `src.policy_loader.build_policy`.

## Setup

Python 3.10, 3.11, or 3.12 is supported. The repository pins Overcooked-AI 1.1.0,
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

## Competition submission evaluation

The teacher-facing submission is self-contained in `final/`. Run its evaluator
from that directory so `configs/competition.yaml` resolves the bundled router,
model artifacts, layouts, and teacher runner correctly:

```bash
cd final
../.venv/bin/python -m src.evaluate_competition \
  --config configs/competition.yaml --scenario 1
```

Run every enabled scenario with:

```bash
cd final
../.venv/bin/python -m src.evaluate_competition \
  --config configs/competition.yaml --all-scenarios
```

For a rendered single-scenario rollout, add `--render`:

```bash
cd final
../.venv/bin/python -m src.evaluate_competition \
  --config configs/competition.yaml --scenario 1 --render
```

The `submissions` block in `final/configs/competition.yaml` must keep pointing
to `policies/template.py` and `StudentAgent`. That router selects the bundled
specialist for each known layout; do not replace it with the teacher template.
If the teacher supplies its own Python environment, use its `python` command
in place of `../.venv/bin/python` while still running from `final/`.

## Stage A training and evaluation

```bash
.venv/bin/python scripts/train.py --config configs/stage_a/train_self_play.yaml
.venv/bin/python -m src.evaluate --config configs/stage_a/evaluate_checkpoint.yaml
```

Training writes the effective configuration, JSONL metrics, a resumable
training checkpoint, a compact device-neutral inference artifact, and a run
summary below the configured output root. To resume, set
`checkpoint.resume_from` to a training checkpoint. RNG restoration defaults to
`checkpoint.restore_rng_state: true`; fine-tuning can set it to `false` without
changing checkpoint compatibility. Entropy and reward shaping can anneal over
steps completed after resume. Evaluation loads the ego
through `build_policy`, tests both positions, and reports sparse return,
delivery events, timeout/invalid-action replacements, and the canonical
official score.

For checkpoint-aware runs, add `--evaluate-checkpoints`. This exports and
evaluates every saved checkpoint in the configured deterministic and stochastic
modes, from both ego positions, then copies the selected training and inference
artifacts below `checkpoint_evaluation/selected/`:

```bash
.venv/bin/python scripts/train.py \
  --config configs/stage_a/ablation_baseline_200k.yaml \
  --evaluate-checkpoints
```

Selection first maximizes deterministic minimum-position official score, then
deterministic mean official score. Stochastic metrics are diagnostic
and do not override deployment selection.

## Kaggle CPU or GPU execution

Local smoke tests should pass before packaging. The packager follows the
repository Kaggle skill's versioned input/output convention and runs the same
training script and YAML remotely:

```bash
.venv/bin/python scripts/package_kaggle_run.py \
  --version v1 \
  --kernel-id USER/stage-a-self-play \
  --title "Stage A self-play" \
  --commit HEAD \
  --accelerator cpu
kaggle kernels push -p kaggle/v1/input
```

Monitor the kernel with the Kaggle CLI, then download `run_summary.json`,
checkpoints, metrics, logs, and the effective configuration into
`kaggle/v1/outputs` and copy important artifacts into the configured local
output structure. Verify `remote_input_manifest.json` and
`artifact_manifest.json` hashes. Kaggle-specific paths and APIs do not enter
core modules.

See [docs/architecture.md](docs/architecture.md) for module responsibilities,
interfaces, checkpoint profiles, and compatibility guarantees.

## Stage B state augmentation

Stage B is opt-in and reuses the normal environment and PPO collectors. Collect
a short versioned buffer, validate it, then run the committed mixed-reset smoke:

```bash
.venv/bin/python scripts/collect_state_buffer.py \
  --config configs/stage_b/collect_state_buffer_example.yaml
.venv/bin/python scripts/inspect_state_buffer.py \
  --buffer outputs/stage_b/buffers/example.json.gz \
  --environment-config configs/stage_b/collect_state_buffer_example.yaml
.venv/bin/python scripts/train.py \
  --config configs/stage_b/train_mixed_resets_example.yaml
```

Training without a `state_augmentation` section retains standard Stage A
resets. See
[docs/stage_b_state_augmentation.md](docs/stage_b_state_augmentation.md) for
the buffer schema, policy-pairing format, checkpoint cross-play example, reset
modes, compatibility checks, and extension points.

## Stage C frozen-partner training

Stage C is opt-in: existing `partner.sampler: self_play` configurations retain
Stage A behavior. Set the sampler to `weighted_pool` for a configurable frozen
partner distribution or `exact` for one-partner best-response fine-tuning. Both
modes train only the ego and balance it across physical player positions.

The initial pool contains teacher scripted partners, sticky/random wrapper
variants, `random_motion`, and one frozen self-play artifact:

```bash
.venv/bin/python scripts/train.py \
  --config configs/stage_c/train_partner_pool.yaml
.venv/bin/python -m src.evaluate \
  --config configs/stage_c/evaluate_partner_pool.yaml
```

Evaluation reports each partner and ego position separately in deterministic
and stochastic ego modes. See
[docs/stage_c_partners.md](docs/stage_c_partners.md) for the partner interface,
pool schema, exact fine-tuning command, checkpoint requirements, and extension
instructions.

## RunPod experiment jobs

The reusable [RunPod skill](.codex/skills/runpod/SKILL.md) launches isolated
experiment Pods through RunPod's official APIs. It always pins the cloned Git
commit (or uploads an immutable archive), hashes declared inputs, enforces an
approved provisioned hourly-rate cap, validates fetched artifacts, and
terminates recorded Pods on completion or failure.

Configure RunPod outside the repository, then verify the local setup:

```bash
brew install runpod/runpodctl/runpodctl  # if runpodctl is not installed
runpodctl doctor
.venv/bin/python .codex/skills/runpod/scripts/runpod_matrix.py check --require-api
```

Create a YAML/JSON matrix with the selected image, compute choice, setup
command, job command, configuration, and artifact paths; see
[the matrix format](.codex/skills/runpod/references/matrix-format.md). Run a
dry run and a one-worker smoke test before a matrix launch:

```bash
.venv/bin/python .codex/skills/runpod/scripts/runpod_matrix.py launch \
  --matrix PATH/TO/matrix.yaml --dry-run
.venv/bin/python .codex/skills/runpod/scripts/runpod_matrix.py launch \
  --matrix PATH/TO/matrix.yaml --estimate-hours HOURS \
  --max-hourly-rate USD --max-parallel 1
```

For a validated matrix, the parallel command is the same with the desired
worker count:

```bash
.venv/bin/python .codex/skills/runpod/scripts/runpod_matrix.py launch \
  --matrix PATH/TO/matrix.yaml --estimate-hours HOURS \
  --max-hourly-rate USD --max-parallel N
```

Keep `RUNPOD_API_KEY` (for CI) and, when needed, `RUNPOD_GIT_TOKEN` only in the
environment. Locally the launcher can use the API key and SSH key created by
`runpodctl doctor`; it never prints, writes, or commits credentials. Recover
from interruption with `cleanup --manifest OUTPUT_DIR/runpod_state.json`.
