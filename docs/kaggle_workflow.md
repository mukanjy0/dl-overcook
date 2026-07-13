# Kaggle execution workflow

This repository uses the local Kaggle CLI and versioned `kaggle/vN/`
directories. Core training and evaluation remain portable; Kaggle scripts only
install dependencies, invoke the existing entry points, and preserve outputs.

## Before packaging

1. Read `AGENTS.md`, `.codex/skills/kaggle/SKILL.md`, and the runner reference
   linked by that skill.
2. Use `.venv/bin/python`, `.venv/bin/pytest`, and `.venv/bin/kaggle`.
3. Run the focused local tests and a short smoke test before consuming remote
   compute.
4. Select a new unused `kaggle/vN` directory for every remote kernel.
5. For controlled experiments, package a named Git commit rather than the
   working tree. This prevents unrelated or concurrent edits from entering only
   part of a sweep.

Never print or commit Kaggle credentials. The CLI should use the developer's
existing local authentication.

## Standard training run

Package the committed project and one YAML configuration:

```bash
.venv/bin/python scripts/package_kaggle_run.py \
  --version v24 \
  --kernel-id OWNER/KERNEL-SLUG \
  --title KERNEL-TITLE \
  --config configs/stage_a/train_self_play.yaml
```

Inspect `kaggle/v24/input/kernel-metadata.json`, then launch from the input
directory:

```bash
cd kaggle/v24/input
../../../.venv/bin/kaggle kernels push -p . --accelerator NvidiaTeslaT4
```

Kaggle may reject extra GPU jobs rather than queue them. Check every returned
message and keep only the permitted number of sessions active.

Monitor and retrieve with the exact kernel slug:

```bash
.venv/bin/kaggle kernels status OWNER/KERNEL-SLUG
.venv/bin/kaggle kernels output OWNER/KERNEL-SLUG \
  -p kaggle/v24/outputs
```

Do not treat Kaggle `COMPLETE` as sufficient. Also inspect the downloaded
`run_summary.json` and verify its internal `status`, checkpoint counts, metrics,
effective configuration, and expected final step count. Copy accepted artifacts
into the configured `outputs/` run directory; keep the complete remote log and
versioned package under `kaggle/vN/`.

## Immutable CPU-versus-GPU benchmark

The paired benchmark packager archives the requested commit's runtime paths
directly with `git archive`. Runtime-irrelevant papers and generated files are
excluded to stay under Kaggle's script-size limit. It generates an identical
`main.py` for CPU and T4; only kernel identity and accelerator metadata differ.

```bash
.venv/bin/python scripts/package_kaggle_throughput_benchmark.py \
  --commit ae29971 \
  --config configs/stage_a/ablation_baseline_200k.yaml \
  --owner OWNER \
  --cpu-version v24 \
  --gpu-version v25
```

Launch CPU without an accelerator flag and T4 with the explicit accelerator:

```bash
cd kaggle/v24/input
../../../.venv/bin/kaggle kernels push -p .

cd ../../v25/input
../../../.venv/bin/kaggle kernels push -p . --accelerator NvidiaTeslaT4
```

Both kernels perform one unmeasured rollout/update warm-up and then run the
unchanged committed 200k configuration. `benchmark_result.json` records setup,
training wall time, environment throughput, rollout/update timing, CPU and GPU
utilization, RAM/VRAM, dependency versions, and artifact-integrity hashes.

Before interpreting results, confirm:

- identical `commit` and `config_sha256`;
- identical configuration, seed, layout, horizon, model, and PPO values;
- the expected resolved device difference only;
- successful completion and `artifact_integrity.ok: true`;
- the same completed environment-step and update counts.

Compare timed training separately from dependency/setup time. GPU is materially
useful only if it improves the repeated training phase enough to justify scarce
GPU-session concurrency; startup variation alone is not evidence.

## Handoff checklist

Every remote-execution handoff should state:

- commit, config, kernel slugs, and Kaggle versions;
- remote and internal status for every run;
- local output locations;
- steps, checkpoints, metrics, effective configs, and logs retrieved;
- failed or excluded runs and why they were excluded;
- the selected artifact and the exact criterion used.
