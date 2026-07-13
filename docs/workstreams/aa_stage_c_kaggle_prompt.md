# Paste-ready AA Stage C task prompt

Read and follow `AGENTS.md`, `docs/PROJECT_STRATEGY.md`,
`docs/stage_c_partners.md`, `docs/kaggle_workflow.md`,
`docs/parallel_workspaces.md`, and the Kaggle skill under `.codex/` before
acting.

Operate only in `/Users/katharsis/Developer/dl/overcook-aa-stage-c` on branch
`codex/aa-stage-c-kaggle`. Verify both before changing or running anything. Do
not modify either sibling worktree.

Preserve
`outputs/stage_a_asymmetric_seed67/selected/training_step_000900096.pt` and its
inference artifact as the 13-soup fallback. Record their hashes before and after
the work. Also preserve the copied no-agent-index frozen partner artifact.

Run the local test suite, `build_policy` smoke, configuration validation, and a
minimal fresh-optimizer/fresh-RNG resume smoke. Commit any necessary AA-only
orchestration change before packaging. Do not change the model, PPO recipe,
partner weights, training budget, or evaluation suite in the committed configs.

Package the exact same immutable commit into two new, non-overwriting Kaggle
versions using `scripts/package_kaggle_run.py --accelerator cpu`:

- `configs/stage_c/asymmetric_exact_seed67_300k.yaml`;
- `configs/stage_c/asymmetric_weighted_pool_seed68_300k.yaml`.

Launch both CPU-only Kaggle sessions concurrently through the existing skill.
Each run adds exactly 300032 steps to the 900096-step checkpoint, uses a fresh
optimizer and RNG stream, saves every 50176 steps, and evaluates every saved
checkpoint against the disclosed `greedy_full_task` partner and every weighted
pool member in both physical positions and both inference modes.

Monitor internal progress and Kaggle status, continue the other session if one
fails, download all outputs, verify `run_summary.json`, input and artifact hash
manifests, final step counts, effective configs, checkpoint counts, and fallback
hashes, then place accepted artifacts under their configured local `outputs/`
directories. Select by deterministic minimum-position performance first and
deterministic mean official score second; retain stochastic results for
diagnosis. Do not overwrite the fallback or automatically launch another run.
