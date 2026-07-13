# Parallel experiment workspaces

The three layout workstreams share one tested source commit and then diverge in
separate Git worktrees. They must not share virtual environments or generated
output directories.

| Worktree | Branch | Workstream |
| --- | --- | --- |
| `/Users/katharsis/Developer/dl/overcook` | `codex/cc-stage-a-local` | Counter Circuit Stage A, local |
| `/Users/katharsis/Developer/dl/overcook-cr-stage-b` | `codex/cr-stage-b-runpod` | Coordination Ring Stage B, RunPod CPU |
| `/Users/katharsis/Developer/dl/overcook-aa-stage-c` | `codex/aa-stage-c-kaggle` | Asymmetric Advantages Stage C, Kaggle CPU |

Each task must verify its absolute working directory and branch before changing
files. Source changes are committed independently and moved between worktrees
with normal Git operations, never by writing into another task's checkout.
Generated artifacts stay below that worktree's ignored `outputs/`, `kaggle/`,
or `outputs/runpod/` paths.

The shared base adds two continuation controls:

- `checkpoint.restore_rng_state` defaults to `true`. Set it to `false` when a
  fine-tuning run intentionally needs a fresh random stream.
- `training.reward_shaping_final` and
  `training.reward_shaping_anneal_steps` optionally linearly anneal shaping.
  Entropy and reward schedules count steps completed after resume, not the
  checkpoint's historical environment-step counter.

Both remote launchers preserve immutable source identity. Kaggle packages a
verified commit and hashes declared inputs and outputs. RunPod archive mode
uploads a `git archive`, hashes each declared job input, verifies transfer
hashes in the Pod, enforces a provisioned hourly-rate cap before setup, verifies
the downloaded artifact archive, and terminates Pods in a `finally` cleanup.

Handoff prompts are in
[`docs/workstreams/cr_stage_b_runpod_prompt.md`](workstreams/cr_stage_b_runpod_prompt.md)
and
[`docs/workstreams/aa_stage_c_kaggle_prompt.md`](workstreams/aa_stage_c_kaggle_prompt.md).
