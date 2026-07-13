# Paste-ready CR Stage B task prompt

Read and follow `AGENTS.md`, `docs/PROJECT_STRATEGY.md`,
`docs/stage_b_state_augmentation.md`, `docs/parallel_workspaces.md`, and the
RunPod skill under `.codex/` before acting.

Operate only in `/Users/katharsis/Developer/dl/overcook-cr-stage-b` on branch
`codex/cr-stage-b-runpod`. Verify both before changing or running anything. Do
not modify either sibling worktree.

Preserve the copied Stage A Coordination Ring artifacts. Run the local test
suite and `build_policy` smoke first. Build and inspect the larger cross-play
buffer with `configs/stage_b/collect_coordination_ring_crossplay.yaml`; require
all 30 trajectories (six ordered pairings times five seeds), valid restored
states, and retained successful trajectories before proceeding.

Validate and dry-run
`configs/stage_b/runpod_coordination_ring_smoke.yaml`. Use immutable archive
upload, the declared CPU5C/4-vCPU secure non-interruptible Pod, hashed job
inputs, and a user-approved `--max-hourly-rate`. Before any paid launch, report
the exact commit, input hashes, cost cap, and worst-case smoke cost and request
explicit cost approval. Then run exactly one 32768-step Pod lifecycle: provision,
verify rate, upload, train/evaluate, retrieve and verify artifacts, and terminate
the Pod. Confirm through the API that cleanup succeeded even if any earlier step
fails.

Only after that lifecycle succeeds and the user explicitly approves the full
cost, launch `configs/stage_b/runpod_coordination_ring_matrix.yaml` with at most
four parallel CPU workers. It contains standard, mixed 0.25, mixed 0.50, and
augmented resets across seeds 0–2 for 200704 steps each. Continue other jobs if
one fails. Retrieve every checkpoint, metric, log, effective config, evaluation,
manifest, and integrity hash locally, and terminate all Pods.

Do not alter algorithms/hyperparameters, start Stage C, or scale beyond this
matrix automatically. Report per variant/seed self-play deterministic and
stochastic outcomes, state-source reset counts, failures, throughput, actual
cost, artifact locations/hashes, and whether Stage B improves robustness
without losing self-play competence.
