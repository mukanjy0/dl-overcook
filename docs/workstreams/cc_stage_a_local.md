# Counter Circuit Stage A local workstream

Operate only on `/Users/katharsis/Developer/dl/overcook` and branch
`codex/cc-stage-a-local`.

First re-evaluate these promising source checkpoints through the existing
checkpoint evaluator with seeds 0–19, both physical ego positions, and both
inference modes:

- seed 0, step 300032;
- seed 1, step 400384;
- seed 2, step 900096.

The committed continuation configurations use the strongest previously
observed candidate, seed 2 at step 900096:

- `configs/stage_a/counter_circuit_control_continuation_50k.yaml` preserves the
  optimizer and RNG state and changes no PPO or reward coefficients;
- `configs/stage_a/counter_circuit_consolidation_continuation_50k.yaml` starts a
  fresh optimizer/RNG stream and anneals reward shaping 1.0 → 0.1 and entropy
  0.01 → 0.001 over exactly 50176 continuation steps.

Both save at the midpoint and endpoint and evaluate 20 fixed seeds. Do not run
longer training unless stochastic mean soups reaches at least 1.0 and the
zero-soup rate is at most 50%. A longer extension additionally requires
non-zero deterministic progress in both positions or an unambiguous stochastic
improvement over the source checkpoint.
