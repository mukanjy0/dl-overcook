# Counter Circuit long exact-partner continuations

The validated 1,400,832-step standard-reset control repaired deterministic
execution (2.20 soups over 20 seeds, with 2.10 and 2.30 by physical position).
This workstream spends the next two-million-step compute budget as two
independent 1,000,448-step branches instead of one monolithic run. This keeps
the exact same PPO, reset distribution, frozen disclosed partner, and balanced
position sampling, while making optimizer and environment randomness an
explicit comparison variable.

- `counter_circuit_exact_long_seed2_1m.yaml` uses experiment seed 2;
- `counter_circuit_exact_long_seed3_1m.yaml` uses experiment seed 3.

Both resume the validated selected training checkpoint with fresh optimizer/RNG
streams, save at 100,352-step intervals, and evaluate all saved checkpoints on
20 fixed seeds, both positions, and both inference modes. The current
1,400,832-step selected artifact remains the baseline and is never overwritten.

Run both as CPU Kaggle sessions: PPO is environment-bound in this repository,
and the independent branches benefit more from session parallelism than a T4.
Select globally by deterministic minimum-position official score, then
deterministic mean score. A later checkpoint cannot displace the baseline
unless it actually improves that criterion.
