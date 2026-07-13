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

## Result

Both CPU Kaggle kernels completed from commit `47a65f9` at exactly 2,401,280
environment steps with ten evaluated checkpoints each. Their selected training
and inference artifacts match the remote SHA-256 manifests.

| Candidate | Deterministic soups (pos 0 / pos 1) | Zero rate | Mean score | Minimum-position score | Stochastic soups |
| --- | ---: | ---: | ---: | ---: | ---: |
| Previous CC baseline, step 1,400,832 | 2.200 (2.100 / 2.300) | 10.0% | 23,457.7 | 23,045.8 | 2.675 |
| Seed 2 selected, step 2,002,944 | 3.175 (3.300 / 3.050) | 12.5% | 33,104.4 | 31,855.3 | 4.100 |
| **Seed 3 selected, step 1,902,592** | **4.150 (3.850 / 4.450)** | **2.5%** | **42,380.1** | **39,349.8** | 3.700 |

The seed-3 selected inference artifact was replayed locally through the
existing `build_policy` evaluator. It reproduced 4.15 deterministic soups,
2.5% zero-soup rate, mean score 42,387.6, and zero timeouts in both inference
modes. It is the Counter Circuit specialist candidate:
`outputs/counter_circuit_exact_long_seed3_1m/checkpoint_evaluation/selected/inference.pt`.

Later training was not monotonically better: seed 3's final 2,401,280-step
checkpoint fell to 1.55 deterministic soups and a 30% zero-soup rate. Preserve
the selected step-1,902,592 checkpoint; do not substitute either run's final
checkpoint merely because it is newer.
