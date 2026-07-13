# Counter Circuit Stage B reset-robustness matrix

The Stage C exact-partner run reached the two-soup threshold only under
stochastic action selection. This workstream tests whether resets from its
successful trajectories improve deterministic execution without changing the
partner, PPO hyperparameters, checkpoint source, seed, or physical-position
sampling.

`collect_counter_circuit_stage_c_exact_states.yaml` records every fifth state
from 32 stochastic trajectories for each physical placement of the final Stage
C policy with the disclosed sticky/random partner. The buffer has both
assignments, preserves state timestep, and is validated through the normal
state-buffer boundary before use.

The initial two-session CPU Kaggle matrix is deliberately high-signal:

| Config | Reset distribution | Purpose |
| --- | --- | --- |
| `counter_circuit_exact_standard_seed2_200k.yaml` | standard only | Continuing-training control |
| `counter_circuit_exact_mixed050_seed2_200k.yaml` | 50% buffer / 50% standard | Test a material state-coverage intervention |

Both make 200,704 additional steps from the same 1,200,128-step Stage C
training checkpoint with fresh optimizer/RNG streams, balanced ego positions,
and the exact disclosed partner. Each evaluates all saved checkpoints over
20 seeds, both physical positions, and deterministic/stochastic modes.

Use CPU Kaggle sessions, not T4: the repository benchmark found PPO throughput
to be environment-bound. CPU allows the two independent jobs to run in parallel
while preserving GPU quota for workloads that benefit from it.

The gate is deterministic minimum-position performance. A result is promising
only when both positions improve materially toward two soups while the
stochastic two-soup behavior does not regress. If the 50% mix is harmful, the
next bounded follow-up is a 25% mixture; do not launch it automatically.
