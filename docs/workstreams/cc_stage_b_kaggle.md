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
next bounded follow-up may be a 25% mixture, but only if the standard control
has not already met the deployment gate; do not launch it automatically.

## Result

Both CPU Kaggle kernels completed from commit `a1d051c` at exactly 1,400,832
environment steps, with five evaluated checkpoints each:

- `badexample/overcook-cc-stage-b-standard-control` (`kaggle/v33`);
- `badexample/overcook-cc-stage-b-mixed-reset-50` (`kaggle/v34`).

| Run / checkpoint | Deterministic soups (pos 0 / pos 1) | Deterministic zero rate | Deterministic mean score | Stochastic soups (pos 0 / pos 1) |
| --- | ---: | ---: | ---: | ---: |
| Standard selected, step 1,400,832 | 2.200 (2.100 / 2.300) | 10.0% | 23,457.7 | 2.675 (2.700 / 2.650) |
| Mixed 50% selected, step 1,350,656 | 0.325 (0.450 / 0.200) | 75.0% | 3,989.3 | 2.200 (2.400 / 2.000) |
| Mixed 50% final, step 1,400,832 | 0.275 (0.150 / 0.400) | 82.5% | 3,247.9 | 3.000 (2.650 / 3.350) |

The standard continuation is the successful result: it clears the two-soup
target in deterministic inference in both physical positions. Its deterministic
minimum-position score is 23,045.8 and its stochastic zero-soup rate is 2.5%.
The 50% reset mixture is not a successful Stage B intervention for deployment:
it improves stochastic exploration but destroys deterministic execution. It
actually performed 352 standard and 308 augmented resets, so the comparison is
not explained by a missing buffer path or inactive reset source.

The selected standard artifact was replayed locally through the existing
`build_policy` evaluator with zero timeouts and zero invalid-action
replacements in both modes. The remote CPU suite recorded three ego timeouts
across 40 rollouts; treat those as Kaggle timing noise, not a policy-interface
failure. Remote manifests and the standard artifact set validate fully. The
mixed run has a SHA-256 mismatch only for its duplicate `training_final.pt`;
its selected step-1,350,656 checkpoint and selected inference artifact validate
and are sufficient for the comparison, but do not use that raw final file.

Do not launch the 25% mixture automatically. The control demonstrates that
continued exact-partner training, rather than the current state-augmentation
mixture, repaired deterministic collapse. Preserve its selected inference
artifact as the Counter Circuit specialist candidate and only revisit Stage B
with a changed state-selection or reset curriculum.
