# Results and artifact registry

> **Status:** PPO research results are reported first, followed by the separate
> course-evaluation adaptation. Do not combine rows across protocols.

## A. PPO research benchmark and preserved artifacts

The original PPO checkpoints are retained **outside `final/`**. These results
come from the repository checkpoint suites used to choose and diagnose PPO
candidates. They report learning and cross-play behavior under the project's
internal protocol; they are not course-evaluation scores.

| Run | Layout / partner | Mode | Seeds x positions | Mean score | Mean soups | Exact source |
| --- | --- | --- | ---: | ---: | ---: | --- |
| AA step 900,096 | AA / `greedy_full_task` | historical suite | 3 x 2 | 65,273.00 | not reliable under final recipe criterion | `outputs/stage_a_asymmetric_seed67/selected/scenario1_evaluation.json` |
| CR seed-2 final | CR / self-play | deterministic | 3 x 2 | 13,817.00 | 1.00 per position | `outputs/baseline_coordination_ring_seed2_1m/checkpoint_evaluation/checkpoint_evaluation.json` |
| CC seed-3 step 1,902,592 | CC / disclosed sticky-random greedy | deterministic | 20 x 2 | 42,380.08 | 4.15 overall | `outputs/counter_circuit_exact_long_seed3_1m/checkpoint_evaluation/checkpoint_evaluation.json` |

The historical delivery ledger could count an interaction that did **not** earn
sparse reward for the active recipe. That makes the AA soup total invalid under
the course rule, and prevents these rows from being averaged or compared with
the results in section B.

The corresponding selected inference artifacts remain unchanged and are kept
for reproducibility of the PPO research process:

| Artifact | Stage / selection context | Historical protocol | Retained source and hash | Important boundary |
| --- | --- | --- | --- | --- |
| AA selected inference | Stage A self-play; checkpoint sweep selected step 900,096 | 3 seeds x both positions vs `greedy_full_task` | `outputs/stage_a_asymmetric_seed67/selected/inference_step_000900096.pt` `f6c4289d14623895249615b0bc48e11217bef62c53ce0977c050e076eb3187a2` | Legacy report records score although sparse return was zero; do not present it as a course result. |
| CR selected inference | Stage A Coordination Ring seed-2 self-play | 3 seeds x both positions, deterministic self-play checkpoint suite | `outputs/stage_d_specialists/coordination_ring/inference.pt` `806f0fc17587e832aeb939378dd9ffdce0ea413a15cae6a24eb58698ac4bb42b` | The final route instead uses the bundled guided teammate model. |
| CC selected inference | Stage C exact disclosed-partner continuation, seed 3 | 20 seeds x both positions, deterministic exact-partner suite | `outputs/counter_circuit_exact_long_seed3_1m/checkpoint_evaluation/selected/inference.pt` `176623829c3f71c17e73865170e87a8b88db6ae55471590d759cbca917a3bf41` | The suite's delivery accounting was not the final positive-reward criterion. |

The bootstrap artifact manifest at
[`workstreams/bootstrap_artifacts.json`](workstreams/bootstrap_artifacts.json)
records additional preserved AA inputs, including training checkpoints. The
final bundle copies only the artifact code/weights it needs to run after a
fresh clone.

## B. Course-evaluation adaptation and benchmark

The course evaluator measures only sparse-reward valid soups under its fixed
partners and seeds. Rather than altering the original PPO artifacts, the
project placed compact dataset-aggregation actors in `final/` for the disclosed
recipe/navigation failures in Scenarios 1, 3, and 4. Scenario 2 keeps its
guided PPO model because testing it outside its native layout was not better
than the routed specialists. The evaluator itself was not changed.

| Layout | Ego index | Partner | Mode | Seeds | Mean score | Mean soups | Active artifact / route | Source |
| --- | ---: | --- | --- | ---: | ---: | ---: | --- | --- |
| `asymmetric_advantages` | 0 | `greedy_full_task` | deterministic distilled actor | 4 | 140,420.00 | 14.00 | `final/policies/asymmetric_advantages_distilled.pt` | [`final/README_STUDENT_AGENT.md`](../final/README_STUDENT_AGENT.md) |
| `coordination_ring` | 0 | sticky `greedy_full_task` 0.10 | stochastic guided PPO | 4 | 54,358.50 | 5.25 | `final/policies/scenario2_guided_model.pt` | [`final/README_STUDENT_AGENT.md`](../final/README_STUDENT_AGENT.md) |
| `counter_circuit` | 0 | sticky/random `greedy_full_task` 0.15 / 0.05 | deterministic distilled actor | 4 | 76,296.75 | 7.50 | `final/policies/counter_circuit_distilled.pt` | [`final/README_STUDENT_AGENT.md`](../final/README_STUDENT_AGENT.md) |
| `scenario_4` | 0 | noisy `random_motion` | deterministic distilled actor | 4 | 95,526.50 | 9.50 | `final/policies/scenario4_distilled.pt` | [`final/README_STUDENT_AGENT.md`](../final/README_STUDENT_AGENT.md) |
| `scenario_4` | 1 | noisy `random_motion` | deterministic distilled actor | 4 | 93,100.25 | 9.25 | `final/policies/scenario4_distilled.pt` | [`final/README_STUDENT_AGENT.md`](../final/README_STUDENT_AGENT.md) |

Scenario 4 overall: `94,313.38` mean score and `9.375` mean soups over eight
seed/index attempts. All-scenario equal-weight mean: `91,347.16`.

## C. Excluded evidence

- self-play rollout return and shaped reward are optimization diagnostics, not
  benchmark score;
- planner/FSM sweeps and visual rollouts are one-off diagnostics unless their
  exact partner/seed protocol is stated;
- rejected, interrupted, or unretrieved Kaggle/RunPod jobs are excluded;
- the Scenario 2 guided model forced into other layouts is excluded from final
  selection because it lost to the routed specialists.
