# Final Stage D Submission

## Decision

The final submitted policy is the Stage D router in
`configs/stage_d/specialists.yaml`. It preserves all prior artifact-backed
specialists and adds the validated Scenario 4 scripted specialist plus an
explicit recovery-planner fallback for unknown layouts. No checkpoint was
overwritten and no new model was trained.

| Route | Physical position | Selected policy |
| --- | ---: | --- |
| `asymmetric_advantages` | 0 | `aa_rl_position0_900096` |
| `asymmetric_advantages` | 1 | `aa_greedy_position1` |
| `coordination_ring` | 0, 1 | `cr_p010_s11_step1050624` |
| `counter_circuit` | 0, 1 | `cc_exact_long_seed3_step1902592` |
| `scenario_4` | 0, 1 | `scenario4_exact_fixed_pot_b` |
| unknown layout | 0, 1 | `generic_task_planner_recovery` |

The Scenario 1 isolated probe was rejected because it scored zero at position
1 against `greedy_full_task`. The Scenario 3 2M probe was rejected because its
deterministic minimum-position result was below the retained CC specialist.
The generic recovery profile was selected from the existing fallback tournament:
49,246 mean official score, ahead of the 43,290 balanced/baseline variants.

## Preserved artifacts

| Specialist | Artifact | SHA-256 |
| --- | --- | --- |
| AA position 0 | `outputs/stage_a_asymmetric_seed67/selected/inference_step_000900096.pt` | `f6c4289d14623895249615b0bc48e11217bef62c53ce0977c050e076eb3187a2` |
| CR both positions | `outputs/stage_d_specialists/coordination_ring/inference.pt` | `806f0fc17587e832aeb939378dd9ffdce0ea413a15cae6a24eb58698ac4bb42b` |
| CC both positions | `outputs/counter_circuit_exact_long_seed3_1m/checkpoint_evaluation/selected/inference.pt` | `176623829c3f71c17e73865170e87a8b88db6ae55471590d759cbca917a3bf41` |

Scenario 4 and the generic fallback are deterministic built-ins, so they have
no checkpoint artifact. The machine-readable registry, including resolved paths
and rechecked hashes, is at `outputs/stage_d_finalization/candidate_registry.json`.

## Fresh router-level benchmark

Every ego policy below was loaded through `stage_d_router` and `build_policy`.
Scores are official scores. S1 uses seeds 67–69, S2 uses 0–4, S3 uses 0–19,
and S4 is a fresh random-motion window of seeds 30–59. The score columns are
means by physical ego position; `min-pos` is the lower of those two means.

| Scenario | Rollouts | P0 score | P1 score | min-pos | Mean score | Mean soups | Zero soup | Timeouts / invalid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S1 AA vs greedy | 6 | 130,546.0 | 13,861.0 | 13,861.0 | 72,203.5 | 7.00 | 0.0% | 0 / 0 |
| S2 CR vs sticky greedy 0.10 | 10 | 15,903.2 | 17,887.4 | 15,903.2 | 16,895.3 | 1.30 | 0.0% | 0 / 0 |
| S3 CC vs sticky/random greedy 0.10 | 40 | 39,354.8 | 45,420.4 | 39,354.8 | 42,387.6 | 4.15 | 2.5% | 0 / 0 |
| S4 exact layout vs random motion | 60 | 97,850.8 | 98,864.8 | 97,850.8 | 98,357.8 | 9.78 | 0.0% | 0 / 0 |

Delivery timing was consistent with the retained reports: S1 first/last mean
delivery 41.5/215.5, S2 36.7/46.8, S3 120.9/336.9, and S4 40.7/383.5 timesteps.

| Scenario | Mean rollout wall time | Maximum single action latency |
| --- | ---: | ---: |
| S1 | 0.128 s | 45.9 ms |
| S2 | 0.180 s | 8.5 ms |
| S3 | 0.208 s | 61.0 ms |
| S4 | 0.050 s | 0.4 ms |

All are below the configured 100 ms safety limit; startup and full per-rollout
timing are recorded in `outputs/stage_d_finalization/benchmark.json`.

## Generic fallback proxy

This is a proxy, not an official scenario score. `custom_room` was not part of
the fallback selection tournament. It exposes a real limitation: against a
second full greedy worker in that narrow layout, both positions deadlock (0/10
successful soups). Against a random-motion moving obstacle on the same held-out
layout it is reliable: 80,576 mean score, 70,764 minimum score, 8.0 mean soups,
and 0% zero-soup rate over ten position/seed rollouts. A separate `cramped_room`
noisy-partner diagnostic reached 30,047 mean but had a 40% zero-soup rate.

The fallback is therefore kept as an explicit, valid-action generic recovery
policy, not represented as a robust replacement for a known-layout specialist.

## Validation and reproduction

- `tests/test_stage_d_router.py`: 5 passed, including the clean-process smoke
  across all eight mapped routes.
- The three preserved hashes above match before and after the benchmark.
- No absolute path is stored in the specialist mapping; router path resolution
  is relative to that mapping.

Run the final benchmark from a clean process with:

```bash
.venv/bin/python -B scripts/benchmark_final_stage_d.py --output-dir outputs/stage_d_finalization
```

For the short route-loading smoke:

```bash
.venv/bin/python -B scripts/smoke_stage_d_router.py --output-dir outputs/stage_d_smoke_final --horizon 3
```
