# Scenario 4 speed run

Winner: `scenario4_exact_fixed_pot_b`, a deterministic native planner that
uses the existing collision-aware greedy task FSM and assigns ingredient work
to fixed pot B.  No learned artifact is required.

## Selection evidence

All reported planner evaluations used the disclosed `random_motion` partner,
both physical ego positions, and a 400-step horizon.  The winner aggregates
460 rollouts: seeds 0--229, once at each physical position.

| Candidate | Rollouts | Minimum score | Mean score | Zero-soup rate | Mean soups |
| --- | ---: | ---: | ---: | ---: | ---: |
| Exact fixed pot B (winner) | 460 | 80,659 | 98,287.6 | 0.0% | 9.776 |
| Exact nearest pot | 60 | 80,488 | 96,689.4 | 0.0% | 9.617 |
| Recovery/two-pot | 60 | 80,918 | 97,347.9 | 0.0% | 9.700 |
| Exact fixed pot A | 60 | 70,605 | 78,766.9 | 0.0% | 7.817 |

Winner position metrics: position 0 minimum/mean `80,659 / 97,613.5`; position
1 minimum/mean `90,371 / 98,961.7`.  Mean first and last delivery timesteps
were `40.57` and `383.23`. There were two total safe-action timeouts and zero
invalid actions. The current generic runner does not persist the planner's
per-step blocked-target flag, so blocked time is not recoverable from the
completed remote traces; the two timeout events are the retained obstruction
diagnostic.

The PPO hedge was stopped without a deployable checkpoint after the planner
established this robust score floor. The initial three scripted kernels were
also terminated after identifying their incompatible observation-policy
adapter path; they are excluded from selection.

## Stage D mapping

`configs/stage_d/specialists.yaml` now maps both `scenario_4` physical indexes
to the built-in `scenario4_planner` with:

```yaml
pot_strategy: fixed_b
blocked_threshold: 999
avoid_teammate: false
```

The policy implementation is `policies/scenario4_policy.py`; it reuses
`GreedyFullTaskPolicy` navigation and task logic, filtering only ingredient
assignment. There is no checkpoint artifact to ship.

## Remote artifacts and visuals

- FSM sweep: `kaggle/v96/outputs/run_summary.json`
- Symbolic sweep: `kaggle/v97/outputs/run_summary.json`
- Recovery sweep: `kaggle/v98/outputs/run_summary.json`
- Greedy variants: `kaggle/v99/outputs/run_summary.json`
- Fixed-B diagnostics: `kaggle/v100/outputs/run_summary.json` and
  `kaggle/v101/outputs/run_summary.json`
- Worst evaluated rollout: `outputs/scenario4_speed/rollouts/worst_seed123_pos0.gif`
- Best evaluated rollout: `outputs/scenario4_speed/rollouts/best_seed223_pos1.gif`

Teacher-compatible smoke command:

```bash
.venv/bin/python scripts/render_scenario4_rollout.py \
  --seed 123 --position 0 \
  --gif outputs/scenario4_speed/rollouts/smoke.gif
```
