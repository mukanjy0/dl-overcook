# Stage A asymmetric ablations

The initial controlled comparison uses four 200,000-step self-play PPO runs on
`asymmetric_advantages`, all with seed 67 and otherwise identical model,
rollout, reward-shaping, and optimizer settings:

| Configuration | Agent index | Entropy coefficient |
| --- | --- | --- |
| `ablation_baseline_200k.yaml` | included | constant 0.01 |
| `ablation_no_agent_index_200k.yaml` | omitted | constant 0.01 |
| `ablation_entropy_anneal_200k.yaml` | included | 0.01 → 0.001 over 200k steps |
| `ablation_no_index_entropy_anneal_200k.yaml` | omitted | 0.01 → 0.001 over 200k steps |

Checkpoints are saved approximately every 50,000 environment steps. Every
checkpoint is evaluated through `build_policy` against the disclosed
`greedy_full_task` partner using seeds 67–71, ego positions 0 and 1, and both
deterministic and stochastic inference.

The report at `checkpoint_evaluation/checkpoint_evaluation.json` contains, for
each mode and position:

- soup counts for every episode and their mean;
- official scores for every episode and their mean;
- zero-soup rate;
- mean and minimum position score.

Deployment selection maximizes deterministic minimum-position score first and
deterministic mean official score second. Later environment steps are only a
tie-breaker. Stochastic performance is retained to diagnose policy-distribution
quality and deterministic argmax collapse.

Run locally or remotely through the same entry point:

```bash
.venv/bin/python scripts/train.py \
  --config configs/stage_a/ablation_baseline_200k.yaml \
  --evaluate-checkpoints
```

## Initial 200k result

All four runs completed 200,704 actual environment steps and evaluated five
saved checkpoints. Their selected final checkpoints produced the following
five-seed results against `greedy_full_task`:

| Variant | Deterministic pos. 0 soups / score | Deterministic pos. 1 soups / score | Stochastic pos. 0 soups / score | Stochastic pos. 1 soups / score |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 1.0 / 14,026 | 0.0 / 0 | 2.0 / 22,554.0 | 1.6 / 19,345.0 |
| No agent index | 1.0 / 14,026 | 0.0 / 0 | 5.4 / 56,322.0 | 1.6 / 19,376.6 |
| Entropy annealing | 1.0 / 14,026 | 0.0 / 0 | 4.2 / 44,232.2 | 1.0 / 13,779.6 |
| Both changes | 1.0 / 14,026 | 0.0 / 0 | 2.8 / 29,494.2 | 1.0 / 13,623.4 |

Every variant therefore had deterministic minimum-position score 0,
deterministic mean score 7,013, and a 50% deterministic zero-soup rate. Removing
the explicit index improved stochastic behavior but did not change argmax
behavior. The preserved 900,096-step checkpoint also has minimum-position score
0, but remains the deployment selection through the second criterion with mean
score 65,273. Extending any unchanged 200k recipe is not supported by this
comparison; the next experiment should change the optimization target rather
than only add steps.
