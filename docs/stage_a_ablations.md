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
