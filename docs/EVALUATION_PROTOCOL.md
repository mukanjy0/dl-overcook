# Evaluation protocol

> **Status:** stable for reported final results. Historical reports may follow
> earlier protocols and are explicitly labeled as such in
> [`RESULTS.md`](RESULTS.md).

## Separate the protocols

| Protocol | Purpose | What it may support | What it must not support |
| --- | --- | --- | --- |
| Training metrics | optimize PPO and diagnose learning | loss, entropy, shaped/sparse rollout trends | final competition claims |
| Checkpoint suite | compare saved research checkpoints | deterministic/stochastic, partner, seed, and position diagnostics | claims about a changed teacher runner |
| Final teacher benchmark | select the submission | teacher-compatible score and positive-reward soups | claims about broad unseen-layout generalization |
| One-off debugging | isolate a failure | diagnosis and regression tests | leaderboard or mean-score claims |

## Research checkpoint evaluation

`scripts/train.py --evaluate-checkpoints` exports every saved checkpoint and
runs configured cases through the normal `build_policy` path. The formal
selection order recorded in current reports is:

1. deterministic minimum position score;
2. deterministic mean official score;
3. environment steps as a tie-breaker.

Stochastic evaluation is retained as a diagnostic. Each case records layout,
partner, mode, seed, ego index, sparse return, delivery events, timeouts,
invalid-action replacements, soup count, and score.

## Final teacher benchmark

The canonical command is run from the self-contained bundle:

```bash
cd final
../.venv/bin/python -m src.evaluate_competition \
  --config configs/competition.yaml --all-scenarios
```

The fixed seed set is `67, 607, 6007, 60007`. Scenarios 1–3 evaluate the
student at index 0; Scenario 4 evaluates both physical indexes. The evaluator
uses its configured horizon, partner, action wrapper, and score function.

For final reporting, a soup is counted only when the environment grants
positive sparse reward. This prevents a raw interaction or legacy delivery
event from being mistaken for a completed active-order recipe.

## Reporting rules

- Always state partner, stochasticity/noise, seed count, and player positions.
- Do not average results across different partners, layouts, or scoring
  implementations without labeling the aggregation as a new protocol.
- Quote score and soup counts together when possible; a high score with
  zero valid sparse reward is a debugging signal, not a final result.
- Report zero-soup rate, timeouts, and invalid actions where the suite records
  them.
- Treat a result as verified only when its source report/configuration remains
  available and its protocol can be identified.

## Reproducing an individual scenario

```bash
cd final
../.venv/bin/python -m src.evaluate_competition \
  --config configs/competition.yaml --scenario 3
```

The evaluator writes CSV attempts and the effective competition configuration
under `final/results/competition_eval/`. Those generated files are ignored;
the committed `final/README_STUDENT_AGENT.md` records the accepted benchmark
summary and model hashes.
