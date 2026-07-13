# Counter Circuit Stage A local workstream

Operate only on `/Users/katharsis/Developer/dl/overcook` and branch
`codex/cc-stage-a-local`.

First re-evaluate these promising source checkpoints through the existing
checkpoint evaluator with seeds 0–19, both physical ego positions, and both
inference modes:

- seed 0, step 300032;
- seed 1, step 400384;
- seed 2, step 900096.

The committed continuation configurations use the strongest previously
observed candidate, seed 2 at step 900096:

- `configs/stage_a/counter_circuit_control_continuation_50k.yaml` preserves the
  optimizer and RNG state and changes no PPO or reward coefficients;
- `configs/stage_a/counter_circuit_consolidation_continuation_50k.yaml` starts a
  fresh optimizer/RNG stream and anneals reward shaping 1.0 → 0.1 and entropy
  0.01 → 0.001 over exactly 50176 continuation steps.

Both save at the midpoint and endpoint and evaluate 20 fixed seeds. Do not run
longer training unless stochastic mean soups reaches at least 1.0 and the
zero-soup rate is at most 50%. A longer extension additionally requires
non-zero deterministic progress in both positions or an unambiguous stochastic
improvement over the source checkpoint.

## Completed result

The 20-seed diagnostic confirmed seed 2 at step 900096 as the source: it had
0.45 stochastic mean soups, a 60% stochastic zero-soup rate, and 5,303.95 mean
official score. Every diagnostic deterministic rollout delivered zero soup.

| Checkpoint | Deterministic soups / zero rate | Stochastic soups / zero rate | Stochastic mean score |
| --- | ---: | ---: | ---: |
| Source, seed 2 step 900096 | 0.00 / 100% | 0.45 / 60% | 5,303.95 |
| Control, step 925696 | 0.00 / 100% | 0.50 / 55% | 5,465.25 |
| Control, step 950272 | 0.00 / 100% | 0.75 / 40% | 8,014.15 |
| Consolidation, step 925696 | 0.00 / 100% | 0.55 / 45% | 6,200.15 |
| Consolidation, step 950272 | 0.00 / 100% | 0.00 / 100% | 0.00 |

Both ego positions had identical metrics in this self-play suite. Neither
continuation created deterministic task progress. The unchanged control was the
best stochastic result, but it missed the 1.0-soup threshold. Consequently the
extension gate is not met and this workstream must not launch a longer Counter
Circuit Stage A run without a newly approved intervention.

## Stage C exact-partner pilot

The next bounded intervention is
`configs/stage_c/counter_circuit_exact_partner_seed2_150k.yaml`. It begins from
the productive 900,096-step seed-2 checkpoint, resets optimizer and RNG state,
and makes 150,528 additional environment steps against the disclosed frozen
partner: `greedy_full_task` with both `sticky_action_prob` and
`random_action_prob` set to `0.10`. Training balances physical ego positions.

`configs/stage_c/evaluate_counter_circuit_disclosed_partner_ceiling.yaml` first
checks that this partner/layout combination can produce task progress with two
scripted agents. The pilot evaluates every saved checkpoint on seeds 0–19, both
positions, and both inference modes. It extends only if the exact-partner suite
shows meaningful movement toward the two-soup target; otherwise the next
intervention is a paper-aligned Stage A competence recovery, not more unchanged
self-play.

## Stage C result

The ceiling check was healthy: a scripted ego paired with the disclosed partner
averaged 4.75 total deliveries (5.15 with ego at physical position 0 and 4.35
at position 1), with a 2.5% zero-soup rate across 20 seeds. The environment and
partner therefore do not impose a sub-two-soup ceiling.

The exact-partner training resumed the selected 900,096-step seed-2 checkpoint
with a fresh optimizer/RNG stream for the first 150,528 steps, then continued
the same stream to 300,032 additional steps. It used exactly balanced training
exposure to physical ego positions (150,016 steps per position overall).

| Checkpoint step | Deterministic soups (pos 0 / pos 1) | Stochastic soups (pos 0 / pos 1) | Stochastic zero rate | Stochastic mean score |
| ---: | ---: | ---: | ---: | ---: |
| 1,000,448 | 0.125 (0.150 / 0.100) | 1.825 (1.950 / 1.700) | 5.0% | 19,467.2 |
| 1,050,624 | 0.100 (0.100 / 0.100) | 1.900 (2.050 / 1.750) | 12.5% | 20,121.8 |
| 1,100,800 | 0.125 (0.250 / 0.000) | 2.100 (1.950 / 2.250) | 5.0% | 22,092.7 |
| 1,150,976 | 1.600 (3.150 / 0.050) | 2.475 (2.100 / 2.850) | 2.5% | 25,966.7 |
| 1,200,128 | 0.125 (0.200 / 0.050) | 2.400 (2.350 / 2.450) | 2.5% | 25,226.8 |

Stage C is successful as a stochastic exact-partner intervention: it reaches
the two-soup target in both positions at the final checkpoint, with a low
zero-soup rate. It is **not yet a successful deterministic deployment**. The
deterministic policy remains mostly collapsed, and the best deterministic
checkpoint is sharply role-specialized. Across both exact-partner runs, the
deterministic-first criterion selects step 1,000,448 (minimum position score
1,226.0; mean score 1,617.4), which is far below the score target. The 300k
continuation considered in isolation selects step 1,150,976, but that artifact
is also not balanced across positions. Preserve the final 1,200,128 stochastic
candidate for diagnosis, but do not replace the deployment artifact or relax
the deterministic selection rule based on this result alone.
