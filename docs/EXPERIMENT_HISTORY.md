# Experiment history

> **Status:** concise decision record. It favors decisions that changed the
> next experiment over a complete list of every run directory.

| Stage | Hypothesis / question | Experiment design | Result used for the decision | Next decision |
| --- | --- | --- | --- | --- |
| A | Can the pipeline learn and deploy a PPO policy end to end? | Seeded self-play PPO with versioned checkpoints and `build_policy` smoke tests. | Training, export, resume, and inference contracts worked; task behavior emerged. | Preserve the baseline and examine robustness rather than replacing core interfaces. |
| A | Does an explicit player-index feature or entropy schedule remove positional collapse? | Four controlled 200k AA ablations: index on/off × constant/annealed entropy. | Stochastic behavior changed, but deterministic position-1 performance remained weak in the recorded suite. | Do not justify an unchanged longer run solely from the 200k comparison. |
| B | Can reset-state coverage reduce cross-play brittleness? | Serialize/fingerprint valid states from policy pairings and train with an explicit mixed reset probability. | The mechanism was integrated as an opt-in extension, with standard-start evaluation left unchanged. | Keep state augmentation experimental; use partner-aware evaluation before claiming benefit. |
| C | Does exposure to frozen partners help? | Exact teacher-partner fine-tuning and weighted pools with seeded partner wrappers and balanced ego positions. | Produced reusable partner/session infrastructure and multiple selected research artifacts. | Keep reports separated by partner/mode/position; do not collapse them into teacher results. |
| C | Can additional Counter Circuit PPO training solve the task? | Long exact-partner continuations and checkpoint suites. | Historical suites produced artifacts, but the later teacher check exposed invalid-recipe counting. | Preserve the selected research checkpoint; change final metric and diagnosis instead of relabeling it a teacher winner. |
| D | Can one deployment policy handle known layouts? | Router mapped specialists by layout and physical index with artifact integrity checks. | The router preserved `build_policy` compatibility and made selection explicit. | Use narrow teachers only for disclosed deterministic failures. |
| Final | Can the submission satisfy the teacher's actual recipe/reward semantics? | Teacher-compatible benchmark plus short dataset aggregation from validated specialists. | Distilled actors produced positive-reward soups for Scenarios 1, 3, and 4; Scenario 2 guided PPO remained native-only. | Ship the self-contained `final/` bundle; retain original checkpoints outside it. |

## PPO experiment loop

The project used one implementation path for local and remote PPO experiments.
The goal was to make experiment changes configuration changes, not forks of the
training loop.

1. **Specify the hypothesis in YAML.** A config fixes layout, horizon,
   observation contract, model, PPO hyperparameters, partner sampler, seed,
   checkpoint cadence, and evaluation suite. Stage A uses self-play; Stage C
   changes only the partner/position sampler to use an exact partner or weighted
   pool.
2. **Run a cheap vertical smoke.** The 64-step
   [`ppo_smoke.yaml`](../configs/examples/ppo_smoke.yaml) validates config
   loading, vectorized rollout collection, one PPO update, checkpoint save, and
   inference export before a larger local/Kaggle/RunPod job.
3. **Train with resumable state.** A training checkpoint stores model,
   optimizer, counters, RNG state, effective configuration, and environment
   metadata. An inference artifact stores only the validated model/observation/
   action envelope and CPU weights.
4. **Evaluate saved candidates, not just the final update.** Checkpoint suites
   export each candidate and evaluate fixed seeds, partners, inference modes,
   and physical player positions. Selection is deterministic minimum-position
   score, then deterministic mean score, then steps.
5. **Interpret the protocol before acting.** Shaped rollout reward, sparse
   return, raw delivery events, and course score answer different questions.
   The legacy Counter Circuit discrepancy is why final selection was moved to
   the teacher's positive-reward protocol.

### Baseline PPO configuration

The representative Stage A configuration uses a 128×128 tanh MLP actor-critic,
six Overcooked actions, eight environments, 128 rollout steps, GAE
(`γ=0.99`, `λ=0.95`), clipping `0.2`, entropy `0.01`, and reward shaping `1.0`.
The precise values live in
[`configs/stage_a/train_self_play.yaml`](../configs/stage_a/train_self_play.yaml).
Continuations use a fresh optimizer/RNG when the experimental question calls
for it; the original checkpoint is still preserved.

## From internal benchmark to course benchmark

The distinction below prevents a common but misleading narrative:

| Phase | What was selected | Metric | Why it was useful | Why it was insufficient |
| --- | --- | --- | --- | --- |
| PPO research | checkpoint-suite selected inference artifact | internal delivery/score suite, partner/position diagnostics | compares learning runs under a controlled internal protocol | legacy delivery accounting could disagree with active-order sparse reward |
| Course adaptation | distilled actor in `final/` | unchanged course evaluator, four official seeds, positive-reward soups | verifies the policy the course will actually load and score | specializes disclosed layouts; it is not broad generalization evidence |

The adaptation did **not** overwrite the original selected PPO weights. It uses
them only as historical baselines/compatible initializations where applicable;
the active distilled files are confined to `final/policies/`.

## What the stages mean

- **Stable:** the PPO/checkpoint/evaluation interfaces and the `final/`
  submission contract.
- **Experimental:** state augmentation, partner pools, exact-partner
  continuations, and cross-play comparisons.
- **Historical:** large local/Kaggle/RunPod output directories and legacy
  delivery-ledger reports. They are preserved for auditability, not used as
  final benchmark evidence.

## Decision checkpoints worth reviewing

1. [`PROJECT_STRATEGY.md`](PROJECT_STRATEGY.md) states the initial
   self-play → state coverage → partner robustness → specialization sequence.
2. [`stage_a_ablations.md`](stage_a_ablations.md) records the controlled AA
   ablations and their deterministic limitation.
3. [`stage_b_state_augmentation.md`](stage_b_state_augmentation.md) documents
   serialization, environment fingerprints, and reset semantics.
4. [`stage_c_partners.md`](stage_c_partners.md) documents frozen partner pools
   and exact-partner configurations.
5. [`FAILURE_ANALYSIS.md`](FAILURE_ANALYSIS.md) explains why the final benchmark
   did not simply reuse the historical checkpoint-selection outcome.

## Excluded history

Interrupted runs, rejected remote kernels, and one-off planner probes are not
ranked in the canonical results table. They may remain in ignored `outputs/` or
`kaggle/` directories because they are useful for debugging provenance, but
they are not included in the reported results unless their protocol is
reconstructed and reported separately.
