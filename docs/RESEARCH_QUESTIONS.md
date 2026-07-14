# Research questions

> **Status:** research framing. These are experiment questions, not claims of
> new RL theory.

## RQ1 — Does self-play produce usable task competence?

**Hypothesis.** A compact PPO actor-critic trained in self-play can learn a
non-trivial soup-production policy when observation, action, checkpoint, and
evaluation contracts are kept fixed.

**Design.** Stage A trains the same MLP actor-critic with seeded vectorized
rollouts and exports both resumable training checkpoints and CPU-neutral
inference artifacts. The first criterion is vertical compatibility:
`train → checkpoint → export → build_policy → episode`.

**Interpretation boundary.** Self-play return and shaped reward establish
learning progress, but do not establish compatibility with a teacher partner or
a competition evaluator.

## RQ2 — Does state augmentation improve robustness to state-distribution shift?

**Paper-inspired premise.** The project strategy was informed by the idea that
cross-play failures can arise because a policy has not encountered states
created by another policy's coordination convention. Resetting from validated
states collected under varied pairings is a controlled way to widen coverage
without changing the environment transition rules.

**Design.** Stage B serializes exact valid states, fingerprints the source
environment, samples them with an explicit probability, and keeps evaluation on
the standard start-state distribution. The state source is an opt-in extension
point, not an implicit change to all experiments.

**Decision rule.** Retain augmentation only if cross-play/partner evaluation
improves without materially eroding task competence. State augmentation was not
treated as a universal fix; it remains an experimental mechanism.

## RQ3 — Does training against a frozen partner distribution generalize better than self-play?

**Hypothesis.** A policy trained only with its current copy may overfit to a
shared convention. Sampling teacher scripted policies, sticky/random variants,
and historical policies can make the ego policy less dependent on one partner.

**Design.** Stage C freezes the partner, samples it per episode, balances the
ego's physical player index, and sends only ego transitions through PPO. Exact
partner fine-tuning and weighted pools use the same interface and differ only
by configuration.

## RQ4 — Which metric should decide deployment?

**Question.** A checkpoint can look good under a training-side delivery ledger,
shaped reward, or proxy suite while failing the teacher's recipe/reward rules.

**Decision.** Keep those measurements as diagnostics, but select the final
bundle with the unchanged teacher evaluator, positive-reward soups, fixed
official seeds, and both required positions. This distinction is central to
the final distillation decision.

## RQ5 — When is a targeted deployment adapter justified?

**Observation.** The disclosed layouts exposed deterministic failure modes:
unreachable Manhattan targets on the asymmetric layout, invalid mixed recipes
on Counter Circuit, and partner obstruction on Scenario 4.

**Decision.** Use narrow specialist teachers to produce valid behavior, then
distill them into the same compact actor architecture used by the final router.
This was scoped to `final/` compatibility; it does not replace the original
PPO research artifacts or claim general planner learning.

## What was not pursued

The project deliberately did not introduce a learned theory-of-mind model,
transformer partner model, or a large population-training system. Those would
add modeling and evaluation uncertainty before the simpler self-play,
state-coverage, partner-distribution, and deployment-contract questions were
answered.
