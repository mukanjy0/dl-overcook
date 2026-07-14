# Failure analysis

> **Status:** documented design lessons. These failures informed the final
> deployment adapter; they are not hidden as failed runs.

## 1. Delivery events were not always valid soups

**Symptom.** Some early checkpoint reports recorded delivery-like events and
non-zero score while the environment sparse reward stayed at zero.

**Cause.** The historical path used a recipe-agnostic delivery ledger from
`game_stats`. On Counter Circuit this could include single-ingredient or
otherwise inactive-order soups. The teacher evaluator correctly awards no
sparse reward for those deliveries.

**Response.** The final benchmark counts only positive-reward soups. The
Counter Circuit teacher was made recipe-aware: it complements the disclosed
onion partner with tomato only when the pot remains compatible with an active
order. Distillation then learns the behavior needed by the `final/` contract.

**Lesson.** An instrumentation event is not automatically the task objective.
Validate score attribution against the actual reward and active-order semantics.

## 2. Manhattan distance selected unreachable targets

**Symptom.** On `asymmetric_advantages`, the generic policy could deliver once
then wait indefinitely.

**Cause.** The layout duplicates dispensers and serving windows across an
impassable wall. Manhattan distance favored a slightly nearer feature on the
unreachable half.

**Response.** The narrow teacher filters targets through the existing BFS
interaction path before comparing distances. The final distilled actor learns
this route without shipping a new navigation planner in the active route.

**Lesson.** Geometry heuristics must respect reachability, especially in
asymmetric layouts.

## 3. Self-play competence did not guarantee partner compatibility

**Symptom.** Historical self-play and exact-partner checkpoints showed
position asymmetry, zero-soup episodes, or weak behavior with sticky/random
partners.

**Cause.** Cooperative policies can encode implicit roles and timing
assumptions. A frozen partner with a different convention changes both traffic
and task allocation.

**Response.** Stage C introduced exact partner and weighted-pool configurations
with fresh partner sessions and balanced physical positions. Reports preserve
deterministic and stochastic diagnostics instead of hiding variance.

## 4. A general guided model did not transfer safely across layouts

**Symptom.** The Scenario 2 guided PPO model was strong enough to retain for
`coordination_ring`, but forced cross-layout evaluation produced no valid
Counter Circuit soups and failed every Scenario 4 index-1 case.

**Response.** The router keeps it only on its native layout. The model's
deterministic mode also scored zero in the forced comparison, so its native
route intentionally retains stochastic inference plus its pot-safety override.

**Lesson.** Reusing a model because its observation shape matches is not a
transfer result. Evaluate it under every target protocol before routing it.

## 5. Distillation is a deployment bridge, not a substitute for research evaluation

The final distilled actors are intentionally located in `final/`. They are
selected by teacher-compatible behavior over the four benchmark seeds. The
original PPO artifacts remain under `outputs/` with their historical
training-side reports. Mixing these artifact classes would obscure whether a
claim is about the research pipeline or the submission adapter.
