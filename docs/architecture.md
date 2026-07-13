# Stage A architecture and opt-in Stage B/C extensions

## Compatibility spine

The existing execution path remains authoritative:

```text
YAML -> build_env -> ObservationBuilder -> build_policy
     -> StudentAgentAdapter -> safety/random/sticky wrappers
     -> AgentPair -> run_episode -> runner/evaluator hooks
```

`src.policy_loader.build_policy` retains its signature and wrapper order.
`policies/rl_policy.py` is intentionally thin: it loads a compact inference
artifact, validates observations, selects a device, calls the actor-critic in
inference mode, and returns one integer in the existing six-action convention.
It does not import PPO, optimizers, rollout collection, evaluation,
demonstrations, rendering, or Kaggle code.

## Module responsibilities

- `src/environment.py` owns upstream MDP/environment construction. Its optional
  `StateSource` hook defaults to the unchanged upstream standard start state.
- `src/observations.py`, `src/constants.py`, and `src/policy_loader.py` remain
  the sources of observation, action, and policy-loading semantics.
- `src/episode.py` owns the canonical normal episode loop and immutable episode
  result. Logging, rendering, and demonstration collection attach as hooks.
- `src/evaluation/scoring.py` is the only official-score implementation.
  Delivery timestamps come directly from upstream `soup_delivery` events;
  timeouts are counted for ego, partner, and team.
- `src/models/` owns the stable inference/trainable policy contracts,
  observation metadata, and the shared actor-critic.
- `src/training/` owns self-play and frozen-partner rollout collection, PPO
  updates, and reusable training orchestration. Both player positions use the
  current model in Stage A. Stage C emits PPO transitions only for the ego;
  configured partner sessions never enter the optimizer.
- `src/state_augmentation/` owns Stage B trajectory collection, exact state
  serialization/restoration, versioned buffer storage, deterministic sampling,
  compatibility validation, and buffered reset sources. PPO only sees the
  existing `StateSource` interface.
- `src/partners/` owns partner specifications, fresh-session construction, and
  sampling. Stage A selects current-policy self-play. Opt-in Stage C selects a
  weighted pool or one exact frozen partner and balances the ego across physical
  player positions.
- `src/evaluation/` owns layout/partner/seed/position suites built on the normal
  runner. Canonical compatibility evaluation loads the ego via `build_policy`.
  Suite reports separate deterministic and stochastic inference and summarize
  soup counts, official scores, and zero-soup rates by ego position.
  Checkpoint suites may use the `self_play` partner sentinel to load a second
  independent session from the artifact currently being evaluated. Configured
  disclosed partners may apply `sticky_action_prob` and
  `random_action_prob` through the normal policy-wrapper path.
- `src/evaluation/checkpoint_selection.py` exports and evaluates every saved
  training checkpoint. It selects deployment artifacts lexicographically by
  deterministic minimum-position score and deterministic mean official score.
- `src/checkpointing.py` is the only checkpoint serialization and validation
  boundary.
- `src/experiment_config.py` validates Stage A and opt-in Stage B/C training
  configuration and resolves paths relative to the YAML file. Existing runtime
  YAMLs opt into this path resolution only with
  `paths_relative_to_config: true`.
- `scripts/` contains thin local/remote orchestration. Kaggle code packages and
  invokes these entry points; it does not implement learning or evaluation.

## Stable interfaces

- `InferencePolicy.reset/act` provides one session's inference state.
- `TrainablePolicy.act_batch/evaluate_actions` provides PPO operations without
  owning optimizer or rollout state.
- `PartnerFactory.build` creates a fresh configured partner through the existing
  loader; `PartnerSampler.sample` selects a `PartnerSpec`.
- `EgoPositionSampler.sample` selects the ego's physical player position. The
  Stage C implementation alternates positions with a seeded initial choice.
- `StateSource.sample` optionally supplies an upstream `OvercookedState` at
  reset. `StandardStateSource` selects the standard start.
- `evaluate_from_config` evaluates configured suites through `run_from_config`.
- `calculate_official_score` is pure and takes delivery timestamps, horizon,
  and total team timeouts.
- checkpoint functions distinguish resumable training state from deployable
  inference state and fail clearly on schema, model, observation, action, or
  dependency incompatibility.

## Checkpoint profiles

Training checkpoints contain model/optimizer state, trainer counters, RNG
states, the effective configuration, environment metadata, and dependency/code
versions. They are trusted local artifacts loaded only for resume/export.

Inference artifacts contain schema/model/observation/action/environment and
dependency metadata plus CPU weights. They never store a fixed device. The
adapter maps weights to CPU first, resolves `auto|cpu|cuda`, moves the model,
sets evaluation mode, and performs a metadata-sized CUDA warm-up. Explicit CUDA
also falls back to CPU when unavailable.

The vertical compatibility contract is:

```text
short train -> training checkpoint -> resume -> inference export
-> policies/rl_policy.py -> build_policy -> positions 0 and 1
-> full episode without wrapper replacements
```

## Configuration and execution environments

Training YAML separates experiment, environment, observation, model, training,
partner, state-augmentation, evaluation, checkpoint, and output sections. The
experiment seed is the source for deterministic
model/environment/partner/position/reset streams. Stage B is disabled unless
`state_augmentation.reset_mode` is explicitly changed from `standard`; Stage C
is disabled unless `partner.sampler` is explicitly changed from `self_play` to
`weighted_pool` or `exact`. No core module contains a Kaggle path, layout
choice, checkpoint location, partner name, or device choice.

Local `.venv` and Kaggle use the same `scripts/train.py` and YAML. Kaggle only
overrides the output root and records the effective dependency/CUDA state and a
durable summary. Important remote artifacts must be downloaded; temporary
Kaggle storage is not their only intended copy.

`observation.include_agent_index` controls the explicit physical-player one-hot
stored in the checkpoint observation contract. PPO entropy can remain constant,
or linearly anneal when `entropy_final_coefficient` and optionally
`entropy_anneal_steps` are set under `training.ppo`. Each training metrics record
contains the effective entropy coefficient used for that update.

## Stage C frozen-partner path

At each new episode, the frozen-partner collector samples a `PartnerSpec`,
samples the balanced ego position, and asks `ConfiguredPartnerFactory` for a
fresh partner session. The factory delegates to the unchanged `build_policy`
signature, so teacher builtins, perturbation wrappers, and exported inference
artifacts share normal runner semantics. A checkpoint partner may declare its
own observation builder without changing the ego observation contract.

Weighted sampling is normalized from positive configuration weights. Exact
sampling selects one named entry from the same pool format, which keeps
best-response fine-tuning a configuration change rather than a separate
algorithm. Training metrics record partner and ego-position environment-step
counts. Evaluation remains the normal suite evaluator and reports every partner
case separately with per-position metrics in deterministic and stochastic ego
modes.

See [stage_c_partners.md](stage_c_partners.md) for the configuration contract and
commands.

## Stage B state-augmentation path

Collection configurations define complete physical policy pairings. Each
policy session is built through `ConfiguredPartnerFactory`, and the canonical
episode loop exposes reset/step hooks for sampling without a second environment
implementation. Buffers record the full collection configuration and source
provenance, then validate every state against an exact environment fingerprint
before PPO starts.

At reset, a seeded `BufferedStateSource` selects the normal state, a uniformly
sampled augmented state, or a configured mixture. The environment adapter
deep-copies each restored state. Evaluation and official scoring do not use this
training-only reset path. See
[stage_b_state_augmentation.md](stage_b_state_augmentation.md) for the schema,
commands, and failure modes.

Restoration preserves the state's absolute timestep, so the normal case retains
the source trajectory's remaining horizon. Horizon itself is not part of the
transition fingerprint: a target horizon may differ if every stored state is
still nonterminal. Environment reset still clears upstream event/reward stats,
and policy sessions are reset through their existing lifecycle.

## Later extension points

Stage D can add broader configuration suites and checkpoint selection. Later
additions should implement the existing interfaces without replacing the
teacher-compatible spine.
