# Stage B state augmentation

Stage B adds trajectory-backed episode resets to the existing PPO path. It is
disabled unless a training YAML explicitly selects `augmented` or `mixed`
resets. Environment construction, observations, PPO, checkpointing, evaluation,
official scoring, and `build_policy` remain unchanged.

The implementation follows the repository's state-augmentation paper: collect
trajectories from configured policy pairings, retain every `k`-th non-terminal
state, and use those states as a wider training reset distribution. The paper's
iterative layout sweeps are deliberately outside this infrastructure task.

## Architecture

The responsibilities are separated under `src/state_augmentation/`:

- `collection.py` runs configured policy pairings through the existing
  `build_env`, `ConfiguredPartnerFactory`, `build_policy`, and `run_episode`
  paths. It never implements a second episode loop.
- `serialization.py` uses `OvercookedState.to_dict/from_dict`, fingerprints the
  effective environment, restores states, and probes restored states through
  the upstream transition function.
- `buffer.py` owns the versioned JSON/JSON.GZ schema, atomic storage, full
  validation, and inspection summaries.
- `sampling.py` uniformly samples records with a caller-owned NumPy generator.
  A fixed generator seed therefore gives a fixed record sequence.
- `sources.py` adapts a validated buffer to the existing `StateSource.sample`
  reset interface and implements standard-only, augmented-only, and mixed
  reset distributions.

`src/training/trainer.py` only builds the configured `StateSource` and gives it
to the existing self-play or frozen-partner rollout collector. Evaluation never
uses training state augmentation.

## State-buffer schema

The artifact profile is `overcooked_state_buffer`, currently at schema version
2. JSON and gzip-compressed JSON have identical contents.

| Field | Purpose |
| --- | --- |
| `schema_version`, `profile`, `created_at_utc` | Format identity and provenance |
| `environment` | Layout name, layout fingerprint, horizon, Overcooked-AI version, state-serialization version, dynamics fingerprint, and effective environment config |
| `source_policies` | Stable identifier, declared source, policy type/name, policy config, provenance path, checkpoint SHA-256, and portable `sha256:<digest>` identity |
| `collection_config` | Complete effective collection YAML, including `every_k`, pairings, and seeds |
| `trajectories` | Pairing, physical assignment, seed, returns, deliveries, official score, and completion/success metadata for every source trajectory |
| `records[].physical_player_assignment` | Source policy identifier controlling physical players `0` and `1` |
| `records[].episode_id`, `trajectory_id`, `timestep`, `seed` | Exact trajectory context |
| `records[].serialized_state` | Canonical JSON-compatible `OvercookedState.to_dict()` payload |

Saving or loading validates every record. Unsupported schema/environment/state
versions, wrong layouts or dynamics, malformed state data,
unknown policy identifiers, duplicate record IDs, terminal states, and states
that the configured MDP cannot transition from fail with a clear compatibility
error.

Inspection additionally reports exact and timestep-agnostic duplicates,
per-assignment and per-pairing balance, timestep quantiles/regions, coarse task
progress, and successful/failed trajectory outcomes. A trajectory is successful
when it contains at least one upstream `soup_delivery` event.

## Reset-time semantics

Restoration preserves the serialized state's absolute `timestep`. With the same
horizon, an augmented episode therefore receives the original remaining time,
not a fresh full horizon. This preserves cooking/task state and the original
time-to-go context: future upstream event and score timestamps continue from
the restored timestep.

`OvercookedEnv.reset` still creates fresh event lists and zero cumulative sparse
and shaped rewards. Rollout return accumulators and external episode identifiers
are reset normally. Canonical results record `start_timestep` and calculate
`episode_length` as post-reset transitions rather than the absolute final
timestep. `run_episode` resets both agents; frozen-partner PPO creates and
resets a fresh partner session; the trainable actor-critic is stateless. Wrapper
history such as sticky previous actions is therefore not inherited from the
trajectory source.

The collection horizon is provenance rather than a transition-format contract.
A different target horizon is accepted when layout, dynamics, and dependency
metadata match and every sampled state has `timestep < target_horizon`. The
target episode then has `target_horizon - restored_timestep` steps remaining.
A target horizon that makes any record terminal is rejected.

## Collect, validate, and inspect

The short committed example collects every fifth state from one scripted
cross-play pairing:

```bash
.venv/bin/python scripts/collect_state_buffer.py \
  --config configs/stage_b/collect_state_buffer_example.yaml

.venv/bin/python scripts/inspect_state_buffer.py \
  --buffer outputs/stage_b/buffers/example.json.gz \
  --environment-config configs/stage_b/collect_state_buffer_example.yaml
```

The coordination-ring preflight using all ordered cross-play assignments of the
three Stage A checkpoints is:

```bash
.venv/bin/python scripts/collect_state_buffer.py \
  --config configs/stage_b/collect_coordination_ring_preflight.yaml

.venv/bin/python scripts/inspect_state_buffer.py \
  --buffer outputs/stage_b/buffers/coordination_ring_stage_a_crossplay_preflight.json.gz \
  --environment-config configs/stage_b/collect_coordination_ring_preflight.yaml
```

The relevant collection format is:

```yaml
paths_relative_to_config: true
seed: 67
environment: {layout_name: cramped_room, horizon: 400, old_dynamics: true}
observation: {type: featurized, include_agent_index: true}
collection:
  output_path: ../../outputs/stage_b/buffers/states.json.gz
  every_k: 10
  include_initial_state: true
  num_episodes: 8
  episode_seeds: [67, 68, 69, 70, 71, 72, 73, 74]
  pairings:
    - id: frozen_cross_play
      player_0:
        identifier: sp_seed_67
        source: frozen_checkpoint
        observation: {type: featurized, include_agent_index: true}
        policy:
          type: python_class
          path: ../../policies/rl_policy.py
          class_name: StudentAgent
          config: {checkpoint_path: ../../outputs/seed67/inference.pt, device: cpu}
      player_1:
        identifier: sp_seed_68
        source: frozen_checkpoint
        observation: {type: featurized, include_agent_index: true}
        policy:
          type: python_class
          path: ../../policies/rl_policy.py
          class_name: StudentAgent
          config: {checkpoint_path: ../../outputs/seed68/inference.pt, device: cpu}
```

Use the same policy identifier/configuration in both positions for self-play.
Use two checkpoint identifiers for frozen cross-play. A teacher-provided
scripted source uses the same partner format, for example:

```yaml
player_1:
  identifier: greedy_full_task
  source: teacher_scripted
  policy: {type: builtin, name: greedy_full_task}
```

All policies are fresh sessions built behind the existing partner interface.
Sticky and random-action wrapper fields can be placed in `policy` exactly as in
normal evaluation/Stage C configuration.

## Train with augmented resets

Training adds one optional section:

```yaml
state_augmentation:
  reset_mode: mixed             # standard | augmented | mixed
  buffer_path: ../../outputs/stage_b/buffers/states.json.gz
  augmented_probability: 0.5    # required strictly between 0 and 1 for mixed
```

- `standard` always uses the upstream initial state. It is also the implicit
  default when the section is absent.
- `augmented` always samples a buffer state.
- `mixed` independently selects an augmented state at each environment reset
  with `augmented_probability`; otherwise it uses the standard state.

The short local vertical run is:

```bash
.venv/bin/python scripts/train.py \
  --config configs/stage_b/train_mixed_resets_example.yaml
```

The effective config, JSONL update metrics, progress, and final summary record
the reset mode plus cumulative standard/augmented reset counts. Normal training
and resume/export behavior are otherwise unchanged.

## Add a new state source

For a new trajectory producer already supported by `build_policy`, add a
pairing entry with a unique `identifier`, its normal `policy` mapping, and an
optional policy-specific `observation` mapping. No collection code changes are
needed. A new policy family should first be added behind the existing partner
factory/build-policy boundary.

For a different reset sampler, implement the small `StateSource.sample(mdp,
rng)` interface in `src/state_initialization.py`, validate its artifact before
the first rollout, and pass it to the existing collectors. Keep storage,
sampling, and restoration concerns in their current modules.

## Compatibility limitations and failures

- Buffers are tied to exact layout content, transition-relevant dynamics,
  state-serialization version, and installed Overcooked-AI version. Horizon
  changes are permitted only when every record remains nonterminal.
- Only physical environment state is restored. Recurrent policy hidden state,
  wrapper history, and partial episode statistics are intentionally reset for a
  new training episode.
- Terminal states are not collected and are rejected if manually inserted.
- Checkpoint paths in metadata are provenance only. Portable identity and
  validation use checkpoint SHA-256 values; collection must have readable
  checkpoint files, while training only needs the completed buffer.
- Generated buffers live below `outputs/`, which is already ignored. The Kaggle
  packager copies a non-standard reset buffer when its path is relative, inside
  the project, and already exists. For RunPod, make the buffer available during
  setup at the same config-relative path. Preserve remote artifacts locally;
  temporary Kaggle/Pod storage is not a durable source of truth.

The infrastructure is ready for layout-specific collection and short
validation. It makes no claim about the reset mixture, source-policy set,
`every_k`, or training budget that will perform best on a given layout.

## Coordination Ring parallel matrix

The committed Coordination Ring workstream expands collection to all six
ordered Stage A cross-play assignments over five seeds, then compares standard,
mixed 0.25, mixed 0.50, and augmented resets at seeds 0–2 for 200704 steps.
`configs/stage_b/runpod_coordination_ring_smoke.yaml` defines the required
single-Pod 32768-step lifecycle validation;
`configs/stage_b/runpod_coordination_ring_matrix.yaml` defines the 12 jobs.
Both use immutable archive upload and explicit file inputs rather than relying
on repository authentication or transient Pod state. The complete guarded
handoff is
[`workstreams/cr_stage_b_runpod_prompt.md`](workstreams/cr_stage_b_runpod_prompt.md).
