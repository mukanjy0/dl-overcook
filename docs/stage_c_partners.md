# Stage C frozen-partner training

Stage C adds opt-in PPO best-response training against frozen partner sessions.
It reuses Stage A environment construction, observations, checkpoints, PPO,
episode results, official scoring, and evaluation. Existing configurations with
`partner.sampler: self_play` are unchanged.

Behavior cloning is not part of this stage.

## Partner interfaces

`PartnerSpec` is the episode-independent description passed between the sampler
and factory. It contains:

- `name`: stable reporting and exact-selection name;
- `policy_config`: the normal configuration accepted by `build_policy`;
- `observation_config`: an optional partner-specific observation builder;
- `source`: descriptive metadata such as `teacher_scripted` or
  `frozen_checkpoint`.

`PartnerSampler.sample(rng, episode_context)` selects a `PartnerSpec`.
`ConfiguredPartnerFactory.build(...)` creates a fresh session through
`build_policy`. `EgoPositionSampler.sample(...)` selects physical position 0 or
1. These are the small extension points for new distributions, partner sources,
and position curricula.

Partner sessions are frozen by construction: only the ego actor-critic produces
PPO observations, log probabilities, values, advantages, and optimizer updates.
A checkpoint partner is loaded as an inference artifact and is never attached
to the ego optimizer.

## Partner-pool configuration

Use `weighted_pool` to sample one partner per episode. Weights must be positive
and are normalized; they do not need to sum to one.

```yaml
partner:
  sampler: weighted_pool
  position_sampler: balanced
  policies:
    - name: greedy_full_task
      weight: 3.0
      source: teacher_scripted
      policy:
        type: builtin
        name: greedy_full_task
        max_action_time_ms: 100

    - name: sticky_greedy
      weight: 2.0
      source: teacher_scripted_variant
      policy:
        type: builtin
        name: greedy_full_task
        sticky_action_prob: 0.20
        max_action_time_ms: 100

    - name: frozen_sp
      weight: 2.0
      source: frozen_checkpoint
      observation:
        type: featurized
        include_agent_index: false
      policy:
        type: python_class
        name: frozen_sp
        path: ../../policies/rl_policy.py
        class_name: StudentAgent
        config:
          checkpoint_path: ../../outputs/RUN/checkpoints/inference.pt
          device: auto
          deterministic: false
        max_action_time_ms: 100
```

The partner-level `observation` section is useful when the frozen artifact was
trained with a different `include_agent_index` setting from the ego. Paths in a
training YAML are resolved relative to that YAML.

The initial complete five-partner pool is
[train_partner_pool.yaml](../configs/stage_c/train_partner_pool.yaml). It includes
`greedy_full_task`, sticky greedy, sticky plus random greedy, `random_motion`, and
one exported frozen self-play checkpoint. The example probabilities (`0.20`
sticky and `0.10` random) are configuration values, not core defaults. Replace
them if the teacher publishes different scenario perturbation probabilities.

Train against the pool with:

```bash
.venv/bin/python scripts/train.py \
  --config configs/stage_c/train_partner_pool.yaml
```

`checkpoint.resume_from` is a resumable training checkpoint used to initialize
the ego. Set `checkpoint.load_optimizer_state: false` for fine-tuning with the
new configuration's learning rate; the backward-compatible default is to resume
the old optimizer. `training.total_steps` is the absolute target, including the
checkpoint's recorded environment steps. Each partner checkpoint under
`policy.config.checkpoint_path` must instead be a deployable inference artifact.

## Exact-partner fine-tuning

Use the same pool-entry format with `sampler: exact` and name one entry:

```yaml
partner:
  sampler: exact
  exact_partner: greedy_full_task
  position_sampler: balanced
  policies:
    - name: greedy_full_task
      source: teacher_scripted
      policy:
        type: builtin
        name: greedy_full_task
```

The short Scenario 1 configuration performs four 8-by-128 rollout updates after
the selected 900,096-step Stage A checkpoint:

```bash
.venv/bin/python scripts/train.py \
  --config configs/stage_c/scenario1_exact_short.yaml \
  --evaluate-checkpoints
```

## Adding a partner

For an existing teacher builtin or wrapper variant, add one named entry with a
normal `policy` mapping and a positive weight. Sticky and random actions are
configured with `sticky_action_prob` and `random_action_prob` and use seeded
wrapper RNGs.

For a self-play or historical checkpoint, first export a Stage A inference
artifact. Add a `python_class` entry pointing to `policies/rl_policy.py` and the
artifact, plus its observation settings when they differ from the ego. Missing,
incompatible, or training-profile partner checkpoints fail through the existing
checkpoint validation boundary.

For a new implementation, make it loadable by the existing `build_policy`
contract (builtin or `python_class`) and then add only the pool entry. A new
sampler implementation is needed only when weighted or exact episode sampling
does not express the desired distribution.

## Per-partner and per-position evaluation

The standalone initial suite is
[evaluate_partner_pool.yaml](../configs/stage_c/evaluate_partner_pool.yaml):

```bash
.venv/bin/python -m src.evaluate \
  --config configs/stage_c/evaluate_partner_pool.yaml
```

Each report case identifies its layout, partner, and ego inference mode. Its
`position_metrics` separates physical ego positions 0 and 1, including soup
counts, official scores, minimum and mean score, and zero-soup rate. The suite
runs deterministic and stochastic ego inference with fixed episode seeds.
Stochastic scripted/wrapped partners remain reproducible because every rollout
is rebuilt with its episode seed.

## Asymmetric Advantages 300k pair

The parallel Stage C pair resumes the preserved 900096-step seed-67 checkpoint
with fresh optimizers and random streams:

- `configs/stage_c/asymmetric_exact_seed67_300k.yaml` trains only against the
  disclosed greedy partner;
- `configs/stage_c/asymmetric_weighted_pool_seed68_300k.yaml` trains against the
  declared five-member weighted pool.

Both add exactly 300032 environment steps, save every 50176 steps, and evaluate
all checkpoints against the exact partner and pool in both ego positions and
inference modes. They use CPU-only Kaggle packages because the measured PPO
workload is environment/CPU-bound. The operational handoff is
[`workstreams/aa_stage_c_kaggle_prompt.md`](workstreams/aa_stage_c_kaggle_prompt.md).
