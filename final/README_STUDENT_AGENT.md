# Student-agent integration and benchmark

`policies/template.py` is the only teacher-facing entry point. It implements
`StudentAgent.act(obs) -> int` and routes the raw teacher state by layout:

| Layout | Selected policy |
| --- | --- |
| `asymmetric_advantages` | Bundled validated Stage D index-0 inference policy |
| `coordination_ring` | Bundled guided PPO specialist from `scenario2_guided.py` |
| `counter_circuit` | Bundled validated Stage D inference policy |
| `scenario_4` | Existing fixed-pot-B scripted specialist |
| Other layouts | Existing generic greedy fallback |

The agent requires the teacher state observation:

```yaml
observation:
  type: state
  include_agent_index: true
```

This setting is present in `configs/competition.yaml`. It provides the MDP and
physical player index needed for routing and for the guided Scenario 2 encoder.

## Run the benchmark

From the `final/` directory, use any Python environment containing the teacher
dependencies (`torch`, `numpy`, `pyyaml`, and `overcooked_ai_py`):

```bash
cd final
PYTHONPATH=. python -B run_student_agent_benchmark.py
```

The command executes the teacher policy-loading, wrapper, environment, partner,
seed, horizon, and role-swap paths defined in `configs/competition.yaml`. It
writes one row per rollout to `results/competition_agent_audit/per_attempt.csv`
and prints a mean score and soup count for each enabled scenario.

Scenario 2 deliberately samples from its trained action distribution. The
teacher-provided per-rollout seed is forwarded to Torch before that sampling,
so a benchmark is reproducible for a fixed scenario/seed configuration.

The benchmark reads `env.game_stats['soup_delivery']`, which is the canonical
delivery ledger for the installed old-dynamics environment. This avoids treating
valid deliveries as zero when a transition omits a sparse reward/event field.

## Self-contained bundle

All project files needed at runtime are inside `final/`: the teacher harness,
layouts, policy code, Scenario 2 inference helpers, and all three required
model artifacts. The agent only resolves paths relative to its own files; it
does not read the parent repository, external checkpoints, or local output
directories. A Python environment with the listed third-party dependencies is
still required, as it is for the teacher harness itself.

## What changed in `final/`

Only student-agent integration assets were added or changed:

- `policies/template.py`: Stage D router and teacher-compatible `StudentAgent`.
- `policies/scenario2_guided.py`, `scenario2_agent/`, and
  `policies/scenario2_guided_model.pt`: the validated higher-scoring Scenario 2
  guided specialist and its inference-only dependencies.
- `policies/stage_d_aa_position0.pt` and `policies/stage_d_counter_circuit.pt`:
  the existing validated Stage D inference artifacts, bundled so the router has
  no dependency on a parent checkout.
- `src/policy_wrappers.py`: forwards normal `set_mdp` and `set_agent_index`
  lifecycle calls to `StudentAgent`; this pre-initializes inference state outside
  the 100 ms action limit.
- `configs/competition.yaml`: requests the raw state observation required by the
  student agent.

The teacher environment, layouts, partners, scoring implementation, and
competition evaluator were not changed.

## Bundled model hashes

| Asset | SHA-256 |
| --- | --- |
| `policies/stage_d_aa_position0.pt` | `f6c4289d14623895249615b0bc48e11217bef62c53ce0977c050e076eb3187a2` |
| `policies/stage_d_counter_circuit.pt` | `176623829c3f71c17e73865170e87a8b88db6ae55471590d759cbca917a3bf41` |
| `policies/scenario2_guided_model.pt` | `7f7a20cc7a9e6d5d7821597bcac567a3372ec1588f17d85e3a0c983213c9711f` |
