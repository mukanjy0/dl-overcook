# Student-agent integration and benchmark

`policies/template.py` is the only teacher-facing entry point. It implements
`StudentAgent.act(obs) -> int` and routes the raw teacher state by layout:

| Layout | Selected policy |
| --- | --- |
| `asymmetric_advantages` | Bundled validated Stage D index-0 inference policy |
| `coordination_ring` | Bundled guided PPO specialist from `scenario2_guided.py` |
| `counter_circuit` | Deterministic mixed-recipe tomato specialist paired with the onion partner |
| `scenario_4` | Existing fixed-pot-B scripted specialist |
| Other layouts | Existing generic greedy fallback |

### Why Scenario 3 uses a heuristic

The disclosed `counter_circuit` orders all require both onion and tomato, while
the disclosed greedy partner fetches onions. The previous PPO specialist and
partner delivered single-ingredient or triple-onion soups; Overcooked records
those interactions in `game_stats`, but the teacher correctly awards zero
sparse reward because they do not match an active order.

`_CounterCircuitMixedRecipe` therefore takes the complementary tomato role. It
waits for the onion partner to seed a pot, then adds a tomato only when the
resulting ingredient multiset can still extend an active order. Navigation,
dish handling, plating, and delivery continue to use `GreedyFullTaskPolicy`.
This keeps the workaround deterministic and narrowly scoped to recipe choice.

On the four official Scenario 3 seeds, the unchanged teacher evaluator reports
11, 11, 6, and 5 valid soups (mean score `83,968.75`). Scenario 4 remains
unchanged at mean score `94,281.50` across both physical positions.

The agent requires the teacher state observation:

```yaml
observation:
  type: state
  include_agent_index: true
```

This setting is present in `configs/competition.yaml`. It provides the MDP and
physical player index needed for routing and for the guided Scenario 2 encoder.

## Run the benchmark

The teacher-facing submission block in `configs/competition.yaml` is:

```yaml
submissions:
- name: grupo_XX
  path: policies/template.py
  class_name: StudentAgent
```

From `final/`, run one scenario directly with the teacher command:

```bash
python -m src.evaluate_competition --config configs/competition.yaml --scenario 1
```

Run with rendering:

```bash
python -m src.evaluate_competition --config configs/competition.yaml --render
```

Use the Python interpreter supplied by the teacher environment. Locally, if
`python` is a pyenv shim without the required version, invoke the same command
with the repository's virtual-environment interpreter instead:

```bash
../.venv/bin/python -m src.evaluate_competition --config configs/competition.yaml --scenario 1
```

## Self-contained bundle

All project files needed at runtime are inside `final/`: the teacher harness,
layouts, policy code, Scenario 2 inference helpers, and all bundled
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
  preserved Stage D inference artifacts, bundled so the router has no dependency
  on a parent checkout. The counter-circuit artifact is retained for provenance;
  the active route is the recipe-safe scripted specialist in `template.py`.
- `src/policy_wrappers.py`: forwards normal `set_mdp` and `set_agent_index`
  lifecycle calls to `StudentAgent`; this pre-initializes inference state outside
  the 100 ms action limit.
- `configs/competition.yaml`: requests the raw state observation required by the
  student agent and declares the standard three-field submission block.

The teacher environment, layouts, partners, scoring implementation, and
competition evaluator were not changed.

## Bundled model hashes

| Asset | SHA-256 |
| --- | --- |
| `policies/stage_d_aa_position0.pt` | `f6c4289d14623895249615b0bc48e11217bef62c53ce0977c050e076eb3187a2` |
| `policies/stage_d_counter_circuit.pt` | `176623829c3f71c17e73865170e87a8b88db6ae55471590d759cbca917a3bf41` |
| `policies/scenario2_guided_model.pt` | `7f7a20cc7a9e6d5d7821597bcac567a3372ec1588f17d85e3a0c983213c9711f` |
