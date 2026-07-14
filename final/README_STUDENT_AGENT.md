# Student-agent integration and benchmark

`policies/template.py` is the only teacher-facing entry point. It implements
`StudentAgent.act(obs) -> int` and routes the raw teacher state by layout:

| Layout | Selected policy |
| --- | --- |
| `asymmetric_advantages` | Distilled reachability-aware full-task neural specialist |
| `coordination_ring` | Bundled guided PPO specialist from `scenario2_guided.py` |
| `counter_circuit` | Distilled mixed-recipe neural specialist paired with the onion partner |
| `scenario_4` | Existing fixed-pot-B scripted specialist |
| Other layouts | Existing generic greedy fallback |

### Scenario 1 reachability repair

The two halves of `asymmetric_advantages` contain duplicate dispensers and
serving windows separated by an impassable wall. The original PPO and generic
greedy behavior selected targets using Manhattan distance, so after reaching a
pot they could select a slightly closer feature on the unreachable opposite
side and wait indefinitely.

`_AsymmetricReachableFullTask` keeps the existing full-task logic and BFS
navigation, but filters candidate features to those with a reachable adjacent
interaction tile before comparing path lengths. Its behavior was distilled
into the existing PPO-compatible actor-critic using learner-visited states.
The unchanged teacher command produces 14 valid three-onion soups on every
official seed, with score `140,420` per seed and no invalid recipe deliveries.

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
This keeps the teacher deterministic and narrowly scoped to recipe choice.

The deployed Scenario 3 network was repaired with dataset aggregation rather
than another long PPO run. The existing PPO-compatible actor-critic was rolled
out against the exact disclosed noisy partner, every learner-visited state was
labelled by the recipe-safe teacher, and checkpoints were selected using only
positive sparse-reward soups. Invalid deliveries in Overcooked's event ledger
were deliberately excluded. This retains a small deterministic neural policy
at inference while correcting the previous interaction-only failure mode.

The unchanged teacher command produced 8, 9, 6, and 7 positive-reward soups on
the four official Scenario 3 seeds, with mean score `76,296.75` and no zero-soup
seed. Scenario 4 remains unchanged at mean score `94,281.50` across both
physical positions.

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

Run every enabled scenario with:

```bash
python -m src.evaluate_competition --config configs/competition.yaml --all-scenarios
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
  on a parent checkout. The old counter-circuit artifact is retained for
  provenance.
- `policies/counter_circuit_distilled.pt`: PPO-compatible Scenario 3 network
  repaired from recipe-safe learner-visited demonstrations.
- `policies/asymmetric_advantages_distilled.pt`: PPO-compatible Scenario 1
  network repaired from reachability-aware learner-visited demonstrations.
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
| `policies/asymmetric_advantages_distilled.pt` | `55f5738f1b64e2fd233904ce0fad25d5e7ece2e802fcdebeafbb69c040cf1500` |
| `policies/stage_d_counter_circuit.pt` | `176623829c3f71c17e73865170e87a8b88db6ae55471590d759cbca917a3bf41` |
| `policies/counter_circuit_distilled.pt` | `68887668dce05d7589458241e225f250cfd2037ffd9bc78c7165ca3805724ce4` |
| `policies/scenario2_guided_model.pt` | `7f7a20cc7a9e6d5d7821597bcac567a3372ec1588f17d85e3a0c983213c9711f` |
