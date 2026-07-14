# Student-agent integration and benchmark

`policies/template.py` is the only teacher-facing entry point. It implements
`StudentAgent.act(obs) -> int` and routes the raw teacher state by layout:

| Layout | Selected policy |
| --- | --- |
| `asymmetric_advantages` | Distilled reachability-aware full-task neural specialist |
| `coordination_ring` | Bundled guided PPO specialist from `scenario2_guided.py` |
| `counter_circuit` | Distilled mixed-recipe neural specialist paired with the onion partner |
| `scenario_4` | Distilled fixed-pot-B neural specialist for both indexes |
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

### Scenario 4 two-index distillation

The validated fixed-pot-B policy was distilled into the same compact
PPO-compatible actor-critic while alternating physical player indexes and
rolling out against the disclosed noisy `random_motion` partner. Selection
covered all eight official seed/index attempts and improved the minimum from 8
to 9 soups while preserving the 9.375-soup mean. The scripted implementation is
retained as the reproducible teacher, but the active route is neural.

### Scenario 2 model cross-layout comparison

The bundled Scenario 2 guided PPO model was also forced through every scenario
before routing was finalized. It remained selected only for `coordination_ring`:
it averaged 4--6 soups on Scenario 1 versus the selected model's 14, produced
zero valid mixed soups on Scenario 3, and produced zero soups in every Scenario
4 index-1 attempt. Its deterministic mode produced zero soups on all scenarios,
so Scenario 2 keeps the intended stochastic inference mode.

| Scenario | Guided model forced mean soups | Selected route mean soups |
| --- | ---: | ---: |
| 1 | 5.000 | 14.000 |
| 2 | 4.000 | Same guided model in its native layout |
| 3 | 0.000 | 7.500 |
| 4 | 0.875 | 9.375 |

### Why the Scenario 3 teacher uses a heuristic

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
seed. The distilled Scenario 4 route scores `94,313.38` across both physical
positions.

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
- `policies/scenario4_distilled.pt`: position-aware neural distillation of the
  fixed-pot-B Scenario 4 teacher.
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
| `policies/asymmetric_advantages_distilled.pt` | `09483cc3ff720de841f11145a87a786ddaef04910434176a8074f50caa05ea16` |
| `policies/stage_d_counter_circuit.pt` | `176623829c3f71c17e73865170e87a8b88db6ae55471590d759cbca917a3bf41` |
| `policies/counter_circuit_distilled.pt` | `68887668dce05d7589458241e225f250cfd2037ffd9bc78c7165ca3805724ce4` |
| `policies/scenario4_distilled.pt` | `44c08353f16a776e3c746f38e4c7cea60316d1f3bd6b2266fe7666377f52dfa5` |
| `policies/scenario2_guided_model.pt` | `7f7a20cc7a9e6d5d7821597bcac567a3372ec1588f17d85e3a0c983213c9711f` |
