# AGENTS.md

## Purpose

Maintain this repository as a clean, modular extension of the teacher-provided Overcooked-AI codebase.

The main compatibility contract is that the submitted agent must remain loadable through the teacher's existing `build_policy` workflow.

Do not include project strategy, experiment planning, or competition-roadmap details in this file. Keep this document focused on stable engineering and execution rules.

## Core Engineering Principles

- Prefer simple, readable code over clever or overly abstract solutions.
- Avoid duplicated logic. Reuse shared components, configuration, interfaces, and helpers.
- Keep important control flow explicit and easy to trace.
- Keep functions and classes focused on one responsibility.
- Use clear names that communicate intent.
- Add type hints to important public functions, classes, and data structures.
- Add concise docstrings when behavior, assumptions, inputs, or outputs are not obvious.
- Avoid global mutable state unless required by the upstream framework.
- Prefer deterministic behavior where practical and expose random seeds explicitly.
- Do not introduce abstractions unless they reduce real duplication or isolate a likely point of change.

## Compatibility with the Teacher's Workflow

- Preserve compatibility with the teacher's existing runner and environment.
- The submitted agent must be loadable through `build_policy`.
- Inspect the current `build_policy` contract before changing integration code.
- Preserve its expected signature, return type, and runtime behavior.
- Keep the teacher-facing integration layer thin.
- Prefer adapters or wrappers around project internals instead of embedding all logic directly inside `build_policy`.
- Do not require custom startup steps outside the teacher's workflow.
- Do not silently change:
  - observation formats;
  - action formats;
  - reward semantics;
  - episode horizons;
  - layout definitions;
  - partner behavior;
  - scoring behavior.
- Ensure inference does not depend on:
  - training-only packages;
  - Kaggle-specific paths;
  - local-only files;
  - external services;
  - interactive setup.
- Add or maintain a smoke test that imports and instantiates the agent through `build_policy`.
- When modifying teacher-provided code:
  - keep changes minimal;
  - preserve existing behavior by default;
  - document the reason;
  - add regression coverage when feasible.

## Repository Structure

Follow the existing repository structure when it is already clear and consistent.

When adding new modules, use this separation as guidance:

- package or `src/` directory:
  - agents;
  - policies;
  - models;
  - training logic;
  - partner interfaces;
  - environment adapters;
  - evaluation logic.
- `scripts/`:
  - training entry points;
  - evaluation entry points;
  - export and data-processing commands;
  - thin orchestration only.
- `configs/`:
  - experiment and runtime configuration.
- `tests/`:
  - unit tests;
  - integration tests;
  - compatibility and regression tests.
- `docs/`:
  - architecture summaries;
  - usage guides;
  - extension points;
  - meaningful design decisions.
- generated-output directories such as `runs/`, `outputs/`, or `checkpoints/`:
  - generated artifacts only.

Do not create a new top-level directory when an existing one already serves the same purpose.

## Core Logic and Utilities

- Keep domain logic close to the module that owns it.
- Do not use a generic `utils.py` as a dumping ground.
- Put code in utility modules only when it is broadly reusable and not tied to one algorithm or workflow.
- Prefer focused utility modules, for example:
  - `checkpointing.py`;
  - `serialization.py`;
  - `seed_utils.py`;
  - `metrics.py`;
  - `logging_utils.py`.
- Keep algorithm-specific helpers inside the relevant algorithm package.
- Separate:
  - core algorithmic behavior;
  - configuration parsing;
  - CLI handling;
  - filesystem operations;
  - logging;
  - plotting;
  - experiment orchestration.
- Scripts should call reusable package functions rather than contain substantial implementation logic.

## Modularity and Extension Points

Design likely-to-change components behind small, stable interfaces.

Typical extension points may include:

- ego agent or policy;
- partner policy;
- partner sampler or partner distribution;
- state buffer or augmentation source;
- training curriculum;
- evaluator;
- score calculator;
- checkpoint loader;
- layout-specific configuration.

Prefer composition over large inheritance hierarchies.

A new partner, evaluator, augmentation method, or training component should usually be addable through one focused implementation and configuration change, not by editing many unrelated conditionals.

Avoid hard-coding:

- layout names;
- partner names;
- filesystem paths;
- random seeds;
- hyperparameters;
- checkpoint locations;
- device choices.

## Python Environment and Dependencies

- Use the repository-local `.venv` for all Python work.
- Prefer explicit commands through the local environment, such as:
  - `.venv/bin/python`;
  - `.venv/bin/pytest`;
  - `.venv/bin/<tool>`.
- Do not use the system Python for project commands.
- Do not install packages globally.
- If `.venv` is missing, create it using the repository's existing dependency workflow.
- Preserve the project's existing dependency manager when one is already configured.
- If no dependency manager is established, prefer `uv`.
- With `uv`, prefer:
  - `uv sync`;
  - `uv run python ...`;
  - `uv run pytest`.
- When adding or changing dependencies:
  - update the appropriate dependency file;
  - update the lockfile when applicable;
  - avoid adding unnecessary packages;
  - verify that inference dependencies remain minimal.
- Do not create a second virtual environment unless explicitly required.

## Configuration

- Centralize experiment and runtime parameters in configuration files or structured configuration objects.
- Validate required fields early and fail with useful error messages.
- Avoid multiple sources of truth for the same parameter.
- Keep defaults conservative and compatible with the teacher's baseline.
- Separate:
  - environment configuration;
  - algorithm hyperparameters;
  - partner configuration;
  - evaluation configuration;
  - output paths.
- Record the effective configuration for meaningful training and evaluation runs.

## Training and Evaluation

- Keep training and evaluation code separate.
- Evaluation must not update:
  - model parameters;
  - optimizer state;
  - running statistics;
  - partner state;
  unless explicitly intended.
- Keep one canonical implementation of the official competition score.
- Do not duplicate score formulas across scripts.
- Make stochastic runs reproducible through explicit seeds.
- Record enough metadata to reproduce important results:
  - checkpoint;
  - layout;
  - partner;
  - seed;
  - horizon;
  - effective configuration;
  - code version when available.
- Support evaluation from both player positions when the environment permits it.
- Fail clearly on incompatible checkpoints, missing assets, or configuration mismatches.

## Kaggle and GPU Execution

- A Kaggle execution skill is available under `.codex/`.
- Inspect and follow the relevant skill documentation before using Kaggle.
- Use Kaggle automatically when GPU acceleration is materially beneficial, including substantial training runs or expensive batched evaluation.
- Run local smoke tests before launching remote workloads.
- Do not use Kaggle for trivial tests or lightweight commands.
- Keep Kaggle-specific orchestration outside core training and evaluation logic.
- Keep commands and configuration portable so the same workflow can run locally or remotely with minimal changes.
- Codex may update the Kaggle skill when necessary, but changes must remain reusable and documented.
- Preserve important remote outputs:
  - checkpoints;
  - metrics;
  - logs;
  - effective configs;
  - summaries.
- Download important artifacts back into the expected local output structure.
- Do not rely on temporary Kaggle storage as the only copy of an important result.
- Warn the user when Kaggle compute, memory, runtime, or storage is likely insufficient.
- When more compute is needed, clearly state whether the workload is better suited to:
  - RunPod;
  - an HPC cluster;
  - another GPU environment.

## Logging and Generated Outputs

- Use structured, readable logging.
- Avoid noisy per-step logs by default.
- Log major lifecycle events, including:
  - configuration loaded;
  - training started or completed;
  - checkpoint saved or loaded;
  - evaluation summary;
  - important warnings;
  - compatibility fallbacks.
- Keep generated files under dedicated output directories.
- Do not write large artifacts into the repository root.
- Use stable run-directory or filename conventions with enough context to identify the experiment.
- Do not commit:
  - large checkpoints;
  - raw datasets;
  - cache directories;
  - temporary videos;
  - local environment files;
  - profiling outputs;
  unless explicitly required.

## Testing and Validation

For meaningful changes:

- run relevant existing tests;
- add focused tests for new behavior;
- verify imports and CLI entry points;
- verify `build_policy` compatibility;
- run a minimal local smoke test;
- verify checkpoint save/load behavior when affected;
- verify that evaluation remains deterministic under fixed seeds when expected.

Prefer small, targeted tests over slow end-to-end tests when they provide equivalent confidence.

Do not claim a change works without running the relevant validation when the environment permits it.

## Documentation

Document meaningful capabilities and structural changes, not every small edit.

Update `README.md` for:

- initial setup;
- common commands;
- entry-level usage;
- the primary training or evaluation workflow;
- important compatibility requirements.

Use `docs/` for:

- architecture;
- module responsibilities;
- extension points;
- checkpoint formats;
- training and evaluation workflows;
- remote execution;
- important design decisions;
- non-obvious modification guidance.

Documentation should explain:

- what was added or changed;
- why it exists at a high level;
- how to use it;
- where to modify or extend it.

Avoid microupdates, implementation diaries, and overly specific change logs that do not help a developer understand or use the system.

Keep code and documentation consistent. When a meaningful command, interface, path, or workflow changes, update the relevant documentation in the same task.

## `.gitignore` Maintenance

After meaningful changes, review `.gitignore`.

Update it when new generated or local-only artifacts are introduced, including:

- `.venv`;
- checkpoints;
- runs and outputs;
- logs;
- datasets;
- videos;
- notebook checkpoints;
- Kaggle downloads;
- caches;
- profiling outputs;
- temporary configuration files;
- local secrets.

Do not add overly broad ignore patterns that could hide source code, documentation, configuration, tests, or intentionally committed fixtures.

Never commit credentials, API tokens, private keys, or local machine-specific secrets.

## Change Discipline

- Keep changes scoped to the requested task.
- Do not perform unrelated refactors unless they are necessary to complete the task safely.
- Preserve backward compatibility where practical.
- Remove dead code only when its role is understood and tests or references confirm it is unused.
- Avoid temporary compatibility hacks without documenting them.
- When a workaround is necessary, isolate it and explain the constraint in code or documentation.
- Before finishing, review the diff for:
  - duplication;
  - accidental generated files;
  - stale documentation;
  - broken imports;
  - unnecessary dependency changes;
  - compatibility regressions.
