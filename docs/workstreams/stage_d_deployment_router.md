# Stage D deployment router

Stage D adds no training and no specialist behavior.  It dispatches the existing,
validated CPU inference specialists through the normal `build_policy` path.

## Deployment map

| Layout | Physical ego index | Specialist | Artifact SHA-256 |
| --- | ---: | --- | --- |
| `asymmetric_advantages` | 0 | `aa_rl_position0_900096` | `f6c4289d14623895249615b0bc48e11217bef62c53ce0977c050e076eb3187a2` |
| `asymmetric_advantages` | 1 | `aa_greedy_position1` (`greedy_full_task`) | built-in |
| `coordination_ring` | 0 or 1 | `cr_p010_s11_step1050624` | `806f0fc17587e832aeb939378dd9ffdce0ea413a15cae6a24eb58698ac4bb42b` |
| `counter_circuit` | 0 or 1 | `cc_exact_long_seed3_step1902592` | `176623829c3f71c17e73865170e87a8b88db6ae55471590d759cbca917a3bf41` |
| unknown layout | 0 or 1 | `generic_greedy_fallback` (`greedy_full_task`) | built-in |

Artifact paths are relative to `configs/stage_d/specialists.yaml`:

- AA: `../../outputs/stage_a_asymmetric_seed67/selected/inference_step_000900096.pt`
- CR: `../../outputs/stage_d_specialists/coordination_ring/inference.pt`
- CC: `../../outputs/counter_circuit_exact_long_seed3_1m/checkpoint_evaluation/selected/inference.pt`

The router verifies a configured checkpoint exists and matches its SHA-256 before
first load.  Checkpoints are lazily constructed, cached inside the router, reset
between episodes, and continue to use the repository's safe-action wrapper.

## Implementation

- `configs/stage_d/specialists.yaml`: central route and artifact configuration.
- `src/deployment/stage_d_router.py`: layout/index dispatch, integrity checks,
  lazy cache, and delegation.
- `src/policy_loader.py`: `stage_d_router` policy type, retaining the existing
  teacher-compatible `build_policy` entry point.
- `src/experiment_config.py`: config-relative mapping path resolution.
- `scripts/smoke_stage_d_router.py`: one short rollout for each of six mapped
  layout/index routes.
- `tests/test_stage_d_router.py`: routing, fallback, reset/cache,
  missing/corrupt artifact, and clean-process smoke coverage.

## Validation

- `.venv/bin/pytest -q` — `83 passed, 1 skipped` (the existing CUDA-only test).
- `scripts/smoke_stage_d_router.py --horizon 20` — six CPU rollouts completed,
  all with zero timeouts and zero invalid-action replacements.
- `python -m src.evaluate --config configs/stage_d/smoke_deployment_router.yaml`
  — teacher-facing config resolution and both AA physical positions completed.

Run the final complete routing smoke with:

```bash
.venv/bin/python scripts/smoke_stage_d_router.py --output-dir outputs/stage_d_smoke_final
```

## Remaining assumptions

- The teacher environment exposes the standard layout identity via
  `env.mdp.layout_name`; unknown identities intentionally receive the explicit
  greedy fallback.
- Generated selected checkpoints are deployment inputs and must accompany the
  source tree; the mapping rejects a missing or changed checkpoint rather than
  silently substituting another policy.
- The CR selected inference artifact is retained at the documented generated
  output path; no checkpoint is modified or committed by Stage D.
