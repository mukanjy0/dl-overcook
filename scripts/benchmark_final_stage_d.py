"""Benchmark the final Stage D router through its public policy-loading path.

This is intentionally an evaluation-only tool: every ego policy is built as a
``stage_d_router`` and every rollout uses the canonical ``run_episode`` loop.
It never imports a specialist directly or changes any training artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from src.environment import build_env
from src.episode import EpisodeStep, run_episode
from src.observations import ObservationBuilder
from src.policy_loader import build_two_policies
from src.seed_utils import set_global_seed
from src.deployment.stage_d_router import select_specialist


@dataclass(frozen=True)
class BenchmarkCase:
    """One teacher-style direct benchmark or clearly labelled generic proxy."""

    name: str
    group: str
    layout_name: str | None
    layout_file: str | None
    partner: dict[str, Any]
    seeds: tuple[int, ...]


def _policy(name: str, **settings: Any) -> dict[str, Any]:
    return {
        "type": "builtin",
        "name": name,
        "max_action_time_ms": 100,
        "invalid_action": "stay",
        "timeout_action": "stay",
        **settings,
    }


def _parse_seed_range(value: str) -> tuple[int, ...]:
    try:
        start_text, stop_text = value.split(":", 1)
        start, stop = int(start_text), int(stop_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seed range must be START:STOP") from exc
    if stop <= start:
        raise argparse.ArgumentTypeError("seed range must have STOP > START")
    return tuple(range(start, stop))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _router_policy(mapping: Path) -> dict[str, Any]:
    return {
        "type": "stage_d_router",
        "specialist_mapping": str(mapping),
        "max_action_time_ms": 100,
        "invalid_action": "stay",
        "timeout_action": "stay",
    }


def _summarize(rows: list[dict[str, Any]], startup_seconds: float) -> dict[str, Any]:
    def position_summary(position: int) -> dict[str, Any]:
        subset = [row for row in rows if row["ego_position"] == position]
        scores = [row["official_score"] for row in subset]
        soups = [row["soups"] for row in subset]
        return {
            "episodes": len(subset),
            "mean_score": mean(scores),
            "minimum_score": min(scores),
            "mean_soups": mean(soups),
            "zero_soup_rate": mean(soup == 0 for soup in soups),
        }

    scores = [row["official_score"] for row in rows]
    soups = [row["soups"] for row in rows]
    latencies = [row["max_action_latency_ms"] for row in rows]
    successful_first = [row["first_delivery_timestep"] for row in rows if row["first_delivery_timestep"] is not None]
    successful_last = [row["last_delivery_timestep"] for row in rows if row["last_delivery_timestep"] is not None]
    total_wall = sum(row["rollout_wall_seconds"] for row in rows)
    by_position = {str(position): position_summary(position) for position in (0, 1)}
    return {
        "episodes": len(rows),
        "mean_score": mean(scores),
        "minimum_score": min(scores),
        "minimum_position_mean_score": min(item["mean_score"] for item in by_position.values()),
        "mean_soups": mean(soups),
        "zero_soup_rate": mean(soup == 0 for soup in soups),
        "first_delivery_mean_successful": mean(successful_first) if successful_first else None,
        "last_delivery_mean_successful": mean(successful_last) if successful_last else None,
        "timeouts": sum(row["timeouts"] for row in rows),
        "invalid_actions": sum(row["invalid_actions"] for row in rows),
        "blocked_steps": sum(row["blocked_steps"] for row in rows),
        "startup_seconds": startup_seconds,
        "inference_wall_seconds": total_wall,
        "mean_rollout_wall_seconds": total_wall / len(rows),
        "max_action_latency_ms": max(latencies),
        "mean_max_action_latency_ms": mean(latencies),
        "by_position": by_position,
    }


def _run_case(case: BenchmarkCase, mapping: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    environment = {
        "layout_name": case.layout_name,
        "layout_file": case.layout_file,
        "horizon": 400,
        "old_dynamics": True,
    }
    startup_start = time.perf_counter()
    env = build_env(environment)
    obs_builder = ObservationBuilder(env, {"type": "featurized", "include_agent_index": True})
    startup_seconds = time.perf_counter() - startup_start
    selection_ids = {
        str(position): select_specialist(mapping, str(env.mdp.layout_name), position).specialist_id
        for position in (0, 1)
    }
    rows: list[dict[str, Any]] = []
    config = {
        "policies": {
            "agent_0": _router_policy(mapping),
            "agent_1": case.partner,
        }
    }
    for seed in case.seeds:
        for ego_position in (0, 1):
            set_global_seed(seed)
            policy_start = time.perf_counter()
            ego_agent, partner_agent = build_two_policies(config, env, obs_builder, seed=seed)
            policy_setup_seconds = time.perf_counter() - policy_start
            agents = (ego_agent, partner_agent) if ego_position == 0 else (partner_agent, ego_agent)
            action_latencies: list[float] = []
            blocked_steps = 0

            def on_step(step: EpisodeStep, _env: Any) -> bool:
                nonlocal blocked_steps
                info = step.joint_infos[ego_position]
                elapsed_ms = info.get("elapsed_ms")
                if elapsed_ms is not None:
                    action_latencies.append(float(elapsed_ms))
                blocked_steps += int(bool(info.get("scenario4_blocked", False)))
                return False

            rollout_start = time.perf_counter()
            result = run_episode(
                env=env,
                agents=agents,
                episode_id=len(rows),
                seed=seed,
                ego_player_index=ego_position,
                role_swap=ego_position == 1,
                on_step=on_step,
            )
            rollout_wall_seconds = time.perf_counter() - rollout_start
            deliveries = list(result.delivery_timesteps)
            rows.append(
                {
                    "case": case.name,
                    "group": case.group,
                    "layout": str(env.mdp.layout_name),
                    "seed": seed,
                    "ego_position": ego_position,
                    "specialist_id": selection_ids[str(ego_position)],
                    "official_score": result.official_score,
                    "soups": len(deliveries),
                    "first_delivery_timestep": deliveries[0] if deliveries else None,
                    "last_delivery_timestep": deliveries[-1] if deliveries else None,
                    "timeouts": result.timeout_count_total,
                    "invalid_actions": sum(result.invalid_action_replacements_by_agent),
                    "blocked_steps": blocked_steps,
                    "policy_setup_seconds": policy_setup_seconds,
                    "rollout_wall_seconds": rollout_wall_seconds,
                    "max_action_latency_ms": max(action_latencies, default=0.0),
                }
            )
    return _summarize(rows, startup_seconds), rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping", default="configs/stage_d/specialists.yaml")
    parser.add_argument("--output-dir", default="outputs/stage_d_finalization")
    parser.add_argument(
        "--scenario4-seeds",
        type=_parse_seed_range,
        default=tuple(range(30, 60)),
        help="fresh Scenario 4 random-motion validation range (default: 30:60)",
    )
    parser.add_argument("--skip-generic-proxy", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    mapping = Path(args.mapping).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    direct_cases = (
        BenchmarkCase("scenario_1", "direct", "asymmetric_advantages", None, _policy("greedy_full_task"), (67, 68, 69)),
        BenchmarkCase("scenario_2", "direct", "coordination_ring", None, _policy("greedy_full_task", sticky_action_prob=0.10), (0, 1, 2, 3, 4)),
        BenchmarkCase("scenario_3", "direct", "counter_circuit", None, _policy("greedy_full_task", sticky_action_prob=0.10, random_action_prob=0.10), tuple(range(20))),
        BenchmarkCase("scenario_4", "direct", None, str(project_root / "configs/layouts/scenario_4.layout"), _policy("random_motion"), args.scenario4_seeds),
    )
    proxy_cases = (
        BenchmarkCase("generic_custom_narrow_direct", "generic_proxy_held_out", None, str(project_root / "configs/layouts/custom_room.layout"), _policy("greedy_full_task"), (201, 202, 203, 204, 205)),
        BenchmarkCase("generic_custom_narrow_moving", "generic_proxy_held_out", None, str(project_root / "configs/layouts/custom_room.layout"), _policy("random_motion"), (201, 202, 203, 204, 205)),
        BenchmarkCase("generic_cramped_noisy", "generic_proxy_diagnostic", "cramped_room", None, _policy("greedy_full_task", sticky_action_prob=0.10, random_action_prob=0.10), (201, 202, 203, 204, 205)),
    )
    cases = direct_cases if args.skip_generic_proxy else direct_cases + proxy_cases
    summaries: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    registry: dict[str, Any] = {}
    for case in cases:
        summary, rows = _run_case(case, mapping)
        summaries[case.name] = summary
        all_rows.extend(rows)
        for row in rows:
            specialist_id = row["specialist_id"]
            if specialist_id in registry:
                continue
            selection = select_specialist(mapping, row["layout"], row["ego_position"])
            checkpoint = (selection.policy.get("config", {}) or {}).get("checkpoint_path")
            registry[specialist_id] = {
                "layout_route": selection.route_layout,
                "physical_position": row["ego_position"],
                "checkpoint_path": checkpoint,
                "expected_sha256": selection.artifact_sha256,
                "actual_sha256": _sha256(Path(checkpoint)) if checkpoint else None,
            }
    payload = {
        "status": "complete",
        "mapping": str(mapping),
        "scenario4_seed_range": [args.scenario4_seeds[0], args.scenario4_seeds[-1]],
        "summaries": summaries,
        "artifact_registry": registry,
        "rollouts": all_rows,
    }
    (output_dir / "benchmark.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_dir / "candidate_registry.json").write_text(json.dumps(registry, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", "summaries": summaries}, indent=2))


if __name__ == "__main__":
    main()
