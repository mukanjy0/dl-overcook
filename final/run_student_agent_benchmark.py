"""Benchmark this bundled StudentAgent with the teacher rollout loop.

The installed old-dynamics environment records deliveries in ``env.game_stats``
rather than per-transition rewards, so this runner reads that canonical ledger
when calculating the documented official score.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path


FINAL_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(FINAL_ROOT))

import yaml
from overcooked_ai_py.agents.agent import AgentPair

from src.competition_evaluation import (
    _submission_policy_config,
    _teammate_policy_config,
    _wrapper_timeout_count,
    validate_competition_config,
)
from src.environment import build_env
from src.observations import ObservationBuilder
from src.policy_loader import build_policy
from src.runner import set_global_seed
from src.scoring import compute_attempt_score


def _run_attempt(
    env,
    obs_builder: ObservationBuilder,
    submission: dict,
    scenario: dict,
    seed: int,
    student_index: int,
) -> dict:
    set_global_seed(seed)
    policies = [
        _submission_policy_config(submission, scenario["max_action_time_ms"]),
        _teammate_policy_config(scenario["teammate"], scenario["max_action_time_ms"]),
    ]
    if student_index == 1:
        policies.reverse()
    agents = [
        build_policy(policies[0], env, obs_builder, seed=seed + 1000),
        build_policy(policies[1], env, obs_builder, seed=seed + 2000),
    ]
    student = agents[student_index]
    pair = AgentPair(*agents)
    env.reset(regen_mdp=False)
    pair.reset()
    pair.set_mdp(env.mdp)
    done = False
    while not done:
        actions_and_infos = pair.joint_action(env.state)
        actions, infos = zip(*actions_and_infos)
        _, _, done, _ = env.step(actions, infos)

    delivery_steps = sorted(
        int(step)
        for player_steps in env.game_stats.get("soup_delivery", ())
        for step in player_steps
    )
    first = delivery_steps[0] + 1 if delivery_steps else None
    last = delivery_steps[-1] + 1 if delivery_steps else None
    timeouts = _wrapper_timeout_count(student)
    return {
        "scenario_id": scenario["id"],
        "seed": seed,
        "student_index": student_index,
        "soups": len(delivery_steps),
        "first_soup_timestep": first,
        "last_soup_timestep": last,
        "student_timeouts": timeouts,
        "score": compute_attempt_score(
            soups=len(delivery_steps),
            horizon=scenario["horizon"],
            first_soup_timestep=first,
            last_soup_timestep=last,
            student_timeouts=timeouts,
        ),
    }


def main() -> None:
    config_path = FINAL_ROOT / "configs/competition.yaml"
    config = validate_competition_config(
        yaml.safe_load(config_path.read_text(encoding="utf-8")),
        base_dir=FINAL_ROOT,
    )
    submission = config["submissions"][0]
    rows: list[dict] = []
    for scenario in (item for item in config["scenarios"] if item.get("enabled", True)):
        env = build_env(
            {
                "layout_name": scenario.get("layout_name"),
                "layout_file": scenario.get("layout_file"),
                "horizon": scenario["horizon"],
                "old_dynamics": bool(scenario.get("old_dynamics", True)),
                "mdp_overrides": scenario.get("mdp_overrides", {}) or {},
            }
        )
        observation = scenario.get("observation", config.get("observation", {}))
        obs_builder = ObservationBuilder(env, observation)
        positions = (0, 1) if scenario.get("swap_roles", False) else (0,)
        for seed in scenario["seeds"]:
            for student_index in positions:
                rows.append(
                    _run_attempt(
                        env,
                        obs_builder,
                        submission,
                        scenario,
                        int(seed),
                        student_index,
                    )
                )

    output_dir = FINAL_ROOT / "results/competition_agent_audit"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "per_attempt.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for scenario_id in sorted({row["scenario_id"] for row in rows}):
        scenario_rows = [row for row in rows if row["scenario_id"] == scenario_id]
        mean_score = sum(row["score"] for row in scenario_rows) / len(scenario_rows)
        mean_soups = sum(row["soups"] for row in scenario_rows) / len(scenario_rows)
        print(f"scenario={scenario_id} mean_score={mean_score:.2f} mean_soups={mean_soups:.3f}")


if __name__ == "__main__":
    main()
