"""Evaluate lightweight Scenario 4 scripted candidates over stochastic seeds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

import yaml

from src.runner import run_from_config


def _policy(variant: dict, project_root: Path) -> dict:
    if variant.get("builtin"):
        return {
            "type": "builtin",
            "name": variant["builtin"],
            "max_action_time_ms": 100,
            "invalid_action": "stay",
            "timeout_action": "stay",
        }
    return {
        "type": "python_class",
        "path": str(project_root / "policies/scenario4_policy.py"),
        "class_name": "Scenario4PlannerPolicy",
        "config": variant,
        "max_action_time_ms": 100,
        "invalid_action": "stay",
        "timeout_action": "stay",
    }


def _summarize(name: str, records: list[dict]) -> dict:
    scores = [int(r["official_score"]) for r in records]
    soups = [int(r["sparse_return"] // 20) for r in records]
    deliveries = [r["delivery_timesteps"] for r in records]
    return {
        "variant": name,
        "episodes": len(records),
        "minimum_score": min(scores),
        "mean_score": mean(scores),
        "zero_soup_rate": mean(value == 0 for value in soups),
        "mean_soups": mean(soups),
        "first_delivery_mean": mean((items[0] if items else 400) for items in deliveries),
        "last_delivery_mean": mean((items[-1] if items else 400) for items in deliveries),
        "timeouts": sum(int(r["timeout_count_total"]) for r in records),
        "invalid_actions": sum(sum(r["invalid_action_replacements_by_agent"]) for r in records),
        "by_position": {
            str(position): {
                "minimum_score": min(int(r["official_score"]) for r in records if r["ego_player_index"] == position),
                "mean_score": mean(int(r["official_score"]) for r in records if r["ego_player_index"] == position),
            }
            for position in (0, 1)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="outputs/scenario4_sweep")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    project_root = Path(__file__).resolve().parents[1]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    seeds = list(range(int(cfg.get("seed_start", 0)), int(cfg.get("seed_start", 0)) + int(cfg.get("num_seeds", 30))))
    all_rows: list[dict] = []
    summaries = []
    for variant in cfg["variants"]:
        name = variant["name"]
        result = run_from_config(
            {
                "seed": seeds[0],
                "environment": {"layout_name": None, "layout_file": str(project_root / "configs/layouts/scenario_4.layout"), "horizon": 400, "old_dynamics": True},
                "observation": {"type": "featurized", "include_agent_index": True},
                "policies": {
                    "agent_0": _policy(variant, project_root),
                    "agent_1": {"type": "builtin", "name": "random_motion", "max_action_time_ms": 100, "invalid_action": "stay", "timeout_action": "stay"},
                },
                "execution": {"num_episodes": len(seeds), "episode_seeds": seeds, "ego_player_positions": [0, 1]},
                "rendering": {"mode": "none"},
                "logging": {"output_dir": str(output / name), "save_step_log": False, "save_episode_summary": True},
                "data_collection": {"enabled": False},
            }
        )
        rows = result["episode_results"]
        for row in rows:
            row["variant"] = name
        all_rows.extend(rows)
        summaries.append(_summarize(name, rows))
    summaries.sort(key=lambda item: (-item["minimum_score"], item["zero_soup_rate"], -item["mean_score"]))
    with (output / "variant_summary.json").open("w", encoding="utf-8") as stream:
        json.dump({"status": "complete", "variants": summaries}, stream, indent=2)
    with (output / "episodes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({key for row in all_rows for key in row}))
        writer.writeheader(); writer.writerows(all_rows)
    print(json.dumps({"status": "complete", "winner": summaries[0], "variants": summaries}, indent=2))


if __name__ == "__main__":
    main()
