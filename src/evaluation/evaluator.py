"""Evaluation suites built on the teacher-compatible runner."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from src.evaluation.types import EvaluationCaseResult, EvaluationReport
from src.runner import run_from_config


class Evaluator:
    """Stable evaluation interface backed by the teacher-compatible runner."""

    def evaluate(self, config: dict[str, Any]) -> dict[str, Any]:
        return evaluate_from_config(config)


def _layout_label(layout: str | dict[str, Any]) -> str:
    if isinstance(layout, str):
        return layout
    if layout.get("layout_name"):
        return str(layout["layout_name"])
    if layout.get("layout_file"):
        return Path(str(layout["layout_file"])).stem
    raise ValueError("Each evaluation layout needs layout_name or layout_file")


def _environment_for_layout(
    base_environment: dict[str, Any],
    layout: str | dict[str, Any],
) -> dict[str, Any]:
    environment = deepcopy(base_environment)
    environment.pop("layout_name", None)
    environment.pop("layout_file", None)
    if isinstance(layout, str):
        environment["layout_name"] = layout
    else:
        environment.update(deepcopy(layout))
    return environment


def _partner_entry(entry: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if not isinstance(entry, dict):
        raise ValueError("Each evaluation partner must be a mapping")
    policy = entry.get("policy", entry)
    if not isinstance(policy, dict):
        raise ValueError("evaluation partner.policy must be a mapping")
    name = str(entry.get("name", policy.get("name", "partner")))
    return name, deepcopy(policy)


def evaluate_from_config(config: dict[str, Any]) -> dict[str, Any]:
    """Run legacy single evaluations or a Stage A layout/partner suite."""
    evaluation = config.get("evaluation", {}) or {}
    layouts = evaluation.get("layouts")
    partners = evaluation.get("partners")
    if not layouts or not partners:
        return run_from_config(config)

    base_policies = config.get("policies", {}) or {}
    if "agent_0" not in base_policies:
        raise ValueError("Suite evaluation requires policies.agent_0 as the ego policy")

    seeds = [int(seed) for seed in evaluation.get("seeds", [0])]
    positions = [int(position) for position in evaluation.get("player_positions", [0, 1])]
    output_root = Path(
        str((config.get("logging", {}) or {}).get("output_dir", "outputs/evaluation"))
    )
    cases: list[EvaluationCaseResult] = []
    all_scores: list[float] = []
    all_returns: list[float] = []

    for layout in layouts:
        layout_label = _layout_label(layout)
        for partner_entry in partners:
            partner_name, partner_policy = _partner_entry(partner_entry)
            case_config = deepcopy(config)
            case_config.pop("evaluation", None)
            case_config["environment"] = _environment_for_layout(
                config["environment"],
                layout,
            )
            ego_policy = deepcopy(base_policies["agent_0"])
            if "deterministic" in evaluation and ego_policy.get("type") == "python_class":
                ego_policy.setdefault("config", {})["deterministic"] = bool(
                    evaluation["deterministic"]
                )
            case_config["policies"] = {
                "agent_0": ego_policy,
                "agent_1": partner_policy,
            }
            case_config["execution"] = {
                **(case_config.get("execution", {}) or {}),
                "num_episodes": len(seeds),
                "episode_seeds": seeds,
                "ego_player_positions": positions,
            }
            case_config["logging"] = {
                **(case_config.get("logging", {}) or {}),
                "output_dir": str(output_root / layout_label / partner_name),
            }
            aggregate = run_from_config(case_config)
            cases.append(
                EvaluationCaseResult(
                    layout=layout_label,
                    partner=partner_name,
                    aggregate=aggregate,
                )
            )
            all_scores.extend(aggregate["official_scores"])
            all_returns.extend(aggregate["returns_sparse"])

    report = EvaluationReport(
        cases=tuple(cases),
        mean_official_score=float(np.mean(all_scores)) if all_scores else 0.0,
        mean_sparse_return=float(np.mean(all_returns)) if all_returns else 0.0,
        num_rollouts=len(all_scores),
    )
    return report.to_dict()
