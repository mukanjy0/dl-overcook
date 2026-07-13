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


def _inference_modes(evaluation: dict[str, Any]) -> list[tuple[str, bool]]:
    configured = evaluation.get("inference_modes")
    if configured is None:
        deterministic = bool(evaluation.get("deterministic", True))
        return [("deterministic" if deterministic else "stochastic", deterministic)]
    if not isinstance(configured, list) or not configured:
        raise ValueError("evaluation.inference_modes must be a non-empty list")

    modes: list[tuple[str, bool]] = []
    for value in configured:
        if isinstance(value, bool):
            mode = ("deterministic" if value else "stochastic", value)
        else:
            normalized = str(value).strip().lower()
            if normalized not in {"deterministic", "stochastic"}:
                raise ValueError(
                    "evaluation.inference_modes values must be deterministic or stochastic"
                )
            mode = (normalized, normalized == "deterministic")
        if mode[0] not in {name for name, _ in modes}:
            modes.append(mode)
    return modes


def _position_metrics(episode_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize delivery and official-score outcomes by physical ego position."""
    metrics: dict[str, Any] = {}
    for position in sorted({int(item["ego_player_index"]) for item in episode_results}):
        episodes = [
            item for item in episode_results if int(item["ego_player_index"]) == position
        ]
        soup_counts = [len(item.get("delivery_timesteps", [])) for item in episodes]
        scores = [float(item.get("official_score", 0.0)) for item in episodes]
        metrics[str(position)] = {
            "num_episodes": len(episodes),
            "soup_counts": soup_counts,
            "mean_soup_count": float(np.mean(soup_counts)) if soup_counts else 0.0,
            "official_scores": scores,
            "mean_official_score": float(np.mean(scores)) if scores else 0.0,
            "minimum_official_score": float(np.min(scores)) if scores else 0.0,
            "zero_soup_rate": (
                float(np.mean([count == 0 for count in soup_counts])) if soup_counts else 0.0
            ),
        }
    return metrics


def _mode_metrics(cases: list[EvaluationCaseResult]) -> dict[str, Any]:
    episodes = [
        episode
        for case in cases
        for episode in case.aggregate.get("episode_results", [])
    ]
    position_metrics = _position_metrics(episodes)
    position_scores = [
        float(values["mean_official_score"]) for values in position_metrics.values()
    ]
    soup_counts = [len(episode.get("delivery_timesteps", [])) for episode in episodes]
    scores = [float(episode.get("official_score", 0.0)) for episode in episodes]
    return {
        "num_rollouts": len(episodes),
        "position_metrics": position_metrics,
        "mean_soup_count": float(np.mean(soup_counts)) if soup_counts else 0.0,
        "zero_soup_rate": (
            float(np.mean([count == 0 for count in soup_counts])) if soup_counts else 0.0
        ),
        "mean_official_score": float(np.mean(scores)) if scores else 0.0,
        "mean_position_score": (
            float(np.mean(position_scores)) if position_scores else 0.0
        ),
        "min_position_score": float(np.min(position_scores)) if position_scores else 0.0,
    }


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
    cases_by_mode: dict[str, list[EvaluationCaseResult]] = {}
    all_scores: list[float] = []
    all_returns: list[float] = []

    for mode_name, deterministic in _inference_modes(evaluation):
        mode_cases: list[EvaluationCaseResult] = []
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
                if ego_policy.get("type") == "python_class":
                    ego_policy.setdefault("config", {})["deterministic"] = deterministic
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
                    "output_dir": str(
                        output_root / mode_name / layout_label / partner_name
                    ),
                }
                aggregate = run_from_config(case_config)
                case = EvaluationCaseResult(
                    layout=layout_label,
                    partner=partner_name,
                    inference_mode=mode_name,
                    aggregate=aggregate,
                    position_metrics=_position_metrics(aggregate["episode_results"]),
                )
                cases.append(case)
                mode_cases.append(case)
                all_scores.extend(aggregate["official_scores"])
                all_returns.extend(aggregate["returns_sparse"])
        cases_by_mode[mode_name] = mode_cases

    modes = {name: _mode_metrics(mode_cases) for name, mode_cases in cases_by_mode.items()}
    primary_mode = "deterministic" if "deterministic" in modes else next(iter(modes))
    primary_scores = [
        score
        for case in cases_by_mode[primary_mode]
        for score in case.aggregate["official_scores"]
    ]
    primary_returns = [
        value
        for case in cases_by_mode[primary_mode]
        for value in case.aggregate["returns_sparse"]
    ]

    report = EvaluationReport(
        cases=tuple(cases),
        mean_official_score=(
            float(np.mean(primary_scores)) if primary_scores else 0.0
        ),
        mean_sparse_return=(
            float(np.mean(primary_returns)) if primary_returns else 0.0
        ),
        num_rollouts=len(all_scores),
        modes=modes,
    )
    return report.to_dict()
