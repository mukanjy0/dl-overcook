"""Serializable evaluation result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EvaluationCaseResult:
    """Results for one configured layout and partner combination."""

    layout: str
    partner: str
    aggregate: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "layout": self.layout,
            "partner": self.partner,
            "aggregate": self.aggregate,
        }


@dataclass(frozen=True)
class EvaluationReport:
    """Aggregate report for a configuration-driven evaluation suite."""

    cases: tuple[EvaluationCaseResult, ...] = field(default_factory=tuple)
    mean_official_score: float = 0.0
    mean_sparse_return: float = 0.0
    num_rollouts: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "cases": [case.to_dict() for case in self.cases],
            "mean_official_score": self.mean_official_score,
            "mean_return_sparse": self.mean_sparse_return,
            "num_rollouts": self.num_rollouts,
        }
