"""Configuration-driven evaluation and canonical scoring."""

from src.evaluation.scoring import OfficialScoreCalculator, calculate_official_score
from src.evaluation.types import EvaluationReport

__all__ = [
    "EvaluationReport",
    "OfficialScoreCalculator",
    "calculate_official_score",
]
