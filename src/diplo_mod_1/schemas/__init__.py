"""Pipeline-stage result schemas."""

from diplo_mod_1.schemas.evaluation import EvaluationResult, evaluate_predictor
from diplo_mod_1.schemas.pipeline import PreprocessingResult

__all__ = [
    "EvaluationResult",
    "PreprocessingResult",
    "evaluate_predictor",
]
