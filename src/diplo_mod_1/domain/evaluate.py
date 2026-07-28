"""evaluate_predictor — shared train/val/test scoring for any WineScorePredictor."""

from typing import Literal

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from diplo_mod_1.domain.metrics import ModelMetrics
from diplo_mod_1.domain.predictor import WineScorePredictor
from diplo_mod_1.schemas.evaluation import EvaluationResult


def evaluate_predictor(
    model: WineScorePredictor,
    splits: dict[Literal["train", "val", "test"], tuple[np.ndarray, np.ndarray]],
    model_type: Literal["xgboost", "neural_net"],
) -> EvaluationResult:
    """Score ``model`` on each ``(X, y)`` split and collect the metrics.

    Shared by notebooks 03 and 04 so both models are scored identically,
    for the notebook 05 head-to-head comparison.
    """
    result = EvaluationResult()
    for split_name, (X, y) in splits.items():
        preds = model.predict(X)
        result.add(
            ModelMetrics(
                model_type=model_type,
                split=split_name,
                rmse=root_mean_squared_error(y, preds),
                mae=mean_absolute_error(y, preds),
                r2=r2_score(y, preds),
            )
        )
    return result
