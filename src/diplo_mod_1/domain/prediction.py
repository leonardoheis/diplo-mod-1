"""WinePrediction — a single model prediction for a wine's quality score."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from diplo_mod_1.domain.base import BaseDomainModel


class WinePrediction(BaseDomainModel):
    """Predicted quality score produced by a trained model.

    Usage::

        pred = WinePrediction(predicted_points=91.4, model_type="xgboost")
    """

    predicted_points: float = Field(ge=80.0, le=100.0)
    model_type: Literal["xgboost", "neural_net"]
