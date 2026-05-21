"""ModelMetrics — evaluation result for one model on one data split."""

from typing import Literal

from pydantic import Field

from diplo_mod_1.domain.base import BaseDomainModel


class ModelMetrics(BaseDomainModel):
    """Regression metrics for a trained model evaluated on a single split.

    Usage::

        m = ModelMetrics(
            model_type="xgboost",
            split="test",
            rmse=1.83,
            mae=1.41,
            r2=0.51,
        )
    """

    model_type: Literal["xgboost", "neural_net"]
    split: Literal["train", "val", "test"]
    rmse: float = Field(ge=0.0)
    mae: float = Field(ge=0.0)
    r2: float
