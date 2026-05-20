"""EvaluationResult — accumulated metrics across models and splits."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from diplo_mod_1.domain.metrics import ModelMetrics


class EvaluationResult(BaseModel):
    """Collects ModelMetrics entries produced during notebook 05 evaluation.

    Mutable by design: metrics are appended as each model/split is scored.

    Usage::

        result = EvaluationResult()
        result.add(ModelMetrics(model_type="xgboost", split="test", rmse=1.83, mae=1.41, r2=0.51))
        result.add(ModelMetrics(model_type="neural_net", split="test", rmse=1.71, mae=1.33, r2=0.54))
        best = result.best(split="test")
    """

    metrics: list[ModelMetrics] = []

    def add(self, m: ModelMetrics) -> None:
        """Append a metrics record."""
        self.metrics.append(m)

    def best(self, split: Literal["train", "val", "test"] = "test") -> ModelMetrics:
        """Return the entry with the lowest RMSE for the given split."""
        candidates = [m for m in self.metrics if m.split == split]
        if not candidates:
            raise ValueError(f"No metrics recorded for split '{split}'.")
        return min(candidates, key=lambda m: m.rmse)

    def for_model(self, model_type: Literal["xgboost", "neural_net"]) -> list[ModelMetrics]:
        """Return all entries for a specific model type."""
        return [m for m in self.metrics if m.model_type == model_type]
