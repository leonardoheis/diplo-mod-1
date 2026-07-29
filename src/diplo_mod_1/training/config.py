"""Pydantic configuration and result records for XGBoost hyperparameter tuning."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from diplo_mod_1.constants import RANDOM_STATE
from diplo_mod_1.domain.metrics import ModelMetrics


class XGBoostSearchSpace(BaseModel):
    """Optuna suggestion ranges (lower, upper) for each XGBoost hyperparameter."""

    max_depth: tuple[int, int] = (3, 8)
    learning_rate: tuple[float, float] = (0.01, 0.3)
    n_estimators: tuple[int, int] = (100, 1500)
    subsample: tuple[float, float] = (0.5, 1.0)
    colsample_bytree: tuple[float, float] = (0.5, 1.0)
    min_child_weight: tuple[int, int] = (1, 10)
    reg_alpha: tuple[float, float] = (1e-8, 10.0)
    reg_lambda: tuple[float, float] = (1e-8, 10.0)
    gamma: tuple[float, float] = (1e-8, 5.0)

    @model_validator(mode="after")
    def bounds_ordered(self) -> "XGBoostSearchSpace":
        for name, (lo, hi) in self.model_dump().items():
            if lo > hi:
                raise ValueError(f"{name}: lower bound {lo} exceeds upper bound {hi}")
        return self


class XGBoostTuningConfig(BaseModel):
    """Hyperparameters for XGBoostTuner.

    Usage::

        # defaults
        config = XGBoostTuningConfig()

        # load from JSON file — lets you swap search spaces without editing code
        config = XGBoostTuningConfig.from_json(Path("configs/xgboost_tuning.json"))
    """

    n_trials: int = Field(default=50, ge=1)
    early_stopping_rounds: int = Field(default=50, ge=1)
    random_state: int = RANDOM_STATE
    search_space: XGBoostSearchSpace = Field(default_factory=XGBoostSearchSpace)

    @classmethod
    def from_json(cls, path: Path) -> "XGBoostTuningConfig":
        """Load config from a JSON file (same shape as ``model_dump()``)."""
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


class RunRecord(BaseModel):
    """One completed tuning run — the config used, the winning params, and its metrics."""

    run_id: str
    tuning_config: str
    model_filename: str
    best_params: dict[str, float]
    metrics: list[ModelMetrics]


class TuningHistory(BaseModel):
    """Every ``RunRecord`` on file, plus a pointer to the best one — persisted as one JSON file.

    Usage::

        history = TuningHistory()
        history.runs.append(record)
        history.best_run_id = history.best_run(split="test").run_id
    """

    runs: list[RunRecord] = Field(default_factory=list)
    best_run_id: str | None = None

    def best_run(self, split: Literal["train", "val", "test"] = "test") -> RunRecord | None:
        """Return the run with the lowest RMSE on ``split``, or None if no run has it."""
        candidates = [r for r in self.runs if any(m.split == split for m in r.metrics)]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda r: next(m.rmse for m in r.metrics if m.split == split),
        )
