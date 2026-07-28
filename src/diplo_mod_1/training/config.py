"""Pydantic configuration for XGBoost hyperparameter tuning."""

from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from diplo_mod_1.constants import RANDOM_STATE


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
