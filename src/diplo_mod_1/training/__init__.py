"""XGBoost hyperparameter tuning package."""

from diplo_mod_1.training.config import (
    RunRecord,
    TuningHistory,
    XGBoostSearchSpace,
    XGBoostTuningConfig,
)
from diplo_mod_1.training.registry import ModelRegistry
from diplo_mod_1.training.xgboost_tuner import XGBoostTuner, detect_device

__all__ = [
    "ModelRegistry",
    "RunRecord",
    "TuningHistory",
    "XGBoostSearchSpace",
    "XGBoostTuner",
    "XGBoostTuningConfig",
    "detect_device",
]
