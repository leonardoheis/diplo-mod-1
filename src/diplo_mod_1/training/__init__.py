"""XGBoost and PyTorch NN training package."""

from diplo_mod_1.training.config import (
    RunRecord,
    TuningHistory,
    XGBoostSearchSpace,
    XGBoostTuningConfig,
)
from diplo_mod_1.training.device import detect_torch_device, detect_xgboost_device
from diplo_mod_1.training.nn_config import NNSearchSpace, NNTuningConfig
from diplo_mod_1.training.nn_model import WineScoreNet, WineScorePredictorNet
from diplo_mod_1.training.nn_registry import NNModelRegistry
from diplo_mod_1.training.nn_tuner import NNTuner
from diplo_mod_1.training.registry import ModelRegistry
from diplo_mod_1.training.xgboost_tuner import XGBoostTuner

__all__ = [
    "ModelRegistry",
    "NNModelRegistry",
    "NNSearchSpace",
    "NNTuner",
    "NNTuningConfig",
    "RunRecord",
    "TuningHistory",
    "WineScoreNet",
    "WineScorePredictorNet",
    "XGBoostSearchSpace",
    "XGBoostTuner",
    "XGBoostTuningConfig",
    "detect_torch_device",
    "detect_xgboost_device",
]
