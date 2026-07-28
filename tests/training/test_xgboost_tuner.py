"""Tests for XGBoostTuner."""

import numpy as np
import optuna

from diplo_mod_1.training.config import XGBoostSearchSpace, XGBoostTuningConfig
from diplo_mod_1.training.xgboost_tuner import XGBoostTuner

_FAST_CONFIG = XGBoostTuningConfig(
    n_trials=2,
    early_stopping_rounds=3,
    search_space=XGBoostSearchSpace(n_estimators=(5, 10)),
)


def test_tune_returns_completed_study(
    regression_split: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    X_train, y_train, X_val, y_val = regression_split
    tuner = XGBoostTuner(_FAST_CONFIG)
    study = tuner.tune(X_train, y_train, X_val, y_val)
    assert isinstance(study, optuna.Study)
    assert len(study.trials) == 2
    assert study.best_value >= 0


def test_tune_invokes_callback(
    regression_split: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    X_train, y_train, X_val, y_val = regression_split
    tuner = XGBoostTuner(_FAST_CONFIG)
    calls: list[int] = []
    tuner.tune(X_train, y_train, X_val, y_val, callbacks=[lambda s, t: calls.append(t.number)])
    assert calls == [0, 1]


def test_fit_best_returns_fitted_model(
    regression_split: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    X_train, y_train, X_val, y_val = regression_split
    tuner = XGBoostTuner(_FAST_CONFIG)
    study = tuner.tune(X_train, y_train, X_val, y_val)
    model = tuner.fit_best(X_train, y_train, X_val, y_val, study)
    preds = model.predict(X_val)
    assert preds.shape == y_val.shape
