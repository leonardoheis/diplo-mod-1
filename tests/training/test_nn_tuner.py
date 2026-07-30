"""Tests for NNTuner."""

import optuna

from diplo_mod_1.training.nn_config import NNSearchSpace, NNTuningConfig
from diplo_mod_1.training.nn_tuner import NNTuner

_FAST_CONFIG = NNTuningConfig(
    n_trials=2,
    max_epochs=3,
    early_stopping_patience=2,
    search_space=NNSearchSpace(architecture=["4_2"], batch_size=[8]),
)


def test_tune_returns_completed_study(regression_split) -> None:
    X_train, y_train, X_val, y_val = regression_split
    tuner = NNTuner(_FAST_CONFIG)
    study = tuner.tune(X_train, y_train, X_val, y_val)
    assert isinstance(study, optuna.Study)
    assert len(study.trials) == 2
    assert study.best_value >= 0


def test_tune_invokes_callback(regression_split) -> None:
    X_train, y_train, X_val, y_val = regression_split
    tuner = NNTuner(_FAST_CONFIG)
    calls: list[int] = []
    tuner.tune(X_train, y_train, X_val, y_val, callbacks=[lambda s, t: calls.append(t.number)])
    assert calls == [0, 1]


def test_fit_best_returns_fitted_model(regression_split) -> None:
    X_train, y_train, X_val, y_val = regression_split
    tuner = NNTuner(_FAST_CONFIG)
    study = tuner.tune(X_train, y_train, X_val, y_val)
    model = tuner.fit_best(X_train, y_train, X_val, y_val, study)
    preds = model.predict(X_val)
    assert preds.shape == y_val.shape
