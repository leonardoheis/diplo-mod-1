"""Tests for NNTuner."""

import optuna
import pytest

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


def test_fit_best_invokes_callback_once_per_epoch(regression_split) -> None:
    X_train, y_train, X_val, y_val = regression_split
    tuner = NNTuner(_FAST_CONFIG)
    study = tuner.tune(X_train, y_train, X_val, y_val)
    calls: list[int] = []
    model = tuner.fit_best(
        X_train,
        y_train,
        X_val,
        y_val,
        study,
        callbacks=[lambda epoch, train_loss, val_loss: calls.append(epoch)],
    )
    assert calls == list(range(len(model.train_losses_)))


def test_make_model_threads_activation_through(regression_split) -> None:
    X_train, y_train, X_val, y_val = regression_split
    config = NNTuningConfig(
        n_trials=1,
        max_epochs=2,
        early_stopping_patience=1,
        search_space=NNSearchSpace(architecture=["4_2"], batch_size=[8], activation=["gelu"]),
    )
    tuner = NNTuner(config)
    study = tuner.tune(X_train, y_train, X_val, y_val)
    assert study.best_params["activation"] == "gelu"
    model = tuner.fit_best(X_train, y_train, X_val, y_val, study)
    assert model.activation == "gelu"


def test_pruning_callback_raises_when_trial_should_prune() -> None:
    pruner = optuna.pruners.MedianPruner(n_startup_trials=0, n_warmup_steps=0)
    study = optuna.create_study(direction="minimize", pruner=pruner)

    good_trial = study.ask()
    good_trial.report(0.1, step=0)
    study.tell(good_trial, 0.1)

    bad_trial = study.ask()
    callback = NNTuner._make_pruning_callback(bad_trial)
    with pytest.raises(optuna.TrialPruned):
        callback(0, train_loss=0.0, val_loss=100.0)
