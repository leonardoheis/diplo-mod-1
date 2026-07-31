"""NNTuner — Optuna-based hyperparameter search for WineScorePredictorNet."""

from collections.abc import Callable
from typing import Any

import numpy as np
import optuna
from sklearn.metrics import root_mean_squared_error

from diplo_mod_1.domain.predictor import FeatureMatrix
from diplo_mod_1.training.device import detect_torch_device
from diplo_mod_1.training.nn_config import NNTuningConfig
from diplo_mod_1.training.nn_model import WineScorePredictorNet

OptunaTrialCallback = Callable[[optuna.Study, optuna.trial.FrozenTrial], None]


class NNTuner:
    """Bayesian (TPE) hyperparameter search for ``WineScorePredictorNet``,
    validated by early stopping — mirrors ``XGBoostTuner``'s shape.

    Usage::

        tuner = NNTuner()
        study = tuner.tune(X_train, y_train, X_val, y_val)
        model = tuner.fit_best(X_train, y_train, X_val, y_val, study)
    """

    def __init__(self, config: NNTuningConfig | None = None) -> None:
        self.config = config or NNTuningConfig()

    def _make_model(self, input_dim: int, params: dict[str, Any]) -> WineScorePredictorNet:
        hidden_sizes = [int(x) for x in str(params["architecture"]).split("_")]
        return WineScorePredictorNet(
            input_dim=input_dim,
            hidden_sizes=hidden_sizes,
            dropout=float(params["dropout"]),
            activation=str(params["activation"]),
            learning_rate=float(params["learning_rate"]),
            weight_decay=float(params["weight_decay"]),
            batch_size=int(params["batch_size"]),
            max_epochs=self.config.max_epochs,
            early_stopping_patience=self.config.early_stopping_patience,
            random_state=self.config.random_state,
            device=detect_torch_device(),
        )

    @staticmethod
    def _make_pruning_callback(trial: optuna.Trial) -> Callable[[int, float, float], None]:
        """Report each epoch's val loss to Optuna and stop the trial if it should be pruned."""

        def callback(epoch: int, train_loss: float, val_loss: float) -> None:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return callback

    def _build_objective(
        self, X_train: FeatureMatrix, y_train: np.ndarray, X_val: FeatureMatrix, y_val: np.ndarray
    ) -> Callable[[optuna.Trial], float]:
        space = self.config.search_space

        def objective(trial: optuna.Trial) -> float:
            params = {
                "architecture": trial.suggest_categorical("architecture", space.architecture),
                "dropout": trial.suggest_float("dropout", *space.dropout),
                "activation": trial.suggest_categorical("activation", space.activation),
                "learning_rate": trial.suggest_float(
                    "learning_rate", *space.learning_rate, log=True
                ),
                "weight_decay": trial.suggest_float("weight_decay", *space.weight_decay, log=True),
                "batch_size": trial.suggest_categorical("batch_size", space.batch_size),
            }
            model = self._make_model(X_train.shape[1], params)
            model.fit(
                X_train,
                y_train,
                X_val=X_val,
                y_val=y_val,
                callbacks=[self._make_pruning_callback(trial)],
            )
            return root_mean_squared_error(y_val, model.predict(X_val))

        return objective

    def tune(
        self,
        X_train: FeatureMatrix,
        y_train: np.ndarray,
        X_val: FeatureMatrix,
        y_val: np.ndarray,
        callbacks: list[OptunaTrialCallback] | None = None,
    ) -> optuna.Study:
        """Run the Optuna search; returns the completed study."""
        objective = self._build_objective(X_train, y_train, X_val, y_val)
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self.config.random_state),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=3),
        )
        study.optimize(
            objective, n_trials=self.config.n_trials, show_progress_bar=True, callbacks=callbacks
        )
        return study

    def fit_best(
        self,
        X_train: FeatureMatrix,
        y_train: np.ndarray,
        X_val: FeatureMatrix,
        y_val: np.ndarray,
        study: optuna.Study,
        *,
        callbacks: list[Callable[[int, float, float], None]] | None = None,
    ) -> WineScorePredictorNet:
        """Refit on the winning trial's params, early-stopped against val."""
        model = self._make_model(X_train.shape[1], study.best_params)
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val, callbacks=callbacks)
        return model
