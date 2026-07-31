"""XGBoostTuner — Optuna-based hyperparameter search for XGBRegressor."""

from collections.abc import Callable

import numpy as np
import optuna
from sklearn.metrics import root_mean_squared_error
from xgboost import XGBRegressor

from diplo_mod_1.domain.predictor import FeatureMatrix
from diplo_mod_1.training.config import XGBoostTuningConfig
from diplo_mod_1.training.device import detect_xgboost_device

OptunaTrialCallback = Callable[[optuna.Study, optuna.trial.FrozenTrial], None]


class XGBoostTuner:
    """Bayesian (TPE) hyperparameter search for XGBRegressor, validated by early stopping.

    Each trial fits on the training split and is scored by RMSE against a
    held-out validation split, which also drives early stopping — the same
    split serves both purposes, so a test split can stay untouched until
    final evaluation.

    Usage::

        tuner = XGBoostTuner()
        study = tuner.tune(X_train, y_train, X_val, y_val)
        model = tuner.fit_best(X_train, y_train, X_val, y_val, study)
    """

    def __init__(self, config: XGBoostTuningConfig | None = None) -> None:
        self.config = config or XGBoostTuningConfig()

    # ── private ───────────────────────────────────────────────────────────────

    def _make_model(self, params: dict[str, int | float]) -> XGBRegressor:
        return XGBRegressor(
            **params,
            random_state=self.config.random_state,
            n_jobs=-1,
            tree_method="hist",
            device=detect_xgboost_device(),
            eval_metric="rmse",
            early_stopping_rounds=self.config.early_stopping_rounds,
        )

    def _build_objective(
        self,
        X_train: FeatureMatrix,
        y_train: np.ndarray,
        X_val: FeatureMatrix,
        y_val: np.ndarray,
    ) -> Callable[[optuna.Trial], float]:
        space = self.config.search_space

        def objective(trial: optuna.Trial) -> float:
            params = {
                "max_depth": trial.suggest_int("max_depth", *space.max_depth),
                "learning_rate": trial.suggest_float(
                    "learning_rate", *space.learning_rate, log=True
                ),
                "n_estimators": trial.suggest_int("n_estimators", *space.n_estimators),
                "subsample": trial.suggest_float("subsample", *space.subsample),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree", *space.colsample_bytree
                ),
                "min_child_weight": trial.suggest_int("min_child_weight", *space.min_child_weight),
                "reg_alpha": trial.suggest_float("reg_alpha", *space.reg_alpha, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", *space.reg_lambda, log=True),
                "gamma": trial.suggest_float("gamma", *space.gamma, log=True),
            }
            model = self._make_model(params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            return root_mean_squared_error(y_val, model.predict(X_val))

        return objective

    # ── public API ────────────────────────────────────────────────────────────

    def tune(
        self,
        X_train: FeatureMatrix,
        y_train: np.ndarray,
        X_val: FeatureMatrix,
        y_val: np.ndarray,
        callbacks: list[OptunaTrialCallback] | None = None,
    ) -> optuna.Study:
        """Run the Optuna search; returns the completed study.

        ``callbacks`` are passed straight to ``study.optimize`` — e.g. a
        function that logs each trial to an external tracker.
        """
        objective = self._build_objective(X_train, y_train, X_val, y_val)
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self.config.random_state),
        )
        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            show_progress_bar=True,
            callbacks=callbacks,
        )
        return study

    def fit_best(
        self,
        X_train: FeatureMatrix,
        y_train: np.ndarray,
        X_val: FeatureMatrix,
        y_val: np.ndarray,
        study: optuna.Study,
    ) -> XGBRegressor:
        """Refit on the winning trial's params, early-stopped against val."""
        model = self._make_model(study.best_params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return model
