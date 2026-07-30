"""Tests for evaluate_predictor."""

import numpy as np
from scipy import sparse

from diplo_mod_1.schemas.evaluation import EvaluationResult, evaluate_predictor


class _ConstantPredictor:
    """Minimal WineScorePredictor stub that always predicts a fixed value."""

    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(X.shape[0], self.value, dtype=np.float32)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_ConstantPredictor":
        return self


def test_evaluate_predictor_scores_all_splits() -> None:
    model = _ConstantPredictor(90.0)
    splits = {
        "train": (np.zeros((5, 2)), np.full(5, 90.0)),
        "val": (np.zeros((3, 2)), np.full(3, 92.0)),
        "test": (np.zeros((4, 2)), np.full(4, 88.0)),
    }
    result = evaluate_predictor(model, splits, model_type="xgboost")
    assert isinstance(result, EvaluationResult)
    assert {m.split for m in result.metrics} == {"train", "val", "test"}


def test_perfect_predictions_yield_zero_rmse() -> None:
    model = _ConstantPredictor(90.0)
    splits = {"test": (np.zeros((5, 2)), np.full(5, 90.0))}
    result = evaluate_predictor(model, splits, model_type="xgboost")
    m = result.best(split="test")
    assert m.rmse == 0.0
    assert m.mae == 0.0


def test_model_type_propagated() -> None:
    model = _ConstantPredictor(90.0)
    splits = {"test": (np.zeros((2, 2)), np.array([90.0, 91.0]))}
    result = evaluate_predictor(model, splits, model_type="neural_net")
    assert result.metrics[0].model_type == "neural_net"


def test_accepts_sparse_feature_matrix() -> None:
    """X may be sparse (e.g. tabular + TF-IDF hstacked) — not just dense np.ndarray."""
    model = _ConstantPredictor(90.0)
    splits = {"test": (sparse.csr_matrix(np.zeros((3, 2))), np.full(3, 90.0))}
    result = evaluate_predictor(model, splits, model_type="xgboost")
    assert result.best(split="test").rmse == 0.0
