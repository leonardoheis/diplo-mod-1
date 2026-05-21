"""Tests for WinePrediction domain model."""

import pytest
from pydantic import ValidationError

from diplo_mod_1.constants import POINTS_MAX, POINTS_MIN
from diplo_mod_1.domain.prediction import WinePrediction


def test_valid_instance_creates() -> None:
    pred = WinePrediction(predicted_points=91.4, model_type="xgboost")
    assert pred.predicted_points == pytest.approx(91.4)
    assert pred.model_type == "xgboost"


def test_neural_net_model_type_accepted() -> None:
    pred = WinePrediction(predicted_points=88.0, model_type="neural_net")
    assert pred.model_type == "neural_net"


def test_predicted_points_below_minimum_raises() -> None:
    with pytest.raises(ValidationError):
        WinePrediction(predicted_points=float(POINTS_MIN) - 0.1, model_type="xgboost")


def test_predicted_points_above_maximum_raises() -> None:
    with pytest.raises(ValidationError):
        WinePrediction(predicted_points=float(POINTS_MAX) + 0.1, model_type="neural_net")


def test_invalid_model_type_raises() -> None:
    with pytest.raises(ValidationError):
        WinePrediction(predicted_points=90.0, model_type="random_forest")  # type: ignore[arg-type]


def test_frozen_rejects_mutation() -> None:
    pred = WinePrediction(predicted_points=91.4, model_type="xgboost")
    with pytest.raises(ValidationError):
        pred.predicted_points = 92.0  # type: ignore[misc]
