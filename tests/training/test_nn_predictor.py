"""Tests for WineScorePredictorNet."""

import numpy as np
import pytest

from diplo_mod_1.domain.predictor import WineScorePredictor
from diplo_mod_1.training.nn_model import WineScorePredictorNet

_FAST_KWARGS = dict(
    input_dim=5,
    hidden_sizes=[4],
    max_epochs=5,
    early_stopping_patience=2,
    batch_size=8,
    device="cpu",
)


def test_implements_wine_score_predictor_protocol() -> None:
    model = WineScorePredictorNet(**_FAST_KWARGS)
    assert isinstance(model, WineScorePredictor)


def test_fit_returns_self(regression_split) -> None:
    X_train, y_train, X_val, y_val = regression_split
    model = WineScorePredictorNet(**_FAST_KWARGS)
    result = model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    assert result is model


def test_fit_tracks_loss_curves(regression_split) -> None:
    X_train, y_train, X_val, y_val = regression_split
    model = WineScorePredictorNet(**_FAST_KWARGS)
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    assert len(model.train_losses_) >= 1
    assert len(model.val_losses_) == len(model.train_losses_)
    assert model.best_epoch_ >= 0


def test_predict_returns_correct_shape(regression_split) -> None:
    X_train, y_train, X_val, y_val = regression_split
    model = WineScorePredictorNet(**_FAST_KWARGS)
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    preds = model.predict(X_val)
    assert preds.shape == y_val.shape


def test_fit_without_validation_set_still_works(regression_split) -> None:
    X_train, y_train, _, _ = regression_split
    model = WineScorePredictorNet(**_FAST_KWARGS)
    model.fit(X_train, y_train)
    assert len(model.train_losses_) == _FAST_KWARGS["max_epochs"]  # no early stop without val


def test_predict_before_fit_raises_clear_error() -> None:
    model = WineScorePredictorNet(**_FAST_KWARGS)
    with pytest.raises(RuntimeError, match="fit"):
        model.predict(np.zeros((2, 5), dtype=np.float32))


def test_fit_drops_trailing_batch_of_size_one(regression_split) -> None:
    """A trailing batch of exactly 1 sample would otherwise crash BatchNorm1d
    ('Expected more than 1 value per channel when training')."""
    X_train, y_train, X_val, y_val = regression_split
    kwargs = dict(_FAST_KWARGS)
    kwargs["batch_size"] = len(X_train) - 1  # last batch would have 1 sample
    model = WineScorePredictorNet(**kwargs)
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)  # must not raise


def test_fit_invokes_callback(regression_split) -> None:
    X_train, y_train, X_val, y_val = regression_split
    model = WineScorePredictorNet(**_FAST_KWARGS)
    calls: list[int] = []
    model.fit(
        X_train,
        y_train,
        X_val=X_val,
        y_val=y_val,
        callbacks=[lambda epoch, train_loss, val_loss: calls.append(epoch)],
    )
    assert calls == list(range(len(calls)))
