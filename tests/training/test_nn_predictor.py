"""Tests for WineScorePredictorNet."""

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
