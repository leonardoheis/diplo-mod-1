"""WineScorePredictor — structural contract for any trained scoring model."""

from typing import Protocol, runtime_checkable

from typing_extensions import Self

import numpy as np


@runtime_checkable
class WineScorePredictor(Protocol):
    """Duck-typed contract any trained wine-scoring model must satisfy.

    Both the XGBoost regressor (notebook 03) and the PyTorch network
    (notebook 04) implement this interface, enabling polymorphic evaluation
    in notebook 05 without a shared base class.

    Usage::

        def evaluate(model: WineScorePredictor, X: np.ndarray) -> np.ndarray:
            assert isinstance(model, WineScorePredictor)
            return model.predict(X)
    """

    def predict(self, X: np.ndarray) -> np.ndarray: ...

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self: ...
