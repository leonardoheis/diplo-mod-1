"""WineScorePredictor — structural contract for any trained scoring model."""

from typing import Protocol, TypeAlias, runtime_checkable

from typing_extensions import Self

import numpy as np
from scipy import sparse

FeatureMatrix: TypeAlias = np.ndarray | sparse.spmatrix | sparse.sparray
"""Dense or sparse feature input — e.g. tabular columns widened with sparse
TF-IDF columns via ``hstack``. XGBoost's sklearn API accepts both natively.

Includes both scipy sparse class families: ``spmatrix`` (what
``TfidfVectorizer``/``scipy.sparse.hstack`` actually return today) and the
newer ``sparray`` (scipy's recommended direction going forward, per
https://docs.scipy.org/doc/scipy/reference/sparse.migration_to_sparray.html).
Widen here, not narrow — nothing in this codebase can drop ``spmatrix``
until sklearn itself returns ``sparray`` from ``TfidfVectorizer``.
"""


@runtime_checkable
class WineScorePredictor(Protocol):
    """Duck-typed contract any trained wine-scoring model must satisfy.

    Both the XGBoost regressor (notebook 03) and the PyTorch network
    (notebook 04) implement this interface, enabling polymorphic evaluation
    in notebook 05 without a shared base class.

    Usage::

        def evaluate(model: WineScorePredictor, X: FeatureMatrix) -> np.ndarray:
            assert isinstance(model, WineScorePredictor)
            return model.predict(X)
    """

    def predict(self, X: FeatureMatrix) -> np.ndarray: ...

    def fit(self, X: FeatureMatrix, y: np.ndarray) -> Self: ...
