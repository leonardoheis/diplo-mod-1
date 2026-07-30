"""WineScoreNet — the PyTorch regression MLP, and its dense/sparse Dataset."""

import numpy as np
import torch
from scipy import sparse
from torch import nn
from torch.utils.data import Dataset

from diplo_mod_1.domain.predictor import FeatureMatrix


class WineScoreNet(nn.Module):
    """Feed-forward MLP regressor: ``[Linear -> BatchNorm1d -> ReLU -> Dropout]``
    per hidden layer, then a single ``Linear(*, 1)`` output.

    Trained on the full concatenated feature matrix (tabular + TF-IDF, 2044
    columns) — same "just concatenate and let the model learn" approach that
    already worked for XGBoost, not a two-branch architecture.
    """

    def __init__(self, input_dim: int, hidden_sizes: list[int], dropout: float = 0.2) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class SparseTabularDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Wraps a dense-or-sparse feature matrix for PyTorch's DataLoader.

    Densifies one row at a time in ``__getitem__`` rather than the whole
    matrix up front — the TF-IDF block alone would be ~665MB dense despite
    being ~1% non-zero; this keeps memory proportional to batch size instead.
    """

    def __init__(self, X: FeatureMatrix, y: np.ndarray | None = None) -> None:
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.X[idx]
        if sparse.issparse(row):
            row = row.toarray()
        row = np.asarray(row, dtype=np.float32).ravel()
        features = torch.from_numpy(row)
        target = (
            torch.tensor(self.y[idx], dtype=torch.float32)
            if self.y is not None
            else torch.tensor(0.0)
        )
        return features, target
