"""WineScoreNet — the PyTorch regression MLP, and its dense/sparse Dataset."""

from collections.abc import Callable

import numpy as np
import torch
from scipy import sparse
from torch import nn
from torch.utils.data import DataLoader, Dataset

from diplo_mod_1.domain.predictor import FeatureMatrix
from diplo_mod_1.training.device import detect_torch_device


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


class WineScorePredictorNet:
    """Sklearn-style wrapper around ``WineScoreNet`` — implements
    ``WineScorePredictor`` (``domain/predictor.py``) so ``evaluate_predictor``
    can score it identically to the XGBoost model.

    ``fit``'s extra keyword-only args (``X_val``, ``y_val``, ``callbacks``)
    are additional to the Protocol's minimal ``fit(X, y)`` — fine, since
    ``@runtime_checkable`` Protocols only check method presence, not exact
    signatures.

    Usage::

        model = WineScorePredictorNet(input_dim=2044, hidden_sizes=[256, 64])
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        preds = model.predict(X_test)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: list[int],
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int = 128,
        max_epochs: int = 100,
        early_stopping_patience: int = 10,
        random_state: int = 42,
        device: str | None = None,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.random_state = random_state
        self.device = device

    def fit(
        self,
        X: FeatureMatrix,
        y: np.ndarray,
        *,
        X_val: FeatureMatrix | None = None,
        y_val: np.ndarray | None = None,
        callbacks: list[Callable[[int, float, float], None]] | None = None,
    ) -> "WineScorePredictorNet":
        torch.manual_seed(self.random_state)
        device = self.device or detect_torch_device()
        self.model_ = WineScoreNet(self.input_dim, self.hidden_sizes, self.dropout).to(device)
        optimizer = torch.optim.AdamW(
            self.model_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        loss_fn = nn.MSELoss()

        train_loader = DataLoader(
            SparseTabularDataset(X, y), batch_size=self.batch_size, shuffle=True
        )
        has_val = X_val is not None and y_val is not None
        val_loader = (
            DataLoader(SparseTabularDataset(X_val, y_val), batch_size=self.batch_size)
            if has_val
            else None
        )

        self.train_losses_: list[float] = []
        self.val_losses_: list[float] = []
        self.best_epoch_ = 0
        best_val_loss = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        epochs_without_improvement = 0

        for epoch in range(self.max_epochs):
            self.model_.train()
            running_loss, n_samples = 0.0, 0
            for features, target in train_loader:
                features, target = features.to(device), target.to(device)
                optimizer.zero_grad()
                preds = self.model_(features)
                loss = loss_fn(preds, target)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * features.size(0)
                n_samples += features.size(0)
            train_loss = running_loss / n_samples
            self.train_losses_.append(train_loss)

            val_loss = train_loss
            if val_loader is not None:
                self.model_.eval()
                running_val, n_val = 0.0, 0
                with torch.no_grad():
                    for features, target in val_loader:
                        features, target = features.to(device), target.to(device)
                        running_val += loss_fn(
                            self.model_(features), target
                        ).item() * features.size(0)
                        n_val += features.size(0)
                val_loss = running_val / n_val
            self.val_losses_.append(val_loss)

            if callbacks:
                for callback in callbacks:
                    callback(epoch, train_loss, val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self.model_.state_dict().items()}
                self.best_epoch_ = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if has_val and epochs_without_improvement >= self.early_stopping_patience:
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)

        return self

    def predict(self, X: FeatureMatrix) -> np.ndarray:
        device = self.device or detect_torch_device()
        self.model_.to(device)
        self.model_.eval()
        loader = DataLoader(SparseTabularDataset(X), batch_size=self.batch_size)
        predictions: list[np.ndarray] = []
        with torch.no_grad():
            for features, _ in loader:
                predictions.append(self.model_(features.to(device)).cpu().numpy())
        return np.concatenate(predictions)
