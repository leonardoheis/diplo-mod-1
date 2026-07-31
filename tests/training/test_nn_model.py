"""Tests for WineScoreNet and SparseTabularDataset."""

import numpy as np
import torch
from scipy import sparse
from torch import nn

from diplo_mod_1.training.nn_model import SparseTabularDataset, WineScoreNet


def test_wine_score_net_forward_shape() -> None:
    net = WineScoreNet(input_dim=10, hidden_sizes=[8, 4], dropout=0.1)
    x = torch.randn(5, 10)
    out = net(x)
    assert out.shape == (5,)


def test_wine_score_net_handles_single_hidden_layer() -> None:
    net = WineScoreNet(input_dim=6, hidden_sizes=[4], dropout=0.0)
    out = net(torch.randn(3, 6))
    assert out.shape == (3,)


def test_wine_score_net_uses_requested_activation() -> None:
    net = WineScoreNet(input_dim=6, hidden_sizes=[4], dropout=0.0, activation="gelu")
    assert any(isinstance(layer, nn.GELU) for layer in net.network)
    assert not any(isinstance(layer, nn.ReLU) for layer in net.network)


def test_dataset_len_matches_row_count() -> None:
    X = np.zeros((7, 4), dtype=np.float32)
    y = np.arange(7, dtype=np.float32)
    ds = SparseTabularDataset(X, y)
    assert len(ds) == 7


def test_dataset_getitem_dense_input() -> None:
    X = np.arange(12, dtype=np.float32).reshape(3, 4)
    y = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    ds = SparseTabularDataset(X, y)
    features, target = ds[1]
    assert features.shape == (4,)
    assert torch.equal(features, torch.tensor(X[1]))
    assert target.item() == 20.0


def test_dataset_getitem_sparse_input_densifies_one_row() -> None:
    X = sparse.csr_matrix(np.eye(5, dtype=np.float32))
    y = np.arange(5, dtype=np.float32)
    ds = SparseTabularDataset(X, y)
    features, target = ds[2]
    assert isinstance(features, torch.Tensor)
    assert features.shape == (5,)
    assert features[2].item() == 1.0
    assert target.item() == 2.0


def test_dataset_without_targets_returns_placeholder() -> None:
    X = np.zeros((2, 3), dtype=np.float32)
    ds = SparseTabularDataset(X)
    features, target = ds[0]
    assert features.shape == (3,)
    assert target.item() == 0.0
