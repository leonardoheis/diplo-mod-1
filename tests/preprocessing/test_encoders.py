"""Tests for FeatureMatrix, TabularEncoder, and TextEncoder."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from diplo_mod_1.preprocessing.config import TextEncoderConfig
from diplo_mod_1.preprocessing.encoders import FeatureMatrix, TabularEncoder, TextEncoder


# ── FeatureMatrix ─────────────────────────────────────────────────────────────


def test_feature_matrix_positional_unpacking() -> None:
    mat = FeatureMatrix(
        X=np.zeros((5, 3), dtype=np.float32),
        feature_names=["a", "b", "c"],
        column_groups={"continuous": [0, 1], "binary": [2], "ohe": []},
    )
    X, names, groups = mat
    assert X.shape == (5, 3)
    assert names == ["a", "b", "c"]
    assert groups["continuous"] == [0, 1]


def test_feature_matrix_attribute_access() -> None:
    mat = FeatureMatrix(
        X=np.zeros((5, 3), dtype=np.float32),
        feature_names=["a", "b", "c"],
        column_groups={"continuous": [0], "binary": [1, 2], "ohe": []},
    )
    assert mat.X.shape == (5, 3)
    assert mat.feature_names[0] == "a"
    assert "binary" in mat.column_groups


# ── TabularEncoder ────────────────────────────────────────────────────────────


def test_tabular_transform_before_fit_raises(featured_df: pd.DataFrame) -> None:
    with pytest.raises(RuntimeError, match="fit"):
        TabularEncoder().transform(featured_df)


def test_tabular_fit_transform_returns_feature_matrix(featured_df: pd.DataFrame) -> None:
    train_df = featured_df.iloc[:40]
    y_train = train_df["points"].to_numpy(dtype=np.float32)
    result = TabularEncoder().fit_transform(train_df, y_train)
    assert isinstance(result, FeatureMatrix)


def test_tabular_feature_names_length_matches_x_columns(featured_df: pd.DataFrame) -> None:
    train_df = featured_df.iloc[:40]
    y_train = train_df["points"].to_numpy(dtype=np.float32)
    result = TabularEncoder().fit_transform(train_df, y_train)
    assert len(result.feature_names) == result.X.shape[1]


def test_tabular_column_groups_have_expected_keys(featured_df: pd.DataFrame) -> None:
    train_df = featured_df.iloc[:40]
    y_train = train_df["points"].to_numpy(dtype=np.float32)
    result = TabularEncoder().fit_transform(train_df, y_train)
    assert "continuous" in result.column_groups
    assert "binary" in result.column_groups
    assert "ohe" in result.column_groups


def test_tabular_transform_matches_fit_transform_feature_count(featured_df: pd.DataFrame) -> None:
    train_df = featured_df.iloc[:40]
    val_df = featured_df.iloc[40:]
    y_train = train_df["points"].to_numpy(dtype=np.float32)
    encoder = TabularEncoder()
    train_result = encoder.fit_transform(train_df, y_train)
    val_result = encoder.transform(val_df)
    assert val_result.X.shape[1] == train_result.X.shape[1]


def test_tabular_x_dtype_is_float32(featured_df: pd.DataFrame) -> None:
    train_df = featured_df.iloc[:40]
    y_train = train_df["points"].to_numpy(dtype=np.float32)
    result = TabularEncoder().fit_transform(train_df, y_train)
    assert result.X.dtype == np.float32


# ── TextEncoder ───────────────────────────────────────────────────────────────


def test_text_transform_before_fit_raises(featured_df: pd.DataFrame) -> None:
    with pytest.raises(RuntimeError, match="fit"):
        TextEncoder().transform(featured_df)


def test_text_fit_transform_returns_sparse_matrix(featured_df: pd.DataFrame) -> None:
    config = TextEncoderConfig(max_features=50, min_df=1)
    train_df = featured_df.iloc[:40]
    enc = TextEncoder(config)
    enc.fit(train_df)
    result = enc.transform(train_df)
    assert sparse.issparse(result)


def test_text_transform_row_count_matches_input(featured_df: pd.DataFrame) -> None:
    config = TextEncoderConfig(max_features=50, min_df=1)
    train_df = featured_df.iloc[:40]
    val_df = featured_df.iloc[40:]
    enc = TextEncoder(config)
    enc.fit(train_df)
    result = enc.transform(val_df)
    assert result.shape[0] == len(val_df)


def test_text_transform_column_count_bounded_by_max_features(featured_df: pd.DataFrame) -> None:
    max_features = 50
    config = TextEncoderConfig(max_features=max_features, min_df=1)
    train_df = featured_df.iloc[:40]
    enc = TextEncoder(config)
    enc.fit(train_df)
    result = enc.transform(train_df)
    assert result.shape[1] <= max_features
