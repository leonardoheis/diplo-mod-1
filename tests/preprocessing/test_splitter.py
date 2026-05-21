"""Tests for DataSplitter."""

import numpy as np
import pandas as pd

from diplo_mod_1.constants import TEST_SIZE, VAL_SIZE
from diplo_mod_1.preprocessing.splitter import DataSplitter


def test_split_returns_expected_keys(featured_df: pd.DataFrame) -> None:
    splits = DataSplitter().split(featured_df)
    assert set(splits.keys()) == {"train", "val", "test"}


def test_splits_are_non_overlapping(featured_df: pd.DataFrame) -> None:
    splits = DataSplitter().split(featured_df)
    train_set = set(splits["train"].tolist())
    val_set = set(splits["val"].tolist())
    test_set = set(splits["test"].tolist())
    assert not train_set & val_set
    assert not train_set & test_set
    assert not val_set & test_set


def test_splits_cover_full_dataset(featured_df: pd.DataFrame) -> None:
    splits = DataSplitter().split(featured_df)
    all_idx = (
        set(splits["train"].tolist()) | set(splits["val"].tolist()) | set(splits["test"].tolist())
    )
    assert all_idx == set(range(len(featured_df)))


def test_test_split_approximate_size(featured_df: pd.DataFrame) -> None:
    splits = DataSplitter().split(featured_df)
    actual_ratio = len(splits["test"]) / len(featured_df)
    assert abs(actual_ratio - TEST_SIZE) < 0.05


def test_val_split_approximate_size(featured_df: pd.DataFrame) -> None:
    splits = DataSplitter().split(featured_df)
    actual_ratio = len(splits["val"]) / len(featured_df)
    assert abs(actual_ratio - VAL_SIZE) < 0.05


def test_split_is_deterministic(featured_df: pd.DataFrame) -> None:
    s1 = DataSplitter().split(featured_df)
    s2 = DataSplitter().split(featured_df)
    np.testing.assert_array_equal(s1["train"], s2["train"])
    np.testing.assert_array_equal(s1["val"], s2["val"])
    np.testing.assert_array_equal(s1["test"], s2["test"])
