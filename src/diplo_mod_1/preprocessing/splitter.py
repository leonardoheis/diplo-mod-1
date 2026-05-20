"""Stratified train / validation / test splitter."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from diplo_mod_1.constants import POINTS_BINS

from .config import DataSplitterConfig


class DataSplitter:
    """Produces stratified train / validation / test index arrays.

    Stratification is on binned ``points`` so each split retains the same
    score distribution as the full dataset.

    Usage::

        splitter = DataSplitter()
        split_idx = splitter.split(featured_df)
        # split_idx["train"], split_idx["val"], split_idx["test"]
    """

    def __init__(self, config: DataSplitterConfig | None = None) -> None:
        self.config = config or DataSplitterConfig()

    def split(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Return positional index arrays for train, val, and test splits."""
        y = df["points"]
        strata = pd.cut(y, bins=POINTS_BINS, labels=False)
        idx = np.arange(len(df))

        train_val_idx, test_idx = train_test_split(
            idx,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=strata,
        )
        strata_tv = strata.iloc[train_val_idx]
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=self.config.val_size,
            random_state=self.config.random_state,
            stratify=strata_tv,
        )
        return {"train": train_idx, "val": val_idx, "test": test_idx}
