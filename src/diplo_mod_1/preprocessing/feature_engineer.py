"""FeatureEngineer — derives new columns from a cleaned Wine Reviews DataFrame."""

from __future__ import annotations

import numpy as np
import pandas as pd

from diplo_mod_1.preprocessing.cleaner import _VINTAGE_RE
from diplo_mod_1.preprocessing.config import FeatureEngineerConfig


class FeatureEngineer:
    """Derives engineered features from cleaned wine review data.

    Expects the output of ``DataCleaner.clean()`` as input.  Fits on the
    full dataset before the train/val/test split so that ``median_vintage_``
    is computed from the complete distribution — no leakage.

    Usage::

        engineer = FeatureEngineer()
        engineer.fit(raw_df)                  # learn median_vintage_ from full data
        featured = engineer.transform(cleaned) # apply to cleaned DataFrame
    """

    def __init__(self, config: FeatureEngineerConfig | None = None) -> None:
        self.config = config or FeatureEngineerConfig()
        self.median_vintage_: float | None = None

    def fit(self, df: pd.DataFrame) -> "FeatureEngineer":
        """Learn the median vintage year from the raw dataset."""
        vintage_year = pd.to_numeric(
            df["title"].astype(str).str.extract(_VINTAGE_RE, expand=False),
            errors="coerce",
        )
        self.median_vintage_ = float(vintage_year.median())
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add engineered columns. Expects output of DataCleaner.clean()."""
        if self.median_vintage_ is None:
            raise RuntimeError("Call fit() before transform().")
        out = df.copy()
        out["log_price"] = np.log1p(out["price"])
        out["is_luxury"] = (out["price"] > self.config.luxury_threshold).astype(int)
        out["is_us"] = (out["country"] == "US").astype(int)
        out["has_designation"] = out["designation"].notna().astype(int)
        out["description_length"] = out["description"].astype(str).str.len()
        out["vintage_year"] = out["vintage_year"].fillna(self.median_vintage_)
        out["wine_age"] = self.config.ref_year - out["vintage_year"]
        return out

    def fit_transform(self, df: pd.DataFrame, cleaned: pd.DataFrame) -> pd.DataFrame:
        """Fit on raw data and transform cleaned data in one step."""
        return self.fit(df).transform(cleaned)
