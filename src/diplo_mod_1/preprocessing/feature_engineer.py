"""FeatureEngineer — derives new columns from a cleaned Wine Reviews DataFrame."""

import numpy as np
import pandas as pd

from diplo_mod_1.constants import VINTAGE_RE
from diplo_mod_1.preprocessing.columns import TASTING_KEYWORDS
from diplo_mod_1.preprocessing.config import FeatureEngineerArtifacts, FeatureEngineerConfig


class FeatureEngineer:
    """Derives engineered features from cleaned wine review data.

    Expects the output of ``DataCleaner.clean()`` as input.  Fits on the
    full dataset before the train/val/test split so that ``median_vintage_``
    is computed from the complete distribution — no leakage.

    Usage::

        engineer = FeatureEngineer()
        engineer.fit(raw_df)                   # learn median_vintage_ from full data
        featured = engineer.transform(cleaned)  # apply to cleaned DataFrame
    """

    def __init__(self, config: FeatureEngineerConfig | None = None) -> None:
        self.config = config or FeatureEngineerConfig()
        self.median_vintage_: float | None = None
        self.variety_avg_log_price_: dict[str, float] | None = None

    def fit(self, df: pd.DataFrame) -> "FeatureEngineer":
        """Learn the median vintage year and per-variety average log-price."""
        vintage_year = pd.to_numeric(
            df["title"].astype(str).str.extract(VINTAGE_RE, expand=False),
            errors="coerce",
        )
        self.median_vintage_ = float(vintage_year.median())

        log_price = np.log1p(df["price"])
        self.variety_avg_log_price_ = log_price.groupby(df["variety"]).mean().to_dict()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add engineered columns. Expects output of DataCleaner.clean()."""
        if self.median_vintage_ is None or self.variety_avg_log_price_ is None:
            raise RuntimeError("Call fit() before transform().")
        out = df.copy()
        out["log_price"] = np.log1p(out["price"])
        out["is_luxury"] = (out["price"] > self.config.luxury_threshold).astype(int)
        out["is_us"] = (out["country"] == "US").astype(int)
        out["has_designation"] = out["designation"].notna().astype(int)
        out["description_length"] = out["description"].astype(str).str.len()
        out["vintage_year"] = out["vintage_year"].fillna(self.median_vintage_)
        out["wine_age"] = self.config.ref_year - out["vintage_year"]

        variety_avg_log_price = out["variety"].map(self.variety_avg_log_price_)
        out["price_vs_variety"] = out["log_price"] - variety_avg_log_price.fillna(
            out["log_price"].mean()
        )

        description = out["description"].astype(str)
        for term in TASTING_KEYWORDS:
            out[f"has_{term}"] = description.str.contains(term, case=False, regex=False).astype(int)

        return out

    @property
    def artifacts(self) -> FeatureEngineerArtifacts:
        """Fitted values as a typed Pydantic model — serialisable to JSON."""
        if self.median_vintage_ is None or self.variety_avg_log_price_ is None:
            raise RuntimeError("Call fit() before accessing artifacts.")
        return FeatureEngineerArtifacts(
            luxury_threshold=self.config.luxury_threshold,
            ref_year=self.config.ref_year,
            median_vintage=self.median_vintage_,
            variety_avg_log_price=self.variety_avg_log_price_,
        )
