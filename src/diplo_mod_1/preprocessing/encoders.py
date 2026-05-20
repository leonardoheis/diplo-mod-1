"""Tabular and text feature encoders for the Wine Reviews pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, TargetEncoder

from diplo_mod_1.preprocessing.columns import (
    BINARY_COLS,
    DIRECT_CONTINUOUS,
    FREQ_COLS,
    FREQ_FEATURE_COLS,
    OHE_COLS,
    TARGET_ENCODE_COLS,
    TARGET_FEATURE_COLS,
)
from diplo_mod_1.preprocessing.config import TabularEncoderConfig, TextEncoderConfig


def _assemble_matrix(
    df: pd.DataFrame,
    target_df: pd.DataFrame,
    freq_df: pd.DataFrame,
    ohe_arr: np.ndarray,
    ohe_names: list[str],
    *,
    include_strictness: bool = False,
    taster_strictness: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str], dict[str, list[int]]]:
    blocks: list[np.ndarray] = []
    names: list[str] = []

    blocks.append(df[DIRECT_CONTINUOUS].to_numpy(dtype=np.float32))
    names.extend(DIRECT_CONTINUOUS)

    blocks.append(df[BINARY_COLS].to_numpy(dtype=np.float32))
    names.extend(BINARY_COLS)

    blocks.append(target_df[TARGET_FEATURE_COLS].to_numpy(dtype=np.float32))
    names.extend(TARGET_FEATURE_COLS)

    if include_strictness and taster_strictness is not None:
        blocks.append(taster_strictness.reshape(-1, 1).astype(np.float32))
        names.append("taster_strictness")

    blocks.append(freq_df[FREQ_FEATURE_COLS].to_numpy(dtype=np.float32))
    names.extend(FREQ_FEATURE_COLS)

    blocks.append(ohe_arr.astype(np.float32))
    names.extend(ohe_names)

    x = np.hstack(blocks).astype(np.float32)

    n_cont = len(DIRECT_CONTINUOUS)
    n_bin = len(BINARY_COLS)
    n_target = len(TARGET_FEATURE_COLS)
    n_strict = 1 if include_strictness else 0
    n_freq = len(FREQ_FEATURE_COLS)

    # Column layout:
    #   [0 .. n_cont-1]               → DIRECT_CONTINUOUS
    #   [n_cont .. +n_bin-1]          → BINARY_COLS        (do NOT scale)
    #   [n_cont+n_bin .. +n_target-1] → TARGET_FEATURE_COLS
    #   [... +n_strict]               → taster_strictness (optional)
    #   [... +n_freq-1]               → FREQ_FEATURE_COLS
    #   rest                          → OHE
    binary_start = n_cont
    target_start = binary_start + n_bin
    ohe_start = target_start + n_target + n_strict + n_freq

    groups: dict[str, list[int]] = {
        "continuous": list(range(0, n_cont)) + list(range(target_start, ohe_start)),
        "binary": list(range(binary_start, binary_start + n_bin)),
        "ohe": list(range(ohe_start, len(names))),
    }
    return x, names, groups


class TabularEncoder:
    """Fits and applies tabular feature encoders (target, frequency, OHE).

    Follows sklearn's fit/transform convention — all encoders are fit on
    training data only.  ``fit_transform`` uses cross-validated target
    encoding on the training split to avoid leakage; ``transform`` uses the
    fitted (non-CV) encoders for val and test.

    Usage::

        encoder = TabularEncoder()
        x_train, names, groups = encoder.fit_transform(train_df, y_train)
        x_val,   _, _ = encoder.transform(val_df)
        x_test,  _, _ = encoder.transform(test_df)
    """

    def __init__(self, config: TabularEncoderConfig | None = None) -> None:
        self.config = config or TabularEncoderConfig()
        self.target_encoders_: dict[str, TargetEncoder] = {}
        self.freq_maps_: dict[str, dict[str, float]] = {}
        self.ohe_: OneHotEncoder | None = None
        self.global_points_mean_: float | None = None

    # ── private ───────────────────────────────────────────────────────────────

    def _fit_core(
        self,
        train_df: pd.DataFrame,
        y_train: np.ndarray,
    ) -> pd.DataFrame:
        """Fit all encoders; return CV-encoded target DataFrame for the train split."""
        self.global_points_mean_ = float(y_train.mean())

        target_parts: dict[str, np.ndarray] = {}
        for src, feat in TARGET_ENCODE_COLS:
            enc = TargetEncoder(
                cv=self.config.target_encoder_cv,
                random_state=self.config.random_state,
                target_type="continuous",
            )
            target_parts[feat] = enc.fit_transform(train_df[[src]], y_train).ravel()
            self.target_encoders_[src] = enc

        for src, feat in FREQ_COLS:
            counts = train_df[src].value_counts()
            self.freq_maps_[feat] = {k: float(np.log1p(v)) for k, v in counts.items()}

        self.ohe_ = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        if OHE_COLS:
            self.ohe_.fit(train_df[OHE_COLS])

        return pd.DataFrame(target_parts, index=train_df.index)

    def _encode_target(self, df: pd.DataFrame) -> pd.DataFrame:
        parts = {
            feat: self.target_encoders_[src].transform(df[[src]]).ravel()
            for src, feat in TARGET_ENCODE_COLS
        }
        return pd.DataFrame(parts, index=df.index)

    def _encode_freq(self, df: pd.DataFrame) -> pd.DataFrame:
        parts = {feat: df[src].map(self.freq_maps_[feat]).fillna(0.0) for src, feat in FREQ_COLS}
        return pd.DataFrame(parts, index=df.index)

    def _encode_ohe(self, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        if not OHE_COLS or self.ohe_ is None:
            return np.empty((len(df), 0), dtype=np.float32), []
        return self.ohe_.transform(df[OHE_COLS]), list(self.ohe_.get_feature_names_out(OHE_COLS))

    # ── public API ────────────────────────────────────────────────────────────

    def fit(self, train_df: pd.DataFrame, y_train: np.ndarray) -> "TabularEncoder":
        """Fit all encoders on the training split."""
        self._fit_core(train_df, y_train)
        return self

    def fit_transform(
        self,
        train_df: pd.DataFrame,
        y_train: np.ndarray,
        *,
        include_strictness: bool = False,
    ) -> tuple[np.ndarray, list[str], dict[str, list[int]]]:
        """Fit and return CV-encoded feature matrix for the training split."""
        train_target_df = self._fit_core(train_df, y_train)
        freq_df = self._encode_freq(train_df)
        ohe_arr, ohe_names = self._encode_ohe(train_df)
        taster_strictness = None
        if include_strictness and self.global_points_mean_ is not None:
            taster_strictness = (
                train_target_df["taster_avg_points"].to_numpy() - self.global_points_mean_
            )
        return _assemble_matrix(
            train_df,
            train_target_df,
            freq_df,
            ohe_arr,
            ohe_names,
            include_strictness=include_strictness,
            taster_strictness=taster_strictness,
        )

    def transform(
        self,
        df: pd.DataFrame,
        *,
        include_strictness: bool = False,
    ) -> tuple[np.ndarray, list[str], dict[str, list[int]]]:
        """Apply fitted encoders to any split (val, test, or non-CV train)."""
        if not self.target_encoders_:
            raise RuntimeError("Call fit() or fit_transform() before transform().")
        target_df = self._encode_target(df)
        freq_df = self._encode_freq(df)
        ohe_arr, ohe_names = self._encode_ohe(df)
        taster_strictness = None
        if include_strictness and self.global_points_mean_ is not None:
            taster_strictness = target_df["taster_avg_points"].to_numpy() - self.global_points_mean_
        return _assemble_matrix(
            df,
            target_df,
            freq_df,
            ohe_arr,
            ohe_names,
            include_strictness=include_strictness,
            taster_strictness=taster_strictness,
        )


class TextEncoder:
    """TF-IDF encoder for the wine description column.

    Usage::

        text_enc = TextEncoder()
        text_enc.fit(train_df)
        x_txt_train = text_enc.transform(train_df)
        x_txt_val   = text_enc.transform(val_df)
    """

    def __init__(self, config: TextEncoderConfig | None = None) -> None:
        self.config = config or TextEncoderConfig()
        self.vectorizer_: TfidfVectorizer | None = None

    def fit(self, train_df: pd.DataFrame) -> "TextEncoder":
        """Fit the TF-IDF vectorizer on training descriptions."""
        self.vectorizer_ = TfidfVectorizer(
            max_features=self.config.max_features,
            ngram_range=self.config.ngram_range,
            min_df=self.config.min_df,
            stop_words="english",
        )
        self.vectorizer_.fit(train_df["description"].astype(str))
        return self

    def transform(self, df: pd.DataFrame) -> sparse.csr_matrix:
        """Transform descriptions to a sparse TF-IDF matrix."""
        if self.vectorizer_ is None:
            raise RuntimeError("Call fit() before transform().")
        return self.vectorizer_.transform(df["description"].astype(str))
