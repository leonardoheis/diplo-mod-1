"""Tests for DatasetExporter static methods."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.preprocessing import StandardScaler

from diplo_mod_1.preprocessing.config import TextEncoderConfig
from diplo_mod_1.preprocessing.encoders import TabularEncoder, TextEncoder
from diplo_mod_1.preprocessing.exporter import DatasetExporter


@pytest.fixture(scope="module")
def encoder_artifacts(featured_df: pd.DataFrame) -> dict:
    """Fitted encoder, split arrays, and text matrices for exporter tests."""
    train_df = featured_df.iloc[:40]
    val_df = featured_df.iloc[40:50]
    test_df = featured_df.iloc[50:]

    y = {
        "train": train_df["points"].to_numpy(dtype=np.float32),
        "val": val_df["points"].to_numpy(dtype=np.float32),
        "test": test_df["points"].to_numpy(dtype=np.float32),
    }

    encoder = TabularEncoder()
    train_res = encoder.fit_transform(train_df, y["train"])
    val_res = encoder.transform(val_df)
    test_res = encoder.transform(test_df)
    x_tab = {"train": train_res.X, "val": val_res.X, "test": test_res.X}

    cont_idx = train_res.column_groups["continuous"]
    scaler = StandardScaler()
    scaler.fit(x_tab["train"][:, cont_idx])

    text_enc = TextEncoder(TextEncoderConfig(max_features=50, min_df=1))
    text_enc.fit(train_df)
    x_txt: dict[str, sparse.csr_matrix] = {
        "train": text_enc.transform(train_df),
        "val": text_enc.transform(val_df),
        "test": text_enc.transform(test_df),
    }

    return {
        "x_tab": x_tab,
        "x_txt": x_txt,
        "y": y,
        "encoder": encoder,
        "feature_names": train_res.feature_names,
        "column_groups": train_res.column_groups,
        "scaler": scaler,
        "text_enc": text_enc,
    }


# ── export_xgboost ────────────────────────────────────────────────────────────


def test_export_xgboost_creates_split_arrays(tmp_path: Path, encoder_artifacts: dict) -> None:
    ea = encoder_artifacts
    DatasetExporter.export_xgboost(
        tmp_path, ea["x_tab"], ea["y"], ea["encoder"], ea["feature_names"], ea["column_groups"]
    )
    for split in ("train", "val", "test"):
        assert (tmp_path / f"X_{split}.npy").exists()
        assert (tmp_path / f"y_{split}.npy").exists()


def test_export_xgboost_creates_metadata_files(tmp_path: Path, encoder_artifacts: dict) -> None:
    ea = encoder_artifacts
    DatasetExporter.export_xgboost(
        tmp_path, ea["x_tab"], ea["y"], ea["encoder"], ea["feature_names"], ea["column_groups"]
    )
    assert (tmp_path / "feature_names.json").exists()
    assert (tmp_path / "target_encoders.joblib").exists()
    assert (tmp_path / "ohe.joblib").exists()


def test_export_xgboost_feature_names_json_structure(
    tmp_path: Path, encoder_artifacts: dict
) -> None:
    ea = encoder_artifacts
    DatasetExporter.export_xgboost(
        tmp_path, ea["x_tab"], ea["y"], ea["encoder"], ea["feature_names"], ea["column_groups"]
    )
    meta = json.loads((tmp_path / "feature_names.json").read_text())
    assert "feature_names" in meta
    assert "column_groups" in meta
    assert meta["scaled"] is False


# ── export_nn ─────────────────────────────────────────────────────────────────


def test_export_nn_creates_tabular_and_text_arrays(tmp_path: Path, encoder_artifacts: dict) -> None:
    ea = encoder_artifacts
    DatasetExporter.export_nn(
        tmp_path,
        ea["x_tab"],
        ea["x_txt"],
        ea["y"],
        ea["feature_names"],
        ea["column_groups"],
        ea["scaler"],
        ea["text_enc"],
    )
    for split in ("train", "val", "test"):
        assert (tmp_path / f"X_tab_{split}.npy").exists()
        assert (tmp_path / f"X_txt_{split}.npz").exists()
        assert (tmp_path / f"y_{split}.npy").exists()


def test_export_nn_creates_artifact_files(tmp_path: Path, encoder_artifacts: dict) -> None:
    ea = encoder_artifacts
    DatasetExporter.export_nn(
        tmp_path,
        ea["x_tab"],
        ea["x_txt"],
        ea["y"],
        ea["feature_names"],
        ea["column_groups"],
        ea["scaler"],
        ea["text_enc"],
    )
    assert (tmp_path / "scaler.joblib").exists()
    assert (tmp_path / "tfidf_vectorizer.joblib").exists()


# ── write_manifest ────────────────────────────────────────────────────────────


def test_write_manifest_produces_valid_json(tmp_path: Path, encoder_artifacts: dict) -> None:
    ea = encoder_artifacts
    manifest_path = tmp_path / "dataset_manifest.json"
    DatasetExporter.write_manifest(
        manifest_path,
        tmp_path / "split_indices.npz",
        {k: v.shape for k, v in ea["x_tab"].items()},
        {k: v.shape for k, v in ea["x_tab"].items()},
        {"train": 100, "val": 25, "test": 30},
    )
    manifest = json.loads(manifest_path.read_text())
    for key in ("target", "random_state", "split", "xgboost", "nn"):
        assert key in manifest
