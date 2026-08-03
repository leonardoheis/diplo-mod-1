"""Disk I/O: persist processed datasets, fitted artifacts, and manifest."""

import json
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse
from sklearn.preprocessing import StandardScaler

from diplo_mod_1.constants import RANDOM_STATE
from diplo_mod_1.preprocessing.encoders import TabularEncoder, TextEncoder


class DatasetExporter:
    """Writes processed splits, fitted encoders, and a dataset manifest to disk.

    All methods are static — the class groups I/O concerns and keeps import
    paths explicit without carrying state.

    Usage::

        DatasetExporter.export_xgboost(xgb_dir, x_splits, y_splits, encoder, names, groups)
        DatasetExporter.export_nn(nn_dir, x_tab, x_txt, y_splits, names, groups, scaler, text_enc)
        DatasetExporter.write_manifest(...)
    """

    @staticmethod
    def export_xgboost(
        out_dir: Path,
        x_splits: dict[str, np.ndarray],
        y_splits: dict[str, np.ndarray],
        encoder: TabularEncoder,
        feature_names: list[str],
        column_groups: dict[str, list[int]],
    ) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        for split in ("train", "val", "test"):
            np.save(out_dir / f"X_{split}.npy", x_splits[split])
            np.save(out_dir / f"y_{split}.npy", y_splits[split])
        meta = {"feature_names": feature_names, "column_groups": column_groups, "scaled": False}
        (out_dir / "feature_names.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        joblib.dump(encoder.target_encoders_, out_dir / "target_encoders.joblib")
        joblib.dump(encoder.ohe_, out_dir / "ohe.joblib")

    @staticmethod
    def export_nn(
        out_dir: Path,
        x_tab_splits: dict[str, np.ndarray],
        x_txt_splits: dict[str, sparse.csr_matrix],
        y_splits: dict[str, np.ndarray],
        feature_names: list[str],
        column_groups: dict[str, list[int]],
        scaler: StandardScaler,
        text_encoder: TextEncoder,
    ) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        for split in ("train", "val", "test"):
            np.save(out_dir / f"X_tab_{split}.npy", x_tab_splits[split])
            sparse.save_npz(out_dir / f"X_txt_{split}.npz", x_txt_splits[split])
            np.save(out_dir / f"y_{split}.npy", y_splits[split])
        meta = {
            "feature_names": feature_names,
            "column_groups": column_groups,
            "scaled": "continuous_only",
            "continuous_column_indices": column_groups["continuous"],
        }
        (out_dir / "feature_names.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        (out_dir / "continuous_column_indices.json").write_text(
            json.dumps(column_groups["continuous"], indent=2), encoding="utf-8"
        )
        joblib.dump(scaler, out_dir / "scaler.joblib")
        joblib.dump(text_encoder.vectorizer_, out_dir / "tfidf_vectorizer.joblib")

    @staticmethod
    def write_manifest(
        path: Path,
        split_path: Path,
        x_shapes: dict[str, tuple[int, ...]],
        nn_tab_shapes: dict[str, tuple[int, ...]],
        txt_nnz: dict[str, int],
    ) -> None:
        manifest = {
            "target": "points",
            "random_state": RANDOM_STATE,
            "split": str(split_path.name),
            "xgboost": {
                "path": "xgboost/",
                "scaled": False,
                "X_shape_train": list(x_shapes["train"]),
                "X_shape_val": list(x_shapes["val"]),
                "X_shape_test": list(x_shapes["test"]),
            },
            "nn": {
                "path": "nn/",
                "scaled": "continuous_only",
                "X_tab_shape_train": list(nn_tab_shapes["train"]),
                "X_tab_shape_val": list(nn_tab_shapes["val"]),
                "X_tab_shape_test": list(nn_tab_shapes["test"]),
                "X_txt_nnz_train": txt_nnz["train"],
                "X_txt_nnz_val": txt_nnz["val"],
                "X_txt_nnz_test": txt_nnz["test"],
            },
            "note": "Training not performed in notebook 02.",
        }
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
