"""Device auto-detection for XGBoost (CUDA-or-CPU) and PyTorch (CUDA/MPS/CPU).

Kept in one module rather than split by library — "which accelerator is
available" is a single cross-cutting concern, not something that belongs
inside a tuner class named after one specific model type.
"""

from functools import lru_cache

import numpy as np
from xgboost import XGBRegressor


@lru_cache(maxsize=1)
def detect_xgboost_device() -> str:
    """CUDA if XGBoost can actually train on it here, else CPU.

    XGBoost has no Apple Silicon/MPS backend — CUDA is the only GPU path.
    Probes XGBoost directly rather than trusting ``torch.cuda.is_available()``:
    XGBoost bundles its own CUDA runtime independent of torch's, so a
    CPU-only torch build says nothing about whether XGBoost's CUDA works.
    Cached — the throwaway fit only runs once per process.
    """
    try:
        XGBRegressor(tree_method="hist", device="cuda", n_estimators=1).fit(
            np.zeros((2, 1), dtype=np.float32), np.zeros(2, dtype=np.float32)
        )
        device = "cuda"
    except Exception:
        device = "cpu"

    print(f"XGBoost device: {device}")
    return device


@lru_cache(maxsize=1)
def detect_torch_device() -> str:
    """CUDA, then Apple Silicon MPS, then CPU — whichever this machine has.

    Unlike XGBoost, torch's own ``torch.cuda.is_available()``/
    ``torch.backends.mps.is_available()`` are trustworthy here — there's no
    separate bundled-runtime mismatch to work around, torch is the only
    library involved.
    """
    import torch

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"PyTorch device: {device}")
    return device
