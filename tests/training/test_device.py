"""Tests for device auto-detection (XGBoost's CUDA probe, PyTorch's CUDA/MPS/CPU)."""

from diplo_mod_1.training.device import detect_torch_device, detect_xgboost_device


def test_detect_xgboost_device_returns_cpu_or_cuda() -> None:
    assert detect_xgboost_device() in {"cpu", "cuda"}


def test_detect_torch_device_returns_valid_string() -> None:
    assert detect_torch_device() in {"cpu", "cuda", "mps"}
