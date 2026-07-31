"""Pydantic configuration for PyTorch NN hyperparameter tuning."""

from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from diplo_mod_1.constants import RANDOM_STATE


class NNSearchSpace(BaseModel):
    """Optuna suggestion ranges/choices for each NN hyperparameter.

    ``architecture`` encodes hidden-layer sizes as underscore-joined strings
    (e.g. ``"256_64"`` -> ``[256, 64]``) — a categorical choice among a few
    fixed depths, since tuning variable-depth architectures directly in
    Optuna is awkward.
    """

    architecture: list[str] = ["128_64", "256_64", "512_128_32"]
    dropout: tuple[float, float] = (0.1, 0.5)
    learning_rate: tuple[float, float] = (1e-4, 1e-2)
    weight_decay: tuple[float, float] = (1e-6, 1e-2)
    batch_size: list[int] = [64, 128, 256]
    activation: list[str] = ["relu"]

    @field_validator("dropout", "learning_rate", "weight_decay")
    @classmethod
    def bounds_ordered(cls, v: tuple[float, float]) -> tuple[float, float]:
        if v[0] > v[1]:
            raise ValueError(f"lower bound {v[0]} exceeds upper bound {v[1]}")
        return v


class NNTuningConfig(BaseModel):
    """Hyperparameters for NNTuner.

    Usage::

        # defaults
        config = NNTuningConfig()

        # load from JSON file
        config = NNTuningConfig.from_json(Path("configs/nn_training.json"))
    """

    n_trials: int = Field(default=30, ge=1)
    max_epochs: int = Field(default=100, ge=1)
    early_stopping_patience: int = Field(default=10, ge=1)
    random_state: int = RANDOM_STATE
    search_space: NNSearchSpace = Field(default_factory=NNSearchSpace)

    @classmethod
    def from_json(cls, path: Path) -> "NNTuningConfig":
        """Load config from a JSON file (same shape as ``model_dump()``)."""
        return cls.model_validate_json(path.read_text(encoding="utf-8"))
