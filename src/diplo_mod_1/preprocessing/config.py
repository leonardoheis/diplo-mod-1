"""Pydantic v2 configuration and artifact models for the preprocessing pipeline."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator

from diplo_mod_1.constants import LUXURY_THRESHOLD, RANDOM_STATE, REF_YEAR


class DataCleanerConfig(BaseModel):
    """Hyperparameters for DataCleaner."""


class FeatureEngineerConfig(BaseModel):
    """Hyperparameters for FeatureEngineer."""

    luxury_threshold: float = LUXURY_THRESHOLD
    ref_year: int = REF_YEAR

    @field_validator("luxury_threshold")
    @classmethod
    def must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("luxury_threshold must be positive")
        return v

    @field_validator("ref_year")
    @classmethod
    def valid_year(cls, v: int) -> int:
        if not (1900 <= v <= 2100):
            raise ValueError("ref_year must be between 1900 and 2100")
        return v


class DataSplitterConfig(BaseModel):
    """Hyperparameters for DataSplitter."""

    random_state: int = RANDOM_STATE
    test_size: float = 0.2
    val_size: float = 0.2

    @model_validator(mode="after")
    def valid_sizes(self) -> "DataSplitterConfig":
        if not (0 < self.test_size < 1):
            raise ValueError("test_size must be in (0, 1)")
        if not (0 < self.val_size < 1):
            raise ValueError("val_size must be in (0, 1)")
        if self.test_size + self.val_size >= 1:
            raise ValueError("test_size + val_size must be < 1")
        return self


class TabularEncoderConfig(BaseModel):
    """Hyperparameters for TabularEncoder."""

    random_state: int = RANDOM_STATE
    target_encoder_cv: int = 5

    @field_validator("target_encoder_cv")
    @classmethod
    def cv_at_least_two(cls, v: int) -> int:
        if v < 2:
            raise ValueError("target_encoder_cv must be >= 2")
        return v


class TextEncoderConfig(BaseModel):
    """Hyperparameters for TextEncoder."""

    max_features: int = 2000
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 5

    @field_validator("max_features", "min_df")
    @classmethod
    def must_be_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("must be >= 1")
        return v

    @field_validator("ngram_range")
    @classmethod
    def valid_ngram_range(cls, v: tuple[int, int]) -> tuple[int, int]:
        if v[0] < 1:
            raise ValueError("ngram_range values must be >= 1")
        if v[0] > v[1]:
            raise ValueError("ngram_range[0] must be <= ngram_range[1]")
        return v


class PipelineConfig(BaseModel):
    """Top-level config for PreprocessingPipeline — composes all step configs.

    Usage::

        # defaults — identical to calling PreprocessingPipeline()
        config = PipelineConfig()

        # override one step
        config = PipelineConfig(text_encoder=TextEncoderConfig(max_features=5000))

        # load from JSON file
        config = PipelineConfig.model_validate_json(path.read_text())
    """

    cleaner: DataCleanerConfig = Field(default_factory=DataCleanerConfig)
    feature_engineer: FeatureEngineerConfig = Field(default_factory=FeatureEngineerConfig)
    splitter: DataSplitterConfig = Field(default_factory=DataSplitterConfig)
    encoder: TabularEncoderConfig = Field(default_factory=TabularEncoderConfig)
    text_encoder: TextEncoderConfig = Field(default_factory=TextEncoderConfig)


class CleaningArtifacts(BaseModel):
    """Fitted values produced by DataCleaner.fit() — serialisable to JSON."""

    global_price_median: float
    random_state: int = RANDOM_STATE
