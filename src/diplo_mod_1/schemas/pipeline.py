"""PreprocessingResult — typed summary returned by PreprocessingPipeline.run()."""

from __future__ import annotations

from pydantic import BaseModel, Field

from diplo_mod_1.constants import CERO


class PreprocessingResult(BaseModel):
    """Summary produced at the end of a full preprocessing run.

    Replaces the plain ``dict[str, Any]`` previously returned by
    ``PreprocessingPipeline.run()``, making each field accessible by name.

    Usage::

        result = pipeline.run(raw_csv, interim_dir, processed_dir)
        print(result.xgb_features)
        print(result.split_sizes["train"])
    """

    n_rows: int = Field(gt=CERO)
    xgb_features: int = Field(gt=CERO)
    nn_tab_features: int = Field(gt=CERO)
    txt_features: int = Field(gt=CERO)
    split_sizes: dict[str, int] = Field(
        description="Row counts per split keyed by 'train', 'val', 'test'."
    )
