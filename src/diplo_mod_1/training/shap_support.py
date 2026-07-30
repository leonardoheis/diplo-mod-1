"""Compatibility shim: shap can't yet parse XGBoost 3.x's base_score format.

ponytail: XGBoost 3.x serializes ``base_score`` as e.g. ``"[8.844811E1]"``
(array notation) instead of a plain float string; shap 0.49.1 (the newest
release compatible with this project's Python 3.10 — shap 0.50+ requires
Python 3.11+) does ``float(base_score)`` directly and crashes on that format.
Ceiling: remove this once shap ships a fix for XGBoost's new format, or this
project's Python floor moves to 3.11+ so a newer shap becomes installable.
"""

import re
from functools import lru_cache
from typing import Any


def _fix_base_score(learner_model_param: dict[str, Any]) -> None:
    """Strip XGBoost 3.x's array-notation ``base_score`` to a plain float string.

    Mutates ``learner_model_param`` in place; a no-op if the value doesn't
    look like the array-bracketed format.
    """
    base_score = learner_model_param.get("base_score")
    if isinstance(base_score, str):
        learner_model_param["base_score"] = re.sub(r"[\[\]]", "", base_score)


@lru_cache(maxsize=1)
def patch_shap_xgboost_base_score() -> None:
    """Wrap shap's UBJSON model loader to tolerate XGBoost 3.x's base_score.

    Idempotent (cached) — call this once before constructing any
    ``shap.TreeExplainer`` for an XGBoost model in this process.
    """
    import shap.explainers._tree as shap_tree

    original_decode = shap_tree.decode_ubjson_buffer

    def _decode_with_fixed_base_score(fd: Any) -> dict[str, Any]:
        result = original_decode(fd)
        _fix_base_score(result["learner"]["learner_model_param"])
        return result

    shap_tree.decode_ubjson_buffer = _decode_with_fixed_base_score
