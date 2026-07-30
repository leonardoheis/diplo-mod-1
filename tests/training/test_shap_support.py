"""Tests for the shap/XGBoost 3.x base_score compatibility shim."""

from diplo_mod_1.training.shap_support import _fix_base_score, patch_shap_xgboost_base_score


def test_fix_base_score_strips_brackets() -> None:
    param = {"base_score": "[8.844811E1]"}
    _fix_base_score(param)
    assert param["base_score"] == "8.844811E1"


def test_fix_base_score_leaves_plain_value_untouched() -> None:
    param = {"base_score": "88.44811"}
    _fix_base_score(param)
    assert param["base_score"] == "88.44811"


def test_patch_wraps_decode_ubjson_buffer() -> None:
    import shap.explainers._tree as shap_tree

    original = shap_tree.decode_ubjson_buffer
    try:
        patch_shap_xgboost_base_score.cache_clear()
        patch_shap_xgboost_base_score()
        assert shap_tree.decode_ubjson_buffer is not original
    finally:
        shap_tree.decode_ubjson_buffer = original
        patch_shap_xgboost_base_score.cache_clear()
