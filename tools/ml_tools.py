"""
tools/ml_tools.py — LangChain tools that wrap the ML pipeline.

Uses the active model (models/active_model.txt); falls back to XGBoost if
no selection has been made. The Regressor singleton is cache-busted whenever
the active model key changes.
"""

from __future__ import annotations

import json
import os

import pandas as pd
from langchain_core.tools import tool

from pipeline.data_processor import DataProcessor
from pipeline.regressor      import Regressor

# ── Lazy-loaded singletons ─────────────────────────────────────────────────────

_data_processor: DataProcessor | None = None
_regressor: Regressor | None          = None
_regressor_key: str | None            = None   # tracks which key the cached regressor was built for


def _get_data_processor() -> DataProcessor:
    global _data_processor
    if _data_processor is None:
        step = DataProcessor(verbose=False)
        step._load_ohe_schema()
        _data_processor = step
    return _data_processor


def _get_regressor(model_key: str | None = None) -> Regressor:
    """Return a cached Regressor, rebuilding if model_key changed."""
    global _regressor, _regressor_key

    if model_key is None:
        from ml_evaluation.evaluator import get_active_model
        model_key = get_active_model()

    if _regressor is None or _regressor_key != model_key:
        _regressor     = Regressor(model_key=model_key, verbose=False)
        _regressor_key = model_key

    return _regressor


def _preprocess(contract: dict) -> pd.DataFrame:
    return _get_data_processor().preprocess_single(contract)


# ── LangChain tools ────────────────────────────────────────────────────────────

@tool
def predict_regression(contract_json: str) -> str:
    """
    Run the regression model to get a dollar-value point estimate.

    Args:
        contract_json: JSON string with the pre-award contract fields.

    Returns:
        JSON string with point_estimate_aud, ci_low_90_aud, ci_high_90_aud,
        log_prediction, and model_key.
    """
    try:
        contract = json.loads(contract_json)
        X = _preprocess(contract)
        result = _get_regressor().predict(X)
        return json.dumps(result)
    except Exception as exc:
        return json.dumps({"error": str(exc)})
