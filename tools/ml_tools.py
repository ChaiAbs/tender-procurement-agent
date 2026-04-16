"""
tools/ml_tools.py — LangChain tools that wrap the existing ML pipeline.

These tools load the trained XGBoost models on first call and cache them
for the lifetime of the process (lazy singleton pattern).

Both tools accept a JSON string of the 7 pre-award contract fields and
return a JSON string so they work cleanly with LLM tool-calling.
"""

import json
import os

import pandas as pd
from langchain_core.tools import tool

from pipeline.data_processor import DataProcessor
from pipeline.regressor      import Regressor

# ── Lazy-loaded singletons ─────────────────────────────────────────────────────

_data_processor: DataProcessor | None = None
_regressor: Regressor | None = None


def _get_data_processor() -> DataProcessor:
    global _data_processor
    if _data_processor is None:
        step = DataProcessor(verbose=False)
        step._load_ohe_schema()
        _data_processor = step
    return _data_processor


def _get_regressor() -> Regressor:
    global _regressor
    if _regressor is None:
        _regressor = Regressor(verbose=False)
    return _regressor


def _preprocess(contract: dict) -> pd.DataFrame:
    """Preprocess a contract dict into an ML feature vector."""
    return _get_data_processor().preprocess_single(contract)


# ── LangChain tools ────────────────────────────────────────────────────────────

@tool
def predict_regression(contract_json: str) -> str:
    """
    Run the XGBoost regression model to get a dollar-value point estimate.

    Args:
        contract_json: JSON string containing any of the 7 pre-award fields:
            procurement_method, disposition, is_consultancy_services,
            publisher_gov_type, category_code, parent_category_code,
            publisher_cofog_level

    Returns:
        JSON string with keys:
            point_estimate_aud  — predicted contract value in AUD
            ci_low_90_aud       — lower bound of 90% confidence interval
            ci_high_90_aud      — upper bound of 90% confidence interval
            log_prediction      — raw log-space prediction (diagnostic)
    """
    try:
        contract = json.loads(contract_json)
        X = _preprocess(contract)
        result = _get_regressor().predict(X)
        return json.dumps(result)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


