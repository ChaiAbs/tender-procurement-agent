"""
tools/ml_runner.py — Subprocess entry point for ML predictions.

Isolates XGBoost/OpenMP from LangChain's async context (macOS ARM segfault).

Usage:
    python tools/ml_runner.py '<contract_json>'
    python tools/ml_runner.py '<contract_json>' xgboost
    python tools/ml_runner.py '<contract_json>' lightgbm

Prints a single JSON line with regression and validation results.
Price range is derived from KNN similar contracts in the LangGraph pipeline.
"""

import json
import sys
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_evaluation.evaluator import get_active_model
from pipeline.data_processor import DataProcessor
from pipeline.regressor      import Regressor
from pipeline.validator      import Validator

contract_json = sys.argv[1]
contract      = json.loads(contract_json)

# Resolve model key: explicit CLI arg → active model file → default
model_key = sys.argv[2] if len(sys.argv) > 2 else get_active_model()

# Run regression prediction directly (avoid the LangChain tool layer in subprocess)
dp = DataProcessor(verbose=False)
dp._load_ohe_schema()
X_infer = dp.preprocess_single(contract)

regressor  = Regressor(model_key=model_key, verbose=False)
regression = regressor.predict(X_infer)

validator_context = {
    "contract":              contract,
    "regression_prediction": regression,
}
Validator(verbose=False).run(validator_context)
validation = validator_context.get("validation", {})

print(json.dumps({
    "regression": regression,
    "validation": validation,
}))
