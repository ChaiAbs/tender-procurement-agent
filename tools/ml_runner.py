"""
tools/ml_runner.py — Subprocess entry point for ML predictions + validation.

Run by graph.predict() in a subprocess to isolate XGBoost/OpenMP from
LangChain's async context, which causes a segfault on macOS ARM otherwise.

Usage (internal):
    python tools/ml_runner.py '<contract_json>'

Prints a single JSON line with regression, bucket, and validation results.
"""

import json
import sys
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.ml_tools import predict_regression
from pipeline.bucket_classifier import predict_from_regression
from pipeline.validator import Validator

contract_json = sys.argv[1]
contract      = json.loads(contract_json)

regression = json.loads(predict_regression.invoke({"contract_json": contract_json}))
bucket     = predict_from_regression(regression.get("point_estimate_aud", 0))

# Run deterministic validation — no LLM needed
validator_context = {
    "contract":               contract,
    "regression_prediction":  regression,
    "bucket_prediction":      bucket,
}
Validator(verbose=False).run(validator_context)
validation = validator_context.get("validation", {})

print(json.dumps({
    "regression": regression,
    "bucket":     bucket,
    "validation": validation,
}))
