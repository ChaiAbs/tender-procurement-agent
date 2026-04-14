"""
config.py — Central configuration for the Tender Price Prediction Multi-Agent System.
All feature definitions, bucket boundaries, and model hyperparameters live here.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Features ───────────────────────────────────────────────────────────────────
# These are the 7 categorical features available BEFORE a contract is awarded.
PRE_AWARD_FEATURES = [
    "procurement_method",
    "disposition",
    "is_consultancy_services",
    "publisher_gov_type",
    "category_code",
    "parent_category_code",
    "publisher_cofog_level",
]

# Numeric pre-award features — passed through without one-hot encoding.
NUMERIC_FEATURES = ["duration_days"]

# Contract duration: cap at 10 years to filter data-entry errors in the export.
DURATION_CAP_DAYS = 3650

TARGET_COLUMN = "value"

# ── Bucket definitions ─────────────────────────────────────────────────────────
BUCKET_ORDER = ["Small", "Medium", "Large", "Very Large"]

BUCKET_RANGES = {
    "Small":      (0,          50_000),
    "Medium":     (50_000,    500_000),
    "Large":     (500_000,  5_000_000),
    "Very Large":(5_000_000, float("inf")),
}

SUBRANGE_DEFINITIONS = {
    "Small": [
        ("$0 – $10K",    0,          10_000),
        ("$10K – $20K",  10_000,     20_000),
        ("$20K – $35K",  20_000,     35_000),
        ("$35K – $50K",  35_000,     50_000),
    ],
    "Medium": [
        ("$50K – $150K",   50_000,   150_000),
        ("$150K – $250K", 150_000,   250_000),
        ("$250K – $375K", 250_000,   375_000),
        ("$375K – $500K", 375_000,   500_000),
    ],
    "Large": [
        ("$500K – $1.5M",   500_000, 1_500_000),
        ("$1.5M – $2.75M", 1_500_000,2_750_000),
        ("$2.75M – $4M",   2_750_000,4_000_000),
        ("$4M – $5M",      4_000_000,5_000_000),
    ],
    "Very Large": [
        ("$5M – $20M",     5_000_000,  20_000_000),
        ("$20M – $60M",   20_000_000,  60_000_000),
        ("$60M – $150M",  60_000_000, 150_000_000),
        (">$150M",        150_000_000, float("inf")),
    ],
}

# ── Model hyperparameters ──────────────────────────────────────────────────────
XGBOOST_REGRESSOR_PARAMS = {
    "n_estimators":    200,
    "max_depth":       6,
    "learning_rate":   0.05,
    "subsample":       0.8,
    "colsample_bytree":0.8,
    "random_state":    42,
    "nthread":         1,
}

XGBOOST_CLASSIFIER_PARAMS = {
    "n_estimators":    200,
    "max_depth":       6,
    "learning_rate":   0.05,
    "subsample":       0.8,
    "colsample_bytree":0.8,
    "random_state":    42,
    "eval_metric":     "mlogloss",
    "nthread":         1,
}

# ── Pipeline thresholds ────────────────────────────────────────────────────────
# ML bucket_probability below this → treat as novel contract → route to LLM
NOVEL_CONTRACT_PROB_THRESHOLD = 0.35

# Confidence label shown to the user
CONFIDENCE_THRESHOLDS = {
    "High":   {"bucket": "Small",      "stage1_prob": 0.70},
    "Medium": {"bucket": "Medium",     "stage1_prob": 0.50},
    "Low":    {"bucket": "Large",      "stage1_prob": 0.30},
    "Very Low":{"bucket": "Very Large","stage1_prob": 0.10},
}

RANDOM_BASELINE_HIT_RATE = 1 / 16  # 6.25%

# ── LLM settings ───────────────────────────────────────────────────────────────
LLM_MODEL       = os.environ.get("LLM_MODEL", "claude-sonnet-4-6")
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.1"))

# ── RAG settings ───────────────────────────────────────────────────────────────
RAG_DIR             = os.path.join(BASE_DIR, "rag_store")
RAG_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RAG_COLLECTION_NAME = "tenders"
RAG_INDEX_SAMPLE    = 50_000   # rows to index (None = all)
RAG_N_RESULTS       = 5        # similar contracts to retrieve per query

# ── Prompts directory ──────────────────────────────────────────────────────────
PROMPTS_DIR = os.path.join(BASE_DIR, "prompts")
