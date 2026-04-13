"""
backtest.py — Measure ML model accuracy on the held-out test set.

Focuses on Large/Very Large and novel contracts where the model is weakest.
Loads models once in-process for speed (no subprocess per contract).

Metrics:
  - MAPE    : Mean Absolute Percentage Error
  - MdAPE   : Median Absolute Percentage Error (robust to outliers)
  - Hit rate : % where actual value falls within predicted sub-range
  - Bucket accuracy: % where predicted bucket matches actual bucket

Usage:
    python backtest.py --data tenders_export.xlsx --mode large
    python backtest.py --data tenders_export.xlsx --mode novel
    python backtest.py --data tenders_export.xlsx --mode both --output results.csv
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    PRE_AWARD_FEATURES,
    TARGET_COLUMN,
    BUCKET_RANGES,
    SUBRANGE_DEFINITIONS,
    NOVEL_CONTRACT_PROB_THRESHOLD,
)


# ── Bucket helpers ─────────────────────────────────────────────────────────────

def actual_bucket(value: float) -> str:
    for name, (lo, hi) in BUCKET_RANGES.items():
        if lo <= value < hi:
            return name
    return "Very Large"


def in_subrange(value: float, subrange_label: str, bucket: str) -> bool:
    for label, lo, hi in SUBRANGE_DEFINITIONS.get(bucket, []):
        if label == subrange_label:
            return lo <= value < hi
    return False


# ── ML — load once, run in-process ────────────────────────────────────────────

_dp = _reg = _bkt = None

def _load_ml_models():
    global _dp, _reg, _bkt
    if _dp is not None:
        return
    from pipeline.data_processor    import DataProcessor
    from pipeline.regressor         import Regressor
    from pipeline.bucket_classifier import BucketClassifier
    print("Loading ML models...", flush=True)
    _dp  = DataProcessor(verbose=False)
    _dp._load_freq_maps()
    _reg = Regressor(verbose=False)
    _bkt = BucketClassifier(verbose=False)
    print("ML models ready.\n", flush=True)


def run_ml(contract: dict) -> tuple[dict, dict]:
    X = _dp.preprocess_single(contract)
    return _reg.predict(X), _bkt.predict(X)


# ── Metrics ────────────────────────────────────────────────────────────────────

def ape(predicted, actual: float):
    if not predicted or predicted <= 0:
        return None
    return abs(predicted - actual) / actual * 100


def summarise(errors: list) -> dict:
    valid = [e for e in errors if e is not None]
    if not valid:
        return {"mape": None, "mdape": None, "n": 0}
    return {
        "mape":  round(float(np.mean(valid)),   1),
        "mdape": round(float(np.median(valid)), 1),
        "n":     len(valid),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Backtest ML model accuracy")
    parser.add_argument("--data",   required=True, help="Path to tenders_export.xlsx or CSV")
    parser.add_argument("--mode",   choices=["large", "novel", "both", "all"], default="large",
                        help="Which contracts to evaluate (default: large)")
    parser.add_argument("--output", help="Save detailed results to CSV")
    parser.add_argument("--seed",   type=int, default=99)
    args = parser.parse_args()

    # ── Load & split (same split as training) ──────────────────────────────────
    print(f"Loading {args.data} ...", flush=True)
    ext = Path(args.data).suffix.lower()
    df  = pd.read_excel(args.data) if ext in (".xlsx", ".xls") else pd.read_csv(args.data, low_memory=False)

    df = df[df[TARGET_COLUMN].notna()].copy()
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    df = df[df[TARGET_COLUMN] > 0].copy()

    for col in PRE_AWARD_FEATURES:
        if col not in df.columns:
            df[col] = "unknown"
        else:
            df[col] = df[col].fillna("unknown").astype(str).str.strip().str.lower()

    df["actual_bucket"] = df[TARGET_COLUMN].apply(actual_bucket)
    _, test_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Test set: {len(test_df):,} contracts\n", flush=True)

    # ── Run ML on all test contracts ───────────────────────────────────────────
    _load_ml_models()
    print(f"Running ML predictions on {len(test_df):,} contracts...", flush=True)

    rows = []
    for i, (_, row) in enumerate(test_df.iterrows()):
        contract = {f: row[f] for f in PRE_AWARD_FEATURES}
        actual   = float(row[TARGET_COLUMN])
        try:
            reg, bkt = run_ml(contract)
            ml_point = reg.get("point_estimate_aud")
            rows.append({
                "actual":         actual,
                "actual_bucket":  row["actual_bucket"],
                "ml_point":       ml_point,
                "ml_bucket":      bkt.get("predicted_bucket"),
                "bucket_prob":    bkt.get("bucket_probability", 1.0),
                "predicted_sub":  bkt.get("predicted_subrange"),
                "ml_ape":         ape(ml_point, actual),
                "ml_hit":         in_subrange(actual, bkt.get("predicted_subrange", ""), bkt.get("predicted_bucket", "")),
                "is_novel":       bkt.get("bucket_probability", 1.0) < NOVEL_CONTRACT_PROB_THRESHOLD,
            })
        except Exception:
            pass

        if (i + 1) % 1000 == 0:
            print(f"  {i + 1:,}/{len(test_df):,}", flush=True)

    results_df = pd.DataFrame(rows)
    print(f"Done: {len(results_df):,} predictions\n", flush=True)

    # ── Filter ─────────────────────────────────────────────────────────────────
    is_large  = results_df["actual_bucket"].isin(["Large", "Very Large"])
    is_novel  = results_df["is_novel"] & ~is_large

    if args.mode == "large":
        subsets = {"Large/Very Large": results_df[is_large]}
    elif args.mode == "novel":
        subsets = {"Novel": results_df[is_novel]}
    elif args.mode == "both":
        subsets = {
            "Large/Very Large": results_df[is_large],
            "Novel":            results_df[is_novel],
        }
    else:  # all
        subsets = {"All": results_df}

    # ── Print summary ──────────────────────────────────────────────────────────
    sep = "=" * 55
    print(sep)
    print("ML ACCURACY BACKTEST")
    print(sep)

    for label, sub in subsets.items():
        if len(sub) == 0:
            print(f"\n{label}: no contracts found")
            continue

        s        = summarise(sub["ml_ape"].tolist())
        hit_rate = sub["ml_hit"].mean() * 100
        bkt_acc  = (sub["ml_bucket"] == sub["actual_bucket"]).mean() * 100

        print(f"\n{label} (n={len(sub):,})")
        print(f"  MAPE            {s['mape']:.1f}%")
        print(f"  MdAPE           {s['mdape']:.1f}%")
        print(f"  Sub-range hit   {hit_rate:.1f}%")
        print(f"  Bucket accuracy {bkt_acc:.1f}%")

        # Break down by bucket
        print(f"  {'Bucket':<14} {'MdAPE':>7} {'Hit%':>7} {'n':>6}")
        print(f"  {'-'*38}")
        for bucket in ["Small", "Medium", "Large", "Very Large"]:
            b = sub[sub["actual_bucket"] == bucket]
            if not len(b):
                continue
            bs = summarise(b["ml_ape"].tolist())
            bh = b["ml_hit"].mean() * 100
            print(f"  {bucket:<14} {bs['mdape']:>6.1f}% {bh:>6.1f}% {len(b):>6,}")

    print(f"\n{sep}")

    if args.output:
        out = Path(args.output).resolve()
        results_df.to_csv(str(out), index=False)
        print(f"Detailed results: {out}")


if __name__ == "__main__":
    main()
