"""
train_models.py — Train and evaluate all registered ML models.

Usage:
    python train_models.py                        # train all, set best R² as active
    python train_models.py --set xgboost          # train all, force a specific active model
    python train_models.py --model xgboost        # train only one model
    python train_models.py --list                 # print registry without training

Saves trained models to models/ and writes models/model_comparison.json.
"""

import argparse
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

from ml_evaluation.evaluator import MultiModelEvaluator, set_active_model, ordinal_encode_split
from ml_evaluation.model_registry import MODEL_REGISTRY
from pipeline.data_processor import DataProcessor

DATA_PATH = "tenders_export.xlsx"


def _print_table(df: pd.DataFrame) -> None:
    ok = df[df.get("status", pd.Series(["ok"] * len(df))) == "ok"].copy()
    if ok.empty:
        print("No successful results to display.")
        return

    ok = ok.sort_values("r2", ascending=False)
    cols = ["display_name", "r2", "rmse_log", "mae_log", "mae_dollar", "within_50pct", "train_time_s"]
    cols = [c for c in cols if c in ok.columns]

    header = f"{'Model':<26} {'R²':>7} {'RMSE(log)':>10} {'MAE(log)':>9} {'MAE($)':>12} {'≤50%':>7} {'Time(s)':>8}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for _, row in ok.iterrows():
        mae_d = f"${row['mae_dollar']:>10,.0f}" if "mae_dollar" in row else ""
        print(
            f"{row['display_name']:<26} "
            f"{row['r2']:>7.4f} "
            f"{row['rmse_log']:>10.4f} "
            f"{row['mae_log']:>9.4f} "
            f"{mae_d:>12} "
            f"{row.get('within_50pct', 0):>6.1f}% "
            f"{row.get('train_time_s', 0):>7.1f}s"
        )
    print("=" * len(header))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate ML models for tender price prediction.")
    parser.add_argument("--list",  action="store_true", help="List registered models and exit")
    parser.add_argument("--model", metavar="KEY",       help="Train only this model key")
    parser.add_argument("--set",   metavar="KEY",       help="Set this model as active after training")
    args = parser.parse_args()

    if args.list:
        print(f"\n{'Key':<16} {'Display Name':<28} {'Categorical':<14} {'Scaler'}")
        print("-" * 68)
        for key, spec in MODEL_REGISTRY.items():
            print(
                f"{key:<16} {spec['display_name']:<28} "
                f"{'native' if spec['native_categorical'] else 'ordinal':<14} "
                f"{'yes' if spec['needs_scaler'] else 'no'}"
            )
        return

    # ── Load and preprocess data ───────────────────────────────────────────────
    print(f"Loading data from {DATA_PATH} …")
    dp = DataProcessor(verbose=True)
    context = dp.run({"data_path": DATA_PATH})

    X_train, X_test = context["X_train"], context["X_test"]
    y_train, y_test = context["y_train"], context["y_test"]
    print(f"Train={len(X_train):,}  Test={len(X_test):,}  Features={X_train.shape[1]}\n")

    # ── Evaluate ───────────────────────────────────────────────────────────────
    evaluator = MultiModelEvaluator(verbose=True)

    if args.model:
        if args.model not in MODEL_REGISTRY:
            print(f"Unknown model key '{args.model}'. Run --list to see options.")
            sys.exit(1)
        # Evaluate only the requested model
        X_tr_enc, X_te_enc, encoder, cat_cols = ordinal_encode_split(X_train, X_test)
        spec = MODEL_REGISTRY[args.model]
        row = evaluator._train_evaluate_one(
            args.model, spec, X_train, X_test,
            X_tr_enc, X_te_enc, encoder, cat_cols, y_train, y_test
        )
        df = pd.DataFrame([row])
        print(f"\nTrained {spec['display_name']}:")
        _print_table(df)

        # Merge into existing comparison JSON so the UI reflects the update
        existing = evaluator.load_comparison()
        merged   = [r for r in existing if r.get("model_key") != args.model] + [row]
        import json
        from config import MODELS_DIR
        with open(os.path.join(MODELS_DIR, "model_comparison.json"), "w") as f:
            json.dump(merged, f, indent=2)
    else:
        df = evaluator.evaluate_all(X_train, X_test, y_train, y_test)
        _print_table(df)

    # ── Set active model ───────────────────────────────────────────────────────
    if args.set:
        set_active_model(args.set)
        print(f"\nActive model set to: {args.set}")
    elif not args.model:
        ok = df[df.get("status", pd.Series(["ok"] * len(df))) == "ok"]
        if not ok.empty:
            best_key = ok.loc[ok["r2"].idxmax(), "model_key"]
            set_active_model(best_key)
            print(f"\nBest model by R²: {best_key}  — set as active.")


if __name__ == "__main__":
    main()
