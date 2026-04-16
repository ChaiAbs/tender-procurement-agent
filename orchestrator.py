"""
orchestrator.py — Master Orchestrator for the Tender Price Prediction Pipeline

The Orchestrator coordinates all pipeline steps:

  ┌─────────────────────────────────────────────────────────────────┐
  │                       ORCHESTRATOR                              │
  │                                                                 │
  │  DataProcessor → Regressor                      (training)     │
  │                                                                 │
  │  DataProcessor.preprocess_single()                              │
  │    → Regressor.predict()                                        │
  │    → predict_from_regression()  (bucket derived deterministically)
  │    → Validator.run()                                            │
  │    → Presenter.run()                            (inference)     │
  └─────────────────────────────────────────────────────────────────┘

Usage:
  # Train (once — saves models to ./models/)
  python orchestrator.py --mode train --data tenders_export.xlsx

  # Predict a single contract
  python orchestrator.py --mode predict \
      --procurement_method "Open tender" \
      --disposition "Contract Notice" \
      --is_consultancy_services "No" \
      --publisher_gov_type "FED" \
      --category_code "81111500" \
      --parent_category_code "81000000" \
      --publisher_cofog_level "2"

  # Evaluate end-to-end on the dataset
  python orchestrator.py --mode evaluate --data tenders_export.xlsx
"""

import argparse
import json
import os
import pickle
import sys

from pipeline import (
    DataProcessor,
    Regressor,
    Validator,
    Presenter,
)
from pipeline.bucket_classifier import predict_from_regression
from config import MODELS_DIR, PRE_AWARD_FEATURES

SCHEMA_PATH = os.path.join(MODELS_DIR, "feature_schema.pkl")


class Orchestrator:
    """Coordinates all pipeline steps for training and inference."""

    def __init__(self, verbose: bool = True):
        self.verbose        = verbose
        self.data_processor = DataProcessor(verbose=verbose)
        self.regressor      = Regressor(verbose=verbose)
        self.validator      = Validator(verbose=verbose)
        self.presenter      = Presenter(verbose=verbose)

    # ── Training ───────────────────────────────────────────────────────────────

    def train(self, data_path: str) -> dict:
        """
        Full training pipeline.
        Runs: DataProcessor → Regressor → BucketClassifier
        Returns the final context dict (metrics, etc.).
        """
        self._banner("TRAINING MODE")
        context = {"data_path": data_path}

        # 1. Data preparation
        self._section("Step 1 / 2 — Data Processor")
        context = self.data_processor.run(context)
        self._save_schema(context["feature_schema"])

        # 2. Regression model
        self._section("Step 2 / 2 — Regressor")
        context = self.regressor.run(context)

        self._banner("TRAINING COMPLETE")
        self._print_train_summary(context)
        return context

    # ── Inference ──────────────────────────────────────────────────────────────

    def predict(self, contract: dict) -> dict:
        """
        Inference pipeline for a single contract.
        Runs: preprocess → Regression → Bucket → Validator → Presenter
        Returns the final context dict (report_dict, report_text, etc.).
        """
        self._banner("PREDICTION MODE")

        # Load saved schema into data_processor
        self._load_schema()

        # 1. Preprocess the raw contract dict
        self._section("Step 1 / 4 — Data Processor  (preprocess)")
        X_infer = self.data_processor.preprocess_single(contract)
        self.data_processor.log(f"Feature vector shape: {X_infer.shape}")

        # 2. Regression point estimate
        self._section("Step 2 / 4 — Regressor  (predict)")
        reg_pred = self.regressor.predict(X_infer)
        self.regressor.log(
            f"Point estimate: ${reg_pred['point_estimate_aud']:,.0f}  "
            f"| CI: ${reg_pred['ci_low_90_aud']:,.0f} – "
            f"${reg_pred['ci_high_90_aud']:,.0f}"
        )

        # 3. Bucket + sub-range (derived from regression point estimate)
        self._section("Step 3 / 4 — Bucket  (derived from regression)")
        bucket_pred = predict_from_regression(reg_pred["point_estimate_aud"])
        self.regressor.log(
            f"Bucket: {bucket_pred['predicted_bucket']} "
            f"| Sub-range: {bucket_pred['predicted_subrange']}"
        )

        # 4. Validation + confidence
        self._section("Step 4a / 4 — Validator")
        context = {
            "contract":               contract,
            "regression_prediction":  reg_pred,
            "bucket_prediction":      bucket_pred,
        }
        context = self.validator.run(context)

        # 5. Present
        self._section("Step 4b / 4 — Presenter")
        context = self.presenter.run(context)

        return context

    # ── Evaluation ─────────────────────────────────────────────────────────────

    def evaluate(self, data_path: str, n_samples: int = 500) -> None:
        """
        Quick evaluation: sample n_samples rows from the dataset,
        run inference on each, and report summary statistics.
        """
        import pandas as pd
        import numpy  as np

        self._banner("EVALUATION MODE")
        self._load_schema()

        ext = os.path.splitext(data_path)[1].lower()
        df  = pd.read_excel(data_path) if ext in (".xlsx", ".xls") else pd.read_csv(data_path)
        df  = df[df["value"].notna() & (pd.to_numeric(df["value"], errors="coerce") > 0)]
        df  = df.sample(min(n_samples, len(df)), random_state=42)

        hits = 0
        for _, row in df.iterrows():
            contract = {f: row.get(f, "unknown") for f in PRE_AWARD_FEATURES}
            try:
                ctx    = self.predict(contract)
                sr     = ctx["report_dict"]["subrange_prediction"]
                actual = float(row["value"])
                lo     = sr["low_aud"]
                hi     = sr["high_aud"] if sr["high_aud"] else float("inf")
                if lo <= actual < hi:
                    hits += 1
            except Exception:
                pass

        hit_rate = hits / len(df)
        print(f"\n{'='*50}")
        print(f"Evaluation over {len(df)} samples")
        print(f"Hit rate: {hit_rate:.1%}  (random baseline: 6.25%)")
        print(f"{'='*50}\n")

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _save_schema(self, schema: list) -> None:
        with open(SCHEMA_PATH, "wb") as f:
            pickle.dump(schema, f)

    def _load_schema(self) -> None:
        self.data_processor._load_ohe_schema()

    def _banner(self, text: str) -> None:
        if self.verbose:
            w = 58
            print(f"\n\033[1m\033[96m{'═' * w}\033[0m")
            print(f"\033[1m\033[96m  {text}\033[0m")
            print(f"\033[1m\033[96m{'═' * w}\033[0m\n")

    def _section(self, text: str) -> None:
        if self.verbose:
            print(f"\n\033[1m▶ {text}\033[0m")


    def _print_train_summary(self, context: dict) -> None:
        reg   = context.get("regression_metrics", {})
        stats = context.get("stats",              {})
        print("\n\033[1mTraining Summary\033[0m")
        print(f"  Dataset          : {stats.get('n_contracts', '?'):,} contracts")
        print(f"  Regression R²    : {reg.get('r2', '?')}")
        print(f"  Regression RMSE  : {reg.get('rmse_log', '?')} (log space)")
        print(f"  Bucket/sub-range : derived from regression point estimate")
        print()


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Tender Price Prediction — Multi-Agent Pipeline"
    )
    parser.add_argument(
        "--mode", choices=["train", "predict", "evaluate"],
        required=True, help="Pipeline mode"
    )
    parser.add_argument(
        "--data", type=str, default="tenders_export.xlsx",
        help="Path to tenders Excel/CSV file (required for train/evaluate)"
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress agent logs")

    # Contract fields for predict mode
    for feat in PRE_AWARD_FEATURES:
        parser.add_argument(f"--{feat}", type=str, default="unknown")

    args = parser.parse_args()
    orch = Orchestrator(verbose=not args.quiet)

    if args.mode == "train":
        if not os.path.exists(args.data):
            print(f"ERROR: Data file not found: {args.data}", file=sys.stderr)
            sys.exit(1)
        orch.train(args.data)

    elif args.mode == "predict":
        contract = {feat: getattr(args, feat) for feat in PRE_AWARD_FEATURES}
        ctx = orch.predict(contract)
        # Also dump structured dict for scripting
        if args.quiet:
            print(json.dumps(ctx["report_dict"], indent=2))

    elif args.mode == "evaluate":
        if not os.path.exists(args.data):
            print(f"ERROR: Data file not found: {args.data}", file=sys.stderr)
            sys.exit(1)
        orch.evaluate(args.data)


if __name__ == "__main__":
    main()
