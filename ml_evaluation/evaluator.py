"""
evaluator.py — Multi-model evaluation for tender price prediction.

Trains all registered models on the same train/test split, computes a rich set
of metrics (log-space and dollar-space), saves each trained model to disk, and
writes a comparison JSON to models/model_comparison.json.

Usage:
    from ml_evaluation.evaluator import MultiModelEvaluator
    evaluator = MultiModelEvaluator()
    df = evaluator.evaluate_all(X_train, X_test, y_train, y_test)
    print(df.to_string(index=False))
"""

from __future__ import annotations

import json
import os
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from config import MODELS_DIR
from ml_evaluation.model_registry import DEFAULT_MODEL, MODEL_REGISTRY

COMPARISON_PATH   = os.path.join(MODELS_DIR, "model_comparison.json")
ACTIVE_MODEL_PATH = os.path.join(MODELS_DIR, "active_model.txt")


# ── Encoding helpers ───────────────────────────────────────────────────────────

def ordinal_encode_split(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, OrdinalEncoder, list[str]]:
    """
    Fit OrdinalEncoder on X_train categorical columns, transform both splits.
    Returns (X_tr_encoded, X_te_encoded, fitted_encoder, cat_col_names).
    All columns are cast to float so any sklearn estimator accepts them.
    """
    cat_cols = [c for c in X_train.columns if hasattr(X_train[c], "cat")]

    enc = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        dtype=float,
    )

    X_tr = X_train.copy()
    X_te = X_test.copy()

    if cat_cols:
        X_tr[cat_cols] = enc.fit_transform(X_train[cat_cols].astype(str))
        X_te[cat_cols] = enc.transform(X_test[cat_cols].astype(str))

    return X_tr.astype(float), X_te.astype(float), enc, cat_cols


def apply_ordinal_encoding(
    X: pd.DataFrame,
    encoder: OrdinalEncoder,
    cat_cols: list[str],
) -> pd.DataFrame:
    """Apply a fitted OrdinalEncoder to a single inference row."""
    X = X.copy()
    if cat_cols:
        X[cat_cols] = encoder.transform(X[cat_cols].astype(str))
    return X.astype(float)


# ── Active model helpers ───────────────────────────────────────────────────────

def get_active_model() -> str:
    if os.path.exists(ACTIVE_MODEL_PATH):
        with open(ACTIVE_MODEL_PATH) as f:
            key = f.read().strip()
        if key in MODEL_REGISTRY:
            return key
    return DEFAULT_MODEL


def set_active_model(key: str) -> None:
    if key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model key '{key}'. Valid keys: {list(MODEL_REGISTRY.keys())}"
        )
    with open(ACTIVE_MODEL_PATH, "w") as f:
        f.write(key)


# ── Evaluator ─────────────────────────────────────────────────────────────────

class MultiModelEvaluator:
    """Trains all registered models, computes metrics, saves artifacts."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results: list[dict] = []

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[Evaluator] {msg}", flush=True)

    def evaluate_all(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> pd.DataFrame:
        """Train every model, return a comparison DataFrame and save artifacts."""
        X_tr_enc, X_te_enc, encoder, cat_cols = ordinal_encode_split(X_train, X_test)

        self.results = []
        for key, spec in MODEL_REGISTRY.items():
            self._log(f"Training {spec['display_name']} …")
            try:
                row = self._train_evaluate_one(
                    key, spec,
                    X_train, X_test,
                    X_tr_enc, X_te_enc,
                    encoder, cat_cols,
                    y_train, y_test,
                )
                self.results.append(row)
                self._log(
                    f"  R²={row['r2']:.4f}  RMSE(log)={row['rmse_log']:.4f}"
                    f"  MAE($)={row['mae_dollar']:,.0f}  t={row['train_time_s']:.1f}s"
                )
            except Exception as exc:
                self._log(f"  FAILED: {exc}")
                self.results.append({
                    "model_key":    key,
                    "display_name": spec["display_name"],
                    "status":       "failed",
                    "error":        str(exc),
                })

        df = pd.DataFrame(self.results)
        self._save_comparison(df)
        return df

    def _train_evaluate_one(
        self,
        key: str,
        spec: dict,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        X_tr_enc: pd.DataFrame,
        X_te_enc: pd.DataFrame,
        encoder: OrdinalEncoder,
        cat_cols: list[str],
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> dict:
        use_native = spec["native_categorical"]
        X_tr = X_train if use_native else X_tr_enc.copy()
        X_te = X_test  if use_native else X_te_enc.copy()

        model = spec["factory"]()
        scaler: StandardScaler | None = None

        # StandardScaler for linear models
        if spec["needs_scaler"]:
            scaler = StandardScaler()
            X_tr_vals = scaler.fit_transform(X_tr)
            X_te_vals  = scaler.transform(X_te)
            X_tr = pd.DataFrame(X_tr_vals, columns=X_tr.columns)
            X_te = pd.DataFrame(X_te_vals, columns=X_te.columns)

        t0 = time.perf_counter()
        if spec["is_catboost"]:
            cat_indices = [list(X_tr.columns).index(c) for c in cat_cols]
            X_tr = X_tr.copy()
            X_te = X_te.copy()
            for c in cat_cols:
                X_tr[c] = X_tr[c].astype(int)
                X_te[c] = X_te[c].astype(int)
            model.fit(X_tr, y_train.values, cat_features=cat_indices)
        else:
            model.fit(X_tr, y_train.values)
        train_time = time.perf_counter() - t0

        y_pred = model.predict(X_te)

        # Log-space metrics
        r2       = round(float(r2_score(y_test.values, y_pred)), 4)
        rmse_log = round(float(np.sqrt(mean_squared_error(y_test.values, y_pred))), 4)
        mae_log  = round(float(mean_absolute_error(y_test.values, y_pred)), 4)

        # Dollar-space metrics (back-transform from log1p)
        y_true_d  = np.expm1(y_test.values)
        y_pred_d  = np.expm1(y_pred)
        rmse_dollar = round(float(np.sqrt(mean_squared_error(y_true_d, y_pred_d))), 2)
        mae_dollar  = round(float(mean_absolute_error(y_true_d, y_pred_d)), 2)

        # Practical accuracy: % predictions within 50% of actual value
        within_50 = round(
            float(np.mean(np.abs(y_pred_d - y_true_d) / (y_true_d + 1) < 0.5) * 100), 1
        )

        # Save model + encoder artifacts
        self._save_model_artifacts(key, model, encoder if not use_native else None,
                                   cat_cols, scaler, rmse_log)

        return {
            "model_key":     key,
            "display_name":  spec["display_name"],
            "r2":            r2,
            "rmse_log":      rmse_log,
            "mae_log":       mae_log,
            "rmse_dollar":   rmse_dollar,
            "mae_dollar":    mae_dollar,
            "within_50pct":  within_50,
            "train_time_s":  round(train_time, 2),
            "status":        "ok",
        }

    def _save_model_artifacts(
        self,
        key: str,
        model,
        encoder: OrdinalEncoder | None,
        cat_cols: list[str],
        scaler: StandardScaler | None,
        rmse: float,
    ) -> None:
        def _dump(obj, path: str) -> None:
            with open(path, "wb") as f:
                pickle.dump(obj, f)

        _dump(model,  os.path.join(MODELS_DIR, f"regression_model_{key}.pkl"))
        _dump(rmse,   os.path.join(MODELS_DIR, f"regression_rmse_{key}.pkl"))

        enc_artifact = {"encoder": encoder, "cat_cols": cat_cols, "scaler": scaler}
        _dump(enc_artifact, os.path.join(MODELS_DIR, f"regression_encoder_{key}.pkl"))

    def _save_comparison(self, df: pd.DataFrame) -> None:
        records = df.to_dict(orient="records")
        with open(COMPARISON_PATH, "w") as f:
            json.dump(records, f, indent=2)
        self._log(f"Comparison saved → {COMPARISON_PATH}")

    @staticmethod
    def load_comparison() -> list[dict]:
        if not os.path.exists(COMPARISON_PATH):
            return []
        with open(COMPARISON_PATH) as f:
            return json.load(f)
