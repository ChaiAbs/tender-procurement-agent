"""
regressor.py — Model-agnostic regression step.

Supports all models registered in ml_evaluation.model_registry.
Each model key has its own saved artifacts under models/:
  regression_model_{key}.pkl    — trained estimator
  regression_rmse_{key}.pkl     — training RMSE (log-space), used for CI
  regression_encoder_{key}.pkl  — {"encoder", "cat_cols", "scaler"} for sklearn models

At inference the active model key is resolved via ml_evaluation.evaluator.get_active_model().
"""

from __future__ import annotations

import os
import pickle

import numpy  as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .base  import PipelineStep
from config import MODELS_DIR


def _model_path(key: str)   -> str: return os.path.join(MODELS_DIR, f"regression_model_{key}.pkl")
def _rmse_path(key: str)    -> str: return os.path.join(MODELS_DIR, f"regression_rmse_{key}.pkl")
def _encoder_path(key: str) -> str: return os.path.join(MODELS_DIR, f"regression_encoder_{key}.pkl")

# Legacy path kept so existing XGBoost model still loads if no key-specific file exists yet.
_LEGACY_MODEL_PATH = os.path.join(MODELS_DIR, "regression_model.pkl")
_LEGACY_RMSE_PATH  = os.path.join(MODELS_DIR, "regression_rmse.pkl")


class Regressor(PipelineStep):
    """Trains and runs a regression model for point-estimate prediction."""

    def __init__(self, model_key: str = "xgboost", verbose: bool = True):
        super().__init__(name=f"Regressor[{model_key}]", verbose=verbose)
        self.model_key   = model_key
        self.model       = None
        self._encoder_artifact: dict | None = None   # {encoder, cat_cols, scaler}
        self.train_rmse: float | None = None

    # ── Training ───────────────────────────────────────────────────────────────

    def run(self, context: dict) -> dict:
        """Training mode: fit model on context splits, evaluate, save artifacts."""
        self._start()
        try:
            from ml_evaluation.model_registry import MODEL_REGISTRY
            from ml_evaluation.evaluator import ordinal_encode_split

            spec    = MODEL_REGISTRY[self.model_key]
            X_train = context["X_train"]
            X_test  = context["X_test"]
            y_train = context["y_train"]
            y_test  = context["y_test"]

            # Encode features for sklearn models
            if spec["native_categorical"]:
                X_tr, X_te = X_train, X_test
                enc_artifact = {"encoder": None, "cat_cols": [], "scaler": None}
            else:
                X_tr, X_te, encoder, cat_cols = ordinal_encode_split(X_train, X_test)
                enc_artifact = {"encoder": encoder, "cat_cols": cat_cols, "scaler": None}

                if spec["needs_scaler"]:
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    X_tr = pd.DataFrame(scaler.fit_transform(X_tr), columns=X_tr.columns)
                    X_te = pd.DataFrame(scaler.transform(X_te),     columns=X_te.columns)
                    enc_artifact["scaler"] = scaler

            self.model = spec["factory"]()
            self.log(f"Training {spec['display_name']} …")

            if spec["is_catboost"]:
                cat_indices = [list(X_tr.columns).index(c) for c in enc_artifact["cat_cols"]]
                for c in enc_artifact["cat_cols"]:
                    X_tr[c] = X_tr[c].astype(int)
                    X_te[c] = X_te[c].astype(int)
                self.model.fit(X_tr, y_train.values, cat_features=cat_indices)
            else:
                self.model.fit(X_tr, y_train.values)

            y_pred          = self.model.predict(X_te)
            metrics         = self._evaluate(y_test.values, y_pred)
            self.train_rmse = metrics["rmse_log"]

            self.log(
                f"Results — R²={metrics['r2']:.4f} | "
                f"RMSE(log)={metrics['rmse_log']:.4f} | "
                f"MAE(log)={metrics['mae_log']:.4f}"
            )

            self._encoder_artifact = enc_artifact
            self._save()
            context["regression_model"]   = self.model
            context["regression_metrics"] = metrics
            self._finish()
        except Exception as exc:
            self._fail(exc)
            raise
        return context

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, X_infer: pd.DataFrame) -> dict:
        """Return a point estimate and 90% CI. Loads model from disk if needed."""
        if self.model is None:
            self._load()

        print(f"[trace] │   ┌── Regressor.predict()  [{self.model_key}]", file=__import__("sys").stderr, flush=True)
        print(f"[trace] │   │   Purpose: run XGBoost/LightGBM/CatBoost on encoded features", file=__import__("sys").stderr, flush=True)
        print(f"[trace] │   │            predict log1p(value), then expm1 back to AUD", file=__import__("sys").stderr, flush=True)
        X = self._apply_encoding(X_infer)
        log_pred  = float(self.model.predict(X)[0])
        point_est = float(np.expm1(log_pred))
        print(f"[trace] │   │   log_pred={log_pred:.4f}  →  point_estimate=${point_est:,.0f}", file=__import__("sys").stderr, flush=True)
        print(f"[trace] │   └── Regressor.predict() done", file=__import__("sys").stderr, flush=True)

        rmse    = self.train_rmse or 1.3
        ci_low  = float(np.expm1(log_pred - 1.645 * rmse))
        ci_high = float(np.expm1(log_pred + 1.645 * rmse))

        return {
            "point_estimate_aud": round(point_est, 2),
            "ci_low_90_aud":      round(max(ci_low, 0), 2),
            "ci_high_90_aud":     round(ci_high, 2),
            "log_prediction":     round(log_pred, 4),
            "model_key":          self.model_key,
        }

    # ── Private helpers ────────────────────────────────────────────────────────

    def _apply_encoding(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply saved encoder/scaler (no-op for native categorical models)."""
        if self._encoder_artifact is None:
            return X
        enc      = self._encoder_artifact["encoder"]
        cat_cols = self._encoder_artifact["cat_cols"]
        scaler   = self._encoder_artifact["scaler"]

        if enc is None:
            return X  # native categorical model, no encoding needed

        from ml_evaluation.evaluator import apply_ordinal_encoding
        from ml_evaluation.model_registry import MODEL_REGISTRY
        X = apply_ordinal_encoding(X, enc, cat_cols)

        if MODEL_REGISTRY.get(self.model_key, {}).get("is_catboost"):
            for c in cat_cols:
                X[c] = X[c].astype(int)

        if scaler is not None:
            X = pd.DataFrame(scaler.transform(X), columns=X.columns)
        return X

    def _evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        return {
            "r2":       round(r2_score(y_true, y_pred), 4),
            "rmse_log": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
            "mae_log":  round(float(mean_absolute_error(y_true, y_pred)), 4),
        }

    def _save(self) -> None:
        key = self.model_key
        with open(_model_path(key), "wb") as f:
            pickle.dump(self.model, f)
        with open(_rmse_path(key), "wb") as f:
            pickle.dump(self.train_rmse, f)
        with open(_encoder_path(key), "wb") as f:
            pickle.dump(self._encoder_artifact, f)
        # Keep legacy paths in sync for the default XGBoost model
        if key == "xgboost":
            with open(_LEGACY_MODEL_PATH, "wb") as f:
                pickle.dump(self.model, f)
            with open(_LEGACY_RMSE_PATH, "wb") as f:
                pickle.dump(self.train_rmse, f)
        self.log(f"Artifacts saved → models/regression_*_{key}.pkl")

    def _load(self) -> None:
        key = self.model_key
        mp  = _model_path(key)

        # Fall back to legacy path for xgboost if key-specific file not yet created
        if not os.path.exists(mp) and key == "xgboost" and os.path.exists(_LEGACY_MODEL_PATH):
            mp = _LEGACY_MODEL_PATH

        if not os.path.exists(mp):
            raise FileNotFoundError(
                f"No trained model found for key '{key}' at {_model_path(key)}. "
                "Run training first: python train_models.py"
            )

        with open(mp, "rb") as f:
            self.model = pickle.load(f)

        rp = _rmse_path(key)
        if not os.path.exists(rp) and key == "xgboost" and os.path.exists(_LEGACY_RMSE_PATH):
            rp = _LEGACY_RMSE_PATH
        if os.path.exists(rp):
            with open(rp, "rb") as f:
                self.train_rmse = pickle.load(f)

        ep = _encoder_path(key)
        if os.path.exists(ep):
            with open(ep, "rb") as f:
                self._encoder_artifact = pickle.load(f)
        else:
            # Older XGBoost artifact with no encoder file — native categorical, no encoding needed
            self._encoder_artifact = {"encoder": None, "cat_cols": [], "scaler": None}

        self.log(f"Model '{key}' loaded from disk.")
