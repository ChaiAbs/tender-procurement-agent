"""
regressor.py — Regressor

Responsibilities:
  1. Train an XGBoost regressor on log-transformed contract values.
  2. Evaluate on the held-out test set (R², RMSE, MAE — all in log space).
  3. Save the trained model to disk.
  4. At inference time, load the model and return a dollar-scale point estimate
     plus a confidence interval derived from training RMSE.

Context keys consumed:  X_train, X_test, y_train, y_test  (training mode)
Context keys produced:  regression_model, regression_metrics,
                        regression_prediction  (dict with point estimate + CI)
"""

import os
import pickle

import numpy  as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from .base  import PipelineStep
from config import MODELS_DIR, XGBOOST_REGRESSOR_PARAMS


REGRESSION_MODEL_PATH = os.path.join(MODELS_DIR, "regression_model.pkl")
REGRESSION_RMSE_PATH  = os.path.join(MODELS_DIR, "regression_rmse.pkl")


class Regressor(PipelineStep):
    """Trains and runs the XGBoost regression model for point-estimate prediction."""

    def __init__(self, verbose: bool = True):
        super().__init__(name="Regressor", verbose=verbose)
        self.model      = None
        self.train_rmse: float | None = None

    # ── Public interface ───────────────────────────────────────────────────────

    def run(self, context: dict) -> dict:
        """Training mode: fit model, evaluate, save."""
        self._start()
        try:
            from xgboost import XGBRegressor

            X_train = context["X_train"]
            X_test  = context["X_test"]
            y_train = context["y_train"]
            y_test  = context["y_test"]

            self.log("Training XGBoost regressor …")
            self.model = XGBRegressor(**XGBOOST_REGRESSOR_PARAMS)
            self.model.fit(X_train, y_train)

            y_pred          = self.model.predict(X_test)
            metrics         = self._evaluate(y_test.values, y_pred)
            self.train_rmse = metrics["rmse_log"]

            self.log(
                f"Results — R²={metrics['r2']:.4f} | "
                f"RMSE(log)={metrics['rmse_log']:.4f} | "
                f"MAE(log)={metrics['mae_log']:.4f}"
            )

            self._save()
            context["regression_model"]   = self.model
            context["regression_metrics"] = metrics
            self._finish()
        except Exception as exc:
            self._fail(exc)
            raise
        return context

    def predict(self, X_infer: pd.DataFrame) -> dict:
        """
        Inference mode: return a point estimate and 90% confidence interval.
        Loads the model from disk if not already in memory.
        """
        if self.model is None:
            self._load()

        log_pred  = float(self.model.predict(X_infer)[0])
        point_est = float(np.expm1(log_pred))

        # 90% CI: ±1.645 * RMSE in log space, back-transformed
        rmse    = self.train_rmse or 1.3
        ci_low  = float(np.expm1(log_pred - 1.645 * rmse))
        ci_high = float(np.expm1(log_pred + 1.645 * rmse))

        return {
            "point_estimate_aud": round(point_est, 2),
            "ci_low_90_aud":      round(max(ci_low, 0), 2),
            "ci_high_90_aud":     round(ci_high, 2),
            "log_prediction":     round(log_pred, 4),
        }

    # ── Private helpers ────────────────────────────────────────────────────────

    def _evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        return {
            "r2":       round(r2_score(y_true, y_pred), 4),
            "rmse_log": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
            "mae_log":  round(float(mean_absolute_error(y_true, y_pred)), 4),
        }

    def _save(self) -> None:
        with open(REGRESSION_MODEL_PATH, "wb") as f:
            pickle.dump(self.model, f)
        with open(REGRESSION_RMSE_PATH, "wb") as f:
            pickle.dump(self.train_rmse, f)
        self.log(f"Model saved → {REGRESSION_MODEL_PATH}")

    def _load(self) -> None:
        if not os.path.exists(REGRESSION_MODEL_PATH):
            raise FileNotFoundError(
                f"No trained regression model found at {REGRESSION_MODEL_PATH}. "
                "Run training first."
            )
        with open(REGRESSION_MODEL_PATH, "rb") as f:
            self.model = pickle.load(f)
        if os.path.exists(REGRESSION_RMSE_PATH):
            with open(REGRESSION_RMSE_PATH, "rb") as f:
                self.train_rmse = pickle.load(f)
        self.log("Regression model loaded from disk.")
