"""
compare_models.py — Quick comparison of mean vs median regression.
Trains a single mean (MSE) regression model and prints MAE/R² for comparison.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pipeline.data_processor import DataProcessor
from config import XGBOOST_REGRESSOR_PARAMS, MODELS_DIR
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

DATA_PATH = "tenders_export.xlsx"

print("Loading and preprocessing data...")
dp = DataProcessor(verbose=False)
context = dp.run({"data_path": DATA_PATH})

X_train = context["X_train"]
X_test  = context["X_test"]
y_train = context["y_train"]
y_test  = context["y_test"]

print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

from xgboost import XGBRegressor

# Mean (MSE) regression — old approach
mean_params = {k: v for k, v in XGBOOST_REGRESSOR_PARAMS.items()
               if k not in ("objective", "quantile_alpha")}
mean_params["objective"] = "reg:squarederror"

print("\nTraining mean (MSE) regression model...")
model_mean = XGBRegressor(**mean_params)
model_mean.fit(X_train, y_train)

y_pred_mean = model_mean.predict(X_test)
r2_mean   = round(r2_score(y_test.values, y_pred_mean), 4)
rmse_mean = round(float(np.sqrt(mean_squared_error(y_test.values, y_pred_mean))), 4)
mae_mean  = round(float(mean_absolute_error(y_test.values, y_pred_mean)), 4)

print(f"\n{'='*45}")
print(f"{'Model':<20} {'R²':>6} {'RMSE(log)':>10} {'MAE(log)':>10}")
print(f"{'='*45}")
print(f"{'Mean (MSE)':<20} {r2_mean:>6} {rmse_mean:>10} {mae_mean:>10}")
print(f"{'Median (quantile)':<20} {'see training output':>28}")
print(f"{'='*45}")
print("\nNote: lower MAE(log) = better typical accuracy")
print("      higher R²      = more variance explained")
