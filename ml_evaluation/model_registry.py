"""
model_registry.py — Registry of all ML models available for tender price prediction.

Each entry:
  display_name       — label shown in UI and reports
  factory            — callable returning an untrained estimator
  native_categorical — True → pass pd.Categorical columns as-is (XGBoost, LightGBM)
                       False → apply OrdinalEncoder before fitting
  needs_scaler       — True → also apply StandardScaler (Ridge only)
  is_catboost        — True → pass cat_feature indices to .fit()
"""

from __future__ import annotations


def _xgboost_factory():
    from xgboost import XGBRegressor
    from config import XGBOOST_REGRESSOR_PARAMS
    return XGBRegressor(**XGBOOST_REGRESSOR_PARAMS)


def _lightgbm_factory():
    import lightgbm as lgb
    return lgb.LGBMRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        verbose=-1,
    )


def _catboost_factory():
    from catboost import CatBoostRegressor
    return CatBoostRegressor(
        iterations=200, depth=6, learning_rate=0.05,
        random_seed=42, verbose=0,
    )


def _random_forest_factory():
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(
        n_estimators=200, max_depth=12, random_state=42, n_jobs=-1,
    )


def _extra_trees_factory():
    from sklearn.ensemble import ExtraTreesRegressor
    return ExtraTreesRegressor(
        n_estimators=200, max_depth=12, random_state=42, n_jobs=-1,
    )


def _hist_gb_factory():
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(
        max_iter=200, max_depth=6, learning_rate=0.05,
        random_state=42,
    )


def _gradient_boost_factory():
    from sklearn.ensemble import GradientBoostingRegressor
    return GradientBoostingRegressor(
        n_estimators=100, max_depth=5, learning_rate=0.05,
        subsample=0.8, random_state=42,
    )


def _ridge_factory():
    from sklearn.linear_model import Ridge
    return Ridge(alpha=1.0)


MODEL_REGISTRY: dict[str, dict] = {
    "xgboost": {
        "display_name":       "XGBoost",
        "factory":            _xgboost_factory,
        "native_categorical": True,
        "needs_scaler":       False,
        "is_catboost":        False,
    },
    "lightgbm": {
        "display_name":       "LightGBM",
        "factory":            _lightgbm_factory,
        "native_categorical": True,
        "needs_scaler":       False,
        "is_catboost":        False,
    },
    "catboost": {
        "display_name":       "CatBoost",
        "factory":            _catboost_factory,
        "native_categorical": False,
        "needs_scaler":       False,
        "is_catboost":        True,
    },
    "random_forest": {
        "display_name":       "Random Forest",
        "factory":            _random_forest_factory,
        "native_categorical": False,
        "needs_scaler":       False,
        "is_catboost":        False,
    },
    "extra_trees": {
        "display_name":       "Extra Trees",
        "factory":            _extra_trees_factory,
        "native_categorical": False,
        "needs_scaler":       False,
        "is_catboost":        False,
    },
    "hist_gb": {
        "display_name":       "Hist Gradient Boosting",
        "factory":            _hist_gb_factory,
        "native_categorical": False,
        "needs_scaler":       False,
        "is_catboost":        False,
    },
    "gradient_boost": {
        "display_name":       "Gradient Boosting",
        "factory":            _gradient_boost_factory,
        "native_categorical": False,
        "needs_scaler":       False,
        "is_catboost":        False,
    },
    "ridge": {
        "display_name":       "Ridge Regression",
        "factory":            _ridge_factory,
        "native_categorical": False,
        "needs_scaler":       True,
        "is_catboost":        False,
    },
}

DEFAULT_MODEL = "xgboost"
