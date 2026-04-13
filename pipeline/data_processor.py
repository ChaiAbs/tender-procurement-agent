"""
data_processor.py — DataProcessor

Responsibilities:
  1. Load the raw tender data (Excel or CSV).
  2. Filter out invalid rows (missing / zero contract values).
  3. Clean and normalise the 7 pre-award categorical features.
  4. One-hot encode categoricals (drop_first=False) — produces ~350-400 columns.
  5. Log-transform the target variable (contract value).
  6. Expose a preprocess_single() helper for inference on new contracts.

Context keys consumed:  data_path  (str) — path to tenders_export.xlsx
Context keys produced:  X_train, X_test, y_train, y_test,
                        feature_schema, raw_df, stats
"""

import os
import pickle

import numpy  as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .base   import PipelineStep
from config  import PRE_AWARD_FEATURES, TARGET_COLUMN, MODELS_DIR


OHE_SCHEMA_PATH = os.path.join(MODELS_DIR, "ohe_schema.pkl")


class DataProcessor(PipelineStep):
    """Loads, cleans, and transforms tender data for the pipeline."""

    def __init__(self, verbose: bool = True):
        super().__init__(name="DataProcessor", verbose=verbose)
        self.feature_schema: list[str] | None = None   # one-hot column names

    # ── Public interface ───────────────────────────────────────────────────────

    def run(self, context: dict) -> dict:
        """Training mode: load data, preprocess, split, update context."""
        self._start()
        try:
            data_path = context.get("data_path")
            if not data_path or not os.path.exists(data_path):
                raise FileNotFoundError(f"Data file not found: {data_path}")

            df = self._load(data_path)
            df = self._filter(df)
            df = self._clean_features(df)
            X, y = self._encode_features(df, fit=True)

            self.feature_schema = list(X.columns)
            context["feature_schema"] = self.feature_schema

            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            context.update({
                "X_train": X_tr,
                "X_test":  X_te,
                "y_train": y_tr,
                "y_test":  y_te,
                "raw_df":  df,
                "stats":   self._compute_stats(df),
            })
            self.log(
                f"Data ready — {len(df):,} rows | "
                f"{X_tr.shape[1]} features | "
                f"train={len(X_tr):,} test={len(X_te):,}"
            )
            self._finish()
        except Exception as exc:
            self._fail(exc)
            raise
        return context

    def preprocess_single(self, contract: dict) -> pd.DataFrame:
        """
        Inference mode: convert a raw contract dict into a feature row.
        One-hot encodes the single row then aligns columns to the training schema.
        """
        if self.feature_schema is None:
            self._load_ohe_schema()

        row = pd.DataFrame([contract])
        row = self._clean_features(row)
        X, _ = self._encode_features(row, fit=False)
        return X

    # ── Private helpers ────────────────────────────────────────────────────────

    # Minimum number of times a category value must appear to get its own column.
    # Values below this threshold are grouped into an '_other' bucket per feature.
    # Keeps feature count close to the report's ~350-400 columns.
    MIN_CATEGORY_FREQ = 50

    def _encode_features(
        self, df: pd.DataFrame, fit: bool
    ) -> tuple[pd.DataFrame, pd.Series | None]:
        """
        One-hot encode all categorical pre-award features.
        fit=True:  compute kept-category sets, build schema, save.
        fit=False: apply saved schema — unknown/rare values become '_other'.
        Column names are sanitised so XGBoost accepts them (no [, ], <).
        """
        feature_df = df[PRE_AWARD_FEATURES].copy()

        if fit:
            self._kept_categories: dict[str, set] = {}
            for col in PRE_AWARD_FEATURES:
                counts = feature_df[col].value_counts()
                self._kept_categories[col] = set(
                    counts[counts >= self.MIN_CATEGORY_FREQ].index
                )

        # Replace rare / unseen values with '_other'
        for col in PRE_AWARD_FEATURES:
            kept = getattr(self, "_kept_categories", {}).get(col, set())
            feature_df[col] = feature_df[col].where(
                feature_df[col].isin(kept), other=f"{col}_other"
            )

        X = pd.get_dummies(feature_df, columns=PRE_AWARD_FEATURES, drop_first=False)

        # Sanitise column names — XGBoost rejects [, ], <
        X.columns = [
            c.replace("[", "_").replace("]", "_").replace("<", "_")
            for c in X.columns
        ]

        if fit:
            self.feature_schema = list(X.columns)
            self._save_ohe_schema()
            self.log(f"One-hot encoding: {len(self.feature_schema)} features "
                     f"(min_freq={self.MIN_CATEGORY_FREQ})")
        else:
            X = X.reindex(columns=self.feature_schema, fill_value=0)

        y = None
        if fit and TARGET_COLUMN in df.columns:
            y = pd.Series(np.log1p(df[TARGET_COLUMN].values), name="log_value")

        return X, y

    def _save_ohe_schema(self) -> None:
        os.makedirs(MODELS_DIR, exist_ok=True)
        with open(OHE_SCHEMA_PATH, "wb") as f:
            pickle.dump(
                {"schema": self.feature_schema, "kept": self._kept_categories}, f
            )

    def _load_ohe_schema(self) -> None:
        if not os.path.exists(OHE_SCHEMA_PATH):
            raise FileNotFoundError(
                f"No OHE schema found at {OHE_SCHEMA_PATH}. Run training first."
            )
        with open(OHE_SCHEMA_PATH, "rb") as f:
            data = pickle.load(f)
        self.feature_schema     = data["schema"]
        self._kept_categories   = data["kept"]

    def _load(self, path: str) -> pd.DataFrame:
        self.log(f"Loading data from {os.path.basename(path)} …")
        ext = os.path.splitext(path)[1].lower()
        if ext in (".xlsx", ".xls"):
            df = pd.read_excel(path)
        else:
            df = pd.read_csv(path, low_memory=False)
        self.log(f"Loaded {len(df):,} rows × {df.shape[1]} columns")
        return df

    def _filter(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df[df[TARGET_COLUMN].notna()]
        df = df[pd.to_numeric(df[TARGET_COLUMN], errors="coerce").notna()]
        df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
        df = df[df[TARGET_COLUMN] > 0].copy()
        self.log(
            f"Filtered rows: {before:,} → {len(df):,} "
            f"({before - len(df):,} dropped)"
        )
        return df

    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalise the 7 pre-award features in place."""
        for col in PRE_AWARD_FEATURES:
            if col not in df.columns:
                df[col] = "unknown"
                continue
            # Always convert to lowercase string regardless of source dtype
            # (newer pandas returns StringDtype for Excel string columns, not object)
            df[col] = (
                df[col]
                .fillna("unknown")
                .astype(str)
                .str.strip()
                .str.lower()
            )
        return df

    def _compute_stats(self, df: pd.DataFrame) -> dict:
        vals = df[TARGET_COLUMN]
        return {
            "n_contracts":  len(df),
            "value_mean":   round(vals.mean(),   2),
            "value_median": round(vals.median(), 2),
            "value_min":    round(vals.min(),    2),
            "value_max":    round(vals.max(),    2),
        }
