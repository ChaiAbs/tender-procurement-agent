"""
data_processor.py — DataProcessor

Responsibilities:
  1. Load the raw tender data (Excel or CSV).
  2. Filter out invalid rows (missing / zero contract values).
  3. Clean and normalise all categorical features.
  4. Encode categoricals using XGBoost native categorical support (pd.Categorical).
     Rare values (< MIN_CATEGORY_FREQ) are collapsed to '<col>_other'.
  5. Log-transform the target variable (contract value).
  6. Build a publisher_name → (portfolio, cofog_level) lookup table for inference.
  7. Expose a preprocess_single() helper for inference on new contracts.

Context keys consumed:  data_path  (str) — path to tenders_export.xlsx
Context keys produced:  X_train, X_test, y_train, y_test,
                        feature_schema, raw_df, stats
"""

import os
import pickle

import numpy  as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .base  import PipelineStep
from config import (
    CATEGORICAL_FEATURES, PRE_AWARD_FEATURES,
    NUMERIC_FEATURES, DURATION_CAP_DAYS, TARGET_COLUMN, MODELS_DIR,
)


SCHEMA_PATH          = os.path.join(MODELS_DIR, "ohe_schema.pkl")
PUBLISHER_LOOKUP_PATH = os.path.join(MODELS_DIR, "publisher_lookup.pkl")


class DataProcessor(PipelineStep):
    """Loads, cleans, and transforms tender data for the pipeline."""

    def __init__(self, verbose: bool = True):
        super().__init__(name="DataProcessor", verbose=verbose)
        self.feature_schema: list[str] | None = None
        self._kept_categories: dict[str, set] = {}
        self._category_dtypes: dict[str, list] = {}   # col → sorted list of kept categories
        self._duration_median: float = 331.0
        self._publisher_lookup: dict = {}

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
            df = self._compute_duration(df, fit=True)
            self._build_publisher_lookup(df)
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
        - Auto-fills publisher_portfolio and publisher_cofog_level from publisher_name
        - Imputes duration_days with training median if not provided
        """
        if self.feature_schema is None:
            self._load_schema()

        contract = dict(contract)  # don't mutate caller's dict

        # Auto-fill portfolio and cofog from publisher_name lookup
        pub_name = str(contract.get("publisher_name", "unknown")).strip().lower()
        lookup = self._publisher_lookup.get(pub_name, {})
        if not contract.get("publisher_portfolio") or contract["publisher_portfolio"] in (None, "", "unknown"):
            contract["publisher_portfolio"] = lookup.get("publisher_portfolio", "unknown")
        if not contract.get("publisher_cofog_level") or contract["publisher_cofog_level"] in (None, "", "unknown"):
            contract["publisher_cofog_level"] = lookup.get("publisher_cofog_level", "unknown")

        # Compute or impute duration_days
        if not contract.get("duration_days") or contract["duration_days"] in (None, "", "unknown"):
            if "contract_start" in contract and "contract_end" in contract:
                try:
                    start = pd.to_datetime(contract["contract_start"])
                    end   = pd.to_datetime(contract["contract_end"])
                    days  = (end - start).days
                    contract["duration_days"] = float(min(max(days, 1), DURATION_CAP_DAYS))
                except Exception:
                    contract["duration_days"] = self._duration_median
            else:
                contract["duration_days"] = self._duration_median

        row = pd.DataFrame([contract])
        row = self._clean_features(row)
        row = self._compute_duration(row, fit=False)
        X, _ = self._encode_features(row, fit=False)
        return X

    # ── Private helpers ────────────────────────────────────────────────────────

    # Minimum times a category value must appear to get its own category.
    # Values below this are collapsed to '<col>_other'.
    MIN_CATEGORY_FREQ = 50

    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalise all categorical features to lowercase strings."""
        for col in CATEGORICAL_FEATURES:
            if col not in df.columns:
                df[col] = "unknown"
                continue
            df[col] = (
                df[col]
                .fillna("unknown")
                .astype(str)
                .str.strip()
                .str.lower()
            )
        return df

    def _compute_duration(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """
        Derive duration_days, contract_start_year, and contract_start_quarter
        from contract_start / contract_end.
        fit=True: compute from date columns, save medians.
        fit=False: duration_days already set in preprocess_single; year/quarter derived from contract_start.
        """
        if fit:
            if "contract_start" in df.columns and "contract_end" in df.columns:
                start = pd.to_datetime(df["contract_start"], errors="coerce")
                end   = pd.to_datetime(df["contract_end"],   errors="coerce")
                days  = (end - start).dt.days
                days  = days.clip(lower=1, upper=DURATION_CAP_DAYS)
                self._duration_median = float(days.median())
                df["duration_days"] = days.fillna(self._duration_median)

                # Year and quarter from contract_start
                df["contract_start_year"]    = start.dt.year.fillna(start.dt.year.median()).astype(float)
                df["contract_start_quarter"] = start.dt.quarter.fillna(2.0).astype(float)
            else:
                self._duration_median = 331.0
                df["duration_days"]          = self._duration_median
                df["contract_start_year"]    = 2020.0
                df["contract_start_quarter"] = 2.0
            self.log(
                f"duration_days: median={self._duration_median:.0f} days | "
                f"contract_start_year/quarter derived"
            )
        else:
            # Inference: derive year/quarter from contract_start if available
            if "contract_start" in df.columns:
                start = pd.to_datetime(df["contract_start"], errors="coerce")
                df["contract_start_year"]    = start.dt.year.fillna(2020.0).astype(float)
                df["contract_start_quarter"] = start.dt.quarter.fillna(2.0).astype(float)
            else:
                import datetime
                df["contract_start_year"]    = float(datetime.datetime.now().year)
                df["contract_start_quarter"] = float((datetime.datetime.now().month - 1) // 3 + 1)
        return df

    def _build_publisher_lookup(self, df: pd.DataFrame) -> None:
        """
        Build publisher_name → {portfolio, cofog_level} lookup from training data.
        Uses the most common value per agency. Saved to disk for inference.
        """
        lookup = {}
        for pub_name, group in df.groupby("publisher_name"):
            portfolio = group["publisher_portfolio"].mode() if "publisher_portfolio" in group else pd.Series()
            cofog     = group["publisher_cofog_level"].mode() if "publisher_cofog_level" in group else pd.Series()
            lookup[pub_name] = {
                "publisher_portfolio":  portfolio.iloc[0] if len(portfolio) > 0 else "unknown",
                "publisher_cofog_level": cofog.iloc[0]    if len(cofog)     > 0 else "unknown",
            }
        self._publisher_lookup = lookup
        os.makedirs(MODELS_DIR, exist_ok=True)
        with open(PUBLISHER_LOOKUP_PATH, "wb") as f:
            pickle.dump(lookup, f)
        self.log(f"Publisher lookup: {len(lookup):,} agencies saved → {PUBLISHER_LOOKUP_PATH}")

    def _encode_features(
        self, df: pd.DataFrame, fit: bool
    ) -> tuple[pd.DataFrame, pd.Series | None]:
        """
        Encode all categorical features using XGBoost native categorical support.
        Each column is cast to pd.Categorical with a fixed list of known categories.
        Rare / unseen values are replaced with '<col>_other' before casting.
        Numeric features are appended as plain floats.
        """
        result = pd.DataFrame(index=df.index)

        if fit:
            self._kept_categories = {}
            self._category_dtypes = {}

        for col in CATEGORICAL_FEATURES:
            # Get or create column data
            if col in df.columns:
                col_data = df[col].copy()
            else:
                col_data = pd.Series(["unknown"] * len(df), index=df.index, dtype=str)

            if fit:
                counts = col_data.value_counts()
                kept   = set(counts[counts >= self.MIN_CATEGORY_FREQ].index)
                self._kept_categories[col] = kept
            else:
                kept = self._kept_categories.get(col, set())

            # Collapse rare / unseen values
            other = f"{col}_other"
            col_data = col_data.where(col_data.isin(kept), other=other)

            if fit:
                cats = sorted(col_data.unique().tolist())
                # Ensure _other is always a valid category
                if other not in cats:
                    cats.append(other)
                self._category_dtypes[col] = cats
            else:
                cats = self._category_dtypes.get(col, [other])
                # Map anything not in training categories to _other
                col_data = col_data.where(col_data.isin(cats), other=other)

            result[col] = pd.Categorical(col_data, categories=cats)

        # Append numeric features
        for col in NUMERIC_FEATURES:
            result[col] = df[col].values if col in df.columns else 0.0

        if fit:
            self.feature_schema = list(result.columns)
            self._save_schema()
            self.log(
                f"Native categorical encoding: {len(CATEGORICAL_FEATURES)} categorical + "
                f"{len(NUMERIC_FEATURES)} numeric | min_freq={self.MIN_CATEGORY_FREQ}"
            )
        else:
            # Reindex to training column order, re-apply categorical dtypes
            result = result.reindex(columns=self.feature_schema)
            for col in CATEGORICAL_FEATURES:
                if col in result.columns:
                    cats = self._category_dtypes.get(col, [f"{col}_other"])
                    result[col] = pd.Categorical(
                        result[col].fillna(f"{col}_other"), categories=cats
                    )

        y = None
        if fit and TARGET_COLUMN in df.columns:
            y = pd.Series(np.log1p(df[TARGET_COLUMN].values), name="log_value")

        return result, y

    def _save_schema(self) -> None:
        os.makedirs(MODELS_DIR, exist_ok=True)
        with open(SCHEMA_PATH, "wb") as f:
            pickle.dump({
                "schema":           self.feature_schema,
                "kept":             self._kept_categories,
                "category_dtypes":  self._category_dtypes,
                "duration_median":  self._duration_median,
            }, f)

    def _load_schema(self) -> None:
        if not os.path.exists(SCHEMA_PATH):
            raise FileNotFoundError(
                f"No schema found at {SCHEMA_PATH}. Run training first."
            )
        with open(SCHEMA_PATH, "rb") as f:
            data = pickle.load(f)
        self.feature_schema     = data["schema"]
        self._kept_categories   = data["kept"]
        self._category_dtypes   = data.get("category_dtypes", {})
        self._duration_median   = data.get("duration_median", 331.0)

        # Load publisher lookup
        if os.path.exists(PUBLISHER_LOOKUP_PATH):
            with open(PUBLISHER_LOOKUP_PATH, "rb") as f:
                self._publisher_lookup = pickle.load(f)

    # Keep old name as alias for backward compatibility with ml_tools.py
    def _load_ohe_schema(self) -> None:
        self._load_schema()

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

    def _compute_stats(self, df: pd.DataFrame) -> dict:
        vals = df[TARGET_COLUMN]
        return {
            "n_contracts":  len(df),
            "value_mean":   round(vals.mean(),   2),
            "value_median": round(vals.median(), 2),
            "value_min":    round(vals.min(),    2),
            "value_max":    round(vals.max(),    2),
        }
