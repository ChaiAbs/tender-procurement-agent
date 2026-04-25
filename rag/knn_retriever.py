"""
rag/knn_retriever.py — KNN-based retrieval using the ML feature space.

Replaces ChromaDB semantic search with sklearn NearestNeighbors on the same
feature vectors the ML model uses. Contracts that are close in the model's
feature space are genuinely similar — unlike semantic embeddings which have no
understanding of UNSPSC codes or COFOG levels.

Artifacts saved to models/:
  knn_retriever.pkl  — fitted NearestNeighbors + OrdinalEncoder + StandardScaler
  knn_data.pkl       — raw contract rows aligned with the training index
"""

from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from config import MODELS_DIR, RAG_N_RESULTS

KNN_MODEL_PATH = os.path.join(MODELS_DIR, "knn_retriever.pkl")
KNN_DATA_PATH  = os.path.join(MODELS_DIR, "knn_data.pkl")

DISPLAY_COLS = [
    "procurement_method",
    "disposition",
    "is_consultancy_services",
    "publisher_gov_type",
    "category_code",
    "parent_category_code",
    "publisher_cofog_level",
    "publisher_name",
    "value",
]


# ── Build ──────────────────────────────────────────────────────────────────────

def build_knn_index(
    X_train: pd.DataFrame,
    raw_df: pd.DataFrame,
    n_neighbors: int = RAG_N_RESULTS,
) -> None:
    """
    Fit KNN on training feature vectors and save artifacts to models/.

    Args:
        X_train:     Feature matrix from DataProcessor (pd.Categorical columns).
        raw_df:      Full filtered DataFrame from DataProcessor (same index as X).
        n_neighbors: k for NearestNeighbors (matches RAG_N_RESULTS by default).
    """
    from ml_evaluation.evaluator import ordinal_encode_split

    print("[KNN] Encoding features …")
    X_enc, _, encoder, cat_cols = ordinal_encode_split(X_train, X_train)

    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X_enc)

    print(f"[KNN] Fitting NearestNeighbors on {len(X_train):,} contracts …")
    knn = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric="euclidean",
        algorithm="ball_tree",
        n_jobs=-1,
    )
    knn.fit(X_scaled)

    with open(KNN_MODEL_PATH, "wb") as f:
        pickle.dump({
            "knn":      knn,
            "encoder":  encoder,
            "cat_cols": cat_cols,
            "scaler":   scaler,
        }, f)

    # Store raw rows aligned to X_train — these are the contracts we'll surface
    keep_cols = [c for c in DISPLAY_COLS if c in raw_df.columns]
    knn_data  = raw_df.loc[X_train.index, keep_cols].copy().reset_index(drop=True)
    with open(KNN_DATA_PATH, "wb") as f:
        pickle.dump(knn_data, f)

    print(f"[KNN] Index saved → {KNN_MODEL_PATH}  ({len(knn_data):,} contracts)")


# ── Query ──────────────────────────────────────────────────────────────────────

_artifacts = None
_knn_data: pd.DataFrame | None = None


def _load() -> tuple[dict, pd.DataFrame]:
    global _artifacts, _knn_data
    if _artifacts is None:
        if not os.path.exists(KNN_MODEL_PATH):
            raise FileNotFoundError(
                f"KNN index not found at {KNN_MODEL_PATH}. "
                "Run training first: python train_models.py"
            )
        with open(KNN_MODEL_PATH, "rb") as f:
            _artifacts = pickle.load(f)
        with open(KNN_DATA_PATH, "rb") as f:
            _knn_data = pickle.load(f)
    return _artifacts, _knn_data


def search_contracts(contract: dict, n_results: int = RAG_N_RESULTS) -> list[dict]:
    """
    Find the most similar historical contracts using KNN in ML feature space.

    Returns the same schema as rag.retriever.search_contracts so the rest of
    the pipeline is unaffected.
    """
    from pipeline.data_processor import DataProcessor
    from ml_evaluation.evaluator import apply_ordinal_encoding

    artifacts, knn_data = _load()

    dp = DataProcessor(verbose=False)
    dp._load_schema()
    X_query = dp.preprocess_single(contract)

    X_enc    = apply_ordinal_encoding(X_query, artifacts["encoder"], artifacts["cat_cols"])
    X_scaled = artifacts["scaler"].transform(X_enc)

    k = min(n_results, len(knn_data))
    distances, indices = artifacts["knn"].kneighbors(X_scaled, n_neighbors=k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        row    = knn_data.iloc[idx].to_dict()
        val    = float(row.get("value", 0))
        row["value"] = val

        if val >= 1_000_000:
            row["value_formatted"] = f"${val / 1_000_000:,.2f}M"
        elif val >= 1_000:
            row["value_formatted"] = f"${val / 1_000:,.1f}K"
        else:
            row["value_formatted"] = f"${val:,.0f}"

        # Convert euclidean distance to a 0–1 similarity score
        row["similarity_score"] = round(float(1 / (1 + dist)), 4)

        results.append(row)

    return results
