"""
rag/indexer.py — Index tenders_export data into a local ChromaDB vector store.

Each contract is converted to a text representation, embedded with a small
HuggingFace model (all-MiniLM-L6-v2, runs locally, no API cost), and stored
with its features and actual value as metadata.

Usage:
    from rag.indexer import index_tenders
    index_tenders("tenders_export.xlsx", sample_size=50_000)

    # or from the CLI:
    python run_agent.py index --data tenders_export.xlsx
"""

import os

import pandas as pd
import chromadb
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

from config import (
    PRE_AWARD_FEATURES,
    TARGET_COLUMN,
    RAG_DIR,
    RAG_COLLECTION_NAME,
    RAG_INDEX_SAMPLE,
)


def _contract_to_text(row: pd.Series) -> str:
    """
    Convert a contract row to a plain-text string for embedding.

    Example output:
      "procurement: open tender | disposition: contract notice |
       consultancy: no | gov_type: fed | category: 81111500 |
       parent_category: 81000000 | cofog: 2 | value: $500.0K"
    """
    field_map = {
        "procurement_method":      "procurement",
        "disposition":             "disposition",
        "is_consultancy_services": "consultancy",
        "publisher_gov_type":      "gov_type",
        "category_code":           "category",
        "parent_category_code":    "parent_category",
        "publisher_cofog_level":   "cofog",
    }
    parts = [f"{short}: {row.get(col, 'unknown')}" for col, short in field_map.items()]

    val = row.get(TARGET_COLUMN)
    if val and pd.notna(val):
        if val >= 1_000_000:
            parts.append(f"value: ${val / 1_000_000:.2f}M")
        elif val >= 1_000:
            parts.append(f"value: ${val / 1_000:.1f}K")
        else:
            parts.append(f"value: ${val:,.0f}")

    return " | ".join(parts)


def index_tenders(
    data_path: str,
    sample_size: int | None = RAG_INDEX_SAMPLE,
    batch_size: int = 500,
) -> int:
    """
    Index contracts from Excel/CSV into ChromaDB.

    Args:
        data_path:   Path to tenders_export.xlsx or a CSV file.
        sample_size: Rows to index. None = all rows (slow for 261 MB file).
        batch_size:  Embedding batch size (tune for memory vs speed).

    Returns:
        Number of documents successfully indexed.
    """
    os.makedirs(RAG_DIR, exist_ok=True)

    print(f"[RAG] Loading {data_path} ...")
    ext = os.path.splitext(data_path)[1].lower()
    df = pd.read_excel(data_path) if ext in (".xlsx", ".xls") else pd.read_csv(data_path, low_memory=False)

    # Keep only valid contracts (same filter as DataAgent)
    df = df[df[TARGET_COLUMN].notna()].copy()
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    df = df[df[TARGET_COLUMN] > 0]

    # Normalise feature columns
    for col in PRE_AWARD_FEATURES:
        if col not in df.columns:
            df[col] = "unknown"
        else:
            df[col] = df[col].fillna("unknown").astype(str).str.strip().str.lower()

    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
        print(f"[RAG] Sampled {sample_size:,} rows from {len(df):,} valid contracts")
    else:
        print(f"[RAG] Indexing all {len(df):,} contracts")

    ef = ONNXMiniLM_L6_V2()
    client = chromadb.PersistentClient(path=RAG_DIR)

    # Delete existing collection if present so we start fresh
    try:
        client.delete_collection(RAG_COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=RAG_COLLECTION_NAME,
        embedding_function=ef,
    )

    texts: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []

    for idx, (_, row) in enumerate(df.iterrows()):
        texts.append(_contract_to_text(row))
        meta = {col: str(row.get(col, "unknown")) for col in PRE_AWARD_FEATURES}
        meta["value"] = float(row.get(TARGET_COLUMN, 0))
        metadatas.append(meta)
        ids.append(str(idx))

    total = len(texts)
    for i in range(0, total, batch_size):
        collection.add(
            documents=texts[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
            ids=ids[i : i + batch_size],
        )
        print(f"[RAG] Indexed {min(i + batch_size, total):,} / {total:,}")

    print(f"[RAG] Done. Store at: {RAG_DIR}")
    return total
