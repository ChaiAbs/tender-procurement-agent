"""
rag/retriever.py — Query the ChromaDB vector store for similar tenders.

The vectorstore is loaded once and cached for the process lifetime.
"""

import os

import chromadb
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

from config import (
    RAG_DIR,
    RAG_COLLECTION_NAME,
    RAG_N_RESULTS,
)

_collection = None


def _get_collection():
    global _collection
    if _collection is None:
        if not os.path.exists(RAG_DIR):
            raise FileNotFoundError(
                f"RAG index not found at '{RAG_DIR}'.\n"
                "Index the dataset first:\n"
                "  python run_agent.py index --data tenders_export.xlsx"
            )
        client = chromadb.PersistentClient(path=RAG_DIR)
        _collection = client.get_collection(
            name=RAG_COLLECTION_NAME,
            embedding_function=ONNXMiniLM_L6_V2(),
        )
    return _collection


def search_contracts(contract: dict, n_results: int = RAG_N_RESULTS) -> list[dict]:
    """
    Find the most similar historical contracts to the given one.

    Args:
        contract:  Dict with any subset of the 7 pre-award feature fields.
        n_results: Number of results to return.

    Returns:
        List of dicts, each containing:
            - The 7 pre-award feature values from the historical contract
            - value           — actual contract value in AUD
            - value_formatted — human-readable string (e.g. "$1.23M")
            - similarity_score — 0–1, higher = closer match
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
    parts = [
        f"{short}: {contract.get(col, 'unknown')}"
        for col, short in field_map.items()
    ]
    query = " | ".join(parts)

    col = _get_collection()
    results_raw = col.query(
        query_texts=[query],
        n_results=n_results,
        include=["metadatas", "distances"],
    )

    results = []
    metadatas = results_raw["metadatas"][0]
    distances = results_raw["distances"][0]

    for meta, dist in zip(metadatas, distances):
        result = dict(meta)
        # ChromaDB returns L2 distance — convert to 0-1 similarity
        result["similarity_score"] = round(max(0.0, 1 - dist / 2), 4)

        val = result.get("value", 0)
        try:
            val = float(val)
        except (TypeError, ValueError):
            val = 0.0

        if val >= 1_000_000:
            result["value_formatted"] = f"${val / 1_000_000:,.2f}M"
        elif val >= 1_000:
            result["value_formatted"] = f"${val / 1_000:,.1f}K"
        else:
            result["value_formatted"] = f"${val:,.0f}"

        results.append(result)

    return results
