"""
rag/domain_retriever.py — Query the domain knowledge index.

Used by the agent to look up:
  - UNSPSC commodity codes from a natural language description
  - COFOG classification levels from an agency function description
  - AusTender valid field values
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DOMAIN_RAG_COLLECTION_NAME, DOMAIN_RAG_DIR, DOMAIN_RAG_N_RESULTS

_collection = None


def _get_collection():
    global _collection
    if _collection is None:
        import chromadb
        from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

        if not os.path.exists(DOMAIN_RAG_DIR):
            raise FileNotFoundError(
                f"Domain RAG index not found at '{DOMAIN_RAG_DIR}'.\n"
                "Build it first: python rag/domain_indexer.py"
            )
        client      = chromadb.PersistentClient(path=DOMAIN_RAG_DIR)
        _collection = client.get_collection(
            name=DOMAIN_RAG_COLLECTION_NAME,
            embedding_function=ONNXMiniLM_L6_V2(),
        )
    return _collection


def search_domain(query: str, n_results: int = DOMAIN_RAG_N_RESULTS, source: str | None = None) -> list[dict]:
    """
    Search the domain knowledge index.

    Args:
        query:     Natural language description (e.g. "IT security consulting")
        n_results: Number of results to return
        source:    Filter by source — "unspsc", "cofog", or "austender" (None = all)

    Returns:
        List of dicts with keys: source, text, metadata, similarity_score
    """
    col = _get_collection()

    where = {"source": source} if source else None
    raw   = col.query(
        query_texts=[query],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    results = []
    for doc, meta, dist in zip(
        raw["documents"][0],
        raw["metadatas"][0],
        raw["distances"][0],
    ):
        results.append({
            "source":           meta.get("source"),
            "text":             doc,
            "metadata":         meta,
            "similarity_score": round(max(0.0, 1 - dist / 2), 4),
        })

    return results
