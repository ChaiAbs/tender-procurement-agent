"""
tools/rag_tools.py — LangChain tool for searching the tenders RAG database.
"""

import json
from langchain_core.tools import tool
from config import RAG_N_RESULTS


@tool
def search_similar_contracts(contract_json: str, n_results: int = RAG_N_RESULTS) -> str:
    """
    Search the historical tenders database for contracts similar to the given one.

    Uses semantic similarity over the 7 pre-award features to find the closest
    historical matches. Each result includes the actual contract value so the
    LLM can reason about what similar contracts cost in practice.

    Args:
        contract_json: JSON string with any subset of the 7 pre-award fields.
        n_results: Number of similar contracts to return (default 5, max 20).

    Returns:
        JSON array. Each element has:
            procurement_method, disposition, is_consultancy_services,
            publisher_gov_type, category_code, parent_category_code,
            publisher_cofog_level — the historical contract's features
            value                  — actual contract value in AUD (float)
            value_formatted        — human-readable value string (e.g. "$1.23M")
            similarity_score       — 0–1, higher = more similar
    """
    # Import here to avoid loading ChromaDB at module import time
    from rag.retriever import search_contracts

    try:
        contract = json.loads(contract_json)
        results = search_contracts(contract, n_results=min(n_results, 20))
        return json.dumps(results, indent=2)
    except FileNotFoundError as exc:
        return json.dumps({
            "error": str(exc),
            "hint": "Index the dataset first: python run_agent.py index --data tenders_export.xlsx",
        })
    except Exception as exc:
        return json.dumps({"error": str(exc)})
