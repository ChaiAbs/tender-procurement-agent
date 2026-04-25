"""
tools/domain_tools.py — LangChain tool for domain knowledge lookups.
"""

import json
from langchain_core.tools import tool
from config import DOMAIN_RAG_N_RESULTS


@tool
def lookup_procurement_codes(description: str, field: str = "all") -> str:
    """
    Look up correct AusTender field values from a natural language description.

    Use this tool whenever you need to determine:
      - The correct UNSPSC category_code or parent_category_code for a contract
      - The correct publisher_cofog_level for a government agency function
      - The exact valid string for procurement_method, disposition, or publisher_gov_type

    Args:
        description: Natural language description of what to look up.
                     Examples:
                       "IT security consulting services"
                       "Department of Defence function"
                       "open competitive procurement method"
        field:       Which field to look up. One of:
                       "category_code"    — UNSPSC commodity code
                       "cofog"            — COFOG level for agency function
                       "procurement_method" — valid procurement method string
                       "gov_type"         — valid publisher_gov_type string
                       "all"              — search across all sources (default)

    Returns:
        JSON array of matches, each with the relevant code/value and context.
    """
    from rag.domain_retriever import search_domain

    source_map = {
        "category_code":    "unspsc",
        "cofog":            "cofog",
        "procurement_method": "austender",
        "gov_type":         "austender",
        "all":              None,
    }
    source = source_map.get(field)

    try:
        results = search_domain(description, n_results=DOMAIN_RAG_N_RESULTS, source=source)

        simplified = []
        for r in results:
            meta = r["metadata"]
            entry = {"similarity": r["similarity_score"], "source": r["source"]}

            if r["source"] == "unspsc":
                entry["category_code"]        = meta.get("code")
                entry["name"]                 = meta.get("title")
                entry["hierarchy"]            = meta.get("hierarchy")

            elif r["source"] == "cofog":
                entry["cofog_code"]  = meta.get("code")
                entry["cofog_title"] = meta.get("title")
                entry["cofog_level"] = meta.get("level")
                entry["text"]        = r["text"][:300]

            elif r["source"] == "austender":
                entry["field"] = meta.get("field")
                entry["text"]  = r["text"]

            simplified.append(entry)

        return json.dumps(simplified, indent=2)

    except FileNotFoundError as exc:
        return json.dumps({
            "error": str(exc),
            "hint": "Build the domain index first: python rag/domain_indexer.py",
        })
    except Exception as exc:
        return json.dumps({"error": str(exc)})
