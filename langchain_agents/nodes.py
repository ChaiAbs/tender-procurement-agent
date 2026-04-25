"""
langchain_agents/nodes.py — LangGraph node functions.

Two-node pipeline:

    analysis → reporting

analysis  : KNN search + computes price range from similar contracts
reporting : plausibility assessment + final procurement briefing synthesis
"""

import json

import numpy as np
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from config import LLM_MODEL, LLM_TEMPERATURE
from prompts import load_prompt
from tools.rag_tools import search_similar_contracts

from .state import TenderState


def _llm() -> ChatAnthropic:
    return ChatAnthropic(model=LLM_MODEL, temperature=LLM_TEMPERATURE)


def _fmt(val: float) -> str:
    if val >= 1_000_000:
        return f"${val / 1_000_000:,.2f}M"
    elif val >= 1_000:
        return f"${val / 1_000:,.1f}K"
    return f"${val:,.0f}"


def _compute_knn_range(similar_contracts: list[dict]) -> dict:
    """
    Derive price range from KNN similar contracts.

    Two-pass outlier removal:
      1. Tukey fences (Q1 - 1.5*IQR, Q3 + 1.5*IQR) — catches high-end outliers
      2. Ratio filter (median/5 to median*5) — catches low-end outliers that
         Tukey misses when the IQR is very wide relative to the median
    """
    values = [float(c["value"]) for c in similar_contracts if c.get("value")]
    if not values:
        return {}
    arr = np.array(values)

    if len(arr) >= 4:
        # Pass 1: Tukey fences
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        filtered = arr[(arr >= q1 - 1.5 * iqr) & (arr <= q3 + 1.5 * iqr)]
        if len(filtered) == 0:
            filtered = arr

        # Pass 2: ratio filter relative to median — removes extreme low/high outliers
        # that Tukey misses when IQR is very wide (common with small n)
        median = float(np.median(filtered))
        if median > 0:
            filtered = filtered[(filtered >= median / 5) & (filtered <= median * 5)]
        if len(filtered) == 0:
            filtered = arr
    else:
        filtered = arr

    low    = float(filtered.min())
    high   = float(filtered.max())
    median = float(np.median(arr))

    return {
        "low":              low,
        "median":           median,
        "high":             high,
        "low_formatted":    _fmt(low),
        "median_formatted": _fmt(median),
        "high_formatted":   _fmt(high),
        "n_contracts":      len(values),
        "n_used":           int(len(filtered)),
    }


def analysis_node(state: TenderState) -> dict:
    """
    KNN search + interpretation of similar historical contracts.
    Computes a KNN-grounded price range (10th–90th percentile) from results.
    """
    similar_raw = search_similar_contracts.invoke({"contract_json": json.dumps(state["contract"])})
    try:
        similar_contracts: list[dict] = json.loads(similar_raw)
        if isinstance(similar_contracts, dict) and "error" in similar_contracts:
            similar_contracts = []
            similar_json = "No similar contracts available (KNN index not built)."
        else:
            similar_json = json.dumps(similar_contracts, indent=2)
    except Exception:
        similar_contracts = []
        similar_json = similar_raw

    knn_range = _compute_knn_range(similar_contracts)

    prompt = load_prompt("analysis_agent")
    messages = [
        SystemMessage(content=prompt["system"]),
        HumanMessage(content=prompt["human"].format(
            contract_json=json.dumps(state["contract"], indent=2),
            similar_contracts_json=similar_json,
        )),
    ]

    response = _llm().invoke(messages)

    return {
        "similar_contracts": similar_contracts,
        "knn_range":         knn_range,
        "analysis":          response.content,
        "messages":          messages + [response],
    }


def reporting_node(state: TenderState) -> dict:
    """
    Plausibility assessment + final procurement briefing report.
    Synthesises ML outputs, KNN analysis, and validation results into
    a document for procurement officers.
    """
    prompt = load_prompt("reporting_agent")
    messages = [
        SystemMessage(content=prompt["system"]),
        HumanMessage(content=prompt["human"].format(
            contract_json=json.dumps(state["contract"], indent=2),
            regression_json=json.dumps(state.get("regression_prediction", {}), indent=2),
            knn_range_json=json.dumps(state.get("knn_range", {}), indent=2),
            validation_json=json.dumps(state.get("validation_result", {}), indent=2),
            similar_contracts_json=json.dumps(state.get("similar_contracts", []), indent=2),
            analysis=state.get("analysis", "Not available."),
        )),
    ]

    response = _llm().invoke(messages)

    return {
        "report":   response.content,
        "messages": messages + [response],
    }
