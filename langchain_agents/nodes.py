"""
langchain_agents/nodes.py — LangGraph node functions.

Two-node pipeline:

    analysis → reporting

analysis  : KNN search + computes price range from similar contracts
reporting : plausibility assessment + final procurement briefing synthesis
"""

import json
import time

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
    print(f"\n[trace] ┌── analysis_node()", flush=True)
    print(f"[trace] │   Purpose: search training data for the 5 most similar historical contracts", flush=True)
    print(f"[trace] │            using KNN in ML feature space, then compute a price range", flush=True)

    t0 = time.time()
    print(f"[trace] │   → search_similar_contracts.invoke()  [KNN lookup in pkl index]", flush=True)
    similar_raw = search_similar_contracts.invoke({"contract_json": json.dumps(state["contract"])})
    print(f"[timing] KNN search: {time.time() - t0:.1f}s", flush=True)

    try:
        similar_contracts: list[dict] = json.loads(similar_raw)
        if isinstance(similar_contracts, dict) and "error" in similar_contracts:
            similar_contracts = []
            similar_json = "No similar contracts available (KNN index not built)."
        else:
            similar_json = json.dumps(similar_contracts, indent=2)
            vals = [c.get("value", 0) for c in similar_contracts]
            print(f"[trace] │   KNN found {len(similar_contracts)} neighbours  values={[f'${v:,.0f}' for v in vals]}", flush=True)
    except Exception:
        similar_contracts = []
        similar_json = similar_raw

    knn_range = _compute_knn_range(similar_contracts)
    if knn_range:
        print(f"[trace] │   → _compute_knn_range()  [Tukey fence outlier removal + min/max]", flush=True)
        print(f"[trace] │   KNN range: {knn_range.get('low_formatted')} – {knn_range.get('high_formatted')}  (median {knn_range.get('median_formatted')})", flush=True)

    t1 = time.time()
    print(f"[trace] │   → analysis LLM call  [Claude interprets KNN neighbours]", flush=True)
    prompt = load_prompt("analysis_agent")
    messages = [
        SystemMessage(content=prompt["system"]),
        HumanMessage(content=prompt["human"].format(
            contract_json=json.dumps(state["contract"], indent=2),
            similar_contracts_json=similar_json,
        )),
    ]

    response = _llm().invoke(messages)
    analysis_latency_ms = (time.time() - t1) * 1000
    print(f"[timing] analysis LLM call: {analysis_latency_ms/1000:.1f}s", flush=True)
    print(f"[trace] └── analysis_node() done", flush=True)

    usage = response.response_metadata.get("usage", {})
    return {
        "similar_contracts":   similar_contracts,
        "knn_range":           knn_range,
        "analysis":            response.content,
        "messages":            messages + [response],
        "analysis_latency_ms": analysis_latency_ms,
        "total_input_tokens":  usage.get("input_tokens", 0),
        "total_output_tokens": usage.get("output_tokens", 0),
    }


def reporting_node(state: TenderState) -> dict:
    """
    Plausibility assessment + final procurement briefing report.
    Synthesises ML outputs, KNN analysis, and validation results into
    a document for procurement officers.
    """
    print(f"\n[trace] ┌── reporting_node()", flush=True)
    print(f"[trace] │   Purpose: synthesise ML point estimate + KNN range + validation", flush=True)
    print(f"[trace] │            into a full procurement briefing report", flush=True)
    reg = state.get("regression_prediction", {})
    val = state.get("validation_result", {})
    print(f"[trace] │   inputs: point_estimate=${reg.get('point_estimate_aud', 0):,.0f}  confidence={val.get('confidence', 'N/A')}", flush=True)
    print(f"[trace] │   → reporting LLM call  [Claude writes the final briefing report]", flush=True)

    prompt = load_prompt("reporting_agent")
    messages = [
        SystemMessage(content=prompt["system"]),
        HumanMessage(content=prompt["human"].format(
            contract_json=json.dumps(state["contract"], indent=2),
            regression_json=json.dumps(reg, indent=2),
            knn_range_json=json.dumps(state.get("knn_range", {}), indent=2),
            validation_json=json.dumps(val, indent=2),
            similar_contracts_json=json.dumps(state.get("similar_contracts", []), indent=2),
            analysis=state.get("analysis", "Not available."),
        )),
    ]

    t0 = time.time()
    response = _llm().invoke(messages)
    reporting_latency_ms = (time.time() - t0) * 1000
    print(f"[timing] reporting LLM call: {reporting_latency_ms/1000:.1f}s", flush=True)
    print(f"[trace] └── reporting_node() done — report length: {len(response.content)} chars", flush=True)

    usage = response.response_metadata.get("usage", {})
    prior_in  = state.get("total_input_tokens", 0)
    prior_out = state.get("total_output_tokens", 0)
    return {
        "report":                response.content,
        "messages":              messages + [response],
        "reporting_latency_ms":  reporting_latency_ms,
        "total_input_tokens":    prior_in  + usage.get("input_tokens", 0),
        "total_output_tokens":   prior_out + usage.get("output_tokens", 0),
    }
