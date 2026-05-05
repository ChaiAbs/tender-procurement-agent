"""
langchain_agents/state.py — Shared state for the LangGraph tender pipeline.

The state dict is passed between every node in the graph.  Each node reads
what it needs and returns a partial dict of keys it wants to update.
LangGraph merges those updates automatically.
"""

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage


class TenderState(TypedDict):
    # ── Input ──────────────────────────────────────────────────────────────────
    contract: dict                   # Raw contract dict with 7 pre-award fields

    # ── Pre-computed before graph (subprocess) ─────────────────────────────────
    regression_prediction: dict      # Point estimate + CI from active ML model
    validation_result: dict          # Deterministic: field checks, confidence

    # ── Populated by analysis_node ─────────────────────────────────────────────
    similar_contracts: list[dict]    # KNN results
    knn_range: dict                  # Price range derived from KNN: p10, median, p90
    analysis: str                    # Interpretation of similar contracts

    # ── Populated by reporting_node ────────────────────────────────────────────
    report: str                      # Final procurement briefing report

    # ── Accumulated across all nodes ───────────────────────────────────────────
    messages: Annotated[list[BaseMessage], operator.add]
    errors: list[str]

    # ── Metrics (populated by each node, consumed by app.py) ───────────────────
    analysis_latency_ms: float
    reporting_latency_ms: float
    total_input_tokens: int
    total_output_tokens: int
