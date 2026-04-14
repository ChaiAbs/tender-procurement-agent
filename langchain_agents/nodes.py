"""
langchain_agents/nodes.py — LangGraph node functions.

Three-node pipeline:

    ml_critique → analysis → reporting

ml_critique  : assesses plausibility of ML model outputs (no RAG)
analysis     : RAG search + interpretation of similar historical contracts
reporting    : synthesises all prior outputs into a final procurement briefing
"""

import json

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from config import LLM_MODEL, LLM_TEMPERATURE
from prompts import load_prompt
from tools.rag_tools import search_similar_contracts

from .state import TenderState


def _llm() -> ChatAnthropic:
    return ChatAnthropic(model=LLM_MODEL, temperature=LLM_TEMPERATURE)


def ml_critique_node(state: TenderState) -> dict:
    """
    Assesses whether the ML model outputs are plausible for this contract.
    Rates plausibility, identifies upside/downside risks, and gives a
    one-sentence recommendation — all based on ML outputs alone (no RAG).
    """
    prompt = load_prompt("ml_critique_agent")
    messages = [
        SystemMessage(content=prompt["system"]),
        HumanMessage(content=prompt["human"].format(
            contract_json=json.dumps(state["contract"], indent=2),
            regression_json=json.dumps(state.get("regression_prediction", {}), indent=2),
            bucket_json=json.dumps(state.get("bucket_prediction", {}), indent=2),
            validation_json=json.dumps(state.get("validation_result", {}), indent=2),
        )),
    ]

    response = _llm().invoke(messages)

    return {
        "ml_critique": response.content,
        "messages": messages + [response],
    }


def analysis_node(state: TenderState) -> dict:
    """
    RAG search + interpretation of similar historical contracts.
    Uses the ml_critique from the previous node as context so the
    interpretation is grounded in the model's plausibility assessment.
    """
    # RAG search
    similar_raw = search_similar_contracts.invoke({"contract_json": json.dumps(state["contract"])})
    try:
        similar_contracts: list[dict] = json.loads(similar_raw)
        if isinstance(similar_contracts, dict) and "error" in similar_contracts:
            similar_contracts = []
            similar_json = "No similar contracts available (RAG index not built)."
        else:
            similar_json = json.dumps(similar_contracts, indent=2)
    except Exception:
        similar_contracts = []
        similar_json = similar_raw

    prompt = load_prompt("analysis_agent")
    messages = [
        SystemMessage(content=prompt["system"]),
        HumanMessage(content=prompt["human"].format(
            contract_json=json.dumps(state["contract"], indent=2),
            similar_contracts_json=similar_json,
            ml_critique=state.get("ml_critique", "Not available."),
        )),
    ]

    response = _llm().invoke(messages)

    return {
        "similar_contracts": similar_contracts,
        "analysis": response.content,
        "messages": messages + [response],
    }


def reporting_node(state: TenderState) -> dict:
    """
    Final procurement briefing report.
    Synthesises ML outputs, plausibility critique, RAG analysis, and
    validation results into a document for procurement officers.
    """
    prompt = load_prompt("reporting_agent")
    messages = [
        SystemMessage(content=prompt["system"]),
        HumanMessage(content=prompt["human"].format(
            contract_json=json.dumps(state["contract"], indent=2),
            regression_json=json.dumps(state.get("regression_prediction", {}), indent=2),
            bucket_json=json.dumps(state.get("bucket_prediction", {}), indent=2),
            validation_json=json.dumps(state.get("validation_result", {}), indent=2),
            similar_contracts_json=json.dumps(state.get("similar_contracts", []), indent=2),
            ml_critique=state.get("ml_critique", "Not available."),
            analysis=state.get("analysis", "Not available."),
        )),
    ]

    response = _llm().invoke(messages)

    return {
        "report": response.content,
        "messages": messages + [response],
    }
