"""
langchain_agents/nodes.py — LangGraph node functions.

Single node pipeline:

    reporting

The reporting node does a RAG search for similar contracts then synthesises
ML outputs, validation results, and historical comparisons into a final
procurement briefing report.
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


def reporting_node(state: TenderState) -> dict:
    """
    RAG search + final procurement briefing report.
    Retrieves similar historical contracts then synthesises all ML outputs
    and validation results into a document for procurement officers.
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

    prompt = load_prompt("reporting_agent")
    messages = [
        SystemMessage(content=prompt["system"]),
        HumanMessage(content=prompt["human"].format(
            contract_json=json.dumps(state["contract"], indent=2),
            regression_json=json.dumps(state.get("regression_prediction", {}), indent=2),
            bucket_json=json.dumps(state.get("bucket_prediction", {}), indent=2),
            validation_json=json.dumps(state.get("validation_result", {}), indent=2),
            similar_contracts_json=similar_json,
        )),
    ]

    response = _llm().invoke(messages)

    return {
        "similar_contracts": similar_contracts,
        "report": response.content,
        "messages": messages + [response],
    }
