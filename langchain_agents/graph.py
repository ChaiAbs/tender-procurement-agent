"""
langchain_agents/graph.py — LangGraph pipeline for tender price prediction.

Graph topology:

    START → analysis → reporting → END

analysis  : KNN search + interpretation of similar historical contracts
reporting : plausibility assessment + final procurement briefing synthesis
"""

from langgraph.graph import END, START, StateGraph

from .nodes import analysis_node, reporting_node
from .state import TenderState


def build_graph():
    builder = StateGraph(TenderState)
    builder.add_node("analysis", analysis_node)
    builder.add_node("reporting", reporting_node)
    builder.add_edge(START, "analysis")
    builder.add_edge("analysis", "reporting")
    builder.add_edge("reporting", END)
    return builder.compile()


_graph = None

def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def predict(contract: dict) -> dict:
    """
    Run the full pipeline for a single tender contract.
    ML predictions and deterministic validation run in a subprocess first,
    then the three-node LangGraph pipeline runs in sequence.
    """
    import json
    import os
    import subprocess
    import sys

    contract_json = json.dumps(contract)
    runner = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tools", "ml_runner.py")

    try:
        result = subprocess.run(
            [sys.executable, runner, contract_json],
            capture_output=True, text=True, timeout=60,
            env={**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"},
        )
        ml_results            = json.loads(result.stdout)
        regression_prediction = ml_results.get("regression", {})
        validation_result     = ml_results.get("validation", {})
    except Exception as e:
        regression_prediction = {"error": str(e)}
        validation_result     = {"error": str(e)}

    initial: TenderState = {
        "contract":               contract,
        "regression_prediction":  regression_prediction,
        "validation_result":      validation_result,
        "similar_contracts":      [],
        "knn_range":              {},
        "analysis":               "",
        "report":                 "",
        "messages":               [],
        "errors":                 [],
    }

    return get_graph().invoke(initial)
