"""
app.py — FastAPI web application for the Tender Price Prediction agent.

The agent is conversational: the user describes a contract in natural language
and the agent collects the 7 pre-award fields through dialogue, then runs
the ML pipeline and returns a procurement briefing report.

Run locally:
    .venv/bin/python app.py

Docker / cloud:
    docker build -t tender-agent .
    docker run -p 8000:8000 --env-file .env tender-agent
"""

import json
import os
import subprocess
import sys
import uuid
from typing import Any, Optional

import anthropic
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("CHROMA_CACHE_DIR", "/app/.chroma_cache")
load_dotenv()

app = FastAPI(title="Tender Price Prediction Agent")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.on_event("startup")
def _preload_onnx():
    """
    Pre-load ONNX embedding model and warm up both ChromaDB collections.
    Without this, the first RAG request triggers a 79MB download mid-request,
    causing Cloud Run timeouts.
    """
    try:
        from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
        ONNXMiniLM_L6_V2()
        print("[startup] ChromaDB ONNX model ready.", flush=True)
    except Exception as exc:
        print(f"[startup] Warning: could not pre-load ONNX model: {exc}", flush=True)

    try:
        from rag.domain_retriever import _get_collection
        _get_collection()
        print("[startup] Domain RAG collection loaded.", flush=True)
    except Exception as exc:
        print(f"[startup] Warning: could not pre-load domain RAG: {exc}", flush=True)

# ── In-memory session store (swap for Redis in production) ─────────────────────
_sessions: dict[str, list[dict]] = {}

# ── Anthropic client ───────────────────────────────────────────────────────────
_client = anthropic.Anthropic()

SYSTEM_PROMPT = """You are a tender price prediction assistant for Australian Government procurement.

Your job is to collect the following contract details from the user, then call
the predict_contract tool to get an ML-based price estimate.

Required fields:
  1. procurement_method      — e.g. "Open tender", "Direct sourcing", "Select tender"
  2. disposition             — e.g. "Contract Notice", "Standing Offer Notice"
  3. is_consultancy_services — always "no" (constant in training data, set automatically)
  4. publisher_gov_type      — "fed" (Federal), or state: "qld", "nsw", "vic", "wa", "act", "sa", "tas", "nt"
  5. category_code           — UNSPSC commodity code, e.g. "81111500"
  6. parent_category_code    — Parent UNSPSC code, e.g. "81000000"
  7. publisher_cofog_level   — COFOG level, e.g. "2"
  8. publisher_name          — the publishing agency name, e.g. "Department of Defence"

Optional but important:
  9. duration_days           — intended contract duration from the procurement plan,
     in days. If the user mentions years or months, convert (1 year = 365,
     6 months = 183, etc.). If unknown, omit it and the model will use a
     statistical default.

Guidelines:
- Be conversational and helpful. Extract details from what the user tells you.
- Ask only for the fields you are missing — don't ask for everything at once.
- Always ask for the intended contract duration from the procurement plan — it significantly improves prediction accuracy.
- If a field cannot be determined, use "unknown".
- Once you have at least procurement_method, disposition, and publisher_gov_type,
  call predict_contract (use "unknown" for the rest).

Field extraction rules:
- For category_code and parent_category_code: ALWAYS call lookup_procurement_codes with
  field="category_code" and a description of what is being purchased before setting these.
  Never guess a UNSPSC code from memory. The tool returns both category_code AND
  parent_category_code — use the top result directly. Do NOT ask the user to confirm
  the code — just use the best match and proceed.
- For publisher_cofog_level: only two values exist — "1.0" or "2.0". Use "2.0" for all
  health, education, defence, social services, and most line agencies. Use "1.0" for
  central/cross-government agencies (e.g. Finance, Treasury, APS Commission). Do NOT
  call lookup_procurement_codes for COFOG — just apply this rule directly.
- For publisher_gov_type: any agency named "Department of [X]" or "Australian [X]" or
  "Commonwealth [X]" without a state name is "fed". State agencies include the state name
  (e.g. "NSW Health", "Queensland Treasury"). Use the state code: "qld", "nsw", "vic",
  "wa", "act", "sa", "tas", "nt". Do NOT ask the user to confirm this — infer it from
  the agency name.
- For procurement_method: use exact lowercase strings. Common values: "open tender",
  "limited tender", "direct sourcing", "select tender", "demand driven".

After getting the prediction, reply with exactly these three sections.
In the Price Prediction section, include the model used as a bullet point: "**ML model:** XGBoost" — use the display name matching the model_key (xgboost→XGBoost, lightgbm→LightGBM, catboost→CatBoost, random_forest→Random Forest, extra_trees→Extra Trees, hist_gb→Hist Gradient Boosting, gradient_boost→Gradient Boosting, ridge→Ridge Regression).



## Price Prediction
- **ML model:** <model display name>
- **Point estimate:** the exact value from `regression.point_estimate_aud`, formatted (e.g. $706,000)
- **KNN price range:** the exact values from `knn_range.low_formatted` to `knn_range.high_formatted` — do NOT invent a different range
- **Confidence:** High / Medium / Low / Very Low — state how many of the 7 fields were provided (e.g. "High — all 7 fields known")

## Similar Historical Contracts
Show a markdown table with columns: Category | Gov Type | Method | Value
List up to 5 similar contracts from the results. If none available, say so.

## Recommendation
One sentence only. State: the point estimate is $X, the KNN range of similar contracts is $Y to $Z (use the exact formatted values from the tool output). Do not synthesize, widen, or narrow these numbers.

Keep the full briefing report on the right panel — the chat reply is the concise summary only.
"""

TOOLS = [
    {
        "name": "lookup_procurement_codes",
        "description": (
            "Look up correct AusTender field values (UNSPSC codes, COFOG level, "
            "procurement method strings) from a natural language description. "
            "Use this before calling predict_contract whenever you need to determine "
            "category_code, parent_category_code, publisher_cofog_level, or the exact "
            "string for procurement_method."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "Natural language description, e.g. 'IT security consulting' or 'Department of Defence function'",
                },
                "field": {
                    "type": "string",
                    "enum": ["category_code", "cofog", "procurement_method", "gov_type", "all"],
                    "description": "Which field to look up. Use 'all' when unsure.",
                },
            },
            "required": ["description"],
        },
    },
    {
        "name": "predict_contract",
        "description": (
            "Run the ML price prediction pipeline for a tender contract. "
            "Returns a structured prediction report."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "procurement_method":      {"type": "string", "description": "How the contract is procured"},
                "disposition":             {"type": "string", "description": "Type of notice"},
                "is_consultancy_services": {"type": "string", "description": "Yes or No"},
                "publisher_gov_type":      {"type": "string", "description": "fed for federal, or state abbreviation: qld, nsw, vic, wa, act, sa, tas, nt"},
                "category_code":           {"type": "string", "description": "UNSPSC commodity code"},
                "parent_category_code":    {"type": "string", "description": "Parent UNSPSC code"},
                "publisher_cofog_level":   {"type": "string", "description": "COFOG classification level"},
                "publisher_name":          {"type": "string", "description": "Publishing agency name, e.g. Department of Defence"},
                "duration_days":           {"type": "number", "description": "Intended contract duration from the procurement plan, in days"},
            },
            "required": [
                "procurement_method", "disposition", "publisher_gov_type",
            ],
        },
    }
]


# ── ML prediction (reuses ml_runner subprocess) ───────────────────────────────

def _run_ml_prediction(contract: dict, model_key: str | None = None) -> dict:
    """Call the ML pipeline in a subprocess (isolates XGBoost/OpenMP from async)."""
    from ml_evaluation.evaluator import get_active_model
    key    = model_key or get_active_model()
    runner = os.path.join(os.path.dirname(__file__), "tools", "ml_runner.py")
    result = subprocess.run(
        [sys.executable, runner, json.dumps(contract), key],
        capture_output=True, text=True, timeout=60,
        env={**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"},
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(result.stderr or "ML runner returned no output")
    return json.loads(result.stdout)


def _run_langchain_report(contract: dict, ml_results: dict) -> tuple[str, list[dict], dict]:
    """
    Generate the full briefing report via the three-node LangGraph pipeline.
    Returns (report_text, similar_contracts, knn_range).
    """
    from langchain_agents.graph import get_graph

    initial = {
        "contract":               contract,
        "regression_prediction":  ml_results.get("regression", {}),
        "validation_result":      ml_results.get("validation", {}),
        "ml_critique":            "",
        "similar_contracts":      [],
        "knn_range":              {},
        "analysis":               "",
        "report":                 "",
        "messages":               [],
        "errors":                 [],
    }
    result = get_graph().invoke(initial)
    return (
        result.get("report", "Report generation failed."),
        result.get("similar_contracts", []),
        result.get("knn_range", {}),
    )


# ── API models ─────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    model_key: Optional[str] = None   # override active model for this request


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    report: Optional[str] = None        # full briefing report (when prediction runs)
    prediction: Optional[dict] = None   # raw ML numbers


# ── Chat endpoint ──────────────────────────────────────────────────────────────

@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    history    = _sessions.setdefault(session_id, [])

    history.append({"role": "user", "content": req.message})

    report     = None
    prediction = None
    reply      = ""

    # Agentic loop — handles tool use transparently
    messages = list(history)
    while True:
        response = _client.messages.create(
            model=os.environ.get("LLM_MODEL", "claude-sonnet-4-6"),
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        # Collect any text from this response turn
        for block in response.content:
            if block.type == "text":
                reply += block.text

        if response.stop_reason != "tool_use":
            break

        # Handle tool calls
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            # Domain lookup tool
            if block.name == "lookup_procurement_codes":
                from tools.domain_tools import lookup_procurement_codes
                result = lookup_procurement_codes.invoke({
                    "description": block.input.get("description", ""),
                    "field":       block.input.get("field", "all"),
                })
                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": block.id,
                    "content":     result,
                })
                continue

            contract = {
                "procurement_method":      block.input.get("procurement_method",      "unknown"),
                "disposition":             block.input.get("disposition",             "unknown"),
                "is_consultancy_services": "no",
                "publisher_gov_type":      block.input.get("publisher_gov_type",      "unknown"),
                "category_code":           block.input.get("category_code",           "unknown"),
                "parent_category_code":    block.input.get("parent_category_code",    "unknown"),
                "publisher_cofog_level":   block.input.get("publisher_cofog_level",   "unknown"),
                "publisher_name":          block.input.get("publisher_name",          "unknown"),
                "duration_days":           block.input.get("duration_days"),
            }

            try:
                ml_results = _run_ml_prediction(contract, model_key=req.model_key)
                prediction = ml_results

                report, similar, knn_range = _run_langchain_report(contract, ml_results)
                tool_output = json.dumps({
                    "regression":        ml_results.get("regression", {}),
                    "knn_range":         knn_range,
                    "validation":        ml_results.get("validation", {}),
                    "similar_contracts": similar,
                })
            except Exception as exc:
                import traceback
                tb = traceback.format_exc()
                print(f"[predict_contract error]\n{tb}", flush=True)
                reply += f"\n\n**Debug error:**\n```\n{tb}\n```"
                tool_output = json.dumps({"error": str(exc)})

            tool_results.append({
                "type":        "tool_result",
                "tool_use_id": block.id,
                "content":     tool_output,
            })

        # Feed tool results back and continue the loop
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user",      "content": tool_results})

    history.append({"role": "assistant", "content": reply})

    return ChatResponse(
        session_id=session_id,
        reply=reply,
        report=report,
        prediction=prediction,
    )


@app.get("/api/models")
def list_models():
    """Return all registered models, their metrics (if trained), and the active key."""
    from ml_evaluation.model_registry import MODEL_REGISTRY
    from ml_evaluation.evaluator      import MultiModelEvaluator, get_active_model

    active     = get_active_model()
    comparison = MultiModelEvaluator.load_comparison()
    metrics_by_key = {
        r["model_key"]: r for r in comparison if r.get("status") == "ok"
    }

    models: list[dict[str, Any]] = []
    for key, spec in MODEL_REGISTRY.items():
        entry: dict[str, Any] = {
            "key":          key,
            "display_name": spec["display_name"],
            "is_active":    key == active,
        }
        if key in metrics_by_key:
            m = metrics_by_key[key]
            entry.update({
                "r2":           m["r2"],
                "rmse_log":     m["rmse_log"],
                "mae_log":      m["mae_log"],
                "mae_dollar":   m.get("mae_dollar"),
                "within_50pct": m.get("within_50pct"),
                "train_time_s": m.get("train_time_s"),
            })
        models.append(entry)

    return {"models": models, "active": active}


class SetModelRequest(BaseModel):
    model_key: str


@app.post("/api/models/active")
def set_active_model(body: SetModelRequest):
    """Set the active prediction model."""
    from ml_evaluation.evaluator import set_active_model as _set
    from fastapi import HTTPException
    try:
        _set(body.model_key)
        return {"active": body.model_key}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.delete("/api/session/{session_id}")
def clear_session(session_id: str):
    _sessions.pop(session_id, None)
    return {"cleared": True}


@app.get("/api/health")
def health():
    return {"status": "ok"}


# ── Serve UI ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def index():
    with open("static/index.html") as f:
        return f.read()


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
