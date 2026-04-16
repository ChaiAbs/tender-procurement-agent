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
from typing import Optional

import anthropic
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
load_dotenv()

app = FastAPI(title="Tender Price Prediction Agent")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.on_event("startup")
def _preload_onnx():
    """
    Download and cache the ChromaDB ONNX embedding model at container startup.
    Without this, the first request that hits the RAG triggers a 79MB download
    mid-request, which causes Cloud Run timeouts.
    """
    try:
        from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
        ONNXMiniLM_L6_V2()
        print("[startup] ChromaDB ONNX model ready.", flush=True)
    except Exception as exc:
        print(f"[startup] Warning: could not pre-load ONNX model: {exc}", flush=True)

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
  3. is_consultancy_services — "Yes" or "No"
  4. publisher_gov_type      — "FED" (Federal), "STATE", or "LOCAL"
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
- Once you have at least procurement_method, disposition, is_consultancy_services,
  and publisher_gov_type, call predict_contract (use "unknown" for the rest).

After getting the prediction, reply with exactly these three sections:

## Price Prediction
- **Predicted range:** the sub-range (e.g. $50K – $150K)
- **Price bucket:** Small / Medium / Large / Very Large with confidence %
- **Point estimate:** regression value in AUD
- **Confidence:** High / Medium / Low / Very Low and a one-line reason

## Similar Historical Contracts
Show a markdown table with columns: Category | Gov Type | Method | Value
List up to 5 similar contracts from the results. If none available, say so.

## Recommendation
Write 2-3 sentences of plain prose. Do NOT mention "Alignment Case", "Scenario", "Case 1/2/3/4", or any internal logic labels whatsoever.

Simply state which dollar range to use and why, based on whether the point estimate and/or bucket sub-range agree with similar historical contracts. Always give ONE clear recommended dollar range.

Keep the full briefing report on the right panel — the chat reply is the concise summary only.
"""

TOOLS = [
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
                "publisher_gov_type":      {"type": "string", "description": "FED, STATE, or LOCAL"},
                "category_code":           {"type": "string", "description": "UNSPSC commodity code"},
                "parent_category_code":    {"type": "string", "description": "Parent UNSPSC code"},
                "publisher_cofog_level":   {"type": "string", "description": "COFOG classification level"},
                "publisher_name":          {"type": "string", "description": "Publishing agency name, e.g. Department of Defence"},
                "duration_days":           {"type": "number", "description": "Intended contract duration from the procurement plan, in days"},
            },
            "required": [
                "procurement_method", "disposition",
                "is_consultancy_services", "publisher_gov_type",
            ],
        },
    }
]


# ── ML prediction (reuses ml_runner subprocess) ───────────────────────────────

def _run_ml_prediction(contract: dict) -> dict:
    """Call the ML pipeline in a subprocess (same path as run_agent.py)."""
    runner = os.path.join(os.path.dirname(__file__), "tools", "ml_runner.py")
    result = subprocess.run(
        [sys.executable, runner, json.dumps(contract)],
        capture_output=True, text=True, timeout=60,
        env={**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"},
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(result.stderr or "ML runner returned no output")
    return json.loads(result.stdout)


def _run_langchain_report(contract: dict, ml_results: dict) -> tuple[str, list[dict]]:
    """
    Generate the full briefing report via the three-node LangGraph pipeline.
    Returns (report_text, similar_contracts).
    """
    from langchain_agents.graph import get_graph

    initial = {
        "contract":               contract,
        "regression_prediction":  ml_results.get("regression", {}),
        "bucket_prediction":      ml_results.get("bucket", {}),
        "validation_result":      ml_results.get("validation", {}),
        "ml_critique":            "",
        "similar_contracts":      [],
        "analysis":               "",
        "report":                 "",
        "messages":               [],
        "errors":                 [],
    }
    result = get_graph().invoke(initial)
    return result.get("report", "Report generation failed."), result.get("similar_contracts", [])


# ── API models ─────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


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

            contract = {
                "procurement_method":      block.input.get("procurement_method",      "unknown"),
                "disposition":             block.input.get("disposition",             "unknown"),
                "is_consultancy_services": block.input.get("is_consultancy_services", "unknown"),
                "publisher_gov_type":      block.input.get("publisher_gov_type",      "unknown"),
                "category_code":           block.input.get("category_code",           "unknown"),
                "parent_category_code":    block.input.get("parent_category_code",    "unknown"),
                "publisher_cofog_level":   block.input.get("publisher_cofog_level",   "unknown"),
                "publisher_name":          block.input.get("publisher_name",          "unknown"),
                "duration_days":           block.input.get("duration_days"),
            }

            try:
                ml_results = _run_ml_prediction(contract)
                prediction = ml_results

                report, similar = _run_langchain_report(contract, ml_results)
                tool_output = json.dumps({
                    "regression":        ml_results.get("regression", {}),
                    "bucket":            ml_results.get("bucket", {}),
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
