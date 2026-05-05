"""
metrics/cloud_logger.py — Structured metrics logging to Google Cloud Logging.

Each prediction emits two log entries:
  1. tender_agent/prediction  — ML output, confidence, contract fields
  2. tender_agent/llm         — per-node latency, token counts, estimated cost

On Cloud Logging, create log-based metrics on these log names to build
dashboards in Cloud Monitoring (e.g. latency p50/p95, cost/day, confidence
distribution, prediction volume by jurisdiction).

Fallback: if google-cloud-logging is unavailable or credentials are missing,
logs are written as structured JSON to stdout — Cloud Run picks these up
automatically via its built-in log driver.

Claude Sonnet 4.6 pricing (as of May 2025):
  Input  tokens: $3.00 / 1M
  Output tokens: $15.00 / 1M
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

# Pricing constants for claude-sonnet-4-6
_INPUT_COST_PER_TOKEN  = 3.00  / 1_000_000
_OUTPUT_COST_PER_TOKEN = 15.00 / 1_000_000

_GCP_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCLOUD_PROJECT")
_LOG_NAME    = "tender_agent"


def _estimate_cost(input_tokens: int, output_tokens: int) -> float:
    return round(
        input_tokens  * _INPUT_COST_PER_TOKEN +
        output_tokens * _OUTPUT_COST_PER_TOKEN,
        6,
    )


class TenderMetricsLogger:
    """
    Singleton logger. Call TenderMetricsLogger.get() to obtain the shared instance.
    Initialises the Cloud Logging client once; falls back to stdout JSON on error.
    """

    _instance: "TenderMetricsLogger | None" = None

    def __init__(self):
        self._cloud_available = False
        self._logger_prediction = None
        self._logger_llm = None
        self._init_cloud()

    @classmethod
    def get(cls) -> "TenderMetricsLogger":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_cloud(self):
        if not _GCP_PROJECT:
            print(
                "[metrics] GOOGLE_CLOUD_PROJECT not set — metrics will be "
                "written as structured JSON to stdout.",
                flush=True,
            )
            return
        try:
            import google.cloud.logging as gcl
            client = gcl.Client(project=_GCP_PROJECT)
            self._logger_prediction = client.logger(f"{_LOG_NAME}/prediction")
            self._logger_llm        = client.logger(f"{_LOG_NAME}/llm")
            self._cloud_available   = True
            print(
                f"[metrics] Cloud Logging initialised → project={_GCP_PROJECT} "
                f"log={_LOG_NAME}/*",
                flush=True,
            )
        except Exception as exc:
            print(
                f"[metrics] Cloud Logging unavailable ({exc}) — "
                "falling back to stdout JSON.",
                flush=True,
            )

    # ── Public API ────────────────────────────────────────────────────────────

    def log_prediction(
        self,
        *,
        session_id: str,
        contract: dict,
        ml_results: dict,
        knn_range: dict,
        ml_latency_ms: float,
        total_latency_ms: float,
    ) -> None:
        """
        Emit a structured prediction log entry.

        Fields visible in Cloud Monitoring log-based metrics:
          point_estimate_aud, confidence, bucket, ml_latency_ms,
          total_latency_ms, publisher_gov_type, category_code, model_key
        """
        reg = ml_results.get("regression", {})
        val = ml_results.get("validation", {})

        payload: dict[str, Any] = {
            # Identifiers
            "event_type":          "prediction",
            "session_id":          session_id[:8],

            # ML output
            "model_key":           reg.get("model_key", "unknown"),
            "point_estimate_aud":  reg.get("point_estimate_aud"),
            "ci_low_aud":          reg.get("ci_low_90_aud"),
            "ci_high_aud":         reg.get("ci_high_90_aud"),
            "log_prediction":      reg.get("log_prediction"),

            # Confidence / validation
            "confidence":          val.get("confidence", "unknown"),
            "bucket":              val.get("bucket", "unknown"),
            "bucket_probability":  val.get("bucket_probability"),
            "is_novel_contract":   val.get("is_novel_contract", False),

            # KNN range
            "knn_low_aud":         knn_range.get("low"),
            "knn_high_aud":        knn_range.get("high"),
            "knn_median_aud":      knn_range.get("median"),
            "knn_n_contracts":     knn_range.get("n_contracts"),

            # Contract fields (no PII)
            "publisher_gov_type":  contract.get("publisher_gov_type", "unknown"),
            "category_code":       contract.get("category_code", "unknown"),
            "procurement_method":  contract.get("procurement_method", "unknown"),
            "disposition":         contract.get("disposition", "unknown"),
            "publisher_cofog_level": contract.get("publisher_cofog_level", "unknown"),
            "duration_days":       contract.get("duration_days"),

            # Latency
            "ml_latency_ms":       round(ml_latency_ms),
            "total_latency_ms":    round(total_latency_ms),
        }

        self._emit("prediction", payload)

    def log_llm_call(
        self,
        *,
        session_id: str,
        node: str,                   # "analysis" | "reporting" | "agent_turn"
        latency_ms: float,
        input_tokens: int  = 0,
        output_tokens: int = 0,
        model: str         = "claude-sonnet-4-6",
        error: str | None  = None,
    ) -> None:
        """
        Emit a structured LLM call log entry.

        Fields visible in Cloud Monitoring log-based metrics:
          node, latency_ms, input_tokens, output_tokens,
          estimated_cost_usd, model
        """
        cost = _estimate_cost(input_tokens, output_tokens)

        payload: dict[str, Any] = {
            "event_type":          "llm_call",
            "session_id":          session_id[:8],
            "node":                node,
            "model":               model,
            "latency_ms":          round(latency_ms),
            "input_tokens":        input_tokens,
            "output_tokens":       output_tokens,
            "total_tokens":        input_tokens + output_tokens,
            "estimated_cost_usd":  cost,
        }
        if error:
            payload["error"] = error

        self._emit("llm", payload)

    def log_rag_call(
        self,
        *,
        session_id: str,
        field: str,
        latency_ms: float,
        n_results: int = 0,
    ) -> None:
        """Emit a RAG lookup log entry."""
        payload: dict[str, Any] = {
            "event_type":  "rag_call",
            "session_id":  session_id[:8],
            "field":       field,
            "latency_ms":  round(latency_ms),
            "n_results":   n_results,
        }
        self._emit("llm", payload)   # write to same llm log for simplicity

    # ── Internal ──────────────────────────────────────────────────────────────

    def _emit(self, log_type: str, payload: dict) -> None:
        """Write to Cloud Logging or fall back to stdout JSON."""
        if self._cloud_available:
            try:
                logger = (
                    self._logger_prediction
                    if log_type == "prediction"
                    else self._logger_llm
                )
                logger.log_struct(payload, severity="INFO")
                return
            except Exception as exc:
                print(f"[metrics] Cloud Logging write failed: {exc}", file=sys.stderr, flush=True)

        # Stdout fallback — Cloud Run ships this to Cloud Logging automatically
        # when the log entry contains a `severity` field.
        entry = {"severity": "INFO", "logName": f"{_LOG_NAME}/{log_type}", **payload}
        print(json.dumps(entry), flush=True)
