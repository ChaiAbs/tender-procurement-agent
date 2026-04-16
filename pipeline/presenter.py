"""
presenter.py — Presenter

Responsibilities:
  Takes the assembled context (regression prediction + bucket prediction +
  validation) and formats it into a clean, human-readable prediction report.

  Two output formats:
    - text  : coloured terminal report (default)
    - dict  : structured dict for programmatic consumption

Context keys consumed:  contract, regression_prediction,
                        bucket_prediction, validation
Context keys produced:  report_text, report_dict
"""

import math
from .base import PipelineStep


# ANSI colours for terminal output
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_CYAN   = "\033[96m"
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_DIM    = "\033[2m"

_CONFIDENCE_COLOUR = {
    "High":     _GREEN,
    "Medium":   _CYAN,
    "Low":      _YELLOW,
    "Very Low": _RED,
}


class Presenter(PipelineStep):
    """Formats the final prediction into a readable report."""

    def __init__(self, verbose: bool = True):
        super().__init__(name="Presenter", verbose=verbose)

    # ── Public interface ───────────────────────────────────────────────────────

    def run(self, context: dict) -> dict:
        self._start()
        try:
            report_dict = self._build_dict(context)
            report_text = self._build_text(report_dict)

            context["report_dict"] = report_dict
            context["report_text"] = report_text

            if self.verbose:
                print(report_text)

            self._finish()
        except Exception as exc:
            self._fail(exc)
            raise
        return context

    # ── Report builders ────────────────────────────────────────────────────────

    def _build_dict(self, ctx: dict) -> dict:
        reg  = ctx.get("regression_prediction", {})
        bkt  = ctx.get("bucket_prediction",     {})
        val  = ctx.get("validation",             {})
        con  = ctx.get("contract",               {})

        return {
            "input_features":   con,
            "point_estimate":   {
                "value_aud":    reg.get("point_estimate_aud"),
                "ci_low_aud":   reg.get("ci_low_90_aud"),
                "ci_high_aud":  reg.get("ci_high_90_aud"),
            },
            "bucket_prediction":{
                "bucket":       bkt.get("predicted_bucket"),
                "probability":  bkt.get("bucket_probability"),
                "all_probs":    bkt.get("all_bucket_probs"),
            },
            "subrange_prediction": {
                "label":        bkt.get("predicted_subrange"),
                "low_aud":      bkt.get("subrange_low_aud"),
                "high_aud":     bkt.get("subrange_high_aud"),
                "probability":  bkt.get("subrange_probability"),
            },
            "confidence": {
                "label":        val.get("confidence_label"),
                "score":        val.get("confidence_score"),
            },
            "warnings":         val.get("warnings", []),
            "consistency":      val.get("consistency", {}),
        }

    def _build_text(self, d: dict) -> str:
        reg    = d["point_estimate"]
        bkt    = d["bucket_prediction"]
        sr     = d["subrange_prediction"]
        conf   = d["confidence"]
        warns  = d["warnings"]
        inputs = d["input_features"]

        conf_colour = _CONFIDENCE_COLOUR.get(conf["label"], _RESET)

        lines = [
            "",
            f"{_BOLD}{_CYAN}╔══════════════════════════════════════════════════════╗{_RESET}",
            f"{_BOLD}{_CYAN}║       TENDER PRICE PREDICTION REPORT                 ║{_RESET}",
            f"{_BOLD}{_CYAN}╚══════════════════════════════════════════════════════╝{_RESET}",
            "",
            f"{_BOLD}── Input Features ─────────────────────────────────────{_RESET}",
        ]

        for k, v in inputs.items():
            lines.append(f"  {_DIM}{k:<30}{_RESET} {v}")

        lines += [
            "",
            f"{_BOLD}── Point Estimate (XGBoost Regression) ────────────────{_RESET}",
            f"  Predicted value   : {_BOLD}{self._fmt(reg['value_aud'])}{_RESET}",
            f"  90% CI            : {self._fmt(reg['ci_low_aud'])}  –  {self._fmt(reg['ci_high_aud'])}",
            "",
            f"{_BOLD}── Range Prediction ────────────────────────────────────{_RESET}",
            f"  Bucket            : {_BOLD}{bkt['bucket']}{_RESET}",
            f"  Sub-range         : {_BOLD}{_GREEN}{sr['label']}{_RESET}",
            f"  Range in AUD      : {self._fmt(sr['low_aud'])}  –  "
            f"{self._fmt(sr['high_aud']) if sr['high_aud'] else '>$150M'}",
            "",
            f"  Bucket probabilities:",
        ]

        for b, p in (bkt.get("all_probs") or {}).items():
            bar   = "█" * int(p * 20)
            lines.append(f"    {b:<12} {bar:<20} {p:.0%}")

        lines += [
            "",
            f"{_BOLD}── Confidence Assessment ───────────────────────────────{_RESET}",
            f"  Confidence label  : {conf_colour}{_BOLD}{conf['label']}{_RESET}  "
            f"(score {conf['score']:.2f} / 1.00)",
        ]

        if warns:
            lines += [
                "",
                f"{_BOLD}{_YELLOW}── Warnings ────────────────────────────────────────────{_RESET}",
            ]
            for w in warns:
                lines.append(f"  ⚠  {_YELLOW}{w}{_RESET}")

        lines += [
            "",
            f"{_DIM}Model: XGBoost native categoricals (pre-award features only) | "
            f"R²=0.59 | Bucket derived from regression point estimate{_RESET}",
            "",
        ]

        return "\n".join(lines)

    @staticmethod
    def _fmt(value: float | None) -> str:
        if value is None:
            return "N/A"
        if value >= 1_000_000:
            return f"${value/1_000_000:,.2f}M"
        if value >= 1_000:
            return f"${value/1_000:,.1f}K"
        return f"${value:,.0f}"
