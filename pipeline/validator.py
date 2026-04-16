"""
validator.py — Validator

Responsibilities:
  1. Validate that a new contract has all required pre-award fields.
  2. Check for unusual / suspicious feature values and warn the user.
  3. Assess prediction confidence based on:
       - Which bucket was predicted (Small = most reliable).
       - The Stage 1 class probability (how certain the bucket classifier was).
       - Whether the regression point estimate falls inside the predicted sub-range.
  4. Surface a plain-English confidence label (High / Medium / Low / Very Low).
  5. Append any warnings or caveats to the context.

Context keys consumed:  contract (dict), regression_prediction, bucket_prediction
Context keys produced:  validation (dict with is_valid, warnings, confidence_label,
                                    consistency_check)
"""

import math
from .base  import PipelineStep
from config import PRE_AWARD_FEATURES, BUCKET_ORDER
from utils  import fmt_dollar


# Confidence scoring weights
_BUCKET_CONFIDENCE_BONUS = {
    "Small":      0.30,
    "Medium":     0.15,
    "Large":     -0.10,
    "Very Large":-0.25,
}


class Validator(PipelineStep):
    """
    Validates input and assesses the quality / reliability of a prediction.
    Does NOT change the prediction — it only annotates it with confidence
    metadata and human-readable warnings.
    """

    def __init__(self, verbose: bool = True):
        super().__init__(name="Validator", verbose=verbose)

    # ── Public interface ───────────────────────────────────────────────────────

    def run(self, context: dict) -> dict:
        """
        Called by the orchestrator during inference.
        Reads contract + predictions from context, writes validation results back.
        """
        self._start()
        try:
            contract    = context.get("contract", {})
            reg_pred    = context.get("regression_prediction", {})
            bucket_pred = context.get("bucket_prediction", {})

            warnings: list[str] = []

            # 1. Input validation
            missing = self._check_missing_fields(contract)
            if missing:
                warnings.append(
                    f"Missing pre-award fields (filled with 'unknown'): "
                    f"{', '.join(missing)}"
                )

            # 2. Suspicious value warnings
            warnings += self._check_suspicious_values(contract)

            # 3. Confidence scoring
            confidence_score, confidence_label = self._score_confidence(
                bucket_pred, reg_pred
            )

            # 4. Consistency check: does regression estimate land in predicted sub-range?
            consistency = self._check_consistency(reg_pred, bucket_pred)
            if not consistency["consistent"]:
                warnings.append(
                    f"Regression estimate ({fmt_dollar(reg_pred.get('point_estimate_aud', 0))}) "
                    f"falls outside predicted sub-range "
                    f"({bucket_pred.get('predicted_subrange', '?')}). "
                    "The bucket prediction is considered more reliable for a price range."
                )

            # 5. Bucket-level caveat
            bucket = bucket_pred.get("predicted_bucket", "Unknown")
            if bucket in ("Large", "Very Large"):
                warnings.append(
                    f"'{bucket}' contracts are harder to predict from pre-tender data alone "
                    f"(paper hit rate: {10 if bucket == 'Large' else 3.6}% vs 25% baseline). "
                    "Treat this prediction as directional only."
                )

            for w in warnings:
                self.warn(w)

            self.log(
                f"Confidence = {confidence_label} ({confidence_score:.2f}) | "
                f"Warnings = {len(warnings)}"
            )

            context["validation"] = {
                "is_valid":         len(missing) < len(PRE_AWARD_FEATURES),
                "missing_fields":   missing,
                "warnings":         warnings,
                "confidence_score": round(confidence_score, 3),
                "confidence_label": confidence_label,
                "consistency":      consistency,
            }
            self._finish()
        except Exception as exc:
            self._fail(exc)
            raise
        return context

    # ── Private helpers ────────────────────────────────────────────────────────

    def _check_missing_fields(self, contract: dict) -> list[str]:
        return [
            f for f in PRE_AWARD_FEATURES
            if f not in contract or contract[f] in (None, "", "unknown")
        ]

    def _check_suspicious_values(self, contract: dict) -> list[str]:
        warnings = []
        pm = str(contract.get("procurement_method", "")).lower()
        if "direct" in pm:
            warnings.append(
                "Procurement method is 'direct sourcing' — prices are "
                "negotiated rather than competed, which reduces model accuracy."
            )
        cat = str(contract.get("category_code", ""))
        if cat in ("", "unknown", "nan"):
            warnings.append(
                "No commodity category code provided — category_code is one "
                "of the strongest pre-award price signals."
            )
        return warnings

    def _score_confidence(
        self, bucket_pred: dict, reg_pred: dict
    ) -> tuple[float, str]:
        """
        Score: starts at 0.50, adjusted by:
          - Which bucket (bonus/penalty)
          - Stage 1 class probability
        """
        bucket  = bucket_pred.get("predicted_bucket", "Medium")
        s1_prob = bucket_pred.get("bucket_probability")  # None when derived from regression

        score = 0.50
        score += _BUCKET_CONFIDENCE_BONUS.get(bucket, 0.0)
        if s1_prob is not None:
            score += (s1_prob - 0.5) * 0.4   # scale probability deviation

        score = max(0.0, min(1.0, score))

        if score >= 0.65:
            label = "High"
        elif score >= 0.45:
            label = "Medium"
        elif score >= 0.25:
            label = "Low"
        else:
            label = "Very Low"

        return score, label

    def _check_consistency(self, reg_pred: dict, bucket_pred: dict) -> dict:
        point_est = reg_pred.get("point_estimate_aud", None)
        lo        = bucket_pred.get("subrange_low_aud",  None)
        hi        = bucket_pred.get("subrange_high_aud", None)

        if point_est is None or lo is None:
            return {"consistent": True, "note": "Cannot assess (missing data)."}

        hi_check = hi if hi is not None else math.inf
        inside   = lo <= point_est < hi_check
        return {
            "consistent": inside,
            "regression_estimate": fmt_dollar(point_est),
            "subrange":            f"{fmt_dollar(lo)} – {fmt_dollar(hi) if hi else '∞'}",
        }

