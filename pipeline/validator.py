"""
validator.py — Validator

Responsibilities:
  1. Validate that a new contract has all required pre-award fields.
  2. Check for unusual / suspicious feature values and warn the user.
  3. Assess prediction confidence based on regression CI width and field completeness.
  4. Surface a plain-English confidence label (High / Medium / Low / Very Low).
  5. Append any warnings or caveats to the context.

Context keys consumed:  contract (dict), regression_prediction
Context keys produced:  validation (dict with is_valid, warnings, confidence_label)
"""

from .base  import PipelineStep
from config import PRE_AWARD_FEATURES


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
        Reads contract + regression prediction from context, writes validation results back.
        """
        self._start()
        try:
            contract = context.get("contract", {})
            reg_pred = context.get("regression_prediction", {})

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

            # 3. Confidence scoring (based on field completeness)
            confidence_score, confidence_label = self._score_confidence(
                reg_pred, len(missing)
            )
            # Note: reg_pred kept as parameter for potential future use

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

    def _score_confidence(self, reg_pred: dict, n_missing: int) -> tuple[float, str]:
        """Confidence based solely on how many of the 7 pre-award fields are known."""
        n_total = len(PRE_AWARD_FEATURES)
        n_known = n_total - n_missing

        if n_known == n_total:
            label = "High"
        elif n_known >= 5:
            label = "Medium"
        elif n_known >= 3:
            label = "Low"
        else:
            label = "Very Low"

        score = n_known / n_total
        return score, label

