"""
bucket_classifier.py — BucketClassifier

Implements the two-stage hierarchical classification pipeline described in the paper.

Stage 1 — assigns a contract to one of four broad buckets:
    Small (<$50K), Medium ($50K–$500K), Large ($500K–$5M), Very Large (>$5M)

Stage 2 — within the predicted bucket, assigns a sub-range (one of four),
    giving a 16-way prediction overall.

At training time both stages are fit on the full training set and saved.
At inference time both stages are loaded from disk.

Context keys consumed:  X_train, X_test, y_train, y_test, raw_df  (training)
                        X_infer                                     (inference)
Context keys produced:  bucket_metrics, bucket_prediction
"""

import os
import pickle

import numpy  as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from .base  import PipelineStep
from config import (
    MODELS_DIR, XGBOOST_CLASSIFIER_PARAMS,
    BUCKET_ORDER, BUCKET_RANGES, SUBRANGE_DEFINITIONS, TARGET_COLUMN,
)


STAGE1_MODEL_PATH  = os.path.join(MODELS_DIR, "stage1_classifier.pkl")
STAGE1_LABELS_PATH = os.path.join(MODELS_DIR, "stage1_labels.pkl")
STAGE2_MODELS_PATH = os.path.join(MODELS_DIR, "stage2_classifiers.pkl")


def _assign_bucket(value: float) -> str:
    for bucket, (lo, hi) in BUCKET_RANGES.items():
        if lo <= value < hi:
            return bucket
    return "Very Large"


def _assign_subrange_label(bucket: str, value: float) -> int:
    for idx, (_, lo, hi) in enumerate(SUBRANGE_DEFINITIONS[bucket]):
        if lo <= value < hi:
            return idx
    return len(SUBRANGE_DEFINITIONS[bucket]) - 1


class BucketClassifier(PipelineStep):
    """Two-stage hierarchical price-range classifier."""

    def __init__(self, verbose: bool = True):
        super().__init__(name="BucketClassifier", verbose=verbose)
        self.stage1_model   = None
        self.stage1_classes: list[str] = BUCKET_ORDER
        self.stage2_models: dict       = {}   # bucket → fitted classifier

    # ── Public interface ───────────────────────────────────────────────────────

    def run(self, context: dict) -> dict:
        """Training mode: fit Stage 1 + Stage 2, evaluate, save."""
        self._start()
        try:
            from xgboost import XGBClassifier

            X_train  = context["X_train"]
            X_test   = context["X_test"]
            raw_df   = context["raw_df"]
            n_train  = len(X_train)

            # Build all labels first
            values_all   = raw_df[TARGET_COLUMN].values
            bucket_all   = np.array([_assign_bucket(v) for v in values_all])
            bucket_train = bucket_all[:n_train]
            bucket_test  = bucket_all[n_train:]

            subrange_all   = np.array(
                [_assign_subrange_label(b, v) for b, v in zip(bucket_all, values_all)]
            )
            subrange_train = subrange_all[:n_train]

            # Cap training size for speed — 50k rows is enough for good accuracy
            MAX_TRAIN = 200_000
            if n_train > MAX_TRAIN:
                idx = np.random.RandomState(42).choice(n_train, MAX_TRAIN, replace=False)
                X_train        = X_train.iloc[idx]
                bucket_train   = bucket_train[idx]
                subrange_train = subrange_train[idx]
                self.log(f"Sampled {MAX_TRAIN:,} / {n_train:,} rows for bucket classifier training")

            # ── Stage 1 ──────────────────────────────────────────────────────
            self.log("Training Stage 1 bucket classifier …")
            label_map = {b: i for i, b in enumerate(BUCKET_ORDER)}
            y1_train  = np.array([label_map[b] for b in bucket_train])
            y1_test   = np.array([label_map[b] for b in bucket_test])

            self.stage1_model = XGBClassifier(
                **XGBOOST_CLASSIFIER_PARAMS,
                num_class=4,
                objective="multi:softprob",
            )
            self.stage1_model.fit(X_train, y1_train)

            y1_pred = self.stage1_model.predict(X_test)
            s1_acc  = round(accuracy_score(y1_test, y1_pred), 4)
            s1_f1   = round(f1_score(y1_test, y1_pred, average="macro"), 4)
            self.log(f"Stage 1 — Accuracy={s1_acc:.4f} | Macro-F1={s1_f1:.4f}")

            # ── Stage 2 ──────────────────────────────────────────────────────
            self.log("Training Stage 2 within-bucket classifiers …")

            s2_metrics: dict[str, dict] = {}
            for bucket in BUCKET_ORDER:
                mask   = bucket_train == bucket
                n_rows = mask.sum()
                if n_rows < 50:
                    self.warn(f"  Bucket '{bucket}' has only {n_rows} rows — skipping Stage 2.")
                    continue

                Xb = X_train[mask]
                yb = subrange_train[mask]

                clf = XGBClassifier(
                    **XGBOOST_CLASSIFIER_PARAMS,
                    num_class=4,
                    objective="multi:softprob",
                )
                clf.fit(Xb, yb)
                self.stage2_models[bucket] = clf

                # Quick train-set eval (no separate validation per bucket)
                yb_pred = clf.predict(Xb)
                s2_metrics[bucket] = {
                    "accuracy": round(accuracy_score(yb, yb_pred), 4),
                    "macro_f1": round(f1_score(yb, yb_pred, average="macro"), 4),
                    "n_train":  int(n_rows),
                }
                self.log(
                    f"  Stage 2 [{bucket}] — "
                    f"Accuracy={s2_metrics[bucket]['accuracy']:.4f} | "
                    f"n={n_rows:,}"
                )

            self._save()
            context["bucket_metrics"] = {
                "stage1_accuracy": s1_acc,
                "stage1_f1":       s1_f1,
                "stage2":          s2_metrics,
            }
            self._finish()
        except Exception as exc:
            self._fail(exc)
            raise
        return context

    def predict(self, X_infer: pd.DataFrame) -> dict:
        """
        Inference mode.
        Returns predicted bucket, sub-range label, and Stage 1 class probabilities.
        """
        if self.stage1_model is None:
            self._load()

        # Stage 1 — predict bucket
        s1_probs  = self.stage1_model.predict_proba(X_infer)[0]
        s1_label  = int(self.stage1_model.predict(X_infer)[0])
        bucket    = BUCKET_ORDER[s1_label]
        bucket_confidence = round(float(s1_probs[s1_label]), 4)

        # Stage 2 — predict sub-range within bucket
        clf = self.stage2_models.get(bucket)
        if clf is None:
            subrange_idx  = 0
            subrange_conf = 0.0
        else:
            s2_probs     = clf.predict_proba(X_infer)[0]
            subrange_idx = int(clf.predict(X_infer)[0])
            subrange_conf = round(float(s2_probs[subrange_idx]), 4)

        subrange_name, lo, hi = SUBRANGE_DEFINITIONS[bucket][subrange_idx]

        # All bucket probs for transparency
        bucket_probs = {
            BUCKET_ORDER[i]: round(float(p), 4)
            for i, p in enumerate(s1_probs)
        }

        return {
            "predicted_bucket":      bucket,
            "bucket_probability":    bucket_confidence,
            "all_bucket_probs":      bucket_probs,
            "predicted_subrange":    subrange_name,
            "subrange_probability":  subrange_conf,
            "subrange_low_aud":      lo,
            "subrange_high_aud":     hi if hi != float("inf") else None,
        }

    # ── Private helpers ────────────────────────────────────────────────────────

    def _save(self) -> None:
        with open(STAGE1_MODEL_PATH, "wb") as f:
            pickle.dump(self.stage1_model, f)
        with open(STAGE2_MODELS_PATH, "wb") as f:
            pickle.dump(self.stage2_models, f)
        self.log(f"Stage 1 + Stage 2 models saved → {MODELS_DIR}")

    def _load(self) -> None:
        for path, label in [
            (STAGE1_MODEL_PATH, "Stage 1"),
            (STAGE2_MODELS_PATH, "Stage 2"),
        ]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"No trained {label} model found at {path}. Run training first."
                )
        with open(STAGE1_MODEL_PATH, "rb") as f:
            self.stage1_model = pickle.load(f)
        with open(STAGE2_MODELS_PATH, "rb") as f:
            self.stage2_models = pickle.load(f)
        self.log("Stage 1 + Stage 2 models loaded from disk.")
