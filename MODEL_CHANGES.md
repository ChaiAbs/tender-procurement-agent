# Model & Architecture Changes — Tender Price Prediction Agent

## Overview

This document summarises all model and architectural changes made since the initial commit, in chronological order. Includes what changed, why, and the measured impact where available.

---

## 1. Initial Architecture

**Baseline system at first commit:**

- **Model:** XGBoost regressor + two-stage XGBoost bucket classifier
- **Features:** 7 pre-award categoricals encoded with one-hot encoding (OHE)
  - `procurement_method`, `disposition`, `is_consultancy_services`, `publisher_gov_type`, `category_code`, `parent_category_code`, `publisher_cofog_level`
- **Pipeline:** DataProcessor → Regressor → BucketClassifier → Validator → Presenter
- **Agent:** Single reporting node (no RAG integration in report)
- **R²:** ~0.38

---

## 2. Three-Node LangGraph Pipeline Restored

**Change:** Restored the full three-node LangGraph pipeline that had been collapsed to a single node.

**Nodes:**
- `ml_critique` — assesses plausibility of ML outputs, identifies risks, no RAG
- `analysis` — performs RAG search, interprets similar historical contracts
- `reporting` — synthesises all prior outputs into final procurement briefing

**Why:** The single-node pipeline was bypassing RAG entirely — the report was being generated without any similar historical contracts. Restoring the three-node topology fixed RAG population in the report.

**Impact:** Qualitative — reports now include relevant historical contract comparisons.

---

## 3. RAG Fix — Full Graph Invocation

**Change:** `app.py` was calling `reporting_node` directly instead of invoking the full graph. Fixed to call `get_graph().invoke(initial)`.

**Why:** Direct node call passed `similar_contracts: []`, so the analysis node was never reached and RAG results never made it into the report.

**Impact:** RAG results now consistently appear in all reports.

---

## 4. Duration Days Added as ML Feature

**Change:** Added `duration_days` as a numeric feature derived from `contract_start` and `contract_end`.

**Implementation:**
- `DataProcessor._compute_duration()` computes duration, caps at 3,650 days (10 years), fills missing with training median
- `preprocess_single()` accepts intended duration at inference, falls back to median if not provided
- User asked for "intended contract duration from the procurement plan"

**Why:** Contract duration is knowable pre-award (agencies plan term before going to market). It is a strong proxy for contract scope and strongly correlated with value.

**R² impact:** 0.38 → 0.52 (+0.14)

---

## 5. Bucket Classifier Removed

**Change:** Replaced the two-stage XGBoost bucket classifier with a deterministic `predict_from_regression()` function.

**Why:** The classifier was performing worse than random on Large and Very Large contracts. It added complexity and caused model inconsistency warnings (point estimate and bucket prediction disagreeing). Deriving the bucket directly from the regression point estimate eliminates the inconsistency entirely.

**Implementation:**
- `predict_from_regression(point_estimate_aud)` in `bucket_classifier.py` maps the regression output to a bucket and sub-range using the fixed `BUCKET_RANGES` and `SUBRANGE_DEFINITIONS` from `config.py`
- Training pipeline reduced to: DataProcessor → Regressor (2 steps instead of 3)
- Validator updated to handle `bucket_probability: None`

**Impact:** Eliminated model inconsistency warnings. Training faster. Hit rate on evaluate: 16% vs 6.25% random baseline.

---

## 6. OHE Replaced with XGBoost Native Categorical Encoding

**Change:** Replaced `pd.get_dummies` (one-hot encoding) with XGBoost native categorical support (`pd.Categorical` + `enable_categorical=True`).

**Why:** OHE on high-cardinality features like `publisher_name` was creating 700–1,000+ columns on 1M rows (~3GB feature matrix). Native categoricals keep the matrix at 10 columns (~40MB). Prior attempt at target encoding had hurt performance significantly.

**Implementation:**
- `DataProcessor._encode_features()` now uses `pd.Categorical` with `_category_dtypes` dict
- Rare values (< 50 occurrences) collapsed to `col_other` per feature
- Schema pickle updated to store `category_dtypes` key
- XGBoost params: added `tree_method="hist"`, `enable_categorical=True`, `nthread=-1`

**Impact:** ~4-5x training speedup. Memory usage reduced from ~3GB to ~40MB for feature matrix.

---

## 7. Publisher Name and Portfolio Added as Features

**Change:** Added `publisher_name` and `publisher_portfolio` as categorical features.

**Implementation:**
- `publisher_name` collected from user at inference
- `publisher_portfolio` and `publisher_cofog_level` auto-derived at inference from a lookup table (`publisher_lookup.pkl`) built during training
- Lookup maps `publisher_name → {portfolio, cofog_level}` using most common value per agency
- `CATEGORICAL_FEATURES` in `config.py` expanded from 7 to 9 features

**Why:** Agency identity is a strong predictor — Defence contracts behave very differently from local council contracts of the same category. Variance explained by `publisher_name` alone: 0.185.

**R² impact:** 0.52 → 0.59 (+0.07)

---

## 8. Cloud Run Cold-Start Fix

**Change:** Pre-downloaded the ChromaDB ONNX embedding model (79MB) into the Docker image at build time.

**Implementation:**
- `Dockerfile`: added `RUN python -c "from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2; ONNXMiniLM_L6_V2()"` before `COPY . .`
- `app.py`: added `@app.on_event("startup")` to warm the model on container start
- `cloudbuild.yaml`: Kaniko cache disabled temporarily to force clean rebuild, then re-enabled

**Why:** On cold starts, ChromaDB was downloading the ONNX model mid-request, causing Cloud Run to timeout before returning a response.

**Impact:** Eliminated cold-start timeouts on Cloud Run.

---

## 9. Reporting Prompt — ML Estimate as Primary Anchor

**Change:** Updated `reporting_agent.yaml` to explicitly make the ML point estimate the primary anchor for the recommendation, with RAG results as supporting evidence only.

**Why:** The LLM was overriding the ML prediction based on a single historical contract from RAG, giving users a recommended range inconsistent with the model output (e.g. model says $925K, report recommends $1.5M–$3M based on one similar contract).

**Rule added:** Only adjust the recommendation away from the ML range if 3 or more similar contracts consistently point to a different range.

---

## 10. Contract Start Year and Quarter Added as Features

**Change:** Added `contract_start_year` and `contract_start_quarter` as numeric features derived from the existing `contract_start` column.

**Why:** Contract values grow over time due to inflation (~25–30% cumulative over 10 years). Year captures both the inflation trend and discrete policy/spending regime shifts. Quarter captures end-of-financial-year (Q4) budget rush patterns common in Australian Government procurement.

**Implementation:**
- `DataProcessor._compute_duration()` extended to derive `contract_start_year` and `contract_start_quarter` from `contract_start`
- At inference, year/quarter default to current date if `contract_start` not provided
- `NUMERIC_FEATURES` in `config.py` expanded from 1 to 3 features

**R² impact:** 0.59 → 0.607 (+0.017)

---

## 11. Quantile Regression Evaluated and Rejected

**Change:** Tested XGBoost quantile regression (median, alpha=0.5) with separate lower (0.05) and upper (0.95) models for CI bounds, to address systematic overestimation caused by right-skewed contract value distribution.

**Result:** MAE(log) was identical (0.90 vs 0.9066) between median and mean regression, while R² dropped from 0.607 to 0.573 and training time tripled (3 models vs 1). No meaningful accuracy improvement.

**Decision:** Reverted to mean (MSE) regression. Overestimation is noted as a known limitation of the training distribution rather than a fixable modelling choice.

---

## Current State

| Metric | Value |
|---|---|
| R² | 0.607 |
| Features | 9 categorical + 3 numeric (12 total) |
| Training rows | ~1M |
| Sub-range hit rate | 16% (vs 6.25% random) |
| Encoding | XGBoost native categoricals |
| Bucket assignment | Deterministic from regression |
| Agent nodes | ml_critique → analysis → reporting |

---

## Known Limitations

- **Systematic overestimation** — model targets the mean, inflated by large-contract outliers. Quantile regression was tested but offered no improvement.
- **Missing category_code** — one of the strongest features; missing at inference significantly reduces accuracy.
- **Fundamental ceiling** — contract value is partly determined by scope, negotiations, and supplier bids, none of which are knowable pre-award. Estimated practical ceiling is R² ~0.65–0.70.
