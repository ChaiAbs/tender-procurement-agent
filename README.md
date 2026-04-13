# Tender Price Prediction — Multi-Agent System

A multi-agent ML pipeline that predicts Australian Government contract prices from pre-award tender information only.

---

## Architecture

```
Orchestrator
├── DataAgent          — loads, cleans, one-hot-encodes tender data
├── RegressionAgent    — XGBoost regressor → point estimate + 90% CI
├── BucketAgent        — Stage 1 (bucket) + Stage 2 (sub-range) classifier
├── ValidatorAgent     — confidence scoring + input/prediction warnings
└── PresenterAgent     — formatted terminal report
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train (runs once — saves models to ./models/)
```bash
python orchestrator.py --mode train --data tenders_export.xlsx
```

### 3. Predict a new contract
```bash
python orchestrator.py --mode predict \
  --procurement_method "open tender" \
  --disposition "contract notice" \
  --is_consultancy_services "no" \
  --publisher_gov_type "FED" \
  --category_code "81111500" \
  --parent_category_code "81000000" \
  --publisher_cofog_level "2"
```

### 4. Evaluate on the dataset
```bash
python orchestrator.py --mode evaluate --data tenders_export.xlsx
```

---

## What each agent does

| Agent | Responsibility |
|-------|---------------|
| **DataAgent** | Loads Excel/CSV, filters invalid rows, cleans 7 pre-award features, one-hot encodes, log-transforms target |
| **RegressionAgent** | Trains XGBoost regressor, saves model, returns point estimate + 90% confidence interval at inference |
| **BucketAgent** | Stage 1: classifies into Small/Medium/Large/Very Large. Stage 2: predicts sub-range within bucket. Implements the paper's two-stage pipeline |
| **ValidatorAgent** | Checks for missing fields, suspicious values, assesses prediction confidence, checks consistency between regression and bucket outputs |
| **PresenterAgent** | Formats everything into a colour-coded terminal report |
| **Orchestrator** | Coordinates all agents, manages training vs. inference modes, saves/loads feature schema |

---

## Pre-award features used (no data leakage)

- `procurement_method` — open tender, limited tender, direct sourcing, etc.
- `disposition` — contract notice, amendment, standing offer
- `is_consultancy_services` — Yes / No
- `publisher_gov_type` — FED / WA / NSW / etc.
- `category_code` — 8-digit UNSPSC commodity code
- `parent_category_code` — top-level UNSPSC category
- `publisher_cofog_level` — government functional classification level

---

## Expected performance (from paper)

| Metric | Value |
|--------|-------|
| Regression R² (pre-award) | ~0.39 |
| Pipeline hit rate | ~31.8% (vs 6.25% random) |
| Small contract hit rate | ~47% (vs 25% random) |
| Large / Very Large | Unreliable — treat as directional only |
