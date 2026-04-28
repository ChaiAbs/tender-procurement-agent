"""
evaluation/run_evaluation.py — Research evaluation for the Tender Price Prediction system.

Generates a Word document with three evaluation sections:
  1. ML Model Evaluation   — regression metrics on held-out test set
  2. RAG Evaluation        — UNSPSC retrieval accuracy
  3. End-to-End Evaluation — full pipeline: NL description → predicted price vs actual

Requirements:
  - tenders_export.xlsx must exist in project root
  - Trained models must exist in models/
  - Domain RAG index must be built (domain_rag_store/)
  - App must be running at localhost:8000 for end-to-end evaluation

Usage:
    python evaluation/run_evaluation.py --data tenders_export.xlsx
    python evaluation/run_evaluation.py --data tenders_export.xlsx --e2e-samples 20 --rag-samples 100
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MODELS_DIR, UNSPSC_FILE


# ── Helpers ────────────────────────────────────────────────────────────────────

def fmt_dollar(val: float) -> str:
    if val >= 1_000_000:
        return f"${val / 1_000_000:,.2f}M"
    elif val >= 1_000:
        return f"${val / 1_000:,.1f}K"
    return f"${val:,.0f}"


def load_unspsc_lookup() -> dict[str, str]:
    """Build a code → title lookup from the UNSPSC Excel file."""
    try:
        df = pd.read_excel(UNSPSC_FILE, dtype=str)
        col_map = {c.lower().strip(): c for c in df.columns}
        code_col  = col_map.get("code") or col_map.get("commodity code") or df.columns[0]
        title_col = col_map.get("title") or col_map.get("commodity title") or df.columns[1]
        df[code_col] = df[code_col].str.strip().str.split(".").str[0]
        return dict(zip(df[code_col], df[title_col]))
    except Exception as exc:
        print(f"[WARN] Could not load UNSPSC lookup: {exc}")
        return {}


def generate_nl_prompt(row: dict, unspsc_lookup: dict) -> str:
    """Convert ground-truth contract fields to a natural language procurement description."""
    method      = str(row.get("procurement_method", "open tender")).lower()
    disposition = str(row.get("disposition", "contract notice")).lower()
    publisher   = str(row.get("publisher_name", "a government agency"))
    duration    = row.get("duration_days")
    year        = row.get("contract_start_year", 2024)
    quarter     = row.get("contract_start_quarter", 2)

    # UNSPSC title
    code = str(row.get("category_code", "")).split(".")[0].split(".")[0]
    category_title = unspsc_lookup.get(code, "goods and services").lower()

    # Duration string
    if duration and not pd.isna(duration):
        months = round(float(duration) / 30)
        duration_str = f"approximately {months} months" if months < 24 else f"approximately {round(months/12, 1)} years"
    else:
        duration_str = "an unspecified duration"

    # Start period
    quarter_map = {1: "early", 2: "mid", 3: "mid-late", 4: "late"}
    period = f"{quarter_map.get(int(quarter) if quarter and not pd.isna(quarter) else 2, 'mid')}-{int(year)}"

    return (
        f"I need to procure {category_title} for {publisher} via {method}. "
        f"It will be a {disposition}, {duration_str} duration, starting {period}."
    )


# ── 1. ML Evaluation ───────────────────────────────────────────────────────────

def run_ml_evaluation(data_path: str) -> dict:
    """Evaluate ML models on held-out test set."""
    print("\n[1/3] Running ML evaluation …")

    from ml_evaluation.evaluator import MultiModelEvaluator, get_active_model, ordinal_encode_split
    from pipeline.data_processor import DataProcessor
    from pipeline.regressor import Regressor

    # Load stored comparison metrics
    comparison = MultiModelEvaluator.load_comparison()
    active_key = get_active_model()

    # Load and split data
    dp = DataProcessor(verbose=False)
    dp._load_ohe_schema()

    df_raw = pd.read_excel(data_path)
    df_raw = df_raw[df_raw["value"] > 0].dropna(subset=["value"])
    df_raw["log_value"] = np.log1p(df_raw["value"])

    # Rebuild test set with same random_state=42
    from sklearn.model_selection import train_test_split
    context = {"data_path": data_path}
    dp.run(context)
    X_train = context["X_train"]
    X_test  = context["X_test"]
    y_train = context["y_train"]
    y_test  = context["y_test"]
    raw_df  = context["raw_df"]

    # Predictions from active model
    regressor = Regressor(model_key=active_key, verbose=False)
    regressor.load()

    X_tr_enc, X_te_enc, encoder, cat_cols = ordinal_encode_split(X_train, X_test)

    from ml_evaluation.model_registry import MODEL_REGISTRY
    spec = MODEL_REGISTRY[active_key]
    X_eval = X_test if spec["native_categorical"] else X_te_enc

    y_pred_log = regressor.model.predict(X_eval)
    y_true_d   = np.expm1(y_test.values)
    y_pred_d   = np.expm1(y_pred_log)

    # Within-range accuracy
    ratios = np.abs(y_pred_d - y_true_d) / (y_true_d + 1)
    within_25 = float(np.mean(ratios < 0.25) * 100)
    within_50 = float(np.mean(ratios < 0.50) * 100)
    within_2x = float(np.mean((y_pred_d / (y_true_d + 1) <= 2) & (y_true_d / (y_pred_d + 1) <= 2)) * 100)

    # Median baseline (predict median value for every contract)
    median_val  = np.median(y_true_d)
    base_ratios = np.abs(median_val - y_true_d) / (y_true_d + 1)
    baseline_within_50 = float(np.mean(base_ratios < 0.50) * 100)

    # Category median baseline
    test_idx = y_test.index
    if "category_code" in raw_df.columns:
        cat_medians = raw_df.groupby("category_code")["value"].median()
        cat_preds   = raw_df.loc[test_idx, "category_code"].map(cat_medians).fillna(median_val)
        cat_ratios  = np.abs(cat_preds.values - y_true_d) / (y_true_d + 1)
        cat_baseline_within_50 = float(np.mean(cat_ratios < 0.50) * 100)
    else:
        cat_baseline_within_50 = baseline_within_50

    active_metrics = next((r for r in comparison if r.get("model_key") == active_key and r.get("status") == "ok"), {})

    print(f"  Active model: {active_key}  R²={active_metrics.get('r2', 'N/A')}  Within-50%={within_50:.1f}%")

    return {
        "active_key":             active_key,
        "active_metrics":         active_metrics,
        "all_models":             [r for r in comparison if r.get("status") == "ok"],
        "within_25pct":           within_25,
        "within_50pct":           within_50,
        "within_2x":              within_2x,
        "baseline_within_50":     baseline_within_50,
        "cat_baseline_within_50": cat_baseline_within_50,
        "n_test":                 len(y_test),
    }


# ── 2. RAG Evaluation ──────────────────────────────────────────────────────────

def run_rag_evaluation(data_path: str, n_samples: int = 100) -> dict:
    """Evaluate UNSPSC retrieval accuracy using domain RAG."""
    print(f"\n[2/3] Running RAG evaluation (n={n_samples}) …")

    from tools.domain_tools import lookup_procurement_codes

    unspsc_lookup = load_unspsc_lookup()
    if not unspsc_lookup:
        return {"error": "UNSPSC lookup not available"}

    df = pd.read_excel(data_path)
    df = df[df["value"] > 0].dropna(subset=["value", "category_code"])
    df["category_code_str"] = df["category_code"].astype(str).str.split(".").str[0]
    df = df[df["category_code_str"].isin(unspsc_lookup)]

    sample = df.sample(min(n_samples, len(df)), random_state=42)

    top1_exact = 0
    top3_exact = 0
    top1_segment = 0
    results = []

    for i, (_, row) in enumerate(sample.iterrows()):
        actual_code = row["category_code_str"]
        title = unspsc_lookup.get(actual_code, "")
        if not title or title.lower() in ("nan", ""):
            continue

        try:
            raw = lookup_procurement_codes.invoke({"description": title, "field": "category_code"})
            matches = json.loads(raw)
            if isinstance(matches, list) and matches:
                top1_code = str(matches[0].get("category_code", "")).split(".")[0]
                top3_codes = [str(m.get("category_code", "")).split(".")[0] for m in matches[:3]]

                exact_top1 = top1_code == actual_code
                exact_top3 = actual_code in top3_codes
                seg_top1   = top1_code[:2] == actual_code[:2]

                if exact_top1: top1_exact += 1
                if exact_top3: top3_exact += 1
                if seg_top1:   top1_segment += 1

                results.append({
                    "actual_code":    actual_code,
                    "actual_title":   title,
                    "top1_code":      top1_code,
                    "top1_title":     matches[0].get("name", ""),
                    "exact_top1":     exact_top1,
                    "exact_top3":     exact_top3,
                    "segment_match":  seg_top1,
                    "similarity":     matches[0].get("similarity", 0),
                })

            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(sample)} — top-1: {top1_exact/(i+1)*100:.1f}%")

        except Exception as exc:
            print(f"  [WARN] RAG error for code {actual_code}: {exc}")

    n = len(results)
    return {
        "n_samples":        n,
        "top1_exact_pct":   round(top1_exact / n * 100, 1) if n else 0,
        "top3_exact_pct":   round(top3_exact / n * 100, 1) if n else 0,
        "top1_segment_pct": round(top1_segment / n * 100, 1) if n else 0,
        "results":          results[:20],  # store first 20 for report table
    }


# ── 3. End-to-End Evaluation ───────────────────────────────────────────────────

def run_e2e_evaluation(data_path: str, n_samples: int = 50, api_url: str = "http://localhost:8000") -> dict:
    """Evaluate the full pipeline: NL description → predicted price vs actual."""
    print(f"\n[3/3] Running end-to-end evaluation (n={n_samples}) …")

    unspsc_lookup = load_unspsc_lookup()

    df = pd.read_excel(data_path)
    df = df[df["value"] > 0].dropna(subset=["value", "category_code", "publisher_name", "procurement_method"])
    df["category_code_str"] = df["category_code"].astype(str).str.split(".").str[0]

    sample = df.sample(min(n_samples, len(df)), random_state=7)
    results = []

    for i, (_, row) in enumerate(sample.iterrows()):
        actual_value = float(row["value"])
        prompt = generate_nl_prompt(row.to_dict(), unspsc_lookup)

        try:
            # Start a fresh session for each contract
            resp = requests.post(
                f"{api_url}/api/chat",
                json={"message": prompt},
                timeout=180,
            )
            resp.raise_for_status()
            data = resp.json()
            session_id = data.get("session_id")

            prediction = data.get("prediction")
            if not prediction:
                # Model might need a follow-up — try once more
                resp2 = requests.post(
                    f"{api_url}/api/chat",
                    json={"message": "please run the prediction now", "session_id": session_id},
                    timeout=180,
                )
                resp2.raise_for_status()
                data = resp2.json()
                prediction = data.get("prediction")

            if prediction and prediction.get("regression"):
                predicted_value = float(prediction["regression"].get("point_estimate_aud", 0))
                ratio = abs(predicted_value - actual_value) / (actual_value + 1)

                results.append({
                    "prompt":          prompt[:120] + "…",
                    "actual":          actual_value,
                    "predicted":       predicted_value,
                    "actual_fmt":      fmt_dollar(actual_value),
                    "predicted_fmt":   fmt_dollar(predicted_value),
                    "pct_error":       round(ratio * 100, 1),
                    "within_25":       ratio < 0.25,
                    "within_50":       ratio < 0.50,
                    "within_2x":       (predicted_value / (actual_value + 1) <= 2) and (actual_value / (predicted_value + 1) <= 2),
                })
            else:
                results.append({
                    "prompt":    prompt[:120] + "…",
                    "actual":    actual_value,
                    "predicted": None,
                    "error":     "No prediction returned",
                })

            # Clear session
            if session_id:
                requests.delete(f"{api_url}/api/session/{session_id}", timeout=10)

            print(f"  {i+1}/{len(sample)} — actual={fmt_dollar(actual_value)}  predicted={fmt_dollar(predicted_value) if prediction else 'N/A'}")
            time.sleep(2)  # avoid rate limiting

        except Exception as exc:
            print(f"  [WARN] E2E error for contract {i+1}: {exc}")
            results.append({"prompt": prompt[:80], "error": str(exc)})

    valid = [r for r in results if r.get("predicted") is not None]
    n = len(valid)

    return {
        "n_samples":      len(results),
        "n_successful":   n,
        "within_25pct":   round(sum(r["within_25"] for r in valid) / n * 100, 1) if n else 0,
        "within_50pct":   round(sum(r["within_50"] for r in valid) / n * 100, 1) if n else 0,
        "within_2x":      round(sum(r["within_2x"] for r in valid) / n * 100, 1) if n else 0,
        "median_pct_error": round(float(np.median([r["pct_error"] for r in valid])), 1) if n else 0,
        "results":        results,
    }


# ── Report Generation ──────────────────────────────────────────────────────────

def generate_report(ml: dict, rag: dict, e2e: dict, output_path: str) -> None:
    """Generate a Word document with all evaluation results."""
    from docx import Document
    from docx.shared import Pt, RGBColor, Inches
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    doc = Document()

    # Title
    title = doc.add_heading("Tender Price Prediction — Evaluation Report", 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_paragraph(
        "This report evaluates the Tender Price Prediction system across three dimensions: "
        "ML model performance, domain RAG retrieval accuracy, and full end-to-end pipeline accuracy."
    )
    doc.add_paragraph()

    # ── Section 1: ML Evaluation ──────────────────────────────────────────────
    doc.add_heading("1. ML Model Evaluation", level=1)
    doc.add_paragraph(
        "The ML pipeline was evaluated on a held-out test set (20% of the full dataset, "
        "random_state=42). The target variable is the contract awarded value in AUD. "
        "All models were trained on the same split using log1p-transformed values."
    )

    # All models comparison table
    doc.add_heading("1.1 Model Comparison", level=2)
    all_models = ml.get("all_models", [])
    if all_models:
        headers = ["Model", "R²", "RMSE (log)", "MAE (log)", "MAE ($)", "Within 50%", "Train time (s)"]
        table = doc.add_table(rows=1, cols=len(headers))
        table.style = "Table Grid"
        hdr = table.rows[0].cells
        for i, h in enumerate(headers):
            hdr[i].text = h
            hdr[i].paragraphs[0].runs[0].bold = True
        for m in all_models:
            row_cells = table.add_row().cells
            row_cells[0].text = m.get("display_name", m.get("model_key", ""))
            row_cells[1].text = str(m.get("r2", ""))
            row_cells[2].text = str(m.get("rmse_log", ""))
            row_cells[3].text = str(m.get("mae_log", ""))
            row_cells[4].text = f"${m.get('mae_dollar', 0):,.0f}" if m.get("mae_dollar") else ""
            row_cells[5].text = f"{m.get('within_50pct', '')}%"
            row_cells[6].text = str(m.get("train_time_s", ""))

    doc.add_paragraph()

    # Within-range accuracy table
    doc.add_heading("1.2 Within-Range Accuracy (Active Model)", level=2)
    doc.add_paragraph(
        f"Active model: {ml.get('active_key', 'N/A')}. "
        f"Test set size: {ml.get('n_test', 'N/A'):,} contracts."
    )

    acc_table = doc.add_table(rows=1, cols=4)
    acc_table.style = "Table Grid"
    acc_headers = ["Metric", "Active Model", "Global Median Baseline", "Category Median Baseline"]
    for i, h in enumerate(acc_headers):
        acc_table.rows[0].cells[i].text = h
        acc_table.rows[0].cells[i].paragraphs[0].runs[0].bold = True

    rows_data = [
        ("Within 25% of actual",  f"{ml.get('within_25pct', 0):.1f}%",  "—",  "—"),
        ("Within 50% of actual",  f"{ml.get('within_50pct', 0):.1f}%",  f"{ml.get('baseline_within_50', 0):.1f}%", f"{ml.get('cat_baseline_within_50', 0):.1f}%"),
        ("Within 2× of actual",   f"{ml.get('within_2x', 0):.1f}%",     "—",  "—"),
    ]
    for rd in rows_data:
        r = acc_table.add_row().cells
        for i, v in enumerate(rd):
            r[i].text = v

    doc.add_paragraph()

    # ── Section 2: RAG Evaluation ─────────────────────────────────────────────
    doc.add_heading("2. Domain RAG Evaluation", level=1)
    doc.add_paragraph(
        "The domain RAG index was evaluated on its ability to retrieve the correct UNSPSC "
        "commodity code given a natural language description of the procurement. "
        "Ground truth codes were taken from the training dataset. For each sampled contract, "
        "the UNSPSC title corresponding to its category_code was used as the query."
    )

    if "error" in rag:
        doc.add_paragraph(f"Evaluation skipped: {rag['error']}")
    else:
        doc.add_heading("2.1 Retrieval Accuracy", level=2)
        doc.add_paragraph(f"Sample size: {rag.get('n_samples', 0)} contracts.")

        rag_table = doc.add_table(rows=1, cols=2)
        rag_table.style = "Table Grid"
        rag_table.rows[0].cells[0].text = "Metric"
        rag_table.rows[0].cells[1].text = "Accuracy"
        rag_table.rows[0].cells[0].paragraphs[0].runs[0].bold = True
        rag_table.rows[0].cells[1].paragraphs[0].runs[0].bold = True

        rag_rows = [
            ("Top-1 exact match",    f"{rag.get('top1_exact_pct', 0)}%"),
            ("Top-3 exact match",    f"{rag.get('top3_exact_pct', 0)}%"),
            ("Top-1 segment match (first 2 digits)", f"{rag.get('top1_segment_pct', 0)}%"),
        ]
        for rr in rag_rows:
            r = rag_table.add_row().cells
            r[0].text = rr[0]
            r[1].text = rr[1]

        doc.add_paragraph()

        # Sample results table
        doc.add_heading("2.2 Sample Results (first 10)", level=2)
        sample_results = rag.get("results", [])[:10]
        if sample_results:
            t = doc.add_table(rows=1, cols=4)
            t.style = "Table Grid"
            h = t.rows[0].cells
            for i, hh in enumerate(["Actual Code", "Actual Title", "Retrieved Code", "Match"]):
                h[i].text = hh
                h[i].paragraphs[0].runs[0].bold = True
            for sr in sample_results:
                r = t.add_row().cells
                r[0].text = sr.get("actual_code", "")
                r[1].text = str(sr.get("actual_title", ""))[:40]
                r[2].text = sr.get("top1_code", "")
                r[3].text = "✓" if sr.get("exact_top1") else ("~" if sr.get("segment_match") else "✗")

    doc.add_paragraph()

    # ── Section 3: End-to-End Evaluation ─────────────────────────────────────
    doc.add_heading("3. End-to-End Evaluation", level=1)
    doc.add_paragraph(
        "The full pipeline was evaluated end-to-end: a natural language procurement description "
        "was generated from ground-truth contract fields and submitted to the conversational agent. "
        "The agent extracted fields via dialogue, resolved UNSPSC codes via the domain RAG, "
        "and ran the ML prediction pipeline. The predicted price was compared to the actual awarded value."
    )

    doc.add_heading("3.1 Summary Metrics", level=2)
    doc.add_paragraph(
        f"Sample size: {e2e.get('n_samples', 0)} contracts. "
        f"Successful predictions: {e2e.get('n_successful', 0)}."
    )

    e2e_table = doc.add_table(rows=1, cols=2)
    e2e_table.style = "Table Grid"
    e2e_table.rows[0].cells[0].text = "Metric"
    e2e_table.rows[0].cells[1].text = "Value"
    e2e_table.rows[0].cells[0].paragraphs[0].runs[0].bold = True
    e2e_table.rows[0].cells[1].paragraphs[0].runs[0].bold = True

    e2e_rows = [
        ("Within 25% of actual",       f"{e2e.get('within_25pct', 0)}%"),
        ("Within 50% of actual",       f"{e2e.get('within_50pct', 0)}%"),
        ("Within 2× of actual",        f"{e2e.get('within_2x', 0)}%"),
        ("Median % error",             f"{e2e.get('median_pct_error', 0)}%"),
    ]
    for er in e2e_rows:
        r = e2e_table.add_row().cells
        r[0].text = er[0]
        r[1].text = er[1]

    doc.add_paragraph()

    # Sample predictions table
    doc.add_heading("3.2 Sample Predictions", level=2)
    valid_results = [r for r in e2e.get("results", []) if r.get("predicted") is not None][:15]
    if valid_results:
        t = doc.add_table(rows=1, cols=4)
        t.style = "Table Grid"
        h = t.rows[0].cells
        for i, hh in enumerate(["Actual Value", "Predicted Value", "% Error", "Within 50%"]):
            h[i].text = hh
            h[i].paragraphs[0].runs[0].bold = True
        for vr in valid_results:
            r = t.add_row().cells
            r[0].text = vr.get("actual_fmt", "")
            r[1].text = vr.get("predicted_fmt", "")
            r[2].text = f"{vr.get('pct_error', 0):.1f}%"
            r[3].text = "✓" if vr.get("within_50") else "✗"

    doc.save(output_path)
    print(f"\n[✓] Report saved → {output_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run evaluation and generate report")
    parser.add_argument("--data",         default="tenders_export.xlsx", help="Path to tenders Excel file")
    parser.add_argument("--output",       default="evaluation_report.docx", help="Output Word document path")
    parser.add_argument("--rag-samples",  type=int, default=100, help="Number of contracts for RAG evaluation")
    parser.add_argument("--e2e-samples",  type=int, default=50,  help="Number of contracts for E2E evaluation")
    parser.add_argument("--api-url",      default="http://localhost:8000", help="App URL for E2E evaluation")
    parser.add_argument("--skip-e2e",     action="store_true", help="Skip end-to-end evaluation")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"Error: data file not found: {args.data}")
        sys.exit(1)

    print(f"Starting evaluation — data: {args.data}")
    t_start = time.time()

    ml_results  = run_ml_evaluation(args.data)
    rag_results = run_rag_evaluation(args.data, n_samples=args.rag_samples)

    if args.skip_e2e:
        e2e_results = {"n_samples": 0, "n_successful": 0, "results": [], "skipped": True}
        print("\n[3/3] End-to-end evaluation skipped (--skip-e2e)")
    else:
        e2e_results = run_e2e_evaluation(args.data, n_samples=args.e2e_samples, api_url=args.api_url)

    generate_report(ml_results, rag_results, e2e_results, args.output)
    print(f"\nTotal time: {(time.time() - t_start) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
