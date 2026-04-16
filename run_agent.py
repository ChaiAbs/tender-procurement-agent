"""
run_agent.py — CLI entry point for the LangChain tender prediction agent.

Commands:

  # 1. Train the ML models (once — saves to ./models/)
  python orchestrator.py --mode train --data tenders_export.xlsx

  # 2. Index tenders into the RAG vector database (once — saves to ./rag_store/)
  python run_agent.py index --data tenders_export.xlsx

  # 3. Run the full LangChain multi-agent prediction pipeline
  python run_agent.py predict \\
      --procurement-method "open tender" \\
      --disposition "contract notice" \\
      --is-consultancy-services "no" \\
      --publisher-gov-type "FED" \\
      --category-code "81111500" \\
      --parent-category-code "81000000" \\
      --publisher-cofog-level "2" \\
      --publisher-name "Department of Defence"

Environment variables (set in .env or shell):
  ANTHROPIC_API_KEY   — required for LLM calls
  LLM_MODEL           — default: claude-sonnet-4-6
  LLM_TEMPERATURE     — default: 0.1
"""

import argparse
import json
import os
import sys

from utils import fmt_dollar

# Must be set before any native libraries (XGBoost/OpenMP) are loaded
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from dotenv import load_dotenv
load_dotenv()


# ── Subcommand: index ─────────────────────────────────────────────────────────

def cmd_index(args):
    from rag.indexer import index_tenders
    print(f"\nIndexing '{args.data}' into RAG database...")
    sz = args.sample_size if args.sample_size > 0 else None
    n = index_tenders(args.data, sample_size=sz)
    print(f"\nDone. {n:,} contracts indexed into ./rag_store/")


# ── Subcommand: predict ───────────────────────────────────────────────────────

def cmd_predict(args):
    from langchain_agents.graph import predict

    contract = {
        "procurement_method":      args.procurement_method,
        "disposition":             args.disposition,
        "is_consultancy_services": args.is_consultancy_services,
        "publisher_gov_type":      args.publisher_gov_type,
        "category_code":           args.category_code,
        "parent_category_code":    args.parent_category_code,
        "publisher_cofog_level":   args.publisher_cofog_level,
        "publisher_name":          args.publisher_name,
        "duration_days":           args.duration_days,
    }

    print("\nRunning LangChain tender prediction pipeline...")
    print("  Step 1/3 — ML critique")
    print("  Step 2/3 — RAG search + analysis")
    print("  Step 3/3 — Procurement briefing report\n")

    result = predict(contract)

    # ── Accuracy check (if actual value provided) ─────────────────────────────
    if args.actual:
        actual   = args.actual
        reg      = result.get("regression_prediction", {})
        bkt      = result.get("bucket_prediction", {})
        ml_point = reg.get("point_estimate_aud")
        ml_lo    = reg.get("ci_low_90_aud")
        ml_hi    = reg.get("ci_high_90_aud")
        sub_lo   = bkt.get("subrange_low_aud")
        sub_hi   = bkt.get("subrange_high_aud")

        sep = "=" * 70
        print(f"\n{sep}")
        print("ACCURACY CHECK")
        print(sep)
        print(f"  Actual value      {fmt_dollar(actual)}")
        print(f"  ML point estimate {fmt_dollar(ml_point)}")
        print(f"  90% CI            {fmt_dollar(ml_lo)} – {fmt_dollar(ml_hi)}")
        print(f"  Predicted range   {bkt.get('predicted_subrange', 'N/A')}")

        if ml_point:
            ape = abs(ml_point - actual) / actual * 100
            print(f"\n  Absolute error    {fmt_dollar(abs(ml_point - actual))}")
            print(f"  % error           {ape:.1f}%")

        in_ci  = ml_lo and ml_hi and ml_lo <= actual <= ml_hi
        in_sub = sub_lo is not None and sub_lo <= actual < (sub_hi or float("inf"))
        print(f"\n  Actual in 90% CI      {'YES' if in_ci  else 'NO'}")
        print(f"  Actual in sub-range   {'YES' if in_sub else 'NO'}")

        # Flag inconsistency between regression and bucket classifier
        pt_in_sub = ml_point and sub_lo is not None and sub_lo <= ml_point < (sub_hi or float("inf"))
        if not pt_in_sub:
            print(f"\n  ⚠  Model inconsistency: point estimate ({fmt_dollar(ml_point)}) falls")
            print(f"     outside the predicted sub-range ({bkt.get('predicted_subrange','?')}).")
            print(f"     The two models disagreed — treat this prediction with caution.")
        print(sep)

    if args.output:
        from exporter import export_to_word
        path = export_to_word(result, args.output)
        print(f"\nReport saved to: {path}")
        return

    if args.json:
        output = {
            "report":                result.get("report"),
            "validation_result":     result.get("validation_result"),
            "regression_prediction": result.get("regression_prediction"),
            "bucket_prediction":     result.get("bucket_prediction"),
            "similar_contracts":     result.get("similar_contracts"),
        }
        print(json.dumps(output, indent=2))
        return

    sep = "=" * 70

    print(f"\n{sep}")
    print("PROCUREMENT BRIEFING REPORT")
    print(sep)
    print(result.get("report", "No report generated."))

    if not args.report_only:
        reg = result.get("regression_prediction", {})
        bkt = result.get("bucket_prediction", {})
        if reg or bkt:
            print(f"\n{sep}")
            print("RAW MODEL OUTPUTS")
            print(sep)
            print("Regression:", json.dumps(reg, indent=2))
            print("Bucket:    ", json.dumps(bkt, indent=2))


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Tender Procurement Agent — LangChain multi-agent pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── index ──────────────────────────────────────────────────────────────────
    idx = sub.add_parser("index", help="Index tenders into the RAG database")
    idx.add_argument("--data", required=True, help="Path to tenders_export.xlsx or CSV")
    idx.add_argument(
        "--sample-size", type=int, default=50_000,
        help="Rows to index (0 = all rows, default: 50000)",
    )

    # ── predict ────────────────────────────────────────────────────────────────
    pred = sub.add_parser("predict", help="Run the full prediction pipeline")
    pred.add_argument("--procurement-method",      required=True)
    pred.add_argument("--disposition",             required=True)
    pred.add_argument("--is-consultancy-services", required=True)
    pred.add_argument("--publisher-gov-type",      required=True)
    pred.add_argument("--category-code",           default="unknown")
    pred.add_argument("--parent-category-code",    default="unknown")
    pred.add_argument("--publisher-cofog-level",   default="unknown")
    pred.add_argument("--publisher-name",          default="unknown")
    pred.add_argument("--duration-days",           type=float, default=None)
    pred.add_argument(
        "--report-only", action="store_true",
        help="Print only the final report (skip intermediate agent outputs)",
    )
    pred.add_argument(
        "--json", action="store_true",
        help="Output raw JSON instead of formatted text",
    )
    pred.add_argument(
        "--output", metavar="FILE.docx",
        help="Write the full report to a Word document (.docx)",
    )
    pred.add_argument(
        "--actual", type=float, metavar="VALUE",
        help="Actual awarded contract value in AUD — shows prediction accuracy",
    )

    args = parser.parse_args()

    if args.command == "index":
        cmd_index(args)
    elif args.command == "predict":
        cmd_predict(args)


if __name__ == "__main__":
    main()
