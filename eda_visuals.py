"""
eda_visuals.py — Exploratory Data Analysis for Tendertrace dataset.
Generates PNG figures saved to eda_output/.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.colors import LogNorm

warnings.filterwarnings("ignore")
os.makedirs("eda_output", exist_ok=True)

STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "font.family":      "sans-serif",
    "axes.titlesize":   13,
    "axes.labelsize":   11,
}
plt.rcParams.update(STYLE)
BLUE = "#2563EB"
ORANGE = "#F59E0B"
GREEN = "#10B981"
RED = "#EF4444"
PALETTE = [BLUE, ORANGE, GREEN, RED, "#8B5CF6", "#EC4899", "#14B8A6", "#F97316", "#6366F1", "#84CC16"]

print("Loading tenders_export.xlsx …")
df_raw = pd.read_excel("tenders_export.xlsx")
print(f"Loaded {len(df_raw):,} rows × {df_raw.shape[1]} columns")

# ── 1. MISSING VALUES HEATMAP ─────────────────────────────────────────────────
print("Plot 1: Missing values …")
missing = df_raw.isnull().mean() * 100
missing = missing.sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(12, 6))
colors = [RED if v > 30 else ORANGE if v > 10 else BLUE for v in missing.values]
bars = ax.barh(missing.index, missing.values, color=colors, edgecolor="none", height=0.7)
ax.set_xlabel("Missing (%)", fontsize=11)
ax.set_title("Missing Values by Column", fontsize=14, fontweight="bold", pad=12)
ax.invert_yaxis()
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
for bar, val in zip(bars, missing.values):
    if val > 0.5:
        ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", ha="left", fontsize=8)
# legend
from matplotlib.patches import Patch
legend_els = [Patch(fc=RED, label=">30% missing"), Patch(fc=ORANGE, label="10–30%"), Patch(fc=BLUE, label="<10%")]
ax.legend(handles=legend_els, loc="lower right", fontsize=9, framealpha=0.8)
fig.tight_layout()
fig.savefig("eda_output/01_missing_values.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  saved 01_missing_values.png")

# ── clean for subsequent plots ────────────────────────────────────────────────
df = df_raw.copy()
df = df[df["value"].notna()]
df["value"] = pd.to_numeric(df["value"], errors="coerce")
df = df[df["value"] > 0].copy()
df["contract_start"] = pd.to_datetime(df.get("contract_start"), errors="coerce")
df["contract_end"]   = pd.to_datetime(df.get("contract_end"),   errors="coerce")
df["year"]           = df["contract_start"].dt.year
df["duration_days"]  = (df["contract_end"] - df["contract_start"]).dt.days.clip(1, 3650)
df["log_value"]      = np.log10(df["value"])

for col in ["procurement_method", "disposition", "publisher_gov_type",
            "publisher_cofog_level", "publisher_name", "category_code"]:
    if col in df.columns:
        df[col] = df[col].fillna("unknown").astype(str).str.strip().str.lower()

print(f"After filtering: {len(df):,} valid contracts")

# ── 2. CONTRACT VALUE DISTRIBUTION ───────────────────────────────────────────
print("Plot 2: Contract value distribution …")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: histogram of log10 values
ax = axes[0]
ax.hist(df["log_value"], bins=80, color=BLUE, edgecolor="white", linewidth=0.3)
ax.set_xlabel("Contract Value (log₁₀ AUD)", fontsize=11)
ax.set_ylabel("Count", fontsize=11)
ax.set_title("Distribution of Contract Values (log scale)", fontsize=13, fontweight="bold")
xticks = [3, 4, 5, 6, 7, 8, 9, 10]
ax.set_xticks(xticks)
ax.set_xticklabels([f"$10^{{{x}}}$" for x in xticks], fontsize=9)
# annotate median
med = np.median(df["value"])
ax.axvline(np.log10(med), color=ORANGE, lw=2, linestyle="--", label=f"Median ${med:,.0f}")
ax.legend(fontsize=9)

# Right: size bucket pie
ax = axes[1]
buckets = pd.cut(df["value"],
    bins=[0, 50_000, 500_000, 5_000_000, float("inf")],
    labels=["Small\n(<$50K)", "Medium\n($50K–$500K)", "Large\n($500K–$5M)", "Very Large\n(>$5M)"])
counts = buckets.value_counts().sort_index()
wedge_colors = [BLUE, GREEN, ORANGE, RED]
wedges, texts, autotexts = ax.pie(
    counts, labels=counts.index, autopct="%1.1f%%",
    colors=wedge_colors, startangle=90,
    wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    textprops={"fontsize": 10}
)
for at in autotexts:
    at.set_fontsize(9)
    at.set_fontweight("bold")
ax.set_title("Contract Size Distribution", fontsize=13, fontweight="bold")

fig.tight_layout()
fig.savefig("eda_output/02_value_distribution.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  saved 02_value_distribution.png")

# ── 3. CONTRACTS OVER TIME ────────────────────────────────────────────────────
print("Plot 3: Contracts over time …")
yearly = df.groupby("year").agg(
    n_contracts=("value", "count"),
    total_value=("value", "sum"),
    median_value=("value", "median"),
).reset_index()
yearly = yearly[(yearly["year"] >= 2010) & (yearly["year"] <= 2025)]

fig, ax1 = plt.subplots(figsize=(13, 5))
ax2 = ax1.twinx()
ax1.bar(yearly["year"], yearly["n_contracts"], color=BLUE, alpha=0.7, label="# Contracts")
ax2.plot(yearly["year"], yearly["median_value"] / 1000, color=ORANGE, lw=2.5, marker="o", ms=5, label="Median value (AUD K)")
ax1.set_xlabel("Year", fontsize=11)
ax1.set_ylabel("Number of Contracts", fontsize=11, color=BLUE)
ax2.set_ylabel("Median Contract Value (AUD K)", fontsize=11, color=ORANGE)
ax1.set_title("Australian Government Procurement Activity Over Time", fontsize=13, fontweight="bold")
ax1.tick_params(axis="y", labelcolor=BLUE)
ax2.tick_params(axis="y", labelcolor=ORANGE)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)
fig.tight_layout()
fig.savefig("eda_output/03_contracts_over_time.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  saved 03_contracts_over_time.png")

# ── 4. TOP PUBLISHERS BY VOLUME ───────────────────────────────────────────────
print("Plot 4: Top publishers …")
top_pub = (df.groupby("publisher_name")["value"]
             .agg(["count", "sum", "median"])
             .sort_values("count", ascending=False)
             .head(20))
top_pub.index = [n.title()[:35] for n in top_pub.index]

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
ax = axes[0]
ax.barh(top_pub.index[::-1], top_pub["count"][::-1], color=BLUE, edgecolor="none")
ax.set_xlabel("Number of Contracts")
ax.set_title("Top 20 Publishers by Contract Count", fontsize=12, fontweight="bold")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))

ax = axes[1]
ax.barh(top_pub.index[::-1], top_pub["sum"][::-1] / 1e9, color=GREEN, edgecolor="none")
ax.set_xlabel("Total Contract Value (AUD Billion)")
ax.set_title("Top 20 Publishers by Total Value", fontsize=12, fontweight="bold")
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("$%.1fB"))

fig.tight_layout()
fig.savefig("eda_output/04_top_publishers.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  saved 04_top_publishers.png")

# ── 5. VALUE BY JURISDICTION ──────────────────────────────────────────────────
print("Plot 5: Value by jurisdiction …")
if "publisher_gov_type" in df.columns:
    jur_map = {"fed":"Federal","nsw":"NSW","vic":"VIC","qld":"QLD","wa":"WA",
               "sa":"SA","act":"ACT","tas":"TAS","nt":"NT"}
    df["jurisdiction"] = df["publisher_gov_type"].map(jur_map).fillna("Other")
    jur = df.groupby("jurisdiction")["value"].agg(["count","median","sum"]).reset_index()
    jur = jur.sort_values("count", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.bar(jur["jurisdiction"], jur["count"], color=PALETTE[:len(jur)], edgecolor="white")
    ax.set_ylabel("Number of Contracts")
    ax.set_title("Contracts by Jurisdiction", fontsize=12, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))
    for bar, val in zip(ax.patches, jur["count"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
                f"{val/1000:.0f}K", ha="center", va="bottom", fontsize=8)

    ax = axes[1]
    ax.bar(jur["jurisdiction"], jur["median"] / 1000, color=PALETTE[:len(jur)], edgecolor="white")
    ax.set_ylabel("Median Contract Value (AUD K)")
    ax.set_title("Median Contract Value by Jurisdiction", fontsize=12, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0fK"))

    fig.tight_layout()
    fig.savefig("eda_output/05_value_by_jurisdiction.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved 05_value_by_jurisdiction.png")

# ── 6. PROCUREMENT METHOD BREAKDOWN ──────────────────────────────────────────
print("Plot 6: Procurement method …")
# Simplify procurement method names
def simplify_method(m):
    m = str(m).lower().strip()
    if "open" in m: return "Open"
    if "limited" in m or "closed" in m or "restricted" in m: return "Limited/Closed"
    if "direct" in m or "sole" in m or "single" in m or "non-competitive" in m: return "Direct/Sole Source"
    if "cua" in m or "arrangement" in m or "soa" in m: return "Standing Offer/CUA"
    if "select" in m or "invited" in m or "prequalif" in m: return "Select/Invited"
    if "quotation" in m or "rfq" in m or "quote" in m: return "Quotation/RFQ"
    if "tender" in m and "limited" not in m: return "Tender"
    if "unknown" in m or "not" in m or "other" in m: return "Other/Unknown"
    return "Other/Unknown"

df["method_simple"] = df["procurement_method"].apply(simplify_method)
meth = df.groupby("method_simple")["value"].agg(["count","median"]).reset_index()
meth = meth.sort_values("count", ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
ax.barh(meth["method_simple"][::-1], meth["count"][::-1], color=BLUE, edgecolor="none")
ax.set_xlabel("Number of Contracts")
ax.set_title("Contracts by Procurement Method (Simplified)", fontsize=12, fontweight="bold")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))

ax = axes[1]
ax.barh(meth["method_simple"][::-1], meth["median"][::-1] / 1000, color=ORANGE, edgecolor="none")
ax.set_xlabel("Median Contract Value (AUD K)")
ax.set_title("Median Value by Procurement Method", fontsize=12, fontweight="bold")
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0fK"))

fig.tight_layout()
fig.savefig("eda_output/06_procurement_method.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  saved 06_procurement_method.png")

# ── 7. CONTRACT DURATION DISTRIBUTION ────────────────────────────────────────
print("Plot 7: Duration distribution …")
dur = df["duration_days"].dropna()
dur = dur[(dur > 0) & (dur <= 3650)]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
ax.hist(dur, bins=100, color=GREEN, edgecolor="white", linewidth=0.3)
ax.axvline(dur.median(), color=RED, lw=2, linestyle="--", label=f"Median {dur.median():.0f} days")
ax.axvline(365, color=ORANGE, lw=1.5, linestyle=":", label="1 year (365d)")
ax.axvline(730, color=ORANGE, lw=1.5, linestyle="-.", label="2 years (730d)")
ax.set_xlabel("Duration (days)")
ax.set_ylabel("Count")
ax.set_title("Contract Duration Distribution", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)

ax = axes[1]
dur_bins = pd.cut(dur, bins=[0,90,180,365,730,1095,1825,3650],
    labels=["<3mo","3–6mo","6mo–1yr","1–2yr","2–3yr","3–5yr","5–10yr"])
dur_counts = dur_bins.value_counts().sort_index()
ax.bar(dur_counts.index, dur_counts.values, color=GREEN, edgecolor="white")
ax.set_xlabel("Duration Bucket")
ax.set_ylabel("Count")
ax.set_title("Contracts by Duration Bucket", fontsize=12, fontweight="bold")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))
for bar, val in zip(ax.patches, dur_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
            f"{val/1000:.0f}K", ha="center", va="bottom", fontsize=8)

fig.tight_layout()
fig.savefig("eda_output/07_duration_distribution.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  saved 07_duration_distribution.png")

# ── 8. CORRELATION MATRIX (numeric + ordinal-encoded categoricals) ────────────
print("Plot 8: Correlation matrix …")
from sklearn.preprocessing import OrdinalEncoder
corr_df = df[["value", "duration_days", "year",
              "procurement_method", "disposition", "publisher_gov_type",
              "publisher_cofog_level", "is_consultancy_services" if "is_consultancy_services" in df.columns else "disposition"]].copy()

# Encode categoricals ordinally for correlation
cat_cols = [c for c in corr_df.columns if corr_df[c].dtype == object]
enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
corr_df[cat_cols] = enc.fit_transform(corr_df[cat_cols].fillna("unknown").astype(str))
for c in cat_cols:
    corr_df[c] = pd.to_numeric(corr_df[c], errors="coerce")
corr_df["log_value"] = np.log1p(df["value"])
corr_df = corr_df.drop(columns=["value"])
rename = {
    "log_value": "Log Value",
    "duration_days": "Duration (days)",
    "year": "Year",
    "procurement_method": "Proc. Method",
    "disposition": "Disposition",
    "publisher_gov_type": "Jurisdiction",
    "publisher_cofog_level": "COFOG Level",
    "is_consultancy_services": "Consultancy",
}
corr_df = corr_df.rename(columns={c: rename.get(c, c) for c in corr_df.columns})
corr_matrix = corr_df.corr()

fig, ax = plt.subplots(figsize=(9, 7))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, ax=ax, annot=True, fmt=".2f", cmap="RdYlBu_r",
            center=0, vmin=-1, vmax=1, linewidths=0.5, linecolor="white",
            annot_kws={"size": 9}, square=True, mask=False)
ax.set_title("Pearson Correlation Matrix (Ordinal-Encoded Features)", fontsize=12, fontweight="bold", pad=12)
fig.tight_layout()
fig.savefig("eda_output/08_correlation_matrix.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  saved 08_correlation_matrix.png")

# ── 9. VALUE BY COFOG LEVEL ───────────────────────────────────────────────────
print("Plot 9: Value by COFOG level …")
if "publisher_cofog_level" in df.columns:
    cofog_map = {"1.0":"Level 1\n(General Govt)", "2.0":"Level 2\n(State Owned)", "unknown":"Unknown"}
    df["cofog_label"] = df["publisher_cofog_level"].map(cofog_map).fillna(df["publisher_cofog_level"])
    cofog = df.groupby("cofog_label")["value"].agg(["count","median","mean"]).reset_index()
    cofog = cofog.sort_values("count", ascending=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(cofog))
    width = 0.35
    bars1 = ax.bar([i - width/2 for i in x], cofog["median"] / 1000, width, label="Median", color=BLUE)
    bars2 = ax.bar([i + width/2 for i in x], cofog["mean"] / 1000, width, label="Mean", color=ORANGE)
    ax.set_xticks(list(x))
    ax.set_xticklabels(cofog["cofog_label"], fontsize=10)
    ax.set_ylabel("Contract Value (AUD K)")
    ax.set_title("Contract Value by COFOG Government Level", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0fK"))
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"${bar.get_height():.0f}K", ha="center", va="bottom", fontsize=8, color=BLUE)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"${bar.get_height():.0f}K", ha="center", va="bottom", fontsize=8, color=ORANGE)
    fig.tight_layout()
    fig.savefig("eda_output/09_value_by_cofog.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved 09_value_by_cofog.png")

# ── 10. TOP UNSPSC CATEGORIES ─────────────────────────────────────────────────
print("Plot 10: Top categories …")
if "category_code" in df.columns:
    top_cat = (df.groupby("category_code")["value"]
                 .agg(["count","median"])
                 .sort_values("count", ascending=False)
                 .head(15))
    top_cat.index = [str(c)[:20] for c in top_cat.index]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(top_cat.index[::-1], top_cat["count"][::-1], color=BLUE, edgecolor="none")
    ax2 = ax.twiny()
    ax2.plot(top_cat["median"][::-1] / 1000, top_cat.index[::-1],
             "o-", color=ORANGE, lw=2, ms=6, label="Median value (K)")
    ax.set_xlabel("Number of Contracts", color=BLUE)
    ax2.set_xlabel("Median Contract Value (AUD K)", color=ORANGE)
    ax.set_title("Top 15 UNSPSC Category Codes by Contract Count", fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", labelcolor=BLUE)
    ax2.tick_params(axis="x", labelcolor=ORANGE)
    ax2.xaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0fK"))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))
    fig.tight_layout()
    fig.savefig("eda_output/10_top_categories.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved 10_top_categories.png")

# ── 11. CONSULTANCY vs NON-CONSULTANCY ────────────────────────────────────────
print("Plot 11: Consultancy split …")
if "is_consultancy_services" in df.columns:
    df["consult"] = df["is_consultancy_services"].astype(str).str.lower()
    consult = df.groupby("consult")["value"].agg(["count","median"]).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(consult["consult"], consult["count"], color=[BLUE, GREEN, ORANGE], edgecolor="white")
    axes[0].set_title("Count: Consultancy vs Non", fontsize=11, fontweight="bold")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))
    axes[1].bar(consult["consult"], consult["median"] / 1000, color=[BLUE, GREEN, ORANGE], edgecolor="white")
    axes[1].set_title("Median Value: Consultancy vs Non", fontsize=11, fontweight="bold")
    axes[1].yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0fK"))
    fig.tight_layout()
    fig.savefig("eda_output/11_consultancy_split.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved 11_consultancy_split.png")

# ── 12. YEAR × JURISDICTION HEATMAP ──────────────────────────────────────────
print("Plot 12: Year × Jurisdiction heatmap …")
if "jurisdiction" in df.columns:
    pivot = df[(df["year"] >= 2015) & (df["year"] <= 2024)].pivot_table(
        index="jurisdiction", columns="year", values="value", aggfunc="count", fill_value=0
    )
    fig, ax = plt.subplots(figsize=(13, 5))
    sns.heatmap(pivot, ax=ax, cmap="Blues", annot=True, fmt=",d",
                annot_kws={"size": 8}, linewidths=0.4, linecolor="white",
                cbar_kws={"label": "# Contracts"})
    ax.set_title("Contract Volume: Jurisdiction × Year", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Year")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig("eda_output/12_jurisdiction_year_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved 12_jurisdiction_year_heatmap.png")

print("\n✓ All plots saved to eda_output/")
print("Files:", sorted(os.listdir("eda_output")))
