"""
eda_part2.py — Plots 8–12 (correlation matrix, COFOG, categories, consultancy, heatmap).
Loads a 100K-row sample to stay within memory limits.
"""
import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder

warnings.filterwarnings("ignore")
os.makedirs("eda_output", exist_ok=True)

STYLE = {
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "sans-serif",
}
plt.rcParams.update(STYLE)
BLUE="#2563EB"; ORANGE="#F59E0B"; GREEN="#10B981"; RED="#EF4444"
PALETTE=[BLUE,ORANGE,GREEN,RED,"#8B5CF6","#EC4899","#14B8A6","#F97316","#6366F1","#84CC16"]

print("Loading data (sample=150K)…")
df_full = pd.read_excel("tenders_export.xlsx")
df_full["value"] = pd.to_numeric(df_full.get("value"), errors="coerce")
df_full = df_full[df_full["value"] > 0].copy()
df_full["contract_start"] = pd.to_datetime(df_full.get("contract_start"), errors="coerce")
df_full["contract_end"]   = pd.to_datetime(df_full.get("contract_end"),   errors="coerce")
df_full["year"]           = df_full["contract_start"].dt.year
df_full["duration_days"]  = (df_full["contract_end"] - df_full["contract_start"]).dt.days.clip(1, 3650)
for col in ["procurement_method","disposition","publisher_gov_type",
            "publisher_cofog_level","publisher_name","category_code","is_consultancy_services"]:
    if col in df_full.columns:
        df_full[col] = df_full[col].fillna("unknown").astype(str).str.strip().str.lower()

jur_map = {"fed":"Federal","nsw":"NSW","vic":"VIC","qld":"QLD","wa":"WA",
           "sa":"SA","act":"ACT","tas":"TAS","nt":"NT"}
df_full["jurisdiction"] = df_full["publisher_gov_type"].map(jur_map).fillna("Other")

df = df_full.sample(n=min(150_000, len(df_full)), random_state=42)
print(f"Working sample: {len(df):,} rows")

# ── 8. CORRELATION MATRIX ─────────────────────────────────────────────────────
print("Plot 8: Correlation matrix…")
num = pd.DataFrame()
num["Log Value"]  = np.log1p(df["value"].values.astype(float))
num["Duration"]   = pd.to_numeric(df["duration_days"], errors="coerce").values
num["Year"]       = pd.to_numeric(df["year"], errors="coerce").values

for col, label in [
    ("procurement_method", "Proc Method"),
    ("disposition",        "Disposition"),
    ("publisher_gov_type", "Jurisdiction"),
    ("publisher_cofog_level", "COFOG Level"),
]:
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    num[label] = enc.fit_transform(
        df[[col]].fillna("unknown").astype(str)
    )[:, 0].astype(float)

corr = num.corr()

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(
    corr, ax=ax, annot=True, fmt=".2f", cmap="RdYlBu_r",
    center=0, vmin=-1, vmax=1, linewidths=0.5, linecolor="white",
    annot_kws={"size": 10}, square=True,
    cbar_kws={"shrink": 0.8},
)
ax.set_title("Pearson Correlation Matrix\n(150K-row sample, categoricals ordinal-encoded)",
             fontsize=12, fontweight="bold", pad=12)
fig.tight_layout()
fig.savefig("eda_output/08_correlation_matrix.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  saved 08_correlation_matrix.png")
del num, corr

# ── 9. VALUE BY COFOG LEVEL ───────────────────────────────────────────────────
print("Plot 9: Value by COFOG level…")
cofog_map = {"1.0": "Level 1\n(General Govt)", "2.0": "Level 2\n(State-Owned)"}
df_full["cofog_label"] = df_full["publisher_cofog_level"].map(cofog_map).fillna(
    df_full["publisher_cofog_level"].str.title()
)
cofog = df_full.groupby("cofog_label")["value"].agg(["count","median","mean"]).reset_index()
cofog = cofog.sort_values("count", ascending=False)

fig, ax = plt.subplots(figsize=(10, 5))
x = range(len(cofog))
w = 0.35
b1 = ax.bar([i - w/2 for i in x], cofog["median"]/1000, w, label="Median", color=BLUE)
b2 = ax.bar([i + w/2 for i in x], cofog["mean"]/1000,   w, label="Mean",   color=ORANGE)
ax.set_xticks(list(x)); ax.set_xticklabels(cofog["cofog_label"], fontsize=9)
ax.set_ylabel("Contract Value (AUD K)")
ax.set_title("Contract Value by COFOG Government Level", fontsize=12, fontweight="bold")
ax.legend()
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0fK"))
for b in b1:
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+1,
            f"${b.get_height():.0f}K", ha="center", va="bottom", fontsize=8, color=BLUE)
for b in b2:
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+1,
            f"${b.get_height():.0f}K", ha="center", va="bottom", fontsize=8, color=ORANGE)
fig.tight_layout()
fig.savefig("eda_output/09_value_by_cofog.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  saved 09_value_by_cofog.png")

# ── 10. TOP UNSPSC CATEGORIES ─────────────────────────────────────────────────
print("Plot 10: Top categories…")
top_cat = (df_full.groupby("category_code")["value"]
             .agg(["count","median"])
             .sort_values("count", ascending=False)
             .head(15))
top_cat.index = [str(c)[:22] for c in top_cat.index]

fig, ax = plt.subplots(figsize=(12, 6))
ax.barh(top_cat.index[::-1], top_cat["count"][::-1], color=BLUE, edgecolor="none")
ax2 = ax.twiny()
ax2.plot(top_cat["median"][::-1]/1000, top_cat.index[::-1],
         "o-", color=ORANGE, lw=2, ms=6, label="Median value")
ax.set_xlabel("Number of Contracts", color=BLUE)
ax2.set_xlabel("Median Contract Value (AUD K)", color=ORANGE)
ax.set_title("Top 15 UNSPSC Category Codes by Contract Count", fontsize=12, fontweight="bold")
ax.tick_params(axis="x", labelcolor=BLUE)
ax2.tick_params(axis="x", labelcolor=ORANGE)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x/1000:.0f}K"))
ax2.xaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0fK"))
fig.tight_layout()
fig.savefig("eda_output/10_top_categories.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  saved 10_top_categories.png")

# ── 11. CONSULTANCY SPLIT ─────────────────────────────────────────────────────
print("Plot 11: Consultancy split…")
if "is_consultancy_services" in df_full.columns:
    consult = df_full.groupby("is_consultancy_services")["value"].agg(["count","median","sum"]).reset_index()
    consult = consult.sort_values("count", ascending=False)
    label_map = {"true":"Consultancy","false":"Non-Consultancy","1":"Consultancy","0":"Non-Consultancy","yes":"Consultancy","no":"Non-Consultancy"}
    consult["label"] = consult["is_consultancy_services"].map(label_map).fillna(consult["is_consultancy_services"].str.title())

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    colors = PALETTE[:len(consult)]

    axes[0].bar(consult["label"], consult["count"], color=colors, edgecolor="white")
    axes[0].set_title("Contract Count", fontsize=11, fontweight="bold")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x/1000:.0f}K"))
    axes[0].tick_params(axis="x", rotation=15)

    axes[1].bar(consult["label"], consult["median"]/1000, color=colors, edgecolor="white")
    axes[1].set_title("Median Contract Value (AUD K)", fontsize=11, fontweight="bold")
    axes[1].yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0fK"))
    axes[1].tick_params(axis="x", rotation=15)

    axes[2].bar(consult["label"], consult["sum"]/1e9, color=colors, edgecolor="white")
    axes[2].set_title("Total Value (AUD B)", fontsize=11, fontweight="bold")
    axes[2].yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.1fB"))
    axes[2].tick_params(axis="x", rotation=15)

    fig.suptitle("Consultancy vs Non-Consultancy Services", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig("eda_output/11_consultancy_split.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved 11_consultancy_split.png")

# ── 12. YEAR × JURISDICTION HEATMAP ──────────────────────────────────────────
print("Plot 12: Year × Jurisdiction heatmap…")
pivot = df_full[
    (df_full["year"] >= 2015) & (df_full["year"] <= 2024)
].pivot_table(
    index="jurisdiction", columns="year",
    values="value", aggfunc="count", fill_value=0
)
pivot.columns = pivot.columns.astype(int)

fig, ax = plt.subplots(figsize=(14, 5))
sns.heatmap(
    pivot, ax=ax, cmap="Blues", annot=True, fmt=",d",
    annot_kws={"size": 8}, linewidths=0.4, linecolor="white",
    cbar_kws={"label": "# Contracts"},
)
ax.set_title("Contract Volume: Jurisdiction × Year", fontsize=12, fontweight="bold", pad=10)
ax.set_xlabel("Year"); ax.set_ylabel("")
fig.tight_layout()
fig.savefig("eda_output/12_jurisdiction_year_heatmap.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  saved 12_jurisdiction_year_heatmap.png")

print("\nAll plots saved to eda_output/")
print("Files:", sorted(os.listdir("eda_output")))
