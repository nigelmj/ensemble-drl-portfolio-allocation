"""
RQ3 — Return vs Volatility scatter (9 points: v1..v6 + RQ2 floor + 2 BH baselines)

Source: results/rq3/backtest_performance_comparison.csv
  Row "Annual return" / "Annual volatility" for columns:
    v1..v6, RQ2 k10b20, Equal Weight (ew_buyhold), Mean Var (mvo_buyhold)

Output:
  results/rq3/return_volatility_scatter.png
  results/rq3/return_volatility_scatter.csv

Usage:
  python -m pipelines.rq3.rq3_risk_return_scatter
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd

from pipelines.rq3 import rq3_config

RESULTS_ROOT = rq3_config.RESULTS_ROOT

VERSION_ORDER = rq3_config.VERSION_ORDER
VERSION_ORDER_DISPLAY = [v.upper() for v in VERSION_ORDER]
VERSION_LABELS = rq3_config.VERSION_LABELS
RQ2_LABEL = rq3_config.RQ2_BLOCK_LABEL
RQ2_LABEL_DISPLAY = rq3_config.RQ2_BLOCK_LABEL_DISPLAY
EW_LABEL = "Equal Weight"
MVO_LABEL = "Mean Var"
MVO_LABEL_DISPLAY = "MVO"
def _disp_label(l: str) -> str:
    if l == RQ2_LABEL:
        return RQ2_LABEL_DISPLAY
    if l == MVO_LABEL:
        return MVO_LABEL_DISPLAY
    if l.startswith("v") and l[1:].isdigit():
        return l.upper()
    return l

VARIANT_COLOR = "#d62728"
V_COLORS = {v: VARIANT_COLOR for v in VERSION_ORDER}


def main():

    csv_path = os.path.join(RESULTS_ROOT, "backtest_performance_comparison.csv")
    assert os.path.exists(csv_path), f"Missing {csv_path}"
    df = pd.read_csv(csv_path, index_col=0)
    # df index = metric names, columns = series
    # Extract required 9 columns
    needed_cols = [VERSION_LABELS[v] for v in VERSION_ORDER] + [RQ2_LABEL, EW_LABEL, MVO_LABEL]
    for c in needed_cols:
        assert c in df.columns, f"Missing column {c} in {csv_path}. Have {list(df.columns)}"

    vols = df.loc["Annual volatility", needed_cols]
    rets_cagr = df.loc["Annual return", needed_cols]
    sharpes = df.loc["Sharpe ratio", needed_cols]
    # Arithmetic annualised return = Sharpe * Vol = mean(r)*252, Sharpe-consistent
    # (Annual return in pyfolio is geometric CAGR; Sharpe uses arithmetic mean)
    rets_arith = sharpes.astype(float) * vols.astype(float)

    # Build scatter frame with short labels — ret is arithmetic (Sharpe-consistent),
    # ret_cagr kept for transparency
    short_labels = VERSION_ORDER + [RQ2_LABEL, EW_LABEL, MVO_LABEL]
    scatter = pd.DataFrame({
        "label": short_labels,
        "vol": [float(vols[c]) for c in needed_cols],
        "ret": [float(rets_arith[c]) for c in needed_cols],
        "ret_cagr": [float(rets_cagr[c]) for c in needed_cols],
        "full_label": needed_cols,
    })
    print(scatter.to_string(index=False))
    # For backwards compat also expose ret as ret_arith
    scatter["ret_arith"] = scatter["ret"]

    # Save CSV for report
    csv_out_0 = os.path.join(RESULTS_ROOT, "return_volatility_scatter.csv")
    scatter.to_csv(csv_out_0, index=False)
    print(f"Saved {csv_out_0}")

    fig, ax = plt.subplots(figsize=(9, 7))

    for _, row in scatter[scatter["label"].isin(VERSION_ORDER)].iterrows():
        lbl = row["label"]
        disp = _disp_label(lbl)
        is_v4 = lbl == "v4"
        marker = "D" if is_v4 else "o"
        ax.scatter(row["vol"], row["ret"], s=120, marker=marker, color=VARIANT_COLOR,
                   edgecolor="black", linewidth=0.7, zorder=4)
        # annotate — all boxed with same alpha
        ax.annotate(disp, (row["vol"], row["ret"]), xytext=(6, 6), textcoords="offset points",
                    fontsize=11, ha="left", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7))

    # RQ2 block ensemble — the comparison floor for RQ3
    rq2_row = scatter[scatter["label"] == RQ2_LABEL].iloc[0]
    ax.scatter(rq2_row["vol"], rq2_row["ret"], s=200, marker="X", color="black",
               edgecolor="white", linewidth=1.0, zorder=6)
    ax.annotate(RQ2_LABEL_DISPLAY, (rq2_row["vol"], rq2_row["ret"]), xytext=(8, -14),
                textcoords="offset points", fontsize=10, ha="left", va="top",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7))

    # Baselines — both triangle ^, keep grey/dark grey
    ew_row = scatter[scatter["label"] == EW_LABEL].iloc[0]
    mvo_row = scatter[scatter["label"] == MVO_LABEL].iloc[0]
    ax.scatter(ew_row["vol"], ew_row["ret"], s=160, marker="^", color="#888888",
               edgecolor="black", linewidth=0.7, zorder=4)
    ax.scatter(mvo_row["vol"], mvo_row["ret"], s=160, marker="^", color="#444444",
               edgecolor="white", linewidth=0.7, zorder=4)
    ax.annotate("EW", (ew_row["vol"], ew_row["ret"]), xytext=(6, 6), textcoords="offset points", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7))
    ax.annotate("MVO", (mvo_row["vol"], mvo_row["ret"]), xytext=(6, 6), textcoords="offset points", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7))

    ax.set_xlabel("Annual volatility", fontsize=14)
    ax.set_ylabel("Annual return", fontsize=14)
    ax.set_title("Risk-Return Trade-off: RQ3 Versions (V1–V6) vs RQ2 Block Ensemble & Baselines", fontsize=15)
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(True, alpha=0.3)

    # Extra padding so the MVO annotation (far right, bottom) is not clipped
    xmin, xmax = scatter["vol"].min(), scatter["vol"].max()
    ymin, ymax = scatter["ret"].min(), scatter["ret"].max()
    xpad = (xmax - xmin) * 0.15
    ypad = (ymax - ymin) * 0.15
    ax.set_xlim(xmin - xpad, xmax + xpad)
    ax.set_ylim(ymin - ypad, ymax + ypad)

    # Custom legend: single entry for all variants
    variant_handle = mlines.Line2D([], [], color=VARIANT_COLOR, marker="o", linestyle="None",
                                   markersize=8, markeredgecolor="black", markeredgewidth=0.7, label="RQ3 variants (V1–V6)")
    rq2_handle = mlines.Line2D([], [], color="black", marker="X", linestyle="None",
                               markersize=9, markeredgecolor="black", markeredgewidth=1.0, label=RQ2_LABEL_DISPLAY)
    ew_handle = mlines.Line2D([], [], color="#888888", marker="^", linestyle="None",
                              markersize=9, markeredgecolor="black", markeredgewidth=0.7, label=EW_LABEL + " (BH)")
    mvo_handle = mlines.Line2D([], [], color="#444444", marker="^", linestyle="None",
                               markersize=9, markeredgecolor="white", markeredgewidth=0.7, label=MVO_LABEL_DISPLAY + " (BH)")
    ax.legend(handles=[variant_handle, rq2_handle, ew_handle, mvo_handle],
              fontsize=10, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0., framealpha=0.95)

    fig.tight_layout()
    out_0 = os.path.join(RESULTS_ROOT, "return_volatility_scatter.png")
    fig.savefig(out_0, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_0} ({os.path.getsize(out_0)} bytes)")
    print("Done.")


if __name__ == "__main__":
    main()
