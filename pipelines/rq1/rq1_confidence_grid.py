"""
RQ1 — 2x2 confidence/disagreement grids for the 4 confidence mappings.

Generates:
  - results/0rq1/confidence_disagreement_timeseries_2x2.png  (twin-axis, D + c per mapping)
  - results/0rq1/confidence_timeseries_2x2.png               (confidence only per mapping)

Copies are also written to results/report/ with rq1_ prefix.

Data source: results/0rq1/combo_{mapping}_equal_weight/test_actions.csv
Confidence depends only on D and d_ref (p90=1.0345) — safe_strategy
has <1e-4 drift due to trajectory feedback, hence equal_weight is used
as canonical representative.

Usage: python -m pipelines.rq1.rq1_confidence_grid
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from pipelines.rq1 import rq1_config

RESULTS_ROOT = rq1_config.RESULTS_ROOT  # results/0rq1
REPORT_ROOT = "results/report"
MAPPINGS = rq1_config.CONFIDENCE_MAPPINGS  # linear, power, exponential, sigmoid
ORDER = ["linear", "power", "exponential", "sigmoid"]  # fixed 2x2 order
COLOR_D = "#2a5c8a"
COLOR_C = "darkgreen"

with open(os.path.join(RESULTS_ROOT, "calibration_all.json")) as f:
    cal = json.load(f)
D_REF = cal["calibrations"][rq1_config.D_REF_METHOD]["d_ref"]
D_REF_METHOD = rq1_config.D_REF_METHOD


def _load(mapping: str, safe: str = "equal_weight") -> pd.DataFrame:
    path = os.path.join(RESULTS_ROOT, f"combo_{mapping}_{safe}", "test_actions.csv")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def plot_twin():
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharex=True, sharey=False)
    axes = axes.flatten()
    for ax, mapping in zip(axes, ORDER):
        df = _load(mapping, "equal_weight")
        # disagreement on the left axis
        ax.plot(df["date"], df["disagreement"], color=COLOR_D, lw=1.0, label="disagreement")
        ax.set_ylabel("Disagreement (TVD)", color=COLOR_D, fontsize=14)
        ax.tick_params(axis="y", labelcolor=COLOR_D, labelsize=12)
        ax.tick_params(axis="x", labelsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(df["disagreement"].min() - 0.005, df["disagreement"].max() + 0.005)
        # confidence right axis
        ax2 = ax.twinx()
        ax2.plot(df["date"], df["confidence"], color=COLOR_C, lw=1.0, label="confidence")
        ax2.set_ylabel("Confidence", color=COLOR_C, fontsize=14)
        ax2.tick_params(axis="y", labelcolor=COLOR_C, labelsize=12)
        ax2.set_ylim(-0.05, 1.05)
        mean_d = df["disagreement"].mean()
        mean_c = df["confidence"].mean()
        ax.set_title(f"{mapping.capitalize()}: D̄={mean_d:.3f}  c̄={mean_c:.3f}", fontsize=15, pad=10)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.tick_params(axis="x", rotation=30, labelsize=11)

    # shared x label
    for ax in axes[2:]:
        ax.set_xlabel("Date", fontsize=15)
    fig.suptitle(
        "Disagreement and Confidence under Alternative Mappings",
        fontsize=17,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    out_local = os.path.join(RESULTS_ROOT, "confidence_disagreement_timeseries_2x2.png")
    out_report = os.path.join(REPORT_ROOT, "rq1_confidence_disagreement_timeseries_2x2.png")
    fig.savefig(out_local, dpi=150, bbox_inches="tight")
    fig.savefig(out_report, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_local}")
    print(f"Saved {out_report}")


def plot_confidence_only():
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharex=True, sharey=False)
    axes = axes.flatten()
    for ax, mapping in zip(axes, ORDER):
        df = _load(mapping, "equal_weight")
        ax.plot(df["date"], df["confidence"], color=COLOR_C, lw=1.1)
        # per-panel autoscale to make small variations visible
        cmin, cmax = df["confidence"].min(), df["confidence"].max()
        pad = max((cmax - cmin) * 0.15, 0.002)
        ax.set_ylim(max(0, cmin - pad), min(1.0, cmax + pad))
        ax.grid(True, alpha=0.3)
        mean_c = df["confidence"].mean()
        std_c = df["confidence"].std()
        ax.set_title(f"{mapping.capitalize()}: c̄={mean_c:.4f}  σ={std_c:.4f}  [{cmin:.3f},{cmax:.3f}]", fontsize=14, pad=8)
        ax.set_ylabel("Confidence", fontsize=14)
        ax.tick_params(axis="y", labelsize=11)
        ax.tick_params(axis="x", labelsize=11)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.tick_params(axis="x", rotation=30, labelsize=11)

    for ax in axes[2:]:
        ax.set_xlabel("Date", fontsize=15)
    fig.suptitle(
        "RQ1: Confidence Time Series by Mapping",
        fontsize=17,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    out_local = os.path.join(RESULTS_ROOT, "confidence_timeseries_2x2.png")
    out_report = os.path.join(REPORT_ROOT, "rq1_confidence_timeseries_2x2.png")
    fig.savefig(out_local, dpi=150, bbox_inches="tight")
    fig.savefig(out_report, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_local}")
    print(f"Saved {out_report}")


def main():
    os.makedirs(REPORT_ROOT, exist_ok=True)
    # all four mappings must have results before plotting
    for m in ORDER:
        p = os.path.join(RESULTS_ROOT, f"combo_{m}_equal_weight", "test_actions.csv")
        assert os.path.exists(p), f"missing {p}"
    plot_twin()
    plot_confidence_only()
    print("Done. Disagreement-only reuse: results/report/rq1_disagreement_histogram_p90.png")


if __name__ == "__main__":
    main()
