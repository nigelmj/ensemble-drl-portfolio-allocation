"""
RQ1 — Performance of PPO ensemble and Confidence-Weighted PPO Ensembles

5-curve plot: PPO ensemble (mean 5 PPO) vs 4 confidence mappings
(Linear / Power / Exponential / Sigmoid), all with previous safe
(allocation = previous weights). Uses the previous-allocation variants
as requested.

Data:
  results/0rq1/baseline_ensemble_average/test_account.csv
  results/0rq1/combo_linear_previous/test_account.csv
  results/0rq1/combo_power_previous/test_account.csv
  results/0rq1/combo_exponential_previous/test_account.csv
  results/0rq1/combo_sigmoid_previous/test_account.csv

Output:
  results/0rq1/performance_standard_vs_confidence_weighted.png
  results/report/rq1_performance_standard_vs_confidence_weighted.png
  + corresponding CSVs

Usage:
  python -m pipelines.rq1.rq1_ensemble_confidence_plot
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from pipelines.rq1 import rq1_config
from rl_portfolio.utils.metrics import backtest_stats

RESULTS_ROOT = rq1_config.RESULTS_ROOT
REPORT_ROOT = "results/report"


def load_account(path: str, col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"portfolio_value": col})
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df


def main():
    os.makedirs(REPORT_ROOT, exist_ok=True)

    basic_path = os.path.join(RESULTS_ROOT, "baseline_ensemble_average", "test_account.csv")
    paths = {
        "PPO ensemble": basic_path,
        "Linear": os.path.join(RESULTS_ROOT, "combo_linear_previous", "test_account.csv"),
        "Power": os.path.join(RESULTS_ROOT, "combo_power_previous", "test_account.csv"),
        "Exponential": os.path.join(RESULTS_ROOT, "combo_exponential_previous", "test_account.csv"),
        "Sigmoid": os.path.join(RESULTS_ROOT, "combo_sigmoid_previous", "test_account.csv"),
    }
    for label, p in paths.items():
        assert os.path.exists(p), f"Missing {label}: {p}"

    # Load basic as reference for dates
    basic = load_account(basic_path, "PPO ensemble")
    dates = basic["date"].tolist()

    result = pd.DataFrame({"date": dates})
    for label, p in paths.items():
        df = load_account(p, label)
        result[label] = df.set_index("date")[label].reindex(dates).values

    result = result.dropna().reset_index(drop=True)
    print(f"Aligned rows: {len(result)}  {result['date'].iloc[0]} -> {result['date'].iloc[-1]}")

    # Performance table
    cols = [c for c in result.columns if c != "date"]
    series = []
    for col in cols:
        df = result[["date", col]].rename(columns={col: "account_value"})
        series.append(backtest_stats(df, value_col_name="account_value").rename(col))
    table = pd.concat(series, axis=1).round(4)
    print("\n=== Standard vs Confidence-Weighted Performance ===")
    print(table)

    table_path_0 = os.path.join(RESULTS_ROOT, "performance_standard_vs_confidence_weighted.csv")
    table_path_r = os.path.join(REPORT_ROOT, "rq1_performance_standard_vs_confidence_weighted.csv")
    table.to_csv(table_path_0)
    table.to_csv(table_path_r)
    print(f"Saved {table_path_0}")
    print(f"Saved {table_path_r}")

    # Plot — same styling as simple baseline plot
    fig, ax = plt.subplots(figsize=(15, 6))

    colors = {
        "PPO ensemble": "black",
        "Linear": "#2a5c8a",
        "Power": "#1f77b4",
        "Exponential": "#ff7f0e",
        "Sigmoid": "#2ca02c",
    }
    lws = {
        "PPO ensemble": 1.9,
        "Linear": 1.6,
        "Power": 1.6,
        "Exponential": 1.6,
        "Sigmoid": 1.6,
    }

    for col in cols:
        ax.plot(pd.to_datetime(result["date"]), result[col], label=col, color=colors.get(col, "#333"), lw=lws.get(col, 1.6))

    ax.set_ylim(600_000, 2_000_000)
    ax.set_title("Performance of PPO ensemble and Confidence-Weighted PPO Ensembles", fontsize=16)
    ax.set_xlabel("Date", fontsize=14)
    ax.set_ylabel("Portfolio Value ($)", fontsize=14)
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, ncol=2, loc="upper left")

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    for lab in ax.get_xticklabels():
        lab.set_rotation(45)
        lab.set_ha("right")

    fig.tight_layout()
    out_0 = os.path.join(RESULTS_ROOT, "performance_standard_vs_confidence_weighted.png")
    out_r = os.path.join(REPORT_ROOT, "rq1_performance_standard_vs_confidence_weighted.png")
    fig.savefig(out_0, dpi=150, bbox_inches="tight")
    fig.savefig(out_r, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_0} ({os.path.getsize(out_0)} bytes)")
    print(f"Saved {out_r} ({os.path.getsize(out_r)} bytes)")
    print("Done.")


if __name__ == "__main__":
    main()
