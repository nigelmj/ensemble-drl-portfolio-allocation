"""
RQ1 — Performance Comparison of Single and Ensemble PPO Agents

Clean 4-curve plot: Single PPO vs PPO ensemble (mean 5 PPO) vs
Equal Weight Buy-and-Hold vs MVO Buy-and-Hold.

Data (default 1M single):
  results/0rq1/account_value_single_ppo.csv
  results/0rq1/baseline_ensemble_average/test_account.csv
  results/baselines/ew_buyhold.csv
  results/baselines/mvo_buyhold.csv

Data (5M compute-matched, with --use-5m):
  results/0rq1/account_value_single_ppo_5m.csv
  -> outputs are suffixed _5m so the 1M artefacts are never overwritten.

Output:
  results/0rq1/performance_comparison_single_ensemble_ppo.png[.csv]
  results/report/rq1_performance_comparison_single_ensemble_ppo.png[.csv]
  (+ _5m variants when --use-5m)

Usage:
  python -m pipelines.rq1.rq1_simple_baseline_plot
  python -m pipelines.rq1.rq1_simple_baseline_plot --use-5m
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

import argparse

from pipelines.rq1 import rq1_config
from rl_portfolio.utils.metrics import backtest_stats

RESULTS_ROOT = rq1_config.RESULTS_ROOT
REPORT_ROOT = "results/report"
BASELINES_DIR = "results/baselines"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--use-5m", action="store_true",
                   help="Use the compute-matched 5M single (account_value_single_ppo_5m.csv) "
                        "and write outputs with _5m suffix so 1M artefacts are preserved.")
    return p.parse_args()

INITIAL_AMOUNT = rq1_config.ENV_KWARGS["initial_amount"]


def load_account(path: str, col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # handle both "portfolio_value" and "value" columns
    if "portfolio_value" in df.columns:
        df = df.rename(columns={"portfolio_value": col})
    elif "value" in df.columns:
        df = df.rename(columns={"value": col})
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df


def main():
    args = parse_args()
    os.makedirs(REPORT_ROOT, exist_ok=True)

    single_file = "account_value_single_ppo_5m.csv" if args.use_5m else "account_value_single_ppo.csv"
    single_label = "Single PPO"
    tag_suffix = "_5m" if args.use_5m else ""
    single_path = os.path.join(RESULTS_ROOT, single_file)
    baseline_path = os.path.join(RESULTS_ROOT, "baseline_ensemble_average", "test_account.csv")
    ew_path = os.path.join(BASELINES_DIR, "ew_buyhold.csv")
    mvo_path = os.path.join(BASELINES_DIR, "mvo_buyhold.csv")

    for p in [single_path, baseline_path, ew_path, mvo_path]:
        assert os.path.exists(p), f"Missing {p}"

    single = load_account(single_path, single_label)
    basic = load_account(baseline_path, "PPO ensemble")
    ew = load_account(ew_path, "Equal Weight")
    mvo = load_account(mvo_path, "MVO")

    dates = single["date"].tolist()
    initial_amount = float(single[single_label].iloc[0])

    # Build aligned frame
    result = pd.DataFrame({"date": dates})
    result[single_label] = single.set_index("date")[single_label].reindex(dates).values
    result["PPO ensemble"] = basic.set_index("date")["PPO ensemble"].reindex(dates).values

    # Baselines are anchored to initial_amount at first date (shared.py logic)
    for name, df in [("Equal Weight", ew), ("MVO", mvo)]:
        s = df.set_index("date")[name].reindex(dates)
        # ew/mvo CSVs are already scaled to 1e6 at their own TEST_START_DATE (2022-01-01)
        # Re-anchor to RQ1 test start (first date in single = 2022-01-03)
        s = s / s.iloc[0] * initial_amount
        result[name] = s.values

    result = result.dropna().reset_index(drop=True)
    print(f"Aligned rows: {len(result)}  {result['date'].iloc[0]} -> {result['date'].iloc[-1]}")

    # Performance table
    cols = [c for c in result.columns if c != "date"]
    series = []
    for col in cols:
        df = result[["date", col]].rename(columns={col: "account_value"})
        series.append(backtest_stats(df, value_col_name="account_value").rename(col))
    table = pd.concat(series, axis=1).round(4)
    print("\n=== Simple Baseline Performance Comparison ===")
    print(table)

    table_path_0 = os.path.join(RESULTS_ROOT, f"performance_comparison_single_ensemble_ppo{tag_suffix}.csv")
    table_path_r = os.path.join(REPORT_ROOT, f"rq1_performance_comparison_single_ensemble_ppo{tag_suffix}.csv")
    table.to_csv(table_path_0)
    table.to_csv(table_path_r)
    print(f"Saved {table_path_0}")
    print(f"Saved {table_path_r}")

    # Plot
    plt.rcParams["figure.figsize"] = (15, 6)
    fig, ax = plt.subplots(figsize=(15, 6))

    colors = {
        single_label: "#2a5c8a",
        "PPO ensemble": "black",
        "Equal Weight": "#ff7f0e",
        "MVO": "#9467bd",
    }
    # Line widths
    lws = {
        single_label: 1.8,
        "PPO ensemble": 1.8,
        "Equal Weight": 1.6,
        "MVO": 1.6,
    }

    linestyles = {
        single_label: "-",
        "PPO ensemble": "-",
        "Equal Weight": "--",
        "MVO": "--",
    }
    for col in cols:
        ax.plot(pd.to_datetime(result["date"]), result[col], label=col, color=colors.get(col, "#333"), lw=lws.get(col, 1.6), linestyle=linestyles.get(col, "-"), alpha=0.9)

    ax.set_ylim(600_000, 2_000_000)
    ax.set_title("Performance Comparison of Single and Ensemble PPO Agents", fontsize=16)
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
    out_0 = os.path.join(RESULTS_ROOT, f"performance_comparison_single_ensemble_ppo{tag_suffix}.png")
    out_r = os.path.join(REPORT_ROOT, f"rq1_performance_comparison_single_ensemble_ppo{tag_suffix}.png")
    fig.savefig(out_0, dpi=150, bbox_inches="tight")
    fig.savefig(out_r, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_0} ({os.path.getsize(out_0)} bytes)")
    print(f"Saved {out_r} ({os.path.getsize(out_r)} bytes)")
    print("Done.")


if __name__ == "__main__":
    main()
