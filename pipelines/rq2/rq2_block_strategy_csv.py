"""
RQ2 — create CSV showing which block strategy (agent) is used within each block.

Reads sweep outputs (ensemble_sweep_summary.csv + ensemble_agent_weights_*.csv)
and produces a single CSV with per-day strategy selection for the top N block strategies.

Usage:
    python rq2_block_strategy_csv.py --top 3
    python rq2_block_strategy_csv.py --top 4
"""

from __future__ import annotations

import argparse
import os
import re

import pandas as pd


RESULTS_DIR = "results/0rq2"
SWEEP_DIR = os.path.join(RESULTS_DIR, "sweeps")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=3, help="Number of top block strategies to include (by Sharpe).")
    parser.add_argument("--k", nargs="+", type=int, default=[10, 20, 40],
                        help="Trailing metric windows to sweep.")
    parser.add_argument("--block", nargs="+", type=int, default=[5, 10, 20, 40],
                        help="Block tenures to sweep.")
    return parser.parse_args()


def main():
    args = parse_args()
    top_n = args.top
    k_vals = args.k
    block_vals = args.block

    all_rows = []

    summary_path = os.path.join(SWEEP_DIR, "ensemble_sweep_summary.csv")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Summary not found: {summary_path}")
    summary = pd.read_csv(summary_path)

    # Filter only block-mode rows (mode is implicit — all sweep runs use mode="block")
    # summary has columns: k, block_days, tag, final_value, total_return, sharpe, max_drawdown, annual_vol, calmar
    summary["sharpe"] = pd.to_numeric(summary["sharpe"], errors="coerce")
    summary = summary.dropna(subset=["sharpe"])

    # Select top N by sharpe (descending)
    top_summary = summary.sort_values("sharpe", ascending=False).head(top_n)
    top_block_days = top_summary["block_days"].tolist()
    print(f"Top {top_n} block strategies by Sharpe: {top_block_days}")

    # For each top block strategy, read its weights and find the daily winner.
    for bd in top_block_days:
        # Sweep tags are k{k}b{block}; match on the block_days suffix to recover k.
        matching_tags = [t for t in summary["tag"].tolist() if re.search(rf"k\d+b{bd}$", t)]
        if not matching_tags:
            print(f"  No tag found for block_days={bd}, skipping.")
            continue

        # Use the first matching tag (there should be exactly one per k×block combo)
        tag = matching_tags[0]
        # Extract k from tag
        m = re.search(r"k(\d+)", tag)
        if not m:
            print(f"  Cannot parse k from tag {tag}, skipping.")
            continue
        k = int(m.group(1))

        weights_path = os.path.join(SWEEP_DIR, f"ensemble_agent_weights_{tag}.csv")
        if not os.path.exists(weights_path):
            print(f"  Weights file not found: {weights_path}, skipping.")
            continue

        weights_df = pd.read_csv(weights_path)
        # weights_df columns: date, w_log_return, w_dsr, w_dsr_drawdown

        # Determine the winning agent per day (agent with highest weight)
        # In block mode, weights should be ~one-hot, but use argmax for robustness
        weights_df["winning_agent"] = weights_df[
            ["w_log_return", "w_dsr", "w_dsr_drawdown"]
        ].idxmax(axis=1)

        # Map column name to agent label
        agent_map = {"w_log_return": "log_return", "w_dsr": "dsr", "w_dsr_drawdown": "dsr_drawdown"}
        weights_df["agent_label"] = weights_df["winning_agent"].map(agent_map)

        # Add block_days and performance metadata
        weights_df["block_days"] = bd
        top_row = top_summary[top_summary["block_days"] == bd].iloc[0]
        weights_df["final_value"] = top_row["final_value"]
        weights_df["total_return"] = top_row["total_return"]
        weights_df["max_drawdown"] = top_row["max_drawdown"]

        weights_df["k"] = k

        # Rows carry block_days rather than an explicit block index; group by it downstream.

        all_rows.append(weights_df)

    if not all_rows:
        print("No data generated — check that sweep results exist.")
        return

    # Combine all top-strategy data
    result = pd.concat(all_rows, ignore_index=True)

    # Final column order
    out_cols = [
        "block_days",
        "date",
        "agent_label",
        "w_log_return",
        "w_dsr",
        "w_dsr_drawdown",
        "portfolio_value",
        "total_return",
        "sharpe",
        "max_drawdown",
        "k",
    ]
    # Ensure sharpe column exists; if not, compute from summary
    if "sharpe" not in result.columns:
        # merge sharpe from top_summary
        sh_map = top_summary.set_index("block_days")["sharpe"].to_dict()
        result["sharpe"] = result["block_days"].map(sh_map)

    # Keep only the columns actually present (portfolio_value comes from env_real).
    existing_cols = [c for c in out_cols if c in result.columns]
    result = result[existing_cols]

    # Output CSV path
    out_path = os.path.join(SWEEP_DIR, f"block_strategy_top{top_n}.csv")
    result.to_csv(out_path, index=False)
    print(f"\nCSV saved to {out_path}")
    print(result[["block_days", "date", "agent_label"]].head(10).to_string(index=False))
    print(f"Total rows: {len(result)}  block_days values: {sorted(result['block_days'].unique())}")


if __name__ == "__main__":
    main()