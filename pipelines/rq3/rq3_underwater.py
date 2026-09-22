"""
RQ3 — Underwater (drawdown-over-time) plot, sweep-best versions + RQ2 floor only.

Source:
  per-version Sharpe-best k×block from sweeps:
    results/rq3/sweeps/sweep_summary_{v}.csv  -> tag
    results/rq3/sweeps/account_value_{v}_{tag}.csv
  RQ2 floor:
    results/rq2/sweeps/account_value_ensemble_pa_k10b20.csv

  No baselines.

Output:
  results/rq3/rq3_underwater.png
  results/rq3/rq3_underwater.csv

Usage:
  python -m pipelines.rq3.rq3_underwater

Caption for LaTeX:
  Drawdown from running peak over the test window: the six versions
  against the RQ2 floor.
"""
from __future__ import annotations

import os
import glob
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

from pipelines.rq3 import rq3_config

RESULTS_ROOT = rq3_config.RESULTS_ROOT

RQ2_LABEL = "RQ2 k10b20"
RQ2_LABEL_DISPLAY = rq3_config.RQ2_BLOCK_LABEL_DISPLAY
RQ2_PATH = "results/rq2/sweeps/account_value_ensemble_pa_k10b20.csv"
VERSION_ORDER = rq3_config.VERSION_ORDER
VERSION_ORDER_DISPLAY = ["V1", "V2", "V3", "V4", "V5", "V6"]
def _disp_v(v: str) -> str:
    return v.upper()

# Palette for the six versions; the RQ2 reference curve is drawn in black.
VERSION_COLORS = {
    "v1": "#1f77b4",  # blue
    "v2": "#ff7f0e",  # orange
    "v3": "#2ca02c",  # green
    "v4": "#d62728",  # red
    "v5": "#9467bd",  # purple
    "v6": "#8c564b",  # brown
}
RQ2_COLOR = "black"


def load_account(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df


def drawdown_series(vals: np.ndarray) -> np.ndarray:
    peak = np.maximum.accumulate(vals)
    # avoid div0 (vals[0] >0 always)
    return (vals - peak) / peak


def resolve_top_tags() -> dict[str, str]:
    """Read sweep_summary_{v}.csv and pick Sharpe-best tag; fallback to manifest_top.json."""
    best: dict[str, str] = {}
    for v in VERSION_ORDER:
        summary_path = os.path.join(RESULTS_ROOT, "sweeps", f"sweep_summary_{v}.csv")
        if os.path.exists(summary_path):
            df = pd.read_csv(summary_path)
            df = df.sort_values("sharpe", ascending=False)
            best[v] = str(df.iloc[0]["tag"])
        else:
            # fallback to manifest_top.json
            manifest_path = os.path.join(RESULTS_ROOT, "manifest_top.json")
            if os.path.exists(manifest_path):
                with open(manifest_path) as f:
                    man = json.load(f)
                tag = man.get("versions_best", {}).get(v)
                if tag:
                    best[v] = tag
                    continue
            raise SystemExit(f"Missing {summary_path} and no manifest fallback for {v}")
    return best


def main():
    os.makedirs(RESULTS_ROOT, exist_ok=True)

    top_tags = resolve_top_tags()
    print(f"Top tags (Sharpe-best): {top_tags}")

    accounts: dict[str, pd.DataFrame] = {}
    for v in VERSION_ORDER:
        tag = top_tags[v]
        path = os.path.join(RESULTS_ROOT, "sweeps", f"account_value_{v}_{tag}.csv")
        if not os.path.exists(path):
            # fallback to canonical ver_v (should not happen in top-only but warn)
            fallback = os.path.join(RESULTS_ROOT, f"ver_{v}", "account_ensemble.csv")
            print(f"[warn] {path} not found, falling back to {fallback}")
            path = fallback
        if not os.path.exists(path):
            raise SystemExit(f"Missing account for {v}: {path}")
        accounts[v] = load_account(path)
        print(f"{v} ({tag}): {len(accounts[v])} rows {accounts[v]['date'].iloc[0]} -> {accounts[v]['date'].iloc[-1]}")

    if not os.path.exists(RQ2_PATH):
        # try glob fallback results/rq2*/sweeps/...
        cands = glob.glob("results/rq2*/sweeps/account_value_ensemble_pa_k10b20.csv")
        if cands:
            rq2_path = sorted(cands)[0]
            print(f"[note] {RQ2_PATH} not found, using {rq2_path}")
            rq2_src = rq2_path
        else:
            raise SystemExit(f"RQ2 floor not found: {RQ2_PATH}")
    else:
        rq2_src = RQ2_PATH
    rq2_df = load_account(rq2_src)
    print(f"RQ2 floor: {len(rq2_df)} rows {rq2_df['date'].iloc[0]} -> {rq2_df['date'].iloc[-1]}")

    # align on anchor dates (v1)
    anchor = accounts[VERSION_ORDER[0]]
    dates = anchor["date"].tolist()
    # build wide portfolio_value frame
    wide = pd.DataFrame({"date": dates})
    for v in VERSION_ORDER:
        wide[v] = accounts[v].set_index("date")["portfolio_value"].reindex(dates).values
    wide[RQ2_LABEL] = rq2_df.set_index("date")["portfolio_value"].reindex(dates).values
    # drop any misaligned tail
    wide = wide.dropna().reset_index(drop=True)
    print(f"Aligned rows: {len(wide)}  dates {wide['date'].iloc[0]} -> {wide['date'].iloc[-1]}")
    dates_dt = pd.to_datetime(wide["date"])

    # compute drawdowns
    dd = pd.DataFrame({"date": wide["date"]})
    for col in VERSION_ORDER + [RQ2_LABEL]:
        dd[col] = drawdown_series(wide[col].to_numpy())
    # also log max drawdown per series
    print("\nMax drawdown (min of underwater):")
    for col in VERSION_ORDER + [RQ2_LABEL]:
        print(f"  {col:12s} {dd[col].min():.4f} ({dd[col].min()*100:.2f}%)  tag={top_tags.get(col, 'k10b20') if col != RQ2_LABEL else 'k10b20'}")
    # compare to backtest_performance_comparison_top.csv if exists
    comp_path = os.path.join(RESULTS_ROOT, "backtest_performance_comparison_top.csv")
    if os.path.exists(comp_path):
        comp = pd.read_csv(comp_path, index_col=0)
        if "Max drawdown" in comp.index:
            print("\nvs backtest_performance_comparison_top.csv Max drawdown:")
            for col in VERSION_ORDER:
                csv_key = col  # top csv uses simple v1..v6
                if csv_key in comp.columns:
                    csv_dd = float(comp.loc["Max drawdown", csv_key])
                    calc_dd = float(dd[col].min())
                    diff = calc_dd - csv_dd
                    print(f"  {col} csv={csv_dd:.4f} calc={calc_dd:.4f} diff={diff:+.2e}")
            if RQ2_LABEL in comp.columns:
                csv_dd = float(comp.loc["Max drawdown", RQ2_LABEL])
                calc_dd = float(dd[RQ2_LABEL].min())
                print(f"  {RQ2_LABEL} csv={csv_dd:.4f} calc={calc_dd:.4f} diff={calc_dd-csv_dd:+.2e}")

    # save CSV (fractions, not percent, for numeric precision; plot shows %)
    csv_out = os.path.join(RESULTS_ROOT, "rq3_underwater.csv")
    dd.to_csv(csv_out, index=False)
    print(f"\nSaved {csv_out}")

    # plot - use display labels only
    fig, ax = plt.subplots(figsize=(15, 5))

    for v in VERSION_ORDER:
        ax.plot(dates_dt, dd[v].values, label=f"{_disp_v(v)} ({top_tags[v]})", color=VERSION_COLORS[v], lw=1.6, alpha=0.95, zorder=3)
    ax.plot(dates_dt, dd[RQ2_LABEL].values, label=RQ2_LABEL_DISPLAY, color=RQ2_COLOR, lw=2.0, zorder=6)

    ax.axhline(0, color="black", lw=0.8, alpha=0.6, zorder=1)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Drawdown (%)", fontsize=14)
    ax.set_title("Drawdown from Running Peak", fontsize=16)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(True, alpha=0.3)
    # give a little bottom padding so -30% not clipped
    dmin = dd[VERSION_ORDER + [RQ2_LABEL]].min().min()
    # dmin is negative, e.g. -0.26
    ax.set_ylim(dmin * 1.08 - 0.01, 0.02)
    ax.legend(fontsize=13, loc="lower right", framealpha=0.9, edgecolor="0.8")

    fig.tight_layout()
    png_out = os.path.join(RESULTS_ROOT, "rq3_underwater.png")
    fig.savefig(png_out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {png_out} ({os.path.getsize(png_out)} bytes)")
    print("Done.")


if __name__ == "__main__":
    main()
