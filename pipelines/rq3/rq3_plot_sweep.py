"""
RQ3 — Per-version sweep timeline plots (like RQ2's rq2_plot_selection.py).

For each version v1..v6, picks the top 3 block configs by Sharpe from
sweep_summary_{version}.csv and plots:
  top: account value curves (ensemble soft/hard not applicable for RQ3,
       so shows the 3 winners only) + RQ2 k10b20 floor if present;
  bottom: Gantt of which member (ppo/sac/a2c) was selected per block
  for each of the 3 winners.

Usage:
    python -m pipelines.rq3.rq3_plot_sweep
    python -m pipelines.rq3.rq3_plot_sweep --version v4

Output:
    results/0rq3/sweeps/{version}_ensemble_selection_timeline.png (6 files)
"""
from __future__ import annotations

import os
import argparse
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
import pandas as pd


SWEEP_DIR = "results/0rq3/sweeps"
RQ3_ROOT = "results/0rq3"

# Colours for the three members (PPO / SAC / A2C).
MEMBER_COLORS = {
    "ppo": "#4C72B0",
    "sac": "#DD8452",
    "a2c": "#55A868",
}
MEMBER_LABELS = {"ppo": "PPO", "sac": "SAC", "a2c": "A2C"}
MEMBER_ORDER = ["ppo", "sac", "a2c"]
BLOCK_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # for top 3 winners


def top_tags(version: str, n: int = 3) -> list[str]:
    path = f"{SWEEP_DIR}/sweep_summary_{version}.csv"
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    df = df.sort_values("sharpe", ascending=False)
    return df["tag"].head(n).tolist()


def runs_of(winner: pd.Series):
    starts, ends, labels = [], [], []
    if len(winner) == 0:
        return []
    start = winner.index[0]
    prev = winner.iloc[0]
    for i in range(1, len(winner)):
        if winner.iloc[i] != prev:
            ends.append(winner.index[i - 1])
            starts.append(start)
            labels.append(prev)
            start = winner.index[i]
            prev = winner.iloc[i]
    ends.append(winner.index[-1])
    starts.append(start)
    labels.append(prev)
    return list(zip(starts, ends, labels))


def _disp_version(v: str) -> str:
    return v.upper()

def _disp_tag(v: str, tag: str) -> str:
    return f"{v.upper()} {tag}"

def plot_version(version: str):
    tags = top_tags(version)
    if not tags:
        print(f"[{_disp_version(version)}] no summary found, skipping")
        return
    print(f"== {_disp_version(version)}: top {len(tags)} -> {tags}")

    fig, (ax_top, ax_sel) = plt.subplots(
        2, 1, figsize=(15, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 4]},
    )

    # Top: equity curves for top 3 sweep winners
    for tag, color in zip(tags, BLOCK_COLORS):
        path = f"{SWEEP_DIR}/account_value_{version}_{tag}.csv"
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        ax_top.plot(df["date"], df["portfolio_value"], label=_disp_tag(version, tag), color=color, lw=1.4)

    # RQ2 floor for reference
    rq2_path = "results/0rq2/sweeps/account_value_ensemble_pa_k10b20.csv"
    if os.path.exists(rq2_path):
        rq2 = pd.read_csv(rq2_path)
        rq2["date"] = pd.to_datetime(rq2["date"])
        ax_top.plot(rq2["date"], rq2["portfolio_value"], label="RQ2 block ensemble", color="black", lw=1.2, linestyle="--", alpha=0.7)

    ax_top.set_ylabel("Portfolio value ($)")
    ax_top.set_title(f"{_disp_version(version)} — Top 3 Sweep Configurations (k × block)", fontsize=15)
    ax_top.legend(loc="upper left", fontsize=9, ncol=2)
    ax_top.grid(True, alpha=0.3)

    n = len(tags)
    # Determine member algo for each row from weights file: which of ppo/sac/a2c had max weight?
    # weights_{version}_{tag}.csv has columns w_{reward} (one per member reward).
    for i, tag in enumerate(tags):
        wpath = f"{SWEEP_DIR}/weights_{version}_{tag}.csv"
        if not os.path.exists(wpath):
            continue
        w = pd.read_csv(wpath)
        w["date"] = pd.to_datetime(w["date"])
        w = w.set_index("date")
        # detect weight columns
        cols = [c for c in w.columns if c.startswith("w_")]
        # map to member order: try w_ppo, w_sac, w_a2c
        # fallback to sorted cols
        if set(["w_ppo", "w_sac", "w_a2c"]).issubset(set(cols)):
            order = ["w_ppo", "w_sac", "w_a2c"]
            label_map = {"w_ppo": "ppo", "w_sac": "sac", "w_a2c": "a2c"}
        else:
            order = sorted(cols)
            label_map = {c: c.replace("w_", "") for c in order}
        winner = w[order].idxmax(axis=1)
        # map to algo label
        winner = winner.map(lambda x: label_map.get(x, x))
        runs = runs_of(winner)
        y = n - 1 - i
        for (s, e, win) in runs:
            left = mdates.date2num(s)
            width = mdates.date2num(e) - left + 1.0
            ax_sel.barh(y, width, left=left, height=0.6, color=MEMBER_COLORS.get(win, "#888888"), edgecolor="none")

    ax_sel.set_yticks([])
    ax_sel.set_ylim(-0.6, n - 0.4)
    ax_sel.set_xlabel("Date")
    ax_sel.xaxis.set_major_locator(mdates.YearLocator())
    ax_sel.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_sel.grid(True, axis="x", alpha=0.3)

    trans = blended_transform_factory(ax_sel.transAxes, ax_sel.transData)
    for i, tag in enumerate(tags):
        ax_sel.text(-0.02, n - 1 - i, _disp_tag(version, tag), va="center", ha="right", fontsize=9, transform=trans)

    ax_sel.set_title("Member Selection Over the Test Period", fontsize=13)
    handles = [plt.Rectangle((0, 0), 1, 1, color=MEMBER_COLORS[k]) for k in MEMBER_ORDER]
    ax_sel.legend(handles, [MEMBER_LABELS[k] for k in MEMBER_ORDER], loc="upper left", fontsize=9, ncol=3)

    fig.tight_layout()
    out = f"{SWEEP_DIR}/{version}_ensemble_selection_timeline.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved to {out}")
    # also copy to report
    rpt = f"results/report/rq3_sweep_timeline_{version}.png"
    try:
        os.makedirs("results/report", exist_ok=True)
        shutil.copy(out, rpt)
        print(f"  Copied to {rpt}")
    except Exception as e:
        print(f"  Warning copy failed: {e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--version", default=None, help="one of v1..v6")
    args = p.parse_args()
    versions = [args.version] if args.version else [f"v{i}" for i in range(1, 7)]
    for v in versions:
        plot_version(v)


if __name__ == "__main__":
    main()
