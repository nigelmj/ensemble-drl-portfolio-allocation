"""
RQ2 — Plot which reward-variant model is selected, by period, during trading.

Auto-discovers every results/rq2*/sweeps/ensemble_sweep_summary.csv, picks the
top 3 block configs by Sharpe from each summary, and plots the account-value
curves (top) and a Gantt-style timeline (bottom) of which agent the ensemble
selected at each point in time for those top-3 configs.

Usage:
    python -m pipelines.rq2.rq2_sweep
    python -m pipelines.rq2.rq2_plot_selection

Output:
    results/rq2*/sweeps/ensemble_selection_timeline.png (one per folder)
"""
from __future__ import annotations

import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
import pandas as pd

from rl_portfolio.utils.io import check_and_make_directories

AGENT_COLORS = {
    "w_log_return": "#4C72B0",
    "w_dsr": "#DD8452",
    "w_dsr_drawdown": "#55A868",
}
AGENT_LABELS = {"w_log_return": "PPO LogReturn", "w_dsr": "PPO DSR",
                "w_dsr_drawdown": "PPO DSR-DD"}
AGENT_ORDER = ["w_log_return", "w_dsr", "w_dsr_drawdown"]

BLOCK_COLORS = ["black", "#2ca02c", "#ff7f0e"]  # k10b20 black, then green, orange per request


def find_sweep_roots() -> list[str]:
    summary_paths = sorted(
        glob.glob("results/rq2*/sweeps/ensemble_sweep_summary.csv")
    )
    return [os.path.dirname(os.path.dirname(p)) for p in summary_paths]


def top_tags(root: str, n: int = 3) -> list[str]:
    summary = pd.read_csv(f"{root}/sweeps/ensemble_sweep_summary.csv")
    summary = summary.sort_values("sharpe", ascending=False)
    return summary["tag"].head(n).tolist()


def runs_of(winner: pd.Series):
    starts, ends, labels = [], [], []
    if len(winner) == 0:
        return
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


def plot_root(root: str):
    check_and_make_directories([f"{root}/sweeps"])

    tags = top_tags(root)
    print(f"== {root}: top {len(tags)} by Sharpe -> {tags}")

    fig, (ax_top, ax_sel) = plt.subplots(
        2, 1, figsize=(15, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 4]},
    )

    soft = pd.read_csv(f"{root}/account_value_ensemble_pa_soft.csv")
    soft["date"] = pd.to_datetime(soft["date"])
    # keep handles for explicit legend ordering (first column top3, second column soft/hard)
    handles_dict = {}
    # display helpers - just config names for top 3
    def _disp_tag(t: str) -> str:
        return t
    l_soft, = ax_top.plot(soft["date"], soft["portfolio_value"], label="Soft Blend",
                color="#1f77b4", lw=1.2)
    handles_dict["Soft Blend"] = l_soft

    hard = pd.read_csv(f"{root}/account_value_ensemble_pa_hard.csv")
    hard["date"] = pd.to_datetime(hard["date"])
    l_hard, = ax_top.plot(hard["date"], hard["portfolio_value"], label="Hard Blend",
                color="#d62728", lw=1.2)
    handles_dict["Hard Blend"] = l_hard

    for tag, color in zip(tags, BLOCK_COLORS):
        df = pd.read_csv(f"{root}/sweeps/account_value_ensemble_pa_{tag}.csv")
        df["date"] = pd.to_datetime(df["date"])
        disp = _disp_tag(tag)
        l, = ax_top.plot(df["date"], df["portfolio_value"], label=disp, color=color,
                    lw=1.4)
        handles_dict[disp] = l
    # Buy-and-hold baselines (dashed) — anchored to ensemble start value
    try:
        initial_amount = float(soft["portfolio_value"].iloc[0])
        for base_name, base_label, base_color in [
            ("ew_buyhold", "Equal Weight", "#888888"),
            ("mvo_buyhold", "MVO", "#444444"),
        ]:
            base_path = f"results/baselines/{base_name}.csv"
            if os.path.exists(base_path):
                base = pd.read_csv(base_path)
                base["date"] = pd.to_datetime(base["date"])
                base = base.set_index("date").reindex(soft["date"]).dropna()
                if not base.empty:
                    # anchor to the ensemble's start value so the curves are comparable
                    s = base["value"] / base["value"].iloc[0] * initial_amount
                    l_base, = ax_top.plot(s.index, s.values, label=base_label,
                                color=base_color, lw=1.2, linestyle="--", alpha=0.85)
                    handles_dict[base_label] = l_base
    except Exception as e:
        print(f"  Warning: could not add buy-hold baselines: {e}")
    ax_top.set_ylabel("Portfolio value ($)")
    ax_top.set_title("Performance of RQ2 Block Ensemble (Top 3) Against Soft and Hard Blends", fontsize=15)
    # Legend: first column top3 (k10b20 black, then green/orange), second column soft (blue) & hard (red)
    # Legend fills column-wise with ncol=3, so order must be column-major: [top3 | soft/hard | baselines]
    if tags:
        order = [_disp_tag(tags[0]), _disp_tag(tags[1]), _disp_tag(tags[2]), "Soft Blend", "Hard Blend", "Equal Weight", "MVO"]
    else:
        order = ["k10b20", "k10b10", "k10b5", "Soft Blend", "Hard Blend", "Equal Weight", "MVO"]
    ordered_handles = [handles_dict[l] for l in order if l in handles_dict]
    ordered_labels = [l for l in order if l in handles_dict]
    ax_top.legend(ordered_handles, ordered_labels, loc="upper left", fontsize=11, ncol=3)
    ax_top.grid(True, alpha=0.3)

    n = len(tags)
    for i, tag in enumerate(tags):
        w = pd.read_csv(f"{root}/sweeps/ensemble_agent_weights_{tag}.csv")
        w["date"] = pd.to_datetime(w["date"])
        w = w.set_index("date")
        winner = w[AGENT_ORDER].idxmax(axis=1)

        runs = runs_of(winner)
        y = n - 1 - i
        for (s, e, win) in runs:
            left = mdates.date2num(s)
            width = mdates.date2num(e) - left + 1.0
            ax_sel.barh(y, width, left=left, height=0.6,
                        color=AGENT_COLORS[win], edgecolor="none")

    ax_sel.set_yticks([])
    ax_sel.set_ylim(-0.6, n - 0.4)
    ax_sel.set_xlabel("Date")
    ax_sel.xaxis.set_major_locator(mdates.YearLocator())
    ax_sel.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_sel.grid(True, axis="x", alpha=0.3)

    trans = blended_transform_factory(ax_sel.transAxes, ax_sel.transData)
    for i, tag in enumerate(tags):
        ax_sel.text(-0.02, n - 1 - i, _disp_tag(tag), va="center",
                    ha="right", fontsize=10, transform=trans)

    ax_sel.set_title("RQ2 Block Ensemble — Reward Variant Selection Over the Test Period", fontsize=13)
    handles = [plt.Rectangle((0, 0), 1, 1, color=AGENT_COLORS[k])
               for k in AGENT_ORDER]
    ax_sel.legend(handles, [AGENT_LABELS[k] for k in AGENT_ORDER],
                  loc="upper left", fontsize=11, ncol=3)

    fig.tight_layout()
    out = f"{root}/sweeps/ensemble_selection_timeline.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved to {out}")


def main():
    roots = find_sweep_roots()
    if not roots:
        print("No results/rq2*/sweeps/ensemble_sweep_summary.csv found.")
        return
    for root in roots:
        plot_root(root)


if __name__ == "__main__":
    main()
