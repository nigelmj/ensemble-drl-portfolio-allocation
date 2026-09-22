"""
RQ2 — Backtest comparison against shared baselines + manifest.

Compares the single-agent PPO account value, the 3 reward-variant standalone
curves, the blended ensemble PPO account value (soft/hard/block/k10b20), and
the shared transaction-cost-aware baselines, then writes a manifest logging the
training lineage.

Usage:
    python -m pipelines.rq2.rq2_train
    python -m pipelines.rq2.rq2_test_members
    python -m pipelines.rq2.rq2_ensemble
    python -m pipelines.rq2.rq2_sweep
    python -m pipelines.rq2.rq2_backtest

Output:
    results/rq2/backtest_performance_comparison.csv
    results/rq2/backtest_result.png
    results/rq2/ensemble_agent_weights.png
    results/rq2/manifest.json
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from pipelines.baselines.shared import LABELS, NAMES

from pipelines.rq2 import rq2_config
from rl_portfolio.config import RESULTS_DIR
from rl_portfolio.utils.io import check_and_make_directories
from rl_portfolio.utils.metrics import backtest_stats

VARIANT_NAMES = {
    "ensemble_hard": "Ensemble Hard",
    "ensemble_block20": "RQ2 block ensemble (k20b10)",
    "ensemble_block10": "RQ2 block ensemble (k10b10)",
    "ensemble_k10b20": "RQ2 block ensemble",
}

VARIANT_PATHS = {
    "ensemble_hard": f"{rq2_config.RESULTS_ROOT}/account_value_ensemble_pa_hard.csv",
    "ensemble_block20": f"{rq2_config.RESULTS_ROOT}/sweeps/account_value_ensemble_pa_k20b10.csv",
    "ensemble_block10": f"{rq2_config.RESULTS_ROOT}/sweeps/account_value_ensemble_pa_k10b10.csv",
    "ensemble_k10b20": f"{rq2_config.RESULTS_ROOT}/sweeps/account_value_ensemble_pa_k10b20.csv",
}

VARIANT_WEIGHT_PATHS = {
    "ensemble_hard": f"{rq2_config.RESULTS_ROOT}/ensemble_agent_weights_hard.csv",
    "ensemble_block20": f"{rq2_config.RESULTS_ROOT}/sweeps/ensemble_agent_weights_k20b10.csv",
    "ensemble_block10": f"{rq2_config.RESULTS_ROOT}/sweeps/ensemble_agent_weights_k10b10.csv",
    "ensemble_k10b20": f"{rq2_config.RESULTS_ROOT}/sweeps/ensemble_agent_weights_k10b20.csv",
}

LINE_LABELS = {
    "ensemble_ppo": "PPO ensemble",
    **rq2_config.MEMBER_NAMES,
    **VARIANT_NAMES,
}
LINE_COLORS = {
    "ensemble_ppo": "#1f77b4",
    "log_return": "#2ca02c",
    "dsr": "#ff7f0e",
    "dsr_drawdown": "#9467bd",
    "ensemble_hard": "#d62728",
    "ensemble_block20": "#bcbd22",
    "ensemble_block10": "#17becf",
    "ensemble_k10b20": "#000000",
    **{v: "#888888" for v in ["Equal Weight", "Equal Weight Rebalanced", "EW Rebalanced",
                               "MVO", "MVO Rebalanced", "Mean Var"]},
}


def load_account(path: str, col: str) -> pd.DataFrame:
    df = pd.read_csv(path).rename(columns={"portfolio_value": col})
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df


def main():
    check_and_make_directories([RESULTS_DIR, rq2_config.RESULTS_ROOT])

    member_accounts = {}
    for key in rq2_config.MEMBER_NAMES:
        path = f"{rq2_config.RESULTS_ROOT}/account_value_ppo_pa_{key}.csv"
        member_accounts[key] = load_account(path, key)
        print(f"Member {key}: {len(member_accounts[key])} rows")

    initial_amount = float(member_accounts["log_return"]["log_return"].iloc[0])
    dates = member_accounts["log_return"]["date"].tolist()

    df_ensemble = load_account(
        f"{rq2_config.RESULTS_ROOT}/account_value_ensemble_pa_soft.csv", "ensemble_ppo"
    )
    print(f"PPO ensemble: {len(df_ensemble)} rows")

    ensemble_variants = {}
    for key, path in VARIANT_PATHS.items():
        if os.path.exists(path):
            ensemble_variants[key] = load_account(path, key)
            print(f"{key}: {len(ensemble_variants[key])} rows")

    # Build the aligned frame with shared baselines.
    result = pd.DataFrame({"date": dates})
    result["ensemble_ppo"] = df_ensemble.set_index("date")["ensemble_ppo"].values
    for key, df in member_accounts.items():
        result[key] = df.set_index("date")[key].values
    for key, df in ensemble_variants.items():
        result[key] = df.set_index("date")[key].reindex(dates).values

    for name in NAMES:
        base = pd.read_csv(f"results/baselines/{name}.csv")
        base["date"] = pd.to_datetime(base["date"]).dt.strftime("%Y-%m-%d")
        s = base.set_index("date")["value"].reindex(dates)
        result[LABELS[name]] = (s / s.iloc[0] * initial_amount).values
    result = result.dropna().reset_index(drop=True)
    print(f"Aligned rows: {len(result)}")

    # Performance comparison table.
    cols = [c for c in result.columns if c != "date"]
    series = []
    for col in cols:
        df = result[["date", col]].rename(columns={col: "account_value"})
        series.append(backtest_stats(df, value_col_name="account_value").rename(col))
    table = pd.concat(series, axis=1).round(4)
    table.to_csv(f"{rq2_config.RESULTS_ROOT}/backtest_performance_comparison.csv")
    print("\n=== RQ2 Performance Comparison ===")
    print(table)

    # Equity plot.
    fig, ax = plt.subplots(figsize=(15, 6))
    plot_cols = [c for c in cols if c != "date"]
    for col in plot_cols:
        color = LINE_COLORS.get(col, "#333333")
        if col == "ensemble_k10b20":
            ax.plot(pd.to_datetime(result["date"]), result[col], color=color,
                    lw=3.0, zorder=5)
        elif col in ("EW Rebalanced", "MVO Rebalanced", "Equal Weight Rebalanced"):
            ax.plot(pd.to_datetime(result["date"]), result[col], color=color,
                    lw=1.0, linestyle="--", alpha=0.9)
        elif col in rq2_config.MEMBER_NAMES:
            ax.plot(pd.to_datetime(result["date"]), result[col], color=color,
                    lw=1.0, linestyle="--", alpha=0.7)
        else:
            ax.plot(pd.to_datetime(result["date"]), result[col], color=color,
                    lw=1.4)
    ax.set_title("RQ2 — Portfolio Value Over Time (Test Set: 2022-2026)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    def _disp(c):
        if c == "Mean Var":
            return "MVO"
        if c == "EW Rebalanced":
            return "Equal Weight Rebalanced"
        return LINE_LABELS.get(c, c)
    ax.legend([_disp(c) for c in plot_cols], loc="upper left",
              fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    for lab in ax.get_xticklabels():
        lab.set_rotation(45)
        lab.set_ha("right")
    fig.savefig(f"{rq2_config.RESULTS_ROOT}/backtest_result.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Blend weights plot.
    weights_df = pd.read_csv(f"{rq2_config.RESULTS_ROOT}/ensemble_agent_weights_soft.csv")
    weights_df["date"] = pd.to_datetime(weights_df["date"])
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(weights_df["date"], weights_df["w_log_return"], label="PPO LogReturn", lw=1.2)
    ax.plot(weights_df["date"], weights_df["w_dsr"], label="PPO DSR", lw=1.2)
    ax.plot(weights_df["date"], weights_df["w_dsr_drawdown"], label="PPO DSR-DD", lw=1.2)
    ax.set_title("Soft-blend Weights over the Test Period", fontsize=15)
    ax.set_xlabel("Date", fontsize=13)
    ax.set_ylabel("Weight", fontsize=13)
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.20, 0.50)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    for lab in ax.get_xticklabels():
        lab.set_rotation(45)
        lab.set_ha("right")
        lab.set_fontsize(11)
    ax.tick_params(axis="y", labelsize=11)
    fig.tight_layout()
    fig.savefig(f"{rq2_config.RESULTS_ROOT}/ensemble_agent_weights.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "research_question": "RQ2 - reward diversity (3-reward PPO ensemble)",
        "models": {
            "lineage": "ALL THREE members trained in rq2_train.py "
                       "(1M steps, seed 42)",
            "reward_scaling_dsr": rq2_config.REWARD_SCALING_DSR,
            "dsr_eta_dd": rq2_config.DSR_ETA_DD,
            "model_dir": rq2_config.MODEL_DIR,
            "total_timesteps": rq2_config.TOTAL_TIMESTEPS,
        },
        "data": {
            "train": rq2_config.DATA_TRAIN,
            "test": rq2_config.DATA_TEST,
            "test_window": f"{dates[0]} to {dates[-1]}",
        },
        "variants": {
            "soft": "k=10 (default in rq2_ensemble.py)",
            "hard": "daily one-hot",
            "block": "k=10, block_days=20 (k10b20 comparison floor)",
            "sweep": "12-config grid results/rq2/sweeps/",
        },
        "baselines": {
            "source": "results/baselines/ (compute_baselines.py)",
            "names": NAMES,
            "labels": LABELS,
        },
        "outputs": rq2_config.RESULTS_ROOT,
    }
    with open(f"{rq2_config.RESULTS_ROOT}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved to {rq2_config.RESULTS_ROOT}/manifest.json")


if __name__ == "__main__":
    main()
