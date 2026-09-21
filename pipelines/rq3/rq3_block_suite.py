"""
RQ3 — Cross-model version suite (block mode, k10b20).

Runs the six reward-algo versions (v1..v6) of RQ3 under the block blend mode
at the k10b20 configuration (the RQ2 sweep winner) over
the shared test window (2022-2026). Each version is written to its own folder:

    results/0rq3/ver_{vN}/
        account_ensemble.csv
        actions_ensemble.csv
        weights_ensemble.csv
        member_account_{algo}_{reward}.csv
        member_actions_{algo}_{reward}.csv
        backtest_performance_comparison.csv   stats vs RQ2 k10b20 + baselines
        backtest_equity.png

Members: PPO members come from rq_trained_models/rq2 (copied into
rq_trained_models/rq3); SAC and A2C members are the
rq_trained_models/rq3/{sac,a2c}_* models (see rq3_train.py). Copy all nine
into rq_trained_models/rq3 before running.

Usage:
    python -m pipelines.rq3.rq3_train --algo sac --reward-type log_return
    python -m pipelines.rq3.rq3_train --algo a2c --reward-type log_return
    python -m pipelines.rq3.rq3_calibrate
    python -m pipelines.rq3.rq3_block_suite

Output:
    results/0rq3/ver_vN/*
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipelines.baselines.shared import LABELS

from pipelines.rq3 import rq3_config
from rl_portfolio.agents.cross_model_ensemble_agent import (
    CrossModelPerfWeightedAgent,
)
from rl_portfolio.agents.portfolio_allocation_agent import (
    MODELS,
    PortfolioAllocationDRLAgent,
)
from rl_portfolio.config import RESULTS_DIR
from rl_portfolio.utils.io import check_and_make_directories
from rl_portfolio.envs.portfolio_allocation_env import PortfolioAllocationEnv
from rl_portfolio.utils.metrics import backtest_stats

K = 10
BLOCK_DAYS = 20

# RQ2's k10b20 sweep winner = the RQ3 comparison floor.
RQ2_K10B20_ACCOUNT = "results/0rq2/sweeps/account_value_ensemble_pa_k10b20.csv"
RQ2_K10B20_LABEL = "RQ2 k10b20"
RQ2_K10B20_LABEL_DISPLAY = "RQ2 block ensemble"
def _display_col(col: str) -> str:
    if col == RQ2_K10B20_LABEL:
        return RQ2_K10B20_LABEL_DISPLAY
    if col == "Mean Var":
        return "MVO"
    if col == "EW Rebalanced":
        return "Equal Weight Rebalanced"
    if col.startswith("v") and " block k" in col:
        # e.g. v1 block k10b20 -> V1 block k10b20
        return "V" + col[1:]
    if col.startswith("v") and col[1:2].isdigit():
        return col.upper().replace("V", "V", 1)
    return col.replace("Mean Var", "MVO")

REBALANCE_DAYS = 21
FEE_PCT = 0.001


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default=None,
                        help="One of v1..v6 (default: all six versions).")
    parser.add_argument("--k", type=int, default=K,
                        help=f"Trailing metric window in days (default: {K}).")
    parser.add_argument("--block-days", type=int, default=BLOCK_DAYS,
                        help=f"Block tenure in days (default: {BLOCK_DAYS}).")
    return parser.parse_args()


def compute_metrics(account_df: pd.DataFrame, label: str) -> dict:
    vals = account_df["portfolio_value"].values
    total_return = vals[-1] / vals[0] - 1
    daily_returns = np.diff(vals) / vals[:-1]
    # Use ddof=1 to match pyfolio/empyrical (sample std, Sharpe-consistent)
    vol_daily = float(np.std(daily_returns, ddof=1)) if len(daily_returns) > 1 else 0.0
    if vol_daily > 0:
        sharpe = np.sqrt(252) * np.mean(daily_returns) / vol_daily
    else:
        sharpe = np.inf if np.mean(daily_returns) > 0 else 0.0
    peak = np.maximum.accumulate(vals)
    max_drawdown = float(np.min((vals - peak) / peak))
    return {
        "label": label,
        "final_value": float(vals[-1]),
        "total_return": float(total_return),
        "sharpe": float(sharpe),
        "max_drawdown": max_drawdown,
        "avg_daily_return": float(np.mean(daily_returns)),
        "volatility": vol_daily,
    }


def run_individual(df, member_spec, out_dir, name):
    model = MODELS[member_spec["algo"]].load(member_spec["path"])
    env = PortfolioAllocationEnv(df=df, **member_spec["env_kwargs"])
    account, actions, stats = PortfolioAllocationDRLAgent.DRL_prediction(
        model=model, environment=env, deterministic=True
    )
    account.to_csv(f"{out_dir}/member_account_{name}.csv", index=False)
    actions.to_csv(f"{out_dir}/member_actions_{name}.csv")
    metrics = compute_metrics(account, name)
    print(f"  {name:<24} Sharpe={metrics['sharpe']:.3f} "
          f"Return={metrics['total_return']:.2%} "
          f"MaxDD={metrics['max_drawdown']:.2%}")
    return metrics


def run_version(df, version, out_dir, k, block_days):
    disp_ver = version.upper()
    print("\n" + "=" * 60)
    print(f"RQ3 {disp_ver} — block k{k}b{block_days}")
    print(" ".join(f"{m['algo']}={m['reward_type']}"
                   for m in rq3_config.VERSIONS[version]))
    print("=" * 60)

    calibration = rq3_config.load_calibration()
    member_specs = rq3_config.build_member_specs(
        rq3_config.VERSIONS[version], calibration)

    print("  Loading members:")
    for m in member_specs:
        if not os.path.exists(m["path"] + ".zip"):
            raise SystemExit(f"Model not found: {m['path']}.zip")
        print(f"    {m['algo']:<5} {m['reward_type']:<12} {m['path']}")

    agent = CrossModelPerfWeightedAgent(
        member_specs=member_specs,
        mode="block",
        k=k,
        block_days=block_days,
    )
    account, actions, weights = agent.run(df=df, base_env_kwargs=rq3_config.BASE_ENV_KWARGS)
    account.to_csv(f"{out_dir}/account_ensemble.csv", index=False)
    actions.to_csv(f"{out_dir}/actions_ensemble.csv")
    weights.to_csv(f"{out_dir}/weights_ensemble.csv", index=False)

    metrics = compute_metrics(account, version)
    print(f"  {version:<8} Sharpe={metrics['sharpe']:.3f} "
          f"Return={metrics['total_return']:.2%} "
          f"MaxDD={metrics['max_drawdown']:.2%}")

    member_results = [metrics]
    print("\n  Individual members (reward-specific envs):")
    for m in member_specs:
        name = f"{m['algo']}_{m['reward_type']}"
        member_results.append(run_individual(df, m, out_dir, name))
    return member_results


def load_baselines(dates, initial_amount):
    """Shared baselines (results/baselines/*.csv), anchored to the account start."""
    out = {"date": dates}
    for name in ["ew_buyhold", "ew_rebalanced", "mvo_buyhold", "mvo_rebalanced"]:
        df = pd.read_csv(f"results/baselines/{name}.csv")
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        s = df.set_index("date")["value"].reindex(dates)
        out[LABELS[name]] = (s / s.iloc[0] * initial_amount).values
    return out


def backtest_version(out_dir, version, k, block_days, rq2_account):
    print(f"\n  Backtest {version} ...")
    account_path = f"{out_dir}/account_ensemble.csv"
    if not os.path.exists(account_path):
        print(f"[skip] {account_path} not found")
        return None

    def _read(path):
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        return df

    account = _read(account_path)
    member_map = {}
    for f in os.listdir(out_dir):
        if f.startswith("member_account_"):
            member_map[f[len("member_account_"):-4]] = f

    rq2 = _read(rq2_account) if os.path.exists(rq2_account) else None
    if rq2 is None:
        print(f"[skip comparison] {rq2_account} not found")
        rq2 = None

    dates = account["date"].tolist()
    initial_amount = float(account["portfolio_value"].iloc[0])
    print(f"  anchor {len(dates)} days {dates[0]} -> {dates[-1]}, "
          f"initial {initial_amount:.0f}")

    baselines = load_baselines(dates, initial_amount)

    label = f"{version} block k{k}b{block_days}"
    result = pd.DataFrame({"date": dates})
    result[label] = account.set_index("date")["portfolio_value"].reindex(dates).values
    if rq2 is not None:
        result[RQ2_K10B20_LABEL] = (
            rq2.set_index("date")["portfolio_value"].reindex(dates).values)
    for key, f in sorted(member_map.items()):
        result[f"Member {key}"] = (
            _read(f"{out_dir}/{f}").set_index("date")["portfolio_value"]
            .reindex(dates).values)
    for col in LABELS.values():
        result[col] = baselines[col]
    result = result.dropna()

    cols = [c for c in result.columns if c != "date"]
    series = []
    for col in cols:
        df = result[["date", col]].rename(columns={col: "account_value"})
        series.append(backtest_stats(df, value_col_name="account_value").rename(col))
    table = pd.concat(series, axis=1).round(4)
    table.to_csv(f"{out_dir}/backtest_performance_comparison.csv")
    print(f"  Saved stats to {out_dir}/backtest_performance_comparison.csv")

    fig, ax = plt.subplots(figsize=(15, 6))
    for col in result.columns:
        if col == "date":
            continue
        disp = _display_col(col)
        if col in (label, RQ2_K10B20_LABEL):
            ax.plot(pd.to_datetime(result["date"]), result[col], label=disp, lw=2.2)
        elif col.startswith("Member "):
            ax.plot(pd.to_datetime(result["date"]), result[col],
                    label=disp, lw=1.2, linestyle="--")
        else:
            ax.plot(pd.to_datetime(result["date"]), result[col],
                    label=disp, lw=1.4, linestyle=":")
    disp_label = _display_col(label)
    ax.set_title(f"RQ3 {disp_label} — Test 2022-2026")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig(f"{out_dir}/backtest_equity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot to {out_dir}/backtest_equity.png")
    return result


def main():
    args = parse_args()
    versions = [args.version] if args.version else list(rq3_config.VERSIONS)
    check_and_make_directories([RESULTS_DIR, rq3_config.RESULTS_ROOT,
                                rq3_config.MODEL_DIR])

    test = pd.read_pickle(rq3_config.DATA_TEST)
    print(f"Test: {len(test)} rows, {test.date.nunique()} days, "
          f"{test.tic.nunique()} tickers")

    rq2_account = RQ2_K10B20_ACCOUNT
    if not os.path.exists(rq2_account):
        print(f"\n[note] RQ2 comparison floor not found: {rq2_account} "
              f"(will omit from backtest tables)")

    version_rows = []
    for v in versions:
        out_dir = f"{rq3_config.RESULTS_ROOT}/ver_{v}"
        check_and_make_directories([out_dir])
        rows = run_version(test, v, out_dir, k=args.k, block_days=args.block_days)
        backtest_version(out_dir, v, args.k, args.block_days, rq2_account)
        for r in rows:
            r["version"] = v
            version_rows.append(r)

    summary = pd.DataFrame(version_rows)
    summary_path = f"{rq3_config.RESULTS_ROOT}/summary_versions.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\n=== RQ3 version summary (sorted by Sharpe) ===")
    print(summary.sort_values("sharpe", ascending=False)
          [["version", "label", "sharpe", "total_return", "max_drawdown"]]
          .round(4).to_string(index=False))
    print(f"Saved to {summary_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
