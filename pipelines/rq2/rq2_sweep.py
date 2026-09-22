"""
RQ2 — sweep metric window (k) x block tenure (block_days).

Runs the block-mode ensemble over the 12-run grid and writes per-config
account / actions / weights files plus two tables into results/rq2/sweeps/.

Usage:
    python -m pipelines.rq2.rq2_ensemble --mode block --k 10 --block-days 20
    python -m pipelines.rq2.rq2_sweep
    python -m pipelines.rq2.rq2_sweep --k 10 20 40 --block 5 10 20 40

Output:
    results/rq2/sweeps/ensemble_sweep_summary.csv
    results/rq2/sweeps/ensemble_fee_sensitivity.csv
    results/rq2/sweeps/account_value_ensemble_pa_{tag}.csv (12 configs)
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from pipelines.rq2 import rq2_config
from rl_portfolio.agents.portfolio_allocation_ensemble_agent import (
    PortfolioAllocationEnsembleAgent,
)
from rl_portfolio.config import RESULTS_DIR
from rl_portfolio.utils.io import check_and_make_directories
from rl_portfolio.utils.metrics import backtest_stats

SWEEP_DIR = f"{rq2_config.RESULTS_ROOT}/sweeps"

FEE_LEVELS = [0.0, 0.0005, 0.001, 0.002, 0.005]

REFERENCE_ACTIONS = {
    "soft": "actions_ensemble_pa_soft.csv",
    "hard": "actions_ensemble_pa_hard.csv",
    "log_return": "actions_ppo_pa_log_return.csv",
    "dsr": "actions_ppo_pa_dsr.csv",
    "dsr_drawdown": "actions_ppo_pa_dsr_drawdown.csv",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", nargs="+", type=int, default=[10, 20, 40],
                        help="Trailing metric windows to sweep (days).")
    parser.add_argument("--block", nargs="+", type=int, default=[5, 10, 20, 40],
                        help="Block tenures to sweep (days).")
    return parser.parse_args()


def evolve(actions_path: str, pivot: pd.DataFrame, fee_pct: float) -> float:
    """Replay an actions file with the env's exact fee formula."""
    a = pd.read_csv(actions_path)
    cols = list(pivot.columns)
    w = a[["cash"] + cols].fillna(0.0).to_numpy()
    w = w / w.sum(axis=1, keepdims=True)
    pos = {x: i for i, x in enumerate(pivot.index)}
    i = pos[str(a["date"].iloc[0])]

    value = rq2_config.ENV_KWARGS["initial_amount"]
    old = np.array([1.0] + [0.0] * len(cols))
    for t in range(len(w)):
        target = w[t]
        turnover = np.sum(np.abs(target - old))
        value -= turnover * value * fee_pct
        ratio = pivot.iloc[i].to_numpy() / pivot.iloc[i - 1].to_numpy()
        price_ratios = np.concatenate([[1.0], ratio])
        value *= np.sum(target * price_ratios)
        old = (target * price_ratios) / np.sum(target * price_ratios)
        i += 1
    return value


def main():
    args = parse_args()
    check_and_make_directories([RESULTS_DIR, SWEEP_DIR])
    test = pd.read_pickle(rq2_config.DATA_TEST)

    summary_rows = []
    for k in args.k:
        for block in args.block:
            tag = f"k{k}b{block}"
            t0 = time.time()
            print(f"\n=== k={k}, block_days={block} ({tag}) ===")

            agent = PortfolioAllocationEnsembleAgent(
                model_paths=rq2_config.DEFAULT_MODEL_PATHS,
                df=test,
                env_kwargs=rq2_config.ENV_KWARGS,
                k=k,
                temperature=1.0,
                alpha=0.3,
                mode="block",
                block_days=block,
            )
            account_df, actions_df, weights_df = agent.run()

            account_df.to_csv(f"{SWEEP_DIR}/account_value_ensemble_pa_{tag}.csv",
                              index=False)
            actions_df.to_csv(f"{SWEEP_DIR}/actions_ensemble_pa_{tag}.csv")
            weights_df.to_csv(f"{SWEEP_DIR}/ensemble_agent_weights_{tag}.csv",
                              index=False)

            stats = backtest_stats(
                account_df.rename(columns={"portfolio_value": "account_value"}),
                value_col_name="account_value",
            )
            summary_rows.append(
                {
                    "k": k,
                    "block_days": block,
                    "tag": tag,
                    "final_value": account_df["portfolio_value"].iloc[-1],
                    "total_return": stats.get("Cumulative returns", np.nan),
                    "sharpe": stats.get("Sharpe ratio", np.nan),
                    "max_drawdown": stats.get("Max drawdown", np.nan),
                    "annual_vol": stats.get("Annual volatility", np.nan),
                    "calmar": stats.get("Calmar ratio", np.nan),
                }
            )
            print(f"  done in {time.time() - t0:.0f}s  "
                  f"total_return={summary_rows[-1]['total_return']:.2%}  "
                  f"sharpe={summary_rows[-1]['sharpe']:.3f}")

    summary = pd.DataFrame(summary_rows).sort_values("sharpe", ascending=False)
    summary_path = f"{SWEEP_DIR}/ensemble_sweep_summary.csv"
    summary.to_csv(summary_path, index=False)
    print("\n=== Sweep summary (sorted by Sharpe) ===")
    print(summary.round(4).to_string(index=False))
    print(f"\nSummary saved to {summary_path}")

    # Post-hoc fee sensitivity from the saved actions files.
    print("\n=== Fee sensitivity (final portfolio value, $) ===")
    pivot = test.pivot(index="date", columns="tic", values="close").sort_index()
    pivot.index = pivot.index.astype(str)
    fee_rows = []
    configs = {f"k{k}b{block}": f"{SWEEP_DIR}/actions_ensemble_pa_k{k}b{block}.csv"
               for k in args.k for block in args.block}
    configs.update({name: f"{rq2_config.RESULTS_ROOT}/{path}"
                    for name, path in REFERENCE_ACTIONS.items()})
    for name, path in configs.items():
        for fee in FEE_LEVELS:
            fee_rows.append({"strategy": name, "fee_pct": fee,
                             "final_value": evolve(path, pivot, fee)})
    fee_table = pd.DataFrame(fee_rows).pivot(index="strategy", columns="fee_pct",
                                             values="final_value")
    fee_table = fee_table.reindex(
        [f"k{k}b{block}" for k in args.k for block in args.block]
        + list(REFERENCE_ACTIONS.keys())
    )
    fee_table.index.name = "strategy"
    print(fee_table.round(0).to_string())
    fee_path = f"{SWEEP_DIR}/ensemble_fee_sensitivity.csv"
    fee_table.to_csv(fee_path)
    print(f"\nFee sensitivity saved to {fee_path}")


if __name__ == "__main__":
    main()
