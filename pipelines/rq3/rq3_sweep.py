"""
RQ3 — sweep trailing window (k) x block tenure (block_days) per version (v1..v6).

Mirrors RQ2 sweep (rq2_sweep.py) but for cross-model versions.
Runs the block-mode ensemble over the 12-run grid for each of the 6
versions (6 * 12 = 72 runs) and writes per-version summaries plus
per-config account/actions/weights files into results/0rq3/sweeps/.

Usage:
    python -m pipelines.rq3.rq3_sweep
    python -m pipelines.rq3.rq3_sweep --version v4
    python -m pipelines.rq3.rq3_sweep --k 10 20 --block 20 40

Output:
    results/0rq3/sweeps/sweep_summary_v{N}.csv            (6 files, 12 rows each)
    results/0rq3/sweeps/account_value_{version}_k{k}b{block}.csv
    results/0rq3/sweeps/actions_{version}_k{k}b{block}.csv
    results/0rq3/sweeps/weights_{version}_k{k}b{block}.csv
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from pipelines.rq3 import rq3_config
from rl_portfolio.agents.cross_model_ensemble_agent import CrossModelPerfWeightedAgent
from rl_portfolio.config import RESULTS_DIR
from rl_portfolio.utils.io import check_and_make_directories
from rl_portfolio.utils.metrics import backtest_stats

SWEEP_DIR = f"{rq3_config.RESULTS_ROOT}/sweeps"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--k", nargs="+", type=int, default=[10, 20, 40],
                   help="Trailing windows to sweep (days).")
    p.add_argument("--block", nargs="+", type=int, default=[5, 10, 20, 40],
                   help="Block tenures to sweep (days).")
    p.add_argument("--version", default=None,
                   help="One of v1..v6 (default: all six versions).")
    return p.parse_args()


def main():
    args = parse_args()
    versions = [args.version] if args.version else list(rq3_config.VERSIONS)
    for v in versions:
        if v not in rq3_config.VERSIONS:
            raise SystemExit(f"Unknown version {v}. Choose from {list(rq3_config.VERSIONS)}")

    check_and_make_directories([RESULTS_DIR, SWEEP_DIR])
    test = pd.read_pickle(rq3_config.DATA_TEST)
    calibration = rq3_config.load_calibration()

    print(f"Test: {len(test)} rows, {test.date.nunique()} days, {test.tic.nunique()} tickers")
    print(f"Versions: {versions}")
    print(f"Grid: k={args.k} x block={args.block} -> {len(args.k)*len(args.block)} configs per version")
    print(f"Calibrations loaded: {list(calibration.keys()) if calibration else 'none (fallback)'}")

    for version in versions:
        print("\n" + "=" * 60)
        print(f"=== Version {version}: " + " ".join(f"{m['algo']}={m['reward_type']}" for m in rq3_config.VERSIONS[version]))
        print("=" * 60)

        member_specs = rq3_config.build_member_specs(rq3_config.VERSIONS[version], calibration)
        print("  Members:")
        for m in member_specs:
            exists = "OK" if __import__("os").path.exists(m["path"] + ".zip") else "MISSING"
            print(f"    {m['algo']:<5} {m['reward_type']:<12} {m['path']} [{exists}]")

        summary_rows = []
        for k in args.k:
            for block in args.block:
                tag = f"k{k}b{block}"
                label = f"{version}_{tag}"
                t0 = time.time()
                print(f"\n  --- {version} k={k}, block={block} ({tag}) ---")
                agent = CrossModelPerfWeightedAgent(
                    member_specs=member_specs,
                    mode="block",
                    k=k,
                    block_days=block,
                )
                account_df, actions_df, weights_df = agent.run(df=test, base_env_kwargs=rq3_config.BASE_ENV_KWARGS)

                account_df.to_csv(f"{SWEEP_DIR}/account_value_{version}_{tag}.csv", index=False)
                actions_df.to_csv(f"{SWEEP_DIR}/actions_{version}_{tag}.csv")
                weights_df.to_csv(f"{SWEEP_DIR}/weights_{version}_{tag}.csv", index=False)

                stats = backtest_stats(
                    account_df.rename(columns={"portfolio_value": "account_value"}),
                    value_col_name="account_value",
                )
                summary_rows.append({
                    "k": k,
                    "block_days": block,
                    "tag": tag,
                    "final_value": float(account_df["portfolio_value"].iloc[-1]),
                    "total_return": float(stats.get("Cumulative returns", np.nan)),
                    "sharpe": float(stats.get("Sharpe ratio", np.nan)),
                    "max_drawdown": float(stats.get("Max drawdown", np.nan)),
                    "annual_vol": float(stats.get("Annual volatility", np.nan)),
                    "calmar": float(stats.get("Calmar ratio", np.nan)),
                })
                print(f"    done in {time.time()-t0:.0f}s  total_return={summary_rows[-1]['total_return']:.2%}  sharpe={summary_rows[-1]['sharpe']:.3f}")

        summary = pd.DataFrame(summary_rows).sort_values("sharpe", ascending=False)
        summary_path = f"{SWEEP_DIR}/sweep_summary_{version}.csv"
        summary.to_csv(summary_path, index=False)
        print(f"\n  Summary for {version} (sorted by Sharpe):")
        print(summary.round(4).to_string(index=False))
        print(f"  Saved to {summary_path}")

    print("\nAll sweeps done. Per-version files in", SWEEP_DIR)


if __name__ == "__main__":
    main()
