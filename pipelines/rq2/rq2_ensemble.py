"""
RQ2 — Run the performance-weighted ensemble on the test split.

Loads the 3 reward-variant PPO models, runs the ensemble agent
and saves the blended portfolio account value, the blended target actions, and
the per-day smoothed blend weights.

Usage:
    python -m pipelines.rq2.rq2_ensemble                      # soft k10
    python -m pipelines.rq2.rq2_ensemble --mode hard --tag hard
    python -m pipelines.rq2.rq2_ensemble --mode block --k 10 --block-days 20 --tag k10b20

Output:
    results/0rq2/account_value_ensemble_pa{_tag}.csv
    results/0rq2/actions_ensemble_pa{_tag}.csv
    results/0rq2/ensemble_agent_weights{_tag}.csv
"""
from __future__ import annotations

import argparse

import pandas as pd

from pipelines.rq2 import rq2_config
from rl_portfolio.agents.portfolio_allocation_ensemble_agent import (
    PortfolioAllocationEnsembleAgent,
)
from rl_portfolio.config import RESULTS_DIR
from rl_portfolio.utils.io import check_and_make_directories


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        action="append",
        dest="model_paths",
        help="Path to a trained model (3 required). Repeatable. "
             "Defaults to the RQ2 log_return/dsr/dsr_drawdown set.",
    )
    parser.add_argument("--k", type=int, default=10,
                        help="Trailing window length in days (default: 10).")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Softmax sharpness (try 1, 5, 10, 20).")
    parser.add_argument("--alpha", type=float, default=0.3,
                        help="Weight EMA smoothing (try 0.2-0.5).")
    parser.add_argument("--mode", choices=["soft", "hard", "block"],
                        default="soft",
                        help="soft: softmax+EMA blend; hard: one-hot on the "
                             "best agent every day; block: one-hot held for "
                             "--block-days.")
    parser.add_argument("--block-days", type=int, default=20,
                        help="Tenure in days for 'block' mode.")
    parser.add_argument("--tag", default="",
                        help="Optional suffix for output files.")
    return parser.parse_args()


def main():
    args = parse_args()
    check_and_make_directories([RESULTS_DIR, rq2_config.RESULTS_ROOT])

    model_paths = args.model_paths or rq2_config.DEFAULT_MODEL_PATHS
    suffix = f"_{args.tag}" if args.tag else ""

    test = pd.read_pickle(rq2_config.DATA_TEST)

    print("Loading models:")
    for p in model_paths:
        print(f"  {p}")

    agent = PortfolioAllocationEnsembleAgent(
        model_paths=model_paths,
        df=test,
        env_kwargs=rq2_config.ENV_KWARGS,
        k=args.k,
        temperature=args.temperature,
        alpha=args.alpha,
        mode=args.mode,
        block_days=args.block_days,
    )

    print(f"\nRunning ensemble (mode={args.mode}, k={args.k}, "
          f"temperature={args.temperature}, alpha={args.alpha}, "
          f"block_days={args.block_days}) "
          f"over {agent.episode_length} steps...")

    account_df, actions_df, weights_df = agent.run()

    account_path = f"{rq2_config.RESULTS_ROOT}/account_value_ensemble_pa{suffix}.csv"
    actions_path = f"{rq2_config.RESULTS_ROOT}/actions_ensemble_pa{suffix}.csv"
    weights_path = f"{rq2_config.RESULTS_ROOT}/ensemble_agent_weights{suffix}.csv"

    account_df.to_csv(account_path, index=False)
    actions_df.to_csv(actions_path)
    weights_df.to_csv(weights_path, index=False)

    initial_value = rq2_config.ENV_KWARGS["initial_amount"]
    final_value = account_df["portfolio_value"].iloc[-1]
    total_return = final_value / initial_value - 1

    daily_return = account_df["portfolio_value"].pct_change().dropna()
    sharpe = (
        (252**0.5) * daily_return.mean() / daily_return.std()
        if daily_return.std() != 0 else 0.0
    )

    print(f"\nInitial portfolio value: {initial_value:.2f}")
    print(f"Final portfolio value:   {final_value:.2f}")
    print(f"Total return:            {total_return:.2%}")
    print(f"Sharpe ratio:            {sharpe:.3f}")

    mean_w = weights_df[["w_log_return", "w_dsr", "w_dsr_drawdown"]].mean()
    print(f"\nMean blend weights:\n{mean_w.round(3).to_string()}")
    print(f"\nResults saved to {rq2_config.RESULTS_ROOT}")


if __name__ == "__main__":
    main()
