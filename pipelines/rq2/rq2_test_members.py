"""
RQ2 — Test the 3 reward-variant models individually on the test split.

Usage:
    python -m pipelines.rq2.rq2_train
    python -m pipelines.rq2.rq2_test_members

Output:
    results/0rq2/account_value_ppo_pa_{logreturn,dsr,dsr_drawdown}{_tag}.csv
    results/0rq2/actions_ppo_pa_{logreturn,dsr,dsr_drawdown}{_tag}.csv
    results/0rq2/portfolio_stats_ppo_pa_{logreturn,dsr,dsr_drawdown}{_tag}.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from pipelines.rq2 import rq2_config
from rl_portfolio.agents.portfolio_allocation_agent import (
    PortfolioAllocationDRLAgent,
)
from rl_portfolio.config import RESULTS_DIR
from rl_portfolio.utils.io import check_and_make_directories
from rl_portfolio.envs.portfolio_allocation_env import PortfolioAllocationEnv

MODELS = {
    "log_return": f"{rq2_config.MODEL_DIR}/agent_ppo_pa_log_return",
    "dsr": f"{rq2_config.MODEL_DIR}/agent_ppo_pa_dsr",
    "dsr_drawdown": f"{rq2_config.MODEL_DIR}/agent_ppo_pa_dsr_drawdown",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tag",
        default="",
        help="Optional suffix matching the trained-model tag.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    check_and_make_directories([RESULTS_DIR, rq2_config.RESULTS_ROOT])

    test = pd.read_pickle(rq2_config.DATA_TEST)

    for name, path in MODELS.items():
        suffix = f"_{args.tag}" if args.tag else ""
        print("\n" + "=" * 60)
        print(f"Testing model '{name}' from {path}")
        print("=" * 60)

        model = PPO.load(path)
        env_kwargs = rq2_config.build_env_kwargs(name)
        e_test_gym = PortfolioAllocationEnv(df=test, **env_kwargs)

        account_memory, actions_memory, stats = (
            PortfolioAllocationDRLAgent.DRL_prediction(
                model=model, environment=e_test_gym, deterministic=True
            )
        )

        account_memory.to_csv(
            f"{rq2_config.RESULTS_ROOT}/account_value_ppo_pa_{name}{suffix}.csv",
            index=False,
        )
        actions_memory.to_csv(
            f"{rq2_config.RESULTS_ROOT}/actions_ppo_pa_{name}{suffix}.csv"
        )
        pd.DataFrame([stats]).to_csv(
            f"{rq2_config.RESULTS_ROOT}/portfolio_stats_ppo_pa_{name}{suffix}.csv",
            index=False,
        )

        initial_value = env_kwargs["initial_amount"]
        final_value = account_memory["portfolio_value"].iloc[-1]
        total_return = final_value / initial_value - 1

        daily_return = account_memory["portfolio_value"].pct_change().dropna()
        if daily_return.std() != 0:
            sharpe = (252**0.5) * daily_return.mean() / daily_return.std()
        else:
            sharpe = np.inf if daily_return.mean() > 0 else 0.0

        print(f"Initial portfolio value: {initial_value:.2f}")
        print(f"Final portfolio value:   {final_value:.2f}")
        print(f"Total return:            {total_return:.2%}")
        print(f"Sharpe ratio:            {sharpe:.3f}")
        print(f"Average turnover/step:   {stats['average_turnover']:.4f}")


if __name__ == "__main__":
    main()
