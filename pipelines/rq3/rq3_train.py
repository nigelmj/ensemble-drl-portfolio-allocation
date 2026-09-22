"""
RQ3 — Train the SAC / A2C members.

SAC and A2C members are trained in RQ3 (PPO members reuse the RQ2
models). Train sac_log_return and a2c_log_return first, then the dsr pair,
then the dsr_drawdown pair, each at 1M steps. For the DSR-based members the
per-algo scaling is taken from results/rq3/calibration_dsr.json when it
exists (see rq3_calibrate.py); otherwise the fallback is used.

Usage:
    python -m pipelines.rq3.rq3_train --algo sac --reward-type log_return
    python -m pipelines.rq3.rq3_train --algo a2c --reward-type log_return
    python -m pipelines.rq3.rq3_train --algo sac --reward-type dsr
    python -m pipelines.rq3.rq3_train --reward-type dsr_drawdown --tag smoke --total-timesteps 60000

Output:
    rq_trained_models/rq3/{algo}_{reward_type}{_tag}.zip
"""
from __future__ import annotations

import argparse
import os
import time

import pandas as pd
from stable_baselines3.common.logger import configure

from pipelines.rq3 import rq3_config
from rl_portfolio.agents.portfolio_allocation_agent import (
    PortfolioAllocationDRLAgent,
)
from rl_portfolio.config import RESULTS_DIR, TRAINED_MODEL_DIR
from rl_portfolio.utils.io import check_and_make_directories
from rl_portfolio.envs.portfolio_allocation_env import (
    PortfolioAllocationEnv,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["sac", "a2c"], default="sac",
                        help="Which algorithm to train (default: sac).")
    parser.add_argument("--reward-type", choices=rq3_config.REWARD_TYPES,
                        required=True,
                        help="Which reward variant to train.")
    parser.add_argument("--total-timesteps", type=int,
                        default=rq3_config.TOTAL_TIMESTEPS,
                        help=f"Training budget (default: {rq3_config.TOTAL_TIMESTEPS:,}).")
    parser.add_argument("--tag", default="",
                        help="Optional suffix for save paths (e.g. 'smoke').")
    parser.add_argument("--calibration", default=None,
                        help="Path to the calibration JSON to use for DSR "
                             "scaling (default: results/rq3/calibration_dsr.json).")
    return parser.parse_args()


def main():
    args = parse_args()
    suffix = f"_{args.tag}" if args.tag else ""
    algo = args.algo
    name = f"{algo}_{args.reward_type}{suffix}"
    save_path = f"{rq3_config.MODEL_DIR}/{name}"
    tb_path = f"{rq3_config.RESULTS_ROOT}/{name}_tb"

    check_and_make_directories([TRAINED_MODEL_DIR, RESULTS_DIR,
                                rq3_config.MODEL_DIR, rq3_config.RESULTS_ROOT])

    train = pd.read_pickle(rq3_config.DATA_TRAIN)
    print(f"Train: {len(train)} rows, {train.date.nunique()} days, "
          f"{train.tic.nunique()} tickers")

    cal_path = args.calibration or rq3_config.CALIBRATION_PATH
    calibration = rq3_config.load_calibration(cal_path) if os.path.exists(cal_path) else {}
    if calibration:
        print(f"Using calibrated DSR scaling from {cal_path}")
    else:
        print("No calibration file found - using fallback DSR scaling.")

    env_kwargs = rq3_config.build_env_kwargs(algo, args.reward_type, calibration)
    print(f"\nTraining {name}  (algo={algo}, reward={args.reward_type}, "
          f"timesteps={args.total_timesteps:,})")
    print(f"env_kwargs: {env_kwargs}")

    start = time.time()
    set_seed(rq3_config.SEED)
    env = PortfolioAllocationEnv(df=train, **env_kwargs)
    env_train, _ = env.get_sb_env()

    agent = PortfolioAllocationDRLAgent(env=env_train)
    model = agent.get_model(
        algo,
        model_kwargs=rq3_config.MODEL_KWARGS[algo],
        policy_kwargs=rq3_config.ALGO_POLICY_KWARGS[algo],
        seed=rq3_config.SEED,
    )

    new_logger = configure(tb_path, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    trained = agent.train_model(
        model=model,
        tb_log_name=name,
        total_timesteps=args.total_timesteps,
    )
    trained.save(save_path)
    print(f"Saved to {save_path}.zip")

    elapsed = (time.time() - start) / 60
    print(f"\n{algo.upper()} {args.reward_type} training complete in {elapsed:.1f} minutes.")


if __name__ == "__main__":
    main()
