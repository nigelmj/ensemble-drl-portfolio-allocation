"""
RQ2 — Train reward-variant PPO models.

Trains one or more PPO agents on the shared train split, one per reward_type
(log_return, dsr, dsr_drawdown), with ent_coef=0.001, n_steps=2048,
lr=0.00025, batch=64, log_std_init=-2.0, seed=42.

RQ2 trains ALL THREE members (the log_return baseline is NOT reused)
so every RQ2/RQ3 comparison rests on identically-produced models.

Usage:
    python -m pipelines.rq2.rq2_train
    python -m pipelines.rq2.rq2_train --reward-type dsr_drawdown
    python -m pipelines.rq2.rq2_train --tag smoke --total-timesteps 60000

Output:
    rq_trained_models/rq2/agent_ppo_pa_{log_return,dsr,dsr_drawdown}{_tag}.zip
    results/rq2/ppo_pa_{reward_type}{_tag}/... (TensorBoard + CSV logs)
"""
from __future__ import annotations

import argparse

import pandas as pd
from stable_baselines3.common.logger import configure

from pipelines.rq2 import rq2_config
from rl_portfolio.agents.portfolio_allocation_agent import (
    PortfolioAllocationDRLAgent,
)
from rl_portfolio.config import RESULTS_DIR, TRAINED_MODEL_DIR
from rl_portfolio.utils.io import check_and_make_directories
from rl_portfolio.envs.portfolio_allocation_env import PortfolioAllocationEnv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reward-type",
        choices=rq2_config.REWARD_TYPES + ["all"],
        default="all",
        help="Which reward variant(s) to train (default: all).",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=rq2_config.TOTAL_TIMESTEPS,
        help=f"Training budget per model (default: {rq2_config.TOTAL_TIMESTEPS:,}).",
    )
    parser.add_argument(
        "--reward-scaling-dsr",
        type=float,
        default=rq2_config.REWARD_SCALING_DSR,
        help=f"Scaling for the DSR / DSR-drawdown rewards "
             f"(default: {rq2_config.REWARD_SCALING_DSR}).",
    )
    parser.add_argument(
        "--eta-dd",
        type=float,
        default=rq2_config.DSR_ETA_DD,
        help=f"Drawdown penalty weight for the dsr_drawdown reward "
             f"(default: {rq2_config.DSR_ETA_DD}).",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="Optional suffix for save paths (e.g. 'smoke').",
    )
    return parser.parse_args()


def train_one(reward_type: str, args) -> None:
    suffix = f"_{args.tag}" if args.tag else ""
    save_path = f"{rq2_config.MODEL_DIR}/agent_ppo_pa_{reward_type}{suffix}"
    tb_path = f"{rq2_config.RESULTS_ROOT}/ppo_pa_{reward_type}{suffix}"

    print("\n" + "=" * 60)
    print(f"Training reward_type='{reward_type}'  "
          f"timesteps={args.total_timesteps:,}")
    print(f"Save path: {save_path}.zip")
    print("=" * 60)

    train = pd.read_pickle(rq2_config.DATA_TRAIN)
    env_kwargs = rq2_config.build_env_kwargs(reward_type)
    env_kwargs["reward_scaling_dsr"] = args.reward_scaling_dsr
    if reward_type == "dsr_drawdown":
        env_kwargs["dsr_eta_dd"] = args.eta_dd

    e_train_gym = PortfolioAllocationEnv(df=train, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()

    agent = PortfolioAllocationDRLAgent(env=env_train)
    model = agent.get_model(
        "ppo",
        model_kwargs=rq2_config.PPO_PARAMS,
        policy_kwargs=rq2_config.POLICY_KWARGS,
        seed=rq2_config.SEED,
    )

    new_logger = configure(tb_path, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    trained = agent.train_model(
        model=model,
        tb_log_name=f"ppo_{reward_type}",
        total_timesteps=args.total_timesteps,
    )

    trained.save(save_path)
    print(f"Saved to {save_path}.zip")


def main():
    args = parse_args()
    check_and_make_directories([TRAINED_MODEL_DIR, RESULTS_DIR,
                                rq2_config.MODEL_DIR, rq2_config.RESULTS_ROOT])

    reward_types = (
        rq2_config.REWARD_TYPES if args.reward_type == "all" else [args.reward_type]
    )
    for rt in reward_types:
        train_one(rt, args)

    print("\nRQ2 training complete.")


if __name__ == "__main__":
    main()
