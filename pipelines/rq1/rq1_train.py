"""
RQ1 — Train the inter-seed PPO ensemble + single PPO baseline.

Trains N_ENSEMBLE PPO agents with consecutive seeds on the FULL shared train
split (2010-2021), plus a single PPO baseline. The default ensemble budget is
1M per member (5M total); the default single baseline is now 5M for a
compute-matched control. The 1M single (ppo_single.zip) is kept
alongside the new ppo_single_5m.zip so both can be compared / reverted.

Usage:
    python -m pipelines.rq1.rq1_train
    python -m pipelines.rq1.rq1_train --skip-ensemble --single-timesteps 5000000
    python -m pipelines.rq1.rq1_train --tag smoke --total-timesteps 60000 --single-timesteps 60000

Output:
    rq_trained_models/rq1/ppo_0.zip .. ppo_{N-1}.zip
    rq_trained_models/rq1/ppo_single.zip          (1M)
    rq_trained_models/rq1/ppo_single_5m.zip       (5M, compute-matched)
    results/0rq1/tb/...
"""
from __future__ import annotations

import argparse
import time

import pandas as pd
from stable_baselines3.common.logger import configure

from pipelines.rq1.rq1_alloc import EnsemblePPOAllocationAgent
from pipelines.rq1 import rq1_config
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
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=rq1_config.TOTAL_TIMESTEPS,
        help=f"Training budget per ensemble member (default: {rq1_config.TOTAL_TIMESTEPS:,}).",
    )
    parser.add_argument(
        "--single-timesteps",
        type=int,
        default=rq1_config.TOTAL_TIMESTEPS_SINGLE_5M,
        help=f"Training budget for the single PPO baseline (default: {rq1_config.TOTAL_TIMESTEPS_SINGLE_5M:,}). "
             f"Use {rq1_config.TOTAL_TIMESTEPS:,} to reproduce the legacy 1M single.",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="Optional suffix for save paths (e.g. 'smoke').",
    )
    parser.add_argument(
        "--skip-ensemble",
        action="store_true",
        help="Skip ensemble training (use when only the 5M single is needed).",
    )
    parser.add_argument(
        "--skip-single",
        action="store_true",
        help="Skip single-baseline training.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    suffix = f"_{args.tag}" if args.tag else ""
    model_dir = f"{rq1_config.MODEL_DIR}{suffix}"
    tb_root = f"{rq1_config.RESULTS_ROOT}/tb{suffix}"

    check_and_make_directories([TRAINED_MODEL_DIR, RESULTS_DIR, model_dir, tb_root])

    train = pd.read_pickle(rq1_config.DATA_TRAIN)
    print(f"Train: {len(train)} rows, {train.date.nunique()} days, "
          f"{train.tic.nunique()} tickers")

    cfg = rq1_config

    # ---- 1. Train the N-seed ensemble ----
    if not args.skip_ensemble:
        agent = EnsemblePPOAllocationAgent(
            n_ensemble=cfg.N_ENSEMBLE,
            base_seed=cfg.BASE_SEED,
            confidence_mapping=cfg.CONFIDENCE_MAPPINGS[1],  # power (unused at train time)
            safe_strategy=cfg.SAFE_STRATEGIES[0],           # previous
            d_ref_method=cfg.D_REF_METHOD,
            power_p=cfg.POWER_P,
            sigmoid_k=cfg.SIGMOID_K,
        )

        start = time.time()
        agent.train_ensemble(
            df=train,
            env_kwargs=cfg.ENV_KWARGS,
            model_kwargs=cfg.PPO_KWARGS,
            total_timesteps=args.total_timesteps,
            save_dir=model_dir,
            policy_kwargs=cfg.POLICY_KWARGS,
        )
        elapsed = (time.time() - start) / 60
        print(f"\nEnsemble training completed in {elapsed:.1f} minutes")
    else:
        print("\n[skip] Ensemble training (--skip-ensemble)")

    # ---- 2. Train single PPO baseline ----
    if not args.skip_single:
        print("\n" + "=" * 60)
        # Distinguish legacy 1M vs compute-matched 5M via filename suffix.
        # Keeps ppo_single.zip untouched when training the 5M control.
        if args.single_timesteps == cfg.TOTAL_TIMESTEPS_SINGLE_5M:
            single_name = "ppo_single_5m"
        elif args.single_timesteps == cfg.TOTAL_TIMESTEPS:
            single_name = "ppo_single"
        else:
            single_name = f"ppo_single_{args.single_timesteps}"
        print(f"Training single PPO baseline '{single_name}' "
              f"({args.single_timesteps:,} timesteps, seed {cfg.BASE_SEED})...")
        print("=" * 60)

        set_seed(cfg.BASE_SEED)
        env = PortfolioAllocationEnv(df=train, **cfg.ENV_KWARGS)
        env_train, _ = env.get_sb_env()

        agent_single = PortfolioAllocationDRLAgent(env=env_train)
        model_single = agent_single.get_model(
            "ppo",
            model_kwargs=cfg.PPO_KWARGS,
            policy_kwargs=cfg.POLICY_KWARGS,
            seed=cfg.BASE_SEED,
        )
        new_logger = configure(f"{tb_root}/{single_name}", ["stdout", "csv", "tensorboard"])
        model_single.set_logger(new_logger)
        single_model = PortfolioAllocationDRLAgent.train_model(
            model=model_single,
            tb_log_name=single_name,
            total_timesteps=args.single_timesteps,
        )
        single_path = f"{model_dir}/{single_name}"
        single_model.save(single_path)
        print(f"Saved single PPO to {single_path}.zip")
    else:
        print("\n[skip] Single baseline training (--skip-single)")
        single_name = None

    print("\n" + "=" * 60)
    print(f"RQ1 training summary")
    print("=" * 60)
    if not args.skip_ensemble:
        print(f"  Ensemble models: {cfg.N_ENSEMBLE} members (seeds "
              f"{cfg.BASE_SEED}..{cfg.BASE_SEED + cfg.N_ENSEMBLE - 1}) @ {args.total_timesteps:,} each")
    else:
        print(f"  Ensemble models: skipped")
    if not args.skip_single:
        print(f"  Single baseline: 1 model (seed {cfg.BASE_SEED}) @ {args.single_timesteps:,} "
              f"-> {single_name}.zip")
    else:
        print(f"  Single baseline: skipped")
    print(f"  Models saved to: {model_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
