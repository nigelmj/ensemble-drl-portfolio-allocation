#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQ1 — Baseline ensemble average testing.
Allocation = simple average of ensemble PPO agents.
No safe allocation blending, no confidence mapping.
"""
from __future__ import annotations

import json
import os
import numpy as np
import pandas as pd

from pipelines.rq1.rq1_alloc import EnsemblePPOAllocationAgent
from pipelines.rq1 import rq1_config
from rl_portfolio.config import RESULTS_DIR
from rl_portfolio.utils.io import check_and_make_directories
from rl_portfolio.envs.portfolio_allocation_env import PortfolioAllocationEnv


def main():
    check_and_make_directories([RESULTS_DIR, rq1_config.RESULTS_ROOT])

    test = pd.read_pickle(rq1_config.DATA_TEST)
    print(f"Test: {len(test)} rows, {test.date.nunique()} days, "
          f"{test.tic.nunique()} tickers")

    # ---- Calibration ----
    cal = json.load(
        open(f"{rq1_config.RESULTS_ROOT}/calibration_all.json")
    )
    d_ref = cal["calibrations"][rq1_config.D_REF_METHOD]["d_ref"]
    disagreement_stats = cal["disagreement_stats"]
    print(f"\nD_ref ({rq1_config.D_REF_METHOD}): {d_ref:.6f}")

    # ---- Baseline: ensemble average, no safe, no confidence ----
    label = "baseline_ensemble_average"
    out_dir = f"{rq1_config.RESULTS_ROOT}/{label}"
    check_and_make_directories([out_dir])
    print("\n" + "=" * 60)
    print(f"Running baseline: ensemble average (no safe, no confidence)")
    print("=" * 60)

    agent = EnsemblePPOAllocationAgent(
        n_ensemble=rq1_config.N_ENSEMBLE,
        base_seed=rq1_config.BASE_SEED,
        confidence_mapping="power",       # unused, but required by __init__
        safe_strategy="previous",         # unused, but required by __init__
        d_ref_method=rq1_config.D_REF_METHOD,
        power_p=rq1_config.POWER_P,
        sigmoid_k=rq1_config.SIGMOID_K,
    )
    agent.load_models(f"{rq1_config.MODEL_DIR}")
    agent.d_ref = d_ref
    agent.disagreement_stats = disagreement_stats
    agent.portfolio_size = test["tic"].nunique()

    env = PortfolioAllocationEnv(df=test, **rq1_config.ENV_KWARGS)
    test_env, test_obs = env.get_sb_env()
    test_env.reset()

    max_steps = env.episode_length - 1

    account_records = []
    action_records = []

    for step in range(max_steps + 1):
        actions = []
        for model in agent.models:
            action, _ = model.predict(test_obs, deterministic=True)
            actions.append(action.flatten())

        actions_arr = np.array(actions)
        alloc_ensemble = actions_arr.mean(axis=0)   # <-- ensemble average only

        # Use ensemble mean directly; no safe blending, no confidence
        alloc_final = alloc_ensemble.copy()

        date = env._sorted_times[env._time_index] if env._time_index < len(env._sorted_times) else None

        account_records.append({
            "date": str(date) if date is not None else None,
            "portfolio_value": env.portfolio_value,
        })

        action_record = {
            "date": str(date) if date is not None else None,
        }
        for k in range(len(alloc_ensemble)):
            action_record[f"ensemble_alloc_{k}"] = alloc_ensemble[k]
        # No safe_alloc, no final_alloc, no confidence, no disagreement

        action_records.append(action_record)

        test_obs, rewards, dones, info = test_env.step(
            alloc_final.reshape(1, -1).astype(np.float32)
        )

        if dones[0]:
            break

    account_df = pd.DataFrame(account_records)
    actions_df = pd.DataFrame(action_records)

    account_df.to_csv(os.path.join(out_dir, "test_account.csv"), index=False)
    actions_df.to_csv(os.path.join(out_dir, "test_actions.csv"), index=False)

    print(f"Saved {out_dir}/test_account.csv ({len(account_df)} rows)")
    print(f"Saved {out_dir}/test_actions.csv ({len(actions_df)} rows)")

    # ---- Compute metrics ----
    vals = account_df["portfolio_value"].values
    total_return = vals[-1] / vals[0] - 1
    daily_returns = np.diff(vals) / vals[:-1]
    if np.std(daily_returns) > 0:
        sharpe = np.sqrt(252) * np.mean(daily_returns) / np.std(daily_returns)
    else:
        sharpe = np.inf if np.mean(daily_returns) > 0 else 0.0

    peak = np.maximum.accumulate(vals)
    drawdowns = (vals - peak) / peak
    max_drawdown = float(np.min(drawdowns))

    mean_confidence = np.nan
    mean_disagreement = np.nan

    metrics = {
        "label": label,
        "final_value": float(vals[-1]),
        "total_return": float(total_return),
        "sharpe": float(sharpe),
        "max_drawdown": max_drawdown,
        "avg_daily_return": float(np.mean(daily_returns)),
        "volatility": float(np.std(daily_returns)),
        "mean_confidence": mean_confidence,
        "mean_disagreement": mean_disagreement,
    }

    print(f"  Sharpe={metrics['sharpe']:.3f}  "
          f"Return={metrics['total_return']:.2%}  "
          f"MaxDD={metrics['max_drawdown']:.2%}  "
          f"Conf=N/A  Disag=N/A")

    # ---- Append to test_summary ----
    summary_path = f"{rq1_config.RESULTS_ROOT}/test_summary.csv"
    try:
        summary_existing = pd.read_csv(summary_path)
    except FileNotFoundError:
        summary_existing = pd.DataFrame()

    metrics_series = pd.Series(metrics)
    if not summary_existing.empty:
        summary_new = pd.concat([summary_existing, metrics_series.to_frame().T], ignore_index=True)
    else:
        summary_new = metrics_series.to_frame().T

    summary_new.to_csv(summary_path, index=False)
    print(f"\nSummary updated at {summary_path}")


if __name__ == "__main__":
    main()
