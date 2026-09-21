"""
RQ1 — Validate & calibrate D_ref on the last three years of train (2019-2021).

Runs the trained ensemble on a validation window sliced from the shared train
split and calibrates D_ref for each method. The test window (2022-2026) is
never touched here.

Usage:
    python -m pipelines.rq1.rq1_train
    python -m pipelines.rq1.rq1_validate

Output:
    results/0rq1/calibration_all.json
    results/0rq1/validation_disagreements.csv
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from pipelines.rq1.rq1_alloc import EnsemblePPOAllocationAgent, calibrate_d_ref, compute_disagreement, normalize_actions
from pipelines.rq1 import rq1_config
from rl_portfolio.config import RESULTS_DIR
from rl_portfolio.utils.io import check_and_make_directories
from rl_portfolio.envs.portfolio_allocation_env import PortfolioAllocationEnv


def main():
    check_and_make_directories([RESULTS_DIR, rq1_config.RESULTS_ROOT])

    train = pd.read_pickle(rq1_config.DATA_TRAIN)
    val = train[
        (train["date"] >= rq1_config.VAL_START) &
        (train["date"] <= rq1_config.VAL_END)
    ].reset_index(drop=True)
    print(f"Val window {rq1_config.VAL_START}..{rq1_config.VAL_END}: "
          f"{len(val)} rows, {val.date.nunique()} days, {val.tic.nunique()} tickers")

    cfg = rq1_config
    agent = EnsemblePPOAllocationAgent(
        n_ensemble=cfg.N_ENSEMBLE,
        base_seed=cfg.BASE_SEED,
        confidence_mapping=cfg.CONFIDENCE_MAPPINGS[1],
        safe_strategy=cfg.SAFE_STRATEGIES[0],
        d_ref_method=cfg.D_REF_METHOD,
        power_p=cfg.POWER_P,
        sigmoid_k=cfg.SIGMOID_K,
    )
    agent.load_models(cfg.MODEL_DIR)
    print(f"\nLoaded {len(agent.models)} ensemble models from {cfg.MODEL_DIR}")

    env = PortfolioAllocationEnv(df=val, **cfg.ENV_KWARGS)
    test_env, test_obs = env.get_sb_env()
    test_env.reset()

    max_steps = env.episode_length - 1
    disagreements = []

    for step in range(max_steps + 1):
        actions = []
        for model in agent.models:
            action, _ = model.predict(test_obs, deterministic=True)
            actions.append(action.flatten())
        actions_arr = np.array(actions)
        normed = np.array([normalize_actions(a) for a in actions_arr])
        ensemble_mean = normed.mean(axis=0)
        disagreements.append(compute_disagreement(normed, ensemble_mean))
        test_obs, _, dones, _ = test_env.step(
            ensemble_mean.reshape(1, -1).astype(np.float32)
        )
        if dones[0]:
            break

    disagreements = np.array(disagreements)
    print(f"\nCollected {len(disagreements)} disagreement values")
    print(f"  Mean:   {disagreements.mean():.6f}")
    print(f"  Median: {np.median(disagreements):.6f}")
    print(f"  Std:    {disagreements.std():.6f}")
    print(f"  Min:    {disagreements.min():.6f}")
    print(f"  Max:    {disagreements.max():.6f}")

    calibrations = {}
    print(f"\n{'Method':<10} {'D_ref':>12} {'Mean D':>12} {'Ratio':>10}")
    print("-" * 46)
    for method in cfg.D_REF_METHODS:
        d_ref = calibrate_d_ref(disagreements, method=method)
        calibrations[method] = {
            "d_ref": float(d_ref),
            "method": method,
            "mean_disagreement": float(disagreements.mean()),
            "median_disagreement": float(np.median(disagreements)),
            "max_disagreement": float(disagreements.max()),
            "ratio_mean_to_d_ref": float(disagreements.mean() / d_ref if d_ref > 0 else 0),
        }
        print(f"{method:<10} {d_ref:>12.6f} {disagreements.mean():>12.6f} "
              f"{disagreements.mean() / d_ref if d_ref > 0 else 0:>10.4f}")

    disagreement_stats = {
        "d_mean": float(disagreements.mean()),
        "d_std": float(disagreements.std()),
        "d_min": float(disagreements.min()),
        "d_max": float(disagreements.max()),
    }

    output = {
        "n_ensemble": cfg.N_ENSEMBLE,
        "base_seed": cfg.BASE_SEED,
        "validation_period": f"{cfg.VAL_START} to {cfg.VAL_END}",
        "calibrations": calibrations,
        "disagreement_stats": disagreement_stats,
    }

    cal_path = os.path.join(rq1_config.RESULTS_ROOT, "calibration_all.json")
    with open(cal_path, "w") as f:
        json.dump(output, f, indent=2)

    val_df = pd.DataFrame({"disagreement": disagreements})
    val_df.to_csv(
        os.path.join(rq1_config.RESULTS_ROOT, "validation_disagreements.csv"),
        index=False,
    )
    print(f"\nSaved all calibrations to {cal_path}")


if __name__ == "__main__":
    main()
