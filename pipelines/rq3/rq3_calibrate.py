"""
RQ3 — Calibrate reward scaling per algorithm (SAC + A2C).

Pure environment-stepping (no training). Steps each algorithm's already
trained ``{algo}_log_return`` policy (all in rq_trained_models/rq3) on the
train split under the dsr / dsr_drawdown reward functions and computes
per-algorithm scaling so that different algorithms' DSR rewards are
comparable. PPO members use the RQ2 scalings (5.0 / 1400) in
rq3_config and never touched here.

SAC/A2C dsr_drawdown eta is anchored to PPO's 1400:
    heat = eta_dd * mean(penalty>0) / mean|raw DSR|
is first measured for PPO (from its ppo_log_return policy), then
each algo gets eta such that its own heat matches PPO's.
Requires the PPO log_return member to be present in rq_trained_models/rq3
before this step runs.

Usage:
    python -m pipelines.rq3.rq3_train --algo sac --reward-type log_return
    python -m pipelines.rq3.rq3_train --algo a2c --reward-type log_return
    python -m pipelines.rq3.rq3_calibrate

Output:
    results/0rq3/calibration_dsr.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from pipelines.rq3 import rq3_config
from rl_portfolio.agents.portfolio_allocation_agent import MODELS
from rl_portfolio.utils.io import check_and_make_directories
from rl_portfolio.envs.portfolio_allocation_env import PortfolioAllocationEnv

REF_TAG = ""  # reference models are the untagged log_return members


def collect_rewards(algo: str, reward_type: str, df, scaling: float = 1.0,
                    eta_dd: float | None = None, tag: str = "") -> np.ndarray:
    suffix = f"_{tag}" if tag else ""
    model = MODELS[algo].load(f"{rq3_config.MODEL_DIR}/{algo}_log_return{suffix}")
    kwargs = dict(rq3_config.BASE_ENV_KWARGS)
    kwargs["reward_type"] = reward_type
    if reward_type in ("dsr", "dsr_drawdown"):
        kwargs["reward_scaling_dsr"] = scaling
    if reward_type == "dsr_drawdown":
        kwargs["dsr_eta_dd"] = eta_dd or 0.0
    env = PortfolioAllocationEnv(df=df, **kwargs)
    test_env, test_obs = env.get_sb_env()
    rewards = []
    for _ in range(env.episode_length):
        action, _ = model.predict(test_obs, deterministic=True)
        test_obs, r, dones, _ = test_env.step(action)
        rewards.append(float(r[0]))
        if dones[0]:
            break
    return np.array(rewards)


def _penalty_heat(penalty: np.ndarray, dsr_mag: float):
    """heat = mean(penalty>0) / mean|raw DSR| — the frac-equivalent per unit eta."""
    pos = penalty[penalty > 0]
    if len(pos) == 0 or pos.mean() <= 0 or dsr_mag <= 0:
        return None
    return pos.mean() / dsr_mag


def stage_reward(train, algos, tag="", out_path=rq3_config.CALIBRATION_PATH):
    print("=" * 74)
    print("RQ3 CALIBRATION: per-algo scaling for dsr / dsr_drawdown")
    print("=" * 74)
    print(f"PPO anchor: RQ2 scalings "
          f"({rq3_config.PPO_REWARD_SCALING_DSR} / "
          f"{rq3_config.PPO_DSR_ETA_DD:,.0f})")
    check_and_make_directories([out_path.rsplit('/', 1)[0]])

    # Reference heat from PPO's own dsr_drawdown penalty (eta=1.0, scaling=1.0
    # so the ratio is pure; eta and scaling cancel in the heat).
    ppo_dsr_raw = collect_rewards("ppo", "dsr", train, scaling=1.0, tag=tag)
    ppo_dsr_mag = float(np.abs(ppo_dsr_raw).mean())
    ppo_dd = collect_rewards("ppo", "dsr_drawdown", train, scaling=1.0,
                             eta_dd=1.0, tag=tag)
    ppo_pen = np.maximum(ppo_dsr_raw - ppo_dd, 0.0)
    ppo_pos = ppo_pen[ppo_pen > 0]
    ppo_heat = _penalty_heat(ppo_pen, ppo_dsr_mag)
    if ppo_heat is None or not np.isfinite(ppo_heat):
        ppo_heat = 1.0
    heat_ref = rq3_config.PPO_DSR_ETA_DD * ppo_heat
    print(f"  PPO dsr_mag={ppo_dsr_mag:.4f}, "
          f"dd_pen_mean(>0)={float(ppo_pos.mean()) if len(ppo_pos) else 0.0:.3e}")
    print(f"  -> reference penalty heat = {heat_ref:.4f} (frac-equivalent)")

    algo_calibration = {}

    for algo in algos:
        log_mag = float(np.abs(
            collect_rewards(algo, "log_return", train, tag=tag)).mean())

        # raw DSR with its default scaling (scaling factor cancels in the ratio)
        dsr_raw = collect_rewards(algo, "dsr", train, scaling=1.0, tag=tag)
        dsr_mag = float(np.abs(dsr_raw).mean())
        reward_scaling_dsr = (log_mag / dsr_mag if dsr_mag > 0
                              else rq3_config.FALLBACK_REWARD_SCALING_DSR)

        # drawdown penalty, from dsr_drawdown with dd=1.0
        dd_rewards = collect_rewards(algo, "dsr_drawdown", train,
                                     scaling=1.0, eta_dd=1.0, tag=tag)
        penalty = np.maximum(dsr_raw - dd_rewards, 0.0)
        pos = penalty[penalty > 0]
        algo_heat = _penalty_heat(penalty, dsr_mag)
        if len(pos) > 0 and pos.mean() > 0 and algo_heat > 0:
            dsr_eta_dd = heat_ref / algo_heat
        else:
            dsr_eta_dd = rq3_config.FALLBACK_DSR_ETA_DD

        algo_calibration[algo] = {
            "log_return_mean_abs": log_mag,
            "reward_scaling_dsr": float(reward_scaling_dsr),
            "dsr_eta_dd": float(dsr_eta_dd),
            "raw_dsr_mean_abs": dsr_mag,
            "dd_penalty_mean_pos": float(pos.mean()) if len(pos) else 0.0,
        }

        print(f"\n{algo} (reference model = {algo}_log_return)")
        print(f"  mean|log_return reward| = {log_mag:.4f}")
        print(f"  mean|raw DSR reward|    = {dsr_mag:.4f}")
        if len(pos) > 0:
            print(f"  drawdown penalty mean(>0) = {pos.mean():.3e}")
        print(f"  -> reward_scaling_dsr = {reward_scaling_dsr:.3f}")
        print(f"  -> dsr_eta_dd         = {dsr_eta_dd:,.0f}")

    # PPO uses the RQ2 scalings in rq3_config.
    algo_calibration["ppo"] = {
        "log_return_mean_abs": None,
        "reward_scaling_dsr": rq3_config.PPO_REWARD_SCALING_DSR,
        "dsr_eta_dd": rq3_config.PPO_DSR_ETA_DD,
        "raw_dsr_mean_abs": None,
        "dd_penalty_mean_pos": None,
        "note": "RQ2 scalings",
    }

    calibration = {
        "method": ("per-algo, eta anchored to PPO's "
                   f"{rq3_config.PPO_DSR_ETA_DD:,.0f}; ref={algo}_log_return"),
        "heat_ref": float(heat_ref),
        "algo_scaling": algo_calibration,
    }
    with open(out_path, "w") as f:
        json.dump(calibration, f, indent=2)
    print(f"\nSaved per-algo calibration to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", action="append",
                        choices=rq3_config.ALGOS, default=None,
                        help="Which algorithms to calibrate (default: sac a2c).")
    parser.add_argument("--out", default=rq3_config.CALIBRATION_PATH,
                        help="Output JSON path.")
    args = parser.parse_args()

    algos = args.algo or ["sac", "a2c"]
    train = pd.read_pickle(rq3_config.DATA_TRAIN)
    print(f"Train: {len(train)} rows, {train.date.nunique()} days, "
          f"{train.tic.nunique()} tickers")

    stage_reward(train, algos, out_path=args.out)
    print("\nDone.")


if __name__ == "__main__":
    main()
