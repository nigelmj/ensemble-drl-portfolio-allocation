"""
RQ2 — Calibrate reward_scaling_dsr on the trained log_return PPO.

Samples the raw differential-Sharpe-ratio reward (DSR_t) under the RQ2
log_return policy on the train split and prints/saves the suggested
reward_scaling_dsr and dsr_eta_dd at several fracs. Run AFTER rq2_train.py.

Usage:
    python -m pipelines.rq2.rq2_train
    python -m pipelines.rq2.rq2_calibrate

Output:
    results/0rq2/calibration_dsr_scaling.txt
    results/0rq2/calibration_dsr_scaling.json
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from pipelines.rq2 import rq2_config
from rl_portfolio.config import RESULTS_DIR
from rl_portfolio.utils.io import check_and_make_directories
from rl_portfolio.envs.portfolio_allocation_env import PortfolioAllocationEnv

MODEL_PATH = f"{rq2_config.MODEL_DIR}/agent_ppo_pa_log_return"

env_kwargs = dict(rq2_config.ENV_KWARGS)
env_kwargs["reward_scaling_dsr"] = 1.0


def collect_rewards(reward_type: str, **extra_kwargs) -> np.ndarray:
    env = PortfolioAllocationEnv(df=train, reward_type=reward_type,
                                 **env_kwargs, **extra_kwargs)
    test_env, test_obs = env.get_sb_env()
    rewards = []
    for _ in range(env.episode_length):
        action, _ = model.predict(test_obs, deterministic=True)
        test_obs, r, dones, _ = test_env.step(action)
        rewards.append(float(r[0]))
        if dones[0]:
            break
    return np.array(rewards)


train = pd.read_pickle(rq2_config.DATA_TRAIN)
model = PPO.load(MODEL_PATH)

check_and_make_directories([RESULTS_DIR, rq2_config.RESULTS_ROOT])

print(f"Model: {MODEL_PATH}")
print(f"Train split: {len(train)} rows, {train.date.nunique()} days\n")

log_rewards = collect_rewards("log_return")
dsr_rewards = collect_rewards("dsr")
dd_rewards = collect_rewards("dsr_drawdown", dsr_eta_dd=1.0)

lines = []
def report(name: str, rewards: np.ndarray) -> None:
    s = (f"{name:12s} mean={rewards.mean():+.4f} "
         f"mean|.|={np.abs(rewards).mean():.4f} std={rewards.std():.4f} "
         f"min={rewards.min():+.4f} max={rewards.max():+.4f}")
    print(s)
    lines.append(s)
    pcts = np.percentile(rewards, [5, 25, 50, 75, 95])
    s2 = (f"            pct5={pcts[0]:+.4f} p25={pcts[1]:+.4f} "
          f"p50={pcts[2]:+.4f} p75={pcts[3]:+.4f} p95={pcts[4]:+.4f}")
    print(s2)
    lines.append(s2)


report("log_return", log_rewards)
report("dsr_raw", dsr_rewards)

penalty = np.maximum(dsr_rewards - dd_rewards, 0.0)
pos = penalty[penalty > 0]
dd_stats = np.percentile(penalty, [50, 75, 90, 95, 99])
s = (f"\ndrawdown penalty (max(0,dDD)^2, eta_dd=1.0):\n"
     f"            pct_nonzero={100 * len(pos) / len(penalty):.1f}% "
     f"mean={penalty.mean():.3e} p50={dd_stats[0]:.3e} "
     f"p75={dd_stats[1]:.3e} p90={dd_stats[2]:.3e} p95={dd_stats[3]:.3e} "
     f"p99={dd_stats[4]:.3e}")
print(s)
lines.append(s)
if len(pos) > 0:
    s = (f"            mean(positive)={pos.mean():.3e} "
         f"p50(pos)={np.median(pos):.3e} p90(pos)={np.percentile(pos, 90):.3e}")
    print(s)
    lines.append(s)

log_mag = np.abs(log_rewards).mean()
dsr_mag = np.abs(dsr_rewards).mean()
suggested = log_mag / dsr_mag if dsr_mag > 0 else 1.0

s = (f"\nMean |log_return reward| = {log_mag:.4f}\n"
     f"Mean |raw DSR reward|    = {dsr_mag:.4f}\n"
     f"\nSuggested reward_scaling_dsr = {suggested:.2f}")
print(s)
lines.append(s)

eta_by_frac = {}
if len(pos) > 0:
    s = "\nSuggested dsr_eta_dd (typical down day, frac of mean|DSR|):"
    print(s)
    lines.append(s)
    for frac in (0.25, 0.5, 1.0):
        eta = frac * dsr_mag / pos.mean()
        eta_by_frac[frac] = float(eta)
        crash_penalty = eta * suggested * dd_stats[4]
        s = f"  frac={frac:.2f} -> eta_dd = {eta:,.0f} " \
            f"(crash-day scaled penalty ~ {crash_penalty:.1f})"
        print(s)
        lines.append(s)

result = {
    "model_path": MODEL_PATH,
    "log_return_mean_abs": float(log_mag),
    "raw_dsr_mean_abs": float(dsr_mag),
    "reward_scaling_dsr": float(suggested),
    "dsr_eta_dd_frac025": eta_by_frac.get(0.25),
    "dsr_eta_dd_frac050": eta_by_frac.get(0.5),
    "dsr_eta_dd_frac100": eta_by_frac.get(1.0),
    "current_defaults": {
        "reward_scaling_dsr": rq2_config.REWARD_SCALING_DSR,
        "dsr_eta_dd": rq2_config.DSR_ETA_DD,
    },
}
with open(f"{rq2_config.RESULTS_ROOT}/calibration_dsr_scaling.json", "w") as f:
    json.dump(result, f, indent=2)
with open(f"{rq2_config.RESULTS_ROOT}/calibration_dsr_scaling.txt", "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"\nSaved calibration to {rq2_config.RESULTS_ROOT}/calibration_dsr_scaling.json")
print(f"Current train defaults: scaling={rq2_config.REWARD_SCALING_DSR}, "
      f"eta_dd={rq2_config.DSR_ETA_DD}")
