"""
RQ3 — algorithm diversity ensemble: shared configuration.

Six reward-algo versions (v1..v6) as permutations of {log_return, dsr,
dsr_drawdown} over the fixed algorithm order [ppo, sac, a2c].

Members:
    PPO members  -> reuse the RQ2 models (rq_trained_models/rq2),
                    copied into rq_trained_models/rq3/ before training.
                    Their DSR scaling is fixed to RQ2's 5.0 / 1400.
    A2C members  -> trained by rq3_train.py (log_return -> dsr ->
                    dsr_drawdown) at 1M steps, seed 42.
    SAC members  -> trained by rq3_train.py (log_return -> dsr -> dsr_drawdown).

The SAC/A2C dsr_drawdown eta is calibrated so each member carries the same
penalty scale as PPO's 1400 (see rq3_calibrate.py).

Outputs (never overwrite existing results/models):
    models   -> rq_trained_models/rq3/
    results  -> results/0rq3/ver_v1..v6/
"""
from __future__ import annotations

import itertools
import json
import os

from rl_portfolio.config import TIME_WINDOW

ALGOS = ["ppo", "sac", "a2c"]
REWARD_TYPES = ["log_return", "dsr", "dsr_drawdown"]

# RQ2 scalings — PPO members always use these regardless of the calibration json.
PPO_REWARD_SCALING_DSR = 5.0
PPO_DSR_ETA_DD = 1400.0

# SAC/A2C dsr_eta_dd is anchored to PPO's 1400 in the calibrate step:
# each algo gets eta such that eta * penalty / |DSR| matches PPO's,
# so the DD penalty scale is comparable across members.

# Per-algorithm training hyperparameters (used by PortfolioAllocationDRLAgent).
MODEL_KWARGS = {
    "ppo": {
        "n_steps": 2048,
        "ent_coef": 0.001,
        "learning_rate": 0.00025,
        "batch_size": 64,
    },
    "a2c": {
        "n_steps": 5,
        "ent_coef": 0.01,
        "learning_rate": 0.0007,
    },
    "sac": {
        "learning_rate": 0.0003,
        "buffer_size": 300000,
        "batch_size": 256,
        "learning_starts": 10000,
        "train_freq": 1,
        "gradient_steps": 1,
        "ent_coef": "auto",
        "gamma": 0.99,
        "tau": 0.005,
    },
}

SEED = 42

# log_std_init only applies to stochastic (actor-critic) policies ppo/a2c.
# SAC's actor uses the same MLP policy; log_std_init is not applicable to it.
ALGO_POLICY_KWARGS = {
    "ppo": {"log_std_init": -2.0},
    "a2c": {"log_std_init": -2.0},
    "sac": None,
}

BASE_ENV_KWARGS = {
    "initial_amount": 1_000_000,
    "time_window": TIME_WINDOW,
    "fee_pct": 0.001,
    "reward_scaling": 500,
    "market_feature_cols": ["vix", "vol20", "vol20d60"],
}

# All 6 permutations of rewards assigned to the fixed algo order [ppo,sac,a2c].
VERSIONS = {}
for idx, perm in enumerate(itertools.permutations(REWARD_TYPES), start=1):
    VERSIONS[f"v{idx}"] = [
        {"algo": algo, "reward_type": rt} for algo, rt in zip(ALGOS, perm)
    ]

VERSION_ORDER = list(VERSIONS)

# Reward-type abbreviations used in figure labels and CSV column headers.
_REWARD_SHORT = {"log_return": "log_return", "dsr": "dsr", "dsr_drawdown": "dsr_dd"}

# Long per-version labels, derived from VERSIONS so they cannot drift from the
# member specs they describe: "v1 ppo=log_return sac=dsr a2c=dsr_dd".
VERSION_LABELS = {
    version: " ".join(
        [version] + [f"{m['algo']}={_REWARD_SHORT[m['reward_type']]}" for m in specs]
    )
    for version, specs in VERSIONS.items()
}

RQ2_BLOCK_LABEL = "RQ2 k10b20"
RQ2_BLOCK_LABEL_DISPLAY = "RQ2 block ensemble"

DATA_DIR = "data"
MODEL_DIR = "rq_trained_models/rq3"
RESULTS_ROOT = "results/0rq3"
DATA_TRAIN = f"{DATA_DIR}/portfolio_allocation_train.pkl"
DATA_TEST = f"{DATA_DIR}/portfolio_allocation_test.pkl"

CALIBRATION_PATH = f"{RESULTS_ROOT}/calibration_dsr.json"

TOTAL_TIMESTEPS = 1_000_000
SMOKE_TIMESTEPS = 60_000

# Fallbacks if calibration_dsr.json is absent (only used for sac/a2c members).
FALLBACK_REWARD_SCALING_DSR = 5.0
FALLBACK_DSR_ETA_DD = 1400.0


def load_calibration(path: str = CALIBRATION_PATH) -> dict:
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _reward_env_kwargs(algo: str, reward_type: str, calibration: dict) -> dict:
    """Extra env kwargs for a reward type, with the RQ2 override for PPO."""
    if reward_type == "log_return":
        return {}

    if algo == "ppo":
        # PPO members always use the RQ2 scalings.
        if reward_type == "dsr":
            return {"reward_type": "dsr",
                    "reward_scaling_dsr": PPO_REWARD_SCALING_DSR}
        return {"reward_type": "dsr_drawdown",
                "reward_scaling_dsr": PPO_REWARD_SCALING_DSR,
                "dsr_eta_dd": PPO_DSR_ETA_DD}

    algo_cal = calibration.get("algo_scaling", {}).get(algo, {})
    if reward_type == "dsr":
        return {
            "reward_type": "dsr",
            "reward_scaling_dsr": float(
                algo_cal.get("reward_scaling_dsr", FALLBACK_REWARD_SCALING_DSR)
            ),
        }
    return {
        "reward_type": "dsr_drawdown",
        "reward_scaling_dsr": float(
            algo_cal.get("reward_scaling_dsr", FALLBACK_REWARD_SCALING_DSR)
        ),
        "dsr_eta_dd": float(
            algo_cal.get("dsr_eta_dd", FALLBACK_DSR_ETA_DD)
        ),
    }


def build_env_kwargs(
    algo: str,
    reward_type: str,
    calibration: dict | None = None,
) -> dict:
    kwargs = dict(BASE_ENV_KWARGS)
    kwargs.update(_reward_env_kwargs(algo, reward_type, calibration or {}))
    return kwargs


def build_member_specs(
    specs: list, calibration: dict | None = None
) -> list:
    cal = calibration or {}
    out = []
    for m in specs:
        out.append({
            "algo": m["algo"],
            "reward_type": m["reward_type"],
            "path": f"{MODEL_DIR}/{m['algo']}_{m['reward_type']}",
            "env_kwargs": build_env_kwargs(m["algo"], m["reward_type"], cal),
        })
    return out
