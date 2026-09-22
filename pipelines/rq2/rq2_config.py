"""
RQ2 — reward diversity ensemble: shared configuration.

Three PPO members on {log_return, dsr, dsr_drawdown} with
reward_scaling_dsr=5.0 and dsr_eta_dd=1400, plus RQ2 output roots.

Outputs (never overwrite existing results/models):
    models   -> rq_trained_models/rq2/
    results  -> results/rq2/
"""
from __future__ import annotations

from rl_portfolio.config import TIME_WINDOW

REWARD_TYPES = ["log_return", "dsr", "dsr_drawdown"]

REWARD_SCALING_DSR = 5.0
DSR_ETA_DD = 1400.0

PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.001,
    "learning_rate": 0.00025,
    "batch_size": 64,
}

POLICY_KWARGS = {"log_std_init": -2.0}
SEED = 42

TOTAL_TIMESTEPS = 1_000_000

DATA_DIR = "data"
MODEL_DIR = "rq_trained_models/rq2"
RESULTS_ROOT = "results/rq2"
DATA_TRAIN = f"{DATA_DIR}/portfolio_allocation_train.pkl"
DATA_TEST = f"{DATA_DIR}/portfolio_allocation_test.pkl"

ENV_KWARGS = {
    "initial_amount": 1_000_000,
    "time_window": TIME_WINDOW,
    "fee_pct": 0.001,
    "reward_scaling": 500,
    "market_feature_cols": ["vix", "vol20", "vol20d60"],
}

DEFAULT_MODEL_PATHS = [
    f"{MODEL_DIR}/agent_ppo_pa_log_return",
    f"{MODEL_DIR}/agent_ppo_pa_dsr",
    f"{MODEL_DIR}/agent_ppo_pa_dsr_drawdown",
]

MEMBER_NAMES = {
    "log_return": "PPO LogReturn",
    "dsr": "PPO DSR",
    "dsr_drawdown": "PPO DSR-DD",
}


def build_env_kwargs(reward_type: str) -> dict:
    kwargs = dict(ENV_KWARGS)
    if reward_type in ("dsr", "dsr_drawdown"):
        kwargs["reward_type"] = reward_type
        kwargs["reward_scaling_dsr"] = REWARD_SCALING_DSR
        if reward_type == "dsr_drawdown":
            kwargs["dsr_eta_dd"] = DSR_ETA_DD
    return kwargs
