"""
RQ1 — inter-seed uncertainty ensemble: shared configuration.

Trains on the FULL shared train split (2010-2021) and calibrates D_ref on the
last three years of it (2019-2021), leaving the test window (2022-2026)
completely untouched.

Outputs (never overwrite existing results/models):
    models   -> rq_trained_models/rq1/
    results  -> results/rq1/
"""
from __future__ import annotations


from rl_portfolio.config import TIME_WINDOW

N_ENSEMBLE = 5
BASE_SEED = 42
TOTAL_TIMESTEPS = 1_000_000
TOTAL_TIMESTEPS_SINGLE_5M = 5_000_000  # compute-matched single baseline (5 × 1M)
SMOKE_TIMESTEPS = 60_000

D_REF_METHOD = "p90"
POWER_P = 2.0
SIGMOID_K = 1.0

# Beta-steepness for new delta-centred mappings (single beta shared by exp & sigmoid)
#   exponential: c = clip(exp(-beta*(delta-1)),0,1),  delta=D/D_ref
#   sigmoid:     c = 1/(1+exp(beta*(delta-1)))
# Legacy (beta=None) keeps old: exp(-D/D_ref) and 1/(1+exp(k*(D-D_ref)))
BETA_GRID = [10, 20, 30, 40, 60, 80, 100]

CONFIDENCE_MAPPINGS = ["linear", "power", "exponential", "sigmoid"]
SAFE_STRATEGIES = ["previous", "equal_weight", "equal_weight_stocks"]

# Temporary reporting for stocks-only ablation (does not overwrite rq1)
TEMP_RESULTS_ROOT = "results/temp/rq1"

D_REF_METHODS = ["mean", "median", "p75", "p90", "p95"]

VAL_START = "2019-01-01"
VAL_END = "2021-12-31"

DATA_DIR = "data"
MODEL_DIR = "rq_trained_models/rq1"
RESULTS_ROOT = "results/rq1"
DATA_TRAIN = f"{DATA_DIR}/portfolio_allocation_train.pkl"
DATA_TEST = f"{DATA_DIR}/portfolio_allocation_test.pkl"

ENV_KWARGS = {
    "initial_amount": 1_000_000,
    "time_window": TIME_WINDOW,
    "fee_pct": 0.001,
    "reward_scaling": 500,
    "market_feature_cols": ["vix", "vol20", "vol20d60"],
}

PPO_KWARGS = {
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 0.00025,
    "batch_size": 64,
}

POLICY_KWARGS = {"log_std_init": -2.0}


def combo_name(mapping: str, safe_strategy: str) -> str:
    return f"combo_{mapping}_{safe_strategy}"

