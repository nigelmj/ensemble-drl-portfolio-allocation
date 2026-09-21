"""Shared configuration: dates, directories, environment look-back window."""

from __future__ import annotations

# directory
DATA_SAVE_DIR = "datasets"
TRAINED_MODEL_DIR = "trained_models"
TENSORBOARD_LOG_DIR = "tensorboard_log"
RESULTS_DIR = "results"

# date format: '%Y-%m-%d'
TRAIN_START_DATE = "2010-01-01"
TRAIN_END_DATE = "2021-12-31"

TEST_START_DATE = "2022-01-01"
TEST_END_DATE = "2026-03-31"

TRADE_START_DATE = "2022-01-01"
TRADE_END_DATE = "2026-03-31"

# PortfolioAllocationEnv: look-back window (trading days). The environment
# starts trading at row index == TIME_WINDOW, using the prior TIME_WINDOW days
# as the observation window. Must be >= the market-feature warm-up period
# (see pipelines/data/portfolio_allocation_data.py validate_data).
TIME_WINDOW = 60
