"""
Shared transaction-cost-aware baselines for the RQ1/RQ2/RQ3 backtests.

Computes four curves once on the full test window (2022-01-01 -> 2026-03-31)
and writes them to results/baselines/ so every research-question backtest reads
the same numbers instead of recomputing MVO per RQ:

    ew_buyhold.csv       Equal weight, buy and hold
    ew_rebalanced.csv    Equal weight, rebalanced every 21 days with fee
    mvo_buyhold.csv      MVO weights (from full train history), buy and hold
    mvo_rebalanced.csv   MVO rebalanced every 21 days, weights recomputed from
                         all price history <= each rebalance date (no look-ahead)

Conventions match the existing per-RQ backtests:
    REBALANCE_DAYS = 21, FEE_PCT = 0.001, initial amount = 1,000,000,
    MVO: EfficientFrontier max_sharpe -> min_volatility fallback,
    weight_bounds (0, 0.5), clean_weights, renormalised over the 30 DJIA tics.
    Trading begins on TEST_START_DATE (the env starts at row index ==
    TIME_WINDOW, and split_train_test prepends those 60 observation days).

Usage:
    python -m pipelines.baselines.compute_baselines

Output:
    results/baselines/{ew_buyhold,ew_rebalanced,mvo_buyhold,mvo_rebalanced}.csv
    results/baselines/baselines_summary.csv
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from pypfopt.efficient_frontier import EfficientFrontier

from rl_portfolio.config import TEST_START_DATE
from rl_portfolio.utils.io import check_and_make_directories

REBALANCE_DAYS = 21
FEE_PCT = 0.001
INITIAL_AMOUNT = 1_000_000

BASELINES_DIR = "results/baselines"

TRAIN_PICKLE = "data/portfolio_allocation_train.pkl"
TEST_PICKLE = "data/portfolio_allocation_test.pkl"

BASELINE_NAMES = ["ew_buyhold", "ew_rebalanced", "mvo_buyhold", "mvo_rebalanced"]


def stock_returns(prices: np.ndarray) -> np.ndarray:
    """Simple period returns (percent), same convention as the existing scripts."""
    rows, cols = prices.shape
    out = np.zeros([rows - 1, cols])
    for j in range(cols):
        out[:, j] = (prices[1:, j] - prices[:-1, j]) / prices[:-1, j] * 100
    return out


def mvo_weights(prices: np.ndarray, weight_bounds=(0, 0.5)) -> np.ndarray:
    """Max-Sharpe (fallback min-vol) mean-variance weights from a price matrix."""

    rows, cols = prices.shape
    if rows < 2:
        return np.full(cols, 1.0 / cols)
    rets = stock_returns(prices)
    mean_ret = np.mean(rets, axis=0)
    cov_ret = np.cov(rets, rowvar=False)
    ef = EfficientFrontier(mean_ret, cov_ret, weight_bounds=weight_bounds)
    try:
        ef.max_sharpe()
    except Exception:
        ef.min_volatility()
    cleaned = ef.clean_weights()
    return np.array([cleaned[k] for k in sorted(cleaned)])


def main() -> None:
    check_and_make_directories([BASELINES_DIR])

    test = pd.read_pickle(TEST_PICKLE)
    trade = test[test["date"] >= TEST_START_DATE].reset_index(drop=True)
    tic_list = sorted(trade["tic"].unique())
    stock_dim = len(tic_list)
    print(f"Test window: {trade.date.nunique()} days, {stock_dim} tickers, "
          f"{trade['date'].min()} -> {trade['date'].max()}")

    pivot = trade.pivot(index="date", columns="tic", values="close").reindex(
        columns=tic_list
    )
    first_close = (
        trade[trade["date"] == trade["date"].min()]
        .set_index("tic")["close"]
        .reindex(tic_list)
    )
    ratios = pivot / pivot.shift(1)

    # --- buy and hold helper ---
    def buy_and_hold(dollar_allocation: np.ndarray) -> pd.Series:
        shares = dollar_allocation / first_close.to_numpy()
        values = pivot.to_numpy() @ shares
        return pd.Series(values, index=pivot.index, name="value")

    # --- rebalanced helper (target weights chosen per rebalance date) ---
    def rebalanced(target_weights_for) -> pd.Series:
        weights = target_weights_for(pivot.index[0])
        value = float(INITIAL_AMOUNT)
        values = [value]
        for i in range(1, len(pivot)):
            if i % REBALANCE_DAYS == 0:
                target = target_weights_for(pivot.index[i])
                turnover = float(np.sum(np.abs(target - weights)))
                value -= turnover * value * FEE_PCT
                weights = target
            day_ratio = ratios.iloc[i].to_numpy()
            portfolio_return = float(np.sum(weights * day_ratio))
            value *= portfolio_return
            weights = weights * day_ratio / portfolio_return
            values.append(value)
        return pd.Series(values, index=pivot.index, name="value")

    # --- full history for MVO (train + test incl. 60-day lead-in) ---
    train = pd.read_pickle(TRAIN_PICKLE)
    full = pd.concat([train, test], ignore_index=True)
    full = full.drop_duplicates(subset=["date", "tic"]).reset_index(drop=True)
    full_pivot = full.pivot(index="date", columns="tic", values="close").reindex(
        columns=tic_list
    ).sort_index()

    # --- 1. Equal weight buy and hold ---
    ew_dollars = np.full(stock_dim, INITIAL_AMOUNT / stock_dim)
    ew_buyhold = buy_and_hold(ew_dollars).rename("value")

    # --- 2. Equal weight rebalanced ---
    ew_target = np.full(stock_dim, 1.0 / stock_dim)
    ew_rebal = rebalanced(lambda _: ew_target)

    # --- static MVO weights from the train period (for buy and hold) ---
    train_stock = train.pivot(index="date", columns="tic", values="close")
    train_stock = train_stock.dropna(axis=1, how="any")
    mvo_static_full = np.zeros(stock_dim)
    mvo_static = mvo_weights(np.asarray(train_stock))
    mvo_tickers = list(train_stock.columns)
    for i, tic in enumerate(tic_list):
        if tic in mvo_tickers:
            mvo_static_full[i] = mvo_static[mvo_tickers.index(tic)]
    if mvo_static_full.sum() > 0:
        mvo_static_full /= mvo_static_full.sum()
    else:
        mvo_static_full = np.full(stock_dim, 1.0 / stock_dim)

    # --- 3. MVO buy and hold ---
    mvo_buyhold = buy_and_hold(INITIAL_AMOUNT * mvo_static_full).rename("value")

    # --- 4. MVO rebalanced (weights recomputed from history <= date) ---
    def mvo_weights_until(date: str) -> np.ndarray:
        hist = full_pivot.loc[:date].dropna(axis=1, how="any")
        w_full = np.zeros(stock_dim)
        w = mvo_weights(np.asarray(hist))
        hist_tickers = list(hist.columns)
        for i, tic in enumerate(tic_list):
            if tic in hist_tickers:
                w_full[i] = w[hist_tickers.index(tic)]
        if w_full.sum() > 0:
            w_full /= w_full.sum()
        else:
            w_full = np.full(stock_dim, 1.0 / stock_dim)
        return w_full

    mvo_rebal = rebalanced(mvo_weights_until)

    # --- save ---
    curves = {
        "ew_buyhold": ew_buyhold,
        "ew_rebalanced": ew_rebal,
        "mvo_buyhold": mvo_buyhold,
        "mvo_rebalanced": mvo_rebal,
    }
    for name, series in curves.items():
        out = pd.DataFrame({"date": series.index.astype(str),
                            "value": series.values})
        path = os.path.join(BASELINES_DIR, f"{name}.csv")
        out.to_csv(path, index=False)
        print(f"Saved {path}  (final {series.iloc[-1]:,.2f})")

    def sharpe_ratio(series: pd.Series) -> float:
        daily_returns = series.pct_change().dropna()
        if daily_returns.std() == 0:
            return float("inf") if daily_returns.mean() > 0 else 0.0
        return float(np.sqrt(252) * daily_returns.mean() / daily_returns.std())

    summary = pd.DataFrame({
        "baseline": list(curves),
        "final_value": [float(s.iloc[-1]) for s in curves.values()],
        "total_return": [float(s.iloc[-1] / INITIAL_AMOUNT - 1)
                         for s in curves.values()],
        "sharpe": [sharpe_ratio(s) for s in curves.values()],
    })
    summary.to_csv(os.path.join(BASELINES_DIR, "baselines_summary.csv"), index=False)
    print("\n=== Baselines summary ===")
    print(summary.round(4).to_string(index=False))
    print("\nDone.")


if __name__ == "__main__":
    main()
