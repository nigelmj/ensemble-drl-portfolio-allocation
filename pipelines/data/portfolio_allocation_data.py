"""
Portfolio Allocation – Data Preparation

Downloads DJIA constituent data, VIX, and the DJIA index from Yahoo Finance,
engineers features (log returns, realised volatility), and saves a processed
DataFrame compatible with my_env.portfolio_allocation_env.PortfolioAllocationEnv.

Usage:
    python -m pipelines.portfolio_allocation_data

Output:
    data/portfolio_allocation_data.csv
    data/portfolio_allocation_data.pkl
"""

from __future__ import annotations

import os
from bisect import bisect_left
from typing import List

import numpy as np
import pandas as pd
import yfinance as yf

from rl_portfolio.config import TEST_END_DATE
from rl_portfolio.config import TEST_START_DATE
from rl_portfolio.config import TIME_WINDOW
from rl_portfolio.config import TRAIN_END_DATE
from rl_portfolio.config import TRAIN_START_DATE
from rl_portfolio.tickers import DOW_30_TICKER
from rl_portfolio.utils.io import check_and_make_directories

# ── Config ──────────────────────────────────────────────────────────

START_DATE = "2010-01-01"
END_DATE = "2026-03-31"

VOL_WINDOW_20 = 20
VOL_WINDOW_60 = 60

TICKER_LIST = DOW_30_TICKER
VIX_TICKER = "^VIX"
DJI_TICKER = "^DJI"

OUTPUT_CSV = "data/portfolio_allocation_data.csv"
OUTPUT_PICKLE = "data/portfolio_allocation_data.pkl"

TRAIN_CSV = "data/portfolio_allocation_train.csv"
TRAIN_PICKLE = "data/portfolio_allocation_train.pkl"

TEST_CSV = "data/portfolio_allocation_test.csv"
TEST_PICKLE = "data/portfolio_allocation_test.pkl"

# ── Download ────────────────────────────────────────────────────────


def download_stock_data(
    ticker_list: List[str],
    start: str,
    end: str,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """Download daily OHLCV data for each ticker and return a long-format DataFrame."""
    data_frames = []
    for tic in ticker_list:
        df = yf.download(tic, start=start, end=end, auto_adjust=auto_adjust)
        if df.empty:
            print(f"  Warning: no data for {tic}, skipping")
            continue
        if df.columns.nlevels != 1:
            df.columns = df.columns.droplevel(1)
        df = df.reset_index()
        df.rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            },
            inplace=True,
        )
        if "date" not in df.columns:
            df.rename(columns={"index": "date"}, inplace=True)
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        df["tic"] = tic
        data_frames.append(df)

    if not data_frames:
        raise ValueError("No data was fetched for any ticker.")

    result = pd.concat(data_frames, axis=0, ignore_index=True)
    result = result.sort_values(by=["date", "tic"]).reset_index(drop=True)
    return result[["date", "tic", "open", "high", "low", "close", "volume"]]


def download_single_ticker(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download a single ticker (e.g. ^VIX, ^DJI) and return [date, close]."""
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    if df.columns.nlevels != 1:
        df.columns = df.columns.droplevel(1)
    df = df.reset_index()
    df.rename(
        columns={"Date": "date", "Close": "close"},
        inplace=True,
    )
    if "date" not in df.columns:
        df.rename(columns={"index": "date"}, inplace=True)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df[["date", "close"]]


# ── Feature engineering ─────────────────────────────────────────────


def compute_stock_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-ticker log_return column. First row per ticker is NaN."""
    df = df.sort_values(by=["tic", "date"]).reset_index(drop=True)
    df["log_return"] = df.groupby("tic")["close"].transform(
        lambda s: np.log(s / s.shift(1))
    )
    return df


def compute_dji_features(dji_df: pd.DataFrame) -> pd.DataFrame:
    """Compute dji_log_return, vol20, vol60, vol20d60 from DJIA index closes."""
    dji = dji_df.sort_values("date").reset_index(drop=True).copy()
    dji["dji_log_return"] = np.log(dji["close"] / dji["close"].shift(1))

    ann_factor = np.sqrt(252)
    dji["vol20"] = (
        dji["dji_log_return"].rolling(window=VOL_WINDOW_20).std() * ann_factor
    )
    dji["vol60"] = (
        dji["dji_log_return"].rolling(window=VOL_WINDOW_60).std() * ann_factor
    )
    dji["vol20d60"] = dji["vol20"] / dji["vol60"]

    return dji[["date", "dji_log_return", "vol20", "vol60", "vol20d60"]]


# ── Merge ───────────────────────────────────────────────────────────


def merge_features(
    stock_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    dji_features_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge stock data with VIX and DJIA-derived features on date."""

    df = stock_df.merge(vix_df, on="date", how="left", suffixes=("", "_vix"))
    df["vix"] = df["close_vix"].ffill()
    df.drop(columns=["close_vix"], inplace=True)

    df = df.merge(dji_features_df, on="date", how="left")

    df = df.sort_values(by=["tic", "date"]).reset_index(drop=True)
    return df


# ── Preprocess ──────────────────────────────────────────────────────


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the merged DataFrame. Keeps warm-up NaNs intact."""
    df = df.dropna(subset=["close"]).reset_index(drop=True)

    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "log_return",
        "vix",
        "vol20",
        "vol60",
        "vol20d60",
        "dji_log_return",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(np.float32)

    df = df.sort_values(by=["tic", "date"]).reset_index(drop=True)
    return df


def add_day_index(df: pd.DataFrame) -> pd.DataFrame:
    """Add sequential trading-day index column (0..N-1)."""
    all_dates = sorted(df["date"].unique())
    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    df["day_index"] = df["date"].map(date_to_idx).astype(np.int32)
    return df


# ── Validation ──────────────────────────────────────────────────────


def validate_data(df: pd.DataFrame) -> None:
    """Run integrity checks and print a dataset summary."""
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)

    # Required columns
    required = ["date", "tic", "close", "vix", "vol20", "vol20d60"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    print(f"[OK] All required columns present")

    # Duplicate (date, tic) pairs
    dups = df.duplicated(subset=["date", "tic"]).sum()
    if dups > 0:
        raise ValueError(f"Found {dups} duplicate (date, tic) rows")
    print(f"[OK] No duplicate (date, tic) rows")

    # Date ordering within each ticker
    for tic, group in df.groupby("tic"):
        dates = group["date"].values
        if not np.all(dates[:-1] <= dates[1:]):
            raise ValueError(f"Dates not monotonically increasing for {tic}")
    print(f"[OK] Dates are monotonically increasing within each ticker")

    # NaN check in required columns
    # vol60 (and therefore vol20d60) is NaN for the first 60 trading days
    # (indices 0-59): a `window`-length rolling std needs `window` values, and
    # the dji_log_return series starts with a NaN. TIME_WINDOW must be >= this
    # warm-up so the environment's first observation has valid features.
    n_warmup = max(VOL_WINDOW_20, VOL_WINDOW_60)  # 60
    for tic, group in df.groupby("tic"):
        group = group.sort_values("date").reset_index(drop=True)
        if len(group) <= n_warmup:
            continue  # too short to check — will be caught by env's own capacity check
        tail = group.iloc[n_warmup:]  # from index 60 onward
        for col in ["close", "vix"]:
            if tail[col].isna().any():
                bad = tail[tail[col].isna()][["date", col]]
                raise ValueError(f"NaN in {col} for {tic} after warm-up:\n{bad}")
        for col in ["vol20", "vol20d60"]:
            if tail[col].isna().any():
                bad_dates = tail.loc[tail[col].isna(), "date"].tolist()
                raise ValueError(
                    f"NaN in {col} for {tic} on dates {bad_dates[:5]} after warm-up"
                )

    print(f"[OK] No NaN in required columns after warm-up period")

    # Summary
    print(f"\nDataset summary:")
    print(f"  Shape:                {df.shape[0]} rows × {df.shape[1]} cols")
    print(f"  Date range:           {df['date'].min()} to {df['date'].max()}")
    print(f"  Unique tickers:       {df['tic'].nunique()}")
    print(f"  Trading days:         {df['date'].nunique()}")
    missing = df.isna().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        print(f"  Missing values:\n{missing.to_string()}")
    else:
        print(f"  Missing values:       none")
    print(f"  First 5 rows:\n{df.head().to_string()}")
    print("=" * 60)


# ── Save ────────────────────────────────────────────────────────────


def save_data(df: pd.DataFrame, csv_path: str, pkl_path: str) -> None:
    """Save DataFrame to CSV and pickle."""
    out_dir = os.path.dirname(csv_path)
    if out_dir:
        check_and_make_directories([out_dir])

    df.to_csv(csv_path, index=False)
    print(f"Saved CSV to {csv_path}")

    df.to_pickle(pkl_path)
    print(f"Saved pickle to {pkl_path}")


# ── Train / test split ──────────────────────────────────────────────


def split_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train and test periods.

    Warm-up rows are intentionally kept in the splits. PortfolioAllocationEnv
    consumes them as the look-back window for its first observation (it starts
    trading at time index == time_window and never reads market features before
    that point), so dropping them here would shift the effective trading start
    ~time_window days later and waste that data for no benefit.

    The test split additionally prepends TIME_WINDOW trading days of data
    before TEST_START_DATE, so the environment can start trading on the first
    day of the test period instead of TIME_WINDOW days into it. Those look-back
    rows are historical at decision time, so they introduce no look-ahead bias.
    """
    dates = sorted(df["date"].unique())
    test_trade_idx = bisect_left(dates, TEST_START_DATE)
    test_start = dates[max(0, test_trade_idx - TIME_WINDOW)]

    train = df[
        (df["date"] >= TRAIN_START_DATE) & (df["date"] <= TRAIN_END_DATE)
    ].reset_index(drop=True)
    test = df[
        (df["date"] >= test_start) & (df["date"] <= TEST_END_DATE)
    ].reset_index(drop=True)
    return train, test


# ── Main ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Portfolio Allocation — Data Preparation")
    print("=" * 60)

    # 1. Download
    print("\nDownloading stock data...")
    stock_df = download_stock_data(TICKER_LIST, START_DATE, END_DATE)
    print(f"  {len(stock_df.tic.unique())} tickers, {stock_df.date.nunique()} days")

    print("Downloading VIX...")
    vix_df = download_single_ticker(VIX_TICKER, START_DATE, END_DATE)
    print(f"  {len(vix_df)} rows")

    print("Downloading DJIA index...")
    dji_df = download_single_ticker(DJI_TICKER, START_DATE, END_DATE)
    print(f"  {len(dji_df)} rows")

    # 2. Feature engineering
    print("\nComputing features...")
    stock_df = compute_stock_log_returns(stock_df)
    dji_features = compute_dji_features(dji_df)
    print("  Done")

    # 3. Merge
    print("\nMerging features...")
    merged = merge_features(stock_df, vix_df, dji_features)
    print(f"  {len(merged)} rows after merge")

    # 4. Preprocess
    print("\nPreprocessing...")
    cleaned = preprocess(merged)
    print(f"  {len(cleaned)} rows after preprocessing")

    # 5. Add day index
    cleaned = add_day_index(cleaned)
    print("  Added day_index column")

    # 6. Validate
    validate_data(cleaned)

    # 7. Save
    print("\nSaving...")
    save_data(cleaned, OUTPUT_CSV, OUTPUT_PICKLE)

    # 8. Split into train / test and save
    print("\nSplitting into train / test...")
    train, test = split_train_test(cleaned)
    print(f"  Train: {len(train)} rows, {train.date.nunique()} days")
    print(f"  Test:  {len(test)} rows, {test.date.nunique()} days")
    save_data(train, TRAIN_CSV, TRAIN_PICKLE)
    save_data(test, TEST_CSV, TEST_PICKLE)
    print("\nDone.")
