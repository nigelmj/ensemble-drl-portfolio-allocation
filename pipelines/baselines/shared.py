"""
Shared loader for the baseline curves produced by compute_baselines.py.

Every RQ backtest reads the four CSVs from results/baselines/ via this module so
that the Equal-Weight / MVO (buy-and-hold and 21-day rebalanced) curves are
identical across RQ1 / RQ2 / RQ3.
"""

from __future__ import annotations

import os

import pandas as pd

BASELINES_DIR = "results/baselines"

NAMES = ["ew_buyhold", "ew_rebalanced", "mvo_buyhold", "mvo_rebalanced"]

LABELS = {
    "ew_buyhold": "Equal Weight",
    "ew_rebalanced": "EW Rebalanced",
    "mvo_buyhold": "Mean Var",
    "mvo_rebalanced": "MVO Rebalanced",
}

REBALANCED_NAMES = ["ew_rebalanced", "mvo_rebalanced"]


def load_baseline(name: str) -> pd.Series:
    """Load one baseline as a Series indexed by 'date' (YYYY-MM-DD strings)."""
    if name not in NAMES:
        raise ValueError(f"Unknown baseline '{name}'. Got {NAMES}")
    path = os.path.join(BASELINES_DIR, f"{name}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Baseline not found: {path}\nRun "
            "`python -m pipelines.baselines.compute_baselines` first."
        )
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df.set_index("date")["value"]


def attach_baselines(
    result: pd.DataFrame,
    dates,
    label: str = "value",
    initial_amount: float = 1_000_000,
) -> pd.DataFrame:
    """Append the 4 baseline columns to `result` (indexed by its 'date' column).

    The baseline curves were computed from the full 1,000,000 initial amount;
    when a backtest starts later (e.g. after a lead-in) the curves are anchored
    to `initial_amount` at the first in-window date so the equity comparison is
    apples-to-apples.
    """
    out = result.copy()
    first = dates[0]
    for name in NAMES:
        s = load_baseline(name)
        s = s.reindex([str(d) for d in dates])
        s = s / s.iloc[0] * initial_amount  # anchor to the backtest start
        out[LABELS[name]] = s.values
    return out
