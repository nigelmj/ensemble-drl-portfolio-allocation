"""Portfolio-weight helpers shared by the environment and the ensemble agents."""

from __future__ import annotations

import numpy as np


def normalize_actions(actions: np.ndarray) -> np.ndarray:
    """Project a raw action vector onto the simplex (cash + tickers).

    Falls back to a uniform allocation when the action sum is ~0. This is the
    single definition used by both PortfolioAllocationEnv and the ensemble
    agents, so inference-time normalisation cannot drift from the environment's.
    """
    actions = np.asarray(actions, dtype=np.float32).flatten()
    action_sum = actions.sum()
    if action_sum < 1e-8:
        return np.ones_like(actions) / len(actions)
    return actions / action_sum
