# Performance-weighted ensemble of reward-variant PPO agents for portfolio allocation
from __future__ import annotations

from collections import deque
from typing import List, Tuple

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from rl_portfolio.envs.portfolio_allocation_env import PortfolioAllocationEnv
from rl_portfolio.utils.allocation import normalize_actions


class PortfolioAllocationEnsembleAgent:
    """Blend 3 reward-variant PPO policies by trailing risk-adjusted performance.

    Three models (trained with ``log_return``, ``dsr`` and ``dsr_drawdown``
    rewards) each trade their own isolated hypothetical portfolio on the test
    data. A trailing metric (mean / std of daily returns over a rolling window)
    ranks the models. The blend weights are then computed per ``mode``:
    ``soft`` = temperature-scaled softmax of the metrics, EMA-smoothed over
    time; ``hard`` = one-hot on the current best metric every day; ``block`` =
    one-hot on the best metric at each block boundary, held for ``block_days``.
    A fourth ``env_real`` trades the blended allocation and produces the final
    account / action history.

    The rolling metric uses only returns up to and including the previous day
    (today's return is appended after the weights are computed), so there is no
    look-ahead in the blend. ``env_A/B/C`` never see the blended portfolio, so
    their trailing metrics are not contaminated by anything outside their own
    k-day window.
    """

    def __init__(
        self,
        model_paths: List[str],
        df: pd.DataFrame,
        env_kwargs: dict,
        k: int = 20,
        temperature: float = 1.0,
        alpha: float = 0.3,
        mode: str = "soft",
        block_days: int = 20,
    ):
        if len(model_paths) != 3:
            raise ValueError(f"Expected exactly 3 model paths, got {len(model_paths)}")
        if mode not in ("soft", "hard", "block"):
            raise ValueError(f"Unknown mode '{mode}'; expected soft/hard/block")

        self.model_paths = model_paths
        self.k = k
        self.temperature = temperature
        self.alpha = alpha
        self.mode = mode
        self.block_days = block_days

        self.models = [PPO.load(path) for path in model_paths]

        self.envs = [
            PortfolioAllocationEnv(df=df, **env_kwargs) for _ in range(3)
        ]

        self.vec_envs = []
        self.observations = []
        for env in self.envs:
            vec_env, obs = env.get_sb_env()
            self.vec_envs.append(vec_env)
            self.observations.append(obs)

        # env_real is stepped directly (not via a DummyVecEnv): DummyVecEnv
        # auto-resets an env on done=True, which would wipe env_real's asset /
        # action memory on the terminal step. Its observation is never consumed
        # by a model, so stepping it directly loses nothing.
        self.env_real = PortfolioAllocationEnv(df=df, **env_kwargs)

        self.portfolio_size = self.env_real.portfolio_size
        self.episode_length = self.env_real.episode_length

        self.buffers = [deque(maxlen=self.k) for _ in range(3)]
        self.w_smoothed = np.full(3, 1.0 / 3.0)

    def _metrics(self) -> np.ndarray:
        """Trailing mean/std per agent, from the k-day buffers only."""
        metrics = np.zeros(3, dtype=np.float32)
        for i, buf in enumerate(self.buffers):
            if len(buf) < self.k:
                return np.zeros(3, dtype=np.float32)
            arr = np.asarray(buf, dtype=np.float32)
            std = arr.std()
            metrics[i] = arr.mean() / max(std, 1e-12)
        return metrics

    def _blend_weights(self, normalized_actions: List[np.ndarray]) -> np.ndarray:
        """Weighted combination of the per-agent normalized allocations."""
        combined = np.zeros_like(normalized_actions[0], dtype=np.float32)
        for w, norm in zip(self.w_smoothed, normalized_actions):
            combined += w * norm
        return normalize_actions(combined)

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Run the ensemble on the test data.

        Returns
        -------
        account_df : pd.DataFrame
            date + portfolio_value of the blended (real) portfolio.
        actions_df : pd.DataFrame
            per-day blended target weights (cash + tics).
        weights_df : pd.DataFrame
            per-day smoothed blend weights per agent (log_return / dsr / dsr_drawdown).
        """
        weight_records = []
        block_winner = -1

        for step in range(self.episode_length):
            returns = []
            normalized_actions = []

            for i in range(3):
                action, _ = self.models[i].predict(
                    self.observations[i], deterministic=True
                )
                norm = normalize_actions(action)
                normalized_actions.append(norm)

                value_before = self.envs[i].get_portfolio_value()
                self.observations[i], _, dones_i, _ = self.vec_envs[i].step(action)
                value_after = self.envs[i].get_portfolio_value()
                returns.append(value_after / max(value_before, 1e-12) - 1.0)

            metrics = self._metrics()
            if self.mode == "hard":
                # Pick the current best agent every day (one-hot, no EMA).
                if metrics.sum() != 0.0:
                    w_raw = np.zeros(3, dtype=np.float32)
                    w_raw[int(np.argmax(metrics))] = 1.0
                else:
                    w_raw = np.full(3, 1.0 / 3.0)
            elif self.mode == "block":
                # Pick the best agent at each block boundary and hold it for
                # block_days. Ties resolve to the first index (np.argmax).
                if step % self.block_days == 0:
                    block_winner = (
                        int(np.argmax(metrics)) if metrics.sum() != 0.0 else -1
                    )
                if block_winner >= 0:
                    w_raw = np.zeros(3, dtype=np.float32)
                    w_raw[block_winner] = 1.0
                else:
                    w_raw = np.full(3, 1.0 / 3.0)
            else:
                if metrics.sum() != 0.0:
                    w_raw = np.exp(self.temperature * metrics)
                    w_raw = w_raw / w_raw.sum()
                else:
                    w_raw = np.full(3, 1.0 / 3.0)

            if self.mode == "soft":
                self.w_smoothed = (
                    self.alpha * w_raw + (1.0 - self.alpha) * self.w_smoothed
                )
            else:
                self.w_smoothed = w_raw

            for i in range(3):
                self.buffers[i].append(returns[i])

            blended = self._blend_weights(normalized_actions)
            _, _, dones_real, _, _ = self.env_real.step(blended)

            date = self.env_real._sorted_times[self.env_real._time_index]
            weight_records.append(
                {
                    "date": str(date),
                    "w_log_return": self.w_smoothed[0],
                    "w_dsr": self.w_smoothed[1],
                    "w_dsr_drawdown": self.w_smoothed[2],
                }
            )

            if dones_real:
                break

        weights_df = pd.DataFrame(weight_records)
        account_df = self.env_real.save_asset_memory()
        actions_df = self.env_real.save_action_memory()
        return account_df, actions_df, weights_df
