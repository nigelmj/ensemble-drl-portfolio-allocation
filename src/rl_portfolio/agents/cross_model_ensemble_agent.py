"""
RQ3 algorithm-diversity ensemble for the PortfolioAllocationEnv.

Performance-weighted blend over *different* Stable Baselines 3 algorithms
(PPO / SAC / A2C), each trained with its own reward_type (e.g. ppo=log_return,
sac=dsr, a2c=dsr_drawdown).

Each member trades its own isolated tracking environment; a trailing
risk-adjusted metric ranks the members and the blend weights are computed per
mode (soft / hard / block). A final real env trades the blend.
"""
from __future__ import annotations

import os
from collections import deque
from typing import List, Tuple

import numpy as np
import pandas as pd

from rl_portfolio.agents.portfolio_allocation_agent import MODELS
from rl_portfolio.envs.portfolio_allocation_env import PortfolioAllocationEnv
from rl_portfolio.utils.allocation import normalize_actions


class _BaseEnsembleAgent:
    """Shared constructor/loading for a heterogeneous member ensemble.

    member_specs: list of dicts, one per member:
        {"algo": "ppo"|"sac"|"a2c", "reward_type": "...", "path": "..."}
    """

    def __init__(self, member_specs: List[dict]):
        if not member_specs:
            raise ValueError("member_specs cannot be empty.")
        self.member_specs = member_specs
        self.algos = [m["algo"] for m in member_specs]
        self.reward_types = [m["reward_type"] for m in member_specs]
        self.models = []
        for m in member_specs:
            if m["algo"] not in MODELS:
                raise ValueError(f"Unknown algo '{m['algo']}'. Got {list(MODELS)}")
            if not os.path.exists(m["path"] + ".zip"):
                raise FileNotFoundError(f"Model not found: {m['path']}.zip")
            self.models.append(MODELS[m["algo"]].load(m["path"]))

    def _build_env_kwargs(self, base_env_kwargs: dict, reward_type: str) -> dict:
        kwargs = dict(base_env_kwargs)
        kwargs["reward_type"] = reward_type
        return kwargs


class CrossModelPerfWeightedAgent(_BaseEnsembleAgent):
    """Performance-weighted blend of heterogeneous algorithms with different rewards.

    Each member trades its own isolated env; a trailing mean/std metric ranks
    members and weights are derived per mode (soft/hard/block). A final real
    env trades the blend.
    """

    def __init__(
        self,
        member_specs: List[dict],
        k: int = 20,
        temperature: float = 1.0,
        alpha: float = 0.3,
        mode: str = "soft",
        block_days: int = 20,
    ):
        super().__init__(member_specs)
        if mode not in ("soft", "hard", "block"):
            raise ValueError(f"Unknown mode '{mode}'; expected soft/hard/block")
        self.k = k
        self.temperature = temperature
        self.alpha = alpha
        self.mode = mode
        self.block_days = block_days

        self.envs = []
        self.vec_envs = []
        self.observations = []
        self.env_real = None
        self.portfolio_size = None
        self.episode_length = None
        self.buffers = None
        self.w_smoothed = None

    def _setup_envs(self, df: pd.DataFrame, base_env_kwargs: dict):
        self.envs = [
            PortfolioAllocationEnv(
                df=df, **self._build_env_kwargs(base_env_kwargs, rt)
            )
            for rt in self.reward_types
        ]
        self.vec_envs, self.observations = [], []
        for e in self.envs:
            v, o = e.get_sb_env()
            self.vec_envs.append(v)
            self.observations.append(o)
        self.env_real = PortfolioAllocationEnv(df=df, **base_env_kwargs)
        self.portfolio_size = self.env_real.portfolio_size
        self.episode_length = self.env_real.episode_length
        self.buffers = [deque(maxlen=self.k) for _ in self.envs]
        self.w_smoothed = np.full(len(self.envs), 1.0 / len(self.envs))

    def _metrics(self) -> np.ndarray:
        metrics = np.zeros(len(self.buffers), dtype=np.float32)
        for i, buf in enumerate(self.buffers):
            if len(buf) < self.k:
                return np.zeros(len(self.buffers), dtype=np.float32)
            arr = np.asarray(buf, dtype=np.float32)
            metrics[i] = arr.mean() / max(arr.std(), 1e-12)
        return metrics

    def _blend_weights(self, normalized_actions: List[np.ndarray]) -> np.ndarray:
        combined = np.zeros_like(normalized_actions[0], dtype=np.float32)
        for w, norm in zip(self.w_smoothed, normalized_actions):
            combined += w * norm
        return normalize_actions(combined)

    def run(
        self,
        df: pd.DataFrame,
        base_env_kwargs: dict,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Blend on df. Returns (account, actions, weights)."""
        self._setup_envs(df, base_env_kwargs)
        weight_records, block_winner = [], -1

        for step in range(self.episode_length):
            returns, normalized_actions = [], []
            for i in range(len(self.envs)):
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
            n = len(self.envs)
            if self.mode == "hard":
                if metrics.sum() != 0.0:
                    w_raw = np.zeros(n, dtype=np.float32)
                    w_raw[int(np.argmax(metrics))] = 1.0
                else:
                    w_raw = np.full(n, 1.0 / n)
            elif self.mode == "block":
                if step % self.block_days == 0:
                    block_winner = int(np.argmax(metrics)) if metrics.sum() != 0.0 else -1
                if block_winner >= 0:
                    w_raw = np.zeros(n, dtype=np.float32)
                    w_raw[block_winner] = 1.0
                else:
                    w_raw = np.full(n, 1.0 / n)
            else:
                if metrics.sum() != 0.0:
                    w_raw = np.exp(self.temperature * metrics)
                    w_raw = w_raw / w_raw.sum()
                else:
                    w_raw = np.full(n, 1.0 / n)

            if self.mode == "soft":
                self.w_smoothed = (
                    self.alpha * w_raw + (1.0 - self.alpha) * self.w_smoothed
                )
            else:
                self.w_smoothed = w_raw

            for i in range(n):
                self.buffers[i].append(returns[i])

            blended = self._blend_weights(normalized_actions)
            _, _, dones_real, _, _ = self.env_real.step(blended)
            date = self.env_real._sorted_times[self.env_real._time_index]
            weight_records.append({
                "date": str(date),
                **{f"w_{self.reward_types[i]}": self.w_smoothed[i]
                   for i in range(n)},
            })
            if dones_real:
                break

        return (
            self.env_real.save_asset_memory(),
            self.env_real.save_action_memory(),
            pd.DataFrame(weight_records),
        )
