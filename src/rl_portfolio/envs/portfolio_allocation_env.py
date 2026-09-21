from __future__ import annotations

import random
from typing import List, Optional

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from gymnasium import spaces
from gymnasium.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv

from rl_portfolio.config import TIME_WINDOW
from rl_portfolio.utils.allocation import normalize_actions


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PortfolioAllocationEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        initial_amount: float,
        time_window: int = TIME_WINDOW,
        fee_pct: float = 0.00,
        reward_scaling: float = 1,
        reward_type: str = "log_return",
        dsr_eta: float = 1 / 60,
        dsr_eta_dd: float = 0.0,
        reward_scaling_dsr: float = 1.0,
        time_column: str = "date",
        tic_column: str = "tic",
        use_market_features: bool = True,
        market_feature_cols: Optional[List[str]] = None,
        print_verbosity: int = 10,
    ):
        super().__init__()

        if reward_type not in ("log_return", "dsr", "dsr_drawdown"):
            raise ValueError(
                f"Unknown reward_type '{reward_type}'. "
                f"Expected one of: log_return, dsr, dsr_drawdown"
            )

        self._df = df
        self._initial_amount = initial_amount
        self._time_window = time_window
        self._fee_pct = fee_pct
        self._reward_scaling = reward_scaling
        self._reward_type = reward_type
        self._dsr_eta = dsr_eta
        self._dsr_eta_dd = dsr_eta_dd
        self._reward_scaling_dsr = reward_scaling_dsr
        self._time_column = time_column
        self._tic_column = tic_column
        self._use_market_features = use_market_features
        self._market_feature_cols = market_feature_cols or ["vix", "vol20", "vol20d60"]
        self.print_verbosity = print_verbosity
        self.episode = 0
        self._total_cost = 0.0

        self._preprocess_data()

        self._tic_list = list(self._df[self._tic_column].unique())
        self.portfolio_size = len(self._tic_list)

        self.action_space = spaces.Box(
            low=0, high=1, shape=(self.portfolio_size + 1,), dtype=np.float32
        )

        self._n_market_feats = (
            len(self._market_feature_cols) if self._use_market_features else 0
        )
        obs_cols = 1 + self._time_window + self._n_market_feats
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.portfolio_size + 1, obs_cols),
            dtype=np.float32,
        )

        self._sorted_times = sorted(set(self._df[self._time_column]))
        self._n_times = len(self._sorted_times)

        if self._n_times < self._time_window + 2:
            raise ValueError(
                f"Dataset has {self._n_times} time steps, but need at least "
                f"{self._time_window + 2} for time_window={self._time_window} "
                f"(time_window + 2 to build the initial state and take one step)"
            )

        self.episode_length = self._n_times - self._time_window
        self._time_index = self._time_window

        self._seed()
        self._reset_memory()

        self.current_weights = np.array(
            [1.0] + [0.0] * self.portfolio_size, dtype=np.float32
        )
        self.portfolio_value = self._initial_amount
        self._terminal = False

        self._dsr_A = 0.0
        self._dsr_B = 1e-8
        self._peak_value = self._initial_amount
        self._dd_prev = 0.0

    def step(self, actions):
        self._terminal = self._time_index >= self._n_times - 1

        if self._terminal:
            if self.episode % self.print_verbosity == 0:
                self._print_episode_summary()
            return self._state, self._reward, self._terminal, False, self._info

        target_weights = normalize_actions(actions)

        old_weights = self.current_weights.copy()
        old_portfolio_value = self.portfolio_value

        turnover = np.sum(np.abs(target_weights - old_weights))
        cost = turnover * self.portfolio_value * self._fee_pct
        self._total_cost += cost
        self._turnover_memory.append(turnover)
        self.portfolio_value -= cost

        price_ratios = self._get_price_ratios(self._time_index + 1)
        portfolio_return = np.sum(target_weights * price_ratios)
        self.portfolio_value *= portfolio_return
        portfolio_return = float(portfolio_return)
        resulting_weights = (target_weights * price_ratios) / max(portfolio_return, 1e-8)

        rate_of_return = self.portfolio_value / old_portfolio_value
        rate_of_return = max(rate_of_return, 1e-8)
        self._reward = self._compute_reward(rate_of_return)

        self.current_weights = resulting_weights
        self._actions_memory.append(target_weights)
        self._final_weights.append(resulting_weights)
        self._portfolio_return_memory.append(rate_of_return - 1.0)
        self._portfolio_reward_memory.append(np.log(rate_of_return))
        self._asset_memory["initial"].append(old_portfolio_value)
        self._asset_memory["final"].append(self.portfolio_value)

        self._time_index += 1
        self._date_memory.append(self._sorted_times[self._time_index])

        self._state, self._info = self._build_state()

        return self._state, self._reward, self._terminal, False, self._info

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._seed(seed)

        self.episode += 1
        self._total_cost = 0.0
        self._time_index = self._time_window
        self.portfolio_value = self._initial_amount
        self.current_weights = np.array(
            [1.0] + [0.0] * self.portfolio_size, dtype=np.float32
        )
        self._terminal = False
        self._dsr_A = 0.0
        self._dsr_B = 1e-8
        self._peak_value = self._initial_amount
        self._dd_prev = 0.0
        self._reset_memory()
        self._state, self._info = self._build_state()
        return self._state, self._info

    def render(self, mode="human"):
        return self._state

    def get_portfolio_value(self):
        return self.portfolio_value

    def get_portfolio_weights(self):
        return self.current_weights.copy()

    def enumerate_portfolio(self):
        print("Index: 0. Tic: Cash")
        for i, tic in enumerate(self._tic_list):
            print(f"Index: {i + 1}. Tic: {tic}")

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs

    def save_asset_memory(self):
        df_account_value = pd.DataFrame(
            {
                "date": self._date_memory,
                "portfolio_value": self._asset_memory["final"],
            }
        )
        return df_account_value

    def save_action_memory(self):
        df_actions = pd.DataFrame(
            self._actions_memory, columns=["cash"] + self._tic_list
        )
        df_actions.index = self._date_memory
        df_actions.index.name = "date"
        return df_actions

    def _print_episode_summary(self):
        begin_total_asset = self._initial_amount
        end_total_asset = self.portfolio_value
        tot_reward = end_total_asset - begin_total_asset

        action_cash_weights = np.array(
            [w[0] for w in self._actions_memory[1:]], dtype=np.float32
        )
        average_cash_weight = (
            float(np.mean(action_cash_weights)) if len(action_cash_weights) > 0 else 0.0
        )
        average_invested_weight = 1 - average_cash_weight

        stats = self.get_episode_stats()

        df_daily_return = pd.DataFrame(self._portfolio_return_memory[1:])
        df_daily_return.columns = ["daily_return"]
        if df_daily_return["daily_return"].std() != 0:
            sharpe = (
                (252**0.5)
                * df_daily_return["daily_return"].mean()
                / df_daily_return["daily_return"].std()
            )

        print(f"day: {self._time_index}, episode: {self.episode}")
        print(f"begin_total_asset: {begin_total_asset:0.2f}")
        print(f"end_total_asset: {end_total_asset:0.2f}")
        print(f"total_reward: {tot_reward:0.2f}")
        print(f"total_cost: {self._total_cost:0.2f}")
        print(f"average_turnover: {stats['average_turnover']:0.4f}")
        print(f"cost_pct_of_initial: {stats['cost_pct']:0.2f}%")
        print(f"average_cash_weight: {average_cash_weight:0.4f}")
        print(f"average_invested_weight: {average_invested_weight:0.4f}")
        if df_daily_return["daily_return"].std() != 0:
            print(f"Sharpe: {sharpe:0.3f}")
        print("=================================")

    def get_episode_stats(self):
        turnover = np.array(self._turnover_memory, dtype=np.float32)
        average_turnover = (
            float(turnover.mean()) if len(turnover) > 0 else 0.0
        )
        total_cost = self._total_cost
        cost_pct = 100 * total_cost / self._initial_amount
        return {
            "average_turnover": average_turnover,
            "total_cost": total_cost,
            "cost_pct": cost_pct,
        }

    def _compute_reward(self, rate_of_return: float) -> float:
        """Compute the per-step reward for the configured reward_type.

        ``log_return`` (default) is unchanged: log(V_t / V_{t-1}) * reward_scaling,
        where the rate of return is net of transaction costs.

        ``dsr`` uses the differential Sharpe ratio increment (DeepRisk / Deep Stock
        Ratio). The estimate uses the PRE-update EMA values A_{t-1}, B_{t-1}:
            r_t   = V_t / V_{t-1} - 1
            DSR_t = (B_{t-1}(r_t - A_{t-1}) - 0.5*A_{t-1}(r_t**2 - B_{t-1}))
                    / max(B_{t-1} - A_{t-1}**2, eps)**1.5
        then the EMAs advance:
            A_t = A_{t-1} + eta*(r_t - A_{t-1})
            B_t = B_{t-1} + eta*(r_t**2 - B_{t-1})
        The sign of the -0.5*A term follows the standard derivation; verify
        empirically that the reward rises when risk-adjusted performance improves.

        ``dsr_drawdown`` adds a penalty on the squared increase in drawdown:
            DD_t = (peak_so_far - V_t) / peak_so_far
            reward = (DSR_t - eta_dd * max(0, DD_t - DD_{t-1})**2) * reward_scaling_dsr
        """
        if self._reward_type == "log_return":
            return float(np.log(rate_of_return) * self._reward_scaling)

        r_t = rate_of_return - 1.0
        a_prev = self._dsr_A
        b_prev = self._dsr_B
        var = b_prev - a_prev**2
        denominator = max(var, 1e-12) ** 1.5
        dsr = (
            b_prev * (r_t - a_prev)
            - 0.5 * a_prev * (r_t**2 - b_prev)
        ) / denominator

        self._dsr_A = a_prev + self._dsr_eta * (r_t - a_prev)
        self._dsr_B = b_prev + self._dsr_eta * (r_t**2 - b_prev)

        if self._reward_type == "dsr":
            return float(dsr * self._reward_scaling_dsr)

        self._peak_value = max(self._peak_value, self.portfolio_value)
        dd = (self._peak_value - self.portfolio_value) / self._peak_value
        penalty = max(0.0, dd - self._dd_prev) ** 2
        self._dd_prev = dd
        return float((dsr - self._dsr_eta_dd * penalty) * self._reward_scaling_dsr)

    def _preprocess_data(self):
        self._df = self._df.sort_values(
            by=[self._tic_column, self._time_column]
        ).reset_index(drop=True)

        self._df[self._time_column] = pd.to_datetime(self._df[self._time_column])
        self._df["close"] = self._df["close"].astype(np.float32)

        if self._use_market_features:
            for col in self._market_feature_cols:
                if col not in self._df.columns:
                    raise ValueError(
                        f"Market feature column '{col}' not found in DataFrame. "
                        f"Available columns: {list(self._df.columns)}"
                    )
                self._df[col] = self._df[col].astype(np.float32)

        self._sorted_times = sorted(set(self._df[self._time_column]))
        self._tic_list = list(self._df[self._tic_column].unique())
        n_stocks = len(self._tic_list)
        n_times = len(self._sorted_times)

        time_to_idx = {t: j for j, t in enumerate(self._sorted_times)}

        close = np.zeros((n_stocks, n_times), dtype=np.float32)
        for i, tic in enumerate(self._tic_list):
            tic_mask = self._df[self._tic_column] == tic
            tic_df = self._df[tic_mask]
            for _, row in tic_df.iterrows():
                j = time_to_idx[row[self._time_column]]
                close[i, j] = row["close"]

        # Forward-fill zeros (missing data, e.g. IPO dates)
        for i in range(n_stocks):
            last_valid = 0.0
            for j in range(n_times):
                if close[i, j] > 0:
                    last_valid = close[i, j]
                else:
                    close[i, j] = last_valid

        self._close_array = close

        log_ret = np.zeros_like(close)
        safe_cur = np.where(close[:, 1:] > 0, close[:, 1:], 1.0)
        safe_prev = np.where(close[:, :-1] > 0, close[:, :-1], 1.0)
        log_ret[:, 1:] = np.where(
            (close[:, 1:] > 0) & (close[:, :-1] > 0),
            np.log(safe_cur / safe_prev),
            0.0,
        )
        self._log_return_array = log_ret

        price_ratio = np.ones_like(close)
        price_ratio[:, 1:] = np.where(
            (close[:, 1:] > 0) & (close[:, :-1] > 0),
            close[:, 1:] / safe_prev,
            1.0,
        )
        self._price_ratio_array = price_ratio

        if self._use_market_features:
            n_feats = len(self._market_feature_cols)
            mkt = np.zeros((n_times, n_feats), dtype=np.float32)
            for j, t in enumerate(self._sorted_times):
                rows = self._df[self._df[self._time_column] == t]
                if len(rows) > 0:
                    for k, col in enumerate(self._market_feature_cols):
                        vals = rows[col].unique()
                        mkt[j, k] = vals[0] if len(vals) > 0 else 0.0
            self._market_feature_array = mkt

    def _build_state(self):
        i = self._time_index
        t = self._time_window
        n = self.portfolio_size
        n_cols = 1 + t + self._n_market_feats

        state = np.zeros((n + 1, n_cols), dtype=np.float32)

        state[:, 0] = self.current_weights

        state[0, 1 : t + 1] = 0.0

        start = i - t + 1
        end = i + 1
        log_slice = self._log_return_array[:, start:end]
        log_slice = np.flip(log_slice, axis=1)
        state[1:, 1 : t + 1] = log_slice

        if self._use_market_features:
            state[:, t + 1 :] = self._market_feature_array[i]

        info = {
            "tics": list(self._tic_list),
            "time_index": i,
            "end_time": self._sorted_times[i],
        }
        return state, info

    def _reset_memory(self):
        self._asset_memory = {
            "initial": [self._initial_amount],
            "final": [self._initial_amount],
        }
        self._turnover_memory = []
        self._portfolio_return_memory = [0.0]
        self._portfolio_reward_memory = [0.0]
        init_w = np.array([1.0] + [0.0] * self.portfolio_size, dtype=np.float32)
        self._actions_memory = [init_w]
        self._final_weights = [init_w]
        self._date_memory = [self._sorted_times[self._time_index]]

    def _get_price_ratios(self, time_index: int) -> np.ndarray:
        ratios = np.ones(self.portfolio_size + 1, dtype=np.float32)
        ratios[1:] = self._price_ratio_array[:, time_index]
        return ratios

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
