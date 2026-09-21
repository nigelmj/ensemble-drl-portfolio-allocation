# Uncertainty-aware ensemble of PPO agents for portfolio allocation.
#
# RQ1 — inter-seed uncertainty, with save directories under the RQ1 output roots.
from __future__ import annotations

import json
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure

from rl_portfolio.agents.portfolio_allocation_agent import (
    MODEL_KWARGS,
    PortfolioAllocationDRLAgent,
)
from rl_portfolio.utils.allocation import normalize_actions
from rl_portfolio.utils.callbacks import TensorboardCallback
from rl_portfolio.utils.io import check_and_make_directories
from rl_portfolio.envs.portfolio_allocation_env import (
    PortfolioAllocationEnv,
    set_seed,
)

from pipelines.rq1 import rq1_config


# ---------------------------------------------------------------------------
# Disagreement metric
# ---------------------------------------------------------------------------

def compute_disagreement(
    actions: np.ndarray,
    ensemble_mean: np.ndarray,
) -> float:
    """Average Total Variation Distance from ensemble mean.

    For each agent i:
        Di = sum(|ai - alloc_ensemble|) / 2
    Overall:
        D = mean(Di)

    Bounded in [0, 1] when inputs are simplex-normalized (sum to 1).
    Callers must normalize with normalize_actions before calling.
    """
    tvds = np.sum(np.abs(actions - ensemble_mean), axis=1) / 2
    return float(np.mean(tvds))


# ---------------------------------------------------------------------------
# Confidence mappings
# ---------------------------------------------------------------------------

def confidence_score(
    d: float,
    d_ref: float,
    mapping: str = "power",
    power_p: float = 2.0,
    sigmoid_k: float = 1.0,
    beta: Optional[float] = None,
    stats: Optional[dict] = None,
) -> float:
    """Map disagreement D to a confidence score in [0, 1].

    Mappings
    --------
    linear:      c = 1 - clip(D / D_ref, 0, 1)
    power:       c = 1 - clip((D / D_ref)^p, 0, 1)
    exponential: c = exp(-D / D_ref)                          if beta is None (legacy)
                 c = clip(exp(-beta * (delta - 1)), 0, 1)    if beta is not None, delta=D/D_ref, centred on delta=1
    sigmoid:     c = 1 / (1 + exp(k * (D - D_ref)))          if beta is None (legacy, k=sigmoid_k)
                 c = 1 / (1 + exp(beta * (delta - 1)))      if beta is not None, centred on delta=1
    """
    if d_ref <= 0:
        return 1.0

    ratio = d / d_ref

    if mapping == "linear":
        return 1.0 - np.clip(ratio, 0.0, 1.0)
    elif mapping == "power":
        return 1.0 - np.clip(ratio ** power_p, 0.0, 1.0)
    elif mapping == "exponential":
        if beta is None:
            return float(np.exp(-ratio))
        delta = ratio
        return float(np.clip(np.exp(-beta * (delta - 1.0)), 0.0, 1.0))
    elif mapping == "sigmoid":
        if beta is None:
            return float(1.0 / (1.0 + np.exp(sigmoid_k * (d - d_ref))))
        delta = ratio
        return float(1.0 / (1.0 + np.exp(beta * (delta - 1.0))))
    else:
        raise ValueError(f"Unknown confidence mapping: {mapping}")


# ---------------------------------------------------------------------------
# Safe allocation strategies
# ---------------------------------------------------------------------------

def safe_previous(current_weights: np.ndarray) -> np.ndarray:
    """Strategy A: carry forward previous weights."""
    return current_weights.copy()


def safe_equal_weight(portfolio_size: int) -> np.ndarray:
    """Strategy B: equal weight across cash + all stocks (1/31)."""
    return np.ones(portfolio_size + 1, dtype=np.float32) / (portfolio_size + 1)


def safe_equal_weight_stocks(portfolio_size: int) -> np.ndarray:
    """Strategy C: equal weight across stocks only, zero cash (1/30 each stock).

    Cash weight is 0.0; remaining 30 stocks share equally.
    Keeps sum to 1. Portable to env's renormalisation without fallback.
    """
    weights = np.zeros(portfolio_size + 1, dtype=np.float32)
    weights[1:] = 1.0 / portfolio_size
    return weights


def get_safe_allocation(
    strategy: str,
    current_weights: np.ndarray,
    portfolio_size: int,
    is_first: bool = False,
) -> np.ndarray:
    if strategy == "previous":
        if is_first:
            return safe_equal_weight(portfolio_size)
        return safe_previous(current_weights)
    elif strategy == "equal_weight":
        return safe_equal_weight(portfolio_size)
    elif strategy == "equal_weight_stocks":
        return safe_equal_weight_stocks(portfolio_size)
    else:
        raise ValueError(f"Unknown safe strategy: {strategy}")


# ---------------------------------------------------------------------------
# Allocation blending
# ---------------------------------------------------------------------------

def blend_allocation(
    alloc_ensemble: np.ndarray,
    alloc_safe: np.ndarray,
    confidence: float,
) -> np.ndarray:
    """Blend ensemble and safe allocations.

    alloc_final = c * alloc_ensemble + (1-c) * alloc_safe
    Weights are clipped to non-negative and re-normalised to sum to 1.
    """
    blended = confidence * alloc_ensemble + (1.0 - confidence) * alloc_safe
    blended = np.maximum(blended, 0.0)
    total = blended.sum()
    if total > 0:
        blended /= total
    else:
        blended = alloc_safe.copy()
    return blended


# ---------------------------------------------------------------------------
# D_ref calibration
# ---------------------------------------------------------------------------

def calibrate_d_ref(
    validation_disagreements: np.ndarray,
    method: str = "p90",
) -> float:
    """Compute D_ref from the validation disagreement distribution.

    Methods: mean, median, p75, p90, p95.
    """
    if method == "mean":
        return float(np.mean(validation_disagreements))
    elif method == "median":
        return float(np.median(validation_disagreements))
    elif method == "p75":
        return float(np.percentile(validation_disagreements, 75))
    elif method == "p90":
        return float(np.percentile(validation_disagreements, 90))
    elif method == "p95":
        return float(np.percentile(validation_disagreements, 95))
    else:
        raise ValueError(f"Unknown D_ref method: {method}")


# ---------------------------------------------------------------------------
# Ensemble agent
# ---------------------------------------------------------------------------

class EnsemblePPOAllocationAgent:
    """Uncertainty-aware ensemble of PPO agents for portfolio allocation."""

    def __init__(
        self,
        n_ensemble: int = 5,
        base_seed: int = 42,
        confidence_mapping: str = "power",
        safe_strategy: str = "previous",
        d_ref_method: str = "p90",
        power_p: float = 2.0,
        sigmoid_k: float = 1.0,
        beta: Optional[float] = None,
    ):
        self.n_ensemble = n_ensemble
        self.base_seed = base_seed
        self.confidence_mapping = confidence_mapping
        self.safe_strategy = safe_strategy
        self.d_ref_method = d_ref_method
        self.power_p = power_p
        self.sigmoid_k = sigmoid_k
        self.beta = beta

        self.models: List[PPO] = []
        self.d_ref: Optional[float] = None
        self.portfolio_size: Optional[int] = None
        self.disagreement_stats: Optional[dict] = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_ensemble(
        self,
        df: pd.DataFrame,
        env_kwargs: dict,
        model_kwargs: Optional[dict] = None,
        total_timesteps: int = 1_000_000,
        save_dir: str = "",
        policy_kwargs: Optional[dict] = None,
    ) -> List[PPO]:
        """Train N PPO agents with seeds base_seed .. base_seed+N-1.

        Each model is saved to ``save_dir/ppo_{i}.zip``.
        Returns the list of trained models.
        """
        if not save_dir:
            save_dir = os.path.join(rq1_config.MODEL_DIR, "rq1_ppo")
        check_and_make_directories([save_dir])

        if model_kwargs is None:
            model_kwargs = MODEL_KWARGS["ppo"].copy()

        self.portfolio_size = df["tic"].nunique()
        self.models = []

        for i in range(self.n_ensemble):
            seed = self.base_seed + i
            print(f"\n{'='*60}")
            print(f"Training ensemble member {i}/{self.n_ensemble - 1}  (seed={seed})")
            print(f"{'='*60}")

            set_seed(seed)

            env = PortfolioAllocationEnv(df=df, **env_kwargs)
            env_train, _ = env.get_sb_env()

            agent = PortfolioAllocationDRLAgent(env=env_train)
            model = agent.get_model(
                "ppo",
                model_kwargs=model_kwargs,
                policy_kwargs=policy_kwargs,
                seed=seed,
            )

            tmp_path = os.path.join(save_dir, "tb", f"ppo_{i}")
            new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
            model.set_logger(new_logger)

            model = model.learn(
                total_timesteps=total_timesteps,
                tb_log_name=f"ppo_{i}",
                callback=TensorboardCallback(),
            )

            model_path = os.path.join(save_dir, f"ppo_{i}")
            model.save(model_path)
            print(f"Saved model to {model_path}.zip")

            self.models.append(model)

        return self.models

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_ensemble(
        self,
        df: pd.DataFrame,
        env_kwargs: dict,
        save_dir: str = "",
    ) -> Tuple[float, pd.DataFrame]:
        """Run all trained models on a validation set.

        Collects per-step actions, computes disagreement, calibrates D_ref.
        Returns (d_ref, validation_df).
        """
        if not save_dir:
            save_dir = os.path.join(rq1_config.RESULTS_ROOT, "rq1_ppo")
        check_and_make_directories([save_dir])

        if not self.models:
            raise RuntimeError("No models loaded. Call train_ensemble first or load_models.")

        self.portfolio_size = df["tic"].nunique()

        env = PortfolioAllocationEnv(df=df, **env_kwargs)
        test_env, test_obs = env.get_sb_env()
        test_env.reset()

        max_steps = env.episode_length - 1
        records = []

        for step in range(max_steps + 1):
            actions = []
            for model in self.models:
                action, _ = model.predict(test_obs, deterministic=True)
                actions.append(action.flatten())

            actions_arr = np.array(actions)
            # Normalize to simplex before TVD so D is true Total Variation Distance in [0,1]
            normed = np.array([normalize_actions(a) for a in actions_arr])
            ensemble_mean = normed.mean(axis=0)
            d = compute_disagreement(normed, ensemble_mean)

            date = env._sorted_times[env._time_index] if env._time_index < len(env._sorted_times) else None

            record = {
                "step": step,
                "date": str(date) if date is not None else None,
                "disagreement": d,
            }
            for j in range(self.n_ensemble):
                for k in range(len(ensemble_mean)):
                    record[f"agent_{j}_action_{k}"] = normed[j, k]
            for k in range(len(ensemble_mean)):
                record[f"ensemble_action_{k}"] = ensemble_mean[k]

            records.append(record)

            test_obs, rewards, dones, info = test_env.step(
                ensemble_mean.reshape(1, -1).astype(np.float32)
            )
            if dones[0]:
                break

        val_df = pd.DataFrame(records)

        disagreements = val_df["disagreement"].values
        self.d_ref = calibrate_d_ref(disagreements, method=self.d_ref_method)

        self.disagreement_stats = {
            "d_mean": float(np.mean(disagreements)),
            "d_std": float(np.std(disagreements)),
            "d_min": float(np.min(disagreements)),
            "d_max": float(np.max(disagreements)),
        }

        val_df.to_csv(os.path.join(save_dir, "validation_disagreements.csv"), index=False)

        calibration = {
            "d_ref": self.d_ref,
            "d_ref_method": self.d_ref_method,
            "n_ensemble": self.n_ensemble,
            "base_seed": self.base_seed,
            "confidence_mapping": self.confidence_mapping,
            "safe_strategy": self.safe_strategy,
            "validation_mean_disagreement": self.disagreement_stats["d_mean"],
            "validation_max_disagreement": self.disagreement_stats["d_max"],
            "validation_median_disagreement": float(np.median(disagreements)),
            "disagreement_stats": self.disagreement_stats,
        }
        with open(os.path.join(save_dir, "calibration.json"), "w") as f:
            json.dump(calibration, f, indent=2)

        print(f"\nD_ref ({self.d_ref_method}): {self.d_ref:.6f}")
        print(f"Mean disagreement: {np.mean(disagreements):.6f}")
        print(f"Max disagreement:  {np.max(disagreements):.6f}")

        return self.d_ref, val_df

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_ensemble(
        self,
        df: pd.DataFrame,
        env_kwargs: dict,
        d_ref: Optional[float] = None,
        save_dir: str = "",
        use_safe: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run ensemble inference on a dataset.

        Returns (account_df, actions_df).
        """
        if not save_dir:
            save_dir = os.path.join(rq1_config.RESULTS_ROOT, "rq1_ppo")
        check_and_make_directories([save_dir])

        if not self.models:
            raise RuntimeError("No models loaded.")

        if d_ref is None:
            d_ref = self.d_ref
        if d_ref is None:
            raise RuntimeError("D_ref not set. Run validate_ensemble or provide d_ref.")

        self.portfolio_size = df["tic"].nunique()

        env = PortfolioAllocationEnv(df=df, **env_kwargs)
        test_env, test_obs = env.get_sb_env()
        test_env.reset()

        max_steps = env.episode_length - 1
        current_weights = env.current_weights.copy()

        account_records = []
        action_records = []

        for step in range(max_steps + 1):
            actions = []
            for model in self.models:
                action, _ = model.predict(test_obs, deterministic=True)
                actions.append(action.flatten())

            actions_arr = np.array(actions)
            # Normalize to simplex before TVD so D is true Total Variation Distance in [0,1]
            normed = np.array([normalize_actions(a) for a in actions_arr])
            alloc_ensemble = normed.mean(axis=0)
            d = compute_disagreement(normed, alloc_ensemble)
            c = confidence_score(
                d, d_ref,
                mapping=self.confidence_mapping,
                power_p=self.power_p,
                sigmoid_k=self.sigmoid_k,
                beta=self.beta,
                stats=self.disagreement_stats,
            )

            if use_safe:
                alloc_safe = get_safe_allocation(
                    self.safe_strategy, current_weights, self.portfolio_size,
                    is_first=(step == 0),
                )
                alloc_final = blend_allocation(alloc_ensemble, alloc_safe, c)
            else:
                alloc_final = alloc_ensemble.copy()
                alloc_safe = np.zeros_like(alloc_final)
                c = 1.0

            date = env._sorted_times[env._time_index] if env._time_index < len(env._sorted_times) else None

            account_records.append({
                "date": str(date) if date is not None else None,
                "portfolio_value": env.portfolio_value,
                "disagreement": d,
                "confidence": c,
            })

            action_record = {
                "date": str(date) if date is not None else None,
                "disagreement": d,
                "confidence": c,
                "confidence_mapping": self.confidence_mapping,
                "safe_strategy": self.safe_strategy,
            }
            for k in range(len(alloc_ensemble)):
                action_record[f"ensemble_alloc_{k}"] = alloc_ensemble[k]
                action_record[f"safe_alloc_{k}"] = alloc_safe[k]
                action_record[f"final_alloc_{k}"] = alloc_final[k]
            action_records.append(action_record)

            test_obs, rewards, dones, info = test_env.step(
                alloc_final.reshape(1, -1).astype(np.float32)
            )

            current_weights = env.current_weights.copy()

            if dones[0]:
                break

        account_df = pd.DataFrame(account_records)
        actions_df = pd.DataFrame(action_records)

        account_df.to_csv(os.path.join(save_dir, "test_account.csv"), index=False)
        actions_df.to_csv(os.path.join(save_dir, "test_actions.csv"), index=False)

        return account_df, actions_df

    # ------------------------------------------------------------------
    # Model I/O
    # ------------------------------------------------------------------

    def load_models(self, load_dir: str = "") -> List[PPO]:
        """Load N PPO models from ``load_dir/ppo_{i}.zip``."""
        if not load_dir:
            load_dir = os.path.join(rq1_config.MODEL_DIR, "rq1_ppo")
        self.models = []
        for i in range(self.n_ensemble):
            path = os.path.join(load_dir, f"ppo_{i}")
            model = PPO.load(path)
            self.models.append(model)
            print(f"Loaded model from {path}")
        return self.models

    def load_calibration(self, calibration_path: str) -> dict:
        """Load D_ref and config from a calibration.json file."""
        with open(calibration_path) as f:
            cal = json.load(f)
        self.d_ref = cal["d_ref"]
        self.d_ref_method = cal["d_ref_method"]
        self.confidence_mapping = cal.get("confidence_mapping", self.confidence_mapping)
        self.safe_strategy = cal.get("safe_strategy", self.safe_strategy)
        self.disagreement_stats = cal.get("disagreement_stats")
        return cal
