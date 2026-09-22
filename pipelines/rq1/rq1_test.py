"""
RQ1 — Test the 8 confidence-mapping × safe-strategy combinations.

Runs the trained inter-seed ensemble on the shared test split (2022-2026) for
every combination of {linear, power, exponential, sigmoid} × {previous,
equal_weight}, saving per-combo account/actions, plus the single PPO baseline.

The compute-matched 5M single (ppo_single_5m.zip) is kept separate from the
legacy 1M single (ppo_single.zip). Use --single-model ppo_single_5m (or
--only-single) to evaluate the 5M control without overwriting/mixing the 1M
results: outputs become account_value_single_ppo_5m.csv etc.

Usage:
    python -m pipelines.rq1.rq1_train
    python -m pipelines.rq1.rq1_validate
    python -m pipelines.rq1.rq1_test
    python -m pipelines.rq1.rq1_test --only-single --single-model ppo_single_5m
    python -m pipelines.rq1.rq1_test --single-model both   # both baselines in one summary (mixes)

Output:
    results/rq1/combo_{mapping}_{safe}/test_account.csv
    results/rq1/combo_{mapping}_{safe}/test_actions.csv
    results/rq1/account_value_single_ppo.csv         (1M legacy)
    results/rq1/account_value_single_ppo_5m.csv      (5M compute-matched)
    results/rq1/actions_single_ppo.csv / actions_single_ppo_5m.csv
    results/rq1/test_summary.csv
    results/rq1/test_summary_single_5m.csv           (when --only-single + 5M)
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from pipelines.rq1.rq1_alloc import EnsemblePPOAllocationAgent
from pipelines.rq1 import rq1_config
from rl_portfolio.agents.portfolio_allocation_agent import (
    PortfolioAllocationDRLAgent,
)
from rl_portfolio.config import RESULTS_DIR
from rl_portfolio.utils.io import check_and_make_directories
from rl_portfolio.envs.portfolio_allocation_env import PortfolioAllocationEnv

import os
import shutil


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tag",
        default="",
        help="Optional suffix matching the trained-model tag.",
    )
    parser.add_argument(
        "--results_root",
        default=rq1_config.RESULTS_ROOT,
        help="Output directory for accounts/actions/summary. "
             "Use results/temp/rq1 for stocks-only ablation.",
    )
    parser.add_argument(
        "--calibration_path",
        default="",
        help="Path to calibration_all.json. Defaults to <results_root>/calibration_all.json "
             "with fallback to rq1_config.RESULTS_ROOT/calibration_all.json.",
    )
    parser.add_argument(
        "--single-model",
        choices=["ppo_single", "ppo_single_5m", "both"],
        default="ppo_single",
        help="Which single-PPO checkpoint(s) to evaluate (default: ppo_single). "
             "Use ppo_single_5m for the 5M compute-matched control, or 'both' to "
             "evaluate both (adds Single PPO 5M row to test_summary.csv).",
    )
    parser.add_argument(
        "--only-single",
        action="store_true",
        help="Only evaluate the single PPO baseline(s), skip ensemble baseline + combos. "
             "Useful for the 5M control: --only-single --single-model ppo_single_5m "
             "writes account_value_single_ppo_5m.csv without touching combos.",
    )
    parser.add_argument(
        "--skip-combos",
        action="store_true",
        help="Skip confidence-mapping combos (ensemble average baseline still runs).",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip the ensemble-average baseline.",
    )
    return parser.parse_args()


def compute_metrics(account_df: pd.DataFrame, label: str) -> dict:
    """Compute Sharpe, max drawdown, total return, etc."""
    vals = account_df["portfolio_value"].values
    total_return = vals[-1] / vals[0] - 1
    daily_returns = np.diff(vals) / vals[:-1]
    if np.std(daily_returns) > 0:
        sharpe = np.sqrt(252) * np.mean(daily_returns) / np.std(daily_returns)
    else:
        sharpe = np.inf if np.mean(daily_returns) > 0 else 0.0

    peak = np.maximum.accumulate(vals)
    drawdowns = (vals - peak) / peak
    max_drawdown = float(np.min(drawdowns))

    return {
        "label": label,
        "final_value": float(vals[-1]),
        "total_return": float(total_return),
        "sharpe": float(sharpe),
        "max_drawdown": max_drawdown,
        "avg_daily_return": float(np.mean(daily_returns)),
        "volatility": float(np.std(daily_returns)),
    }


def _cash_diagnostics(actions_df: pd.DataFrame) -> dict:
    """Compute cash-weight diagnostics from an actions dataframe.

    Cash is always index 0 (k=0). Handles all three cases:
      - combo runs: ensemble_alloc_0, safe_alloc_0, final_alloc_0
      - baseline/single: only ensemble_alloc_0 (final == ensemble)
    """
    diag = {}
    if "ensemble_alloc_0" in actions_df.columns:
        diag["mean_ensemble_cash"] = float(actions_df["ensemble_alloc_0"].mean())
        diag["median_ensemble_cash"] = float(actions_df["ensemble_alloc_0"].median())
    else:
        diag["mean_ensemble_cash"] = np.nan
        diag["median_ensemble_cash"] = np.nan
    if "safe_alloc_0" in actions_df.columns:
        diag["mean_safe_cash"] = float(actions_df["safe_alloc_0"].mean())
        diag["median_safe_cash"] = float(actions_df["safe_alloc_0"].median())
        # For equal_weight 1/31 ~0.0322, for equal_weight_stocks 0.0, for previous ~drifted
        diag["min_safe_cash"] = float(actions_df["safe_alloc_0"].min())
        diag["max_safe_cash"] = float(actions_df["safe_alloc_0"].max())
    else:
        diag["mean_safe_cash"] = np.nan
        diag["median_safe_cash"] = np.nan
        diag["min_safe_cash"] = np.nan
        diag["max_safe_cash"] = np.nan
    if "final_alloc_0" in actions_df.columns:
        diag["mean_final_cash"] = float(actions_df["final_alloc_0"].mean())
        diag["median_final_cash"] = float(actions_df["final_alloc_0"].median())
        diag["min_final_cash"] = float(actions_df["final_alloc_0"].min())
        diag["max_final_cash"] = float(actions_df["final_alloc_0"].max())
    elif "ensemble_alloc_0" in actions_df.columns:
        # Baseline / no-safe: final == ensemble
        diag["mean_final_cash"] = float(actions_df["ensemble_alloc_0"].mean())
        diag["median_final_cash"] = float(actions_df["ensemble_alloc_0"].median())
        diag["min_final_cash"] = float(actions_df["ensemble_alloc_0"].min())
        diag["max_final_cash"] = float(actions_df["ensemble_alloc_0"].max())
    else:
        diag["mean_final_cash"] = np.nan
        diag["median_final_cash"] = np.nan
        diag["min_final_cash"] = np.nan
        diag["max_final_cash"] = np.nan
    return diag


def main():
    args = parse_args()
    suffix = f"_{args.tag}" if args.tag else ""
    results_root = args.results_root
    # Calibration fallback: temp run reuses rq1 calibration
    if args.calibration_path:
        calibration_path = args.calibration_path
    else:
        candidate = os.path.join(results_root, "calibration_all.json")
        if os.path.exists(candidate):
            calibration_path = candidate
        else:
            calibration_path = os.path.join(rq1_config.RESULTS_ROOT, "calibration_all.json")
    check_and_make_directories([RESULTS_DIR, results_root])
    # If using temp root, ensure calibration is also present there for the run record
    if results_root != rq1_config.RESULTS_ROOT and not os.path.exists(os.path.join(results_root, "calibration_all.json")):
        try:
            if os.path.exists(calibration_path):
                shutil.copy(calibration_path, os.path.join(results_root, "calibration_all.json"))
                print(f"Copied calibration to {results_root}/calibration_all.json")
        except Exception as e:
            print(f"Warning: could not copy calibration to temp root: {e}")

    def _eval_single(model_name: str, label: str, acct_path: str, act_path: str) -> dict | None:
        """Load one single-PPO checkpoint, run DRL_prediction, save, return metrics."""
        model_path = f"{rq1_config.MODEL_DIR}{suffix}/{model_name}"
        if not os.path.exists(model_path + ".zip"):
            print(f"[skip] {label}: missing {model_path}.zip")
            return None
        print("\n" + "=" * 60)
        print(f"Running {label} ({model_name}) on test set...")
        print("=" * 60)
        m = PPO.load(model_path)
        single_env = PortfolioAllocationEnv(df=test, **rq1_config.ENV_KWARGS)
        account_single, actions_single, _ = PortfolioAllocationDRLAgent.DRL_prediction(
            model=m, environment=single_env, deterministic=True,
        )
        account_single.to_csv(acct_path, index=False)
        actions_single.to_csv(act_path, index=False)
        print(f"Saved {acct_path} ({len(account_single)} rows)")
        print(f"Saved {act_path} ({len(actions_single)} rows)")
        metrics = compute_metrics(account_single, label)
        try:
            if "cash" in actions_single.columns:
                avg_cash = float(actions_single["cash"].mean())
                metrics["mean_ensemble_cash"] = avg_cash
                metrics["mean_safe_cash"] = np.nan
                metrics["mean_final_cash"] = avg_cash
                metrics["median_ensemble_cash"] = float(actions_single["cash"].median())
                metrics["median_final_cash"] = float(actions_single["cash"].median())
            else:
                metrics.update(_cash_diagnostics(actions_single))
        except (KeyError, ValueError, TypeError) as exc:
            print(f"[warn] cash diagnostics unavailable for {label}: {exc}")
        # annotate run record
        metrics["model_name"] = model_name
        print(f"  Sharpe={metrics['sharpe']:.3f}  "
              f"Return={metrics['total_return']:.2%}  MaxDD={metrics['max_drawdown']:.2%} "
              f"Cash={metrics.get('mean_final_cash', float('nan')):.4f}")
        return metrics

    results = []

    test = pd.read_pickle(rq1_config.DATA_TEST)
    print(f"Test: {len(test)} rows, {test.date.nunique()} days, "
          f"{test.tic.nunique()} tickers")
    print(f"Results root: {results_root}")
    print(f"Calibration:  {calibration_path}")
    print(f"Single-model: {args.single_model}  only_single={args.only_single}")

    cal = json.load(
        open(calibration_path)
    )
    d_ref = cal["calibrations"][rq1_config.D_REF_METHOD]["d_ref"]
    disagreement_stats = cal["disagreement_stats"]
    print(f"\nD_ref ({rq1_config.D_REF_METHOD}): {d_ref:.6f}")

    # --only-single short-circuit: don't touch ensemble combos/baseline; isolated outputs
    if args.only_single:
        single_results = []
        if args.single_model in ("ppo_single", "both"):
            r = _eval_single("ppo_single", "Single PPO",
                             f"{results_root}/account_value_single_ppo.csv",
                             f"{results_root}/actions_single_ppo.csv")
            if r is not None:
                single_results.append(r)
        if args.single_model in ("ppo_single_5m", "both"):
            r = _eval_single("ppo_single_5m", "Single PPO",
                             f"{results_root}/account_value_single_ppo_5m.csv",
                             f"{results_root}/actions_single_ppo_5m.csv")
            if r is not None:
                single_results.append(r)
        if single_results:
            df = pd.DataFrame(single_results)
            # isolated summary so canonical test_summary.csv stays untouched
            suffix_tag = "5m" if args.single_model == "ppo_single_5m" else "single"
            out = os.path.join(results_root, f"test_summary_single_{suffix_tag}.csv")
            # if 'both', name accordingly
            if args.single_model == "both":
                out = os.path.join(results_root, "test_summary_single_both.csv")
            df.to_csv(out, index=False)
            print(f"\nSingle-only summary saved to {out}")
            print(df[["label","total_return","sharpe","max_drawdown","mean_final_cash"]].round(4).to_string(index=False))
        else:
            print("No single model evaluated (missing .zip?).")
        return

    # ---- Baseline: ensemble average, no safe, no confidence ----
    if not args.skip_baseline:
        label = "baseline_ensemble_average"
        out_dir = f"{results_root}/{label}"
        check_and_make_directories([out_dir])
        print("\n" + "=" * 60)
        print(f"Running baseline: ensemble average (no safe, no confidence)")
        print("=" * 60)

        agent = EnsemblePPOAllocationAgent(
            n_ensemble=rq1_config.N_ENSEMBLE,
            base_seed=rq1_config.BASE_SEED,
            confidence_mapping="power",
            safe_strategy="previous",
            d_ref_method=rq1_config.D_REF_METHOD,
            power_p=rq1_config.POWER_P,
            sigmoid_k=rq1_config.SIGMOID_K,
        )
        agent.load_models(f"{rq1_config.MODEL_DIR}{suffix}")
        agent.d_ref = d_ref
        agent.disagreement_stats = disagreement_stats
        agent.portfolio_size = test["tic"].nunique()

        env = PortfolioAllocationEnv(df=test, **rq1_config.ENV_KWARGS)
        test_env, test_obs = env.get_sb_env()
        test_env.reset()

        max_steps = env.episode_length - 1
        current_weights = env.current_weights.copy()

        account_records = []
        action_records = []

        for step in range(max_steps + 1):
            actions = []
            for model in agent.models:
                action, _ = model.predict(test_obs, deterministic=True)
                actions.append(action.flatten())

            actions_arr = np.array(actions)
            # Normalize to simplex for true ensemble average (as with TVD)
            normed = np.array([a / a.sum() if a.sum() >= 1e-8 else np.ones_like(a)/len(a) for a in actions_arr])
            alloc_ensemble = normed.mean(axis=0)

            # Use ensemble mean directly; no safe blending, no confidence
            alloc_final = alloc_ensemble.copy()

            date = env._sorted_times[env._time_index] if env._time_index < len(env._sorted_times) else None

            account_records.append({
                "date": str(date) if date is not None else None,
                "portfolio_value": env.portfolio_value,
            })

            action_record = {
                "date": str(date) if date is not None else None,
            }
            for k in range(len(alloc_ensemble)):
                action_record[f"ensemble_alloc_{k}"] = alloc_ensemble[k]
            # No safe_alloc, no final_alloc, no confidence, no disagreement

            action_records.append(action_record)

            test_obs, rewards, dones, info = test_env.step(
                alloc_final.reshape(1, -1).astype(np.float32)
            )

            current_weights = env.current_weights.copy()

            if dones[0]:
                break

        account_df = pd.DataFrame(account_records)
        actions_df = pd.DataFrame(action_records)

        account_df.to_csv(os.path.join(out_dir, "test_account.csv"), index=False)
        actions_df.to_csv(os.path.join(out_dir, "test_actions.csv"), index=False)

        print(f"Saved {out_dir}/test_account.csv ({len(account_df)} rows)")
        print(f"Saved {out_dir}/test_actions.csv ({len(actions_df)} rows)")

        # Compute metrics
        vals = account_df["portfolio_value"].values
        total_return = vals[-1] / vals[0] - 1
        daily_returns = np.diff(vals) / vals[:-1]
        if np.std(daily_returns) > 0:
            sharpe = np.sqrt(252) * np.mean(daily_returns) / np.std(daily_returns)
        else:
            sharpe = np.inf if np.mean(daily_returns) > 0 else 0.0

        peak = np.maximum.accumulate(vals)
        drawdowns = (vals - peak) / peak
        max_drawdown = float(np.min(drawdowns))

        metrics = {
            "label": label,
            "final_value": float(vals[-1]),
            "total_return": float(total_return),
            "sharpe": float(sharpe),
            "max_drawdown": max_drawdown,
            "avg_daily_return": float(np.mean(daily_returns)),
            "volatility": float(np.std(daily_returns)),
            "mean_confidence": None,
            "mean_disagreement": None,
            "mapping": None,
            "safe_strategy": None,
            "d_ref": d_ref,
        }
        # Cash-weight diagnostic for baseline (ensemble == final)
        cash_diag = _cash_diagnostics(actions_df)
        metrics.update(cash_diag)
        results.append(metrics)

        print(f"  Sharpe={metrics['sharpe']:.3f}  "
              f"Return={metrics['total_return']:.2%}  "
              f"MaxDD={metrics['max_drawdown']:.2%}  "
              f"Conf=None  Disag=None  Cash={cash_diag['mean_final_cash']:.4f}")
    else:
        print("\n[skip] baseline_ensemble_average (--skip-baseline)")

    if not args.skip_combos:
        for mapping in rq1_config.CONFIDENCE_MAPPINGS:
            for safe in rq1_config.SAFE_STRATEGIES:
                label = rq1_config.combo_name(mapping, safe)
                out_dir = f"{results_root}/{label}"
                check_and_make_directories([out_dir])
                print("\n" + "=" * 60)
                print(f"Running combo: {mapping} / {safe}")
                print("=" * 60)

                agent = EnsemblePPOAllocationAgent(
                    n_ensemble=rq1_config.N_ENSEMBLE,
                    base_seed=rq1_config.BASE_SEED,
                    confidence_mapping=mapping,
                    safe_strategy=safe,
                    d_ref_method=rq1_config.D_REF_METHOD,
                    power_p=rq1_config.POWER_P,
                    sigmoid_k=rq1_config.SIGMOID_K,
                )
                agent.load_models(f"{rq1_config.MODEL_DIR}{suffix}")
                agent.d_ref = d_ref
                agent.disagreement_stats = disagreement_stats

                account_df, actions_df = agent.predict_ensemble(
                    df=test,
                    env_kwargs=rq1_config.ENV_KWARGS,
                    d_ref=d_ref,
                    save_dir=out_dir,
                )

                metrics = compute_metrics(account_df, label)
                metrics["mapping"] = mapping
                metrics["safe_strategy"] = safe
                metrics["d_ref"] = d_ref
                metrics["mean_confidence"] = float(actions_df["confidence"].mean())
                metrics["std_confidence"] = float(actions_df["confidence"].std())
                metrics["min_confidence"] = float(actions_df["confidence"].min())
                metrics["max_confidence"] = float(actions_df["confidence"].max())
                metrics["mean_disagreement"] = float(actions_df["disagreement"].mean())
                # Cash-weight diagnostics
                cash_diag = _cash_diagnostics(actions_df)
                metrics.update(cash_diag)
                results.append(metrics)

                print(f"  Sharpe={metrics['sharpe']:.3f}  "
                      f"Return={metrics['total_return']:.2%}  "
                      f"MaxDD={metrics['max_drawdown']:.2%}  "
                      f"Conf={metrics['mean_confidence']:.4f}  "
                      f"SafeCash={cash_diag['mean_safe_cash']:.4f}  "
                      f"FinalCash={cash_diag['mean_final_cash']:.4f}")
    else:
        print("\n[skip] combos (--skip-combos / --only-single)")

    # ---- Single PPO baseline(s) ----
    # Keep 1M and 5M strictly separate: distinct files and labels.
    # Full run defaults to the legacy 1M single; use --single-model ppo_single_5m
    # or --only-single --single-model ppo_single_5m for the 5M control.
    # Use --single-model both only if you intentionally want both rows in one summary.
    single_targets: list[tuple[str, str, str, str]] = []
    if args.single_model in ("ppo_single", "both"):
        single_targets.append(("ppo_single", "Single PPO",
                              f"{results_root}/account_value_single_ppo.csv",
                              f"{results_root}/actions_single_ppo.csv"))
    if args.single_model in ("ppo_single_5m", "both"):
        single_targets.append(("ppo_single_5m", "Single PPO",
                              f"{results_root}/account_value_single_ppo_5m.csv",
                              f"{results_root}/actions_single_ppo_5m.csv"))
    for model_name, label, acct_path, act_path in single_targets:
        m = _eval_single(model_name, label, acct_path, act_path)
        if m is not None:
            results.append(m)

    summary = pd.DataFrame(results).sort_values("sharpe", ascending=False) if results else pd.DataFrame()
    if not summary.empty:
        summary_path = f"{results_root}/test_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"\nSummary saved to {summary_path}")
        print("\n=== RQ1 Test Summary (sorted by Sharpe) ===")
        cols_to_print = [c for c in ["label", "total_return", "sharpe", "max_drawdown",
                       "mean_confidence", "mean_safe_cash", "mean_final_cash", "mean_ensemble_cash", "d_ref"] if c in summary.columns]
        print(summary[cols_to_print].round(4).to_string(index=False))
        # Also save a cash-focused view
        cash_cols = [c for c in ["label", "safe_strategy", "mapping", "mean_safe_cash", "mean_final_cash", "mean_ensemble_cash",
                                 "median_safe_cash", "median_final_cash", "sharpe", "total_return"] if c in summary.columns]
        if cash_cols:
            cash_summary_path = os.path.join(results_root, "cash_diagnostics.csv")
            summary[cash_cols].to_csv(cash_summary_path, index=False)
            print(f"Cash diagnostics saved to {cash_summary_path}")
            print(summary[cash_cols].round(4).to_string(index=False))
    else:
        print("\nNo results to summarise (all skipped or missing models).")


if __name__ == "__main__":
    main()
