"""
RQ1 power sweep — inference-only test for power=16 and 32.

Reuses already-trained ensemble at rq_trained_models/rq1 (5 seeds, 1M steps)
and existing p90 calibration (D_ref≈1.0345 from 2019-2021). No retraining.

Produces isolated outputs so canonical p2 results stay untouched:
  results/rq1_power16/combo_power16_{previous,equal_weight}/ {test_account.csv, test_actions.csv}
  results/rq1_power32/...
  results/rq1_power{p}/test_summary.csv, calibration copy, manifest snippet

Usage:
  python -m pipelines.rq1.rq1_power_sweep           # runs 16 & 32, both safes
  python -m pipelines.rq1.rq1_power_sweep --powers 16 32 --safes equal_weight
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import numpy as np
import pandas as pd

from pipelines.rq1.rq1_alloc import EnsemblePPOAllocationAgent
from pipelines.rq1 import rq1_config
from rl_portfolio.utils.io import check_and_make_directories


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--powers", nargs="+", type=float, default=[16.0, 32.0], help="power exponents to test")
    p.add_argument("--safes", nargs="+", default=["previous", "equal_weight"], help="safe strategies")
    p.add_argument("--tag", default="", help="optional suffix matching trained-model tag")
    return p.parse_args()


def compute_metrics(account_df: pd.DataFrame, label: str) -> dict:
    vals = account_df["portfolio_value"].values
    total_return = vals[-1] / vals[0] - 1
    daily_returns = np.diff(vals) / vals[:-1]
    sharpe = float(np.sqrt(252) * np.mean(daily_returns) / np.std(daily_returns)) if np.std(daily_returns) > 0 else (float("inf") if np.mean(daily_returns) > 0 else 0.0)
    peak = np.maximum.accumulate(vals)
    max_drawdown = float(np.min((vals - peak) / peak))
    return {
        "label": label,
        "final_value": float(vals[-1]),
        "total_return": float(total_return),
        "sharpe": float(sharpe),
        "max_drawdown": max_drawdown,
        "avg_daily_return": float(np.mean(daily_returns)),
        "volatility": float(np.std(daily_returns)),
    }


def main():
    args = parse_args()
    suffix = f"_{args.tag}" if args.tag else ""
    # load test data & calibration once
    test = pd.read_pickle(rq1_config.DATA_TEST)
    print(f"Test: {len(test)} rows, {test.date.nunique()} days, {test.tic.nunique()} tickers")
    cal_path = os.path.join(rq1_config.RESULTS_ROOT, "calibration_all.json")
    cal = json.load(open(cal_path))
    d_ref = cal["calibrations"][rq1_config.D_REF_METHOD]["d_ref"]
    disagreement_stats = cal["disagreement_stats"]
    print(f"D_ref ({rq1_config.D_REF_METHOD}): {d_ref:.6f}  powers={args.powers}  safes={args.safes}")

    for power in args.powers:
        p_int = int(power) if power.is_integer() else str(power).replace(".", "p")
        results_root = f"results/rq1_power{p_int}"
        check_and_make_directories([results_root])
        # copy calibration for the run record
        dst_cal = os.path.join(results_root, "calibration_all.json")
        if not os.path.exists(dst_cal):
            shutil.copy(cal_path, dst_cal)
        # manifest for the run record
        manifest = {
            "parent": "RQ1 power sweep (isolated)",
            "base_results": rq1_config.RESULTS_ROOT,
            "power_p": float(power),
            "safes": args.safes,
            "mapping": "power",
            "d_ref": d_ref,
            "d_ref_method": rq1_config.D_REF_METHOD,
            "model_dir": f"{rq1_config.MODEL_DIR}{suffix}",
            "note": "Isolated so canonical p2 (results/rq1) untouched; separate test_summary.",
        }
        with open(os.path.join(results_root, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

        results = []
        for safe in args.safes:
            label = f"combo_power{p_int}_{safe}"
            out_dir = os.path.join(results_root, label)
            check_and_make_directories([out_dir])
            print("\n" + "=" * 60)
            print(f"Running power={power} ({p_int}) / {safe}  -> {out_dir}")
            print("=" * 60)
            agent = EnsemblePPOAllocationAgent(
                n_ensemble=rq1_config.N_ENSEMBLE,
                base_seed=rq1_config.BASE_SEED,
                confidence_mapping="power",
                safe_strategy=safe,
                d_ref_method=rq1_config.D_REF_METHOD,
                power_p=float(power),
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
            metrics["mapping"] = "power"
            metrics["safe_strategy"] = safe
            metrics["d_ref"] = d_ref
            metrics["power_p"] = float(power)
            metrics["mean_confidence"] = float(actions_df["confidence"].mean())
            metrics["std_confidence"] = float(actions_df["confidence"].std())
            metrics["min_confidence"] = float(actions_df["confidence"].min())
            metrics["max_confidence"] = float(actions_df["confidence"].max())
            metrics["mean_disagreement"] = float(actions_df["disagreement"].mean())
            results.append(metrics)
            print(f"  Sharpe={metrics['sharpe']:.3f} Return={metrics['total_return']:.2%} "
                  f"MaxDD={metrics['max_drawdown']:.2%} Conf={metrics['mean_confidence']:.4f} "
                  f"[{metrics['min_confidence']:.4f},{metrics['max_confidence']:.4f}]")

        summary = pd.DataFrame(results).sort_values("sharpe", ascending=False)
        summary_path = os.path.join(results_root, "test_summary.csv")
        summary.to_csv(summary_path, index=False)
        print(f"\nSummary saved to {summary_path}")
        print(summary[["label", "total_return", "sharpe", "max_drawdown", "mean_confidence", "mean_disagreement"]].round(4).to_string(index=False))

    print("\nAll sweeps done. Compare to canonical p2 at results/rq1/test_summary.csv")


if __name__ == "__main__":
    main()
