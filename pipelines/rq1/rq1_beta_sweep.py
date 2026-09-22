"""
RQ1 beta sweep — inference-only test for beta-steepened exponential/sigmoid.

Reuses already-trained ensemble at rq_trained_models/rq1 (5 seeds, 1M steps)
and existing p90 calibration (D_ref=0.405021 from 2019-2021). No retraining.
Single beta shared by both mappings, centred on delta=1:

    exponential: c = clip(exp(-beta*(delta-1)),0,1)
    sigmoid:     c = 1/(1+exp(beta*(delta-1)))   delta=D/D_ref

Produces isolated outputs so canonical results/rq1 untouched:
    results/rq1_beta{beta}/combo_exponential_{safe}/ {test_account.csv, test_actions.csv}
    results/rq1_beta{beta}/combo_sigmoid_{safe}/ ...
    results/rq1_beta{beta}/test_summary.csv, calibration copy, manifest

Default betas = BETA_GRID {10,20,30,40,60,80,100}; usually run after picking
one beta from rq1_beta_calibration.py. Supports single-beta or full grid.

Usage:
    python -m pipelines.rq1.rq1_beta_sweep --betas 40            # single beta, both mappings & both safes
    python -m pipelines.rq1.rq1_beta_sweep --betas 40 --mappings sigmoid --safes equal_weight
    python -m pipelines.rq1.rq1_beta_sweep --betas 10 20 30 40 60 80 100
    python -m pipelines.rq1.rq1_beta_sweep --dry-run  # check I/O without inference
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

DEFAULT_BETAS = rq1_config.BETA_GRID
DEFAULT_MAPPINGS = ["exponential", "sigmoid"]
DEFAULT_SAFES = ["previous", "equal_weight"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--betas", nargs="+", type=float, default=DEFAULT_BETAS,
                   help=f"beta values (default: {DEFAULT_BETAS})")
    p.add_argument("--mappings", nargs="+", default=DEFAULT_MAPPINGS,
                   choices=["exponential", "sigmoid", "linear", "power"],
                   help="mappings to test (default: exponential sigmoid)")
    p.add_argument("--safes", nargs="+", default=DEFAULT_SAFES,
                   help="safe strategies (default: previous equal_weight)")
    p.add_argument("--tag", default="", help="optional suffix matching trained-model tag")
    p.add_argument("--dry-run", action="store_true", help="only create dirs/manifest, skip inference")
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

    test = pd.read_pickle(rq1_config.DATA_TEST)
    print(f"Test: {len(test)} rows, {test.date.nunique()} days, {test.tic.nunique()} tickers")
    cal_path = os.path.join(rq1_config.RESULTS_ROOT, "calibration_all.json")
    cal = json.load(open(cal_path))
    d_ref = cal["calibrations"][rq1_config.D_REF_METHOD]["d_ref"]
    disagreement_stats = cal["disagreement_stats"]
    print(f"D_ref ({rq1_config.D_REF_METHOD}): {d_ref:.6f}  betas={args.betas}  mappings={args.mappings}  safes={args.safes}")
    print(f"Cal mean D={disagreement_stats['d_mean']:.6f} std={disagreement_stats['d_std']:.6f}")

    for beta in args.betas:
        b_str = str(int(beta)) if float(beta).is_integer() else str(beta).replace(".", "p")
        results_root = f"results/rq1_beta{b_str}"
        check_and_make_directories([results_root])
        dst_cal = os.path.join(results_root, "calibration_all.json")
        if not os.path.exists(dst_cal):
            shutil.copy(cal_path, dst_cal)
        # copy calibration report for the run record if exists
        cal_report = "results/rq1_beta_calibration/beta_sweep_summary.csv"
        if os.path.exists(cal_report):
            try:
                shutil.copy(cal_report, os.path.join(results_root, "beta_calibration_summary.csv"))
            except OSError as exc:
                print(f"[warn] could not copy {cal_report} into the run record: {exc}")

        manifest = {
            "parent": "RQ1 beta sweep (isolated)",
            "base_results": rq1_config.RESULTS_ROOT,
            "beta": float(beta),
            "betas_grid": [float(b) for b in args.betas],
            "mappings": args.mappings,
            "safes": args.safes,
            "d_ref": d_ref,
            "d_ref_method": rq1_config.D_REF_METHOD,
            "disagreement_stats": disagreement_stats,
            "model_dir": f"{rq1_config.MODEL_DIR}{suffix}",
            "formulas": {
                "exponential": "c = clip(exp(-beta*(delta-1)),0,1), delta=D/D_ref",
                "sigmoid": "c = 1/(1+exp(beta*(delta-1)))",
            },
            "note": "Isolated so canonical results/rq1 untouched; separate test_summary per beta. Pick beta from calibration (rq1_beta_calibration.py) not test.",
        }
        with open(os.path.join(results_root, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

        if args.dry_run:
            print(f"[dry-run] Prepared {results_root}/manifest.json  (skipping inference)")
            continue

        results = []
        for mapping in args.mappings:
            for safe in args.safes:
                label = f"combo_{mapping}_beta{b_str}_{safe}"
                out_dir = os.path.join(results_root, label)
                check_and_make_directories([out_dir])
                print("\n" + "=" * 60)
                print(f"Running beta={beta} ({b_str})  {mapping} / {safe}  -> {out_dir}")
                print("=" * 60)
                agent = EnsemblePPOAllocationAgent(
                    n_ensemble=rq1_config.N_ENSEMBLE,
                    base_seed=rq1_config.BASE_SEED,
                    confidence_mapping=mapping,
                    safe_strategy=safe,
                    d_ref_method=rq1_config.D_REF_METHOD,
                    power_p=rq1_config.POWER_P,
                    sigmoid_k=rq1_config.SIGMOID_K,
                    beta=float(beta),
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
                metrics["beta"] = float(beta)
                metrics["d_ref"] = d_ref
                metrics["mean_confidence"] = float(actions_df["confidence"].mean())
                metrics["std_confidence"] = float(actions_df["confidence"].std())
                metrics["min_confidence"] = float(actions_df["confidence"].min())
                metrics["max_confidence"] = float(actions_df["confidence"].max())
                metrics["p10_confidence"] = float(actions_df["confidence"].quantile(0.10))
                metrics["p90_confidence"] = float(actions_df["confidence"].quantile(0.90))
                metrics["mean_disagreement"] = float(actions_df["disagreement"].mean())
                results.append(metrics)
                print(f"  Sharpe={metrics['sharpe']:.3f} Return={metrics['total_return']:.2%} "
                      f"MaxDD={metrics['max_drawdown']:.2%} Conf={metrics['mean_confidence']:.4f}±{metrics['std_confidence']:.4f} "
                      f"[{metrics['min_confidence']:.4f},{metrics['max_confidence']:.4f}] p10={metrics['p10_confidence']:.4f} p90={metrics['p90_confidence']:.4f}")

        if results:
            summary = pd.DataFrame(results).sort_values("sharpe", ascending=False)
            summary_path = os.path.join(results_root, "test_summary.csv")
            summary.to_csv(summary_path, index=False)
            print(f"\nSummary saved to {summary_path}")
            cols = ["label", "mapping", "safe_strategy", "beta", "total_return", "sharpe", "max_drawdown", "mean_confidence", "std_confidence", "p10_confidence", "p90_confidence", "mean_disagreement"]
            print(summary[cols].round(4).to_string(index=False))
        else:
            print(f"No combos run for beta {beta} (dry-run or empty)")

    print(f"\nAll sweeps done. Canonical results at {rq1_config.RESULTS_ROOT}/test_summary.csv untouched.")


if __name__ == "__main__":
    main()
