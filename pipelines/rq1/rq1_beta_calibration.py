"""
RQ1 beta calibration — calibration-window only (no test).

Loads the calibration-window disagreements (2019-2021, from
rq1_validate.py) and evaluates the *new* delta-centred mappings

    exponential: c = clip(exp(-beta*(delta-1)),0,1)
    sigmoid:     c = 1/(1+exp(beta*(delta-1)))   delta=D/D_ref

for beta in BETA_GRID {10,20,30,40,60,80,100} on D_ref=p90.
Reports mean/std/min/max/p10/p90 per (mapping,beta) — calibration
only, so beta is picked without peeking at test (2022-2026).
Linear/power are included as reference but unchanged.

No model loading, no test window, no overwrite of results/0rq1.
Outputs:
    results/0rq1_beta_calibration/beta_sweep_summary.csv
    results/0rq1_beta_calibration/validation_disagreement_hist.png
    results/0rq1_beta_calibration/calibration_curves.png
Copies also to results/report/rq1_beta_*

Usage:
    python -m pipelines.rq1.rq1_beta_calibration
    python -m pipelines.rq1.rq1_beta_calibration --betas 10 20 30 40 60 80 100
    python -m pipelines.rq1.rq1_beta_calibration --no-plot
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from pipelines.rq1 import rq1_config

RESULTS_ROOT = rq1_config.RESULTS_ROOT  # results/0rq1 (read-only)
CAL_PATH = os.path.join(RESULTS_ROOT, "calibration_all.json")
VAL_CSV = os.path.join(RESULTS_ROOT, "validation_disagreements.csv")
OUT_ROOT = "results/0rq1_beta_calibration"
REPORT_ROOT = "results/report"

DEFAULT_BETAS = rq1_config.BETA_GRID  # [10,20,30,40,60,80,100]


def confidence_exp(beta: float, delta: np.ndarray) -> np.ndarray:
    return np.clip(np.exp(-beta * (delta - 1.0)), 0.0, 1.0)


def confidence_sigmoid(beta: float, delta: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(beta * (delta - 1.0)))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--betas", nargs="+", type=float, default=DEFAULT_BETAS,
                   help="Beta grid (default: BETA_GRID from config)")
    p.add_argument("--d-ref-method", default=rq1_config.D_REF_METHOD,
                   help="D_ref method (default p90; kept fixed)")
    p.add_argument("--no-plot", action="store_true", help="Skip PNG curves/hists")
    return p.parse_args()


def main():
    args = parse_args()
    betas = [float(b) for b in args.betas]
    os.makedirs(OUT_ROOT, exist_ok=True)
    os.makedirs(REPORT_ROOT, exist_ok=True)

    if not os.path.exists(CAL_PATH):
        raise FileNotFoundError(f"Missing {CAL_PATH} — run rq1_validate.py first")
    if not os.path.exists(VAL_CSV):
        raise FileNotFoundError(f"Missing {VAL_CSV} — run rq1_validate.py first")

    with open(CAL_PATH) as f:
        cal = json.load(f)
    method = args.d_ref_method
    if method not in cal["calibrations"]:
        raise ValueError(f"D_ref method {method} not in {CAL_PATH}: {list(cal['calibrations'])}")
    d_ref = float(cal["calibrations"][method]["d_ref"])
    d_stats = cal.get("disagreement_stats", {})

    dis = pd.read_csv(VAL_CSV)["disagreement"].values.astype(float)
    delta = dis / d_ref if d_ref > 0 else np.ones_like(dis)

    print(f"Calibration window: {cal['validation_period']}  method={method}  d_ref={d_ref:.6f}")
    print(f"N={len(dis)}  D mean={dis.mean():.6f} std={dis.std():.6f} min={dis.min():.6f} max={dis.max():.6f} median={np.median(dis):.6f}")
    print(f"delta=D/D_ref  mean={delta.mean():.6f} std={delta.std():.6f} min={delta.min():.6f} max={delta.max():.6f}  delta-1 mean={delta.mean()-1:.6f}")
    print(f"Betas: {betas}")

    rows = []
    # reference: linear and power on same D_ref (beta irrelevant)
    for mapping in ["linear", "power"]:
        if mapping == "linear":
            c = 1.0 - np.clip(delta, 0.0, 1.0)
        else:
            c = 1.0 - np.clip(delta ** rq1_config.POWER_P, 0.0, 1.0)
        rows.append({
            "mapping": mapping, "beta": np.nan,
            "mean": float(c.mean()), "std": float(c.std()),
            "min": float(c.min()), "max": float(c.max()),
            "p10": float(np.percentile(c, 10)), "p90": float(np.percentile(c, 90)),
            "frac_at_1": float((c == 1.0).mean()), "frac_at_0": float((c == 0.0).mean()),
        })

    for beta in betas:
        for mapping in ["exponential", "sigmoid"]:
            if mapping == "exponential":
                c = confidence_exp(beta, delta)
                frac1 = float((c >= 1.0 - 1e-12).mean())
                frac0 = float((c <= 1e-12).mean())
            else:
                c = confidence_sigmoid(beta, delta)
                frac1 = float((c >= 1.0 - 1e-12).mean())
                frac0 = float((c <= 1e-12).mean())
            rows.append({
                "mapping": mapping, "beta": float(beta),
                "mean": float(c.mean()), "std": float(c.std()),
                "min": float(c.min()), "max": float(c.max()),
                "p10": float(np.percentile(c, 10)), "p90": float(np.percentile(c, 90)),
                "frac_at_1": frac1, "frac_at_0": frac0,
            })

    df = pd.DataFrame(rows)
    # order: linear, power, then exponential betas ascending, sigmoid betas ascending
    order = {"linear": 0, "power": 1, "exponential": 2, "sigmoid": 3}
    df["_o"] = df["mapping"].map(order)
    df = df.sort_values(["_o", "beta"], na_position="first").drop(columns=["_o"])

    out_csv = os.path.join(OUT_ROOT, "beta_sweep_summary.csv")
    df.to_csv(out_csv, index=False, float_format="%.6f")
    report_csv = os.path.join(REPORT_ROOT, "rq1_beta_calibration_summary.csv")
    df.to_csv(report_csv, index=False, float_format="%.6f")
    print(f"\nSaved {out_csv}")
    print(f"Saved {report_csv}")

    pd.set_option("display.max_rows", 100)
    pd.set_option("display.float_format", lambda x: f"{x:.5f}")
    print("\n=== Calibration-only confidence distribution (p90=%.6f) ===" % d_ref)
    print(df.to_string(index=False))

    # manifest for the run record
    manifest = {
        "validation_period": cal["validation_period"],
        "d_ref_method": method, "d_ref": d_ref,
        "disagreement_stats": d_stats,
        "delta_stats": {"mean": float(delta.mean()), "std": float(delta.std()),
                        "min": float(delta.min()), "max": float(delta.max()),
                        "p10": float(np.percentile(delta, 10)), "p90": float(np.percentile(delta, 90))},
        "betas": betas, "mappings": ["exponential", "sigmoid"],
        "note": "Calibration-only; no test window. Pick beta from this, not from test.",
    }
    with open(os.path.join(OUT_ROOT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    if args.no_plot:
        print("Skipping plots (--no-plot)")
        return

    # --- plots ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # 1) disagreement histogram (calibration window) with D_ref line
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(dis, bins=50, color="#2a5c8a", edgecolor="white", alpha=0.85)
        ax.axvline(d_ref, color="red", linestyle="--", lw=1.8, label=f"d_ref p90={d_ref:.4f}")
        ax.axvline(dis.mean(), color="darkorange", linestyle=":", lw=1.6, label=f"mean={dis.mean():.4f}")
        ax.set_xlabel("Disagreement (TVD)")
        ax.set_ylabel("Frequency")
        ax.set_title("RQ1: Calibration Disagreement Distribution (2019-2021, p90)")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        out1 = os.path.join(OUT_ROOT, "validation_disagreement_hist.png")
        out1r = os.path.join(REPORT_ROOT, "rq1_beta_validation_disagreement_hist.png")
        fig.savefig(out1, dpi=150, bbox_inches="tight")
        fig.savefig(out1r, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out1}")
        print(f"Saved {out1r}")

        # 2) calibration curves: c vs delta (analytic) for all betas, with rug of actual deltas
        delta_grid = np.linspace(0.85, 1.15, 600)
        # use a qualitative palette that remains distinct for 7 betas
        cmap = plt.cm.viridis
        betas_sorted = sorted(betas)
        colors = [cmap(i / max(len(betas_sorted)-1,1)) for i in range(len(betas_sorted))]

        fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
        for ax, mapping in zip(axes, ["exponential", "sigmoid"]):
            for beta, col in zip(betas_sorted, colors):
                if mapping == "exponential":
                    c_grid = confidence_exp(beta, delta_grid)
                else:
                    c_grid = confidence_sigmoid(beta, delta_grid)
                ax.plot(delta_grid, c_grid, lw=1.8, color=col, label=f"beta={int(beta) if float(beta).is_integer() else beta}")
            # rug of actual calibration deltas (2D histogram at bottom)
            ax.plot(delta, np.full_like(delta, -0.02), "|", color="black", alpha=0.25, markersize=4)
            # shade p10-p90 of delta
            d10, d90 = np.percentile(delta, [10, 90])
            ax.axvspan(d10, d90, color="grey", alpha=0.12, label=f"delta p10-p90 [{d10:.3f},{d90:.3f}]")
            ax.axvline(1.0, color="red", linestyle="--", lw=1.2, alpha=0.8)
            ax.axvline(delta.mean(), color="darkorange", linestyle=":", lw=1.2, alpha=0.9)
            ax.set_xlabel("delta = D / D_ref (p90)")
            ax.set_xlim(delta_grid.min(), delta_grid.max())
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3)
            ax.set_title(f"{mapping.capitalize()}  c(beta, delta)")
            ax.legend(fontsize=8, ncol=2, loc="best")
        axes[0].set_ylabel("Confidence c")
        fig.suptitle(f"RQ1 Beta Calibration Curves (p90 D_ref={d_ref:.4f}, calibration deltas underlay)", fontsize=13)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        out2 = os.path.join(OUT_ROOT, "calibration_curves.png")
        out2r = os.path.join(REPORT_ROOT, "rq1_beta_calibration_curves.png")
        fig.savefig(out2, dpi=150, bbox_inches="tight")
        fig.savefig(out2r, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out2}")
        print(f"Saved {out2r}")

        # 3) histograms of c on calibration deltas, per beta (2 rows x 7 cols small multiples)
        fig, axes = plt.subplots(2, len(betas_sorted), figsize=(3*len(betas_sorted), 6), sharex=True, sharey=True)
        if len(betas_sorted) == 1:
            axes = axes.reshape(2, 1)
        for j, beta in enumerate(betas_sorted):
            for i, mapping in enumerate(["exponential", "sigmoid"]):
                ax = axes[i, j]
                if mapping == "exponential":
                    c = confidence_exp(beta, delta)
                    col = "#ff7f0e"
                else:
                    c = confidence_sigmoid(beta, delta)
                    col = "#2ca02c"
                ax.hist(c, bins=30, color=col, edgecolor="white", alpha=0.85)
                ax.axvline(float(np.mean(c)), color="red", linestyle="--", lw=1.2)
                ax.set_title(f"{mapping[:3]} beta={int(beta) if float(beta).is_integer() else beta}\nmean={c.mean():.3f} p10={np.percentile(c,10):.3f} p90={np.percentile(c,90):.3f}", fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.set_xlim(-0.02, 1.02)
        for ax in axes[1, :]:
            ax.set_xlabel("Confidence c", fontsize=9)
        axes[0,0].set_ylabel("Frequency", fontsize=10)
        axes[1,0].set_ylabel("Frequency", fontsize=10)
        fig.suptitle(f"Calibration c histograms per beta (p90 D_ref={d_ref:.4f}, N={len(delta)})", fontsize=12)
        fig.tight_layout(rect=[0, 0.03, 1, 0.96])
        out3 = os.path.join(OUT_ROOT, "confidence_hist_by_beta.png")
        out3r = os.path.join(REPORT_ROOT, "rq1_beta_confidence_hist_by_beta.png")
        fig.savefig(out3, dpi=150, bbox_inches="tight")
        fig.savefig(out3r, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out3}")
        print(f"Saved {out3r}")

    except Exception as e:
        print(f"Plotting failed: {e}")
        import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
