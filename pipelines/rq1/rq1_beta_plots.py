"""
RQ1 beta plots — per-beta account-value curves + disagreement-confidence twin timeseries.

Per-beta isolated outputs (never touches results/rq1):
  results/rq1_beta{beta}/equity_beta{beta}.png
  results/rq1_beta{beta}/disagreement_confidence_twin_beta{beta}.png

Equity per beta: 4 combos (exponential/sigmoid × previous/equal_weight) + PPO ensemble baseline (black).
Twin per beta: 2×2 grid (row=mapping, col=safe) twin-axis D (left, steelblue) + c (right, darkgreen).

If too many betas, default picks top-3 by Sharpe from existing test_summary.csv (calibration-independent pick is done earlier; this is just plot reduction).
Safes = previous, equal_weight. Baseline = PPO ensemble only (no Single PPO/EW).

Usage:
  python -m pipelines.rq1.rq1_beta_plots
  python -m pipelines.rq1.rq1_beta_plots --betas 10 40 100
  python -m pipelines.rq1.rq1_beta_plots --betas 10 20 30 40 60 80 100 --top3   # auto pick top 3 Sharpe
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from pipelines.rq1 import rq1_config

RESULTS_ROOT = rq1_config.RESULTS_ROOT  # canonical, read-only for baseline + calibration
BETA_GRID = rq1_config.BETA_GRID
MAPPINGS = ["exponential", "sigmoid"]
SAFES = ["previous", "equal_weight"]
COLOR_D = "#2a5c8a"
COLOR_C = "darkgreen"
COLOR_ENSEMBLE = "black"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--betas", nargs="+", type=float, default=None,
                   help="Betas to plot (default: top-3 by Sharpe among existing rq1_beta dirs, else BETA_GRID top3)")
    p.add_argument("--mappings", nargs="+", default=MAPPINGS, choices=MAPPINGS,
                   help="Mappings (default: exponential sigmoid)")
    p.add_argument("--safes", nargs="+", default=SAFES,
                   help="Safes (default: previous equal_weight)")
    p.add_argument("--top3", action="store_true",
                   help="If betas list >3, auto-pick top 3 by Sharpe (default behaviour when --betas omitted)")
    p.add_argument("--no-equity", action="store_true", help="Skip equity curves")
    p.add_argument("--no-twin", action="store_true", help="Skip twin timeseries")
    return p.parse_args()


def beta_str(b: float) -> str:
    return str(int(b)) if float(b).is_integer() else str(b).replace(".", "p")


def discover_existing_betas() -> list[float]:
    betas = []
    for name in os.listdir("results"):
        if name.startswith("rq1_beta") and not name.startswith("rq1_beta_calibration"):
            suf = name.replace("rq1_beta", "")
            try:
                # directory suffixes are "10", "10p5", ... ("p" is the decimal point)
                betas.append(float(suf.replace("p", ".")))
            except ValueError:
                continue  # not a beta directory
    return sorted(betas)


def pick_top3_by_sharpe(candidates: list[float]) -> list[float]:
    scored = []
    for b in candidates:
        bs = beta_str(b)
        path = f"results/rq1_beta{bs}/test_summary.csv"
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
            # best Sharpe among combos for that beta
            best = df["sharpe"].max() if "sharpe" in df.columns else -999
            scored.append((best, b))
        except (OSError, ValueError, KeyError) as exc:
            print(f"[warn] skipping beta {b}: could not read {path}: {exc}")
    if not scored:
        return sorted(candidates)[:3]
    scored.sort(reverse=True)
    return [b for _, b in scored[:3]]


def load_account(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def plot_equity_per_beta(beta: float, mappings, safes):
    bs = beta_str(beta)
    results_root = f"results/rq1_beta{bs}"
    # baseline ensemble (canonical)
    base_path = os.path.join(RESULTS_ROOT, "baseline_ensemble_average", "test_account.csv")
    if not os.path.exists(base_path):
        print(f"[warn] baseline missing {base_path} — skipping equity for beta {beta}")
        return
    series = {}
    base = load_account(base_path)
    series["PPO ensemble"] = base.set_index("date")["portfolio_value"]

    for m in mappings:
        for s in safes:
            label = f"combo_{m}_beta{bs}_{s}"
            p = os.path.join(results_root, label, "test_account.csv")
            if not os.path.exists(p):
                print(f"[warn] missing {p}")
                continue
            df = load_account(p)
            # display name e.g. "Exponential (Previous) beta 10"
            disp = f"{m.capitalize()} ({s.replace('equal_weight','Equal Weight').replace('previous','Previous')}) beta {bs}"
            # shorter: Exp/ Sig
            short = {"exponential": "Exp", "sigmoid": "Sig"}[m]
            disp = f"{short} {s.replace('equal_weight','EW').replace('previous','Prev')} β{bs}"
            series[disp] = df.set_index("date")["portfolio_value"]

    if len(series) <= 1:
        print(f"[warn] no series for beta {beta} equity")
        return

    common_dates = sorted(set.intersection(*[set(s.index) for s in series.values()]))
    for k in series:
        series[k] = series[k].reindex(common_dates)

    fig, ax = plt.subplots(figsize=(16, 7))
    colors = {
        "PPO ensemble": COLOR_ENSEMBLE,
    }
    # colour per combo: orange shades for exponential, green/blue for sigmoid
    cmap_exp = plt.cm.Oranges
    cmap_sig = plt.cm.Greens
    exp_keys = [k for k in series if k.startswith("Exp")]
    sig_keys = [k for k in series if k.startswith("Sig")]
    for i, k in enumerate(sorted(exp_keys)):
        colors[k] = cmap_exp(0.5 + 0.5 * i / max(len(exp_keys)-1, 1))
    for i, k in enumerate(sorted(sig_keys)):
        colors[k] = cmap_sig(0.45 + 0.5 * i / max(len(sig_keys)-1, 1))

    for label, vals in series.items():
        lw = 2.0 if label == "PPO ensemble" else 1.6
        ax.plot(common_dates, vals.values, label=label, color=colors.get(label, "#333"), lw=lw)

    ax.set_title(f"RQ1 Beta {bs} — Account Value (Test 2022-2026, p90, per-beta)  [{results_root}]", fontsize=16)
    ax.set_xlabel("Date", fontsize=14)
    ax.set_ylabel("Portfolio Value ($)", fontsize=14)
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, ncol=2, loc="upper left")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    for lab in ax.get_xticklabels():
        lab.set_rotation(45)
        lab.set_ha("right")
    all_vals = pd.concat(series.values())
    ymin = max(600_000, float(all_vals.min()) * 0.95)
    ymax = float(all_vals.max()) * 1.05
    ax.set_ylim(ymin, ymax)

    fig.tight_layout()
    out = os.path.join(results_root, f"equity_beta{bs}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out} ({os.path.getsize(out)} bytes)")

    # print table
    print(f"Beta {bs} final values:")
    for k, v in series.items():
        print(f"  {k:30s} {v.iloc[-1]:,.0f}  ret {v.iloc[-1]/1e6-1:.2%}")


def plot_twin_per_beta(beta: float, mappings, safes):
    bs = beta_str(beta)
    results_root = f"results/rq1_beta{bs}"
    # 2x2 grid: rows=mappings, cols=safes
    nrows = len(mappings)
    ncols = len(safes)
    fig, axes = plt.subplots(nrows, ncols, figsize=(8*ncols, 5*nrows), sharex=True, sharey=False, squeeze=False)

    for i, m in enumerate(mappings):
        for j, s in enumerate(safes):
            ax = axes[i, j]
            label = f"combo_{m}_beta{bs}_{s}"
            p = os.path.join(results_root, label, "test_actions.csv")
            if not os.path.exists(p):
                ax.set_visible(False)
                continue
            df = pd.read_csv(p)
            df["date"] = pd.to_datetime(df["date"])
            # disagreement left
            ax.plot(df["date"], df["disagreement"], color=COLOR_D, lw=1.0, label="disagreement")
            ax.set_ylabel("Disagreement (TVD)", color=COLOR_D, fontsize=13)
            ax.tick_params(axis="y", labelcolor=COLOR_D, labelsize=11)
            ax.tick_params(axis="x", labelsize=11)
            ax.grid(True, alpha=0.3)
            # dynamic y for D: tight around data
            dmin, dmax = df["disagreement"].min(), df["disagreement"].max()
            pad = max((dmax-dmin)*0.15, 0.002)
            ax.set_ylim(max(0, dmin-pad), min(1, dmax+pad))
            # confidence right
            ax2 = ax.twinx()
            ax2.plot(df["date"], df["confidence"], color=COLOR_C, lw=1.1, label="confidence")
            ax2.set_ylabel("Confidence", color=COLOR_C, fontsize=13)
            ax2.tick_params(axis="y", labelcolor=COLOR_C, labelsize=11)
            # per-panel autoscale for confidence as in rq1_confidence_grid
            cmin, cmax = df["confidence"].min(), df["confidence"].max()
            cpad = max((cmax-cmin)*0.18, 0.015)
            # for exponential which is near 1, allow small range but not 0-1 full
            if cmax - cmin < 0.02:
                ax2.set_ylim(max(0, cmin - 0.02), min(1.0, cmax + 0.02))
            else:
                ax2.set_ylim(max(0, cmin - cpad), min(1.0, cmax + cpad))
            mean_d = df["disagreement"].mean()
            mean_c = df["confidence"].mean()
            std_c = df["confidence"].std()
            ax.set_title(f"{m.capitalize()} / {s.replace('equal_weight','Equal Weight').replace('previous','Previous')}: D̄={mean_d:.3f} c̄={mean_c:.3f} σ={std_c:.3f}", fontsize=12, pad=10)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax.tick_params(axis="x", rotation=30, labelsize=10)

    for j in range(ncols):
        axes[nrows-1, j].set_xlabel("Date", fontsize=13)
    fig.suptitle(f"RQ1 Beta {bs} — Disagreement & Confidence (Test, p90)  [{results_root}]", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    out = os.path.join(results_root, f"disagreement_confidence_twin_beta{bs}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out} ({os.path.getsize(out)} bytes)")


def main():
    args = parse_args()

    # discover or use provided betas
    if args.betas is None:
        candidates = discover_existing_betas()
        if not candidates:
            candidates = BETA_GRID
        betas = pick_top3_by_sharpe(candidates)
        print(f"No --betas given: discovered {candidates}, auto-picked top-3 by Sharpe: {betas}")
    else:
        betas = [float(b) for b in args.betas]
        if len(betas) > 3 and args.top3:
            betas = pick_top3_by_sharpe(betas)
            print(f"--top3: reduced to top-3 by Sharpe: {betas}")
        elif len(betas) > 3:
            top = pick_top3_by_sharpe(betas)
            print(f"Note: {len(betas)} betas requested, keeping top-3 by Sharpe for per-beta plots: {top} (use --betas 10 20 ... to force all)")
            betas = top

    # the canonical baseline must exist before plotting
    base_path = os.path.join(RESULTS_ROOT, "baseline_ensemble_average", "test_account.csv")
    assert os.path.exists(base_path), f"Missing baseline {base_path}"
    # filter betas that have results
    valid = []
    for b in betas:
        bs = beta_str(b)
        if os.path.isdir(f"results/rq1_beta{bs}"):
            valid.append(b)
        else:
            print(f"[skip] no dir results/rq1_beta{bs}")
    if not valid:
        raise SystemExit(f"No valid beta dirs for {betas}")
    betas = valid
    print(f"Plotting betas={betas} mappings={args.mappings} safes={args.safes}")

    for beta in betas:
        if not args.no_equity:
            plot_equity_per_beta(beta, args.mappings, args.safes)
        if not args.no_twin:
            plot_twin_per_beta(beta, args.mappings, args.safes)

    print("Done. Per-beta outputs in results/rq1_beta{beta}/")


if __name__ == "__main__":
    main()
