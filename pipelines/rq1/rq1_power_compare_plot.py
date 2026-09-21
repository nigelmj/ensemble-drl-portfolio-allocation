"""
Compare power p=2 (canonical) vs p=16 vs p=32 confidence across test set.

Reads:
  results/0rq1/combo_power_equal_weight/test_actions.csv            (p2)
  results/0rq1_power16/combo_power16_equal_weight/test_actions.csv  (p16)
  results/0rq1_power32/combo_power32_equal_weight/test_actions.csv  (p32)
and same for previous safe.

Produces (for each safe):
  results/0rq1_power16/confidence_compare_p2_p16_p32_{safe}.png
  results/report/rq1_confidence_compare_power_{safe}.png
plus combined equal_weight twin-style overlay.

Usage: python -m pipelines.rq1.rq1_power_compare_plot
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

COLOR_D = "#2a5c8a"
COLORS = {"p2": "#1f77b4", "p16": "#ff7f0e", "p32": "#d62728"}  # blue, orange, red
LABELS = {"p2": "p=2 (canonical)", "p16": "p=16", "p32": "p=32"}

def load(safe, p):
    if p == 2:
        path = f"results/0rq1/combo_power_equal_weight/test_actions.csv" if safe=="equal_weight" else f"results/0rq1/combo_power_previous/test_actions.csv"
        # Note canonical previous exists under 0rq1/combo_power_previous
        if safe == "previous":
            # fallback: check both
            alt = "results/0rq1/combo_power_previous/test_actions.csv"
            if not os.path.exists(path):
                path = alt
    else:
        path = f"results/0rq1_power{p}/combo_power{p}_{safe}/test_actions.csv"
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df

def plot_for_safe(safe):
    p2 = load(safe, 2)
    p16 = load(safe, 16)
    p32 = load(safe, 32)
    # confidence overlay
    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
    # top: disagreement (single line, since D almost identical)
    axes[0].plot(p2["date"], p2["disagreement"], color=COLOR_D, lw=1.0, label="disagreement (p2; p16/32 nearly identical)")
    # also plot p16/p32 disagreement faint to prove
    axes[0].plot(p16["date"], p16["disagreement"], color=COLOR_D, lw=0.6, alpha=0.4, linestyle="--")
    axes[0].set_ylabel("Disagreement (TVD)", fontsize=14)
    axes[0].tick_params(labelsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=11)
    _safe_disp = {"equal_weight": "Equal Weight", "previous": "Previous"}
    axes[0].set_title(f"Disagreement (test set) — {_safe_disp.get(safe, safe)}", fontsize=15, pad=8)
    # bottom: confidence comparison p2 vs p16 vs p32
    axes[1].plot(p2["date"], p2["confidence"], color=COLORS["p2"], lw=1.1, label=f'{LABELS["p2"]} (mean {p2.confidence.mean():.3f})')
    axes[1].plot(p16["date"], p16["confidence"], color=COLORS["p16"], lw=1.1, label=f'{LABELS["p16"]} (mean {p16.confidence.mean():.3f})')
    axes[1].plot(p32["date"], p32["confidence"], color=COLORS["p32"], lw=1.1, label=f'{LABELS["p32"]} (mean {p32.confidence.mean():.3f})')
    axes[1].set_ylabel("Confidence", fontsize=14)
    axes[1].set_xlabel("Date", fontsize=15)
    axes[1].tick_params(labelsize=11)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(axes[1].get_xticklabels(), rotation=30, ha="right")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=11, ncol=3)
    axes[1].set_title(f"Confidence — Power mapping, {_safe_disp.get(safe, safe)} safe", fontsize=15, pad=8)
    fig.suptitle(f"RQ1: Power Confidence Comparison p2 vs p16 vs p32 — {_safe_disp.get(safe, safe)}", fontsize=17)
    fig.tight_layout(rect=[0,0.03,1,0.96])
    out1 = f"results/0rq1_power16/confidence_compare_p2_p16_p32_{safe}.png"
    out2 = f"results/report/rq1_confidence_compare_power_{safe}.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    fig.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out1} and {out2}")

    # also save stats table
    for df, tag in [(p2,"p2"),(p16,"p16"),(p32,"p32")]:
        print(f"{safe} {tag}: mean_c={df.confidence.mean():.4f} std={df.confidence.std():.4f} min={df.confidence.min():.4f} max={df.confidence.max():.4f} mean_d={df.disagreement.mean():.4f}")

def main():
    os.makedirs("results/report", exist_ok=True)
    for safe in ["equal_weight","previous"]:
        plot_for_safe(safe)
    print("Done")

if __name__ == "__main__":
    main()
