"""
RQ1: Equity plot comparing Power p=2, p=16, p=32 vs Linear vs PPO ensemble.

Reads 5 test_account.csv series (all equal_weight safe for the 4 mapped combos):
 - results/rq1/combo_power_equal_weight/test_account.csv          (p2 canonical)
 - results/rq1_power16/combo_power16_equal_weight/test_account.csv (p16)
 - results/rq1_power32/combo_power32_equal_weight/test_account.csv (p32)
 - results/rq1/combo_linear_equal_weight/test_account.csv         (linear)
 - results/rq1/baseline_ensemble_average/test_account.csv         (basic ensemble avg, no safe/confidence)

Saves:
 - results/rq1/power_comparison_p2_p16_p32_linear_baseline.png
 - plus a sweep isolate copy under results/rq1_power16/
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

def load(path):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["portfolio_value"]

base = "results/rq1/combo_power_equal_weight/test_account.csv"
p16 = "results/rq1_power16/combo_power16_equal_weight/test_account.csv"
p32 = "results/rq1_power32/combo_power32_equal_weight/test_account.csv"
linear = "results/rq1/combo_linear_equal_weight/test_account.csv"
baseline = "results/rq1/baseline_ensemble_average/test_account.csv"

series = {
    "PPO ensemble": load(baseline),
    "Linear (Equal Weight)": load(linear),
    "Power p=2 (canonical)": load(base),
    "Power p=16": load(p16),
    "Power p=32": load(p32),
}

# align on common dates (intersection)
common_dates = sorted(set.intersection(*[set(s.index) for s in series.values()]))
for k in series:
    series[k] = series[k].reindex(common_dates)

plt.rcParams["figure.figsize"] = (16, 7)
fig, ax = plt.subplots()

colors = {
    "PPO ensemble": "black",
    "Linear (Equal Weight)": "#2a5c8a",
    "Power p=2 (canonical)": "#1f77b4",
    "Power p=16": "#ff7f0e",
    "Power p=32": "#d62728",
}
styles = {
    "PPO ensemble": "-",
    "Linear (Equal Weight)": "-",
    "Power p=2 (canonical)": "-",
    "Power p=16": "-",
    "Power p=32": "-",
}
widths = {
    "PPO ensemble": 1.8,
    "Linear (Equal Weight)": 1.8,
    "Power p=2 (canonical)": 1.6,
    "Power p=16": 1.8,
    "Power p=32": 1.8,
}

for label, vals in series.items():
    ax.plot(common_dates, vals.values, label=label, color=colors[label], linestyle=styles[label], lw=widths[label])

ax.set_title("RQ1: Power p2 vs p16 vs p32 vs Linear vs PPO ensemble (2022-2026, Equal Weight)", fontsize=16)
ax.set_xlabel("Date", fontsize=14)
ax.set_ylabel("Portfolio Value ($)", fontsize=14)
ax.tick_params(axis="both", labelsize=11)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11, ncol=2, loc="upper left")

fig.tight_layout()

outs = [
    "results/rq1/power_comparison_p2_p16_p32_linear_baseline.png",
    "results/rq1_power16/power_comparison_p2_p16_p32_linear_baseline.png",
]
for o in outs:
    os.makedirs(os.path.dirname(o), exist_ok=True)
    fig.savefig(o, dpi=150, bbox_inches="tight")
    print(f"Saved {o} ({os.path.getsize(o)} bytes)")

plt.close(fig)

print("\nFinal portfolio values:")
for k, v in series.items():
    print(f"{k:25s} {v.iloc[-1]:,.0f}  return {v.iloc[-1]/1e6-1:.2%}  last date {v.index[-1].date()}")
print(f"Common dates: {len(common_dates)} from {common_dates[0].date()} to {common_dates[-1].date()}")
