"""
RQ3 — Backtest comparison across all six reward-algo versions.

Reads each ver_vN/account_ensemble.csv from rq3_block_suite.py, aligns them on
the shared test window, adds the shared baselines and the RQ2 k10b20 floor, and
writes one comparison table + equity plot + a manifest logging the
training lineage of every member.

Usage:
    python -m pipelines.rq3.rq3_block_suite
    python -m pipelines.rq3.rq3_backtest

Output:
    results/rq3/backtest_performance_comparison.csv
    results/rq3/backtest_all_versions.png
    results/rq3/manifest.json
"""
from __future__ import annotations

import json
import os
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd

from pipelines.baselines.shared import LABELS

from pipelines.rq3 import rq3_config
from rl_portfolio.config import RESULTS_DIR
from rl_portfolio.utils.io import check_and_make_directories
from rl_portfolio.utils.metrics import backtest_stats

RQ2_K10B20_ACCOUNT = "results/rq2/sweeps/account_value_ensemble_pa_k10b20.csv"
RQ2_K10B20_LABEL = rq3_config.RQ2_BLOCK_LABEL
RQ2_K10B20_LABEL_DISPLAY = rq3_config.RQ2_BLOCK_LABEL_DISPLAY
VERSION_LABELS = rq3_config.VERSION_LABELS

# Display labels for plot legends.
VERSION_LABELS_DISPLAY = {k: v.replace("v", "V", 1) for k, v in VERSION_LABELS.items()}
VERSION_LABELS_SIMPLE_DISPLAY = {k: k.replace("v", "V") for k in VERSION_LABELS}

def _display_label(col: str) -> str:
    if col == RQ2_K10B20_LABEL:
        return RQ2_K10B20_LABEL_DISPLAY
    if col == "Mean Var":
        return "MVO"
    if col == "EW Rebalanced":
        return "Equal Weight Rebalanced"
    if col in VERSION_LABELS:
        return VERSION_LABELS_DISPLAY[col]
    if col in VERSION_LABELS_DISPLAY.values():
        return col
    # version string like v1 -> V1
    if col in VERSION_LABELS_SIMPLE_DISPLAY:
        return VERSION_LABELS_SIMPLE_DISPLAY[col]
    if col.startswith("v") and col[1:].isdigit():
        return col.upper()
    return col.replace("Mean Var", "MVO")

# Simple v1..v6 labels for top-sweep plot
VERSION_LABELS_SIMPLE = {k: k for k in VERSION_LABELS}


def load_account(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df


def main():

    parser = argparse.ArgumentParser(description="RQ3 backtest (use --use-sweep-top for per-version best k*b*)")
    parser.add_argument("--use-sweep-top", action="store_true",
                        help="use per-version Sharpe-best k*b* from sweeps/ instead of ver_vN k10b20")
    args = parser.parse_args()

    use_top = args.use_sweep_top
    version_labels = VERSION_LABELS_SIMPLE if use_top else VERSION_LABELS
    # output paths: keep canonical outputs, write *_top alongside when use_top
    out_csv = f"{rq3_config.RESULTS_ROOT}/backtest_performance_comparison{'_top' if use_top else ''}.csv"
    out_png = f"{rq3_config.RESULTS_ROOT}/backtest_all_versions{'_top' if use_top else ''}.png"
    # keep report copies consistent

    check_and_make_directories([RESULTS_DIR, rq3_config.RESULTS_ROOT])

    versions = list(rq3_config.VERSIONS)
    accounts = {}
    sweep_best = {}  # v -> tag
    if use_top:
        # load per-version best tag from sweeps
        for v in versions:
            summary_path = f"{rq3_config.RESULTS_ROOT}/sweeps/sweep_summary_{v}.csv"
            if os.path.exists(summary_path):
                df_sum = pd.read_csv(summary_path)
                df_sum = df_sum.sort_values("sharpe", ascending=False)
                top_tag = df_sum.iloc[0]["tag"]
                sweep_best[v] = top_tag
                path = f"{rq3_config.RESULTS_ROOT}/sweeps/account_value_{v}_{top_tag}.csv"
                if os.path.exists(path):
                    accounts[v] = load_account(path)
                    print(f"{v}: sweep-best {top_tag} from {path} ({len(accounts[v])} rows)")
                else:
                    print(f"[warn] {path} not found, falling back to ver_{v}")
                    fallback = f"{rq3_config.RESULTS_ROOT}/ver_{v}/account_ensemble.csv"
                    if os.path.exists(fallback):
                        accounts[v] = load_account(fallback)
                        print(f"{v}: fallback {fallback} ({len(accounts[v])} rows)")
            else:
                print(f"[warn] {summary_path} not found, using ver_{v}")
                path = f"{rq3_config.RESULTS_ROOT}/ver_{v}/account_ensemble.csv"
                if os.path.exists(path):
                    accounts[v] = load_account(path)
                    print(f"{v}: {len(accounts[v])} rows")
    else:
        for v in versions:
            path = f"{rq3_config.RESULTS_ROOT}/ver_{v}/account_ensemble.csv"
            if os.path.exists(path):
                accounts[v] = load_account(path)
                print(f"{v}: {len(accounts[v])} rows")

    if not accounts:
        raise SystemExit("No version accounts found. Run rq3_block_suite.py / rq3_sweep.py first.")

    anchor = accounts[sorted(accounts)[0]]
    dates = anchor["date"].tolist()
    initial_amount = float(anchor["portfolio_value"].iloc[0])
    print(f"\nAnchor: {len(dates)} days {dates[0]} -> {dates[-1]}, "
          f"initial {initial_amount:.0f}")
    if use_top:
        print(f"Using sweep-best tags: {sweep_best}")

    result = pd.DataFrame({"date": dates})
    for v, df in accounts.items():
        result[version_labels[v]] = (
            df.set_index("date")["portfolio_value"].reindex(dates).values
        )

    if os.path.exists(RQ2_K10B20_ACCOUNT):
        rq2 = load_account(RQ2_K10B20_ACCOUNT)
        result[RQ2_K10B20_LABEL] = (
            rq2.set_index("date")["portfolio_value"].reindex(dates).values
        )
    else:
        print(f"[note] {RQ2_K10B20_ACCOUNT} not found; omitting RQ2 floor.")

    for name, label in LABELS.items():
        df = pd.read_csv(f"results/baselines/{name}.csv")
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        s = df.set_index("date")["value"].reindex(dates)
        result[label] = (s / s.iloc[0] * initial_amount).values
    result = result.dropna().reset_index(drop=True)
    print(f"Aligned rows: {len(result)}")

    cols = [c for c in result.columns if c != "date"]
    series = []
    for col in cols:
        df = result[["date", col]].rename(columns={col: "account_value"})
        series.append(backtest_stats(df, value_col_name="account_value").rename(col))
    table = pd.concat(series, axis=1).round(4)
    table.to_csv(out_csv)
    print(f"\n=== RQ3 Performance Comparison ({'top sweep' if use_top else 'all versions'}) ===")
    print(table)

    fig, ax = plt.subplots(figsize=(15, 6))
    # use display labels for legend but keep internal cols for data
    version_labels_disp = VERSION_LABELS_DISPLAY if not use_top else VERSION_LABELS_SIMPLE_DISPLAY
    version_cols = [version_labels[v] for v in versions if version_labels[v] in result]
    version_cols_disp = [version_labels_disp[v] for v in versions if version_labels[v] in result]
    for col in version_cols:
        ax.plot(pd.to_datetime(result["date"]), result[col], lw=1.5)
    if RQ2_K10B20_LABEL in result:
        ax.plot(pd.to_datetime(result["date"]), result[RQ2_K10B20_LABEL],
                color="black", lw=1.8, zorder=5)
    BH_COLS = [LABELS[n] for n in ("ew_buyhold", "mvo_buyhold")]
    BH_COLS_disp = [_display_label(c) for c in BH_COLS]
    for col in BH_COLS:
        ax.plot(pd.to_datetime(result["date"]), result[col],
                lw=1.0, linestyle="--", alpha=0.8)
    title = "Portfolio Performance of Algorithm-Diverse Ensembles"
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    # ax.plot was called without label=, so handles are in plot order:
    # 6 version lines, 1 RQ2 line (if present), 2 BH lines
    all_lines = ax.get_lines()
    n_ver = len(version_cols)
    has_rq2 = RQ2_K10B20_LABEL in result
    version_handles = all_lines[0:n_ver]
    rq2_handle = all_lines[n_ver] if has_rq2 else mlines.Line2D([], [], color="none", label="")
    bh_start = n_ver + (1 if has_rq2 else 0)
    bh_handles = all_lines[bh_start:bh_start + len(BH_COLS)]
    # pad bh_handles if missing (should not happen)
    while len(bh_handles) < len(BH_COLS):
        bh_handles.append(mlines.Line2D([], [], color="none", label=""))
    blank = mlines.Line2D([], [], color="none", label=" ")
    # Column-major: first 6 = all versions (col1), next 6 = RQ2/EW/MVO + 3 blanks (col2)
    # matplotlib legend with ncol=2 fills column-wise (col1 = first 6 handles)
    handles_cm = version_handles + [rq2_handle, bh_handles[0], bh_handles[1], blank, blank, blank]
    labels_cm = version_cols_disp + [RQ2_K10B20_LABEL_DISPLAY, BH_COLS_disp[0], BH_COLS_disp[1], " ", " ", " "]
    ax.legend(handles_cm, labels_cm, loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved plot to {out_png}")

    # Only overwrite manifest in non-top mode to keep canonical k10b20 lineage
    if not use_top:
        manifest = {
            "research_question": "RQ3 - architecture diversity (ppo/sac/a2c, block k10b20)",
            "versions": {
                v: [
                    {"algo": m["algo"], "reward_type": m["reward_type"]}
                    for m in rq3_config.VERSIONS[v]
                ]
                for v in versions
            },
            "models": {
                "ppo": "REUSED from RQ2 training (rq_trained_models/rq2)",
                "sac": "trained in rq3_train.py (log_return -> dsr -> dsr_drawdown)",
                "a2c": "trained in rq3_train.py (log_return -> dsr -> dsr_drawdown)",
                "staging": "PPO members copied into rq_trained_models/rq3",
            },
            "ppo_scalings": {
                "reward_scaling_dsr": rq3_config.PPO_REWARD_SCALING_DSR,
                "dsr_eta_dd": rq3_config.PPO_DSR_ETA_DD,
            },
            "sac_a2c_calibration": {
                "method": f"eta anchored to PPO's {rq3_config.PPO_DSR_ETA_DD:,.0f}",
                "path": rq3_config.CALIBRATION_PATH,
            },
            "blend": {"mode": "block", "k": 10, "block_days": 20},
            "baselines": {"source": "results/baselines/ (compute_baselines.py)"},
            "comparison_floor": "results/rq2/sweeps/account_value_ensemble_pa_k10b20.csv",
            "outputs": rq3_config.RESULTS_ROOT,
        }
        with open(f"{rq3_config.RESULTS_ROOT}/manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"Manifest saved to {rq3_config.RESULTS_ROOT}/manifest.json")
    else:
        # for top mode, write a separate manifest to avoid overwriting canonical
        manifest_top = {
            "research_question": "RQ3 - architecture diversity (ppo/sac/a2c) — sweep-best per version",
            "versions_best": sweep_best,
            "versions": {
                v: [
                    {"algo": m["algo"], "reward_type": m["reward_type"]}
                    for m in rq3_config.VERSIONS[v]
                ]
                for v in versions
            },
            "blend": "per-version Sharpe-best k*b* from sweeps/sweep_summary_{version}.csv",
            "comparison_floor": "results/rq2/sweeps/account_value_ensemble_pa_k10b20.csv",
        }
        with open(f"{rq3_config.RESULTS_ROOT}/manifest_top.json", "w") as f:
            json.dump(manifest_top, f, indent=2)
        print(f"Top manifest saved to {rq3_config.RESULTS_ROOT}/manifest_top.json")
    print("Done.")


if __name__ == "__main__":
    main()
