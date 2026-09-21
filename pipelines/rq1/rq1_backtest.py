"""
RQ1 — Backtest comparison: N combos vs single PPO + shared baselines.

Reads the per-combo accounts from rq1_test.py and the shared baselines from
results/baselines/, builds a performance comparison table, equity plots,
disagreement/confidence timeseries and histograms, cash-weight diagnostics,
and writes a manifest.

Supports temporary reporting in results/temp/0rq1 for the
equal_weight_stocks ablation (12 combos when SAFE_STRATEGIES includes
equal_weight_stocks).

Usage:
    python -m pipelines.rq1.rq1_test --results_root results/temp/0rq1
    python -m pipelines.rq1.rq1_backtest --results_root results/temp/0rq1

Output:
    <results_root>/backtest_performance_comparison.csv
    <results_root>/backtest_result.png
    <results_root>/disagreement_confidence_timeseries.png
    <results_root>/disagreement_histogram.png
    <results_root>/confidence_histogram.png
    <results_root>/cash_weight_comparison.png
    <results_root>/cash_diagnostics.csv
    <results_root>/manifest.json
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from pipelines.baselines.shared import LABELS, NAMES, attach_baselines

from pipelines.rq1 import rq1_config
from rl_portfolio.config import RESULTS_DIR
from rl_portfolio.utils.io import check_and_make_directories
from rl_portfolio.utils.metrics import backtest_stats

INITIAL_AMOUNT = rq1_config.ENV_KWARGS["initial_amount"]

# Display helpers for plot rendering only
_MAPPING_DISP = {"linear": "Linear", "power": "Power", "exponential": "Exponential", "sigmoid": "Sigmoid"}
_SAFE_DISP = {"previous": "Previous", "equal_weight": "Equal Weight", "equal_weight_stocks": "Equal Weight Stocks"}
_BASELINE_DISP = {"baseline_ensemble_average": "PPO ensemble"}

def _display_combo_name(raw: str) -> str:
    """Map raw combo like 'power_equal_weight' or 'baseline_ensemble_average' to display."""
    if raw in _BASELINE_DISP:
        return _BASELINE_DISP[raw]
    # try to split mapping_safe
    for m in _MAPPING_DISP:
        if raw.startswith(m + "_"):
            safe = raw[len(m)+1:]
            return f"{_MAPPING_DISP[m]} ({_SAFE_DISP.get(safe, safe)})"
        if raw.startswith(m):
            # handle power16 etc - keep as is but capitalize mapping part
            return raw.replace(m, _MAPPING_DISP[m], 1).replace("equal_weight_stocks", "Equal Weight Stocks").replace("equal_weight", "Equal Weight")
    # fallback: replace underscores
    return raw.replace("equal_weight_stocks", "Equal Weight Stocks").replace("equal_weight", "Equal Weight").replace("baseline_ensemble_average", "PPO ensemble")

def _display_label(raw: str) -> str:
    if raw == "Single PPO":
        return raw
    if raw in LABELS.values():
        # Map Mean Var -> MVO for plot
        return raw.replace("Mean Var", "MVO").replace("EW Rebalanced", "Equal Weight Rebalanced")
    return _display_combo_name(raw)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_root",
        default=rq1_config.RESULTS_ROOT,
        help="Directory with per-combo test_account.csv files. "
             "Defaults to RESULTS_ROOT; use results/temp/0rq1 for stocks-only ablation.",
    )
    parser.add_argument(
        "--use-5m",
        action="store_true",
        help="Use the compute-matched 5M single (account_value_single_ppo_5m.csv) and "
             "write _5m-suffixed outputs so 1M artefacts are preserved (separate baseline).",
    )
    return parser.parse_args()


def load_account(path: str, value_col: str = "portfolio_value") -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.rename(columns={value_col: "value"})
    return df


def main():
    args = parse_args()
    results_root = args.results_root
    use_5m = args.use_5m
    tag_suffix = "_5m" if use_5m else ""
    single_label = "Single PPO"
    single_file = "account_value_single_ppo_5m.csv" if use_5m else "account_value_single_ppo.csv"
    check_and_make_directories([RESULTS_DIR, results_root])
    # Fallback for single PPO if temp root not yet populated but the main root exists
    single_path = os.path.join(results_root, single_file)
    if not os.path.exists(single_path):
        fallback = os.path.join(rq1_config.RESULTS_ROOT, single_file)
        if os.path.exists(fallback):
            print(f"{single_label} not found in {results_root}, using fallback {fallback}")
            single_path = fallback

    single = load_account(single_path)
    dates = single["date"].tolist()
    initial_amount = float(single["value"].iloc[0])
    print(f"{single_label}: {len(single)} rows, {dates[0]} -> {dates[-1]}, "
          f"initial {initial_amount:.2f} [{single_file}]")

    combos = []
    for mapping in rq1_config.CONFIDENCE_MAPPINGS:
        for safe in rq1_config.SAFE_STRATEGIES:
            name = rq1_config.combo_name(mapping, safe)
            path = os.path.join(results_root, name, "test_account.csv")
            if os.path.exists(path):
                combos.append((name, load_account(path)))
            else:
                # Also check canonical root for reuse (if temp run reuses some combos)
                alt = os.path.join(rq1_config.RESULTS_ROOT, name, "test_account.csv")
                if alt != path and os.path.exists(alt) and results_root != rq1_config.RESULTS_ROOT:
                    # Only warn, don't double-count; user wants temp-isolated
                    print(f"Missing {path}, skipping (alt exists at {alt} but not copying)")
    print(f"Found {len(combos)} combos in {results_root}: {[c[0] for c in combos]}")

    # ---- Baseline: ensemble average (no safe, no confidence) ----
    baseline_path = os.path.join(results_root, "baseline_ensemble_average", "test_account.csv")
    if not os.path.exists(baseline_path):
        fallback = os.path.join(rq1_config.RESULTS_ROOT, "baseline_ensemble_average", "test_account.csv")
        if os.path.exists(fallback):
            baseline_path = fallback
    if os.path.exists(baseline_path):
        baseline_df = load_account(baseline_path)
        combos.append(("baseline_ensemble_average", baseline_df))

    # ---- Build the aligned result frame with shared baselines ----
    result = pd.DataFrame({"date": dates})
    result[single_label] = single.set_index("date")["value"].reindex(dates).values
    for name, df in combos:
        result[f"{name.replace('combo_', '')}"] = (
            df.set_index("date")["value"].reindex(dates).values
        )
    result = attach_baselines(result, dates, initial_amount=initial_amount)
    result = result.dropna().reset_index(drop=True)
    print(f"Aligned rows: {len(result)}")

    # ---- Performance comparison table ----
    cols = [c for c in result.columns if c != "date"]
    series = []
    for col in cols:
        df = result[["date", col]].rename(columns={col: "account_value"})
        series.append(backtest_stats(df, value_col_name="account_value").rename(col))
    tag = tag_suffix
    table = pd.concat(series, axis=1).round(4)
    table.to_csv(os.path.join(results_root, f"backtest_performance_comparison{tag}.csv"))
    print(f"\n=== RQ1 Performance Comparison ({single_label}) ===")
    print(table)

    # ---- Equity curve plot ----
    plt.rcParams["figure.figsize"] = (15, 6)
    fig, ax = plt.subplots()
    for col in cols:
        disp = _display_label(col)
        if col in LABELS.values():
            ax.plot(pd.to_datetime(result["date"]), result[col], label=disp, lw=1.0,
                    linestyle="--", alpha=0.8)
        elif col in (single_label, "Single PPO", "Single PPO 5M"):
            ax.plot(pd.to_datetime(result["date"]), result[col], label=disp, lw=1.8,
                    color="black")
        else:
            ax.plot(pd.to_datetime(result["date"]), result[col], label=disp, lw=1.4)
    ax.set_title(f"RQ1 — Portfolio Value Over Time (Test Set: 2022-2026) [{results_root}]")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend(fontsize=8, ncol=3, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(results_root, f"backtest_result{tag}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    # also write a copy under report with tag
    try:
        table.to_csv(os.path.join("results/report", f"rq1_backtest_performance_comparison{tag}.csv"))
    except OSError as exc:
        print(f"[warn] could not write the results/report copy of the table: {exc}")

    # ---- Disagreement / confidence timeseries (best combo) ----
    summary_path = os.path.join(results_root, "test_summary.csv")
    if not os.path.exists(summary_path):
        summary_path = os.path.join(rq1_config.RESULTS_ROOT, "test_summary.csv")
    test_summary = pd.read_csv(summary_path)
    # Prefer best by Sharpe if available
    if "sharpe" in test_summary.columns:
        best = test_summary[test_summary["label"].str.startswith("combo_")].sort_values("sharpe", ascending=False).iloc[0]
    else:
        best = test_summary[test_summary["label"].str.startswith("combo_")].iloc[0]
    best_actions_path = os.path.join(results_root, best['label'], "test_actions.csv")
    if not os.path.exists(best_actions_path):
        best_actions_path = os.path.join(rq1_config.RESULTS_ROOT, best['label'], "test_actions.csv")
    best_actions = pd.read_csv(best_actions_path)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    ax1.plot(pd.to_datetime(best_actions["date"]), best_actions["disagreement"],
             linewidth=0.8, color="steelblue")
    ax1.set_ylabel("Disagreement (TVD)")
    ax1.set_title(f"RQ1 — {_display_label(best['label'].replace('combo_',''))}: disagreement & confidence")
    ax1.grid(True, alpha=0.3)
    ax2.plot(pd.to_datetime(best_actions["date"]), best_actions["confidence"],
             linewidth=0.8, color="darkgreen")
    ax2.set_ylabel("Confidence")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(results_root, "disagreement_confidence_timeseries.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- Histograms ----
    for key, color, title, out in [
        ("disagreement", "steelblue", "Distribution of Ensemble Disagreement (Test Set)",
         "disagreement_histogram.png"),
        ("confidence", "darkgreen", "Distribution of Ensemble Confidence (Test Set)",
         "confidence_histogram.png"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(best_actions[key], bins=50, color=color, edgecolor="white", alpha=0.8)
        ax.axvline(best_actions[key].mean(), color="red", linestyle="--",
                   label=f'Mean = {best_actions[key].mean():.4f}')
        ax.set_xlabel(key.capitalize())
        ax.set_ylabel("Frequency")
        ax.set_title(title + f" ({_display_label(best['label'].replace('combo_',''))})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(results_root, out), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ---- Cash-weight diagnostics ----
    try:
        # Use cash_diagnostics.csv if produced by rq1_test.py; otherwise derive from test_summary
        cash_diag_path = os.path.join(results_root, "cash_diagnostics.csv")
        if os.path.exists(cash_diag_path):
            cash_df = pd.read_csv(cash_diag_path)
        else:
            cash_df = test_summary

        # Bar chart: mean final cash per combo (sorted)
        combo_cash = cash_df[cash_df["label"].str.startswith("combo_")].copy()
        if "mean_final_cash" in combo_cash.columns and not combo_cash.empty:
            combo_cash = combo_cash.sort_values("mean_final_cash")
            fig, ax = plt.subplots(figsize=(12, 6))
            colors = []
            for _, row in combo_cash.iterrows():
                s = str(row["safe_strategy"])
                if "equal_weight_stocks" in s:
                    colors.append("#e74c3c")  # red for stocks-only
                elif "equal_weight" in s:
                    colors.append("#2ecc71")  # green for 1/31
                else:
                    colors.append("#3498db")  # blue for previous
            ax.barh(combo_cash["label"], combo_cash["mean_final_cash"], color=colors, edgecolor="white")
            if "mean_safe_cash" in combo_cash.columns:
                ax.barh(combo_cash["label"], combo_cash["mean_safe_cash"], color="none", edgecolor="black", linestyle="--", alpha=0.5, label="mean safe cash")
            ax.set_xlabel("Mean Cash Weight (final allocation)")
            ax.set_title("RQ1 — Mean Cash Weight by Combo (diagnostic)")
            # Annotate values
            for i, (lbl, val) in enumerate(zip(combo_cash["label"], combo_cash["mean_final_cash"])):
                ax.text(val + 0.005, i, f"{val:.3f}", va="center", fontsize=8)
            ax.legend(handles=[
                plt.Rectangle((0,0),1,1, color="#3498db", label="Previous"),
                plt.Rectangle((0,0),1,1, color="#2ecc71", label="Equal Weight (1/31)"),
                plt.Rectangle((0,0),1,1, color="#e74c3c", label="Equal Weight Stocks (1/30, 0 cash)"),
            ], fontsize=8)
            ax.grid(True, axis="x", alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(results_root, "cash_weight_comparison.png"), dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved cash diagnostics to {os.path.join(results_root,'cash_weight_comparison.png')}")

        # Cash timeseries for best per safe_strategy
        safe_strats = rq1_config.SAFE_STRATEGIES
        fig, axes = plt.subplots(len(safe_strats), 1, figsize=(15, 3*len(safe_strats)), sharex=True)
        if len(safe_strats) == 1:
            axes = [axes]
        for ax, safe in zip(axes, safe_strats):
            sub = test_summary[test_summary["safe_strategy"] == safe]
            if sub.empty:
                ax.set_visible(False)
                continue
            best_safe = sub.sort_values("sharpe", ascending=False).iloc[0]
            path = os.path.join(results_root, best_safe["label"], "test_actions.csv")
            if not os.path.exists(path):
                path = os.path.join(rq1_config.RESULTS_ROOT, best_safe["label"], "test_actions.csv")
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path)
            dates_ts = pd.to_datetime(df["date"])
            if "final_alloc_0" in df.columns:
                ax.plot(dates_ts, df["final_alloc_0"], lw=0.8, label="final cash", color="#2c3e50")
            if "safe_alloc_0" in df.columns:
                ax.plot(dates_ts, df["safe_alloc_0"], lw=0.8, label="safe cash", linestyle="--", alpha=0.7)
            if "ensemble_alloc_0" in df.columns:
                ax.plot(dates_ts, df["ensemble_alloc_0"], lw=0.6, label="ensemble cash", alpha=0.5)
            ax.set_ylabel("Cash weight")
            ax.set_title(f"{_display_label(best_safe['label'].replace('combo_',''))} (sharpe {best_safe.get('sharpe',0):.3f}) — cash decomposition")
            ax.legend(fontsize=7, ncol=3)
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("Date")
        fig.tight_layout()
        fig.savefig(os.path.join(results_root, "cash_timeseries_by_safe.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Save a combined cash summary table for report
        if "mean_final_cash" in test_summary.columns:
            cash_table = test_summary[["label","safe_strategy","mapping","sharpe","total_return","max_drawdown",
                                       "mean_final_cash","mean_safe_cash","mean_ensemble_cash","mean_confidence"]].copy()
            # For stocks-only, safe cash should be 0; highlight delta vs 1/31
            cash_table = cash_table.sort_values("sharpe", ascending=False)
            cash_table.to_csv(os.path.join(results_root, "cash_performance_table.csv"), index=False)
            print(f"Cash performance table saved")

    except Exception as e:
        print(f"Cash diagnostics failed: {e}")
        import traceback
        traceback.print_exc()

    # ---- Manifest ----
    manifest = {
        "research_question": "RQ1 - inter-seed uncertainty (5 PPO seeds) + equal_weight_stocks ablation",
        "models": {
            "lineage": "all trained in rq1_train.py",
            "n_ensemble": rq1_config.N_ENSEMBLE,
            "base_seed": rq1_config.BASE_SEED,
            "seeds": list(range(rq1_config.BASE_SEED,
                                rq1_config.BASE_SEED + rq1_config.N_ENSEMBLE)),
            "total_timesteps": rq1_config.TOTAL_TIMESTEPS,
            "total_timesteps_single_5m": rq1_config.TOTAL_TIMESTEPS_SINGLE_5M,
            "single_file_used": single_file,
            "single_label": single_label,
            "model_dir": rq1_config.MODEL_DIR,
        },
        "data": {
            "train": rq1_config.DATA_TRAIN,
            "test": rq1_config.DATA_TEST,
            "val_window": [rq1_config.VAL_START, rq1_config.VAL_END],
            "test_window": f"{dates[0]} to {dates[-1]}",
        },
        "d_ref": {
            "method": rq1_config.D_REF_METHOD,
            "all_methods": rq1_config.D_REF_METHODS,
            "calibration_source": summary_path,
        },
        "combos": {
            "confidence_mappings": rq1_config.CONFIDENCE_MAPPINGS,
            "safe_strategies": rq1_config.SAFE_STRATEGIES,
            "cross_product": len(rq1_config.CONFIDENCE_MAPPINGS) * len(rq1_config.SAFE_STRATEGIES),
            "note": "equal_weight = 1/31 inc cash; equal_weight_stocks = 1/30 stocks only (0 cash)",
        },
        "env_kwargs": rq1_config.ENV_KWARGS,
        "baselines": {
            "source": "results/baselines/ (compute_baselines.py)",
            "names": NAMES,
            "labels": LABELS,
            "note": "External EW baselines are stocks-only 1/30, no cash",
        },
        "cash_diagnostic": {
            "safe_equal_weight": "1/31 cash+stocks",
            "safe_equal_weight_stocks": "0 cash, 1/30 stocks",
            "outputs": ["cash_diagnostics.csv", "cash_weight_comparison.png", "cash_timeseries_by_safe.png", "cash_performance_table.csv"],
        },
        "outputs": results_root,
        "canonical_results_root": rq1_config.RESULTS_ROOT,
        "temp_note": "Temporary ablation reporting; 0rq1 unchanged",
    }
    with open(os.path.join(results_root, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved to {os.path.join(results_root,'manifest.json')}")
    print("Done.")


if __name__ == "__main__":
    main()
