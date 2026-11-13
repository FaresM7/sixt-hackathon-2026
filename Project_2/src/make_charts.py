"""Two slide charts from the committed result files (no model runs).

results/plots/forecast_by_horizon.png : MAE by days since the last reading, ours vs baselines
results/plots/anomaly_vs_starter.png  : alarm rate and large-error recall, ours vs starter rules
Usage: python -m src.make_charts
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .series import HERE

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
OURS, SEG, STARTER = "#2a78d6", "#eb6834", "#1baf7a"     # categorical slots 1-3 (validated)


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def forecast_chart(res, out):
    t = pd.read_csv(res / "forecast_val_tuned.csv")
    t = t[(t.targets == "all") & (t.gap_days != "all")]
    buckets = ["1-14", "15-30", "31-60", "61-90", "91-150"]
    series = [("m_lgbm", "Our model (LightGBM)", OURS), ("b2_segment", "Segment average rate", SEG),
              ("b1_cum_rate", "Car's own average rate (starter)", STARTER)]
    fig, ax = plt.subplots(figsize=(10, 5.2), facecolor=SURFACE)
    _style(ax)
    x, w = np.arange(len(buckets)), 0.26
    for k, (m, label, c) in enumerate(series):
        v = t.set_index(["model", "gap_days"]).loc[m].reindex(buckets).MAE.to_numpy()
        bars = ax.bar(x + (k - 1) * (w + 0.02), v, w, color=c, label=label, edgecolor=SURFACE, linewidth=2)
        for b, val in zip(bars, v):
            ax.text(b.get_x() + b.get_width() / 2, val + 40, f"{val:,.0f}", ha="center", va="bottom",
                    fontsize=8, color=INK if m == "m_lgbm" else INK2,
                    fontweight="bold" if m == "m_lgbm" else "normal")
    ax.set_xticks(x, [f"{b} days" for b in buckets], color=INK2)
    ax.set_ylabel("Mean absolute error (km)", color=INK2)
    ax.set_title("Forecast error by days since the last known reading (held-out vehicles)",
                 color=INK, loc="left", fontsize=12)
    ax.legend(frameon=False, labelcolor=INK2, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "forecast_by_horizon.png", dpi=150, facecolor=SURFACE)


def anomaly_chart(res, out):
    t = pd.read_csv(res / "anomaly_eval.csv")
    st = t[t.detector == "starter_rules (baseline)"].iloc[0]
    ours = t[(t.detector == "rules+two_sided") & (t.budget == "0.5%")].iloc[0]
    names, cols = ["Starter rules", "Our detector"], [STARTER, OURS]
    panels = [("Readings flagged (review workload)", [st.flag_rate_real * 100, ours.flag_rate_real * 100], "{:.1f}%"),
              ("Planted errors >= 1,000 km caught", [st.recall_err_ge_1000km * 100, ours.recall_err_ge_1000km * 100], "{:.0f}%"),
              ("Flags that are planted errors (lower bound)", [st.precision_lb * 100, ours.precision_lb * 100], "{:.0f}%")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), facecolor=SURFACE)
    for ax, (title, vals, fmt) in zip(axes, panels):
        _style(ax)
        bars = ax.bar(names, vals, 0.5, color=cols, edgecolor=SURFACE, linewidth=2)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v * 1.02, fmt.format(v), ha="center", va="bottom",
                    fontsize=11, color=INK, fontweight="bold")
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        ax.set_ylim(0, max(vals) * 1.2)
        ax.tick_params(axis="x", colors=INK2)
        ax.set_yticks([])
        ax.yaxis.grid(False)
    fig.suptitle("Erroneous mileage detection: injected-error evaluation on held-out vehicles",
                 color=INK, x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "anomaly_vs_starter.png", dpi=150, facecolor=SURFACE)


def main() -> None:
    res = HERE / "results"
    out = res / "plots"
    out.mkdir(parents=True, exist_ok=True)
    forecast_chart(res, out)
    anomaly_chart(res, out)
    print("wrote", out / "forecast_by_horizon.png", out / "anomaly_vs_starter.png")


if __name__ == "__main__":
    main()
