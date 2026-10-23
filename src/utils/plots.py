"""Precision-recall curves and one asset's risk timeline."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

from src.utils.paths import FIGURES as DOCS
from src.models.evaluate import TOP_K

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {"Keyword count": "#eda100", "History only (no text)": "#1baf7a",
          "Text + history, logistic": "#2a78d6", "Text + history, LightGBM": "#eb6834"}


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def draw(test, scores, failures, episodes, chosen):
    DOCS.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK})

    fig, ax = plt.subplots(figsize=(7.5, 5.2), dpi=150)
    _style(ax)
    for name, s in scores.items():
        p, r, _ = precision_recall_curve(test.label, s)
        ap = average_precision_score(test.label, s)
        ax.plot(r, p, color=COLORS[name], linewidth=2.2 if name == chosen else 1.6,
                label=f"{name}  (PR-AUC {ap:.2f})")
    base = test.label.mean()
    ax.axhline(base, color=MUTED, linestyle="--", linewidth=1)
    ax.text(0.12, base + 0.015, f"random ranking ({base:.0%} of asset-weeks break down within 30 days)",
            ha="left", fontsize=7.5, color=MUTED)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall: share of asset-weeks before a breakdown that are flagged", fontsize=9.5)
    ax.set_ylabel("Precision: share of flags followed by a breakdown", fontsize=9.5)
    ax.set_title("Will this asset break down in the next 30 days? (test years 2024-25)", fontsize=11.5,
                 loc="left", pad=10)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.text(0.01, 0.01, "Synthetic data. Models trained on 2020-23 only.", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(DOCS / "precision_recall.png")
    plt.close(fig)

    # Timeline: pick a test-period asset that has both a caught breakdown and a repaired degradation.
    s = test.assign(score=scores[chosen])
    s["rank"] = s.groupby("snapshot").score.rank(ascending=False, method="first")
    thr = s[s["rank"] == TOP_K].set_index("snapshot").score.sort_index()
    fx = episodes[(episodes.outcome == "fixed") & (episodes.end >= test.snapshot.min())]
    bd = failures[(failures.date >= test.snapshot.min()) & (failures.failure_mode != "sudden")]
    best, best_n = None, -1
    for a in sorted(set(fx.asset_id) & set(bd.asset_id)):
        n = (s[(s.asset_id == a)]["rank"] <= TOP_K).sum()
        if n > best_n:
            best, best_n = a, n
    a = s[s.asset_id == best].sort_values("snapshot")
    fig, ax = plt.subplots(figsize=(8.5, 3.6), dpi=150)
    _style(ax)
    ax.plot(a.snapshot, a.score, color=COLORS[chosen], linewidth=1.8, label="Breakdown risk score (weekly)")
    ax.plot(thr.index, thr.values, color=MUTED, linewidth=1, linestyle=":", label=f"Score needed to make the weekly top {TOP_K}")
    ymax = max(a.score.max(), thr.max()) * 1.15
    for r in bd[bd.asset_id == best].itertuples():
        ax.axvline(r.date, color="#e34948", linewidth=1.4)
        ax.text(r.date, ymax * 0.97, " breakdown", color=INK, fontsize=8, va="top")
    for r in fx[fx.asset_id == best].itertuples():
        ax.axvline(r.end, color="#1baf7a", linewidth=1.4, linestyle="--")
        ax.text(r.end, ymax * 0.82, " repaired before\n breakdown", color=INK, fontsize=8, va="top")
    ax.set_ylim(0, ymax)
    ax.set_ylabel("Risk score", fontsize=9.5)
    ax.set_title(f"One asset over the test years: {best}", fontsize=11, loc="left", pad=8)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.12), ncol=2)
    fig.tight_layout()
    fig.savefig(DOCS / "asset_timeline.png")
    plt.close(fig)
