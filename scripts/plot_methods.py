#!/usr/bin/env python
"""Histograms of seats by party under each voting method, in the style of img/senate-hist-YYYY.png.

  .venv/bin/python scripts/plot_methods.py            # reads outputs/compass/methods/seats.npz

Writes, under img/compass/:
  <method>/<chamber>-hist-<year>.png     one histogram per method, chamber and year (the old series)
  compare/<chamber>-methods-<year>.png   all methods for one chamber-year on one page
  <chamber>-methods-over-time.png        mean seats by method across 1976-2020
"""
import argparse, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from compass.methods import METHODS, TITLES

COLORS = ["blue", "red", "green", "gray"]            # Democrat, Republican, Other, Libertarian (as in senate.py)
CHAMBER = {"H": ("house", "House"), "S": ("senate", "Senate")}


def draw(ax, sims, actual, parties, legend=True):
    """sims [I, 4] seats per iteration; actual [4]. Same encoding as the dice-roll plots."""
    lo, hi = min(sims.min(), actual.min()), max(sims.max(), actual.max())
    bins = np.arange(lo - 0.5, hi + 1.5, max(1, int(np.ceil((hi - lo + 1) / 60))))
    top = 0
    for p, name in enumerate(parties):
        h = ax.hist(sims[:, p], bins, color=COLORS[p], alpha=0.3, label=name)
        top = max(top, h[0].max())
    for p in range(len(parties)):
        ax.vlines(actual[p], 0, top, color=COLORS[p], label=f"By Count: {actual[p]}")
    for p in range(len(parties)):
        ax.vlines(sims[:, p].mean(), 0, top, color=COLORS[p], linestyles="dashed", label=f"Sim Mean: {sims[:, p].mean():.1f}")
    if legend:
        ax.legend(fontsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="outputs/compass/methods/seats.npz"); ap.add_argument("--out", default="img/compass")
    ap.add_argument("--skip-series", action="store_true"); ap.add_argument("--label", default="compass agents")
    a = ap.parse_args()
    z = np.load(a.npz); years = z["years"]; parties = list(z["parties"]); out = Path(a.out)
    for ch, (slug, label) in CHAMBER.items():
        actual = z[f"actual_{ch}"]
        for t, yr in enumerate(years):
            if not a.skip_series:
                for m in METHODS:
                    fig, ax = plt.subplots(figsize=(14, 7))
                    draw(ax, z[f"{m}_{ch}"][t], actual[t], parties)
                    ax.set_title(f"Histogram of {TITLES[m]} voting outcomes for {label} Election in {yr} ({a.label})")
                    ax.set_xlabel(f"Number of {label} Seats Awarded to each Party")
                    ax.set_ylabel("Count of Scenarios for Each Balance of Power")
                    (out / m).mkdir(parents=True, exist_ok=True)
                    fig.savefig(out / m / f"{slug}-hist-{yr}.png", dpi=70); plt.close(fig)
            fig, axes = plt.subplots(len(METHODS), 1, figsize=(13, 2.6 * len(METHODS)), sharex=True)
            for ax, m in zip(axes, METHODS):
                draw(ax, z[f"{m}_{ch}"][t], actual[t], parties, legend=False)
                mean = z[f"{m}_{ch}"][t].mean(0)
                ax.set_title(f"{TITLES[m]}:  " + "   ".join(f"{p} {mean[i]:.1f}" for i, p in enumerate(parties)), fontsize=10, loc="left")
            axes[-1].set_xlabel(f"Number of {label} seats  (solid line = actual result, dashed = simulation mean)")
            h, l = axes[0].get_legend_handles_labels()
            fig.legend(h[:4], l[:4], loc="upper right", ncol=4, fontsize=9)
            fig.suptitle(f"{label} {yr}: seats by party under each voting method ({a.label})", x=0.02, ha="left", fontsize=13)
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            (out / "compare").mkdir(parents=True, exist_ok=True)
            fig.savefig(out / "compare" / f"{slug}-methods-{yr}.png", dpi=80); plt.close(fig)
        # over time: mean seats by method for each party
        fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
        for p, ax in enumerate(axes.flat):
            ax.plot(years, actual[:, p], color="black", lw=2.5, label="actual")
            for m in METHODS:
                s = z[f"{m}_{ch}"][:, :, p]
                ax.plot(years, s.mean(1), marker="o", ms=3, lw=1.2, label=TITLES[m], ls="--" if m == "random_ballot" else "-")
            ax.set_title(f"{parties[p]} seats"); ax.grid(alpha=0.25)
        axes[0, 0].legend(fontsize=8); fig.suptitle(f"{label}: mean seats by voting method, 1976-2020 ({a.label})")
        fig.tight_layout(); fig.savefig(out / f"{slug}-methods-over-time.png", dpi=90); plt.close(fig)
        print("wrote", label)


if __name__ == "__main__":
    main()
