#!/usr/bin/env python
"""Plots for outputs/compass/scenarios/ -> img/compass/scenarios/."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compass.strategy import LABELS
from plot_methods import draw

SRC, OUT = Path("outputs/compass/scenarios"), Path("img/compass/scenarios")
STYLE = {"plurality_strategic": ("black", "-"), "plurality_honest": ("black", ":"), "random_ballot": ("#718096", "--"),
         "rcv_honest": ("#2b6cb0", "-"), "rcv_strategic": ("#2b6cb0", "--"), "approval_honest": ("#c53030", "-"),
         "approval_strategic": ("#c53030", "--"), "star_honest": ("#d69e2e", "-"), "star_strategic": ("#d69e2e", "--"),
         "borda": ("#38a169", "-"), "condorcet": ("#805ad5", "-")}
POS = {"center": "Centrist entrant (midway between the parties)", "left_flank": "Left-flank entrant", "right_flank": "Right-flank entrant"}


def lines(ax, d, x, y, methods):
    for m in methods:
        k = d[d.method == m].sort_values(x)
        ax.plot(k[x], k[y], color=STYLE[m][0], ls=STYLE[m][1], marker="o", ms=3, lw=1.6, label=LABELS[m])
    ax.grid(alpha=0.25)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sw = pd.read_csv(SRC / "desertion_sweep.csv"); ent = pd.read_csv(SRC / "entrant_scenarios.csv")
    info = __import__("json").loads((SRC / "summary.json").read_text())

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    lines(ax[0], sw, "rho", "third_party_wins", [m for m in STYLE])
    ax[0].axhline(info["actual_third_party_wins"], color="k", lw=0.8); ax[0].set_yscale("log")
    ax[0].set_xlabel("assumed share of third-party supporters who desert under plurality (rho)")
    ax[0].set_ylabel(f"contests won by a third-party candidate (of {info['contests']:,}; log scale)")
    ax[0].set_title("Third-party wins if real votes hide sincere support"); ax[0].legend(fontsize=8)
    lines(ax[1], sw, "rho", "house_seats_from_random_ballot", [m for m in STYLE if m != "random_ballot"])
    ax[1].set_xlabel("rho"); ax[1].set_ylabel("House seats that differ from random ballot's expected split (per year)")
    ax[1].set_title("Distance of each method from random ballot")
    fig.tight_layout(); fig.savefig(OUT / "desertion_sweep.png", dpi=100); plt.close(fig)

    for k in sorted(ent.stretch.unique()):
        e = ent[ent.stretch == k]; tag = "fitted" if k == 1 else f"stretch{k:g}"
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
        for ax, p in zip(axes, POS):
            lines(ax, e[e.position == p], "entrant_first_choice", "entrant_win_rate", [m for m in STYLE])
            ax.set_title(POS[p]); ax.set_xlabel("entrant's sincere first-choice share")
        axes[0].set_ylabel("share of contests the entrant wins"); axes[0].legend(fontsize=8)
        fig.suptitle(f"Hypothetical third candidate in real two-party contests ({'fitted compass' if k == 1 else f'compass stretched x{k:g}: more policy-driven voters'})")
        fig.tight_layout(); fig.savefig(OUT / f"entrant_wins_{tag}.png", dpi=100); plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
        base = e[e.method == "condorcet"].set_index(["position", "entrant_first_choice"]).expression
        for ax, p in zip(axes, POS):
            d = e[e.position == p].copy()
            d["loss"] = [base[(p, s)] - x for s, x in zip(d.entrant_first_choice, d.expression)]
            lines(ax, d, "entrant_first_choice", "loss", ["plurality_strategic", "rcv_honest", "rcv_strategic", "approval_honest",
                                                          "approval_strategic", "star_honest", "star_strategic"])
            ax.set_title(POS[p]); ax.set_xlabel("entrant's sincere first-choice share")
        axes[0].set_ylabel("compass-expression loss vs Condorcet winner (higher = worse)"); axes[0].legend(fontsize=8)
        fig.suptitle(f"Honest vs pragmatic (mainstream-first) ballots ({'fitted compass' if k == 1 else f'stretched x{k:g}'})")
        fig.tight_layout(); fig.savefig(OUT / f"strategy_expression_{tag}.png", dpi=100); plt.close(fig)

    z = np.load(SRC / "random_ballot_sincere.npz"); years = list(z["years"]); parties = list(z["parties"]); rhos = z["rhos"]
    for yr in (1992, 2020):
        t = years.index(yr)
        fig, axes = plt.subplots(len(rhos), 1, figsize=(13, 2.7 * len(rhos)), sharex=True)
        for ax, r in zip(axes, rhos):
            s = z[f"H_rho{r}"][t]; draw(ax, s, z["actual_H"][t], parties, legend=False)
            ax.set_title(f"rho = {r:g}:  " + "   ".join(f"{p} {s[:, i].mean():.1f}" for i, p in enumerate(parties)), fontsize=10, loc="left")
        axes[-1].set_xlabel("House seats (solid = actual result, dashed = random-ballot mean)")
        fig.suptitle(f"House {yr}: random ballot if voters vote sincerely, by assumed desertion rate rho", x=0.02, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(OUT / f"random_ballot_house_{yr}.png", dpi=90); plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for r in rhos:
        ax.plot(years, z[f"H_rho{r}"][:, :, 2:].sum(-1).mean(1), marker="o", ms=3, label=f"rho = {r:g}")
    ax.set_ylabel("expected third-party House seats under random ballot"); ax.legend(); ax.grid(alpha=0.25)
    ax.set_title("Random ballot with sincere voting: third-party House seats, 1976-2020")
    fig.tight_layout(); fig.savefig(OUT / "random_ballot_third_party_over_time.png", dpi=100); plt.close(fig)

    g = pd.read_csv(SRC / "gallup_vs_compass.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for c, col in zip(["D", "lean_D", "pure_I", "lean_R", "R"], ["#2b6cb0", "#63b3ed", "#718096", "#fc8181", "#c53030"]):
        ax.plot(g.index, g[f"gallup_{c}"], color=col, lw=2, label=f"Gallup {c}")
        ax.plot(g.index, g[f"compass_{c}"], color=col, lw=1.5, ls="--", marker="o", ms=3)
    ax.set_ylabel("% of adults / agents"); ax.set_title("Five-way party identification: Gallup (solid) vs compass agents (dashed)")
    ax.legend(ncol=5, fontsize=8); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(OUT / "gallup_vs_compass.png", dpi=100); plt.close(fig)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
