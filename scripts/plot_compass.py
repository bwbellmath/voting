#!/usr/bin/env python
"""Plot a fitted compass: voter types, state trajectories, party/candidate positions, and
predicted-vs-actual on the held-out races of that run.

  .venv/bin/python scripts/plot_compass.py outputs/compass/PSH_random_d2K4_s0
"""
import json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from compass import evaluate as ev
from compass.data import build_races, to_tensors, YEARS, OFFICES
from compass.model import Config, build

BLUE, RED, GREY = "#2b6cb0", "#c53030", "#718096"


def load(stem):
    ck = torch.load(f"{stem}.pt"); args = json.loads(Path(f"{stem}.json").read_text())["args"]
    rd = to_tensors(build_races(offices=tuple(ck["offices"])))
    cfg = Config(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in ck["cfg"].items()})
    m = build(rd, cfg, args["seed"]); m.load_state_dict(ck["state"]); m.eval()
    return m, rd, args


def rotation(m):
    """Axis 1 = mean Democrat->Republican direction; axis 2 = orthogonal (d=2 only gets a full basis)."""
    with torch.no_grad():
        v = (m.party_pos[:, 1] - m.party_pos[:, 0]).mean(0).numpy()
    e1 = v / np.linalg.norm(v)
    if len(e1) == 1:
        return np.array([[np.sign(e1[0])]])
    q, _ = np.linalg.qr(np.column_stack([e1, np.eye(len(e1))[:, : len(e1) - 1]]))
    q[:, 0] *= np.sign(q[:, 0] @ e1)
    return q


def main(stem):
    m, rd, args = load(stem)
    Q = rotation(m); d = m.cfg.d
    S, T = len(rd.states), len(YEARS)
    with torch.no_grad():
        st = torch.arange(S).repeat_interleave(T); yr = torch.arange(T).repeat(S)
        pi = m.weights(st, yr).view(S, T, -1).numpy()
        mu, sig = m.mu.numpy() @ Q, m.log_sig.exp().numpy()
        pp = m.party_pos.numpy() @ Q
        nat = m.national_mean().numpy() @ Q
    means = pi @ mu                                                    # [S,T,d]
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    a = ax[0]
    y2 = (lambda x: x[..., 1]) if d > 1 else (lambda x: np.zeros_like(x[..., 0]))
    for k in range(len(mu)):
        a.scatter(mu[k, 0], y2(mu[k]), s=4000 * pi[:, :, k].mean(), color=GREY, alpha=0.18, zorder=1)
        a.annotate(f"type {k + 1}\n{100 * pi[:, :, k].mean():.0f}%", (mu[k, 0], y2(mu[k])), ha="center", color=GREY, fontsize=9)
    for s, name in enumerate(rd.states):
        a.annotate("", xy=(means[s, -1, 0], y2(means[s, -1])), xytext=(means[s, 0, 0], y2(means[s, 0])),
                   arrowprops=dict(arrowstyle="->", color="k", alpha=0.35, lw=0.8))
        a.annotate(name, (means[s, -1, 0], y2(means[s, -1])), fontsize=7, ha="center", va="center",
                   bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))
    for p, col, lab in [(0, BLUE, "Democratic party"), (1, RED, "Republican party")]:
        a.plot(pp[:, p, 0], y2(pp[:, p]), "-o", color=col, ms=3, lw=1.5, label=f"{lab} 1976→2020")
        a.annotate("2020", (pp[-1, p, 0], y2(pp[-1, p])), color=col, fontsize=8, xytext=(4, 4), textcoords="offset points")
        a.annotate("1976", (pp[0, p, 0], y2(pp[0, p])), color=col, fontsize=8, xytext=(4, 4), textcoords="offset points")
    a.set_title("Fitted compass: state mean voter position, 1976 (tail) → 2020 (label)")
    a.set_xlabel("axis 1: Democrat ← → Republican (party-line direction)"); a.set_ylabel("axis 2 (orthogonal)")
    a.legend(loc="best", fontsize=8); a.grid(alpha=0.2); a.set_aspect("equal", "datalim")

    b = ax[1]
    if args["split"] == "random":
        tr, te = ev.random_split(rd, 0.2, args["seed"])
    elif args["split"] == "temporal":
        tr, te = ev.temporal_split(rd, 2016)
    else:
        te = np.arange(len(rd))
    test = rd.subset(te)
    with torch.no_grad():
        p = ev.d_share(m(test), test)
    ok, yd = test.two_party()
    for o, (name, col) in enumerate(zip(["President (state)", "Senate", "House"], ["k", "#d69e2e", "#38a169"])):
        k = (ok & (test.office == o)).numpy()
        if k.sum():
            rm = 100 * np.sqrt(((p.numpy()[k] - yd.numpy()[k]) ** 2).mean())
            b.scatter(100 * p.numpy()[k], 100 * yd.numpy()[k], s=6 if o == 2 else 14, alpha=0.35 if o == 2 else 0.8,
                      color=col, label=f"{name}: n={k.sum()}, RMSE {rm:.1f} pts", zorder=3 - o)
    b.plot([0, 100], [0, 100], color=GREY, lw=1); b.axhline(50, color=GREY, lw=0.5); b.axvline(50, color=GREY, lw=0.5)
    b.set_xlabel("predicted Democratic share of two-party vote (%)"); b.set_ylabel("actual (%)")
    b.set_title(f"Held-out races ({args['split']} split): predicted vs actual"); b.legend(fontsize=9); b.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(f"{stem}.png", dpi=110)
    print("wrote", f"{stem}.png")


if __name__ == "__main__":
    main(sys.argv[1])
