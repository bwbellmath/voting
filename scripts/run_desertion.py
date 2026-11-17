#!/usr/bin/env python
"""Measure third-party suppression per year, then re-run every election with that suppression removed.

  .venv/bin/python scripts/run_desertion.py --iters 200 --voters 1000        # ~6 min
  .venv/bin/python scripts/plot_methods.py --npz outputs/compass/desertion/seats.npz \\
        --out img/compass/desertion --label "suppression-adjusted compass agents"

Steps
  1. measure_suppression: per-year rate from third-party candidates' appeal with vs without both majors
  2. re-solve candidate appeal so that plurality *with that suppression* reproduces the real results
  3. agents then vote sincerely under every method (including random ballot)
"""
import argparse, json, sys, time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_compass import load
from compass import simulate as sim, strategy as st
from compass.data import YEARS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="outputs/compass/full_d2K4"); ap.add_argument("--out", default="outputs/compass/desertion")
    ap.add_argument("--iters", type=int, default=200); ap.add_argument("--voters", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    model, _, _ = load(a.model)
    ct, c, v0, _ = sim.prepare(model, verbose=False)
    torch.set_grad_enabled(False)
    per_year, info, rho_total, _ = st.measure_suppression(ct, v0)
    per_year.to_csv(out / "suppression_by_year.csv", index=False)
    print(per_year[["year", "one_major_candidates", "gap_raw", "gap_se", "suppression_mean"]].round(3).to_string(index=False))

    C = len(ct.ids); multi = torch.where(ct.ncand >= 3)[0]
    front = torch.tensor([[0, 1]]).expand(len(multi), 2).contiguous()
    ym = ct.y[multi]; top_third = (ym[:, 2:] * ct.mask[multi][:, 2:]).max(1).values
    cap = (1 - top_third / ym[:, 1]).clamp(0, 1)                      # a third candidate cannot sincerely out-poll the runner-up
    rho_c = torch.minimum(rho_total[multi], cap)
    v_adj, err = st.calibrate_strategic(model, ct.pi[multi], c[multi], ym, ct.mask[multi], front, rho_c)
    v = v0.clone(); v[multi] = v_adj
    sincere = ct.y.clone(); sincere[multi] = sim.expected_shares(model, ct.pi[multi], c[multi], v_adj, ct.mask[multi])
    minor = (ct.group >= 2) & ct.mask
    info.update({"calibration_max_abs_err": err, "contests": C, "contests_adjusted": int((rho_c > 0).sum()),
                 "contests_capped": int((rho_c < rho_total[multi] - 1e-6).sum()),
                 "minor_first_choice_pct_observed": 100 * (ct.y * minor).sum().item() / C,
                 "minor_first_choice_pct_sincere": 100 * (sincere * minor).sum().item() / C})
    print(f"strategic plurality reproduces real shares to {err:.1e}; third-party first-choice share "
          f"{info['minor_first_choice_pct_observed']:.1f}% observed -> {info['minor_first_choice_pct_sincere']:.1f}% sincere", flush=True)

    win = sim.simulate(model, ct, c, v, a.iters, a.voters, a.seed, polls=sincere)
    sim.save_contest_wins(out / "contest_wins.npz", ct, win, a.iters, sincere=sincere)
    actual = sim.seat_counts(ct, torch.zeros(C, dtype=torch.long))
    seats = {m: sim.seat_counts(ct, w) for m, w in win.items()}
    np.savez_compressed(out / "seats.npz", years=np.array(YEARS), parties=np.array(sim.PLOT_PARTIES),
                        **{f"actual_{ch}": actual[ch][:, 0] for ch in "HS"},
                        **{f"{m}_{ch}": seats[m][ch] for m in seats for ch in "HS"})
    base = pd.read_csv("outputs/compass/methods/seats_summary.csv").set_index(["method", "chamber", "year", "party"])["mean"]
    rows = []
    for m in seats:
        for ch in "HS":
            for t, yr in enumerate(YEARS):
                for p, name in enumerate(sim.PLOT_PARTIES):
                    x = seats[m][ch][t, :, p]
                    rows.append({"method": m, "chamber": ch, "year": yr, "party": name, "actual": actual[ch][t, 0, p],
                                 "mean": x.mean(), "sd": x.std(), "min": x.min(), "max": x.max(),
                                 "baseline_mean": base.get((m, ch, yr, name), np.nan)})
    pd.DataFrame(rows).to_csv(out / "seats_summary.csv", index=False)
    grp = {m: ct.group[torch.arange(C), w.long()] for m, w in win.items()}                 # [I, C]
    info["third_party_winners"] = {m: float((g >= 2).float().sum(1).mean()) for m, g in grp.items()}
    info["actual_third_party_winners"] = int(minor[:, 0].sum())
    names = ct.df.groupby("contest", sort=False).agg(names=("name", list), groups=("group", list), shares=("share", list))
    ch_rows = []
    for m, w in win.items():
        if m == "random_ballot":
            continue
        modal = torch.mode(w.long(), 0).values; freq = (w.long() == modal).float().mean(0)
        for i in torch.where(modal != 0)[0].tolist():
            r = names.loc[ct.ids[i]]
            ch_rows.append({"method": m, "contest": ct.ids[i], "actual_winner": f"{r.names[0]} ({r.groups[0]})",
                            "method_winner": f"{r.names[modal[i]]} ({r.groups[modal[i]]})", "share_of_runs": round(freq[i].item(), 3),
                            "real_shares": " / ".join(f"{g} {100 * s:.1f}" for g, s in zip(r.groups, r.shares)),
                            "sincere_shares": " / ".join(f"{100 * s:.1f}" for s in sincere[i][ct.mask[i]].tolist())})
    pd.DataFrame(ch_rows).to_csv(out / "changed_winners.csv", index=False)
    info["seconds"] = time.time() - t0; info["args"] = vars(a)
    (out / "run_info.json").write_text(json.dumps(info, indent=1))

    fig, ax = plt.subplots(figsize=(11, 5.5))
    lo = 1 - np.exp(-np.clip(per_year.gap_raw - 2 * per_year.gap_se, 0, None)); hi = 1 - np.exp(-(per_year.gap_raw + 2 * per_year.gap_se))
    ax.errorbar(per_year.year, 1 - np.exp(-per_year.gap_raw.clip(lower=0)), yerr=[1 - np.exp(-per_year.gap_raw.clip(lower=0)) - lo, hi - (1 - np.exp(-per_year.gap_raw.clip(lower=0)))],
                fmt="o", color="#718096", capsize=3, label="raw yearly estimate (±2 se)")
    ax.plot(per_year.year, per_year.suppression_mean, "-o", color="#2b6cb0", lw=2, label="used in simulation (shrunk toward pooled)")
    ax.axhline(info["pooled_suppression_dead_heat"], color="k", lw=0.8, ls="--", label=f"pooled {info['pooled_suppression_dead_heat']:.2f}")
    ax.set_ylim(0, 1); ax.set_ylabel("suppression rate of third-party support when both majors run"); ax.grid(alpha=0.25); ax.legend()
    ax.set_title("Measured third-party suppression by election year")
    Path("img/compass/desertion").mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig("img/compass/desertion/suppression_by_year.png", dpi=100)
    print("third-party winners per method:", {k: round(x, 1) for k, x in info["third_party_winners"].items()}, "actual", info["actual_third_party_winners"])
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
