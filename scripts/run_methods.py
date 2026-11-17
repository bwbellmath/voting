#!/usr/bin/env python
"""Re-run every House and Senate election 1976-2020 under several voting methods with compass agents.

  .venv/bin/python scripts/run_methods.py --model outputs/compass/full_d2K4 --iters 200 --voters 1000

Writes outputs/compass/methods/seats.npz (seat counts per method / year / iteration / party),
seats_summary.csv and changed_winners.csv.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_compass import load
from compass import methods as vm, simulate as sim
from compass.data import YEARS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="outputs/compass/full_d2K4")
    ap.add_argument("--iters", type=int, default=200); ap.add_argument("--voters", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="outputs/compass/methods")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    model, _, _ = load(a.model)
    t0 = time.time()
    ct, c, v, info = sim.prepare(model)
    print(f"{len(ct.ids)} contests; calibration max |share error| = {info['calibration_max_abs_err']:.2e}")
    print("minor-party positions (L, G, X):", np.round(info["minor_positions"].numpy(), 2).tolist())
    win = sim.simulate(model, ct, c, v, a.iters, a.voters, a.seed)
    sim.save_contest_wins(out / "contest_wins.npz", ct, win, a.iters)
    actual = sim.seat_counts(ct, torch.zeros(len(ct.ids), dtype=torch.long))          # slot 0 = most votes
    seats = {m: sim.seat_counts(ct, w) for m, w in win.items()}
    np.savez_compressed(out / "seats.npz", years=np.array(YEARS), parties=np.array(sim.PLOT_PARTIES),
                        **{f"actual_{ch}": actual[ch][:, 0] for ch in "HS"},
                        **{f"{m}_{ch}": seats[m][ch] for m in seats for ch in "HS"})
    rows = []
    for m in seats:
        for ch in "HS":
            for t, yr in enumerate(YEARS):
                for p, name in enumerate(sim.PLOT_PARTIES):
                    x = seats[m][ch][t, :, p]
                    rows.append({"method": m, "chamber": ch, "year": yr, "party": name, "actual": actual[ch][t, 0, p],
                                 "mean": x.mean(), "sd": x.std(), "min": x.min(), "max": x.max()})
    pd.DataFrame(rows).to_csv(out / "seats_summary.csv", index=False)
    # contests where a method's usual winner differs from the real (most-votes) winner
    ch_rows = []
    names = ct.df.groupby("contest", sort=False).agg(names=("name", list), groups=("group", list), shares=("share", list))
    for m, w in win.items():
        if m == "random_ballot":
            continue
        modal = torch.mode(w.long(), 0).values
        freq = (w.long() == modal).float().mean(0)
        for i in torch.where(modal != 0)[0].tolist():
            r = names.loc[ct.ids[i]]
            ch_rows.append({"method": m, "contest": ct.ids[i], "actual_winner": f"{r.names[0]} ({r.groups[0]})",
                            "method_winner": f"{r.names[modal[i]]} ({r.groups[modal[i]]})",
                            "share_of_runs": round(freq[i].item(), 3),
                            "shares": " / ".join(f"{g} {100 * s:.1f}" for g, s in zip(r.groups, r.shares))})
    pd.DataFrame(ch_rows).to_csv(out / "changed_winners.csv", index=False)
    (out / "run_info.json").write_text(json.dumps({"args": vars(a), "seconds": time.time() - t0, "contests": len(ct.ids),
        "calibration_max_abs_err": info["calibration_max_abs_err"],
        "minor_positions_LGX": info["minor_positions"].tolist(), "house_contests_per_year":
        {int(y): int(((ct.meta.office == "H") & (ct.meta.eff_year == y)).sum()) for y in YEARS}}, indent=1))
    print(f"done in {time.time() - t0:.0f}s; {len(ch_rows)} (method, contest) pairs with a different usual winner")


if __name__ == "__main__":
    main()
