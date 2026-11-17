#!/usr/bin/env python
"""Is the fitted compass a property of the data or of the optimiser?

1. Seed stability: fit the real data from several random starts; compare state-year mean
   positions across fits after Procrustes alignment (R^2 = 1 means the same map every time).
2. Synthetic recovery: simulate elections from a known compass on the real race design,
   refit, and measure how much of the true map is recovered.

  .venv/bin/python scripts/identifiability.py --offices PSH --seeds 3
"""
import argparse, itertools, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from compass import evaluate as ev, synth
from compass.data import build_races, to_tensors, YEARS
from compass.fit import fit
from compass.model import Config, build


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offices", default="PSH"); ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--d", type=int, default=2); ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--steps", type=int, default=1000); ap.add_argument("--M", type=int, default=64)
    ap.add_argument("--s-walk", type=float, default=0.03)
    a = ap.parse_args()
    rd = to_tensors(build_races(offices=tuple(a.offices)))
    cfg = Config(d=a.d, K=a.K, M=a.M, s_walk=a.s_walk)
    res = {"args": vars(a)}

    fits = []
    for s in range(a.seeds):
        m = build(rd, cfg, seed=s); fit(m, rd, steps=a.steps, verbose=False); fits.append(m)
        print(f"real fit seed {s}: log-lik/race {m.log_lik(rd).mean().item():.4f}", flush=True)
    sm = [synth.state_means(m) for m in fits]
    res["seed_stability_r2"] = [synth.procrustes_r2(sm[i], sm[j]) for i, j in itertools.combinations(range(a.seeds), 2)]
    # 1-D summary that should be stable even if the 2-D map is not: position along the party axis
    def party_axis(m):
        with torch.no_grad():
            v = (m.party_pos[:, 1] - m.party_pos[:, 0]).mean(0).numpy()
        return synth.state_means(m) @ (v / np.linalg.norm(v))
    ax = [party_axis(m) for m in fits]
    res["party_axis_corr"] = [float(np.corrcoef(ax[i], ax[j])[0, 1]) for i, j in itertools.combinations(range(a.seeds), 2)]
    print("seed stability, full map R^2:", np.round(res["seed_stability_r2"], 3))
    print("seed stability, party-axis corr:", np.round(res["party_axis_corr"], 3), flush=True)

    truth = synth.make_truth(rd, cfg)
    srd, clean = synth.simulate(truth, rd)
    tr, te = ev.random_split(srd, 0.2, 0)
    m = build(srd, cfg, seed=1); fit(m, srd.subset(tr), steps=a.steps, verbose=False)
    test = srd.subset(te)
    with torch.no_grad():
        got = ev.two_party_metrics(ev.d_share(m(test), test), test)
    floor = ev.two_party_metrics(ev.d_share(clean[te], test), test)
    res["synthetic"] = {"map_r2": synth.procrustes_r2(synth.state_means(m), synth.state_means(truth)),
                        "party_axis_corr": float(np.corrcoef(party_axis(m), party_axis(truth))[0, 1]),
                        "heldout_rmse": {k: v["rmse"] for k, v in got.items()},
                        "noise_floor_rmse": {k: v["rmse"] for k, v in floor.items()}}
    print("synthetic recovery:", json.dumps(res["synthetic"], indent=1))
    out = Path("outputs/compass"); out.mkdir(parents=True, exist_ok=True)
    (out / f"identifiability_{a.offices}_d{a.d}K{a.K}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
