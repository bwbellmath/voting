#!/usr/bin/env python
"""Fit the compass model on a train split and score it on held-out real elections.

  .venv/bin/python scripts/run_compass.py --offices PS --split random
  .venv/bin/python scripts/run_compass.py --offices PSH --split temporal --d 2 --K 4
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from compass import baseline, evaluate as ev
from compass.data import build_races, to_tensors
from compass.fit import fit
from compass.model import Config, build


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offices", default="PS")
    ap.add_argument("--split", default="random", choices=["random", "temporal", "none"])
    ap.add_argument("--d", type=int, default=2); ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--steps", type=int, default=2500); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-adapt", action="store_true"); ap.add_argument("--s-u", type=float, default=0.5)
    ap.add_argument("--M", type=int, default=128); ap.add_argument("--s-C", type=float, default=0.5)
    ap.add_argument("--s-G", type=float, default=1.0); ap.add_argument("--quad", action="store_true")
    ap.add_argument("--s-walk", type=float, default=None); ap.add_argument("--s-wave", type=float, default=None)
    ap.add_argument("--tag", default=None); ap.add_argument("--out", default="outputs/compass")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rd = to_tensors(build_races(offices=tuple(a.offices)))
    if a.split == "random":
        tr, te = ev.random_split(rd, 0.2, a.seed)
    elif a.split == "temporal":
        tr, te = ev.temporal_split(rd, 2016)
    else:
        tr, te = np.arange(len(rd)), np.arange(len(rd))
    train, test = rd.subset(tr), rd.subset(te)
    cfg = Config(d=a.d, K=a.K, adapt=not a.no_adapt, s_u=a.s_u, M=a.M, s_C=a.s_C, s_G=a.s_G,
                 quad_trend=a.quad)
    if a.s_walk: cfg.s_walk = a.s_walk
    if a.s_wave: cfg.s_wave = a.s_wave
    model = build(rd, cfg, a.seed)
    print(f"{len(train)} train races, {len(test)} test races, d={a.d} K={a.K}")
    t0 = time.time(); fit(model, train, steps=a.steps)
    with torch.no_grad():
        p_tr, p_te = ev.d_share(model(train), train), ev.d_share(model(test), test)
    res = {"args": vars(a), "fit_seconds": time.time() - t0,
           "train": ev.two_party_metrics(p_tr, train),
           "test": {"compass": ev.two_party_metrics(p_te, test),
                    "baseline_swing": ev.two_party_metrics(baseline.fit_predict(train, test, False), test),
                    "baseline_trend": ev.two_party_metrics(baseline.fit_predict(train, test, True), test)}}
    # calibration: fit the predictive spread on half the test races, check coverage on the other half
    perm = np.random.default_rng(a.seed).permutation(len(test)); h = len(perm) // 2
    phis = ev.calibrate_phi(p_te, test, perm[:h])
    res["test"]["calibrated_phi"] = phis
    res["test"]["coverage90"] = ev.coverage(p_te, test, phis, perm[h:])
    print("\nheld-out two-party D share (RMSE in share points, lower is better)")
    for name in res["test"]["compass"]:
        row = "  ".join(f"{m}: rmse {100 * res['test'][m][name]['rmse']:5.2f} win {100 * res['test'][m][name]['winner_acc']:5.1f}%"
                        for m in ["compass", "baseline_swing", "baseline_trend"])
        print(f"  {name} n={res['test']['compass'][name]['n']:5d}  {row}")
    print("  in-sample compass rmse:", {k: round(100 * v["rmse"], 2) for k, v in res["train"].items()})
    print("  90% interval coverage:", {k: (round(v["coverage"], 3), round(100 * v["mean_halfwidth"], 1)) for k, v in res["test"]["coverage90"].items()})
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tag = a.tag or f"{a.offices}_{a.split}_d{a.d}K{a.K}_s{a.seed}"
    (out / f"{tag}.json").write_text(json.dumps(res, indent=1))
    torch.save({"state": model.state_dict(), "cfg": cfg.__dict__, "offices": a.offices}, out / f"{tag}.pt")


if __name__ == "__main__":
    main()
