#!/usr/bin/env python
"""Export everything the web UI needs into webui/data/app.json.

  .venv/bin/python scripts/build_webui.py
  .venv/bin/python -m http.server 8000 -d webui        # then open http://localhost:8000

Needs outputs/compass/{methods,desertion}/contest_wins.npz (written by run_methods.py / run_desertion.py)
and the district layers from scripts/build_webui_geo.py.
"""
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from plot_compass import load, rotation
from compass import methods as vm, simulate as sim, strategy as st
from compass.data import CSV_DIR, JUNK, YEARS, _era, _norm_name, build_races, to_tensors

OUT = ROOT / "webui" / "data"
GROUP_CLASS = {"D": 0, "R": 1, "L": 2, "G": 2, "X": 2}          # party class: Democrat, Republican, third


def president_contests(ev):
    p = pd.read_csv(CSV_DIR / "1976-2020-president.csv")
    df = pd.DataFrame({"year": p.year, "state": p.state_po, "name": _norm_name(p.candidate),
                       "group": sim._group(p.party_detailed), "votes": p.candidatevotes})
    df = df[~df.name.str.match(JUNK) & (df.votes > 0) & (df.name != "")]
    pri = df.group.map({g: i for i, g in enumerate(sim.GROUPS)})
    df = df.assign(_p=pri).sort_values("_p").groupby(["year", "state", "name"], as_index=False).agg(
        votes=("votes", "sum"), group=("group", "first"))
    out = []
    for (yr, s), x in df.groupby(["year", "state"]):
        x = x.sort_values("votes", ascending=False); tot = int(x.votes.sum())
        x = x[(x.votes / tot >= sim.MIN_SHARE) | (np.arange(len(x)) == 0)].head(sim.MAX_CANDS)
        out.append({"id": f"{yr}-P-{s}", "o": "P", "y": int(yr), "st": s, "tv": tot, "ev": int(ev[(s, _era(yr))]),
                    "c": [[n, g, int(v)] for n, g, v in zip(x.name, x.group, x.votes)]})
    return out


def main():
    model, rd, _ = load(str(ROOT / "outputs/compass/full_d2K4"))
    torch.set_grad_enabled(False)
    main_df = build_races(); main_rd = to_tensors(main_df)
    fitted_other = set(main_df[(main_df.party == "O") & (main_df.office != "P")].cand)
    ct = sim.Contests(sim.load_contests(), model, main_rd, fitted_other)
    runs = {}
    for key, folder in [("sincere", "methods"), ("adjusted", "desertion")]:
        z = np.load(ROOT / f"outputs/compass/{folder}/contest_wins.npz")
        assert list(z["ids"]) == ct.ids, f"{folder}: contest list differs; re-run the simulation"
        runs[key] = z
    n_iter = int(runs["sincere"]["n_iter"])
    methods = list(vm.METHODS)
    Q = rotation(model)
    type_lean = (model.mu.numpy() @ Q)[:, 0]
    type_axis2 = (model.mu.numpy() @ Q)[:, 1] if Q.shape[0] > 1 else np.zeros_like(type_lean)

    names = ct.df.groupby("contest", sort=False).agg(names=("name", list), groups=("group", list), votes=("votes", list))
    contests = []
    for i, cid in enumerate(ct.ids):
        m = ct.meta.iloc[i]; r = names.loc[cid]; n = len(r.names)
        c = {"id": cid, "o": m.office, "y": int(m.eff_year), "st": m.state, "tv": int(sum(r.votes)),
             "c": [[a, b, int(v)] for a, b, v in zip(r.names, r.groups, r.votes)]}
        if m.office == "H":
            c["d"] = int(m.district); c["pi"] = [round(float(x), 3) for x in ct.pi[i]]
        else:
            c["k"] = int(m.klass); c["sp"] = int(bool(m.special))
        if n > 1:
            c["w"] = {key: [[int(x) for x in z[f"wins_{mth}"][i, :n]] for mth in methods] for key, z in runs.items()}
            ss = runs["adjusted"]["sincere_share"][i, :n]
            if np.abs(ss - ct.y[i, :n].numpy()).max() > 5e-4:
                c["ss"] = [round(float(x), 4) for x in ss]
        contests.append(c)

    hd = ct.meta[ct.meta.office == "H"].assign(era=lambda d: d.eff_year.map(_era))
    seats = hd.groupby(["state", "era", "eff_year"]).district.nunique().groupby(["state", "era"]).max()
    ev = {(s, e): n + 2 for (s, e), n in seats.items()}
    for e in set(hd.era):
        ev[("DC", e)] = 3
    pres = president_contests(ev)
    for yr in range(1976, 2021, 4):
        tot = sum(c["ev"] for c in pres if c["y"] == yr)
        print(f"  {yr}: {tot} electoral votes" + ("" if tot == 538 else "  <-- not 538"))

    S, T = len(rd.states), len(YEARS)
    pi = model.weights(torch.arange(S).repeat_interleave(T), torch.arange(T).repeat(S)).view(S, T, -1).numpy()
    scen = json.loads((ROOT / "outputs/compass/scenarios/summary.json").read_text())["gallup"]
    by_type = st.party_id_by_type(model, scen["t1"], scen["t2"])
    gal = st.gallup_party_id()
    sup = pd.read_csv(ROOT / "outputs/compass/desertion/suppression_by_year.csv")
    pp = model.party_pos.numpy() @ Q
    p_names = pd.read_csv(CSV_DIR / "1976-2020-president.csv")[["state_po", "state", "state_fips"]].drop_duplicates("state_po")
    app = {
        "years": YEARS, "n_iter": n_iter,
        "methods": [{"key": m, "label": vm.TITLES.get(m, m)} for m in methods],
        "states": [{"po": r.state_po, "name": r.state.title().replace(" Of ", " of "), "fips": f"{int(r.state_fips):02d}"}
                   for r in p_names.sort_values("state_po").itertuples()],
        "contests": contests + pres,
        "compass": {
            "type_lean": [round(float(x), 4) for x in type_lean], "type_axis2": [round(float(x), 4) for x in type_axis2],
            "party_lean": np.round(pp[:, :2, 0], 4).tolist(),
            "state_pi": {s: np.round(pi[i], 3).tolist() for i, s in enumerate(rd.states)},
            "pid_by_type": np.round(by_type, 4).tolist(),
            "pid_labels": ["Democrat", "Lean Democrat", "Independent", "Lean Republican", "Republican"],
        },
        "gallup": {int(y): [round(float(v), 1) for v in r] for y, r in gal.iterrows()},
        "suppression": {int(r.year): round(float(r.suppression_mean), 3) for r in sup.itertuples()},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "app.json").write_text(json.dumps(app, separators=(",", ":")))
    print(f"wrote {OUT / 'app.json'}: {len(contests)} House/Senate + {len(pres)} presidential contests, "
          f"{(OUT / 'app.json').stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
