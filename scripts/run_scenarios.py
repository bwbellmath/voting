#!/usr/bin/env python
"""Scenarios: what if real plurality votes understate sincere third-party support?

  .venv/bin/python scripts/run_scenarios.py            # ~10 min, writes outputs/compass/scenarios/

  1. closeness test      do third-party shares fall in close races? (falsifiable, real data only)
  2. Gallup reconciliation   five-way party ID from compass agents vs Gallup
  3. desertion sweep     real contests, sincere support = what makes strategic plurality match reality
  4. entrant scenarios   a hypothetical third candidate added to real two-party contests
  5. random ballot       seat distributions under sincere voting
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_compass import load
from compass import simulate as sim, strategy as st
from compass.data import YEARS, load_long

RHOS = [0.0, 0.2, 0.5, 0.75, 0.9]
STRENGTHS = [0.05, 0.10, 0.20, 0.25, 0.30, 0.35, 0.40]
STRETCH = [1.0, 2.0]          # 1 = fitted compass; 2 = voters twice as far apart (more policy-driven)
POSITIONS = ["center", "left_flank", "right_flank"]


# ---------------------------------------------------------------- 1. closeness test
def closeness_test() -> pd.DataFrame:
    df = load_long()
    g = df.pivot_table(index=["race", "year", "office", "state"], columns="party", values="votes", aggfunc="sum", fill_value=0).reset_index()
    g = g[(g.D > 0) & (g.R > 0) & (g[["D", "R", "O"]].sum(1) > 1000)].copy()
    g["minor"] = g.O / g[["D", "R", "O"]].sum(1); g["margin"] = (g.D - g.R).abs() / (g.D + g.R)
    rows = []
    for o, name in zip("PSH", ["President (by state)", "Senate", "House"]):
        d = g[g.office == o]
        Z = np.column_stack([pd.get_dummies(d.year).to_numpy(float), pd.get_dummies(d.state).to_numpy(float)])
        res = lambda x: x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
        xr, yr = res(d.margin.to_numpy(float)), res(d.minor.to_numpy(float))
        b = (xr @ yr) / (xr @ xr); e = yr - b * xr; cl = d.state.to_numpy()
        se = np.sqrt(sum((xr[cl == s] @ e[cl == s]) ** 2 for s in np.unique(cl))) / (xr @ xr)
        close, safe = d[d.margin <= 0.05].minor.mean(), d[d.margin > 0.40].minor.mean()
        rows.append({"office": name, "contests": len(d), "minor_share_close_races": close, "minor_share_blowouts": safe,
                     "slope": b, "se": se, "t": b / se, "implied_rho": max(0.0, 1 - close / safe)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 2. Gallup
def gallup_reconciliation(model, ct):
    gal = st.gallup_party_id()
    yrs = [y for y in YEARS if y in gal.index]
    h = ct.df[ct.df.office == "H"].groupby(["state", "eff_year"]).votes.sum().unstack(fill_value=0)
    states = sorted(set(ct.meta.state))
    main_states = sorted(set(load_long().state))
    W = np.zeros((len(main_states), len(YEARS)))
    for i, s in enumerate(main_states):
        if s in h.index:
            W[i] = h.loc[s, YEARS].to_numpy()
    W = W / W.sum(0, keepdims=True)
    target = gal.loc[yrs].to_numpy(); tix = [YEARS.index(y) for y in yrs]
    best = None
    for t1 in np.arange(0.05, 1.5, 0.05):
        for t2 in np.arange(t1 + 0.1, 6.0, 0.1):
            sh = st.party_id_shares(model, W, t1, t2)[tix]
            err = ((sh - target) ** 2).mean()
            if best is None or err < best[0]:
                best = (err, t1, t2, sh)
    err, t1, t2, sh = best
    cols = ["D", "lean_D", "pure_I", "lean_R", "R"]
    out = pd.DataFrame(target, index=yrs, columns=[f"gallup_{c}" for c in cols]).join(
        pd.DataFrame(sh, index=yrs, columns=[f"compass_{c}" for c in cols]))
    net_g = target[:, 0] + target[:, 1] - target[:, 3] - target[:, 4]; net_c = sh[:, 0] + sh[:, 1] - sh[:, 3] - sh[:, 4]
    info = {"t1": float(t1), "t2": float(t2), "rmse_points": float(np.sqrt(err)),
            "net_dem_corr": float(np.corrcoef(net_g, net_c)[0, 1]),
            "gallup_mean": dict(zip(cols, target.mean(0).round(1).tolist())), "compass_mean": dict(zip(cols, sh.mean(0).round(1).tolist()))}
    return out, info


# ---------------------------------------------------------------- helpers
def house_party_table(ct, probs_multi, multi_idx, base_probs):
    """Expected House seats by plot party [T, 4] given per-contest winner probabilities."""
    P = base_probs.clone(); P[multi_idx] = probs_multi[multi_idx]
    plot = torch.as_tensor(sim.PLOT_OF_GROUP)[ct.group]                                   # [C,J]
    out = np.zeros((len(YEARS), 4))
    for t, yr in enumerate(YEARS):
        hc = torch.as_tensor(((ct.meta.office == "H") & (ct.meta.eff_year == yr)).to_numpy())
        for p in range(4):
            out[t, p] = (P[hc] * (plot[hc] == p) * ct.mask[hc]).sum().item()
    return out


def gallagher(seats, votes):
    s = 100 * seats / seats.sum(-1, keepdims=True); v = 100 * votes / votes.sum(-1, keepdims=True)
    return np.sqrt(0.5 * ((s - v) ** 2).sum(-1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="outputs/compass/full_d2K4"); ap.add_argument("--out", default="outputs/compass/scenarios")
    ap.add_argument("--iters", type=int, default=20); ap.add_argument("--voters", type=int, default=1000)
    ap.add_argument("--entrant-contests", type=int, default=800); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    model, _, _ = load(a.model)
    summary = {"args": vars(a)}

    ctest = closeness_test(); ctest.to_csv(out / "closeness_test.csv", index=False)
    print(ctest.round(4).to_string(), flush=True)

    ct, c, v0, info = sim.prepare(model, verbose=False)
    torch.set_grad_enabled(False)
    gal, ginfo = gallup_reconciliation(model, ct); gal.to_csv(out / "gallup_vs_compass.csv"); summary["gallup"] = ginfo
    print("Gallup reconciliation:", json.dumps(ginfo), flush=True)

    # ------------------------------------------------------------ 3. desertion sweep on real contests
    C = len(ct.ids); multi = torch.where(ct.ncand >= 3)[0]
    front = torch.tensor([[0, 1]]).expand(len(multi), 2).contiguous()
    minor = (ct.group >= 2) & ct.mask
    actual = torch.zeros(C, sim.MAX_CANDS); actual[:, 0] = 1.0                            # slot 0 = real winner
    rows, rb_seats, house_rows = [], {}, []
    gen = np.random.default_rng(a.seed)
    for rho in RHOS:
        # a third candidate cannot sincerely out-poll the real runner-up (they would have been a frontrunner)
        ym = ct.y[multi]; top_minor = (ym[:, 2:] * ct.mask[multi][:, 2:]).max(1).values
        rho_c = torch.minimum(torch.full((len(multi),), rho), (1 - top_minor / ym[:, 1]).clamp(0, 1))
        capped = int((rho_c < rho - 1e-6).sum())
        v, err = st.calibrate_strategic(model, ct.pi[multi], c[multi], ct.y[multi], ct.mask[multi], front, rho_c)
        res = st.run_methods(model, ct.pi[multi], c[multi], v, ct.ncand[multi], front, rho_c, a.iters, a.voters, a.seed)
        sincere = ct.y.clone(); sincere[multi] = sim.expected_shares(model, ct.pi[multi], c[multi], v, ct.mask[multi])
        for m in st.SCENARIO_METHODS:
            w = torch.zeros(C, sim.MAX_CANDS); w[multi] = res[m]["wins"]
            base = sincere if m == "random_ballot" else actual                             # 1-2 candidate contests
            full = base.clone(); full[multi] = w[multi]
            hs = house_party_table(ct, full, torch.arange(C), full)
            hv = house_party_table(ct, sincere, torch.arange(C), sincere)
            rows.append({"rho": rho, "method": m, "calibration_err": err, "contests_capped": capped,
                         "third_party_wins": (full * minor).sum().item(),
                         "changed_winner": (1 - full[:, 0]).sum().item() if m != "random_ballot" else float("nan"),
                         "condorcet_efficiency": res[m]["cw"].mean().item(), "expression": res[m]["expr"].mean().item(),
                         "house_third_party_seats_per_year": hs[:, 2:].sum(1).mean(),
                         "house_gallagher": gallagher(hs, hv).mean()})
            house_rows.append({"rho": rho, "method": m, **{f"{p}_{y}": hs[t, i] for t, y in enumerate(YEARS) for i, p in enumerate(sim.PLOT_PARTIES)}})
        rb = [r for r in house_rows if r["rho"] == rho and r["method"] == "random_ballot"][0]
        for r in rows[-len(st.SCENARIO_METHODS):]:
            hs_m = np.array([[[h for h in house_rows if h["rho"] == rho and h["method"] == r["method"]][0][f"{p}_{y}"]
                              for p in sim.PLOT_PARTIES] for y in YEARS])
            hs_rb = np.array([[rb[f"{p}_{y}"] for p in sim.PLOT_PARTIES] for y in YEARS])
            r["house_seats_from_random_ballot"] = 0.5 * np.abs(hs_m - hs_rb).sum(1).mean()
            r["sincere_minor_first_choice_pct"] = 100 * (sincere * minor).sum().item() / C
        # 5. random-ballot House seat draws under sincere voting: one categorical draw per contest
        cum = torch.cumsum(sincere * ct.mask, 1).numpy(); u = gen.random((200, C, 1))
        win = (u > cum[None] / cum[None, :, -1:]).sum(-1).clip(max=sim.MAX_CANDS - 1)
        rb_seats[rho] = sim.seat_counts(ct, torch.as_tensor(win))
        print(f"rho={rho}: calibration err {err:.1e}; third-party wins: " + ", ".join(
            f"{r['method']} {r['third_party_wins']:.0f}" for r in rows[-len(st.SCENARIO_METHODS):]), flush=True)
    sweep = pd.DataFrame(rows); sweep.to_csv(out / "desertion_sweep.csv", index=False)
    summary["actual_third_party_wins"] = int(minor[:, 0].sum()); summary["contests"] = C; summary["multi_candidate_contests"] = len(multi)
    np.savez_compressed(out / "random_ballot_sincere.npz", years=np.array(YEARS), rhos=np.array(RHOS), parties=np.array(sim.PLOT_PARTIES),
                        actual_H=sim.seat_counts(ct, torch.zeros(C, dtype=torch.long))["H"][:, 0],
                        **{f"H_rho{r}": rb_seats[r]["H"] for r in RHOS}, **{f"S_rho{r}": rb_seats[r]["S"] for r in RHOS})

    # ------------------------------------------------------------ 4. entrant scenarios
    two = torch.where((ct.ncand == 2) & (ct.group[:, :2].sort(1).values == torch.tensor([0, 1])).all(1)
                      & torch.as_tensor((ct.meta.eff_year >= 2012).to_numpy()))[0]
    pick = two[torch.randperm(len(two), generator=torch.Generator().manual_seed(a.seed))[: a.entrant_contests]]
    n = len(pick); is_d0 = ct.group[pick, 0] == 0
    cD = torch.where(is_d0[:, None], c[pick, 0], c[pick, 1]); cR = torch.where(is_d0[:, None], c[pick, 1], c[pick, 0])
    vD = torch.where(is_d0, v0[pick, 0], v0[pick, 1]); vR = torch.where(is_d0, v0[pick, 1], v0[pick, 0])
    mask3 = torch.ones(n, 3, dtype=torch.bool); nc3 = torch.full((n,), 3)
    mainstream = torch.tensor([[0, 1]]).expand(n, 2).contiguous()                        # pragmatic = back a major party
    y2 = torch.stack([torch.where(is_d0, ct.y[pick, 0], ct.y[pick, 1]), torch.where(is_d0, ct.y[pick, 1], ct.y[pick, 0])], 1)
    erows = []
    for k in STRETCH:
        mk = st.stretched(model, k); kD, kR = cD * k, cR * k
        v2, _ = sim.calibrate(mk, ct.pi[pick], torch.stack([kD, kR], 1), y2, torch.ones(n, 2, dtype=torch.bool))
        for pos in POSITIONS:
            cE = {"center": (kD + kR) / 2, "left_flank": kD - (kR - kD), "right_flank": kR + (kR - kD)}[pos]
            c3 = torch.stack([kD, kR, cE], 1)
            for s in STRENGTHS:
                v3 = st.calibrate_entrant(mk, ct.pi[pick], c3, torch.cat([v2, torch.zeros(n, 1)], 1), mask3, 2, s)
                sincere = sim.expected_shares(mk, ct.pi[pick], c3, v3, mask3)
                res = st.run_methods(mk, ct.pi[pick], c3, v3, nc3, mainstream, 1.0, max(5, a.iters // 2), a.voters, a.seed)
                ps = res["plurality_strategic"]["wins"]
                for m in st.SCENARIO_METHODS:
                    w = res[m]["wins"]
                    erows.append({"stretch": k, "position": pos, "entrant_first_choice": s, "method": m,
                                  "entrant_win_rate": w[:, 2].mean().item(), "dem_win_rate": w[:, 0].mean().item(),
                                  "rep_win_rate": w[:, 1].mean().item(), "expression": res[m]["expr"].mean().item() / k ** 2,
                                  "condorcet_efficiency": res[m]["cw"].mean().item(),
                                  "differs_from_strategic_plurality": 0.5 * (w - ps).abs().sum(1).mean().item(),
                                  "entrant_in_top_two": (sincere.topk(2, dim=1).indices == 2).any(1).float().mean().item()})
                print(f"stretch {k} {pos} s={s}: " + ", ".join(f"{r['method']} {r['entrant_win_rate']:.2f}" for r in erows[-len(st.SCENARIO_METHODS):]), flush=True)
    ent = pd.DataFrame(erows)
    # strategy sensitivity: how far does the strategic version move the outcome from the honest one?
    ent.to_csv(out / "entrant_scenarios.csv", index=False)
    summary["entrant_contests"] = n
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print("done")


if __name__ == "__main__":
    main()
