"""Re-run the real House and Senate elections under other voting methods.

Agents are sampled from each constituency's fitted compass distribution. Candidate valences are
calibrated per contest so that the agents' *first-choice* shares equal the real vote shares; the
same agents then fill in ranked / scored / approval ballots from their utilities.

Pipeline
  1. load_contests      every general-election contest with all candidates above MIN_SHARE
  2. place_candidates   positions: fitted party / candidate positions; minor parties get a
                        fitted group position (Libertarian, Green, other)
  3. calibrate          per-contest valences so expected plurality shares == actual shares
  4. simulate           N agents per contest per iteration -> winner under each method
  5. seat_counts        seats by party: House per year, Senate as the sitting chamber
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from . import methods as vm
from .data import CSV_DIR, JUNK, YEARS, _era, _norm_name, _person_key, build_races, to_tensors
from .model import Compass

MIN_SHARE = 0.005
MAX_CANDS = 6
GROUPS = ["D", "R", "L", "G", "X"]            # Democrat, Republican, Libertarian, Green, other
PLOT_PARTIES = ["Democrat", "Republican", "Other", "Libertarian"]
PLOT_OF_GROUP = np.array([0, 1, 3, 2, 2])     # group id -> index into PLOT_PARTIES


def _group(s: pd.Series) -> pd.Series:
    s = s.fillna("").astype(str).str.upper()
    out = pd.Series("X", index=s.index)
    out[s.str.contains("GREEN")] = "G"
    out[s.str.contains("LIBERTARIAN")] = "L"
    out[s.str.contains("REPUBLICAN") & ~s.str.contains("DEMOCRAT")] = "R"
    out[s.str.startswith("DEMOCRAT")] = "D"
    return out


def load_contests(csv_dir: Path = CSV_DIR, min_share: float = MIN_SHARE, max_cands: int = MAX_CANDS) -> pd.DataFrame:
    """One row per (contest, candidate). Uncontested seats are kept as one-candidate contests."""
    s = pd.read_csv(csv_dir / "1976-2020-senate.csv")
    s = s[(s.stage == "gen") & (s.state_po != "DC")]
    sen = pd.DataFrame({"office": "S", "year": s.year, "state": s.state_po, "district": "",
                        "klass": s["class"].round().astype(int), "special": s.special.astype(bool),
                        "name": _norm_name(s.candidate), "group": _group(s.party_detailed), "votes": s.candidatevotes})
    h = pd.read_csv(csv_dir / "1976-2020-house.csv", low_memory=False)
    h = h[(h.stage == "GEN") & (h.runoff.astype(str) != "True") & (~h.special.astype(bool)) & (h.state_po != "DC")]
    hou = pd.DataFrame({"office": "H", "year": h.year, "state": h.state_po, "district": h.district.astype(str).str.zfill(3),
                        "klass": 0, "special": False,
                        "name": _norm_name(h.candidate), "group": _group(h.party), "votes": h.candidatevotes})
    df = pd.concat([sen, hou], ignore_index=True)
    df["eff_year"] = df.year + df.year % 2
    df = df[df.eff_year.isin(YEARS) & ~df.name.str.match(JUNK)].copy()
    df["contest"] = (df.year.astype(str) + "-" + df.office + "-" + df.state + "-" + df.district
                     + np.where(df.office == "S", "-c" + df.klass.astype(str), "") + np.where(df.special, "-sp", ""))
    # an unopposed candidate is often recorded with 0 votes: give them one so they win the seat
    tot = df.groupby("contest").votes.transform("sum")
    df.loc[tot == 0, "votes"] = 1
    df = df[df.votes > 0]
    pri = df.group.map({g: i for i, g in enumerate(GROUPS)})
    df = df.assign(_pri=pri).sort_values("_pri")
    keys = ["contest", "office", "year", "eff_year", "state", "district", "klass", "special", "name"]
    df = df.groupby(keys, as_index=False).agg(votes=("votes", "sum"), group=("group", "first"))
    df["share_all"] = df.votes / df.groupby("contest").votes.transform("sum")
    df["won"] = df.votes == df.groupby("contest").votes.transform("max")
    df = df[(df.share_all >= min_share) | df.won]
    df = df.sort_values(["contest", "votes"], ascending=[True, False])
    df = df[df.groupby("contest").cumcount() < max_cands].copy()
    df["share"] = df.votes / df.groupby("contest").votes.transform("sum")
    df["won"] = df.groupby("contest").cumcount() == 0
    df["cand"] = df.name.map(_person_key) + "|" + df.state
    df["unit"] = np.where(df.office == "H", df.state + "-" + df.district + "-e" + df.eff_year.map(_era).astype(str), "")
    return df.reset_index(drop=True)


class Contests:
    """Padded tensors for all contests plus the bookkeeping needed to count seats."""

    def __init__(self, df: pd.DataFrame, model: Compass, main_rd, fitted_other: set):
        self.df = df
        g = list(df.groupby("contest", sort=False))
        C, J = len(g), MAX_CANDS
        self.ids = [k for k, _ in g]
        self.meta = pd.DataFrame([x.iloc[0][["office", "eff_year", "state", "district", "klass", "special"]] for _, x in g]).reset_index(drop=True)
        self.y = torch.zeros(C, J); self.mask = torch.zeros(C, J, dtype=torch.bool)
        self.group = torch.full((C, J), 4, dtype=torch.long); self.anchor = torch.zeros(C, J, dtype=torch.bool)
        delta_ix = torch.zeros(C, J, dtype=torch.long); has_delta = torch.zeros(C, J, dtype=torch.bool)
        c_ix = {c: i for i, c in enumerate(main_rd.cands)}
        for r, (_, x) in enumerate(g):
            n = len(x)
            self.y[r, :n] = torch.tensor(x.share.to_numpy(), dtype=torch.float32); self.mask[r, :n] = True
            self.group[r, :n] = torch.tensor([GROUPS.index(v) for v in x.group])
            for j, (cand, grp) in enumerate(zip(x.cand, x.group)):
                if grp not in "DR" and cand in fitted_other:
                    delta_ix[r, j] = c_ix[cand]; has_delta[r, j] = True
        s_ix = {s: i for i, s in enumerate(main_rd.states)}; u_ix = {u: i for i, u in enumerate(main_rd.units)}
        self.state = torch.tensor([s_ix[s] for s in self.meta.state])
        self.year = torch.tensor([YEARS.index(y) for y in self.meta.eff_year])
        self.office = torch.tensor([1 if o == "S" else 2 for o in self.meta.office])
        unit = [u_ix.get(x.unit.iloc[0], 0) for _, x in g]
        self.unit = torch.tensor(unit)
        self.ncand = self.mask.sum(1)
        major = self.group < 2
        self.anchor = (major | has_delta) & self.mask
        with torch.no_grad():
            self.pi = model.weights(self.state, self.year, self.unit)                      # [C,K]
            shift = self.pi @ model.mu - model.national_mean()[self.year]
            c = model.party_pos[self.year.unsqueeze(1), self.group.clamp(max=1)]
            c = c + torch.sigmoid(model.lam_raw) * shift.unsqueeze(1) if model.cfg.adapt else c
            self.c_anchor = torch.where(major.unsqueeze(-1), c, model.delta[delta_ix])     # [C,J,d]


def expected_shares(model: Compass, pi, c, v, mask, chunk: int = 4096):
    """Expected first-choice shares [C, J] for candidates at c with valences v."""
    z = model.mu.unsqueeze(1) + model.log_sig.exp().unsqueeze(1) * model.eps.unsqueeze(0)
    z2 = (z ** 2).sum(-1)
    out = []
    for i in range(0, len(pi), chunk):
        cc, vv, mm = c[i:i + chunk], v[i:i + chunk], mask[i:i + chunk]
        util = -(z2[None, :, :, None] - 2 * torch.einsum("kmd,rjd->rkmj", z, cc)
                 + (cc ** 2).sum(-1)[:, None, None, :]) + vv[:, None, None, :]
        util = util.masked_fill(~mm[:, None, None, :], -1e9)
        out.append(torch.einsum("rk,rkj->rj", pi[i:i + chunk], F.softmax(util, -1).mean(2)))
    return torch.cat(out)


@torch.no_grad()
def calibrate(model: Compass, pi, c, y, mask, iters: int = 400, tol: float = 1e-5):
    """Valences that make expected first-choice shares equal y (fixed-point iteration)."""
    y = (y * mask) / (y * mask).sum(1, keepdim=True).clamp_min(1e-12)
    logy = y.clamp_min(1e-12).log()
    v = logy.masked_fill(~mask, 0.0)
    for it in range(iters):
        p = expected_shares(model, pi, c, v, mask)
        err = ((p - y).abs() * mask).max().item()
        if err < tol:
            break
        v = (v + (logy - p.clamp_min(1e-12).log())).masked_fill(~mask, 0.0)
        v = v - (v * mask).sum(1, keepdim=True) / mask.sum(1, keepdim=True)
    return v, err


def place_minor_parties(model: Compass, ct: Contests, steps: int = 400, lr: float = 0.05, verbose: bool = True):
    """Fit one position per minor-party group (L, G, X) from where those parties do better or worse.

    Major-party and already-fitted candidates are frozen at their calibrated valences; a minor
    candidate gets its group's position and a group-by-office valence.
    """
    d = model.cfg.d
    v_anchor, _ = calibrate(model, ct.pi, ct.c_anchor, ct.y, ct.anchor & ct.mask)
    pg = torch.zeros(3, d, requires_grad=True); wg = torch.full((3, 3), -3.0, requires_grad=True)
    opt = torch.optim.Adam([pg, wg], lr=lr)
    gi = (ct.group - 2).clamp(min=0)
    has_minor = ((~ct.anchor) & ct.mask).any(1)
    sub = torch.where(has_minor)[0]
    for step in range(steps + 1):
        opt.zero_grad()
        c = torch.where(ct.anchor.unsqueeze(-1), ct.c_anchor, pg[gi])[sub]
        v = torch.where(ct.anchor, v_anchor, wg[gi, ct.office.unsqueeze(1)])[sub]
        p = expected_shares(model, ct.pi[sub], c, v, ct.mask[sub])
        loss = -(ct.y[sub] * p.clamp_min(1e-9).log()).sum(1).mean() + 0.5 * (pg / 1.5).pow(2).sum() / len(sub)
        loss.backward(); opt.step()
        if verbose and step % 100 == 0:
            print(f"  minor-party placement step {step:4d} loss {loss.item():.5f}", flush=True)
    return pg.detach(), wg.detach()


def all_positions(ct: Contests, pg):
    gi = (ct.group - 2).clamp(min=0)
    return torch.where(ct.anchor.unsqueeze(-1), ct.c_anchor, pg[gi])


@torch.no_grad()
def simulate(model: Compass, ct: Contests, c, v, n_iter: int = 200, n_voters: int = 1000, seed: int = 0,
             chunk_cells: int = 12_000_000, verbose: bool = True, polls=None) -> dict:
    """Winner's candidate slot for every (method, iteration, contest): dict name -> int8 [I, C].

    polls [C, J]: shares voters believe when setting their approval threshold (default: real shares).
    """
    polls = ct.y if polls is None else polls
    g = torch.Generator().manual_seed(seed)
    C = len(ct.ids)
    out = {m: torch.zeros(n_iter, C, dtype=torch.int8) for m in vm.METHODS}          # uncontested -> slot 0
    # two-candidate contests: every majoritarian method agrees, so draw the vote count directly
    two = torch.where(ct.ncand == 2)[0]
    if len(two):
        s0 = (ct.y[two, 0] / (ct.y[two, 0] + ct.y[two, 1])).expand(n_iter, -1)
        n0 = torch.binomial(torch.full_like(s0, float(n_voters)), s0, generator=g)
        coin = torch.rand(s0.shape, generator=g) < 0.5
        w = torch.where(n0 * 2 == n_voters, coin, n0 * 2 < n_voters).to(torch.int8)       # 1 -> slot 1 wins
        for m in vm.METHODS:
            out[m][:, two] = w
        out["random_ballot"][:, two] = (torch.rand(s0.shape, generator=g) >= s0).to(torch.int8)
    mu, sig = model.mu, model.log_sig.exp()
    for J in range(3, MAX_CANDS + 1):
        idx = torch.where(ct.ncand == J)[0]
        if not len(idx):
            continue
        per = max(1, chunk_cells // (n_voters * J * J))                                   # contests x iters per chunk
        cJ, vJ, piJ, yJ = c[idx, :J], v[idx, :J], ct.pi[idx], polls[idx, :J]
        flat_r = torch.arange(len(idx)).repeat(n_iter); flat_i = torch.arange(n_iter).repeat_interleave(len(idx))
        for a in range(0, len(flat_r), per):
            r, i = flat_r[a:a + per], flat_i[a:a + per]
            k = torch.multinomial(piJ[r], n_voters, replacement=True, generator=g)        # [B,N] voter types
            z = mu[k] + sig[k] * torch.randn(len(r), n_voters, mu.shape[1], generator=g)
            U = -((z[:, :, None, :] - cJ[r][:, None, :, :]) ** 2).sum(-1) + vJ[r][:, None, :]
            U = U - torch.log(-torch.log(torch.rand(U.shape, generator=g).clamp(1e-12, 1 - 1e-7)))  # Gumbel taste shock
            mask = torch.ones(len(r), J, dtype=torch.bool)
            for m, f in vm.METHODS.items():
                win = f(U, mask, yJ[r]) if m == "approval" else f(U, mask)   # approval threshold uses poll shares
                out[m][i, idx[r]] = win.to(torch.int8)
        if verbose:
            print(f"  simulated {len(idx):5d} contests with {J} candidates", flush=True)
    return out


def seat_counts(ct: Contests, winner_slot) -> dict:
    """Seats by plot party. winner_slot: [I, C] (or [C] for the actual result).

    House: seats won in each election year. Senate: the sitting chamber after each election,
    i.e. for every (state, class) seat the most recent contest in the last three cycles.
    Returns {"H": [T, I, 4], "S": [T, I, 4]} as numpy int arrays.
    """
    w = torch.as_tensor(winner_slot).long()
    w = w[None] if w.dim() == 1 else w
    grp = ct.group[torch.arange(len(ct.ids)), w]                                       # [I, C] group of the winner
    party = torch.as_tensor(PLOT_OF_GROUP)[grp].numpy()
    meta, I = ct.meta, party.shape[0]
    res = {k: np.zeros((len(YEARS), I, 4), int) for k in "HS"}
    for t, yr in enumerate(YEARS):
        hc = np.where((meta.office == "H") & (meta.eff_year == yr))[0]
        for p in range(4):
            res["H"][t, :, p] = (party[:, hc] == p).sum(1)
        sm = meta[(meta.office == "S") & (meta.eff_year <= yr) & (meta.eff_year >= yr - 4)]
        sc = sm.sort_values(["eff_year", "special"]).groupby(["state", "klass"]).tail(1).index.to_numpy()
        for p in range(4):
            res["S"][t, :, p] = (party[:, sc] == p).sum(1)
    return res


def save_contest_wins(path, ct: Contests, win: dict, n_iter: int, sincere=None):
    """Per-contest results for the web UI: how often each candidate slot won under each method.

    Contests are independent draws, so seat distributions for any subset of contests follow from these.
    """
    C = len(ct.ids)
    counts = {m: torch.zeros(C, MAX_CANDS, dtype=torch.long).scatter_add_(
        1, w.long().T.contiguous(), torch.ones(C, w.shape[0], dtype=torch.long)).numpy().astype(np.uint16) for m, w in win.items()}
    extra = {} if sincere is None else {"sincere_share": np.asarray(sincere, dtype=np.float32)}
    np.savez_compressed(path, ids=np.array(ct.ids), n_iter=n_iter, group=ct.group.numpy().astype(np.int8),
                        mask=ct.mask.numpy(), share=ct.y.numpy(), **extra, **{f"wins_{m}": x for m, x in counts.items()})


def prepare(model: Compass, verbose: bool = True):
    """Steps 1-3: contests, candidate positions and calibrated valences."""
    main_df = build_races()
    main_rd = to_tensors(main_df)
    fitted_other = set(main_df[(main_df.party == "O") & (main_df.office != "P")].cand)
    ct = Contests(load_contests(), model, main_rd, fitted_other)
    pg, wg = place_minor_parties(model, ct, verbose=verbose)
    c = all_positions(ct, pg)
    v, err = calibrate(model, ct.pi, c, ct.y, ct.mask)
    return ct, c, v, {"minor_positions": pg, "minor_valence": wg, "calibration_max_abs_err": err}
