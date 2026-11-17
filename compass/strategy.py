"""Strategic voting on top of the compass: what if real plurality votes hide sincere preferences?

Assumption being explored: under plurality a fraction `rho` of voters whose favourite is not one of
the two frontrunners desert to the frontrunner they prefer. Real results are then *strategic*
plurality results, and sincere third-party support is larger than the vote shows.

  calibrate_strategic   valences such that strategic-plurality shares equal the real shares
  run_methods           honest and strategic ballots under every method, plus outcome metrics
  gallup_party_id       five-way party identification from the two Gallup files
  party_id_shares       the same five groups computed from compass agents
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from . import methods as vm
from .data import CSV_DIR, YEARS
from .model import Compass
from .simulate import expected_shares

BIG = 1e6
SCENARIO_METHODS = ["plurality_strategic", "plurality_honest", "random_ballot", "rcv_honest", "rcv_strategic",
                    "approval_honest", "approval_strategic", "star_honest", "star_strategic", "borda", "condorcet"]
LABELS = {"plurality_strategic": "Plurality, strategic", "plurality_honest": "Plurality, honest",
          "random_ballot": "Random ballot", "rcv_honest": "Ranked choice, honest", "rcv_strategic": "Ranked choice, strategic",
          "approval_honest": "Approval, honest", "approval_strategic": "Approval, strategic",
          "star_honest": "STAR, honest", "star_strategic": "STAR, strategic", "borda": "Borda", "condorcet": "Condorcet"}


# ---------------------------------------------------------------- expected strategic shares
def _type_probs(model: Compass, c, v, mask):
    """[C, K, M, J] sincere first-choice probabilities at each quadrature point of each voter type."""
    z = model.mu.unsqueeze(1) + model.log_sig.exp().unsqueeze(1) * model.eps.unsqueeze(0)
    util = -((z ** 2).sum(-1)[None, :, :, None] - 2 * torch.einsum("kmd,rjd->rkmj", z, c)
             + (c ** 2).sum(-1)[:, None, None, :]) + v[:, None, None, :]
    return F.softmax(util.masked_fill(~mask[:, None, None, :], -1e9), -1)


def strategic_shares(model: Compass, pi, c, v, mask, front, rho, chunk: int = 2048):
    """Expected plurality shares when a fraction rho (float or [C]) of non-frontrunner supporters desert.

    front [C, 2] are the two frontrunner slots. A deserter picks between the frontrunners in
    proportion to their sincere appeal to that voter (the logit second-choice rule).
    """
    out = []
    rho = torch.as_tensor(rho, dtype=torch.float32).expand(len(pi))
    for i in range(0, len(pi), chunk):
        p = _type_probs(model, c[i:i + chunk], v[i:i + chunk], mask[i:i + chunk])
        f = front[i:i + chunk]; rh = rho[i:i + chunk, None, None, None]
        is_front = torch.zeros(p.shape[0], p.shape[-1], dtype=torch.bool).scatter_(1, f, True)[:, None, None, :]
        pf = p * is_front
        minor_mass = (p * ~is_front).sum(-1, keepdim=True)
        s = pf + rh * minor_mass * pf / pf.sum(-1, keepdim=True).clamp_min(1e-12) + (1 - rh) * p * ~is_front
        out.append(torch.einsum("rk,rkj->rj", pi[i:i + chunk], s.mean(2)))
    return torch.cat(out)


@torch.no_grad()
def calibrate_strategic(model: Compass, pi, c, y, mask, front, rho, iters: int = 800, tol: float = 2e-5):
    """Valences such that *strategic* plurality shares equal the observed shares y."""
    logy = y.clamp_min(1e-12).log()
    v = logy.masked_fill(~mask, 0.0)
    for _ in range(iters):
        p = strategic_shares(model, pi, c, v, mask, front, rho)
        err = ((p - y).abs() * mask).max().item()
        if err < tol:
            break
        v = (v + 0.7 * (logy - p.clamp_min(1e-12).log())).masked_fill(~mask, 0.0)
        v = v - (v * mask).sum(1, keepdim=True) / mask.sum(1, keepdim=True)
    return v, err


@torch.no_grad()
def calibrate_entrant(model: Compass, pi, c, v, mask, slot: int, target, iters: int = 200):
    """Set the valence of candidate `slot` so that its sincere first-choice share equals `target`."""
    v = v.clone()
    t = torch.as_tensor(target, dtype=torch.float32).expand(len(pi))
    for _ in range(iters):
        p = expected_shares(model, pi, c, v, mask)[:, slot]
        if (p - t).abs().max() < 1e-5:
            break
        v[:, slot] += torch.logit(t) - torch.logit(p.clamp(1e-9, 1 - 1e-9))
    return v


# ---------------------------------------------------------------- ballots under strategy
def _front_utils(U, front):
    fa = U.gather(2, front[:, None, 0:1].expand(-1, U.shape[1], 1)).squeeze(-1)
    fb = U.gather(2, front[:, None, 1:2].expand(-1, U.shape[1], 1)).squeeze(-1)
    return fa, fb


def compromise(U, front, strat):
    """Strategic voters move the frontrunner they prefer to the top of their ballot."""
    fa, fb = _front_utils(U, front)
    pref = torch.where(fa >= fb, front[:, 0:1], front[:, 1:2])                          # [B,N]
    bump = torch.zeros_like(U).scatter_(2, pref.unsqueeze(-1), BIG)
    return U + bump * strat.unsqueeze(-1)


def approval_frontrunner(U, mask, front):
    """Approve the frontrunner you prefer and everyone you like more than the midpoint between the two."""
    fa, fb = _front_utils(U, front)
    return vm._argmax_valid((U >= ((fa + fb) / 2).unsqueeze(-1)).float().sum(1), mask)


def star_frontrunner(U, mask, front):
    """Exaggerate: preferred frontrunner (and anyone better) gets 5, the other (and anyone worse) gets 0."""
    fa, fb = _front_utils(U, front)
    hi, lo = torch.maximum(fa, fb).unsqueeze(-1), torch.minimum(fa, fb).unsqueeze(-1)
    s = torch.round(5 * (U.clamp(lo, hi) - lo) / (hi - lo).clamp_min(1e-9))
    return vm.star_from_scores(s, mask)


# ---------------------------------------------------------------- simulation
@torch.no_grad()
def run_methods(model: Compass, pi, c, v, ncand, front, rho_plurality, n_iter: int = 20,
                n_voters: int = 1000, seed: int = 0, chunk_cells: int = 10_000_000) -> dict:
    """Simulate contests with >= 3 candidates under every scenario method.

    Returns per method:
      wins  [C, J]  share of runs each candidate slot won
      expr  [C]     "compass expression": minus the mean squared compass distance voter -> winner
      cw    [C]     share of runs in which the method elected the honest Condorcet winner
    and "first" [C, J], the agents' sincere first-choice shares.
    """
    g = torch.Generator().manual_seed(seed)
    C, Jmax = c.shape[0], c.shape[1]
    res = {m: {"wins": torch.zeros(C, Jmax), "expr": torch.zeros(C), "cw": torch.zeros(C)} for m in SCENARIO_METHODS}
    first = torch.zeros(C, Jmax)
    mu, sig = model.mu, model.log_sig.exp()
    rho_c = torch.as_tensor(rho_plurality, dtype=torch.float32).expand(C)
    for J in range(3, Jmax + 1):
        idx = torch.where(ncand == J)[0]
        if not len(idx):
            continue
        per = max(1, chunk_cells // (n_voters * J * J))
        flat = torch.arange(len(idx)).repeat(n_iter)
        for a in range(0, len(flat), per):
            r = flat[a:a + per]; gi = idx[r]; B = len(r)
            k = torch.multinomial(pi[gi], n_voters, replacement=True, generator=g)
            z = mu[k] + sig[k] * torch.randn(B, n_voters, mu.shape[1], generator=g)
            d2 = ((z[:, :, None, :] - c[gi, :J][:, None, :, :]) ** 2).sum(-1)                  # [B,N,J]
            U = -d2 + v[gi, :J][:, None, :]
            U = U - torch.log(-torch.log(torch.rand(U.shape, generator=g).clamp(1e-12, 1 - 1e-7)))
            mask = torch.ones(B, J, dtype=torch.bool); fr = front[gi]
            all_strat = torch.ones(B, n_voters, dtype=torch.bool)
            some_strat = torch.rand(B, n_voters, generator=g) < rho_c[gi][:, None]
            w = {"plurality_honest": vm.plurality(U, mask),
                 "plurality_strategic": vm.plurality(compromise(U, fr, some_strat), mask),
                 "random_ballot": vm.random_ballot(U, mask),
                 "rcv_honest": vm.ranked_choice(U, mask),
                 "rcv_strategic": vm.ranked_choice(compromise(U, fr, all_strat), mask),
                 "approval_honest": vm.approval(U, mask),
                 "approval_strategic": approval_frontrunner(U, mask, fr),
                 "star_honest": vm.star(U, mask),
                 "star_strategic": star_frontrunner(U, mask, fr),
                 "borda": vm.borda(U, mask), "condorcet": vm.condorcet(U, mask)}
            dist = d2.mean(1)                                                                   # [B,J]
            first.index_add_(0, gi, F.one_hot(U.argmax(-1), Jmax).float().mean(1)[:, :Jmax] / n_iter)
            for m, win in w.items():
                res[m]["wins"].index_put_((gi, win), torch.full((B,), 1.0 / n_iter), accumulate=True)
                res[m]["expr"].index_add_(0, gi, -dist.gather(1, win[:, None]).squeeze(1) / n_iter)
                res[m]["cw"].index_add_(0, gi, (win == w["condorcet"]).float() / n_iter)
    res["first"] = first
    return res


# ---------------------------------------------------------------- Gallup party identification
def gallup_party_id(csv_dir=CSV_DIR) -> pd.DataFrame:
    """Yearly five-way party ID (%): Democrat, lean Democrat, pure independent, lean Republican, Republican."""
    a = pd.read_csv(csv_dir / "data-1aTsg.csv", encoding="utf-8-sig").iloc[1:]
    b = pd.read_csv(csv_dir / "data-gOKi4.csv", encoding="utf-8-sig").iloc[1:]
    a.columns = ["date", "R", "I", "D"]; b.columns = ["date", "Rl", "Dl"]
    g = a.merge(b, on="date")
    for col in ["R", "I", "D", "Rl", "Dl"]:
        g[col] = pd.to_numeric(g[col], errors="coerce")
    g["year"] = g.date.str[:4].astype(int)
    g = g.assign(lean_D=g.Dl - g.D, lean_R=g.Rl - g.R, pure_I=100 - g.Dl - g.Rl)
    return g.groupby("year")[["D", "lean_D", "pure_I", "lean_R", "R"]].mean()


@torch.no_grad()
def party_id_shares(model: Compass, state_weights, t1: float, t2: float, use_wave: bool = True) -> np.ndarray:
    """[T, 5] national shares of the five party-ID groups implied by the compass.

    A voter's partisan gap is utility(generic Democrat) - utility(generic Republican) plus a logistic
    taste shock. |gap| < t1 -> pure independent, t1..t2 -> leaner, > t2 -> party identifier.
    state_weights [S, T]: each state's share of the national electorate.
    """
    S, T = model.A.shape[0], len(YEARS)
    z = model.mu.unsqueeze(1) + model.log_sig.exp().unsqueeze(1) * model.eps.unsqueeze(0)       # [K,M,d]
    out = np.zeros((T, 5))
    for t in range(T):
        pd_, pr = model.party_pos[t, 0], model.party_pos[t, 1]
        gap = -((z - pd_) ** 2).sum(-1) + ((z - pr) ** 2).sum(-1)                              # [K,M]
        if use_wave:
            gap = gap + model.wave0[2] + model.wave[t, 2]
        cdf = lambda th: torch.sigmoid(gap - th).mean(1)                                        # P(gap + eps > th) per type
        by_type = torch.stack([cdf(t2), cdf(t1) - cdf(t2), cdf(-t1) - cdf(t1), cdf(-t2) - cdf(-t1), 1 - cdf(-t2)], 1)
        pi = model.weights(torch.arange(S), torch.full((S,), t))                                # [S,K]
        w = torch.as_tensor(state_weights[:, t], dtype=torch.float32)
        out[t] = ((w[:, None] * pi).sum(0) @ by_type).numpy()
    return 100 * out


def stretched(model: Compass, k: float) -> Compass:
    """A copy of the model with the whole compass scaled by k (voters k times further apart).

    Spatial utility differences grow by k^2 relative to the taste shock, i.e. voters are more
    policy-driven. Candidate positions must be scaled by the same k by the caller.
    """
    import copy
    m = copy.deepcopy(model)
    with torch.no_grad():
        m.mu.mul_(k); m.log_sig.add_(float(np.log(k))); m.party_pos.mul_(k); m.delta.mul_(k)
    return m


# ---------------------------------------------------------------- measuring suppression per year
def minor_candidate_table(ct, v) -> pd.DataFrame:
    """One row per third-party candidate: appeal relative to the strongest major-party candidate.

    v must be the valences calibrated to real shares with no desertion assumed (sim.prepare).
    `two` marks contests where both a Democrat and a Republican ran; `margin` is the gap between
    them as a share of their combined vote (0 when only one major party ran).
    """
    g, m = ct.group, ct.mask
    major, minor = (g < 2) & m, (g >= 2) & m
    two = ((g == 0) & m).any(1) & ((g == 1) & m).any(1)
    v_major = torch.where(major, v, torch.tensor(-1e9)).max(1).values
    ym = torch.where(major, ct.y, torch.tensor(0.0)).sort(1, descending=True).values
    margin = torch.where(two, (ym[:, 0] - ym[:, 1]) / (ym[:, 0] + ym[:, 1]).clamp_min(1e-9), torch.tensor(0.0))
    rows = []
    for ci in torch.where(minor.any(1) & major.any(1))[0].tolist():
        for j in torch.where(minor[ci])[0].tolist():
            rows.append({"contest": ci, "year": int(ct.meta.eff_year[ci]), "office": ct.meta.office[ci],
                         "group": int(g[ci, j]), "two": bool(two[ci]), "margin": margin[ci].item(),
                         "rel_appeal": (v[ci, j] - v_major[ci]).item(), "share": ct.y[ci, j].item()})
    return pd.DataFrame(rows)


def measure_suppression(ct, v):
    """Per-year suppression of third-party candidates when both major parties are on the ballot.

    Regression on third-party candidates' relative appeal:
        rel_appeal = year effect + party-group effect + office effect
                     - gap[year] * two  +  kappa * two * margin
    gap[year] is how much less appealing a third-party candidate looks (log-odds) in a dead-heat
    two-major contest than in a contest with only one major party, where there is nobody to spoil
    and the third candidate is the only alternative. It therefore bundles strategic desertion with
    "not a serious contender". kappa is the part that eases as the major-party race gets less close.
    Year gaps are shrunk toward the pooled value in proportion to their noise (empirical Bayes).

    Returns (per-year DataFrame, info dict, per-contest suppression rate tensor [C]).
    """
    d = minor_candidate_table(ct, v)
    years = sorted(d.year.unique()); T = len(years)
    yi = d.year.map({y: i for i, y in enumerate(years)}).to_numpy()
    two = d.two.to_numpy(float); n = len(d)
    X = np.zeros((n, 2 * T + 1 + 2 + 1))
    X[np.arange(n), yi] = 1.0
    X[np.arange(n), T + yi] = -two
    X[:, 2 * T] = two * d.margin.to_numpy()
    X[:, 2 * T + 1] = d.group.to_numpy() == 3; X[:, 2 * T + 2] = d.group.to_numpy() == 4
    X[:, 2 * T + 3] = d.office.to_numpy() == "S"
    n_one = np.array([((yi == t) & (two == 0)).sum() for t in range(T)])
    keep = np.ones(X.shape[1], bool); keep[T + np.where(n_one < 3)[0]] = False        # gap not identified that year
    y = d.rel_appeal.to_numpy()
    Xk = X[:, keep]
    beta, *_ = np.linalg.lstsq(Xk, y, rcond=None)
    resid = y - Xk @ beta
    cov = np.linalg.pinv(Xk.T @ Xk) * (resid @ resid) / (n - Xk.shape[1])
    b = np.full(X.shape[1], np.nan); se = np.full(X.shape[1], np.nan)
    b[keep] = beta; se[keep] = np.sqrt(np.diag(cov))
    gap, gse = b[T:2 * T], se[T:2 * T]; ok = ~np.isnan(gap)
    w = 1 / gse[ok] ** 2
    pooled = float((gap[ok] * w).sum() / w.sum())
    tau2 = max(0.0, float(np.var(gap[ok], ddof=1) - np.mean(gse[ok] ** 2)))
    shrunk = np.where(ok, pooled + tau2 / (tau2 + np.nan_to_num(gse, nan=1e9) ** 2) * (np.nan_to_num(gap) - pooled), pooled)
    kappa_hat, kappa_se = float(b[2 * T]), float(se[2 * T])
    kappa = kappa_hat if kappa_se > 0 and abs(kappa_hat / kappa_se) > 2 else 0.0          # use the closeness term only if it is detectable
    # per-contest rate: only where both majors ran; eases with the major-party margin
    g, m = ct.group, ct.mask
    two_c = ((g == 0) & m).any(1) & ((g == 1) & m).any(1)
    ym = torch.where((g < 2) & m, ct.y, torch.tensor(0.0)).sort(1, descending=True).values
    margin_c = (ym[:, 0] - ym[:, 1]) / (ym[:, 0] + ym[:, 1]).clamp_min(1e-9)
    gap_c = torch.tensor([shrunk[years.index(y)] if y in years else pooled for y in ct.meta.eff_year], dtype=torch.float32)
    rho_total = (1 - torch.exp(-(gap_c - kappa * margin_c).clamp_min(0))) * two_c
    rho_close = (1 - torch.exp(-(kappa * (0.6 - margin_c)).clamp_min(0))) * two_c           # closeness-only, vs a 60-point blowout
    has_minor = ((g >= 2) & m).any(1)
    yr = np.array(ct.meta.eff_year)
    per_year = pd.DataFrame({
        "year": years, "one_major_candidates": n_one, "two_major_candidates": [((yi == t) & (two == 1)).sum() for t in range(T)],
        "gap_raw": gap, "gap_se": gse, "gap_shrunk": shrunk,
        "suppression_dead_heat": 1 - np.exp(-np.clip(shrunk, 0, None)),
        "suppression_mean": [rho_total[torch.as_tensor((yr == y)) & two_c & has_minor].mean().item() for y in years],
        "closeness_only_mean": [rho_close[torch.as_tensor((yr == y)) & two_c & has_minor].mean().item() for y in years],
        "minor_share_two_major": [d[(d.year == y) & d.two].share.mean() for y in years],
        "minor_share_one_major": [d[(d.year == y) & ~d.two].share.mean() for y in years]})
    one = d[~d.two]
    info = {"pooled_gap": pooled, "between_year_sd": tau2 ** 0.5, "kappa": kappa_hat, "kappa_se": kappa_se, "kappa_used": kappa,
            "pooled_suppression_dead_heat": 1 - float(np.exp(-pooled)),
            "seriousness_gap_one_major": float(-one.rel_appeal.mean()),
            "seriousness_gap_by_group": {k: float(-one[one.group == i].rel_appeal.mean()) for k, i in (("L", 2), ("G", 3), ("X", 4))},
            "minor_candidates": int(n), "one_major_candidates": int((two == 0).sum())}
    return per_year, info, rho_total, rho_close


@torch.no_grad()
def party_id_by_type(model: Compass, t1: float, t2: float, use_wave: bool = True) -> np.ndarray:
    """[T, K, 5] five-way party-ID shares of each voter type in each year (same rule as party_id_shares).

    Any electorate's shares are its type mix times this table, so states and districts can be read off cheaply.
    """
    z = model.mu.unsqueeze(1) + model.log_sig.exp().unsqueeze(1) * model.eps.unsqueeze(0)
    out = []
    for t in range(len(YEARS)):
        gap = -((z - model.party_pos[t, 0]) ** 2).sum(-1) + ((z - model.party_pos[t, 1]) ** 2).sum(-1)
        if use_wave:
            gap = gap + model.wave0[2] + model.wave[t, 2]
        cdf = lambda th: torch.sigmoid(gap - th).mean(1)
        out.append(torch.stack([cdf(t2), cdf(t1) - cdf(t2), cdf(-t1) - cdf(t1), cdf(-t2) - cdf(-t1), 1 - cdf(-t2)], 1).numpy())
    return np.stack(out)
