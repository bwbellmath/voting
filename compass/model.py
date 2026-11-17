"""Latent voter-type compass model.

Voters:      K fixed national "types", type k ~ N(mu_k, diag(sig_k^2)) in R^d.
             State s in year t is a mixture of the types with weights
             pi_{s,t} = softmax(A_s + B_s*tau [+ C_s*tau^2 if quad_trend]); a House district adds an offset G_u.
Candidates:  position c_j in the same space plus a scalar valence v_j.
             major-party congressional: c = party_pos[t, party] + lam * (unit mean - national mean)
             presidential / third-party: c = (party_pos[t, party] or 0) + delta_j
             v = (wave0[office] + wave[t, office]) (Democrats only) + beta_inc[office] * incumbent + u_j
             wave0 is the long-run party advantage; wave[t] is that year's national swing, shrunk to 0,
             so an unseen year is forecast with the average environment.
Voting:      P(voter z picks j) = softmax_j( -|z - c_j|^2 + v_j )   (probabilistic nearest-candidate)
Aggregate:   share_j = sum_k pi_k * E_{z ~ type k}[P(j | z)], expectation by quasi-Monte Carlo.
Likelihood:  observed shares ~ Dirichlet(phi[office] * predicted shares).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import qmc, norm

from .data import RaceData, YEARS, OFFICES


@dataclass
class Config:
    d: int = 2
    K: int = 4
    M: int = 128              # QMC points per type
    quad_trend: bool = False  # quadratic state trends over-extrapolate in forecasts
    adapt: bool = True        # congressional candidates shift toward their constituency
    # prior scales (std devs)
    s_mu: float = 1.5
    s_A: float = 2.0
    s_B: float = 1.0
    s_C: float = 0.5
    s_G: float = 1.0
    s_walk: float = 0.03      # party position random walk per 2 years
    s_delta_pres: float = 0.5
    s_delta_other: float = 1.0
    s_u_pres: float = 1.0
    s_u: float = 0.5
    s_wave: float = 0.5
    # Dirichlet concentration per office (P, S, H). Fixed, not learned: with one race per
    # congressional candidate, a learned phi runs to infinity as the valences u_j absorb every race.
    phi: tuple = (1000.0, 300.0, 300.0)


class Compass(nn.Module):
    def __init__(self, n_states: int, n_cands: int, n_units: int, cfg: Config = Config(), seed: int = 0):
        super().__init__()
        self.cfg = cfg
        g = torch.Generator().manual_seed(seed)
        d, K, T = cfg.d, cfg.K, len(YEARS)
        self.mu = nn.Parameter(torch.randn(K, d, generator=g) * 0.7)
        self.log_sig = nn.Parameter(torch.full((K, d), math.log(0.5)))
        self.A = nn.Parameter(torch.randn(n_states, K, generator=g) * 0.1)
        self.B = nn.Parameter(torch.zeros(n_states, K))
        self.C = nn.Parameter(torch.zeros(n_states, K))
        self.G = nn.Parameter(torch.zeros(n_units, K))          # row 0 = statewide, kept at 0
        pp = torch.zeros(T, 2, d); pp[:, 0, 0] = -0.5; pp[:, 1, 0] = 0.5
        self.party_pos = nn.Parameter(pp)
        self.delta = nn.Parameter(torch.zeros(n_cands, d))
        self.u = nn.Parameter(torch.zeros(n_cands))
        self.wave = nn.Parameter(torch.zeros(T, len(OFFICES)))
        self.wave0 = nn.Parameter(torch.zeros(len(OFFICES)))
        self.beta_inc = nn.Parameter(torch.full((len(OFFICES),), 0.3))
        self.lam_raw = nn.Parameter(torch.tensor(-1.0))
        self.register_buffer("log_phi", torch.tensor(cfg.phi).log())
        eps = norm.ppf(qmc.Sobol(d, scramble=True, seed=seed).random(cfg.M).clip(1e-6, 1 - 1e-6))
        self.register_buffer("eps", torch.tensor(eps, dtype=torch.float32))
        self.register_buffer("tau", torch.tensor([(y - 1998) / 22 for y in YEARS], dtype=torch.float32))

    # ---- voters -------------------------------------------------------------
    def state_logits(self, state, year):
        tau = self.tau[year].unsqueeze(-1)
        out = self.A[state] + self.B[state] * tau
        return out + self.C[state] * tau ** 2 if self.cfg.quad_trend else out

    def weights(self, state, year, unit=None):
        lg = self.state_logits(state, year)
        if unit is not None:
            lg = lg + self.G[unit] * (unit > 0).unsqueeze(-1)
        return F.softmax(lg, -1)

    def national_mean(self):
        """[T, d] unweighted mean over states of the state mean position."""
        S, T = self.A.shape[0], len(YEARS)
        st = torch.arange(S).repeat_interleave(T); yr = torch.arange(T).repeat(S)
        return (self.weights(st, yr) @ self.mu).view(S, T, -1).mean(0)

    # ---- candidates ---------------------------------------------------------
    def candidates(self, rd: RaceData, pi):
        major = (rd.party < 2).unsqueeze(-1)
        pres = (rd.office == 0).view(-1, 1, 1)
        base = self.party_pos[rd.year.unsqueeze(1), rd.party.clamp(max=1)] * major          # [R,J,d]
        free = pres | ~major                                                                   # own position
        c = base + self.delta[rd.cand] * free
        if self.cfg.adapt:
            shift = pi @ self.mu - self.national_mean()[rd.year]                               # [R,d]
            c = c + torch.sigmoid(self.lam_raw) * shift.unsqueeze(1) * (major & ~pres)
        is_d = (rd.party == 0).float()
        wave = self.wave0[rd.office] + self.wave[rd.year, rd.office]
        v = self.u[rd.cand] + wave.unsqueeze(1) * is_d + self.beta_inc[rd.office].unsqueeze(1) * rd.inc
        return c, v

    # ---- shares ------------------------------------------------------------
    def forward(self, rd: RaceData, chunk: int = 4096):
        """Predicted shares [R, J] (0 on padded slots)."""
        z = self.mu.unsqueeze(1) + self.log_sig.exp().unsqueeze(1) * self.eps.unsqueeze(0)     # [K,M,d]
        z2 = (z ** 2).sum(-1)                                                                  # [K,M]
        pi = self.weights(rd.state, rd.year, rd.unit)                                          # [R,K]
        c, v = self.candidates(rd, pi)
        out = []
        for i in range(0, len(rd), chunk):
            cc, vv, mm = c[i:i + chunk], v[i:i + chunk], rd.mask[i:i + chunk]
            util = -(z2[None, :, :, None] - 2 * torch.einsum("kmd,rjd->rkmj", z, cc)
                     + (cc ** 2).sum(-1)[:, None, None, :]) + vv[:, None, None, :]
            util = util.masked_fill(~mm[:, None, None, :], -1e9)
            p_type = F.softmax(util, -1).mean(2)                                               # [r,K,J]
            out.append(torch.einsum("rk,rkj->rj", pi[i:i + chunk], p_type))
        return torch.cat(out)

    # ---- objective ---------------------------------------------------------
    def log_lik(self, rd: RaceData, pred=None, phi=None):
        """Per-race Dirichlet log density of the observed shares."""
        pred = self.forward(rd) if pred is None else pred
        phi = (self.log_phi.exp() if phi is None else torch.as_tensor(phi, dtype=torch.float32))[rd.office].unsqueeze(1)
        a = (phi * pred.clamp_min(1e-5)).masked_fill(~rd.mask, 1.0)
        logy = rd.y.clamp_min(1e-9).log().masked_fill(~rd.mask, 0.0)
        a_sum = (a * rd.mask).sum(1)
        return torch.lgamma(a_sum) - (torch.lgamma(a) * rd.mask).sum(1) + ((a - 1) * logy).sum(1)

    def log_prior(self):
        c = self.cfg
        n = lambda x, s: -0.5 * (x / s).pow(2).sum()
        lp = n(self.mu, c.s_mu) + n(self.log_sig - math.log(0.5), 1.0)
        lp = lp + n(self.A, c.s_A) + n(self.B, c.s_B) + n(self.C, c.s_C) + n(self.G, c.s_G)
        lp = lp + n(self.party_pos[1:] - self.party_pos[:-1], c.s_walk) + n(self.party_pos[0], 2.0)
        pres = self._pres_mask
        lp = lp + n(self.delta[pres], c.s_delta_pres) + n(self.delta[~pres], c.s_delta_other)
        lp = lp + n(self.u[pres], c.s_u_pres) + n(self.u[~pres], c.s_u)
        lp = lp + n(self.wave, c.s_wave) + n(self.wave0, 2.0) + n(self.beta_inc, 2.0) + n(self.lam_raw, 3.0)
        return lp

    def set_pres_mask(self, cands: list):
        """Presidential candidate ids are 'NAME|YEAR'; congressional are 'KEY|ST'."""
        m = torch.tensor([c.rsplit("|", 1)[1].isdigit() for c in cands])
        self.register_buffer("_pres_mask", m)


def build(rd: RaceData, cfg: Config = Config(), seed: int = 0) -> Compass:
    m = Compass(len(rd.states), len(rd.cands), len(rd.units), cfg, seed)
    m.set_pres_mask(rd.cands)
    return m
