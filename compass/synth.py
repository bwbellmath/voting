"""Synthetic recovery: simulate elections from a known compass on the real race design, refit, compare."""
from __future__ import annotations

import copy

import numpy as np
import torch

from .data import RaceData, YEARS
from .model import Compass, Config, build


def make_truth(rd: RaceData, cfg: Config, seed: int = 123) -> Compass:
    """A model whose parameters are drawn at random (the 'true' world)."""
    g = torch.Generator().manual_seed(seed)
    m = build(rd, cfg, seed)
    r = lambda *s: torch.randn(*s, generator=g)
    with torch.no_grad():
        m.mu.copy_(r(cfg.K, cfg.d) * 1.0)
        m.log_sig.copy_(torch.log(0.3 + 0.4 * torch.rand(cfg.K, cfg.d, generator=g)))
        m.A.copy_(r(*m.A.shape)); m.B.copy_(r(*m.B.shape) * 0.6); m.C.copy_(r(*m.C.shape) * 0.2)
        m.G.copy_(r(*m.G.shape) * 0.5); m.G[0] = 0
        walk = torch.cumsum(r(len(YEARS), 2, cfg.d) * 0.1, 0)
        base = torch.zeros(2, cfg.d); base[0, 0], base[1, 0] = -0.6, 0.6
        m.party_pos.copy_(base + walk)
        m.delta.copy_(r(*m.delta.shape) * 0.3)
        m.u.copy_(r(*m.u.shape) * 0.3)
        m.wave.copy_(r(*m.wave.shape) * 0.3)
        m.beta_inc.fill_(0.5); m.lam_raw.fill_(-0.8)
    return m


def simulate(truth: Compass, rd: RaceData, phi: float = 300.0, seed: int = 7):
    """Return (rd with simulated noisy shares, noise-free shares)."""
    torch.manual_seed(seed)
    with torch.no_grad():
        clean = truth(rd)
    alpha = (phi * clean).clamp_min(1e-3)
    y = torch._standard_gamma(alpha) * rd.mask
    y = y / y.sum(1, keepdim=True)
    out = copy.copy(rd); out.y = y.float()
    return out, clean


def procrustes_r2(X: np.ndarray, Y: np.ndarray) -> float:
    """R^2 of Y explained by the best translation + rotation/reflection + uniform scale of X."""
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    U, s, Vt = np.linalg.svd(Xc.T @ Yc)
    return float(s.sum() ** 2 / ((Xc ** 2).sum() * (Yc ** 2).sum()))


def state_means(m: Compass) -> np.ndarray:
    """[S*T, d] mean voter position of every state in every election year."""
    S, T = m.A.shape[0], len(YEARS)
    st = torch.arange(S).repeat_interleave(T); yr = torch.arange(T).repeat(S)
    with torch.no_grad():
        return (m.weights(st, yr) @ m.mu).numpy()
