"""Splits, metrics and calibration for checking the model against real elections."""
from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from scipy.stats import beta as beta_dist

from .data import RaceData, YEARS, OFFICES


def random_split(rd: RaceData, frac: float = 0.2, seed: int = 0):
    """Hold out a random fraction of races within each office."""
    g = np.random.default_rng(seed)
    test = np.zeros(len(rd), bool)
    for o in range(len(OFFICES)):
        idx = np.where(rd.office.numpy() == o)[0]
        test[g.choice(idx, int(round(frac * len(idx))), replace=False)] = True
    return np.where(~test)[0], np.where(test)[0]


def temporal_split(rd: RaceData, last_train_year: int = 2016):
    yr = np.array(YEARS)[rd.year.numpy()]
    return np.where(yr <= last_train_year)[0], np.where(yr > last_train_year)[0]


def d_share(pred: torch.Tensor, rd: RaceData) -> torch.Tensor:
    """Predicted D share of the D+R vote."""
    d = ((rd.party == 0) & rd.mask).float(); r = ((rd.party == 1) & rd.mask).float()
    pd_, pr = (pred * d).sum(1), (pred * r).sum(1)
    return pd_ / (pd_ + pr).clamp_min(1e-9)


def two_party_metrics(pd_: torch.Tensor, rd: RaceData) -> dict:
    """RMSE / MAE of the two-party D share and D-vs-R winner accuracy, per office."""
    ok, yd = rd.two_party()
    out = {}
    for o, name in enumerate(OFFICES):
        k = ok & (rd.office == o)
        if k.sum() == 0:
            continue
        e = (pd_ - yd)[k]
        out[name] = {"n": int(k.sum()), "rmse": e.pow(2).mean().sqrt().item(), "mae": e.abs().mean().item(),
                     "winner_acc": ((pd_[k] > 0.5) == (yd[k] > 0.5)).float().mean().item(),
                     "bias": e.mean().item()}
    return out


def calibrate_phi(pd_: torch.Tensor, rd: RaceData, idx) -> list:
    """Per-office Beta concentration that maximises the likelihood of two-party shares on races `idx`."""
    ok, yd = rd.two_party()
    sel = torch.zeros(len(rd), dtype=torch.bool); sel[torch.as_tensor(idx)] = True
    phis = []
    for o in range(len(OFFICES)):
        k = (ok & sel & (rd.office == o)).numpy()
        if k.sum() < 5:
            phis.append(float("nan")); continue
        p, y = pd_.numpy()[k].clip(1e-4, 1 - 1e-4), yd.numpy()[k].clip(1e-4, 1 - 1e-4)
        nll = lambda lp: -beta_dist.logpdf(y, np.exp(lp) * p, np.exp(lp) * (1 - p)).sum()
        phis.append(float(np.exp(minimize_scalar(nll, bounds=(0.0, 12.0), method="bounded").x)))
    return phis


def coverage(pd_: torch.Tensor, rd: RaceData, phis: list, idx, level: float = 0.9) -> dict:
    """Share of races `idx` whose two-party D share falls inside the central Beta interval."""
    ok, yd = rd.two_party()
    sel = torch.zeros(len(rd), dtype=torch.bool); sel[torch.as_tensor(idx)] = True
    out = {}
    for o, name in enumerate(OFFICES):
        k = (ok & sel & (rd.office == o)).numpy()
        if k.sum() < 5 or not np.isfinite(phis[o]):
            continue
        p, y = pd_.numpy()[k].clip(1e-4, 1 - 1e-4), yd.numpy()[k]
        lo = beta_dist.ppf((1 - level) / 2, phis[o] * p, phis[o] * (1 - p))
        hi = beta_dist.ppf(1 - (1 - level) / 2, phis[o] * p, phis[o] * (1 - p))
        out[name] = {"n": int(k.sum()), "coverage": float(((y >= lo) & (y <= hi)).mean()),
                     "mean_halfwidth": float(((hi - lo) / 2).mean())}
    return out
