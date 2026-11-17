"""Non-spatial baselines: ridge regression on the logit of the two-party Democratic share.

B0 "swing":  state effect + (year x office) effect
B1 "trend":  B0 + per-state linear time trend + incumbency by office + House district effect
These answer: does the compass predict held-out races better than state lean + national wave?
"""
from __future__ import annotations

import numpy as np
import torch

from .data import RaceData, YEARS, OFFICES


def _design(rd: RaceData, full: bool):
    S, T, O, U = len(rd.states), len(YEARS), len(OFFICES), len(rd.units)
    R = len(rd)
    r = np.arange(R)
    st, yr, of, un = rd.state.numpy(), rd.year.numpy(), rd.office.numpy(), rd.unit.numpy()
    blocks, ridge = [], []
    X = np.zeros((R, S)); X[r, st] = 1; blocks.append(X); ridge.append(np.full(S, 1e-3))
    X = np.zeros((R, T * O)); X[r, yr * O + of] = 1; blocks.append(X); ridge.append(np.full(T * O, 1e-2))
    if full:
        tau = (np.array(YEARS)[yr] - 1998) / 22
        X = np.zeros((R, S)); X[r, st] = tau; blocks.append(X); ridge.append(np.full(S, 1e-2))
        d = ((rd.party == 0) & rd.mask).float(); rr = ((rd.party == 1) & rd.mask).float()
        net = ((rd.inc * d).sum(1) - (rd.inc * rr).sum(1)).numpy()
        X = np.zeros((R, O)); X[r, of] = net; blocks.append(X); ridge.append(np.full(O, 1e-3))
        X = np.zeros((R, U)); X[r, un] = (un > 0); blocks.append(X); ridge.append(np.full(U, 1.0))
    return np.hstack(blocks), np.concatenate(ridge)


def fit_predict(train: RaceData, test: RaceData, full: bool = True) -> torch.Tensor:
    """Predicted two-party D share for every test race (NaN-free; only meaningful on D-vs-R races)."""
    ok, yd = train.two_party()
    Xtr, ridge = _design(train, full)
    Xtr = Xtr[ok.numpy()]
    y = torch.logit(yd[ok].clamp(0.02, 0.98)).numpy()
    beta = np.linalg.solve(Xtr.T @ Xtr + np.diag(ridge), Xtr.T @ y)
    Xte, _ = _design(test, full)
    return torch.sigmoid(torch.tensor(Xte @ beta, dtype=torch.float32))
