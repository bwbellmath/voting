"""MAP fitting by full-batch Adam."""
from __future__ import annotations

import time

import torch

from .data import RaceData
from .model import Compass


def fit(model: Compass, train: RaceData, steps: int = 2500, lr: float = 0.03, log_every: int = 250,
        verbose: bool = True) -> list:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr * 0.02)
    hist, t0 = [], time.time()
    for step in range(steps + 1):
        opt.zero_grad()
        ll = model.log_lik(train).sum()
        loss = -(ll + model.log_prior()) / len(train)
        loss.backward()
        opt.step(); sched.step()
        hist.append(loss.item())
        if verbose and step % log_every == 0:
            print(f"  step {step:5d}  loss/race {loss.item():8.4f}  loglik/race {ll.item() / len(train):7.4f}  {time.time() - t0:5.0f}s", flush=True)
    return hist
