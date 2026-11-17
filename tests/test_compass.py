"""Run with: .venv/bin/python -m pytest tests -q"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from compass import baseline, evaluate as ev, synth
from compass.data import build_races, to_tensors
from compass.fit import fit
from compass.model import Config, build


@pytest.fixture(scope="module")
def df():
    return build_races(offices=("P", "S"))


@pytest.fixture(scope="module")
def rd(df):
    return to_tensors(df)


def test_shares_are_simplex(df, rd):
    assert np.allclose(df.groupby("race").share.sum(), 1.0)
    assert torch.allclose(rd.y.sum(1), torch.ones(len(rd)), atol=1e-5)
    assert (rd.y[rd.mask] >= 0.05 - 1e-6).all()


def test_known_races(df):
    me92 = df[df.race == "1992-P-ME-"].set_index("name")
    assert set(me92.party) == {"D", "R", "O"} and abs(me92.loc["PEROT, ROSS", "share"] - 0.306) < 0.01
    ny16 = df[df.race == "2016-P-NY-"].set_index("name")      # fusion lines summed onto one candidate
    assert len(ny16) == 2 and ny16.loc["CLINTON, HILLARY", "party"] == "D"
    assert 0.60 < ny16.loc["CLINTON, HILLARY", "share"] < 0.63
    assert df.groupby("race").party.apply(lambda p: (p == "D").sum()).max() <= 4


def test_incumbency(df):
    ak = df[(df.race == "2010-S-AK-") & df.name.str.contains("MURKOWSKI")]
    assert ak.inc.iloc[0] == 1.0
    assert df[df.office == "P"].inc.sum() == 0


def test_forward_is_distribution(rd):
    m = build(rd, Config(K=3, M=32))
    with torch.no_grad():
        p = m(rd)
    assert torch.allclose(p.sum(1), torch.ones(len(rd)), atol=1e-4)
    assert (p[~rd.mask] == 0).all() and (p >= 0).all()


def test_nearest_candidate_wins():
    """A voter cloud sitting on the Democrat's position must favour the Democrat."""
    rd = to_tensors(build_races(offices=("P",)))
    m = build(rd, Config(K=1, M=64, adapt=False))
    with torch.no_grad():
        m.mu.copy_(m.party_pos[0, 0:1]); m.log_sig.fill_(-3.0)
        assert (ev.d_share(m(rd), rd) > 0.5).all()
        m.mu.copy_(m.party_pos[0, 1:2])
        assert (ev.d_share(m(rd), rd) < 0.5).all()


def test_gradients_finite(rd):
    m = build(rd, Config(K=3, M=32))
    (-(m.log_lik(rd).sum() + m.log_prior())).backward()
    for n, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), n


def test_synthetic_recovery(rd):
    """Fit on simulated elections: must beat the regression baseline and approach the noise floor."""
    cfg = Config(K=3, M=64)
    truth = synth.make_truth(rd, cfg)
    srd, clean = synth.simulate(truth, rd)
    tr, te = ev.random_split(srd, 0.2, 0)
    train, test = srd.subset(tr), srd.subset(te)
    m = build(srd, cfg, seed=1)
    fit(m, train, steps=600, verbose=False)
    with torch.no_grad():
        got = ev.two_party_metrics(ev.d_share(m(test), test), test)
    base = ev.two_party_metrics(baseline.fit_predict(train, test, True), test)
    floor = ev.two_party_metrics(ev.d_share(clean[te], test), test)
    assert got["P"]["rmse"] < base["P"]["rmse"]
    assert got["P"]["rmse"] < 2.0 * floor["P"]["rmse"]
    assert got["S"]["rmse"] < 1.1 * base["S"]["rmse"]


def test_procrustes_invariance():
    g = np.random.default_rng(0)
    X = g.normal(size=(50, 2)); th = 0.7
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    assert synth.procrustes_r2(X, 2.5 * X @ R + 3.0) > 0.999
