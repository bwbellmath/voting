import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from compass import methods as vm, strategy as st
from compass.model import Compass, Config
from test_methods import profile


def test_mainstream_first_fixes_rcv_spoiler():
    """35: F>D>R (flank), 30: D>F>R, 35: R>D>F. Honest RCV drops D, then R beats F; D is the Condorcet winner."""
    U, m = profile([(36, [2, 0, 1]), (29, [0, 2, 1]), (35, [1, 0, 2])])   # slots: 0=D 1=R 2=F
    front = torch.tensor([[0, 1]])
    assert vm.condorcet(U, m).item() == 0
    assert vm.ranked_choice(U, m).item() == 2 or vm.ranked_choice(U, m).item() == 1   # not the Condorcet winner
    strat = torch.ones(1, U.shape[1], dtype=torch.bool)
    assert vm.ranked_choice(st.compromise(U, front, strat), m).item() == 0
    assert st.approval_frontrunner(U, m, front).item() == 0
    assert st.star_frontrunner(U, m, front).item() == 0


def test_strategic_calibration_recovers_shares_and_inflates_minor():
    torch.manual_seed(0)
    model = Compass(n_states=1, n_cands=1, n_units=1, cfg=Config(K=3, M=64))
    C = 8
    pi = torch.softmax(torch.randn(C, 3), -1); c = torch.randn(C, 3, 2) * 0.5
    y = torch.tensor([[0.50, 0.45, 0.05]]).expand(C, 3).contiguous(); mask = torch.ones(C, 3, dtype=torch.bool)
    front = torch.tensor([[0, 1]]).expand(C, 2).contiguous()
    with torch.no_grad():
        v, err = st.calibrate_strategic(model, pi, c, y, mask, front, 0.5)
        assert err < 1e-4
        assert torch.allclose(st.strategic_shares(model, pi, c, v, mask, front, 0.5), y, atol=1e-4)
        from compass.simulate import expected_shares
        sincere = expected_shares(model, pi, c, v, mask)
    assert torch.allclose(sincere[:, 2], torch.full((C,), 0.10), atol=1e-3)     # 5% observed / (1 - 0.5)


def test_gallup_five_way_sums_to_100():
    g = st.gallup_party_id()
    assert abs(g.sum(1) - 100).max() < 1e-6 and (g > 0).all().all() and 2020 in g.index


def test_measure_suppression_recovers_known_gap():
    """Third-party appeal is 1.5 log-odds lower when both majors run -> suppression ~ 1 - exp(-1.5)."""
    import numpy as np, pandas as pd
    from types import SimpleNamespace
    g = np.random.default_rng(0)
    years = [2000, 2002, 2004]; C = 900
    yr = g.choice(years, C); two = g.random(C) < 0.7
    group = torch.zeros(C, 3, dtype=torch.long); mask = torch.ones(C, 3, dtype=torch.bool)
    group[:, 0] = 0; group[:, 1] = 1; group[:, 2] = 2
    mask[~torch.as_tensor(two), 1] = False                               # one-major contests: no Republican
    v = torch.zeros(C, 3)
    v[:, 1] = torch.as_tensor(-np.abs(g.standard_normal(C)), dtype=torch.float32)   # varying major-party margins
    v[:, 2] = torch.as_tensor(-1.0 - 1.5 * two + 0.3 * g.standard_normal(C), dtype=torch.float32)
    y = torch.softmax(v.masked_fill(~mask, -1e9), 1)
    ct = SimpleNamespace(group=group, mask=mask, y=y,
                         meta=pd.DataFrame({"eff_year": yr, "office": ["H"] * C}))
    per_year, info, rho_total, _ = st.measure_suppression(ct, v)
    assert abs(info["pooled_gap"] - 1.5) < 0.1
    assert abs(info["pooled_suppression_dead_heat"] - (1 - np.exp(-1.5))) < 0.03
    assert (rho_total[~torch.as_tensor(two)] == 0).all()                 # nothing to remove where one major ran
    assert len(per_year) == 3 and abs(info["seriousness_gap_one_major"] - 1.0) < 0.1
