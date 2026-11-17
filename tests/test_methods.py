import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from compass import methods as vm


def profile(groups):
    """groups: list of (count, ranking best->worst) over candidates 0..J-1 -> U [1, N, J]."""
    J = len(groups[0][1]); rows = []
    for n, order in groups:
        u = torch.zeros(J)
        for rank, c in enumerate(order):
            u[c] = float(J - rank)
        rows += [u] * n
    return torch.stack(rows)[None], torch.ones(1, J, dtype=torch.bool)


def test_centre_squeeze():
    """40: A>B>C, 35: C>B>A, 25: B>A>C. A leads on first choices, B is everyone's compromise."""
    U, m = profile([(40, [0, 1, 2]), (35, [2, 1, 0]), (25, [1, 0, 2])])
    assert vm.plurality(U, m).item() == 0
    assert vm.ranked_choice(U, m).item() == 0        # B eliminated first, B's voters go to A
    assert vm.condorcet(U, m).item() == 1
    assert vm.borda(U, m).item() == 1
    assert vm.approval(U, m).item() == 1             # middle choice sits exactly at the mean -> approved
    assert vm.star(U, m).item() == 1


def test_approval_ignores_fringe_with_poll_weights():
    """48: A>B>F, 52: B>A>F where F is a hopeless fringe candidate everyone ranks last."""
    U, m = profile([(48, [0, 1, 2]), (52, [1, 0, 2])])
    polls = torch.tensor([[0.48, 0.52, 0.0]])
    assert vm.approval(U, m, polls).item() == 1            # behaves like the head-to-head
    naive = ((U >= U.mean(-1, keepdim=True)).float().sum(1))  # naive rule: everyone approves both A and B
    assert naive[0, 0] == naive[0, 1] == 100


def test_irv_transfer_changes_winner():
    """45: A>C>B, 30: B>C>A, 25: C>B>A -> C out, votes go to B, B beats A 55-45."""
    U, m = profile([(45, [0, 2, 1]), (30, [1, 2, 0]), (25, [2, 1, 0])])
    assert vm.plurality(U, m).item() == 0
    assert vm.ranked_choice(U, m).item() == 1
    assert vm.condorcet(U, m).item() == 2


def test_two_candidates_all_agree():
    U, m = profile([(60, [1, 0]), (40, [0, 1])])
    for name, f in vm.METHODS.items():
        if name != "random_ballot":
            assert f(U, m).item() == 1, name


def test_padding_is_ignored():
    U, m = profile([(40, [0, 1, 2]), (35, [2, 1, 0]), (25, [1, 0, 2])])
    Up = torch.cat([U, torch.full((1, 100, 2), 99.0)], -1)         # two fake, very attractive slots
    mp = torch.cat([m, torch.zeros(1, 2, dtype=torch.bool)], -1)
    for name, f in vm.METHODS.items():
        assert f(Up, mp).item() == f(U, m).item(), name


def test_random_ballot_is_proportional():
    g = torch.Generator().manual_seed(0)
    U = torch.zeros(4000, 1, 2); pick = torch.rand(4000, generator=g) < 0.3
    U[pick, 0, 0] = 1.0; U[~pick, 0, 1] = 1.0
    w = vm.random_ballot(U, torch.ones(4000, 2, dtype=torch.bool))
    assert abs((w == 0).float().mean().item() - 0.3) < 0.03
