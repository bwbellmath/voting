"""Voting methods applied to agents' utilities.

Every function takes
    U    [B, N, J]  utility of voter n for candidate j in contest b (higher = better)
    mask [B, J]     which candidate slots are real
and returns the winning candidate index [B]. Ballots are sincere: they are derived from the
utilities with no strategic behaviour.
"""
from __future__ import annotations

import torch

NEG = -1e9


def _clean(U, mask):
    return U.masked_fill(~mask[:, None, :], NEG)


def _first_choice_counts(U, alive):
    """[B, J] number of voters whose favourite among `alive` candidates is j."""
    top = U.masked_fill(~alive[:, None, :], NEG).argmax(-1)                       # [B,N]
    return torch.zeros(alive.shape, dtype=torch.float32).scatter_add_(1, top, torch.ones(top.shape))


def _argmax_valid(score, mask):
    return score.masked_fill(~mask, NEG).argmax(-1)


def plurality(U, mask):
    """Single preference: most first-choice votes wins."""
    return _argmax_valid(_first_choice_counts(U, mask), mask)


def random_ballot(U, mask):
    """The original 'dice' method: one voter is drawn at random and their first choice wins."""
    return U[:, 0].masked_fill(~mask, NEG).argmax(-1)


def ranked_choice(U, mask):
    """Instant-runoff: repeatedly eliminate the candidate with the fewest first choices."""
    alive = mask.clone()
    for _ in range(mask.shape[1] - 1):
        counts = _first_choice_counts(U, alive)
        more = alive.sum(1) > 1
        loser = counts.masked_fill(~alive, 1e9).argmin(-1)
        alive[more, loser[more]] = False
    return alive.float().argmax(-1)


def approval(U, mask, weights=None):
    """Approve every candidate you like at least as much as your threshold.

    weights=None: threshold is the plain average over the candidates on the ballot (zero-information
    rule). A fringe candidate then drags the threshold down and most voters approve both major
    candidates, so results swing on who else is on the ballot.
    weights=[B, J] (e.g. poll shares): threshold is the weighted average, i.e. "approve anyone better
    than what I expect to get". This is what the simulation uses.
    """
    m = mask[:, None, :]
    w = m.float() if weights is None else (weights * mask)[:, None, :]
    thresh = (U.masked_fill(~m, 0.0) * w).sum(-1, keepdim=True) / w.sum(-1, keepdim=True)
    return _argmax_valid(((U >= thresh) & m).float().sum(1), mask)


def _scores(U, mask, top: int = 5):
    """0..top integer scores: favourite gets `top`, least favourite 0, others scaled linearly."""
    m = mask[:, None, :]
    hi = U.masked_fill(~m, NEG).max(-1, keepdim=True).values
    lo = U.masked_fill(~m, -NEG).min(-1, keepdim=True).values
    return (torch.round(top * (U - lo) / (hi - lo).clamp_min(1e-9)) * m)


def score(U, mask):
    """Range voting: highest total 0-5 score wins."""
    return _argmax_valid(_scores(U, mask).sum(1), mask)


def star_from_scores(s, mask):
    """STAR tally from a [B, N, J] score tensor."""
    tot = s.sum(1).masked_fill(~mask, NEG)
    a = tot.argmax(-1)
    b = tot.scatter(1, a[:, None], NEG).argmax(-1)
    sa = s.gather(2, a[:, None, None].expand(-1, s.shape[1], 1)).squeeze(-1)
    sb = s.gather(2, b[:, None, None].expand(-1, s.shape[1], 1)).squeeze(-1)
    b_wins = (sb > sa).sum(1) > (sa > sb).sum(1)
    return torch.where(b_wins & (mask.sum(1) > 1), b, a)


def star(U, mask):
    """Score Then Automatic Runoff: top two by total score, then whoever more voters scored higher."""
    return star_from_scores(_scores(U, mask), mask)


def _pairwise(U, mask):
    """[B, J, J] number of voters preferring a to b."""
    Uc = _clean(U, mask)
    return (Uc[:, :, :, None] > Uc[:, :, None, :]).float().sum(1)


def borda(U, mask):
    """Borda count: one point for every candidate ranked below."""
    P = _pairwise(U, mask) * mask[:, None, :]
    return _argmax_valid(P.sum(-1), mask)


def condorcet(U, mask):
    """Condorcet winner (beats everyone head-to-head); if there is a cycle, minimax margin."""
    P = _pairwise(U, mask)
    margin = P - P.transpose(1, 2)
    eye = torch.eye(mask.shape[1], dtype=torch.bool)[None]
    worst = margin.masked_fill(eye | ~mask[:, None, :], -NEG).min(-1).values
    return _argmax_valid(worst, mask)


METHODS = {
    "plurality": plurality, "random_ballot": random_ballot, "ranked_choice": ranked_choice,
    "approval": approval, "star": star, "borda": borda, "condorcet": condorcet,
}
TITLES = {
    "plurality": "Plurality (single preference)", "random_ballot": "Random ballot (the original dice roll)",
    "ranked_choice": "Ranked choice (instant runoff)", "approval": "Approval", "star": "STAR",
    "borda": "Borda count", "condorcet": "Condorcet (minimax)",
}
