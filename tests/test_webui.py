"""Checks for the web UI data pipeline (scripts/build_webui_geo.py, simulate.save_contest_wins)."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import build_webui_geo as geo
from compass import simulate as sim


def _square(x0, y0, x1, y1, n=40):
    """Clockwise ring (shapefile outer winding) with n points per side."""
    t = np.linspace(0, 1, n, endpoint=False)
    left = np.c_[np.full(n, x0), y0 + (y1 - y0) * t]; top = np.c_[x0 + (x1 - x0) * t, np.full(n, y1)]
    right = np.c_[np.full(n, x1), y1 - (y1 - y0) * t]; bottom = np.c_[x1 - (x1 - x0) * t, np.full(n, y0)]
    r = np.vstack([left, top, right, bottom]); return np.vstack([r, r[:1]])


def test_neighbouring_districts_share_one_border_arc():
    feats = {"AA-1": [geo.snap(_square(0, 0, 1, 1))], "AA-2": [geo.snap(_square(1, 0, 2, 1))]}
    topo = geo.build_topology(feats)
    a, b = [{(i if i >= 0 else ~i) for i in g["arcs"][0][0]} for g in topo["objects"]["cd"]["geometries"]]
    assert len(a & b) == 1                                   # the shared edge is stored once
    assert sum(len(x) for x in topo["arcs"]) <= 12           # straight sides collapse to their corners


def test_simplification_keeps_corners():
    line = np.array([[0, 0], [1, 0.0001], [2, 0], [2, 2]], float)
    assert geo.douglas_peucker(line, 0.01).tolist() == [[0, 0], [2, 0], [2, 2]]


def test_contest_win_counts(tmp_path):
    ct = SimpleNamespace(ids=["a", "b"], group=torch.zeros(2, sim.MAX_CANDS, dtype=torch.long),
                         mask=torch.ones(2, sim.MAX_CANDS, dtype=torch.bool), y=torch.full((2, sim.MAX_CANDS), 1 / sim.MAX_CANDS))
    win = {"plurality": torch.tensor([[0, 1], [0, 2], [1, 2]], dtype=torch.int8)}       # [iterations, contests]
    sim.save_contest_wins(tmp_path / "w.npz", ct, win, 3)
    w = np.load(tmp_path / "w.npz")["wins_plurality"]
    assert w[0].tolist() == [2, 1, 0, 0, 0, 0] and w[1].tolist() == [0, 1, 2, 0, 0, 0]
