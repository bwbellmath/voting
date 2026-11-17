"""Load the 1976-2020 president / senate / house CSVs into one race table and tensors.

A *race* is one contest in one geographic unit: (year, office, state[, district]).
Candidates below MIN_SHARE of the named-candidate vote are dropped and the remaining
shares renormalised, so every race is a point on a small simplex.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

CSV_DIR = Path(__file__).resolve().parent.parent / "csv"
MIN_SHARE = 0.05
OFFICES = ["P", "S", "H"]
PARTIES = ["D", "R", "O"]
YEARS = list(range(1976, 2021, 2))
JUNK = re.compile(
    r"^(|NAN|WRITEIN|WRITE-IN|WRITE IN|SCATTER\w*|BLANK.*|VOID.*|OTHER|OTHERS|OVER ?VOTES?|UNDER ?VOTES?|"
    r"NONE OF .*|NO VOTE.*|NOT DESIGNATED|UNKNOWN|MISCELLANEOUS|TOTAL.*|SPOILED.*|UNDERVOTES|OVERVOTES)$"
)


def _norm_name(s: pd.Series) -> pd.Series:
    s = s.fillna("").astype(str).str.upper().str.replace(r"[^A-Z, ]", "", regex=True)
    return s.str.replace(r"\s+", " ", regex=True).str.strip()


_SUFFIX = {"JR", "SR", "II", "III", "IV"}


def _person_key(name: str) -> str:
    """'TIMOTHY J PENNY' and 'TIM PENNY' -> 'PENNY T': tolerant of first-name spelling drift."""
    toks = [t for t in name.replace(",", " ").split() if t not in _SUFFIX]
    return f"{toks[-1]} {toks[0][0]}" if len(toks) >= 2 else name


def _party(s: pd.Series) -> pd.Series:
    s = s.fillna("").astype(str).str.upper()
    out = pd.Series("O", index=s.index)
    out[s.str.startswith("DEMOCRAT")] = "D"
    out[s.str.contains("REPUBLICAN") & ~s.str.contains("DEMOCRAT")] = "R"
    return out


def _era(year: int) -> int:
    """Redistricting era: district lines are (roughly) fixed within an era."""
    return 0 if year <= 1980 else 1 + (year - 1982) // 10


def load_long(csv_dir: Path = CSV_DIR, offices=("P", "S", "H")) -> pd.DataFrame:
    """One row per (race, candidate) with summed votes, before the share filter."""
    frames = []
    if "P" in offices:
        p = pd.read_csv(csv_dir / "1976-2020-president.csv")
        frames.append(pd.DataFrame({
            "year": p.year, "office": "P", "state": p.state_po, "district": "", "special": False,
            "name": _norm_name(p.candidate), "party": _party(p.party_detailed), "votes": p.candidatevotes}))
    if "S" in offices:
        s = pd.read_csv(csv_dir / "1976-2020-senate.csv")
        s = s[s.stage == "gen"]
        frames.append(pd.DataFrame({
            "year": s.year, "office": "S", "state": s.state_po, "district": "", "special": s.special.astype(bool),
            "name": _norm_name(s.candidate), "party": _party(s.party_detailed), "votes": s.candidatevotes}))
    if "H" in offices:
        h = pd.read_csv(csv_dir / "1976-2020-house.csv", low_memory=False)
        h = h[(h.stage == "GEN") & (h.runoff.astype(str) != "True") & (~h.special.astype(bool))]
        frames.append(pd.DataFrame({
            "year": h.year, "office": "H", "state": h.state_po, "district": h.district.astype(str).str.zfill(3),
            "special": False, "name": _norm_name(h.candidate), "party": _party(h.party), "votes": h.candidatevotes}))
    df = pd.concat(frames, ignore_index=True)
    df = df[df.year.isin(YEARS) & ~df.name.str.match(JUNK) & (df.votes > 0)]
    df["race"] = (df.year.astype(str) + "-" + df.office + "-" + df.state + "-" + df.district
                  + np.where(df.special, "-sp", ""))
    # fusion tickets: one candidate on several party lines -> sum, keep the major-party label
    pri = df.party.map({"D": 0, "R": 1, "O": 2})
    df = df.assign(_pri=pri).sort_values("_pri")
    df = (df.groupby(["race", "year", "office", "state", "district", "name"], as_index=False)
            .agg(votes=("votes", "sum"), party=("party", "first")))
    return df


def build_races(csv_dir: Path = CSV_DIR, offices=("P", "S", "H"), min_share: float = MIN_SHARE) -> pd.DataFrame:
    """Apply the share filter, derive incumbency, return one row per (race, kept candidate)."""
    df = load_long(csv_dir, offices)
    tot = df.groupby("race").votes.transform("sum")
    df = df[df.votes / tot >= min_share].copy()
    df["share"] = df.votes / df.groupby("race").votes.transform("sum")
    df["nvotes"] = df.groupby("race").votes.transform("sum")
    df = df[df.groupby("race").name.transform("size") >= 2].copy()  # uncontested races carry no share info

    # incumbency: same person won this office in this state recently (6y senate, 2y house)
    df["person"] = np.where(df.office == "P", df.name, df.name.map(_person_key))
    win = df.loc[df.groupby("race").share.idxmax(), ["year", "office", "state", "person"]]
    won = set(zip(win.office, win.state, win.person, win.year))
    lag = {"S": range(1, 7), "H": (2,), "P": ()}
    df["inc"] = [float(any((o, s, n, y - k) in won for k in lag[o]))
                 for o, s, n, y in zip(df.office, df.state, df.person, df.year)]
    # candidate identity: presidential runs are per-year, congressional careers persist within a state
    df["cand"] = np.where(df.office == "P", df.name + "|" + df.year.astype(str), df.person + "|" + df.state)
    df["unit"] = np.where(df.office == "H",
                          df.state + "-" + df.district + "-e" + df.year.map(_era).astype(str), "")
    return df.sort_values(["year", "office", "state", "district", "share"],
                          ascending=[True, True, True, True, False]).reset_index(drop=True)


@dataclass
class RaceData:
    """Padded tensors for R races with up to J candidates each."""
    y: torch.Tensor          # [R, J] observed shares (0 where padded)
    mask: torch.Tensor       # [R, J] bool
    cand: torch.Tensor       # [R, J] candidate id
    party: torch.Tensor      # [R, J] 0=D 1=R 2=O
    inc: torch.Tensor        # [R, J]
    year: torch.Tensor       # [R] index into YEARS
    office: torch.Tensor     # [R] index into OFFICES
    state: torch.Tensor      # [R]
    unit: torch.Tensor       # [R] district-unit id, 0 = statewide (no offset)
    race_ids: list
    states: list
    cands: list
    units: list
    nvotes: np.ndarray

    def __len__(self):
        return self.y.shape[0]

    def subset(self, idx) -> "RaceData":
        idx = torch.as_tensor(idx, dtype=torch.long)
        kw = {k: getattr(self, k)[idx] for k in ["y", "mask", "cand", "party", "inc", "year", "office", "state", "unit"]}
        return RaceData(**kw, race_ids=[self.race_ids[i] for i in idx.tolist()], states=self.states,
                        cands=self.cands, units=self.units, nvotes=self.nvotes[idx.numpy()])

    def two_party(self):
        """(row index, observed D share of the D+R vote) for races with exactly one D and one R."""
        d = (self.party == 0) & self.mask
        r = (self.party == 1) & self.mask
        ok = (d.sum(1) == 1) & (r.sum(1) == 1)
        yd = (self.y * d).sum(1)
        yr = (self.y * r).sum(1)
        return ok, yd / (yd + yr).clamp_min(1e-9)


def to_tensors(df: pd.DataFrame, max_cands: int = 4) -> RaceData:
    states = sorted(df.state.unique())
    cands = sorted(df.cand.unique())
    units = [""] + sorted(u for u in df.unit.unique() if u)
    s_ix = {s: i for i, s in enumerate(states)}
    c_ix = {c: i for i, c in enumerate(cands)}
    u_ix = {u: i for i, u in enumerate(units)}
    groups = list(df.groupby("race", sort=False))
    R, J = len(groups), max_cands
    y = np.zeros((R, J), np.float32); mask = np.zeros((R, J), bool)
    cand = np.zeros((R, J), np.int64); party = np.full((R, J), 2, np.int64); inc = np.zeros((R, J), np.float32)
    year = np.zeros(R, np.int64); office = np.zeros(R, np.int64); state = np.zeros(R, np.int64)
    unit = np.zeros(R, np.int64); nv = np.zeros(R); ids = []
    for r, (rid, g) in enumerate(groups):
        g = g.iloc[:J]
        n = len(g)
        sh = g.share.to_numpy(); y[r, :n] = sh / sh.sum(); mask[r, :n] = True
        cand[r, :n] = [c_ix[c] for c in g.cand]; party[r, :n] = g.party.map({"D": 0, "R": 1, "O": 2})
        inc[r, :n] = g.inc
        g0 = g.iloc[0]
        year[r] = YEARS.index(g0.year); office[r] = OFFICES.index(g0.office)
        state[r] = s_ix[g0.state]; unit[r] = u_ix[g0.unit]; nv[r] = g0.nvotes; ids.append(rid)
    t = torch.from_numpy
    return RaceData(t(y), t(mask), t(cand), t(party), t(inc), t(year), t(office), t(state), t(unit),
                    ids, states, cands, units, nv)
