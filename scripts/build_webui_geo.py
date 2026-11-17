#!/usr/bin/env python
"""Download and simplify the congressional-district layers for the web UI (standard library + numpy only).

  .venv/bin/python scripts/build_webui_geo.py            # all Congresses, ~700 MB of temporary downloads
  .venv/bin/python scripts/build_webui_geo.py 95 116     # only these

Writes webui/data/geo/cd_<congress>.json (TopoJSON, one object "cd", feature id "CA-12"; at-large = 0).

Sources
  95th-114th Congress (elections 1976-2014): Lewis, DeVine, Pitcher & Martis, "Digital Boundary Definitions
    of United States Congressional Districts, 1789-2012", https://cdmaps.polisci.ucla.edu
  115th, 116th Congress (elections 2016-2020): US Census Bureau cartographic boundary files (500k)

Method: cut rings into arcs at junctions (vertices where three or more borders meet),
simplify every distinct arc once with Douglas-Peucker, and share it between the districts on both sides.
Shared borders therefore stay shared: no gaps or overlaps between neighbours.
"""
import io, json, struct, sys, urllib.request, zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "webui" / ".geo_cache"
OUT = ROOT / "webui" / "data" / "geo"
GRID = 1e-5           # degrees: vertex matching grid (shared borders in the sources share exact vertices)
TOL = 0.006           # degrees (~600 m): Douglas-Peucker tolerance
QOUT = 0.002          # degrees: coordinate precision of the output
MIN_AREA = 6e-4       # square degrees: drop smaller islands/holes (a feature's largest ring is always kept)

FIPS = {"01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO", "09": "CT", "10": "DE", "11": "DC",
        "12": "FL", "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN", "19": "IA", "20": "KS", "21": "KY",
        "22": "LA", "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS", "29": "MO", "30": "MT",
        "31": "NE", "32": "NV", "33": "NH", "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND", "39": "OH",
        "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT",
        "50": "VT", "51": "VA", "53": "WA", "54": "WV", "55": "WI", "56": "WY"}
NAMES = {"Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA", "Colorado": "CO",
         "Connecticut": "CT", "Delaware": "DE", "District Of Columbia": "DC", "Florida": "FL", "Georgia": "GA",
         "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
         "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA",
         "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO", "Montana": "MT",
         "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM",
         "New York": "NY", "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
         "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD",
         "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
         "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY"}
CENSUS = {115: ("GENZ2016", "cb_2016_us_cd115_500k", "CD115FP"), 116: ("GENZ2018", "cb_2018_us_cd116_500k", "CD116FP")}


def fetch(url: str, dest: Path) -> Path:
    if not dest.exists() or dest.stat().st_size == 0:
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"  downloading {url}", flush=True)
        req = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
        with urllib.request.urlopen(req, timeout=300) as r:
            dest.write_bytes(r.read())
    return dest


def read_dbf(buf: bytes) -> list:
    n, hlen, rlen = struct.unpack("<IHH", buf[4:12])
    fields, pos = [], 32
    while buf[pos] != 0x0D:
        name = buf[pos:pos + 11].split(b"\0")[0].decode()
        fields.append((name, buf[pos + 16])); pos += 32
    rows = []
    for i in range(n):
        rec = buf[hlen + i * rlen: hlen + (i + 1) * rlen]; off, row = 1, {}
        for name, ln in fields:
            row[name] = rec[off:off + ln].decode("latin-1").strip(); off += ln
        rows.append(row)
    return rows


def read_shp(buf: bytes) -> list:
    """List (one per record) of lists of rings; ring = float array [n, 2]. Null shapes give []."""
    out, pos = [], 100
    while pos < len(buf):
        clen = struct.unpack(">i", buf[pos + 4:pos + 8])[0] * 2
        rec = buf[pos + 8: pos + 8 + clen]; pos += 8 + clen
        if struct.unpack("<i", rec[:4])[0] not in (5, 15, 25):
            out.append([]); continue
        nparts, npts = struct.unpack("<ii", rec[36:44])
        parts = np.frombuffer(rec, "<i4", nparts, 44).tolist() + [npts]
        pts = np.frombuffer(rec, "<f8", npts * 2, 44 + 4 * nparts).reshape(-1, 2)
        out.append([pts[a:b] for a, b in zip(parts[:-1], parts[1:])])
    return out


def signed_area(r) -> float:
    x, y = r[:, 0].astype(float), r[:, 1].astype(float)
    return 0.5 * float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))


def snap(ring):
    """Grid-snap a ring, drop repeated vertices and out-and-back spikes; returns int array (closed) or None."""
    q = np.round(ring / GRID).astype(np.int64)
    q = q[np.r_[True, np.any(q[1:] != q[:-1], axis=1)]]
    pts = [tuple(p) for p in q.tolist()]
    if pts[0] == pts[-1]:
        pts.pop()
    stack = []
    for p in pts:
        if len(stack) >= 2 and stack[-2] == p:
            stack.pop()
        elif not stack or stack[-1] != p:
            stack.append(p)
    while len(stack) >= 3 and (stack[0] == stack[-1] or stack[1] == stack[-1] or stack[0] == stack[-2]):
        if stack[0] == stack[-1]:
            stack.pop()
        elif stack[1] == stack[-1]:
            stack.pop(); stack.pop(0)
        else:
            stack.pop(); stack.pop()
    if len(stack) < 3:
        return None
    return np.array(stack + [stack[0]], dtype=np.int64)


def douglas_peucker(pts: np.ndarray, tol: float) -> np.ndarray:
    keep = np.zeros(len(pts), bool); keep[0] = keep[-1] = True
    todo = [(0, len(pts) - 1)]
    while todo:
        a, b = todo.pop()
        if b - a < 2:
            continue
        seg = pts[b] - pts[a]; rel = pts[a + 1:b] - pts[a]; L = float(np.hypot(*seg))
        d = np.hypot(rel[:, 0], rel[:, 1]) if L == 0 else np.abs(seg[0] * rel[:, 1] - seg[1] * rel[:, 0]) / L
        i = int(np.argmax(d))
        if d[i] > tol:
            k = a + 1 + i; keep[k] = True; todo += [(a, k), (k, b)]
    return pts[keep]


def point_in_ring(p, r) -> bool:
    x, y = p; x0, y0 = r[:-1, 0], r[:-1, 1]; x1, y1 = r[1:, 0], r[1:, 1]
    cross = (y0 > y) != (y1 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        xi = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
    return bool(np.sum(cross & (x < xi)) % 2)


def build_topology(features: dict) -> dict:
    """features: id -> list of snapped rings (shapefile winding: outer clockwise). Returns TopoJSON."""
    nbrs = {}
    for rings in features.values():
        for r in rings:
            pts = [tuple(p) for p in r[:-1].tolist()]; n = len(pts)
            for i, p in enumerate(pts):
                nbrs.setdefault(p, set()).update((pts[i - 1], pts[(i + 1) % n]))
    junction = {p for p, s in nbrs.items() if len(s) != 2}
    arcs, arc_ix = [], {}

    def arc_ref(seq):
        key, rev = tuple(seq), tuple(reversed(seq))
        if key in arc_ix:
            return arc_ix[key]
        if rev in arc_ix:
            return ~arc_ix[rev]
        simp = douglas_peucker(np.array(seq, dtype=float), TOL / GRID).astype(np.int64)
        arc_ix[key] = len(arcs); arcs.append(simp)
        return arc_ix[key]

    geoms = []
    for fid, rings in features.items():
        polys, holes = [], []
        for r in rings:
            pts = [tuple(p) for p in r[:-1].tolist()]; n = len(pts)
            cuts = [i for i, p in enumerate(pts) if p in junction]
            if not cuts:                                   # an island: canonical start so twins match
                s = min(range(n), key=lambda i: pts[i]); seq = pts[s:] + pts[:s] + [pts[s]]
                refs = [arc_ref(seq)]
            else:
                refs = []
                for a, b in zip(cuts, cuts[1:] + [cuts[0] + n]):
                    refs.append(arc_ref([pts[i % n] for i in range(a, b + 1)]))
            coords = np.concatenate([(arcs[i] if i >= 0 else arcs[~i][::-1])[:-1] for i in refs])
            if len(np.unique(coords, axis=0)) < 3:
                continue
            closed = np.vstack([coords, coords[:1]]); area = signed_area(closed)
            (polys if area < 0 else holes).append((abs(area), refs, closed))   # clockwise (negative) = outer
        if not polys:
            continue
        polys.sort(key=lambda t: -t[0])
        big = polys[0][0]
        polys = [p for i, p in enumerate(polys) if i == 0 or p[0] * GRID * GRID >= MIN_AREA]
        out = [[p[1]] for p in polys]
        for h in holes:
            if h[0] * GRID * GRID < MIN_AREA:
                continue
            for i in range(len(polys) - 1, -1, -1):        # smallest outer that contains it
                if polys[i][0] > h[0] and point_in_ring(h[2][0], polys[i][2]):
                    out[i].append(h[1]); break
        geoms.append({"type": "MultiPolygon", "id": fid, "arcs": out, "properties": {"area": round(big * GRID * GRID, 4)}})
    used = sorted({(i if i >= 0 else ~i) for g in geoms for poly in g["arcs"] for ring in poly for i in ring})
    new = {old: k for k, old in enumerate(used)}
    for g in geoms:
        g["arcs"] = [[[new[i] if i >= 0 else ~new[~i] for i in ring] for ring in poly] for poly in g["arcs"]]
    enc = []
    for old in used:
        a = np.round(arcs[old] * (GRID / QOUT)).astype(np.int64)
        a = a[np.r_[True, np.any(a[1:] != a[:-1], axis=1)]]
        if len(a) == 1:
            a = np.vstack([a, a])
        d = np.diff(a, axis=0, prepend=np.zeros((1, 2), np.int64)); enc.append(d.tolist())
    return {"type": "Topology", "transform": {"scale": [QOUT, QOUT], "translate": [0, 0]}, "arcs": enc,
            "objects": {"cd": {"type": "GeometryCollection", "geometries": geoms}}}


def load_layer(n: int):
    if n in CENSUS:
        folder, stem, col = CENSUS[n]
        z = zipfile.ZipFile(fetch(f"https://www2.census.gov/geo/tiger/{folder}/shp/{stem}.zip", CACHE / "raw" / f"{stem}.zip"))
        rows = read_dbf(z.read(f"{stem}.dbf")); shapes = read_shp(z.read(f"{stem}.shp"))
        ids = [f"{FIPS[r['STATEFP']]}-{int(r[col])}" if r["STATEFP"] in FIPS and r[col].isdigit() and int(r[col]) < 90
               else None for r in rows]
    else:
        stem = f"districts{n:03d}"
        z = zipfile.ZipFile(fetch(f"https://cdmaps.polisci.ucla.edu/shp/{stem}.zip", CACHE / "raw" / f"{stem}.zip"))
        member = {Path(m).suffix: m for m in z.namelist()}
        rows = read_dbf(z.read(member[".dbf"])); shapes = read_shp(z.read(member[".shp"]))
        ids = []
        for r in rows:
            try:
                d = int(float(r["DISTRICT"]))
            except ValueError:
                d = -1
            st = NAMES.get(r["STATENAME"].title())
            ids.append(f"{st}-{d}" if st and 0 <= d < 90 else None)
    feats = {}
    for fid, rings in zip(ids, shapes):
        if fid is None or fid.startswith("DC"):
            continue
        for r in rings:
            s = snap(r)
            if s is not None:
                feats.setdefault(fid, []).append(s)
    for fid, rings in feats.items():                      # drop tiny islands and holes before building arcs
        area = [abs(signed_area(r)) * GRID * GRID for r in rings]
        feats[fid] = [r for r, x in zip(rings, area) if x >= MIN_AREA or x == max(area)]
    return feats


def main():
    todo = [int(x) for x in sys.argv[1:]] or list(range(95, 117))
    OUT.mkdir(parents=True, exist_ok=True)
    for n in todo:
        feats = load_layer(n)
        topo = build_topology(feats)
        path = OUT / f"cd_{n}.json"
        path.write_text(json.dumps(topo, separators=(",", ":")))
        npts = sum(len(a) for a in topo["arcs"])
        print(f"congress {n}: {len(topo['objects']['cd']['geometries'])} districts, {len(topo['arcs'])} arcs, "
              f"{npts} points, {path.stat().st_size / 1e3:.0f} kB", flush=True)


if __name__ == "__main__":
    main()
