/* Election Explorer: map + results / demographics / simulations for US elections 1976-2020.
   Data: data/app.json (scripts/build_webui.py), data/geo/*.json (scripts/build_webui_geo.py). */
(() => {
"use strict";
const CLASS = { D: 0, R: 1, L: 2, G: 2, X: 2 };
const CLS_NAME = ["Democrat", "Republican", "Third party"];
const CLS_VAR = ["--dem", "--rep", "--third"];
const GROUP_NAME = { D: "Democrat", R: "Republican", L: "Libertarian", G: "Green", X: "Other" };
const OFFICE_NAME = { P: "President", S: "Senate", H: "House" };
const COLORS = [
  ["winner", "Winner"], ["margin", "Two-party margin"], ["third", "Third-party vote share"],
  ["lean", "Compass lean"], ["sim", "Simulated win chance"], ["flip", "Chance the result changes"]];
const DASH = ["6 4", "2 3", "10 3 2 3", "1 4", "12 4", "6 2 1 2 1 2", "3 6"];
const W = 975, H = 610, HEX_R = 10.5;

const S = { office: "H", geo: "state", year: 2020, color: "winner", method: "ranked_choice", base: "sincere",
            sel: [], tab: "sim", plot: "heat", lineMethods: null, showSims: true, ink: "auto", combine: true,
            sitting: false, actual: true, theme: "auto" };
let D, IDX = { P: {}, S: {}, H: {} }, HD = {}, STATE = {}, MIDX = {}, statesTopo, stateFeat, stateByFips = {};
const geoCache = {}, hexCache = {}, seriesCache = new Map();
const $ = (q) => document.querySelector(q);
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const fmt = d3.format(","), pct = (x, d = 1) => (100 * x).toFixed(d) + "%";
const title = (s) => s.toLowerCase().replace(/(^|[\s\-'(])([a-z])/g, (m, a, b) => a + b.toUpperCase());

/* ---------- state <-> URL ---------- */
function readHash() {
  const q = new URLSearchParams(location.hash.slice(1));
  for (const [k, v] of q) {
    if (k === "sel") S.sel = v ? v.split(",") : [];
    else if (k === "lm") S.lineMethods = v ? v.split(",") : [];
    else if (k === "year") S.year = +v;
    else if (["combine", "sitting", "showSims", "actual"].includes(k)) S[k] = v === "1";
    else if (k in S) S[k] = v;
  }
}
function writeHash() {
  const q = new URLSearchParams();
  for (const k of ["office", "geo", "year", "color", "method", "base", "tab", "plot", "ink", "theme"]) q.set(k, S[k]);
  if (S.sel.length) q.set("sel", S.sel.join(","));
  if (S.lineMethods) q.set("lm", S.lineMethods.join(","));
  for (const k of ["combine", "sitting", "showSims", "actual"]) q.set(k, S[k] ? "1" : "0");
  history.replaceState(null, "", "#" + q.toString());
}

/* ---------- data access ---------- */
function indexData() {
  D.methods.forEach((m, i) => MIDX[m.key] = i);
  for (const s of D.states) STATE[s.po] = s, stateByFips[s.fips] = s.po;
  for (const c of D.contests) {
    c.cls = c.c.map((x) => CLASS[x[1]]);
    c.actual = [0, 0, 0]; c.actual[c.cls[0]] = 1;
    c.votes = [0, 0, 0]; c.c.forEach((x, j) => c.votes[c.cls[j]] += x[2]);
    ((IDX[c.o][c.y] ||= {})[c.st] ||= []).push(c);
    if (c.o === "H") (HD[c.y] ||= {})[c.st + "-" + c.d] = c;
  }
}
const officeYears = (o) => o === "P" ? D.years.filter((y) => y % 4 === 0) : D.years;
function snapYear() { if (S.office === "P" && S.year % 4) S.year -= 2; }

function senateSeats(st, year, sitting) {
  if (!sitting) return (IDX.S[year] || {})[st] || [];
  const best = {};
  for (let y = year - 4; y <= year; y += 2) for (const c of ((IDX.S[y] || {})[st] || []))
    if (!best[c.k] || c.y > best[c.k].y || (c.y === best[c.k].y && c.sp)) best[c.k] = c;
  return Object.values(best);
}
function contestsFor(el, office, year) {
  const st = el.slice(0, 2), isD = el.length > 2;
  if (office === "H") { if (isD) { const c = (HD[year] || {})[el]; return c ? [c] : []; } return (IDX.H[year] || {})[st] || []; }
  if (office === "S") return senateSeats(st, year, S.sitting);
  return (IDX.P[year] || {})[st] || [];
}
function selectionContests(sel, office, year) {
  if (!sel.length) {
    if (office === "S") return D.states.flatMap((s) => senateSeats(s.po, year, S.sitting));
    return Object.values(IDX[office][year] || {}).flat();
  }
  const seen = new Set(), out = [];
  for (const el of sel) for (const c of contestsFor(el, office, year)) if (!seen.has(c.id)) seen.add(c.id), out.push(c);
  return out;
}
/* probability that each party class wins a contest under a method */
function simProbs(c, method, base) {
  if (c.w) {
    const w = c.w[base][MIDX[method]], p = [0, 0, 0];
    w.forEach((n, j) => p[c.cls[j]] += n / D.n_iter);
    return p;
  }
  if (c.o === "P" && method === "random_ballot") return c.votes.map((v) => v / c.tv);
  return c.actual;
}
function slotProbs(c, method, base) {
  if (c.w) return c.w[base][MIDX[method]].map((n) => n / D.n_iter);
  if (c.o === "P" && method === "random_ballot") return c.c.map((x) => x[2] / c.tv);
  return c.c.map((x, j) => +(j === 0));
}
const weight = (c) => c.o === "P" ? c.ev : 1;
function pmf(contests, probs, k) {
  const total = d3.sum(contests, weight), a = new Float64Array(total + 1); a[0] = 1; let hi = 0;
  contests.forEach((c, n) => {
    const p = probs[n][k], w = weight(c);
    if (p <= 0) return;
    for (let i = hi; i >= 0; i--) { const v = a[i]; if (v === 0) continue; a[i] = v * (1 - p); a[i + w] += v * p; }
    hi += w;
  });
  return a;
}
/* per-year seat summary for a selection: actual seats, simulated mean and full distribution */
function series(sel, office, method, base) {
  const key = [sel.join(","), office, method, base, S.sitting].join("|");
  if (seriesCache.has(key)) return seriesCache.get(key);
  const out = officeYears(office).map((year) => {
    const cs = selectionContests(sel, office, year), probs = cs.map((c) => simProbs(c, method, base));
    const total = d3.sum(cs, weight), actual = [0, 0, 0], mean = [0, 0, 0], votes = [0, 0, 0];
    cs.forEach((c, n) => { const w = weight(c); for (let k = 0; k < 3; k++) actual[k] += w * c.actual[k], mean[k] += w * probs[n][k], votes[k] += c.votes[k]; });
    return { year, n: cs.length, total, actual, mean, votes, dist: [0, 1, 2].map((k) => pmf(cs, probs, k)) };
  });
  if (seriesCache.size > 200) seriesCache.clear();
  seriesCache.set(key, out);
  return out;
}
const elName = (el) => el.length > 2 ? (el.endsWith("-0") ? el.slice(0, 2) + " at-large" : el) : STATE[el].name;
const selLabel = (sel) => !sel.length ? "Whole country" : sel.length <= 3 ? sel.map(elName).join(", ") : sel.length + " places";

/* ---------- colour ---------- */
const lin = (c) => { c /= 255; return c <= .04045 ? c / 12.92 : Math.pow((c + .055) / 1.055, 2.4); };
const unlin = (c) => Math.round(255 * (c <= .0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - .055));
let CLS_RGB, CLS_HEX;
function loadTheme() {
  CLS_HEX = CLS_VAR.map(css);
  CLS_RGB = CLS_HEX.map((h) => { const c = d3.rgb(h); return [lin(c.r), lin(c.g), lin(c.b)]; });
}
function mixColor(p) {
  const s = p[0] + p[1] + p[2]; if (!(s > 0)) return css("--nodata");
  const o = [0, 0, 0];
  for (let k = 0; k < 3; k++) for (let i = 0; i < 3; i++) o[i] += p[k] / s * CLS_RGB[k][i];
  return `rgb(${unlin(o[0])},${unlin(o[1])},${unlin(o[2])})`;
}
const neutral = () => css("--surface-2");
const diverge = (t) => { // t in [-1, 1]: Democrat ... neutral ... Republican
  const a = Math.min(1, Math.abs(t)); return d3.interpolateRgb(neutral(), CLS_HEX[t < 0 ? 0 : 1])(0.12 + 0.88 * Math.pow(a, 0.8));
};
const seqThird = (t) => d3.interpolateRgb(neutral(), d3.color(CLS_HEX[2]).darker(0.9))(Math.pow(Math.min(1, Math.max(0, t)), 0.6));
const seqFlip = (t) => d3.interpolateRgb(neutral(), css("--ink"))(Math.pow(Math.min(1, Math.max(0, t)), 0.6) * 0.9);

const NAT_LEAN = [];
function leanOf(piRow, t) {      // main-axis position relative to the whole country that year
  const tl = D.compass.type_lean; return d3.sum(piRow, (p, k) => p * tl[k]) - (NAT_LEAN[t] || 0);
}
let LEAN_SCALE = 1;
function elLean(el, year) {
  const t = D.years.indexOf(year);
  if (el.length > 2) { const c = (HD[year] || {})[el]; if (c) return leanOf(c.pi, t); }
  const row = D.compass.state_pi[el.slice(0, 2)]; return row ? leanOf(row[t], t) : null;
}
function elementColor(el) {
  const y = S.year;
  if (S.color === "lean") { const v = elLean(el, y); return v == null ? css("--nodata") : diverge(v / LEAN_SCALE); }
  const cs = contestsFor(el, S.office, y);
  if (!cs.length) return css("--nodata");
  if (S.color === "winner") return mixColor(cs.reduce((a, c) => a.map((v, k) => v + weight(c) * c.actual[k]), [0, 0, 0]));
  if (S.color === "sim") return mixColor(cs.reduce((a, c) => { const p = simProbs(c, S.method, S.base); return a.map((v, k) => v + weight(c) * p[k]); }, [0, 0, 0]));
  const v = cs.reduce((a, c) => a.map((x, k) => x + c.votes[k]), [0, 0, 0]);
  if (S.color === "margin") {
    if (cs.length === 1 && cs[0].cls[0] === 2) return CLS_HEX[2];
    return v[0] + v[1] ? diverge((v[1] - v[0]) / (v[0] + v[1]) / 0.5) : css("--nodata");
  }
  if (S.color === "third") return seqThird(v[2] / (v[0] + v[1] + v[2]) / 0.3);
  const f = d3.mean(cs, (c) => { const p = simProbs(c, S.method, S.base); return 1 - p[c.cls[0]]; });
  return seqFlip(f);
}
function renderLegend() {
  const sw = (k) => `<span class="sw"><i style="background:${CLS_HEX[k]}"></i>${CLS_NAME[k]}</span>`;
  const ramp = (f, a, b) => `<div><div class="ramp" style="background:linear-gradient(90deg,${d3.range(0, 1.01, .1).map(f).join(",")})"></div><div class="ends"><span>${a}</span><span>${b}</span></div></div>`;
  const m = D.methods[MIDX[S.method]].label, o = OFFICE_NAME[S.office];
  const html = {
    winner: `<div class="ttl">${o} winner, ${S.year}</div>${[0, 1, 2].map(sw).join("")}`,
    sim: `<div class="ttl">Win chance: ${m}</div>${[0, 1, 2].map(sw).join("")}<div class="muted">Colours blend by probability; purple is a Democrat–Republican toss-up.</div>`,
    margin: `<div class="ttl">${o} two-party margin, ${S.year}</div>${ramp((t) => diverge(2 * t - 1), "D +50", "R +50")}<span class="sw"><i style="background:${CLS_HEX[2]}"></i>third-party winner</span>`,
    third: `<div class="ttl">${o} third-party vote share, ${S.year}</div>${ramp((t) => seqThird(t), "0%", "30%+")}`,
    lean: `<div class="ttl">Compass lean of the electorate, ${S.year}</div>${ramp((t) => diverge(2 * t - 1), "more Democratic than the country", "more Republican")}`,
    flip: `<div class="ttl">Chance of a different winning party: ${m}</div>${ramp((t) => seqFlip(t), "0%", "100%")}`,
  }[S.color];
  $("#map-legend").innerHTML = html + `<div><span class="sw"><i style="background:${css("--nodata")}"></i>no contest / no data</span></div>`;
}

/* ---------- map ---------- */
const projection = d3.geoAlbersUsa().scale(1280).translate([W / 2, H / 2]);
const path = d3.geoPath(projection);
let gZoom, zoom, mapToken = 0;
const congressOf = (year) => Math.min(116, (year - 1786) / 2);
async function loadDistricts(year) {
  const n = congressOf(year);
  if (!geoCache[n]) geoCache[n] = d3.json(`data/geo/cd_${n}.json`).then((topo) => {
    const numbered = new Set(topo.objects.cd.geometries.filter((g) => !g.id.endsWith("-0")).map((g) => g.id.slice(0, 2)));
    /* a few source layers carry a stale at-large shape for states that already had numbered districts */
    const feats = topojson.feature(topo, topo.objects.cd).features.filter((f) => path.centroid(f).every(isFinite) && !(f.id.endsWith("-0") && numbered.has(f.id.slice(0, 2))));
    feats.sort((a, b) => b.properties.area - a.properties.area);
    return { topo, feats };
  });
  return geoCache[n];
}
function hexPath(cx, cy, r) {
  let s = "";
  for (let i = 0; i < 6; i++) { const a = Math.PI / 3 * i + Math.PI / 6; s += (i ? "L" : "M") + (cx + r * Math.cos(a)).toFixed(2) + "," + (cy + r * Math.sin(a)).toFixed(2); }
  return s + "Z";
}
/* one hexagon per House seat: each state is a compact blob of hexagons near its place on the map */
function hexLayout(n, feats) {
  if (hexCache[n]) return hexCache[n];
  const r = HEX_R, w = Math.sqrt(3) * r, nodes = feats.map((f) => { const [x, y] = path.centroid(f); return { id: f.id, st: f.id.slice(0, 2), x0: x, y0: y }; });
  const states = d3.groups(nodes, (d) => d.st).map(([st, v]) => ({ st, v, n: v.length, x0: d3.mean(v, (d) => d.x0), y0: d3.mean(v, (d) => d.y0) }));
  states.forEach((s) => { s.x = s.x0; s.y = s.y0; });
  const sim = d3.forceSimulation(states).force("x", d3.forceX((d) => d.x0).strength(0.3)).force("y", d3.forceY((d) => d.y0).strength(0.3))
    .force("c", d3.forceCollide((d) => 0.93 * r * Math.sqrt(d.n) + r * 0.75).iterations(4)).stop();
  for (let i = 0; i < 600; i++) sim.tick();
  const used = new Set(), key = (c, rw) => c + "," + rw, cell = (c, rw) => [(c + (rw & 1) / 2) * w, rw * 1.5 * r];
  const nbrs = (c, rw) => (rw & 1 ? [[1, 0], [1, -1], [0, -1], [-1, 0], [0, 1], [1, 1]] : [[1, 0], [0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 1]]).map(([dc, dr]) => [c + dc, rw + dr]);
  for (const s of states.slice().sort((a, b) => b.n - a.n)) {
    const r0 = Math.round(s.y / (1.5 * r)), c0 = Math.round(s.x / w - (r0 & 1) / 2);
    let seed = null, bd = Infinity;
    for (let dr = -9; dr <= 9; dr++) for (let dc = -9; dc <= 9; dc++) {
      if (used.has(key(c0 + dc, r0 + dr))) continue;
      const [x, y] = cell(c0 + dc, r0 + dr), d = Math.hypot(x - s.x, y - s.y); if (d < bd) bd = d, seed = [c0 + dc, r0 + dr];
    }
    const blob = [seed], front = new Map(); used.add(key(...seed));
    const push = (c, rw) => { for (const [a, b] of nbrs(c, rw)) if (!used.has(key(a, b)) && !front.has(key(a, b))) { const [x, y] = cell(a, b); front.set(key(a, b), [a, b, Math.hypot(x - s.x, y - s.y)]); } };
    push(...seed);
    while (blob.length < s.n) {
      const best = d3.least(front.values(), (d) => d[2]); front.delete(key(best[0], best[1]));
      used.add(key(best[0], best[1])); blob.push([best[0], best[1]]); push(best[0], best[1]);
    }
    /* seat districts inside the blob so that neighbours on the map stay neighbours */
    const cells = blob.map(([c, rw]) => cell(c, rw)), cx = d3.mean(cells, (d) => d[0]), cy = d3.mean(cells, (d) => d[1]);
    const sp = (a, f) => Math.sqrt(d3.mean(a, (d) => f(d) ** 2)) || 1;
    const kx = sp(cells, (d) => d[0] - cx) / sp(s.v, (d) => d.x0 - s.x0), ky = sp(cells, (d) => d[1] - cy) / sp(s.v, (d) => d.y0 - s.y0);
    const pairs = [];
    s.v.forEach((d, i) => cells.forEach((c, j) => pairs.push([Math.hypot((d.x0 - s.x0) * kx - (c[0] - cx), (d.y0 - s.y0) * ky - (c[1] - cy)), i, j])));
    pairs.sort((a, b) => a[0] - b[0]);
    const di = new Set(), cj = new Set();
    for (const [, i, j] of pairs) if (!di.has(i) && !cj.has(j)) { di.add(i); cj.add(j); s.v[i].hx = cells[j][0]; s.v[i].hy = cells[j][1]; }
  }
  /* centre the cartogram in the view */
  const ex = d3.extent(nodes, (d) => d.hx), ey = d3.extent(nodes, (d) => d.hy), ox = W / 2 - (ex[0] + ex[1]) / 2, oy = H / 2 - (ey[0] + ey[1]) / 2;
  nodes.forEach((d) => { d.hx += ox; d.hy += oy; });
  /* state outlines: hexagon edges whose neighbour is another state or empty */
  const edges = new Map();
  for (const d of nodes) for (let i = 0; i < 6; i++) {
    const a0 = Math.PI / 3 * i + Math.PI / 6, a1 = a0 + Math.PI / 3;
    const p0 = [d.hx + r * Math.cos(a0), d.hy + r * Math.sin(a0)], p1 = [d.hx + r * Math.cos(a1), d.hy + r * Math.sin(a1)];
    const k = Math.round((p0[0] + p1[0]) * 5) + "," + Math.round((p0[1] + p1[1]) * 5);
    (edges.get(k) || edges.set(k, { p0, p1, st: [] }).get(k)).st.push(d.id.slice(0, 2));
  }
  let border = "";
  for (const e of edges.values()) if (e.st.length === 1 || e.st[0] !== e.st[1]) border += `M${e.p0[0].toFixed(2)},${e.p0[1].toFixed(2)}L${e.p1[0].toFixed(2)},${e.p1[1].toFixed(2)}`;
  const labels = d3.rollups(nodes, (v) => { const cx = d3.mean(v, (d) => d.hx), cy = d3.mean(v, (d) => d.hy); const m = d3.least(v, (d) => Math.hypot(d.hx - cx, d.hy - cy)); return [m.hx, m.hy]; }, (d) => d.id.slice(0, 2));
  return hexCache[n] = { nodes, border, labels };
}
function initMap() {
  const svg = d3.select("#map");
  gZoom = svg.append("g");
  gZoom.append("g").attr("class", "fills"); gZoom.append("path").attr("class", "borders");
  gZoom.append("g").attr("class", "sel"); gZoom.append("g").attr("class", "labs");
  zoom = d3.zoom().scaleExtent([1, 14]).translateExtent([[-80, -60], [W + 80, H + 60]]).on("zoom", (e) => {
    gZoom.attr("transform", e.transform);
    gZoom.selectAll(".lab").attr("font-size", 8 / Math.sqrt(e.transform.k)).style("stroke-width", 2.2 / Math.sqrt(e.transform.k));
  });
  svg.call(zoom).on("dblclick.zoom", null);
  $("#z-in").onclick = () => svg.transition().duration(200).call(zoom.scaleBy, 1.6);
  $("#z-out").onclick = () => svg.transition().duration(200).call(zoom.scaleBy, 1 / 1.6);
  $("#z-reset").onclick = () => svg.transition().duration(250).call(zoom.transform, d3.zoomIdentity);
}
let mapItems = [];   // [{id, d (path string)}] currently drawn
async function drawMap() {
  const token = ++mapToken, g = gZoom, status = $("#map-status");
  let items, border, labels = [];
  if (S.geo === "state") {
    items = stateFeat.map((f) => ({ id: f.po, d: path(f) }));
    border = path(topojson.mesh(statesTopo, statesTopo.objects.states));
  } else {
    status.hidden = false; status.textContent = "Loading district boundaries…";
    let geo;
    try { geo = await loadDistricts(S.year); } catch (e) { status.textContent = "District boundaries are missing: run scripts/build_webui_geo.py"; return; }
    if (token !== mapToken) return;
    status.hidden = true;
    if (S.geo === "district") {
      items = geo.feats.map((f) => ({ id: f.id, d: path(f) }));
      border = path(topojson.mesh(statesTopo, statesTopo.objects.states));
    } else {
      const hx = hexLayout(congressOf(S.year), geo.feats);
      items = hx.nodes.map((d) => ({ id: d.id, d: hexPath(d.hx, d.hy, HEX_R) }));
      border = hx.border; labels = hx.labels;
    }
  }
  mapItems = items;
  g.select(".fills").selectAll("path").data(items, (d) => d.id).join("path").attr("class", "el").attr("d", (d) => d.d)
    .on("click", (e, d) => toggleSel(d.id)).on("mousemove", (e, d) => showTip(e, elementTip(d.id))).on("mouseleave", hideTip);
  g.select(".borders").attr("d", border);
  g.select(".labs").selectAll("text").data(labels, (d) => d[0]).join("text").attr("class", "lab").attr("x", (d) => d[1][0]).attr("y", (d) => d[1][1]).text((d) => d[0]);
  const k = d3.zoomTransform($("#map")).k;
  g.selectAll(".lab").attr("font-size", 8 / Math.sqrt(k));
  paintMap();
}
function paintMap() {
  gZoom.select(".fills").selectAll("path").attr("fill", (d) => elementColor(d.id));
  const isSel = (id) => S.sel.includes(id) || S.sel.includes(id.slice(0, 2));
  const sel = mapItems.filter((d) => isSel(d.id));
  const layer = gZoom.select(".sel");
  layer.selectAll("path.halo").data(sel, (d) => d.id).join("path").attr("class", "selected halo").attr("d", (d) => d.d);
  layer.selectAll("path.ring").data(sel, (d) => d.id).join("path").attr("class", "selected ring").attr("d", (d) => d.d);
  layer.selectAll("path.ring").raise();
  renderLegend();
  const notes = [];
  if (S.geo !== "state" && S.office !== "H") notes.push(`${OFFICE_NAME[S.office]} results exist only by state, so every district shows its state's result.`);
  if (S.geo === "state" && S.office === "H" && ["winner", "sim"].includes(S.color)) notes.push("Each state blends the colours of its House seats.");
  if (S.office === "P" && ["sim", "flip"].includes(S.color) && S.method !== "random_ballot") notes.push("Presidential contests were not simulated: only the random ballot differs from the real result.");
  if (S.office === "S" && S.sitting && S.year < 1980) notes.push("The sitting chamber is incomplete before 1980: seats last elected before 1976 are not in the data.");
  if (S.geo !== "state" && S.year === 2020) notes.push("2020 uses the 116th Congress district lines (North Carolina's 2020 redraw is not shown).");
  $("#map-note").textContent = notes.join(" ");
}
function elementTip(el) {
  const cs = contestsFor(el, S.office, S.year);
  let h = `<b>${elName(el)}</b> · ${OFFICE_NAME[S.office]} ${S.year}`;
  if (el.length > 2 && S.office !== "H") h += `<div class="muted">showing ${STATE[el.slice(0, 2)].name} (state-level office)</div>`;
  if (!cs.length) return h + `<div class="muted">No contest this year${S.office === "H" ? " (or missing from the returns)" : ""}.</div>`;
  const m = D.methods[MIDX[S.method]].label;
  if (cs.length === 1) {
    const c = cs[0], sp = slotProbs(c, S.method, S.base);
    h += c.c.map((x, j) => `<div class="tr"><span><i class="pdot" style="background:${CLS_HEX[c.cls[j]]}"></i>${title(x[0])} (${x[1]})</span><span>${pct(x[2] / c.tv)} · ${pct(sp[j], 0)}</span></div>`).join("");
    h += `<div class="muted">vote share · win chance under ${m}</div>`;
  } else {
    const a = [0, 0, 0], p = [0, 0, 0];
    cs.forEach((c) => { const q = simProbs(c, S.method, S.base); for (let k = 0; k < 3; k++) a[k] += weight(c) * c.actual[k], p[k] += weight(c) * q[k]; });
    h += [0, 1, 2].map((k) => `<div class="tr"><span><i class="pdot" style="background:${CLS_HEX[k]}"></i>${CLS_NAME[k]}</span><span>${a[k]} · ${p[k].toFixed(1)}</span></div>`).join("");
    h += `<div class="muted">seats won · expected under ${m}</div>`;
  }
  if (S.color === "lean") { const v = elLean(el, S.year); if (v != null) h += `<div class="muted">compass lean ${v > 0 ? "+" : ""}${v.toFixed(2)} (positive = more Republican than the country)</div>`; }
  return h;
}
function showTip(e, html) {
  const t = $("#tip"); t.innerHTML = html; t.hidden = false;
  const r = t.getBoundingClientRect(); let x = e.clientX + 14, y = e.clientY + 14;
  if (x + r.width > innerWidth - 8) x = e.clientX - r.width - 14;
  if (y + r.height > innerHeight - 8) y = e.clientY - r.height - 14;
  t.style.left = Math.max(4, x) + "px"; t.style.top = Math.max(4, y) + "px";
}
function hideTip() { $("#tip").hidden = true; }
function toggleSel(id) {
  const i = S.sel.indexOf(id);
  if (i >= 0) S.sel.splice(i, 1); else S.sel.push(id);
  update({ map: "paint" });
}

/* ---------- controls ---------- */
function radioGroup(sel, name, opts, cls) {
  d3.select(sel).selectAll("label").data(opts).join("label").html((d) => `<input type="radio" name="${name}" value="${d[0]}"> ${d[1]}`)
    .select("input").on("change", (e, d) => { S[name] = d[0]; if (name === "office") snapYear(); update({ map: name === "geo" ? "draw" : "paint" }); });
}
function initControls() {
  radioGroup("#ctl-office", "office", [["P", "President"], ["S", "Senate"], ["H", "House"]]);
  radioGroup("#ctl-geo", "geo", [["state", "States"], ["district", "Districts"], ["hex", "House-seat hexes"]]);
  radioGroup("#ctl-color", "color", COLORS);
  d3.select("#ctl-method").selectAll("option").data(D.methods).join("option").attr("value", (d) => d.key).text((d) => d.label);
  $("#ctl-method").onchange = (e) => { S.method = e.target.value; update({ map: "paint" }); };
  $("#ctl-base").onchange = (e) => { S.base = e.target.value; update({ map: "paint" }); };
  $("#ctl-sitting").onchange = (e) => { S.sitting = e.target.checked; update({ map: "paint" }); };
  $("#yr-slider").oninput = (e) => setYear(D.years[+e.target.value]);
  $("#yr-prev").onclick = () => stepYear(-1); $("#yr-next").onclick = () => stepYear(1);
  $("#sel-clear").onclick = () => { S.sel = []; update({ map: "paint" }); };
  $("#theme-btn").onclick = () => { S.theme = { auto: "light", light: "dark", dark: "auto" }[S.theme] || "auto"; applyTheme(); update({ map: "paint" }); };
  document.querySelectorAll(".tabs button").forEach((b) => b.onclick = () => { S.tab = b.dataset.tab; update({}); });
}
function stepYear(d) { const ys = officeYears(S.office), i = ys.indexOf(S.year) + d; if (i >= 0 && i < ys.length) setYear(ys[i]); }
function setYear(y) {
  const old = S.year; S.year = y; snapYear();
  update({ map: S.geo !== "state" && congressOf(old) !== congressOf(S.year) ? "draw" : "paint" });
}
function syncControls() {
  for (const k of ["office", "geo", "color"]) document.querySelectorAll(`input[name=${k}]`).forEach((i) => i.checked = i.value === S[k]);
  $("#ctl-method").value = S.method; $("#ctl-base").value = S.base; $("#ctl-sitting").checked = S.sitting;
  $("#ctl-sitting-wrap").style.display = S.office === "S" ? "" : "none";
  $("#yr-slider").value = D.years.indexOf(S.year); $("#yr-out").textContent = S.year;
  document.querySelectorAll(".tabs button").forEach((b) => b.setAttribute("aria-selected", b.dataset.tab === S.tab));
  const chips = d3.select("#chips");
  if (!S.sel.length) chips.html(`<span class="chip all">Whole country</span>`);
  else chips.html("").selectAll("span").data(S.sel).join("span").attr("class", "chip").html((d) => `${elName(d)}<button aria-label="Remove ${elName(d)}">×</button>`)
    .select("button").on("click", (e, d) => toggleSel(d));
  $("#sel-clear").hidden = !S.sel.length;
}
function update({ map }) {
  S.sel = S.sel.filter((el) => STATE[el.slice(0, 2)]);
  syncControls(); writeHash();
  if (map === "draw") drawMap(); else if (map === "paint") paintMap();
  renderPanel();
}

/* ---------- chart helpers ---------- */
function frame(host, { h = 300, m = { t: 12, r: 16, b: 28, l: 44 } } = {}) {
  const w = Math.max(320, host.clientWidth || 600);
  const wrap = d3.select(host).append("div").attr("class", "chart").style("height", h + "px");
  const svg = wrap.append("svg").attr("viewBox", `0 0 ${w} ${h}`).attr("width", w).attr("height", h);
  return { wrap, svg, w, h, m, iw: w - m.l - m.r, ih: h - m.t - m.b };
}
function yearAxis(f, x, years) {
  const step = f.iw / years.length < 34 ? 2 : 1, g = f.svg.append("g").attr("class", "axis");
  g.selectAll("text").data(years.filter((y, i) => i % step === 0)).join("text").attr("class", "axis-t").attr("x", (d) => x(d)).attr("y", f.h - f.m.b + 16).attr("text-anchor", "middle").text((d) => d);
}
function yAxis(f, y, label, tickFmt) {
  const g = f.svg.append("g").attr("class", "axis");
  const ticks = y.ticks(5).filter((t) => tickFmt || Number.isInteger(t));
  g.selectAll("line").data(ticks).join("line").attr("class", "gridl").attr("x1", f.m.l).attr("x2", f.w - f.m.r).attr("y1", (d) => y(d)).attr("y2", (d) => y(d));
  g.selectAll("text").data(ticks).join("text").attr("class", "axis-t").attr("x", f.m.l - 6).attr("y", (d) => y(d)).attr("dy", "0.32em").attr("text-anchor", "end").text(tickFmt || ((d) => d));
  if (label) f.svg.append("text").attr("class", "axis-t").attr("x", 4).attr("y", 9).text(label);
}
function yearRule(f, x) {
  if (x(S.year) == null) return;
  f.svg.append("line").attr("class", "yr-rule").attr("x1", x(S.year)).attr("x2", x(S.year)).attr("y1", f.m.t).attr("y2", f.h - f.m.b);
}
/* overlay that reports the nearest year on hover and sets the map year on click */
function yearHover(f, x, years, tipFn) {
  const cross = f.svg.append("line").attr("class", "cross").attr("y1", f.m.t).attr("y2", f.h - f.m.b).style("display", "none");
  const near = (e) => { const [mx] = d3.pointer(e, f.svg.node()); return d3.least(years, (y) => Math.abs(x(y) - mx)); };
  f.svg.append("rect").attr("x", f.m.l).attr("y", f.m.t).attr("width", f.iw).attr("height", f.ih).attr("fill", "transparent").style("cursor", "pointer")
    .on("mousemove", (e) => { const y = near(e); cross.style("display", null).attr("x1", x(y)).attr("x2", x(y)); showTip(e, tipFn(y, e)); })
    .on("mouseleave", () => { cross.style("display", "none"); hideTip(); })
    .on("click", (e) => setYear(near(e)));
}
const tipRow = (color, name, val) => `<div class="tr"><span>${color ? `<i class="pdot" style="background:${color}"></i>` : ""}${name}</span><span>${val}</span></div>`;
const seatWord = () => S.office === "P" ? "electoral votes" : "seats";

/* ---------- Simulations tab ---------- */
const isDark = () => S.theme === "auto" ? matchMedia("(prefers-color-scheme: dark)").matches : S.theme === "dark";
function inkMode() { return S.ink !== "auto" ? S.ink : (isDark() ? "rgb" : "cmy"); }
function applyTheme() {
  if (S.theme === "auto") delete document.documentElement.dataset.theme; else document.documentElement.dataset.theme = S.theme;
  $("#theme-btn").textContent = "Theme: " + S.theme; loadTheme();
}
const INK = { rgb: [[0.10, 0.38, 1.0], [1.0, 0.14, 0.10], [0.50, 1.0, 0.0]], cmy: [[0.05, 0.50, 1.0], [1.0, 0.16, 0.30], [0.86, 0.92, 0.0]] };
function inkColor(a, mode) {      // a: three intensities in [0, 1]
  const C = INK[mode], o = mode === "rgb" ? [0, 0, 0] : [1, 1, 1];
  for (let k = 0; k < 3; k++) for (let i = 0; i < 3; i++)
    if (mode === "rgb") o[i] += a[k] * C[k][i]; else o[i] *= 1 - a[k] * (1 - C[k][i]);
  return o.map((v) => Math.round(255 * Math.min(1, v)));
}
function heatmap(host, sel, opts = {}) {
  const ser = series(sel, S.office, S.method, S.base), years = ser.map((d) => d.year), mode = inkMode();
  const f = frame(host, { h: opts.h || 380, m: { t: 14, r: 14, b: 28, l: 44 } });
  let lo = Infinity, hi = 0;
  for (const d of ser) for (let k = 0; k < 3; k++) {
    d.dist[k].forEach((p, i) => { if (p > 2e-4) lo = Math.min(lo, i), hi = Math.max(hi, i); });
    lo = Math.min(lo, d.actual[k]); hi = Math.max(hi, d.actual[k]);
  }
  if (!isFinite(lo)) { lo = 0; hi = 1; }
  const pad = Math.max(1, Math.round((hi - lo) * 0.04)); lo = Math.max(0, lo - pad); hi = hi + pad;
  const rows = hi - lo + 1, band = f.iw / years.length, gap = Math.min(2, band * 0.08);
  const x = (y) => { const i = years.indexOf(y); return i < 0 ? null : f.m.l + (i + 0.5) * band; };
  const y = d3.scaleLinear().domain([lo - 0.5, hi + 0.5]).range([f.h - f.m.b, f.m.t]);
  const dpr = Math.min(2, devicePixelRatio || 1);
  const cv = f.wrap.insert("canvas", "svg").attr("width", Math.round(f.iw * dpr)).attr("height", Math.round(f.ih * dpr))
    .style("left", f.m.l + "px").style("top", f.m.t + "px").style("width", f.iw + "px").style("height", f.ih + "px").node();
  const ctx = cv.getContext("2d"); ctx.imageSmoothingEnabled = false;
  ctx.fillStyle = mode === "rgb" ? "#000" : "#fff"; ctx.fillRect(0, 0, cv.width, cv.height);
  const col = document.createElement("canvas"); col.width = 1; col.height = rows; const cctx = col.getContext("2d");
  ser.forEach((d, i) => {
    const img = cctx.createImageData(1, rows), mx = d.dist.map((a) => d3.max(a) || 1);
    for (let r = 0; r < rows; r++) {
      const s = hi - r, a = [0, 1, 2].map((k) => { const p = d.dist[k][s] || 0; return p > 0 ? Math.pow(p / Math.sqrt(mx[k]), 0.55) : 0; });
      const c = inkColor(a, mode); img.data.set([c[0], c[1], c[2], 255], r * 4);
    }
    cctx.putImageData(img, 0, 0);
    ctx.drawImage(col, 0, 0, 1, rows, Math.round((i * band + gap / 2) * dpr), 0, Math.round((band - gap) * dpr), cv.height);
  });
  f.svg.style("position", "relative");
  yAxis(f, y, opts.ylabel === false ? null : `${seatWord()} won`);
  f.svg.selectAll(".gridl").style("opacity", 0.25);
  yearAxis(f, x, years);
  if (S.actual) for (let k = 0; k < 3; k++)
    f.svg.append("g").selectAll("circle").data(ser.filter((d) => d.n)).join("circle").attr("cx", (d) => x(d.year)).attr("cy", (d) => y(d.actual[k]))
      .attr("r", 3.2).attr("fill", "none").attr("stroke", mode === "rgb" ? "#fff" : "#000").attr("stroke-width", 1.4);
  const xr = x(S.year);
  if (xr != null) f.svg.append("rect").attr("x", xr - band / 2 + gap / 4).attr("y", f.m.t - 3).attr("width", band - gap / 2).attr("height", f.ih + 6).attr("fill", "none").attr("stroke", css("--ink")).attr("stroke-width", 1.5).attr("rx", 2);
  yearHover(f, x, years, (yr, e) => {
    const d = ser[years.indexOf(yr)], s = Math.max(lo, Math.min(hi, Math.round(y.invert(d3.pointer(e, f.svg.node())[1]))));
    return `<b>${yr}</b> · ${d.total} ${seatWord()} at stake` +
      [0, 1, 2].map((k) => { const sd = Math.sqrt(d3.sum(d.dist[k], (p, i) => p * (i - d.mean[k]) ** 2)); return tipRow(CLS_HEX[k], CLS_NAME[k], `actual ${d.actual[k]} · sim ${d.mean[k].toFixed(1)} ± ${sd.toFixed(1)}`); }).join("") +
      `<div class="muted" style="margin-top:4px">Chance of exactly ${s}:</div>` + [0, 1, 2].map((k) => tipRow(CLS_HEX[k], CLS_NAME[k], pct(d.dist[k][s] || 0))).join("");
  });
}
function heatLegend(host) {
  const mode = inkMode(), c = (a) => `rgb(${inkColor(a, mode).join(",")})`, sw = (a, t) => `<span class="sw"><i style="background:${c(a)};outline:1px solid var(--line)"></i>${t}</span>`;
  d3.select(host).append("div").attr("class", "cap").html(
    sw([1, 0, 0], "Democrat") + sw([0, 1, 0], "Republican") + sw([0, 0, 1], "Third party") + sw([1, 1, 0], "D + R overlap") + sw([1, 0, 1], "D + third") + sw([0, 1, 1], "R + third") +
    (S.actual ? `<span class="sw"><i style="background:none;border:1.5px solid var(--ink);border-radius:50%"></i>actual result</span>` : "") +
    `<br>Each column is one election. Brightness${mode === "cmy" ? " (ink density)" : ""} shows how likely that party is to win exactly that many ${seatWord()}; a sharp, certain outcome is stronger than a spread-out one.`);
}
function lineChart(host, sel, opts = {}) {
  const act = series(sel, S.office, "plurality", S.base), years = act.map((d) => d.year);
  const methods = S.showSims ? (S.lineMethods || [S.method]) : [];
  const sims = methods.map((m) => ({ m, ser: series(sel, S.office, m, S.base) }));
  const f = frame(host, { h: opts.h || 360, m: { t: 16, r: 86, b: 28, l: 44 } });
  const x = d3.scalePoint().domain(years).range([f.m.l + 10, f.w - f.m.r - 4]);
  const top = d3.max([act, ...sims.map((s) => s.ser)].flat(), (d) => Math.max(...d.actual, ...d.mean)) || 1;
  const y = d3.scaleLinear().domain([0, top]).nice().range([f.h - f.m.b, f.m.t]);
  yAxis(f, y, `${seatWord()} won`); yearAxis(f, x, years); yearRule(f, x);
  const line = (acc) => d3.line().x((d) => x(d.year)).y(acc);
  sims.forEach((s) => { for (let k = 0; k < 3; k++)
    f.svg.append("path").attr("d", line((d) => y(d.mean[k]))(s.ser)).attr("fill", "none").attr("stroke", CLS_HEX[k]).attr("stroke-width", 1.6).attr("stroke-dasharray", DASH[MIDX[s.m]]).attr("opacity", 0.95); });
  const ends = [];
  for (let k = 0; k < 3; k++) {
    f.svg.append("path").attr("d", line((d) => y(d.actual[k]))(act)).attr("fill", "none").attr("stroke", CLS_HEX[k]).attr("stroke-width", 2);
    f.svg.append("g").selectAll("circle").data(act).join("circle").attr("cx", (d) => x(d.year)).attr("cy", (d) => y(d.actual[k])).attr("r", 2.6).attr("fill", CLS_HEX[k]).attr("stroke", css("--surface")).attr("stroke-width", 1);
    ends.push({ k, y: y(act[act.length - 1].actual[k]) });
  }
  ends.sort((a, b) => a.y - b.y); for (let i = 1; i < ends.length; i++) ends[i].y = Math.max(ends[i].y, ends[i - 1].y + 13);
  f.svg.append("g").selectAll("text").data(ends).join("text").attr("class", "dlab").attr("x", f.w - f.m.r + 4).attr("y", (d) => d.y).attr("dy", "0.32em").text((d) => CLS_NAME[d.k]);
  yearHover(f, x, years, (yr) => {
    const i = years.indexOf(yr), d = act[i];
    return `<b>${yr}</b> · ${d.total} ${seatWord()} at stake` + [0, 1, 2].map((k) => tipRow(CLS_HEX[k], CLS_NAME[k], "actual " + d.actual[k])).join("") +
      sims.map((s) => `<div class="muted" style="margin-top:4px">${D.methods[MIDX[s.m]].label}</div>` + [0, 1, 2].map((k) => tipRow(CLS_HEX[k], CLS_NAME[k], s.ser[i].mean[k].toFixed(1))).join("")).join("");
  });
}
function lineLegend(host) {
  const methods = S.showSims ? (S.lineMethods || [S.method]) : [];
  d3.select(host).append("div").attr("class", "cap").html(
    [0, 1, 2].map((k) => `<span class="sw"><i style="background:${CLS_HEX[k]}"></i>${CLS_NAME[k]}</span>`).join("") +
    `<span class="sw"><i class="ln"></i>actual result</span>` +
    methods.map((m) => `<span class="sw"><svg width="30" height="8"><line x1="0" x2="30" y1="4" y2="4" stroke="${css("--ink-2")}" stroke-width="1.6" stroke-dasharray="${DASH[MIDX[m]]}"/></svg>${D.methods[MIDX[m]].label} (average)</span>`).join(""));
}
function simTable(host, sel) {
  const rows = D.methods.map((m) => ({ m, d: series(sel, S.office, m.key, S.base).find((d) => d.year === S.year) })).filter((r) => r.d);
  if (!rows.length) return;
  const a = rows[0].d, sd = (d, k) => Math.sqrt(d3.sum(d.dist[k], (p, i) => p * (i - d.mean[k]) ** 2));
  const cell = (d, k) => `<td class="n">${d.mean[k].toFixed(1)} <span class="muted">± ${sd(d, k).toFixed(1)}</span></td>`;
  d3.select(host).append("div").attr("class", "card").html(
    `<h3>${S.year}: expected ${seatWord()} by method · ${selLabel(sel)}</h3><div class="cap">${a.total} ${seatWord()} at stake. Average ± one standard deviation across simulated elections.</div>
     <div class="tblwrap"><table><thead><tr><th>Method</th>${[0, 1, 2].map((k) => `<th class="n"><i class="pdot" style="background:${CLS_HEX[k]}"></i>${CLS_NAME[k]}</th>`).join("")}</tr></thead><tbody>
     <tr><td><b>Actual result</b></td>${[0, 1, 2].map((k) => `<td class="n"><b>${a.actual[k]}</b></td>`).join("")}</tr>
     ${rows.map((r) => `<tr><td>${r.m.label}</td>${[0, 1, 2].map((k) => cell(r.d, k)).join("")}</tr>`).join("")}</tbody></table></div>`);
}
function renderSim(panel) {
  const multi = S.sel.length > 1, row = d3.select(panel).append("div").attr("class", "row");
  row.append("div").attr("class", "seg").attr("role", "radiogroup").selectAll("label").data([["heat", "Histogram heatmap"], ["line", "Line: seats over time"]]).join("label")
    .html((d) => `<input type="radio" name="plot" value="${d[0]}" ${S.plot === d[0] ? "checked" : ""}> ${d[1]}`).select("input").on("change", (e, d) => { S.plot = d[0]; update({}); });
  if (multi) row.append("div").attr("class", "seg").selectAll("label").data([[true, "Combined"], [false, "One plot per place"]]).join("label")
    .html((d) => `<input type="radio" name="combine" ${S.combine === d[0] ? "checked" : ""}> ${d[1]}`).select("input").on("change", (e, d) => { S.combine = d[0]; update({}); });
  const chk = (label, key) => row.append("label").html(`<input type="checkbox" ${S[key] ? "checked" : ""}> ${label}`).select("input").on("change", (e) => { S[key] = e.target.checked; update({}); });
  if (S.plot === "heat") {
    chk("Mark actual results", "actual");
    row.append("label").html(`Colour mixing <select><option value="auto">Auto (by theme)</option><option value="rgb">RGB light on black</option><option value="cmy">CMY ink on white</option></select>`)
      .select("select").property("value", S.ink).on("change", (e) => { S.ink = e.target.value; update({}); });
  } else {
    chk("Include simulations", "showSims");
    if (S.showSims) {
      const cur = S.lineMethods || [S.method];
      row.append("div").attr("class", "checks").selectAll("label").data(D.methods).join("label")
        .html((d) => `<input type="checkbox" ${cur.includes(d.key) ? "checked" : ""}> ${d.label.replace(/ \(.*/, "")}`).select("input")
        .on("change", (e, d) => { const s = new Set(S.lineMethods || [S.method]); e.target.checked ? s.add(d.key) : s.delete(d.key); S.lineMethods = D.methods.map((m) => m.key).filter((k) => s.has(k)); update({}); });
    }
  }
  const mLabel = D.methods[MIDX[S.method]].label, bLabel = S.base === "sincere" ? "voters as they actually voted" : "third-party suppression removed";
  const draw = S.plot === "heat" ? heatmap : lineChart;
  const sub = S.plot === "heat" ? `${mLabel}, ${bLabel}` : `actual results${S.showSims ? " and simulation averages" : ""}, ${bLabel}`;
  if (multi && !S.combine) {
    const shown = S.sel.slice(0, 12), grid = d3.select(panel).append("div").attr("class", "grid3");
    (S.plot === "heat" ? heatLegend : lineLegend)(grid.node().parentNode.insertBefore(document.createElement("div"), grid.node()));
    for (const el of shown) { const card = grid.append("div").attr("class", "card"); card.append("h3").text(elName(el)); draw(card.node(), [el], { h: 220 }); }
    if (S.sel.length > 12) d3.select(panel).append("p").attr("class", "callout").text(`Showing the first 12 of ${S.sel.length} selected places. Switch to Combined to see them all together.`);
  } else {
    const card = d3.select(panel).append("div").attr("class", "card");
    card.append("h3").text(`${OFFICE_NAME[S.office]} ${seatWord()} by party · ${selLabel(S.sel)}`);
    card.append("div").attr("class", "cap").text(sub + ". Click a year to move the map to that election.");
    draw(card.node(), S.sel);
    (S.plot === "heat" ? heatLegend : lineLegend)(card.node());
  }
  if (S.office === "P") d3.select(panel).append("p").attr("class", "callout").text("Presidential contests were not simulated with compass agents. Only the random ballot (each state's winner drawn with chance equal to vote share) differs from the real result; the other methods repeat it.");
  if (S.base === "adjusted") d3.select(panel).append("p").attr("class", "callout").text(`Suppression removed: third-party candidates get back the support the model estimates they lose when both major parties run (${pct(D.suppression[S.year] || 0, 0)} of it in ${S.year}). This is an upper estimate.`);
  simTable(d3.select(panel).append("div").style("margin-top", "16px").node(), S.sel);
}

/* ---------- Results tab ---------- */
function renderResults(panel) {
  const cs = selectionContests(S.sel, S.office, S.year), ser = series(S.sel, S.office, S.method, S.base);
  const d = ser.find((x) => x.year === S.year), root = d3.select(panel);
  root.append("h3").text(`${OFFICE_NAME[S.office]} ${S.year} · ${selLabel(S.sel)}`);
  if (!d || !cs.length) root.append("p").attr("class", "muted").text("No contest for this selection in this year.");
  else {
    const tv = d3.sum(d.votes), tiles = root.append("div").attr("class", "tiles");
    for (let k = 0; k < 3; k++) tiles.append("div").attr("class", "tile").html(`<div class="k"><i style="background:${CLS_HEX[k]}"></i>${CLS_NAME[k]}</div><div class="v">${d.actual[k]}</div><div class="s">${seatWord()} · ${pct(d.votes[k] / tv)} of votes</div>`);
    tiles.append("div").attr("class", "tile").html(`<div class="k">Contests</div><div class="v">${cs.length}</div><div class="s">${fmt(tv)} votes counted</div>`);
  }
  const grid = root.append("div").attr("class", "grid2");
  const left = grid.append("div").attr("class", "card");
  left.append("h3").text("Vote share by party over time");
  left.append("div").attr("class", "cap").html([0, 1, 2].map((k) => `<span class="sw"><i style="background:${CLS_HEX[k]}"></i>${CLS_NAME[k]}</span>`).join("") + "Share of all votes in the selected contests. Click a year to move the map.");
  shareChart(left.node(), ser);
  const right = grid.append("div").attr("class", "card");
  const mLabel = D.methods[MIDX[S.method]].label;
  right.append("h3").text(`Contests in ${S.year}`);
  right.append("div").attr("class", "cap").text(`Win chance is under ${mLabel}, ${S.base === "sincere" ? "voters as they voted" : "suppression removed"}.`);
  const sorted = cs.slice().sort((a, b) => d3.ascending(a.st, b.st) || d3.ascending(a.d || 0, b.d || 0));
  const name = (c) => c.o === "H" ? elName(c.st + "-" + c.d) : c.o === "S" ? `${c.st} Senate${c.sp ? " (special)" : ""}${c.y !== S.year ? " · " + c.y : ""}` : `${STATE[c.st].name}${c.ev ? " · " + c.ev + " EV" : ""}`;
  const body = sorted.map((c) => { const sp = slotProbs(c, S.method, S.base); return c.c.map((x, j) =>
    `<tr>${j ? "" : `<td rowspan="${c.c.length}">${name(c)}</td>`}<td><i class="pdot" style="background:${CLS_HEX[c.cls[j]]}"></i>${title(x[0])}${j ? "" : " ✓"}</td><td>${GROUP_NAME[x[1]]}</td><td class="n">${fmt(x[2])}</td><td class="n">${pct(x[2] / c.tv)}</td><td class="n">${pct(sp[j], 0)}</td></tr>`).join(""); }).join("");
  right.append("div").attr("class", "tblwrap").html(`<table><thead><tr><th>Contest</th><th>Candidate</th><th>Party</th><th class="n">Votes</th><th class="n">Share</th><th class="n">Win chance</th></tr></thead><tbody>${body}</tbody></table>`);
}
function shareChart(host, ser) {
  const years = ser.map((d) => d.year), f = frame(host, { h: 300, m: { t: 16, r: 16, b: 28, l: 44 } });
  const x = d3.scalePoint().domain(years).range([f.m.l + 10, f.w - f.m.r - 4]), y = d3.scaleLinear().domain([0, 1]).range([f.h - f.m.b, f.m.t]);
  const sh = (d, k) => { const t = d3.sum(d.votes); return t ? d.votes[k] / t : null; };
  yAxis(f, y, "share of votes", (t) => Math.round(100 * t) + "%"); yearAxis(f, x, years); yearRule(f, x);
  for (let k = 0; k < 3; k++) {
    const pts = ser.filter((d) => sh(d, k) != null);
    f.svg.append("path").attr("d", d3.line().x((d) => x(d.year)).y((d) => y(sh(d, k)))(pts)).attr("fill", "none").attr("stroke", CLS_HEX[k]).attr("stroke-width", 2);
    f.svg.append("g").selectAll("circle").data(pts).join("circle").attr("cx", (d) => x(d.year)).attr("cy", (d) => y(sh(d, k))).attr("r", 2.6).attr("fill", CLS_HEX[k]).attr("stroke", css("--surface")).attr("stroke-width", 1);
  }
  yearHover(f, x, years, (yr) => { const d = ser[years.indexOf(yr)]; return `<b>${yr}</b> · ${fmt(d3.sum(d.votes))} votes` + [0, 1, 2].map((k) => tipRow(CLS_HEX[k], CLS_NAME[k], sh(d, k) == null ? "–" : pct(sh(d, k)))).join(""); });
}

/* ---------- Demographics tab ---------- */
function stateWeight(st, year) { return d3.sum((IDX.H[year] || {})[st] || [], (c) => c.tv) || d3.sum((IDX.P[year] || {})[st] || [], (c) => c.tv) || 1; }
function mixFor(sel, year) {    // vote-weighted voter-type mix of a selection
  const t = D.years.indexOf(year), K = D.compass.type_lean.length, acc = new Array(K).fill(0); let wsum = 0;
  const add = (pi, w) => { for (let k = 0; k < K; k++) acc[k] += w * pi[k]; wsum += w; };
  const els = sel.length ? sel : Object.keys(D.compass.state_pi);
  for (const el of els) {
    if (el.length > 2) { const c = (HD[year] || {})[el]; if (c) add(c.pi, c.tv); }
    else if (D.compass.state_pi[el]) add(D.compass.state_pi[el][t], stateWeight(el, year));
  }
  return wsum ? acc.map((v) => v / wsum) : null;
}
function stackChart(host, data, colors, labels, opts = {}) {
  /* data: [{year, v: [shares]}], shares sum to 1 */
  const years = data.map((d) => d.year), f = frame(host, { h: 250, m: { t: 14, r: opts.right || 16, b: 28, l: 44 } });
  const x = d3.scalePoint().domain(opts.years || years).range([f.m.l + 10, f.w - f.m.r - 4]), y = d3.scaleLinear().domain([0, 1]).range([f.h - f.m.b, f.m.t]);
  yAxis(f, y, null, (t) => Math.round(100 * t) + "%"); yearAxis(f, x, opts.years || years); yearRule(f, x);
  const n = colors.length;
  for (let k = 0; k < n; k++) {
    const base = (d) => d3.sum(d.v.slice(0, k)), area = d3.area().x((d) => x(d.year)).y0((d) => y(base(d))).y1((d) => y(base(d) + d.v[k]));
    f.svg.append("path").attr("d", area(data)).attr("fill", colors[k]).attr("stroke", css("--surface")).attr("stroke-width", 2);
  }
  yearHover(f, x, years, (yr) => { const d = data[years.indexOf(yr)]; return `<b>${yr}</b>` + d.v.map((v, k) => tipRow(colors[k], labels[k], pct(v))).join(""); });
}
function renderDemo(panel) {
  const root = d3.select(panel), C = D.compass, K = C.type_lean.length;
  root.append("h3").text(`What the compass model infers about voters · ${selLabel(S.sel)}`);
  root.append("p").attr("class", "muted").style("margin", "2px 0 12px").text("The repository has no census demographics below national level. These charts show the fitted model's picture of the electorate, not survey measurements, except where marked Gallup.");
  const rows = D.years.map((year, t) => ({ year, t, mix: mixFor(S.sel, year) })).filter((d) => d.mix);
  const grid = root.append("div").attr("class", "grid2");
  const pidCol = [d3.color(CLS_HEX[0]).formatHex(), d3.interpolateRgb(CLS_HEX[0], neutral())(0.55), css("--ink-3"), d3.interpolateRgb(CLS_HEX[1], neutral())(0.55), d3.color(CLS_HEX[1]).formatHex()];
  const legend = (cols, labs) => cols.map((c, i) => `<span class="sw"><i style="background:${c}"></i>${labs[i]}</span>`).join("");
  const a = grid.append("div").attr("class", "card");
  a.append("h3").text("Estimated party identification (model)");
  a.append("div").attr("class", "cap").html(legend(pidCol, C.pid_labels));
  stackChart(a.node(), rows.map((d) => ({ year: d.year, v: C.pid_labels.map((_, j) => d3.sum(d.mix, (p, k) => p * C.pid_by_type[d.t][k][j])) })), pidCol, C.pid_labels);
  const b = grid.append("div").attr("class", "card");
  b.append("h3").text("Gallup party identification (national survey)");
  b.append("div").attr("class", "cap").text("Whole country only, 2004–2020: the survey has no state or district breakdown.");
  const gy = D.years.filter((y) => D.gallup[y]);
  stackChart(b.node(), gy.map((y) => { const v = D.gallup[y], s = d3.sum(v); return { year: y, v: v.map((x) => x / s) }; }), pidCol, C.pid_labels, { years: D.years });
  const order = d3.range(K).sort((i, j) => C.type_lean[i] - C.type_lean[j]);
  const lo = d3.min(C.type_lean), hi = d3.max(C.type_lean), mid = (lo + hi) / 2;
  const tCol = order.map((k) => diverge((C.type_lean[k] - mid) / ((hi - lo) / 2 || 1)));
  const tLab = order.map((k, i) => `Type ${i + 1} (axis position ${C.type_lean[k] > 0 ? "+" : ""}${C.type_lean[k].toFixed(2)})`);
  const c = grid.append("div").attr("class", "card");
  c.append("h3").text("Voter-type mix");
  c.append("div").attr("class", "cap").html(legend(tCol, tLab) + "<br>The model's four national voter types, ordered from the Democratic to the Republican end of the main axis.");
  stackChart(c.node(), rows.map((d) => ({ year: d.year, v: order.map((k) => d.mix[k]) })), tCol, tLab);
  const e = grid.append("div").attr("class", "card");
  e.append("h3").text("Position on the main compass axis");
  e.append("div").attr("class", "cap").html(`<span class="sw"><i class="ln" style="border-color:var(--ink)"></i>${selLabel(S.sel)}</span><span class="sw"><i class="ln" style="border-top-style:dashed"></i>whole country</span><span class="sw"><i style="background:${CLS_HEX[0]}"></i>Democratic party</span><span class="sw"><i style="background:${CLS_HEX[1]}"></i>Republican party</span>`);
  leanChart(e.node(), rows);
  root.append("p").attr("class", "callout").text("Only the main (Democrat–Republican) axis is reliably identified by vote totals. State legislative results and census demographics are not in the repository; adding either would allow finer detail here.");
}
function leanChart(host, rows) {
  const C = D.compass, pos = (mix) => d3.sum(mix, (p, k) => p * C.type_lean[k]);
  const nat = D.years.map((year) => ({ year, v: pos(mixFor([], year)) })), mine = rows.map((d) => ({ year: d.year, v: pos(d.mix) }));
  const party = (j) => D.years.map((year, t) => ({ year, v: C.party_lean[t][j] }));
  const all = [...nat, ...mine, ...party(0), ...party(1)].map((d) => d.v);
  const f = frame(host, { h: 250, m: { t: 14, r: 16, b: 28, l: 44 } });
  const x = d3.scalePoint().domain(D.years).range([f.m.l + 10, f.w - f.m.r - 4]), y = d3.scaleLinear().domain(d3.extent(all)).nice().range([f.h - f.m.b, f.m.t]);
  yAxis(f, y, "axis position", (t) => t.toFixed(1)); yearAxis(f, x, D.years); yearRule(f, x);
  const ln = d3.line().x((d) => x(d.year)).y((d) => y(d.v));
  f.svg.append("path").attr("d", ln(party(0))).attr("fill", "none").attr("stroke", CLS_HEX[0]).attr("stroke-width", 2);
  f.svg.append("path").attr("d", ln(party(1))).attr("fill", "none").attr("stroke", CLS_HEX[1]).attr("stroke-width", 2);
  f.svg.append("path").attr("d", ln(nat)).attr("fill", "none").attr("stroke", css("--ink-2")).attr("stroke-width", 1.6).attr("stroke-dasharray", "5 4");
  if (S.sel.length) f.svg.append("path").attr("d", ln(mine)).attr("fill", "none").attr("stroke", css("--ink")).attr("stroke-width", 2);
  yearHover(f, x, D.years, (yr) => { const t = D.years.indexOf(yr), m = mine.find((d) => d.year === yr);
    return `<b>${yr}</b>` + (S.sel.length && m ? tipRow(css("--ink"), selLabel(S.sel), m.v.toFixed(2)) : "") + tipRow(css("--ink-2"), "Whole country", nat[t].v.toFixed(2)) + tipRow(CLS_HEX[0], "Democratic party", C.party_lean[t][0].toFixed(2)) + tipRow(CLS_HEX[1], "Republican party", C.party_lean[t][1].toFixed(2)); });
}
function renderPanel() {
  const panel = $("#panel"); panel.innerHTML = ""; hideTip();
  ({ results: renderResults, demo: renderDemo, sim: renderSim })[S.tab](panel);
}

/* ---------- start ---------- */
async function main() {
  const [app, topo] = await Promise.all([d3.json("data/app.json"), d3.json("data/geo/states.json")]);
  D = app; statesTopo = topo; indexData(); readHash(); applyTheme();
  if (!D.years.includes(S.year)) S.year = 2020; snapYear();
  stateFeat = topojson.feature(topo, topo.objects.states).features.filter((f) => stateByFips[f.id]).map((f) => Object.assign(f, { po: stateByFips[f.id] }));
  projection.fitExtent([[8, 8], [W - 8, H - 8]], { type: "FeatureCollection", features: stateFeat });
  D.years.forEach((y, t) => { const m = mixFor([], y); NAT_LEAN[t] = d3.sum(m, (p, k) => p * D.compass.type_lean[k]); });
  const devs = []; for (const st in D.compass.state_pi) if (st !== "DC") D.compass.state_pi[st].forEach((row, t) => devs.push(Math.abs(leanOf(row, t))));
  LEAN_SCALE = d3.quantile(devs.sort(d3.ascending), 0.97) || 1;
  initControls(); initMap(); syncControls(); await drawMap(); renderPanel(); writeHash();
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { loadTheme(); paintMap(); renderPanel(); });
  let rt; addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(renderPanel, 150); });
  document.body.dataset.ready = "1";
}
main().catch((e) => { $("#panel").textContent = "Could not load data: " + e.message + ". Run scripts/build_webui.py and serve the webui folder over http."; console.error(e); });
})();
