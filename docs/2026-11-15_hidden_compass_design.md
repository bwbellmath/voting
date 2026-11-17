# Hidden Political Compass: Design and First Results (2026-11-15)

Follows `docs/2026-11-14_hidden_compass.md`. This doc describes the model that is now implemented in `compass/`, how to run it, and how well it reproduces the real 1976–2020 elections in `csv/`.

## 1. What the model is

**One sentence:** every state is a slowly changing mix of a few fixed national voter types that live in a low-dimensional space; candidates sit in the same space; each voter picks a nearby candidate; we fit everything so the predicted vote shares match the real ones.

### Voters

- There are `K` national **voter types**. Type `k` is a Gaussian cloud `N(mu_k, diag(sig_k²))` in `R^d`. Defaults: `d = 2`, `K = 4`.
- The types are **fixed over time**. This is the assumption that anchors the space: populations change slowly, parties and candidates move.
- State `s` in year `t` is a mixture of the types with weights `pi_{s,t} = softmax(A_s + B_s·tau)`, where `tau` is rescaled time. So each state has a starting mix and a straight-line drift.
- A House district adds its own offset `G_u` to its state's logits. Districts are keyed by redistricting era (1976–80, 82–90, 92–2000, 2002–10, 2012–20).

### Candidates

Each candidate has a position `c_j` in the same space and a scalar **valence** `v_j` (appeal that has nothing to do with position).

| Candidate kind | Position | Why |
|---|---|---|
| Major-party House / Senate | party position that year, plus a learned fraction `lam` of the gap between the constituency and the nation | One race gives one number; an individual position is not identifiable. |
| Presidential | party position that year plus an individual offset | Each one faces 51 different electorates, so the offset is identifiable. |
| Third-party / independent | own free position, shrunk to the origin | Perot, Sanders, Murkowski write-in, etc. |

- Party positions follow a **random walk** over years, with a tight step (`s_walk = 0.03`).
- Valence = long-run Democratic advantage for that office + that year's national swing ("wave", shrunk to 0) + incumbency bonus + a per-candidate term (shrunk to 0, shared across a person's races in a state).

### Voting and likelihood

- A voter at `z` picks candidate `j` with probability `softmax_j(−|z − c_j|² + v_j)`. This is "vote for the nearest candidate" with noise, and it handles any number of candidates.
- Predicted share = mix-weighted average of that probability over each type's cloud (64–128 quasi-Monte Carlo points per type).
- Observed shares ~ `Dirichlet(phi · predicted)`. `phi` is fixed per office, not learned (see §4, finding 1).
- Fit: MAP (maximum a posteriori) by full-batch Adam. About 3 minutes for all 10,528 races on an M1 CPU.

### What the data can and cannot pin down

- The space has no natural orientation. Rotating or reflecting everything changes nothing. For plots, axis 1 is defined as the average Democrat→Republican direction.
- A national shift in voters is indistinguishable from a national wave in one year. The model assigns slow movement to positions and state mixes, and fast movement to the wave.
- No turnout model: shares are among votes cast. No demographics: the repo has none below national level.

## 2. Data used

`compass/data.py` builds one table from `csv/1976-2020-{president,senate,house}.csv`:

- General elections only; fusion-ticket lines summed per candidate; junk rows (write-in totals, blank, scattering) dropped.
- Candidates under 5% are dropped and shares renormalised. Races with fewer than two remaining candidates (uncontested) are dropped.
- Incumbency is derived: the same person won that office in that state 2 years earlier (House) or within 6 years (Senate). Names are matched on last name + first initial.
- Result: **10,528 races** — 611 presidential state-years, 775 Senate, 9,142 House.

The Gallup party-ID files are not used yet (national only, 2004–2024).

## 3. How to run

```bash
/usr/bin/python3 -m venv .venv && .venv/bin/pip install -r requirements-compass.txt   # needs an arm64 Python
.venv/bin/python -m pytest tests -q                                   # 8 tests, ~10 s
.venv/bin/python scripts/run_compass.py --offices PSH --split random  # fit + held-out scores vs baselines
.venv/bin/python scripts/run_compass.py --offices PSH --split temporal
.venv/bin/python scripts/plot_compass.py outputs/compass/PSH_random_d2K4_s0
.venv/bin/python scripts/identifiability.py --offices PSH --seeds 3
```

| File | Role |
|---|---|
| `compass/data.py` | CSVs → race table → padded tensors |
| `compass/model.py` | The model, likelihood and priors |
| `compass/fit.py` | MAP fitting |
| `compass/evaluate.py` | Splits, error metrics, interval calibration |
| `compass/baseline.py` | Non-spatial ridge-regression baselines |
| `compass/synth.py` | Simulate elections from a known compass; Procrustes comparison |

## 4. Results on the real elections

All errors are RMSE of the Democratic share of the two-party vote, in percentage points, on races the model never saw. "Trend baseline" is a ridge regression with no compass: state lean + per-state time trend + year×office swing + incumbency + district effect. It is the bar the compass has to clear.

### Test A: random 20% of races held out (3 different splits, `d=2, K=4`)

| Office | Compass | Trend baseline | Verdict |
|---|---|---|---|
| President (state-level), n≈122 | **3.6** (3.3 / 3.6 / 4.1) | 4.6 | Compass better |
| Senate, n≈150 | 7.9 (8.2 / 7.8 / 7.6) | 7.8 | Tie |
| House, n≈1,700 | **6.7** (6.6 / 6.7 / 7.0) | 7.8 | Compass better |

Winner called correctly: president 85–92%, Senate 81–85%, House 93%.

Caveat on the presidential row: a held-out state still has that candidate's other ~40 states in training, so this is the easy version of the test.

### Test B: train on 1976–2016, forecast 2018 and 2020

Nothing from 2018 or 2020 is seen: no candidates, no national swing.

| Office | Compass | Trend baseline | Verdict |
|---|---|---|---|
| President 2020, n=51 | 5.7 (94% of states called) | **4.0** (92%) | Baseline better |
| Senate, n=64 | **5.6** | 7.9 | Compass better |
| House, n=794 | **7.3** | 7.7 | Compass slightly better |

### Calibration

Predictive intervals were sized on half of the held-out races and checked on the other half. Nominal 90% intervals cover 84–96% of results. Typical half-widths: ±6 points (president), ±13 (Senate), ±11 (House). So the intervals are honest, but wide for Congress.

### How many dimensions does the data support?

Random split, seed 0:

| Model | President | Senate | House |
|---|---|---|---|
| `d=1, K=3` | 3.6 | 8.5 | 6.6 |
| `d=2, K=4` | 3.3 | 8.2 | 6.6 |
| `d=3, K=6` | 4.5 | 8.1 | 6.7 |

One dimension gets nearly all of the predictive value. The second dimension helps a little; a third hurts.

### Is the map real? (identifiability)

| Check | Party axis | Full 2-D map |
|---|---|---|
| Real data, 3 random restarts, agreement between fits | correlation 0.88 | R² 0.64–0.69 |
| Simulated elections from a known map, refit | correlation 0.94 | R² 0.65 |

On simulated data the fit's held-out error is 4.2–5.2 points against a noise floor of about 2.6–2.8, so it does not fully recover the true world even when the model is exactly right.

**Reading:** a state's position along the Democrat–Republican axis is a stable finding. The second axis is only partly determined by this data and should not be interpreted yet.

### Face validity

- Most Democratic states on axis 1 in 1976: DC, GA, WV, AR, TN, MD (Carter's South). In 2020: DC, MA, CA, HI, NY, IL. The model found the Southern realignment unprompted.
- Most Republican in 2020: KY, WY, ID, TN, AL, OK.
- The two party positions move apart from 1976 to 2020 (polarization).
- Fitted House national swings have the right sign in the known wave years: 2006 (+), 2008 (+), 2010 (−), 2014 (−), 2018 (+).
- Constituency adaptation `lam` is about 0.04: almost none of the variation between a party's candidates is explained by candidates moving toward their district.

Plots: `outputs/compass/PSH_random_d2K4_s0.png` and `outputs/compass/PSH_temporal_d2K4_s0.png`.

### Three things that went wrong and what fixed them

1. **Learned noise level collapsed.** With one race per congressional candidate, the per-candidate valence absorbed each result exactly and the learned Dirichlet `phi` ran to infinity. Fix: `phi` is fixed per office; predictive spread is calibrated on held-out races instead. In-sample error (about 1 point) is therefore meaningless for Congress; only held-out numbers count.
2. **First forecast was off by 14 points.** The yearly wave and the yearly party positions can trade off exactly, so an unseen year inherited a mismatched pair. Fix: wave = long-run mean + shrunk yearly deviation, and a much tighter random walk on party positions.
3. **Remaining 4–7 point Republican bias in forecasts.** Quadratic state trends over-extrapolated. Fix: linear trends only.

## 5. Bottom line

- The model reproduces real elections about as well as, and for House and held-out presidential states better than, a strong non-spatial regression. It is not dramatically better.
- What it adds over the regression is structure: one shared space that explains president, Senate and House together, handles three-way races, and gives a state map that moves sensibly over 44 years.
- What is being estimated is a **MAP point estimate**, not a posterior. The intervals above are predictive intervals for vote shares, not uncertainty over the compass itself.

## 6. Open questions and next steps

- **Second axis.** It needs information that vote totals don't carry. Candidates: issue-level survey data by state (ANES/CES), roll-call scores (DW-NOMINATE) to pin candidate positions, or ballot-measure results.
- **True posterior.** Options: stochastic variational inference over all parameters, or a bootstrap ensemble with Procrustes alignment. The restart check above is a crude version of the second.
- **Presidential forecasting** loses to the baseline because an unseen candidate is treated as a generic party member. A prior from the previous nominee, or from primary results, may close the gap.
- **Uncontested House races are dropped**, which removes the safest districts and biases district offsets toward competitive seats.
- **No turnout or abstention**, and no demographics.
- **Hyperparameters** (`s_u`, `s_walk`, `s_wave`, `phi`) were set by hand, and `s_walk` and the linear-trend choice were picked after looking at the 2018–2020 forecast. Test B is therefore slightly optimistic; a clean check would forecast a different held-out period.
- **Gallup party ID** is unused. A cheap external check: does the model's national mix track Gallup's leaner series for 2004–2020?
- **Old flow code** (`transformer_flow.py`, `train_posterior.py`) is superseded by `compass/` and can be removed.
