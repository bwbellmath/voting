# Per-Year Third-Party Suppression and the Adjusted Baseline (2026-11-15)

Follows `docs/2026-11-15_strategic_scenarios.md`. That doc treated the desertion rate as an assumed knob. Here it is measured for each election year, and every election is re-run with that suppression removed.

## The measurement

**Idea.** Closeness of the major-party race is a weak proxy for the pressure on third-party voters. A sharper contrast exists in the data: contests where **only one** major party ran. There, nobody can be a spoiler and the third candidate is the only alternative, so both costs you described are switched off at once: strategic desertion and "not a serious contender".

**Method.**

1. Calibrate every candidate's appeal to the real vote shares with no desertion assumed. The compass accounts for who lives in the constituency and where each candidate sits.
2. For every third-party candidate, take its appeal relative to the strongest major-party candidate in the same contest (7,313 candidates; 766 of them in one-major contests).
3. Regress that on year, party group (Libertarian, Green, other), office, and a per-year effect of "both majors on the ballot", plus the major-party margin.
4. The per-year effect is a log-odds gap. Suppression rate = 1 − exp(−gap). Yearly gaps are shrunk toward the pooled value in proportion to their noise.

**Results.**

| | |
|---|---|
| Third-party share, both majors running | 2.5% |
| Third-party share, one major running | 11.0% |
| Pooled suppression rate | 0.81 |
| Range across years | 0.64 (1986) to 0.89 (2000) |
| 1976–1994 average | about 0.73 |
| 1996–2020 average | about 0.84 |
| Closeness effect once measured this way | −0.04 ± 0.06 (none) |

- Suppression rises around 1996–2000 and stays high, with dips in 2008 and 2018.
- Closeness adds nothing once the comparison is "both majors vs one major". That supports your point that closeness is not the real driver.
- **Residual seriousness gap.** Even with one major absent, a third-party candidate's appeal is 1.5 log-odds below the major's (about 22% of the appeal at the same compass position): Libertarian 1.1, Green 1.4, other 1.7. This part is *not* removed in the simulation. It mixes genuine candidate quality with perceived seriousness, and the data cannot separate them.

Per-year table: `outputs/compass/desertion/suppression_by_year.csv`. Plot: `img/compass/desertion/suppression_by_year.png`.

## The adjusted baseline

- Each contest with both majors gets its year's suppression rate. Candidate appeal is re-solved so that plurality *with that suppression* reproduces the real shares (worst error 0.0006).
- **Cap:** a third candidate cannot sincerely out-poll the real runner-up. This binds in 518 of 4,626 adjusted contests.
- Sincere third-party first-choice share rises from **2.3% (observed) to 8.0%**.
- Agents then vote sincerely under every method: 200 runs of 1,000 agents per contest, as in the original baseline.

## Results

Third-party winners across all 10,774 contests (real: 19):

| Plurality, sincere | Ranked choice | Approval | STAR | Condorcet | Borda | Random ballot |
|---|---|---|---|---|---|---|
| 110 | 124 | 127 | 130 | 130 | 134 | 867 |

Third-party seats per election, averaged over 1976–2020:

| Method | House: real | House: adjusted | Senate chamber: real | Senate: adjusted |
|---|---|---|---|---|
| Plurality, sincere | 0.5 | 3.8 | 1.0 | 2.3 |
| Ranked choice | 0.5 | 4.4 | 1.0 | 2.4 |
| Approval | 0.5 | 4.5 | 1.0 | 2.4 |
| STAR / Condorcet | 0.5 | 4.6 | 1.0 | 2.5 |
| Borda | 0.5 | 4.8 | 1.0 | 2.5 |
| Random ballot | 9.6 (unadjusted) | 33.7 | 3.0 (unadjusted) | 10.9 |

House 2020 (real: 222 D, 212 R, 0 third-party):

| Method | Democrat | Republican | Other | Libertarian |
|---|---|---|---|---|
| Plurality, sincere | 224.4 | 204.3 | 3.6 | 1.7 |
| Ranked choice | 221.9 | 203.1 | 3.7 | 5.2 |
| Approval | 222.0 | 201.6 | 4.0 | 6.4 |
| STAR | 221.7 | 201.6 | 4.0 | 6.8 |
| Condorcet | 221.6 | 201.6 | 4.0 | 6.9 |
| Borda | 222.2 | 200.2 | 4.1 | 7.5 |
| Random ballot | 210.7 | 185.2 | 16.1 | 22.0 |

- **Third parties now win under every method**, about nine times as many House seats as in reality (4–5 a year instead of 0.5). The peaks are 2010 (15–19 seats), 2012 and 2000.
- **Sincerity still matters more than the method.** Sincere plurality alone gives 110 winners; the other methods add 14–24. Ranked choice is the lowest of the alternatives and Borda the highest.
- **Random ballot** goes from about 10 third-party House seats a year to about 34, and from 3 Senate seats to 11.
- **Republicans lose slightly more than Democrats** (House: about −2.5 vs −1.5 seats a year). I have not traced why.
- Run-to-run spread is still narrow for the deterministic methods (about ±1.3 House seats) and ±6 for random ballot.

## How far to trust it

- **This is an upper estimate.** In a one-major contest, some of the third candidate's vote is opposition partisans with no candidate of their own. The compass credits part of that to position, but the parties sit close together in the fitted compass, so the rest lands in "appeal" and inflates the measured gap.
- **The winners are third candidates who already did well.** With 81–89% suppression, a real 7% becomes about 46% sincere. Example: Illinois 13th district 2012, real result R 46.5 / D 46.2 / independent 7.2, becomes 28 / 26 / 46. So the model's concrete claim is: third candidates above roughly 5% in a two-major race would be the sincere plurality favourite. That is strong, and it is checkable against first-round counts wherever ranked choice has been adopted.
- **One rate per year is applied to every third candidate.** A fringe candidate and a well-known independent are probably suppressed differently. That is the seriousness axis again, and it suggests the next refinement: let suppression depend on a seriousness measure (prior office, same person's earlier results, spending if added).
- **Selection:** third candidates who choose to run in one-major contests may differ from those who run against both.
- **Lower bound:** measured by closeness alone, suppression is about zero, which gives back the original baseline. The truth is somewhere between that and this.

## Files

```bash
.venv/bin/python scripts/run_desertion.py --iters 200 --voters 1000     # ~5 min
.venv/bin/python scripts/plot_methods.py --npz outputs/compass/desertion/seats.npz \
      --out img/compass/desertion --label "suppression-adjusted compass agents"
```

| Path | Contents |
|---|---|
| `img/compass/desertion/<method>/{house,senate}-hist-<year>.png` | Histogram series under the adjusted baseline (random ballot in `random_ballot/`) |
| `img/compass/desertion/compare/` | All methods on one page per chamber-year |
| `img/compass/desertion/suppression_by_year.png` | The measured rate by year |
| `outputs/compass/desertion/seats_summary.csv` | Seats per method, chamber, year, party, with the unadjusted baseline alongside |
| `outputs/compass/desertion/changed_winners.csv` | Contests that flip, with real and sincere shares |
| `compass/strategy.py` (`measure_suppression`) | The measurement |
