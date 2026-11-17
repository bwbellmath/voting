# Compass Agents Under Other Voting Methods (2026-11-15)

Re-runs every House and Senate general election from 1976–2020 with simulated voters drawn from the fitted compass (`docs/2026-11-15_hidden_compass_design.md`), under seven voting methods, and plots seats by party the same way as the original dice-roll histograms in `img/`.

## How it works

1. **Contests.** All 10,774 House and Senate contests, with every candidate above 0.5% (up to 6). Uncontested seats are kept as automatic wins.
2. **Candidate positions.** Democrats and Republicans sit at their fitted party position for that year. Minor parties get one fitted position per group (Libertarian, Green, other), learned from where those parties do better or worse.
3. **Calibration.** For each contest, each candidate's appeal term is solved so that the agents' expected *first-choice* shares equal the real vote shares. Worst error across all contests: 0.00001.
4. **Agents.** Each run draws 1,000 agents per contest from that constituency's compass distribution. An agent's utility for a candidate is closeness on the compass + the candidate's appeal + a random taste shock.
5. **Ballots.** Sincere, from the same utilities: a full ranking, 0–5 scores, or approvals.
6. **Seats.** 200 runs. House = seats won that year. Senate = the sitting chamber (most recent contest for each state's seat in the last three cycles), as in `senate.py`.

### Methods

| Method | Rule |
|---|---|
| Plurality | Most first choices. Control: should reproduce the real result. |
| Random ballot | One random voter's first choice wins. This is the original dice roll. |
| Ranked choice | Instant runoff: eliminate the last-place candidate until one remains. |
| Approval | Approve anyone better than your poll-weighted expected outcome. |
| STAR | Score 0–5, top two by total, then head-to-head runoff. |
| Borda | One point for every candidate ranked below. |
| Condorcet | Beats every rival head-to-head; minimax if there is a cycle. |

## Results

Average change in seats versus the real result, across all 23 election years:

| Method | House: Democrats | House: Republicans | Senate: Democrats | Senate: Republicans |
|---|---|---|---|---|
| Plurality (control) | −0.1 | +0.1 | 0.0 | 0.0 |
| Ranked choice | −0.1 | +0.1 | +0.2 | −0.2 |
| Approval | −0.1 | +0.1 | +0.2 | −0.2 |
| STAR | 0.0 | 0.0 | +0.3 | −0.3 |
| Condorcet | −0.1 | +0.1 | +0.2 | −0.2 |
| Borda | +0.9 | −0.9 | +0.9 | −0.8 |
| Random ballot | −4.8 | −4.3 | −1.0 | −1.0 |

- **Ranked choice, approval, STAR and Condorcet change almost nothing.** Under sincere voting with the same candidates, they pick the same winner as plurality in nearly every contest, because most contests are effectively two-candidate.
- **Only two contests flip reliably (in more than 90% of runs)** under ranked choice, STAR and Condorcet, and both are well-known spoiler races:
  - New York Senate 1980: D'Amato (R) 45.1, Holtzman (D) 43.8, Javits on a third line 11.1. Holtzman wins.
  - California 1st district 1990: Riggs (R) 43.3, Bosco (D) 41.9, third candidate 14.8. Bosco wins.
- **Borda moves about one seat per chamber to Democrats** on average (Senate 2020: 49–49 instead of 47–51). Borda is sensitive to where minor candidates sit, so this leans on the weakest part of the model.
- **Random ballot** is the only method that gives minor parties seats (about 5 "other" and 4 Libertarian House seats per year), matching the old dice-roll plots.
- **Spread.** The deterministic methods vary by about ±1 seat (standard deviation) across runs; random ballot by ±5.5 in the House. The narrow spread is expected: these methods are not lotteries, so the only randomness is which 1,000 agents get drawn. Fewer agents per contest would widen it.

## Caveats

- **Approval depends heavily on the approval rule.** With the naive rule ("approve anyone above the average candidate on the ballot"), a fringe candidate drags the threshold down, most agents approve both major candidates, and Democrats gain about 4 House seats a year and 6 Senate seats in 2020. The poll-weighted rule used here removes that. Neither is clearly "right". The naive-rule numbers are saved in `outputs/compass/methods/seats_summary_naive_approval.csv`.
- **No strategic behaviour.** Voters are sincere and the candidate field is fixed. In reality other methods would change who runs and how people vote, which is probably their main effect.
- **Minor-party positions are weakly determined**, as is the compass's second axis. Results that depend on them (Borda, any three-way race) are the least reliable.
- **Senate "actual" counts use general-election pluralities**, as the old script did, so runoff states (e.g. Georgia 2020) show the first-round leader.
- **A few House contests are missing** from the source data in 1996 (422) and 2006 (431).

## Files

```bash
.venv/bin/python scripts/run_methods.py --iters 200 --voters 1000   # ~4 min
.venv/bin/python scripts/plot_methods.py                             # ~370 PNGs
```

| Path | Contents |
|---|---|
| `img/compass/<method>/{house,senate}-hist-<year>.png` | The old histogram series, one per method, chamber and year |
| `img/compass/compare/{house,senate}-methods-<year>.png` | All seven methods on one page per chamber-year |
| `img/compass/{house,senate}-methods-over-time.png` | Mean seats by method, 1976–2020 |
| `outputs/compass/methods/seats_summary.csv` | Mean, sd, min, max seats per method, chamber, year, party |
| `outputs/compass/methods/changed_winners.csv` | Contests where a method's usual winner differs from the real one |
| `compass/methods.py`, `compass/simulate.py` | The voting rules and the simulation |
