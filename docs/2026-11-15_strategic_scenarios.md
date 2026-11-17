# Strategic Voting Scenarios on the Compass (2026-11-15)

Builds on `docs/2026-11-15_voting_methods.md`. That baseline assumed real votes are sincere. Here the assumption is relaxed: real plurality votes may be strategic, hiding sincere third-party support. The questions:

1. Would other voting methods elect more third-party candidates, and how can that be tested?
2. Do ranked choice (RCV) and approval differ, and is RCV more exposed to strategy than approval and STAR?
3. Does any deterministic method come close to random ballot, and how does random ballot change if people vote sincerely?

## Setup

- **Desertion rate `rho`.** Under plurality, a fraction `rho` of voters whose favourite is not one of the two frontrunners vote for the frontrunner they prefer. Candidate appeal is re-solved so that this *strategic* plurality reproduces the real vote shares (error under 0.001). A minor candidate's sincere first-choice share is then roughly observed / (1 − `rho`).
- **Feasibility cap.** A third candidate cannot sincerely out-poll the real runner-up, so `rho` is capped per contest where needed (46 contests at 0.2; 1,566 at 0.9).
- **Honest vs pragmatic ballots.** Honest = straight from utilities. Pragmatic = anchored on the two frontrunners: RCV and plurality voters rank their preferred frontrunner first; approval voters approve everyone above the midpoint between the frontrunners; STAR voters give their preferred frontrunner 5 and the other 0.
- **Compass expression.** Minus the mean squared compass distance from voters to the winner. Higher is better.
- **Entrant scenarios.** A hypothetical third candidate is added to 800 real two-party contests from 2012–2020, at the centre (midway between the parties) or on a flank (one party-gap beyond a party), with its sincere first-choice share set from 5% to 40%. Pragmatic here means "back a major party".
- **Stretched compass.** The fitted compass, and a what-if version with voters twice as far apart (policy matters four times as much relative to personal taste). The stretched version is not fitted to anything; it shows what a sharper compass would imply.

## The affiliation data

The two Gallup files combine into five groups: Democrat, lean Democrat, pure independent, lean Republican, Republican (2004–2020 average: 32 / 16 / 10 / 14 / 28%). I treated these as your "far left … far right" categories; they measure party attachment, not ideology.

**Reconciliation with the compass.** Each agent's partisan gap is its utility for a generic Democrat minus a generic Republican. Two thresholds on that gap (0.3 and 1.2), fixed across all years, split agents into the five groups.

| | Democrat | Lean D | Pure ind. | Lean R | Republican |
|---|---|---|---|---|---|
| Gallup, 2004–2020 mean | 31.7 | 15.8 | 10.1 | 13.8 | 28.5 |
| Compass agents | 32.5 | 14.4 | 10.2 | 14.9 | 27.9 |

- Year-by-year error is 3.7 points (RMSE across groups and years).
- The compass's yearly net Democratic lean tracks Gallup's with correlation 0.87 over nine election years. Gallup was not used in fitting the compass, so this is an independent check, though on only nine points.
- Use in the scenarios: 10% (pure independents) and 40% (all independents) are the natural low and high anchors for how much support a third option could plausibly have.

## 1. Would other methods elect more third parties?

### Falsifiable test on real data: the closeness gradient

If people abandon third parties when their vote matters, third-party share should be lower in close races than in blowouts. Contests with both a Democrat and a Republican, year and state effects removed:

| Office | Contests | Third-party share, margin < 5% | Margin > 40% | Slope t-stat | Implied `rho` |
|---|---|---|---|---|---|
| President (by state) | 612 | 5.0% | 3.6% | 0.6 | 0 |
| Senate | 755 | 2.2% | 4.2% | 2.8 | 0.47 |
| House | 8,646 | 1.5% | 1.9% | 2.8 | 0.21 |

**Result:** the signature is absent for president, small for House, and moderate for Senate. The data supports `rho` somewhere in 0–0.5, not 0.75–0.9. Limit of the test: it only detects desertion that responds to closeness. Habitual desertion ("third parties never win anywhere") would not show up.

### What the simulation says

Third-party winners across all 10,774 contests (real count: 19):

| Method | `rho`=0 | 0.2 | 0.5 | 0.75 | 0.9 |
|---|---|---|---|---|---|
| Plurality, strategic (control) | 19 | 19 | 19 | 19 | 19 |
| Plurality, honest | 19 | 19 | 21 | 56 | 471 |
| Ranked choice, honest | 19 | 20 | 23 | 53 | 561 |
| Approval, honest | 16 | 17 | 19 | 48 | 571 |
| STAR, honest | 19 | 20 | 23 | 52 | 572 |
| Condorcet | 19 | 20 | 22 | 52 | 569 |
| Borda | 18 | 19 | 21 | 48 | 596 |
| Approval, pragmatic | 19 | 19 | 20 | 40 | 389 |
| STAR, pragmatic | 19 | 19 | 19 | 39 | 318 |
| Ranked choice, pragmatic | 19 | 19 | 19 | 19 | 19 |
| Random ballot | 245 | 283 | 397 | 682 | 1,300 |

Sincere third-party first-choice share implied: 2.3% (`rho`=0), 3.7% (0.5), 6.3% (0.75), 12.0% (0.9).

- **In the range the data supports (`rho` ≤ 0.5), no method elects meaningfully more third-party candidates:** 19 becomes at most 23.
- **It takes `rho` = 0.9** to get about 570 wins (5% of contests, roughly 21 House seats a year).
- **The method is not what drives it.** At `rho` = 0.9, honest plurality already gives 471; RCV, approval, STAR and Condorcet add about 100 more. What matters is whether voters are sincere.
- **If voters stay pragmatic, the gains shrink or vanish:** RCV with mainstream-first ranking gives 19, the same as today.

**Verdict on the hypothesis:** not supported by this data. It becomes true only if hidden third-party support is four to five times the observed vote, which the closeness test does not show.

### Other ways to test it

- **Direct measurement of `rho`:** ANES or CES surveys record both candidate ratings and the vote. The share of respondents who rate a third-party candidate highest but vote major-party is `rho`. Not in this repo.
- **Natural experiments:** where RCV was adopted (Maine from 2018, Alaska 2022, many cities), third-party first-choice share should rise by a factor of 1/(1 − `rho`) against comparable places. Needs round-by-round counts, which the files here don't have.
- **The model's own prediction to check:** at `rho` = 0.5 it predicts sincere third-party share of about 3.7% versus 2.3% observed. A first-choice share under RCV well above that would point to higher `rho`.

## 2. Ranked choice vs approval, and exposure to strategy

### Where the methods split

Share of contests a **centrist entrant** wins:

| Entrant's first-choice share | Plurality, honest | RCV, honest | Condorcet | STAR, honest | Approval, honest |
|---|---|---|---|---|---|
| Fitted compass, 30% | 0% | 4% | 13% | 13% | 11% |
| Fitted compass, 35% | 13% | 48% | 47% | 48% | 52% |
| Stretched compass, 20% | 0% | 0% | 25% | 26% | 72% |
| Stretched compass, 25% | 0% | 2% | 43% | 44% | 90% |
| Stretched compass, 30% | 0% | 38% | 59% | 61% | 98% |

- **On the fitted compass the methods rarely split.** A centrist needs about 30% first choices before anything but plurality notices, and by 35% all methods agree. Reason: in the fitted model, personal taste is large relative to policy distance, so a candidate with few first choices is also few people's second choice.
- **On the stretched compass RCV and approval diverge sharply.** A centrist with 20% first choices wins 72% of contests under approval, 25% under Condorcet and STAR, and none under RCV or plurality. This is the classic centre squeeze: RCV eliminates the compromise candidate first.
- **Approval over-elects the centrist** relative to head-to-head majorities (72% vs 25%).

Share of contests a **flank entrant** wins at 40% first-choice share:

| | RCV, honest | Condorcet | STAR, honest | Approval, honest |
|---|---|---|---|---|
| Fitted, left flank | 21% | 6% | 6% | 6% |
| Fitted, right flank | 33% | 9% | 10% | 12% |
| Stretched, left flank | 19% | 0% | 0% | 0% |
| Stretched, right flank | 38% | 0% | 0% | 2% |

RCV elects a strong flank candidate two to four times as often as the head-to-head winner would justify. Here the mainstream candidate is the one squeezed out.

### Does pragmatic RCV beat honest RCV?

Yes, in the flank scenarios. Compass expression (higher is better; all-mainstream outcome = −3.594):

| Scenario | RCV honest | RCV pragmatic | Approval honest → pragmatic | STAR honest → pragmatic |
|---|---|---|---|---|
| Fitted, left flank 40% | −3.729 | −3.594 | −3.651 → −3.611 | −3.652 → −3.613 |
| Fitted, right flank 35% | −3.660 | −3.594 | −3.589 → −3.594 | −3.594 → −3.594 |
| Stretched, right flank 35% | −3.699 | −3.594 | −3.604 → −3.596 | −3.594 → −3.595 |

- Ranking a mainstream candidate first improves the outcome under RCV, and the swing (0.07–0.14) is several times larger than the swing for approval or STAR (0–0.04).
- So in spoiler scenarios the claim holds: RCV rewards pragmatic ranking, and approval and STAR are less affected.

### Where the claim does not hold

- **Approval is the most strategy-sensitive method on the real contests.** At `rho` = 0, the honest approval rule changes the winner in about 105 more contests than sampling noise explains (about 1% of all contests), and elects the wrong major-party candidate when a fringe candidate is on the ballot. Pragmatic approval, RCV and STAR do not. Approval's outcome depends heavily on where voters draw their approval line.
- **In centrist scenarios all three are affected.** Fitted compass, centrist at 40%: the entrant's win rate falls from 82% to 0% (RCV), 89% to 55% (approval), 82% to 29% (STAR) when voters turn pragmatic.
- **STAR is the most robust overall:** essentially no honest-vs-pragmatic difference on real contests and small ones in flank scenarios.
- **The expression metric favours the locally dominant party.** Voters in the fitted compass sit far outside both parties, so a "centrist" between the parties is not at the centre of any one electorate, and the metric scores a centrist win as slightly worse. All expression differences are small (under 4%) because the parties are close together relative to the voters.

## 3. Random ballot as the fairness benchmark

### a) Does any deterministic method come close?

No. House, averaged over 23 elections:

| Method | Seats away from random ballot's expected split | Disproportionality (Gallagher index) | Elects the head-to-head winner |
|---|---|---|---|
| Random ballot | 0 | 0.3 | 64% |
| Every deterministic method, `rho`=0 | 19.5–19.9 | 3.8–3.9 | 96–100% |
| Honest deterministic methods, `rho`=0.9 | 33–37 | 5.9–6.6 | 96–99% |
| Pragmatic plurality / RCV, `rho`=0.9 | 52 | 9.2 | 88% |

- The spread between deterministic methods (0.4 seats) is negligible next to their gap from random ballot (about 20 seats).
- This is structural: any rule that picks one winner per district by majority-type logic gives the local majority the whole seat. Random ballot gives seats in proportion to votes but elects the head-to-head loser about a third of the time and has the worst compass expression.
- Random ballot is also the only method here where voting honestly is always the best strategy. On the real contests STAR came closest to that property (honest and pragmatic ballots gave the same winners), but it was still moved in the centrist entrant scenarios.

### b) Random ballot if people vote sincerely

Expected third-party House seats per year under random ballot:

| `rho` | 0 | 0.2 | 0.5 | 0.75 | 0.9 |
|---|---|---|---|---|---|
| Third-party seats | 9.6 | 11.1 | 15.5 | 26.6 | 50.1 |

At the data-supported `rho` (0.2–0.5) that is 11–16 seats instead of 10. Histograms in the old style for House 1992 and 2020 are in `img/compass/scenarios/`.

## Caveats

- **`rho` is the whole story and is not pinned down.** The closeness test bounds only closeness-driven desertion.
- **Candidates are fixed.** Different rules would change who runs. That may be the largest real effect and is not modelled.
- **The stretched compass is a what-if.** The fitted compass says taste noise is large; the stretched one assumes it is not.
- **Minor-party positions and the second axis are weakly determined**, as before.
- **Pragmatic rules are one choice each.** Other strategies (burying, bullet voting) are not modelled.
- **Runs are small:** 20 iterations of 1,000 agents for the sweep, 10 for the entrant scenarios, on 800 sampled contests. Differences of a few contests are noise.

## Files

```bash
.venv/bin/python scripts/run_scenarios.py     # ~10 min
.venv/bin/python scripts/plot_scenarios.py
```

| Path | Contents |
|---|---|
| `compass/strategy.py` | Strategic calibration, pragmatic ballots, Gallup five-way, stretched compass |
| `outputs/compass/scenarios/closeness_test.csv` | The falsifiable test |
| `outputs/compass/scenarios/desertion_sweep.csv` | Every method at every `rho` |
| `outputs/compass/scenarios/entrant_scenarios.csv` | Entrant win rates and expression |
| `outputs/compass/scenarios/gallup_vs_compass.csv` | Five-way party ID by year |
| `img/compass/scenarios/` | Nine plots |
