# Hidden Political Compass: Status (2026-11-14)

**Goal:** infer a latent distribution of voter positions (a "political compass" in ℝ^d) and candidate positions, such that voters choosing the nearest candidate under a learned metric reproduce observed election results and survey splits. Report posteriors, including multi-modal ones such as polarization.

Sources: `README.md` (Methodology), `PHASE1_SUMMARY.md`, `POSTERIOR_INFERENCE_README.md`, `todo.org`, and the code.

## What exists

| Component | Files | State |
|---|---|---|
| Data loading | `data_loaders.py` | **Done.** `SurveyLoader` reads Gallup party ID and leaner CSVs. `ElectionLoader` reads 1976–2020 president, senate and house results with year/state filters. `DataAggregator` handles grouping and vote shares. |
| Preprocessing | `preprocessing.py` | **Done.** Demographic encoding, vote-share normalization, temporal features, survey-to-election date alignment. |
| Data viz | `viz_server.py`, `templates/`, `static/` | **Done.** Flask API with D3 charts (survey trends, results by year, state heatmap, survey vs. election). |
| Environment | `voting.yml`, `download_data.sh` | **Done.** Large CSVs are stored in a GitHub Release. Note: this machine has no conda or torch, so nothing could be run here. |
| Posterior model | `transformer_flow.py`, `train_posterior.py`, `political_compass_config.json` | **Toy, and broken** (details below). A transformer autoregressive flow fit only to a synthetic 4-D, 5-mode GMM. It never sees real data. |
| Posterior viz | `generate_viz.py`, `visualize_posterior_d3.py`, `demo_posterior.py`, `outputs/` | Works on the toy model. |
| Voting simulation | `simulate_and_visualize.py`, `vote_random_distance.py` | Works on synthetic voters, but isn't connected to the compass model. |

### Why the flow is broken (from reading the code, confirmed by the saved plot)

1. **Information leak.** The causal mask (`triu(..., diagonal=1)`) lets token *i* attend to itself, and token *i*'s input is `x_i`. So `shift_i` and `scale_i` depend on `x_i`. That breaks the triangular-Jacobian assumption, which means the `log_det = -Σ log_scale` formula, and therefore the NLL, is invalid. The model can optimize a fake likelihood.
2. **Forward and inverse don't match.** `inverse()` returns dim 0 unchanged (`x0 = z0`) and feeds a zeroed current dimension. `forward()` feeds the real `x_i`. Sampling isn't the inverse of training.
3. **Not expressive enough, even once fixed.** It's a single affine autoregressive layer, so every conditional is Gaussian and the dim-0 marginal is a unimodal Gaussian. It can't represent the 5-mode target.
4. **Evidence.** `outputs/2d_marginals.png` shows dim 0 ≈ N(0,1) and dims 1–3 collapsed to a spike at 0. Training was saved at about epoch 25.

## Current plan (distilled)

1. **Mini-problem:** one state, one demographic, several observables: elections across offices and years, plus surveys.
2. **Voter prior:** p_θ(z), in 4-D to start (economic, social, foreign, cultural), multi-modal. Use a flow or GMM.
3. **Candidates:** positions c_j, learned directly at first and later from text embeddings, plus "support" covariates such as party, money and airtime.
4. **Perception noise:** voter view = z + N(0, Σ_learned).
5. **Voting:** argmin_j d(z, c_j) under a learned metric, extended to RCV and approval via per-system strategies.
6. **Likelihood:** predicted aggregate shares vs. actual results (multinomial), plus survey party-ID splits. Fit with strong regularization.
7. **Validation:** held-out elections, calibration of credible intervals, interpretability of the axes.
8. **Later:** condition on demographics, change over time, transformer VAE for text.

Status: steps 1–8 are designed but not implemented. Only the data layer and the toy density estimator (step 2, broken) exist.

## Suggested first milestone for a FABLE pass

1. Repackage the code (`compass/` with `data`, `model`, `vote`, `fit`) and add pytest.
2. Replace the flow with a correct one (`zuko` or `nflows` MAF/NSF with several layers), or start with a learnable GMM. At d=4 a GMM is likely enough.
3. **Synthetic recovery test first:** sample a known prior and known candidates, simulate aggregate results, fit, and check that shares are recovered and the prior is recovered up to symmetry.
4. Make voting differentiable: softmax(−d(z, c_j)/τ) averaged over Monte Carlo samples gives expected vote shares.
5. Fit one real state across all 1976–2020 races (president, senate, house) with shared voters and per-race candidates.

## Unknowns, open questions, challenges

- **Identifiability (the biggest one).** Aggregate vote shares don't change if you rotate, reflect, translate or scale both voters and candidates together. Axis meanings ("economic", "social") can't be recovered without anchors: fixed candidate positions, text embeddings, or issue-specific survey questions. With 2-candidate races, each race gives roughly 1 scalar of information, so a 4-D posterior is heavily under-determined.
- **Data mismatch.** The surveys are national Gallup data from 2022–24. Elections run 1976–2020 (plus some 538 data). There's no state- or demographic-level survey data yet; README points to ANES/Pew/CES. File names are inconsistent (`data-1aTsg` vs `data-1atsq` / `gOKi4` vs `qOKi4`), and `data-survey1/2.csv` are mentioned in the README but missing.
- **Candidate positions.** Learn them, or take them from DW-NOMINATE, text embeddings, or LLM ratings? "Text → shared frame" is listed as the hardest part and hasn't been started.
- **Turnout and abstention.** These aren't modeled. Abstainers bias the fitted voter distribution.
- **Time.** Is one static voter distribution enough, or is z_t needed, and how is it regularized?
- **Model choice.** `todo.org` asks whether this is an embedding problem or an implicit neural representation and leans toward embedding, but the README lists four architectures (flow, transformer VAE, Bayesian LSTM, GMM) with no decision. Recommendation: GMM/flow prior + differentiable voting + variational or SVI fit (Pyro is already in the env).
- **Posterior over what?** A fitted density p_θ(z) is not a posterior over θ. Is uncertainty over θ wanted (SVI, ensembles), or just a multi-modal point estimate?
- **Gerrymandering goal.** `todo.org` Goal 2, state-level gerrymandering per year for the map, is a separate deliverable from the same data. Decide whether it's in scope.
- **Hygiene.** No tests. Duplicate loaders across `vote_utils.py`, `voting.py` and `data_loaders.py`. `*.py~` backups. Flat layout. Untracked `outputs/` and CSVs.
