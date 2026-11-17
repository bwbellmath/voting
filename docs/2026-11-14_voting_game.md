# Voting Game: Status (2026-11-14)

## Important: the "hidden targets" design was not found

I couldn't find a design for a voting game with hidden targets anywhere I searched:

- this repo, including all branches and stashes;
- `~/code/voting-old`, `~/code/voting_2` and `~/code/voting_back`;
- `~/docs`, Dropbox, and past Claude Code session transcripts.

A search for "hidden target", "secret target", "hidden objective" and "voting game" matched only `todo.org`. If the design exists, it's probably in a claude.ai chat, a notebook, or on another machine. The rest of this doc covers the game-related code that does exist.

## What exists

| File | State | What it does |
|---|---|---|
| `simulate_and_visualize.py` | **Works** (490 lines) | Runs plurality, approval, Borda, IRV and Condorcet on synthetic voters and writes comparison plots to `img/`. Voter types: uniform, polarized, moderate, skewed. |
| `vote_random.py` | **Works** | `VoterModel`: builds voter score vectors from orientation fractions plus noise, then converts them to plurality, ranked or approval ballots. |
| `vote_random_distance.py` | **Works** | Same idea, but scores come from inverse-square distance between candidates placed on a line in D dimensions. |
| `voting_ml_game.py` | **Skeleton only** (171 lines) | Data classes `VoterFeatures` and `ElectionContext`, an enum of four voting systems, `GaussianMixtureModel`, `VoterStrategyNetwork` (encoders, then a GRU, then one output head per voting system), and `VotingSimulation`, whose methods are all `pass`. |
| `senate.py`, `vote.py`, `president.py` | Legacy, partly working | Older "dice-roll" stochastic elections on real 1976–2020 results. `vote.py` is broken: it calls an undefined `extract_elections` and has an empty `for` loop body. |

### Design intent (from `todo.org` and `voting_ml_game.py`)

- **Step 5:** "Each voting system … will generate a set of strategies (weightings of the various axes of the embedding of the politicians) and will execute those strategies according to the voting games."
- **Later goal:** "Convert model to use latent embeddings from a transformer … so that the problem can be converted for the DefCon Game."
- A voter is defined by its demographics plus a latent vector sampled from a GMM. A learned strategy network maps (voter, election context) to a ballot for each voting system.

### Bugs found in `voting_ml_game.py`

- **GRU shape mismatch:** the GRU expects `input_size = hidden*3`, but only two encodings (`hidden*2`) are concatenated. Any forward pass will crash.
- **Unused encoder:** `latent_encoder` is defined but never used.
- **Invalid GMM parameters:** `mixture_weights` is passed straight to `torch.multinomial` without softmax, so it can go negative. `covs` has no positive-definite constraint.
- **No gradients through sampling:** GMM sampling uses `.sample()` inside a Python loop, so samples aren't reparameterized and gradients can't flow back. As written, the GMM can't be trained.
- **No loss or game loop:** there is no loss, reward, opponent, or game loop. Nothing defines what "winning" or a "target" means.

## Current plan (distilled)

1. Define the game: players, what each player's hidden target is, actions, observations, scoring, and the voting systems in play.
2. Voters live in a latent space, and candidates are embedded in the same space.
3. Each voting system has a strategy: a weighting over embedding axes that turns preferences into a ballot.
4. Train the strategy networks, by RL or by differentiable simulation, to hit the targets.
5. Later, swap the GMM latents for transformer embeddings (the "DefCon game" version).

Only the plain simulation part of steps 2–3 runs today.

## Unknowns, open questions, challenges

- **The core spec is missing.** What is a "hidden target"? Is it a desired winner, a vote-share band, or a policy position, and whose target is it (each voter's, each candidate's, or a party's)? Who knows whose target? Is the game one-shot, repeated, or multi-round? Without this, FABLE would be inventing the game, not finishing it.
- **Who learns?** Do voters learn strategic ballots, candidates learn positions, or both, adversarially?
- **Training method:** RL (non-differentiable vote counting), relaxed or differentiable voting (softmax/Sinkhorn ranks), or game-theoretic solvers (fictitious play, CFR)?
- **Link to the hidden-compass project:** is the game meant to sit on top of the learned compass (voters sampled from the fitted posterior)? If so, the game depends on that project.
- **What "DefCon Game" means:** a competition, a demo, or a CTF-style challenge? This sets scope, audience and deadline.
- **Evaluation:** what counts as success? Strategy beats sincere voting? An equilibrium is found? Players can't infer each other's targets?
- **Code hygiene:** stray `*.py~` backups, duplicated loaders (`vote_utils.py` / `voting.py` / `data_loaders.py`), no package structure, and no tests.
