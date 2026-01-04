## Voting Analysis Project

This project performs analysis on voting data using Python and various scientific libraries, with a focus on learning hidden political compass embeddings from survey and election data. The analysis can be enhanced by integrating election results data from the [fivethirtyeight/election-results](https://github.com/fivethirtyeight/election-results) repository.

## Setup

### 1. Clone the Repository

```bash
git clone git@github.com:bwbellmath/voting.git
cd voting
```

### 2. Download Large Data Files

Large CSV files (407 MB total) are hosted on GitHub Releases to avoid exceeding repository size limits.

**Quick download:**
```bash
./download_data.sh
```

**Manual download (if you don't have `gh` CLI):**
```bash
# Install GitHub CLI first
brew install gh
gh auth login

# Then download the files
gh release download v1.0-data -D csv/old/ -R bwbellmath/voting
```

**Or download directly from the web:**
Visit https://github.com/bwbellmath/voting/releases/tag/v1.0-data and download the files to `csv/old/`

**Files included:**
- `1976-2020-house-stoch_test.csv` (282 MB) - House election stochastic test data
- `1976-2020-senate_cur.csv` (93 MB) - Current Senate election data
- `1976-2020-senate-stoch_test.csv` (32 MB) - Senate election stochastic test data

### 3. Set Up Conda Environment

```bash
conda env create -f voting.yml
conda activate voting
```

This installs all required dependencies including PyTorch, PyTorch Lightning, Normalizing Flows libraries, Pyro, Flask, and data processing tools.

### 4. Initialize Submodules (Optional)

To include the fivethirtyeight election results data:

```bash
git submodule update --init --recursive
```

## Quick Start

1. **Test the data loaders**:
   ```bash
   python data_loaders.py
   ```

2. **Launch the D3 visualization interface**:

   **On the server (local or remote):**
   ```bash
   python viz_server.py
   ```
   The server will start on http://localhost:5000.

   **Accessing remotely via SSH tunnel:**

   If the server is running on a remote host, you can access it securely using an SSH tunnel:

   a. **Enable TCP forwarding on the remote server** (one-time setup):
      ```bash
      # On the remote server, edit /etc/ssh/sshd_config
      sudo sed -i 's/AllowTcpForwarding no/AllowTcpForwarding yes/' /etc/ssh/sshd_config
      sudo systemctl restart ssh
      ```

   b. **Create the SSH tunnel** (from your local machine):
      ```bash
      ssh -f -N -L 8000:localhost:5000 <remote-host>
      ```
      This forwards local port 8000 to the remote server's port 5000 through the SSH connection.

      Note: We use port 8000 locally because macOS ControlCenter may occupy port 5000.

   c. **Access the visualization**:
      Open http://localhost:8000 in your browser.

   **How SSH tunneling works:**
   - All traffic flows through the existing SSH connection (no additional firewall ports needed)
   - The tunnel securely encapsulates HTTP traffic within the SSH connection
   - Perfect for accessing services behind firewalls or NAT

3. **Run preprocessing tests**:
   ```bash
   python preprocessing.py
   ```

## Managing Your Environment

### Updating the Environment

If the `voting.yml` file is updated, you can update your existing environment:

```bash
conda env update -f voting.yml --prune
```

Or remove and recreate it:

```bash
conda env remove -n voting
conda env create -f voting.yml
```

### Updating Submodules

To pull the latest election results data:

```bash
git submodule update --remote
```

## Data Documentation

### Survey Data and Election Results in This Repository

This project uses several datasets, both internal to the project and from the **fivethirtyeight/election-results** submodule, to analyze voter behavior, party identification, and election results over time.

#### Internal Datasets:

1. **`data-1atsq.csv`**: 
   - **Description**: This dataset contains longitudinal survey data focused on **party identification**. Categories include strong Democrat, weak Democrat, independent, Republican, and other minor parties. It also tracks how these affiliations have shifted over time, providing a historical overview of voter self-identification.
   
2. **`data-qOKi4.csv`**: 
   - **Description**: This file provides more granular insights into voter preferences. It includes responses from independents, breaking down whether they tend to lean toward a major party or prefer a third-party option. This data is crucial for modeling voter behavior, especially among those who are not strongly aligned with either major party.

3. **`data-survey1.csv`**: 
   - **Description**: Contains raw survey data from a recent poll focused on voter preferences, including data on party affiliation, voter turnout likelihood, and preferences in hypothetical election matchups. This dataset is useful for understanding current voter sentiment.

4. **`data-survey2.csv`**: 
   - **Description**: A follow-up survey to `data-survey1.csv`, capturing additional variables such as voter issue priorities and how those issues impact their party affiliation. It provides deeper context for voters' shifting allegiances.

#### Datasets from the fivethirtyeight/election-results Submodule:

1. **`presidential_results.csv`**: 
   - **Description**: This file contains detailed data on **presidential election results** over time, including state-by-state breakdowns of vote totals, percentages, and winning candidates. This dataset allows for analysis of trends in presidential elections and how they correlate with changes in party identification and voter behavior.

2. **`senate_results.csv`**: 
   - **Description**: A dataset focused on **U.S. Senate election results**. It includes vote counts, percentages, and party affiliations of candidates across multiple election cycles. This is useful for comparing local/state trends with national voter behaviors.

3. **`house_results.csv`**: 
   - **Description**: Provides election results for **U.S. House races**, with district-level data on vote totals, percentages, and party affiliations. This dataset supports analysis of voting patterns in smaller, more localized races compared to national elections.

4. **`governor_results.csv`**: 
   - **Description**: This file contains election results for **gubernatorial races**. It includes state-level data on candidates, vote totals, and party affiliations, offering insights into how state-level executive races reflect or differ from broader national trends.

5. **`special_election_results.csv`**: 
   - **Description**: Tracks the results of **special elections** across various offices. This dataset can be particularly useful for analyzing how non-standard election cycles (such as mid-term vacancies) affect voter turnout and party support.

6. **`primary_results.csv`**: 
   - **Description**: Focuses on **primary election results**, breaking down how party candidates perform during the primary process. It includes vote totals and percentages for each primary candidate, which can help assess party dynamics and internal competition.

#### Usage:
These datasets are utilized to model voter behavior, simulate elections, and explore how shifts in party identification over time may influence election outcomes. By analyzing both survey data and actual election results, this project aims to provide a comprehensive understanding of voter dynamics.

## Methodology: Learning Hidden Political Compass Embeddings

### Overview

This project aims to **estimate posterior distributions of political compass values** that reproduce observed survey results and election outcomes under a voting simulation where voters select the candidate nearest to their preference in a learned metric space. The approach addresses the fundamental challenge that voter preferences are latent (unobserved) variables that must be inferred from aggregate data like surveys and election results.

### The Core Problem

Given:
- Survey data on voter party identification and preferences
- Historical election results (vote shares, turnout, margins)
- Potentially text data on candidates and platforms

Infer:
- Latent political compass embeddings for voters
- Candidate positions in the same embedding space
- A learned metric distance function that predicts voting behavior
- Posterior distributions that capture uncertainty and multi-modality (e.g., political polarization)

### Mathematical Framework

#### Generative Model

1. **Voter Embeddings**: Each voter `i` has a latent political compass position `z_i ∈ ℝ^d` sampled from a learned distribution:
   ```
   z_i ~ p_θ(z | demographics_i, state_i)
   ```

2. **Candidate Embeddings**: Each candidate `j` has a position `c_j ∈ ℝ^d` which can be:
   - Learned from text embeddings of platforms/statements
   - Directly optimized to fit observed data
   - Constrained by party affiliation or other observables

3. **Voting Model**: Voter `i` votes for candidate `j*` where:
   ```
   j* = argmin_j d(z_i, c_j)
   ```
   where `d(·,·)` is a learned metric (potentially non-Euclidean).

4. **Observation Model**: Aggregate election results `y` are generated from individual votes:
   ```
   y ~ Multinomial(votes(z, c, d))
   ```

#### Inference Objective

Learn parameters `θ` to maximize:
```
p(θ | y_surveys, y_elections) ∝ p(y_surveys, y_elections | θ) p(θ)
```

### Machine Learning Architectures

The project explores several state-of-the-art architectures for handling multi-modal posterior distributions:

#### 1. Normalizing Flows (Recommended for Multi-Modal Distributions)

**Advantages**:
- Exact likelihood computation (unlike VAEs)
- Naturally models complex, multi-modal distributions
- Invertible transformations allow sampling and density estimation
- Well-suited for capturing political polarization (bimodal/multimodal voter distributions)

**Architecture**:
```python
# Conditional normalizing flow
z_i ~ Flow_θ(demographics_i, state_i)

# Multiple coupling layers with permutations
Flow = Compose([
    CouplingLayer(dim=d, context_dim=demographics_dim),
    RandomPermutation(),
    CouplingLayer(dim=d, context_dim=demographics_dim),
    ...
])
```

**Implementation Notes**:
- Use conditional flows to incorporate demographic/geographic features
- Coupling layers enable modeling complex multivariate distributions
- Continuous normalizing flows for smooth, expressive densities

#### 2. Transformer-Based Variational Autoencoder (VAE)

**Advantages**:
- Handles sequential decision-making (ranked choice voting)
- Attention mechanisms capture dependencies between voter groups
- Scalable to large datasets
- Good for incorporating text embeddings of candidate platforms

**Architecture**:
```python
class TransformerVAE(nn.Module):
    def __init__(self, latent_dim, context_dim):
        self.encoder = TransformerEncoder(...)  # Demographics, surveys → μ, σ
        self.decoder = TransformerDecoder(...)  # z → predicted votes

    def forward(self, voter_features, election_context):
        # Encode to latent distribution
        μ, log_σ = self.encoder(voter_features)

        # Sample using reparameterization trick
        z = μ + σ * ε, where ε ~ N(0, I)

        # Decode to voting behavior
        votes = self.decoder(z, election_context)

        return votes, μ, log_σ
```

**Loss Function**:
```
L = -E[log p(votes | z)] + KL(q(z | voter_features) || p(z))
```

**Key Challenge**: Standard Gaussian priors produce unimodal posteriors. Solutions:
- Use mixture of Gaussians prior: `p(z) = Σ_k π_k N(μ_k, Σ_k)`
- Variational mixture families for the posterior
- Diffusion-based decoders for greater flexibility

#### 3. Bayesian LSTM for Sequential Voting Decisions

**Advantages**:
- Natural fit for ranked-choice and sequential voting systems
- Captures temporal dependencies in panel surveys
- Posterior uncertainty quantification over parameters

**Architecture**:
```python
class BayesianLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2)
        self.latent_projection = nn.Linear(hidden_dim, latent_dim * 2)  # μ and σ

    def forward(self, voter_sequence):
        # Process temporal voter data
        h_t, _ = self.lstm(voter_sequence)

        # Predict latent political position over time
        μ, log_σ = self.latent_projection(h_t).chunk(2, dim=-1)

        return μ, log_σ
```

**Inference**: Use variational inference or Adaptive Gauss-Hermite Quadrature (AGHQ) to approximate posterior over LSTM weights and hidden states.

#### 4. Gaussian Mixture Models (GMM) for Voter Clusters

**Advantages**:
- Explicitly models voter clustering (party identification, ideological camps)
- Interpretable components
- Can be combined with neural networks

**Architecture**:
```python
class ConditionalGMM(nn.Module):
    def __init__(self, n_components, latent_dim, condition_dim):
        self.mixture_network = nn.Sequential(...)  # demographics → π_k
        self.mean_network = nn.Sequential(...)      # demographics → μ_k
        self.cov_network = nn.Sequential(...)       # demographics → Σ_k

    def forward(self, demographics):
        π = softmax(self.mixture_network(demographics))
        μ = self.mean_network(demographics)
        Σ = positive_definite(self.cov_network(demographics))

        return MixtureOfGaussians(π, μ, Σ)
```

### Regularization and Training Strategies

1. **Regularization Terms**:
   - Encourage smooth embeddings: `R_smooth = ||∇_demographics z||²`
   - Party alignment: Constrain known party affiliations to cluster
   - Temporal consistency: For panel data, penalize rapid shifts in individual embeddings

2. **Multi-Task Learning**:
   - Joint training on surveys + elections + text data
   - Shared encoder, task-specific decoders
   - Weighted loss: `L = λ_survey L_survey + λ_election L_election + λ_text L_text`

3. **Data Augmentation**:
   - Bootstrap resampling of surveys
   - Synthetic voter generation from learned distributions
   - Counterfactual elections (what-if candidate scenarios)

### Training Pipeline

```
1. Data Preprocessing
   ├── Survey data → voter demographics + preferences
   ├── Election results → vote shares by district/state
   └── Text data → candidate platform embeddings (BERT/GPT)

2. Model Training
   ├── Initialize: Random embeddings or pre-trained text embeddings
   ├── Forward pass: demographics → latent z → voting simulation → predicted results
   ├── Compute loss: Compare predicted vs. actual survey/election outcomes
   ├── Backward pass: Update θ via gradient descent
   └── Regularize: Apply smoothness/clustering penalties

3. Posterior Sampling
   ├── For each voter: Sample z_i ~ p_θ(z | demographics_i)
   ├── Estimate uncertainty via Monte Carlo
   └── Visualize distributions (marginals, joint densities)

4. Validation
   ├── Hold-out test set of elections
   ├── Check calibration: Do predicted vote shares match actuals?
   └── Ablation studies: Remove features/data sources, measure impact
```

### Handling Multi-Modal Distributions

Political polarization creates multi-modal voter distributions. Standard approaches fail:
- **Problem**: Gaussian VAEs produce unimodal posteriors
- **Solutions**:
  1. **Normalizing Flows**: Inherently multi-modal via invertible transforms
  2. **Mixture VAEs**: Use mixture-of-Gaussians prior and posterior
  3. **Diffusion Models**: Recent work (MDDVAE) combines diffusion + VAE for multimodal generation
  4. **Product of Experts**: Combine multiple encoder distributions

### Evaluation Metrics

1. **Predictive Accuracy**:
   - Election outcome prediction (winner, vote share)
   - Survey response prediction (party ID, candidate preference)

2. **Calibration**:
   - Do 90% credible intervals contain true values 90% of the time?
   - Reliability diagrams for probabilistic predictions

3. **Interpretability**:
   - Do learned embeddings align with known political axes (left-right, authoritarian-libertarian)?
   - Can we identify voter segments that match known demographics?

4. **Generalization**:
   - Cross-validation across states, years, election types
   - Transfer learning: Train on one region, test on another

### Current Implementation Status

- ✅ Voter preference generation (`vote_random.py`)
- ✅ Voting simulation for multiple systems (`simulate_and_visualize.py`)
- 🚧 Neural architecture for inference (`voting_ml_game.py` - in progress)
- 🚧 Training pipeline on historical data
- 📋 Planned: Normalizing flow implementation
- 📋 Planned: Transformer VAE for text incorporation

---

For setup instructions, see the [Setup](#setup) section at the top of this README.
