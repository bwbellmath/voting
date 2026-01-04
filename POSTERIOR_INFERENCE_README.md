# Political Compass Posterior Inference

## Overview

This project implements a **transformer-based autoregressive normalizing flow** to learn the posterior distribution of political preferences in a multi-dimensional political compass space. The goal is to model voter distributions that can be used to recreate election and poll results.

## Architecture

### Transformer Autoregressive Flow (Based on TarFlow/T-NAF)

The model uses state-of-the-art transformer-based density estimation (Dec 2024):

- **Input**: Each dimension of the political compass is treated as a token
- **Autoregressive Constraint**: Causal attention masking ensures each dimension only depends on previous dimensions
- **Transformation**: Affine coupling layers parameterized by transformer outputs (shift and scale)
- **Base Distribution**: Standard Gaussian (allows for efficient sampling)

**Key Innovation**: Unlike traditional normalizing flows that compose multiple transformations, this uses a single transformer-based flow with ~400K parameters, achieving efficient density estimation through attention mechanisms.

## Political Compass Dimensions

We model voter preferences across **4 dimensions**:

1. **Economic Left-Right** (-1 to 1)
   - Left: Government intervention, redistribution
   - Right: Free market, low taxes

2. **Social Libertarian-Authoritarian** (-1 to 1)
   - Libertarian: Individual freedom, civil liberties
   - Authoritarian: State control, traditional values

3. **Interventionist-Isolationist** (-1 to 1)
   - Interventionist: Global engagement, military action
   - Isolationist: Non-intervention, America First

4. **Progressive-Traditional** (-1 to 1)
   - Progressive: Social change, diversity
   - Traditional: Cultural preservation, stability

## Target Distribution

Multi-modal Gaussian mixture with **5 modes** representing political clusters:

1. **Progressive Left** (25%): Bernie Sanders, AOC
   - Center: [-0.7, -0.6, -0.3, -0.8]

2. **Moderate Left** (30%): Biden, Obama
   - Center: [-0.4, -0.2, -0.1, -0.3]

3. **Centrist** (15%): Romney, Manchin
   - Center: [0.0, 0.0, 0.0, 0.0]

4. **Conservative Right** (25%): Trump, DeSantis
   - Center: [0.6, 0.5, 0.4, 0.7]

5. **Libertarian** (5%): Rand Paul
   - Center: [0.5, -0.7, 0.6, 0.2]

## Files

### Core Implementation

- `political_compass_config.json` - Configuration file with dimension definitions and model hyperparameters
- `transformer_flow.py` - Transformer autoregressive flow implementation
  - `TransformerAutoregressiveFlow` - Main model class
  - `MultiModalGaussianMixture` - Target distribution generator
  - `AutoregressiveTransformerBlock` - Transformer block with causal masking

### Training & Visualization

- `train_posterior.py` - Training script with evaluation and visualization
- `visualize_posterior_d3.py` - Interactive D3.js visualization generator
- `generate_viz.py` - Quick visualization script

### Model Hyperparameters

```json
{
  "d_model": 128,        // Embedding dimension
  "n_heads": 4,          // Attention heads
  "n_layers": 6,         // Transformer blocks
  "d_ff": 512,           // Feedforward dimension
  "dropout": 0.1,
  "batch_size": 256,
  "learning_rate": 0.0001,
  "n_epochs": 100,
  "noise_augmentation_std": 0.01  // TarFlow's Gaussian noise augmentation
}
```

## Usage

### Training

```bash
# Full training (100 epochs)
python3 train_posterior.py --output-dir outputs

# Quick training (10 epochs)
python3 train_posterior.py --output-dir outputs --epochs 10

# CPU-only training
python3 train_posterior.py --cpu
```

### Visualization

```bash
# Generate static 2D marginals (PNG)
python3 generate_viz.py

# Generate interactive D3 visualization (HTML)
python3 visualize_posterior_d3.py --checkpoint outputs/best_model.pt --output outputs/interactive_viz.html

# Open in browser
open outputs/interactive_viz.html
```

### Sampling from Learned Distribution

```python
import torch
from transformer_flow import TransformerAutoregressiveFlow

# Load trained model
checkpoint = torch.load('outputs/best_model.pt')
model = TransformerAutoregressiveFlow(n_dimensions=4, d_model=128, n_heads=4, n_layers=6, d_ff=512)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Sample 1000 voters
with torch.no_grad():
    voter_positions = model.sample(n_samples=1000, device='cpu')

# voter_positions: [1000, 4] tensor with positions in [-1, 1] for each dimension
```

## Visualizations

### 1. 2D Marginal Plots

- **Diagonal**: 1D histograms showing marginal densities per dimension
- **Off-diagonal**: 2D scatter plots showing correlations between dimension pairs
- **Blue**: Target distribution (ground truth multi-modal Gaussian)
- **Red**: Learned model distribution

### 2. Interactive D3 Visualization

Features:
- Select any two dimensions for X/Y axes
- Toggle mode centers on/off
- Interactive exploration of the 4D distribution
- Shows all 5 political modes

## Training Results

The model learns to approximate the multi-modal structure through:

1. **Negative Log-Likelihood Loss**: Maximize probability of samples from target distribution
2. **Noise Augmentation**: Gaussian noise (σ=0.01) during training for robustness
3. **Autoregressive Sampling**: Dimension-by-dimension generation conditioned on previous dimensions

**Note**: The current visualization shows partial training results (~25-50 epochs). For full convergence, train for 100+ epochs.

## Next Steps

### Immediate Improvements

1. **Extended Training**: Run for full 100 epochs for better convergence
2. **Hyperparameter Tuning**: Experiment with model size, learning rate, noise augmentation
3. **Denoising**: Implement post-training denoising (TarFlow methodology)

### Integration with Election Data

1. **Candidate Positioning**: Map real politicians to political compass coordinates
2. **Distance-Based Voting**: Implement voting model based on Euclidean distance in compass space
3. **Gaussian Noise on Voters**: Add perception noise (already mentioned in project goals)
4. **Poll Simulation**: Generate synthetic polls by sampling voters and computing preferences

### Advanced Extensions

1. **Conditional Flow**: Condition on demographics (state, age, education) → `p(compass | demographics)`
2. **Temporal Evolution**: Model how distributions shift over time
3. **Two-Distribution Model**: Separate voter distribution from candidate distribution (future goal)
4. **Real Data Fitting**: Train on actual voter survey data (e.g., ANES, Pew Research)

## Literature

### Key Papers

1. **TarFlow (Dec 2024)**: "Normalizing Flows are Capable Generative Models"
   - arXiv: 2412.06329
   - State-of-the-art likelihood estimation on images
   - Introduces Gaussian noise augmentation and post-training denoising

2. **T-NAF (Jan 2024)**: "Transformer Neural Autoregressive Flows"
   - arXiv: 2401.01855
   - Treats dimensions as tokens with causal attention masking
   - Order of magnitude fewer parameters than previous flows

3. **STARFlow (2025)**: "Scaling Latent Normalizing Flows"
   - Apple ML Research
   - Scalable approach with deep-shallow transformer design

## Model Architecture Details

### Forward Pass (Density Estimation)

```
Input x: [batch, n_dimensions]
  ↓
Project each dimension to d_model: [batch, n_dimensions, d_model]
  ↓
Add positional encoding
  ↓
Transformer blocks with causal masking (6 layers)
  ↓
Predict shift & log_scale: [batch, n_dimensions, 2]
  ↓
Transform: z = (x - shift) * exp(-log_scale)
  ↓
Compute log p(x) = log p(z) + log|det(J)|
```

### Inverse Pass (Sampling)

```
Sample z ~ N(0, I): [batch, n_dimensions]
  ↓
For each dimension i (autoregressive):
  - Condition on previous dimensions x[:i]
  - Get shift_i and log_scale_i from transformer
  - Compute: x_i = z_i * exp(log_scale_i) + shift_i
  ↓
Output x: [batch, n_dimensions]
```

## Performance Metrics

- **Negative Log-Likelihood (NLL)**: Lower is better
- **NLL Gap**: Model NLL - Target NLL (measures approximation quality)
- **Visual Inspection**: 2D marginals should match target distribution

---

**Created**: December 2024
**Based on**: TarFlow (arXiv:2412.06329), T-NAF (arXiv:2401.01855)
**Framework**: PyTorch 2.1.0
