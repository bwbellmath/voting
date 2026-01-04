#!/usr/bin/env python3
"""
Quick demo of the political compass posterior inference system
"""

import torch
import numpy as np
from transformer_flow import TransformerAutoregressiveFlow, MultiModalGaussianMixture
import json

print("=" * 70)
print("Political Compass Posterior Inference - Demo")
print("=" * 70)

# Load config
with open('political_compass_config.json', 'r') as f:
    config = json.load(f)['political_compass']

print("\n1. Political Compass Dimensions:")
for i, dim in enumerate(config['dimensions']):
    print(f"   {i+1}. {dim['name'].replace('_', ' ').title()}")
    print(f"      {dim['description']}")

print("\n2. Political Spectrum Modes:")
modes = config['model_config']['target_distribution']
for i, (name, center, weight) in enumerate(zip(
    modes['mode_descriptions'],
    modes['mode_centers'],
    modes['mode_weights']
)):
    print(f"   {i+1}. {name} ({weight*100:.1f}%)")
    print(f"      Position: {center}")

# Load trained model
print("\n3. Loading trained model...")
checkpoint = torch.load('outputs/best_model.pt', map_location='cpu')
model_config = config['model_config']['transformer']

model = TransformerAutoregressiveFlow(
    n_dimensions=4,
    d_model=model_config['d_model'],
    n_heads=model_config['n_heads'],
    n_layers=model_config['n_layers'],
    d_ff=model_config['d_ff'],
    dropout=0.0
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

n_params = sum(p.numel() for p in model.parameters())
print(f"   Model parameters: {n_params:,}")

# Sample voters
print("\n4. Sampling 10 voters from learned distribution:")
with torch.no_grad():
    samples = model.sample(10, device='cpu').numpy()

dim_names = [d['name'].replace('_', ' ').title() for d in config['dimensions']]
print(f"\n   {'Voter':<8} {dim_names[0]:<25} {dim_names[1]:<25} {dim_names[2]:<25} {dim_names[3]:<25}")
print("   " + "-" * 110)

for i, sample in enumerate(samples):
    print(f"   {i+1:<8} {sample[0]:>6.3f} {' '*19} {sample[1]:>6.3f} {' '*19} {sample[2]:>6.3f} {' '*19} {sample[3]:>6.3f}")

# Compute statistics
print("\n5. Distribution Statistics (1000 samples):")
with torch.no_grad():
    large_sample = model.sample(1000, device='cpu').numpy()

for i, name in enumerate(dim_names):
    mean = large_sample[:, i].mean()
    std = large_sample[:, i].std()
    print(f"   {name}: μ={mean:>6.3f}, σ={std:>5.3f}")

print("\n6. Visualizations:")
print(f"   - Static PNG: outputs/2d_marginals.png")
print(f"   - Interactive D3: outputs/interactive_viz.html")
print(f"\n   Open the HTML file in a browser for interactive exploration!")

print("\n7. Model Performance (from checkpoint):")
if 'metrics' in checkpoint:
    metrics = checkpoint['metrics']
    print(f"   NLL (Model): {metrics['nll']:.4f}")
    print(f"   NLL (Target): {metrics['target_nll']:.4f}")
    print(f"   Gap: {metrics['nll_gap']:.4f}")
else:
    print("   Metrics not available (partial training)")

print("\n" + "=" * 70)
print("Demo complete! See POSTERIOR_INFERENCE_README.md for details.")
print("=" * 70)
