"""Quick script to generate visualizations from trained model"""
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from transformer_flow import TransformerAutoregressiveFlow, MultiModalGaussianMixture


def load_config(config_path='political_compass_config.json'):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config['political_compass']


def create_target_distribution(config):
    target_config = config['model_config']['target_distribution']
    mode_centers = np.array(target_config['mode_centers'])
    mode_weights = np.array(target_config['mode_weights'])
    n_modes = target_config['n_modes']
    mode_scale = target_config['mode_scale']

    return MultiModalGaussianMixture(
        n_modes=n_modes,
        mode_centers=mode_centers,
        mode_weights=mode_weights,
        mode_scale=mode_scale
    )


def visualize_2d_marginals(model, target_dist, config, save_path, n_samples=5000, device='cpu'):
    dimensions = config['dimensions']
    n_dims = len(dimensions)

    model.eval()
    with torch.no_grad():
        model_samples = model.sample(n_samples, device=device).cpu().numpy()
    target_samples = target_dist.sample(n_samples, device='cpu').numpy()

    fig, axes = plt.subplots(n_dims, n_dims, figsize=(16, 16))
    fig.suptitle('Political Compass Posterior - 2D Marginals\n(Blue=Target, Red=Learned Model)',
                 fontsize=16, y=0.995)

    for i in range(n_dims):
        for j in range(n_dims):
            ax = axes[i, j]

            if i == j:
                ax.hist(target_samples[:, i], bins=50, alpha=0.5, density=True,
                       color='blue', label='Target')
                ax.hist(model_samples[:, i], bins=50, alpha=0.5, density=True,
                       color='red', label='Model')
                ax.set_ylabel('Density')
                if i == 0:
                    ax.legend(loc='upper right', fontsize=8)
            else:
                ax.hexbin(target_samples[:, j], target_samples[:, i],
                         gridsize=30, cmap='Blues', alpha=0.3, mincnt=1)
                ax.scatter(model_samples[:, j], model_samples[:, i],
                          s=1, alpha=0.3, color='red')
                ax.set_xlim(-1.5, 1.5)
                ax.set_ylim(-1.5, 1.5)

            if i == n_dims - 1:
                ax.set_xlabel(dimensions[j]['name'].replace('_', ' ').title(), fontsize=8)
            if j == 0:
                ax.set_ylabel(dimensions[i]['name'].replace('_', ' ').title(), fontsize=8)

            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"Saved 2D marginals to {save_path}")
    plt.close()


# Load config and model
config = load_config()
checkpoint = torch.load('outputs/best_model.pt', map_location='cpu')

model_config = config['model_config']['transformer']
model = TransformerAutoregressiveFlow(
    n_dimensions=config['model_config']['n_dimensions'],
    d_model=model_config['d_model'],
    n_heads=model_config['n_heads'],
    n_layers=model_config['n_layers'],
    d_ff=model_config['d_ff'],
    dropout=model_config['dropout']
)

model.load_state_dict(checkpoint['model_state_dict'])
target_dist = create_target_distribution(config)

# Generate visualization
print("Generating 2D marginals...")
visualize_2d_marginals(model, target_dist, config, 'outputs/2d_marginals.png', n_samples=10000)
print("Done!")
