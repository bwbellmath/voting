"""
Training script for Transformer-based Political Compass Posterior Inference

This script:
1. Loads configuration from political_compass_config.json
2. Creates a multi-modal Gaussian target distribution
3. Trains a transformer autoregressive flow to learn the posterior
4. Saves checkpoints and generates visualizations
"""

import torch
import torch.optim as optim
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import argparse

from transformer_flow import TransformerAutoregressiveFlow, MultiModalGaussianMixture


def load_config(config_path: str = 'political_compass_config.json'):
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config['political_compass']


def create_target_distribution(config: dict) -> MultiModalGaussianMixture:
    """Create multi-modal Gaussian target distribution from config"""
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


def train_epoch(
    model: TransformerAutoregressiveFlow,
    target_dist: MultiModalGaussianMixture,
    optimizer: optim.Optimizer,
    batch_size: int,
    n_batches: int,
    noise_std: float,
    device: str,
    gradient_clip: float = 1.0
) -> float:
    """Train for one epoch"""
    model.train()
    total_loss = 0.0

    for _ in range(n_batches):
        # Sample from target distribution
        x = target_dist.sample(batch_size, device=device)

        # Compute negative log likelihood
        log_prob = model.log_prob(x, noise_std=noise_std)
        loss = -log_prob.mean()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        optimizer.step()

        total_loss += loss.item()

    return total_loss / n_batches


def evaluate(
    model: TransformerAutoregressiveFlow,
    target_dist: MultiModalGaussianMixture,
    batch_size: int,
    n_batches: int,
    device: str
) -> dict:
    """Evaluate model performance"""
    model.eval()
    total_nll = 0.0
    total_target_nll = 0.0

    with torch.no_grad():
        for _ in range(n_batches):
            # Sample from target distribution
            x = target_dist.sample(batch_size, device=device)

            # Compute negative log likelihood under model
            log_prob = model.log_prob(x, noise_std=0.0)
            nll = -log_prob.mean()
            total_nll += nll.item()

            # Compute log likelihood under target (for comparison)
            target_log_prob = target_dist.log_prob(x)
            target_nll = -target_log_prob.mean()
            total_target_nll += target_nll.item()

    return {
        'nll': total_nll / n_batches,
        'target_nll': total_target_nll / n_batches,
        'nll_gap': (total_nll - total_target_nll) / n_batches
    }


def visualize_2d_marginals(
    model: TransformerAutoregressiveFlow,
    target_dist: MultiModalGaussianMixture,
    config: dict,
    save_path: str,
    n_samples: int = 10000,
    device: str = 'cpu'
):
    """
    Visualize 2D marginal distributions by projecting the learned distribution
    onto each pair of political compass dimensions.
    """
    dimensions = config['dimensions']
    n_dims = len(dimensions)

    # Sample from both distributions
    model.eval()
    with torch.no_grad():
        model_samples = model.sample(n_samples, device=device).cpu().numpy()
    target_samples = target_dist.sample(n_samples, device='cpu').numpy()

    # Create subplot grid for all pairs
    fig, axes = plt.subplots(n_dims, n_dims, figsize=(16, 16))
    fig.suptitle('Political Compass Posterior - 2D Marginals', fontsize=16, y=0.995)

    for i in range(n_dims):
        for j in range(n_dims):
            ax = axes[i, j]

            if i == j:
                # Diagonal: 1D marginal histograms
                ax.hist(target_samples[:, i], bins=50, alpha=0.5, density=True,
                       color='blue', label='Target')
                ax.hist(model_samples[:, i], bins=50, alpha=0.5, density=True,
                       color='red', label='Model')
                ax.set_ylabel('Density')
                if i == 0:
                    ax.legend(loc='upper right', fontsize=8)

            else:
                # Off-diagonal: 2D marginal scatter + contours
                # Target distribution
                ax.hexbin(target_samples[:, j], target_samples[:, i],
                         gridsize=30, cmap='Blues', alpha=0.3, mincnt=1)

                # Model distribution
                ax.scatter(model_samples[:, j], model_samples[:, i],
                          s=1, alpha=0.3, color='red', label='Model' if i == 0 and j == 1 else None)

                ax.set_xlim(-1.5, 1.5)
                ax.set_ylim(-1.5, 1.5)

                if i == 0 and j == 1:
                    ax.legend(loc='upper right', fontsize=8)

            # Set labels
            if i == n_dims - 1:
                ax.set_xlabel(dimensions[j]['name'].replace('_', ' ').title(), fontsize=8)
            if j == 0:
                ax.set_ylabel(dimensions[i]['name'].replace('_', ' ').title(), fontsize=8)

            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"Saved 2D marginals visualization to {save_path}")
    plt.close()


def plot_training_curves(losses: list, eval_metrics: list, save_path: str):
    """Plot training and evaluation curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Training loss
    ax1.plot(losses, label='Training NLL', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Negative Log Likelihood')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Evaluation metrics
    epochs = list(range(len(eval_metrics)))
    nlls = [m['nll'] for m in eval_metrics]
    target_nlls = [m['target_nll'] for m in eval_metrics]
    gaps = [m['nll_gap'] for m in eval_metrics]

    ax2.plot(epochs, nlls, label='Model NLL', linewidth=2)
    ax2.plot(epochs, target_nlls, label='Target NLL', linewidth=2, linestyle='--')
    ax2.plot(epochs, gaps, label='NLL Gap', linewidth=2, linestyle=':')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Negative Log Likelihood')
    ax2.set_title('Evaluation Metrics')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved training curves to {save_path}")
    plt.close()


def main(args):
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Determine device
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print(f"Using device: {device}")

    # Load configuration
    config = load_config(args.config)
    print(f"Loaded configuration from {args.config}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Create target distribution
    target_dist = create_target_distribution(config)
    print(f"Created {target_dist.n_modes}-mode Gaussian mixture target distribution")

    # Create model
    model_config = config['model_config']['transformer']
    model = TransformerAutoregressiveFlow(
        n_dimensions=config['model_config']['n_dimensions'],
        d_model=model_config['d_model'],
        n_heads=model_config['n_heads'],
        n_layers=model_config['n_layers'],
        d_ff=model_config['d_ff'],
        dropout=model_config['dropout']
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Created transformer flow model with {n_params:,} parameters")

    # Create optimizer
    training_config = config['model_config']['training']
    optimizer = optim.AdamW(model.parameters(), lr=training_config['learning_rate'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_config['n_epochs'])

    # Training loop
    print("\nStarting training...")
    losses = []
    eval_metrics = []
    best_nll_gap = float('inf')

    n_batches_per_epoch = max(1, 10000 // training_config['batch_size'])
    eval_batches = max(1, 5000 // training_config['batch_size'])

    for epoch in range(training_config['n_epochs']):
        # Train
        avg_loss = train_epoch(
            model=model,
            target_dist=target_dist,
            optimizer=optimizer,
            batch_size=training_config['batch_size'],
            n_batches=n_batches_per_epoch,
            noise_std=training_config['noise_augmentation_std'],
            device=device,
            gradient_clip=training_config['gradient_clip']
        )
        losses.append(avg_loss)

        # Evaluate
        if (epoch + 1) % args.eval_interval == 0 or epoch == 0:
            metrics = evaluate(
                model=model,
                target_dist=target_dist,
                batch_size=training_config['batch_size'],
                n_batches=eval_batches,
                device=device
            )
            eval_metrics.append(metrics)

            print(f"Epoch {epoch+1}/{training_config['n_epochs']} | "
                  f"Train NLL: {avg_loss:.4f} | "
                  f"Eval NLL: {metrics['nll']:.4f} | "
                  f"Target NLL: {metrics['target_nll']:.4f} | "
                  f"Gap: {metrics['nll_gap']:.4f}")

            # Save best model
            if metrics['nll_gap'] < best_nll_gap:
                best_nll_gap = metrics['nll_gap']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metrics': metrics,
                    'config': config
                }, output_dir / 'best_model.pt')

        scheduler.step()

        # Save checkpoint
        if (epoch + 1) % args.checkpoint_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'losses': losses,
                'eval_metrics': eval_metrics
            }, output_dir / f'checkpoint_epoch_{epoch+1}.pt')

    # Save final model
    torch.save({
        'epoch': training_config['n_epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses,
        'eval_metrics': eval_metrics,
        'config': config
    }, output_dir / 'final_model.pt')

    print("\nTraining complete!")

    # Generate visualizations
    print("\nGenerating visualizations...")

    # Load best model for visualization
    checkpoint = torch.load(output_dir / 'best_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # 2D marginal visualizations
    visualize_2d_marginals(
        model=model,
        target_dist=target_dist,
        config=config,
        save_path=output_dir / '2d_marginals.png',
        n_samples=args.viz_samples,
        device=device
    )

    # Training curves
    plot_training_curves(
        losses=losses,
        eval_metrics=eval_metrics,
        save_path=output_dir / 'training_curves.png'
    )

    print(f"\nAll outputs saved to {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train transformer flow for political compass posterior')
    parser.add_argument('--config', type=str, default='political_compass_config.json',
                       help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Directory to save outputs')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU usage')
    parser.add_argument('--eval-interval', type=int, default=5,
                       help='Evaluate every N epochs')
    parser.add_argument('--checkpoint-interval', type=int, default=25,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--viz-samples', type=int, default=10000,
                       help='Number of samples for visualization')

    args = parser.parse_args()
    main(args)
