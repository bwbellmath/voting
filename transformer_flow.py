"""
Transformer-based Autoregressive Flow for Political Compass Posterior Inference

Based on T-NAF (Transformer Neural Autoregressive Flows) and TarFlow architectures.
Each dimension of the political compass is treated as a token, and we use attention
masking to enforce autoregressive constraints for density estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, Optional


class PositionalEncoding(nn.Module):
    """Positional encoding for dimension tokens"""
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [seq_len, batch, d_model]"""
        return x + self.pe[:x.size(0)]


class AutoregressiveTransformerBlock(nn.Module):
    """Transformer block with causal masking for autoregressive modeling"""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=False)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [seq_len, batch, d_model]
        attn_mask: [seq_len, seq_len] causal mask
        """
        # Self-attention with causal mask
        x2 = self.norm1(x)
        attn_output, _ = self.self_attn(x2, x2, x2, attn_mask=attn_mask)
        x = x + self.dropout1(attn_output)

        # Feedforward
        x2 = self.norm2(x)
        x2 = self.linear2(self.dropout(F.gelu(self.linear1(x2))))
        x = x + self.dropout2(x2)

        return x


class TransformerAutoregressiveFlow(nn.Module):
    """
    Transformer-based autoregressive normalizing flow for density estimation.

    Architecture:
    - Treats each dimension as a token
    - Uses causal attention masking for autoregressive property
    - Outputs transformation parameters (shift and scale) for each dimension
    """
    def __init__(
        self,
        n_dimensions: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 6,
        d_ff: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.n_dimensions = n_dimensions
        self.d_model = d_model

        # Input projection: project scalar values to d_model dimensions
        self.input_projection = nn.Linear(1, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=n_dimensions)

        # Stack of transformer blocks
        self.transformer_blocks = nn.ModuleList([
            AutoregressiveTransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Output heads: predict shift and log_scale for each dimension
        self.shift_head = nn.Linear(d_model, 1)
        self.log_scale_head = nn.Linear(d_model, 1)

        # Initialize causal mask
        self.register_buffer('causal_mask', self._generate_causal_mask(n_dimensions))

    def _generate_causal_mask(self, size: int) -> torch.Tensor:
        """Generate causal mask to prevent attending to future dimensions"""
        mask = torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1)
        return mask

    def forward(self, x: torch.Tensor, compute_log_det: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the flow.

        Args:
            x: [batch, n_dimensions] input samples
            compute_log_det: whether to compute log determinant of Jacobian

        Returns:
            z: [batch, n_dimensions] transformed samples (base distribution)
            log_det: [batch] log determinant of Jacobian (if compute_log_det=True)
        """
        batch_size = x.size(0)

        # x: [batch, n_dimensions] -> [batch, n_dimensions, 1]
        x_expanded = x.unsqueeze(-1)

        # Project to d_model: [batch, n_dimensions, d_model]
        h = self.input_projection(x_expanded)

        # Transpose for transformer: [n_dimensions, batch, d_model]
        h = h.transpose(0, 1)

        # Add positional encoding
        h = self.pos_encoder(h)

        # Pass through transformer blocks with causal masking
        for block in self.transformer_blocks:
            h = block(h, attn_mask=self.causal_mask)

        # Transpose back: [batch, n_dimensions, d_model]
        h = h.transpose(0, 1)

        # Predict transformation parameters
        shift = self.shift_head(h).squeeze(-1)  # [batch, n_dimensions]
        log_scale = self.log_scale_head(h).squeeze(-1)  # [batch, n_dimensions]

        # Apply affine transformation: z = (x - shift) * exp(-log_scale)
        z = (x - shift) * torch.exp(-log_scale)

        # Compute log determinant of Jacobian
        if compute_log_det:
            # For affine transformation: log|det(J)| = sum(-log_scale)
            log_det = -log_scale.sum(dim=1)  # [batch]
        else:
            log_det = None

        return z, log_det

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """
        Inverse transformation (sampling): z -> x
        Must be done autoregressively dimension by dimension.

        Args:
            z: [batch, n_dimensions] samples from base distribution

        Returns:
            x: [batch, n_dimensions] samples from target distribution
        """
        batch_size = z.size(0)
        x = torch.zeros_like(z)

        # Process each dimension autoregressively
        for dim in range(self.n_dimensions):
            if dim == 0:
                # First dimension: no conditioning, just standard Gaussian
                x[:, dim] = z[:, dim]
            else:
                # Condition on previous dimensions
                x_partial = x[:, :dim+1].clone()
                x_partial[:, dim] = 0  # Zero out current dimension

                # Get transformation parameters
                with torch.no_grad():
                    # Project partial x
                    x_expanded = x_partial.unsqueeze(-1)  # [batch, dim+1, 1]
                    h = self.input_projection(x_expanded)
                    h = h.transpose(0, 1)  # [dim+1, batch, d_model]
                    h = self.pos_encoder(h)

                    # Pass through transformer
                    for block in self.transformer_blocks:
                        h = block(h, attn_mask=self.causal_mask[:dim+1, :dim+1])

                    h = h.transpose(0, 1)  # [batch, dim+1, d_model]

                    # Get parameters for current dimension
                    shift = self.shift_head(h[:, dim, :]).squeeze(-1)  # [batch]
                    log_scale = self.log_scale_head(h[:, dim, :]).squeeze(-1)  # [batch]

                # Inverse transformation: x = z * exp(log_scale) + shift
                x[:, dim] = z[:, dim] * torch.exp(log_scale) + shift

        return x

    def log_prob(self, x: torch.Tensor, noise_std: float = 0.0) -> torch.Tensor:
        """
        Compute log probability of samples under the flow.

        Args:
            x: [batch, n_dimensions] samples
            noise_std: standard deviation of noise augmentation (training only)

        Returns:
            log_prob: [batch] log probability of each sample
        """
        # Add noise augmentation during training
        if noise_std > 0 and self.training:
            x = x + torch.randn_like(x) * noise_std

        # Transform to base distribution
        z, log_det = self.forward(x, compute_log_det=True)

        # Base distribution: standard Gaussian
        base_log_prob = -0.5 * (z ** 2 + math.log(2 * math.pi)).sum(dim=1)

        # Change of variables: log p(x) = log p(z) + log|det(J)|
        log_prob = base_log_prob + log_det

        return log_prob

    def sample(self, n_samples: int, device: str = 'cpu') -> torch.Tensor:
        """
        Sample from the learned distribution.

        Args:
            n_samples: number of samples to generate
            device: device to generate samples on

        Returns:
            samples: [n_samples, n_dimensions]
        """
        self.eval()
        with torch.no_grad():
            # Sample from base distribution (standard Gaussian)
            z = torch.randn(n_samples, self.n_dimensions, device=device)

            # Transform through inverse flow
            x = self.inverse(z)

        return x


class MultiModalGaussianMixture:
    """
    Multi-modal Gaussian mixture model for generating target samples.
    Used as ground truth for training the transformer flow.
    """
    def __init__(
        self,
        n_modes: int,
        mode_centers: np.ndarray,
        mode_covariances: Optional[np.ndarray] = None,
        mode_weights: Optional[np.ndarray] = None,
        mode_scale: float = 0.15
    ):
        """
        Args:
            n_modes: number of mixture components
            mode_centers: [n_modes, n_dimensions] centers of each mode
            mode_covariances: [n_modes, n_dimensions, n_dimensions] or None for scaled identity
            mode_weights: [n_modes] mixture weights (default: uniform)
            mode_scale: scale for identity covariance matrices
        """
        self.n_modes = n_modes
        self.mode_centers = torch.tensor(mode_centers, dtype=torch.float32)
        self.n_dimensions = mode_centers.shape[1]

        # Set covariances
        if mode_covariances is None:
            # Use scaled identity matrices
            self.mode_covariances = torch.eye(self.n_dimensions).unsqueeze(0).repeat(n_modes, 1, 1) * (mode_scale ** 2)
        else:
            self.mode_covariances = torch.tensor(mode_covariances, dtype=torch.float32)

        # Set mixture weights
        if mode_weights is None:
            self.mode_weights = torch.ones(n_modes) / n_modes
        else:
            self.mode_weights = torch.tensor(mode_weights, dtype=torch.float32)
            self.mode_weights = self.mode_weights / self.mode_weights.sum()

    def sample(self, n_samples: int, device: str = 'cpu') -> torch.Tensor:
        """Sample from the mixture model"""
        # Sample mode indices
        mode_indices = torch.multinomial(self.mode_weights, n_samples, replacement=True)

        # Sample from each mode
        samples = torch.zeros(n_samples, self.n_dimensions, device=device)
        for i in range(self.n_modes):
            mask = mode_indices == i
            n_mode_samples = mask.sum().item()
            if n_mode_samples > 0:
                # Create multivariate normal distribution for this mode
                mean = self.mode_centers[i].to(device)
                cov = self.mode_covariances[i].to(device)
                dist = torch.distributions.MultivariateNormal(mean, cov)
                samples[mask] = dist.sample((n_mode_samples,))

        return samples

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Compute log probability of samples"""
        device = x.device
        log_probs = []

        for i in range(self.n_modes):
            mean = self.mode_centers[i].to(device)
            cov = self.mode_covariances[i].to(device)
            dist = torch.distributions.MultivariateNormal(mean, cov)
            log_prob = dist.log_prob(x) + torch.log(self.mode_weights[i].to(device))
            log_probs.append(log_prob)

        # Log sum exp for mixture
        log_probs = torch.stack(log_probs, dim=1)  # [batch, n_modes]
        return torch.logsumexp(log_probs, dim=1)  # [batch]
