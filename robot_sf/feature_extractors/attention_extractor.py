"""
Attention-based feature extractor for robot environments.

This extractor uses self-attention mechanisms to process LiDAR rays,
allowing the model to focus on the most relevant rays for decision making.
"""

from typing import List

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from robot_sf.sensor.sensor_fusion import OBS_DRIVE_STATE, OBS_RAYS


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module for processing sequential data.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: th.Tensor) -> th.Tensor:
        batch_size, seq_len, embed_dim = x.shape

        # Generate queries, keys, values
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scores = th.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_weights = th.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        attended = th.matmul(attention_weights, V)

        # Reshape and project output
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        output = self.output_proj(attended)

        return output


class AttentionFeatureExtractor(BaseFeaturesExtractor):
    """
    Attention-based feature extractor for robot sensor data.

    This extractor treats LiDAR rays as a sequence and applies self-attention
    to learn which rays are most important for navigation decisions.

    Advantages:
    - Can learn to focus on relevant parts of the environment
    - Handles variable-length sequences naturally
    - More interpretable than pure CNNs
    - Better at learning long-range dependencies

    Attributes:
        observation_space: The space of possible observations from the environment
        embed_dim: Embedding dimension for attention mechanism
        num_heads: Number of attention heads
        num_layers: Number of attention layers
        dropout_rate: Dropout rate for regularization
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout_rate: float = 0.1,
        drive_hidden_dims: List[int] = [32, 16],
    ):
        # Extract observation spaces
        rays_space: spaces.Box = observation_space.spaces[OBS_RAYS]
        drive_state_space: spaces.Box = observation_space.spaces[OBS_DRIVE_STATE]

        # Calculate dimensions
        num_timesteps, num_rays = rays_space.shape
        drive_input_dim = np.prod(drive_state_space.shape)
        drive_output_dim = drive_hidden_dims[-1] if drive_hidden_dims else drive_input_dim

        # Total features: attention output + drive state features
        total_features = embed_dim + drive_output_dim

        # Initialize the base feature extractor
        super().__init__(observation_space, features_dim=total_features)

        # Ray embedding layer (maps each ray to embedding dimension)
        self.ray_embedding = nn.Sequential(
            nn.Linear(num_timesteps, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # Attention layers
        self.attention_layers = nn.ModuleList(
            [MultiHeadAttention(embed_dim, num_heads, dropout_rate) for _ in range(num_layers)]
        )

        # Layer normalization for attention
        self.layer_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])

        # Global pooling for final representation
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Drive state processing MLP
        drive_layers = []
        drive_dims = [drive_input_dim] + drive_hidden_dims

        for i in range(len(drive_dims) - 1):
            drive_layers.extend(
                [nn.Linear(drive_dims[i], drive_dims[i + 1]), nn.ReLU(), nn.Dropout(dropout_rate)]
            )

        self.drive_state_extractor = nn.Sequential(nn.Flatten(), *drive_layers)

    def forward(self, obs: dict) -> th.Tensor:
        """
        Extract features using attention mechanism for rays and MLP for drive state.

        Args:
            obs: Dictionary containing ray and drive state observations

        Returns:
            Concatenated features from attention processing and drive state
        """
        # Process rays with attention
        rays = obs[OBS_RAYS]  # Shape: (batch, timesteps, num_rays)

        # Transpose to treat rays as sequence elements
        rays_transposed = rays.transpose(1, 2)  # Shape: (batch, num_rays, timesteps)

        # Embed each ray
        ray_embeddings = self.ray_embedding(rays_transposed)  # Shape: (batch, num_rays, embed_dim)

        # Apply attention layers with residual connections
        attended_rays = ray_embeddings
        for attention, layer_norm in zip(self.attention_layers, self.layer_norms):
            attended_rays = layer_norm(attended_rays + attention(attended_rays))

        # Global average pooling to get fixed-size representation
        attended_rays = attended_rays.transpose(1, 2)  # Shape: (batch, embed_dim, num_rays)
        ray_features = self.global_pool(attended_rays).squeeze(-1)  # Shape: (batch, embed_dim)

        # Process drive state
        drive_features = self.drive_state_extractor(obs[OBS_DRIVE_STATE])

        # Concatenate features
        return th.cat([ray_features, drive_features], dim=1)
