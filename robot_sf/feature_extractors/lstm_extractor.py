"""LSTM-based feature extractor for robot sensor data.

Treats the 1-D ray array as a sequence (one scalar per LSTM timestep,
ordered by bearing angle) to capture spatial context across adjacent rays —
corridor widths, pedestrian clusters, and obstacle arcs all produce
characteristic sequential patterns that the LSTM can learn to summarise.

Drive state is processed by a small MLP.  The concatenation of the LSTM
final hidden state and the MLP output forms the feature vector returned to
the policy network.

Limitation with standard PPO
-----------------------------
SB3's ``PPO`` does not carry hidden state across environment steps; the
LSTM hidden state is zeroed at the start of each forward pass.  This
extractor therefore provides *within-observation* sequential encoding only
(spatial, not temporal).  For true step-to-step retention you need
``RecurrentPPO`` from ``sb3_contrib``; that package is intentionally kept
out-of-scope here so this extractor can be benchmarked directly against
MLP, CNN, and attention extractors under identical PPO conditions.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from robot_sf.sensor.sensor_fusion import OBS_DRIVE_STATE, OBS_RAYS


class LSTMFeatureExtractor(BaseFeaturesExtractor):
    """LSTM-based feature extractor for ray / drive-state observations.

    The ray array (length N) is reshaped to a sequence of N scalar inputs
    fed one-by-one to the LSTM.  The final hidden state of the last LSTM
    layer is concatenated with a small MLP encoding of the drive state.

    Architecture::

        rays  (N,) → LSTM(input_size=1, hidden_size, num_layers)
                      └─ final hidden → (hidden_size,)
        drive (D,) → Linear → ReLU [× len(drive_hidden_dims)]
                      └─ (drive_hidden_dims[-1],)
        concat → features_dim = hidden_size + drive_hidden_dims[-1]

    Args:
        observation_space: Dict observation space with OBS_RAYS and
            OBS_DRIVE_STATE keys.
        hidden_size: LSTM hidden state dimension.  Larger values improve
            capacity at the cost of slower inference.
        num_layers: Number of stacked LSTM layers.  Depth > 1 adds dropout
            between layers when ``lstm_dropout > 0``.
        lstm_dropout: Dropout probability applied between LSTM layers
            (ignored when ``num_layers == 1``).
        drive_hidden_dims: Hidden layer widths for the drive-state MLP.
            The last element determines its output dimension.
        bidirectional: When True the LSTM scans rays in both directions and
            the effective hidden size doubles (hidden_size * 2).  Increases
            capacity but roughly halves throughput.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        hidden_size: int = 64,
        num_layers: int = 1,
        lstm_dropout: float = 0.0,
        drive_hidden_dims: list[int] | None = None,
        bidirectional: bool = False,
    ) -> None:
        if drive_hidden_dims is None:
            drive_hidden_dims = [32, 16]

        rays_space = cast("spaces.Box", observation_space.spaces[OBS_RAYS])
        drive_space = cast("spaces.Box", observation_space.spaces[OBS_DRIVE_STATE])

        ray_seq_len = int(np.prod(rays_space.shape))
        drive_input_dim = int(np.prod(drive_space.shape))
        drive_output_dim = drive_hidden_dims[-1] if drive_hidden_dims else drive_input_dim
        directions = 2 if bidirectional else 1
        lstm_out_dim = hidden_size * directions

        features_dim = lstm_out_dim + drive_output_dim
        super().__init__(observation_space, features_dim=features_dim)

        self._ray_seq_len = ray_seq_len

        # One scalar per LSTM step; batch_first=True for (B, seq, 1) tensors.
        self.ray_lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # Drive-state MLP
        drive_layers: list[nn.Module] = [nn.Flatten()]
        drive_dims = [drive_input_dim] + drive_hidden_dims
        for in_dim, out_dim in zip(drive_dims[:-1], drive_dims[1:]):
            drive_layers += [nn.Linear(in_dim, out_dim), nn.ReLU()]
        self.drive_mlp = nn.Sequential(*drive_layers)

    def forward(self, obs: dict) -> th.Tensor:
        """Extract features from a batch of observations.

        Args:
            obs: Dict with OBS_RAYS (B, N) and OBS_DRIVE_STATE (B, D).

        Returns:
            Feature tensor of shape (B, features_dim).
        """
        rays = obs[OBS_RAYS]
        # Reshape to (B, seq_len, 1) for batch_first LSTM.
        rays_seq = rays.view(rays.shape[0], self._ray_seq_len, 1)
        # lstm_out: (B, seq_len, directions * hidden_size)
        # h_n: (num_layers * directions, B, hidden_size)
        _, (h_n, _) = self.ray_lstm(rays_seq)
        # Concat forward (and backward for bidirectional) last-layer hidden states.
        if self.ray_lstm.bidirectional:
            # h_n[-2]: forward last layer,  h_n[-1]: backward last layer
            lstm_features = th.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            lstm_features = h_n[-1]  # (B, hidden_size)

        drive_features = self.drive_mlp(obs[OBS_DRIVE_STATE])
        return th.cat([lstm_features, drive_features], dim=1)
