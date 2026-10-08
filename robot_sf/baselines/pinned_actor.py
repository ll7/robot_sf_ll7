"""Float64 NumPy actor matching the CNN/MLP policy measured in issue #10211.

Weights are copied once. Unsupported modules fail closed, rather than silently
returning to a float32 or stochastic implementation. Observation adaptation and
Box action clipping remain owned by the existing PPO adapter.
"""

from __future__ import annotations

import numpy as np


def _compile(module):  # noqa: C901 - supported inference layer dispatch
    """Compile supported inference layers.

    Returns:
        Cached float64 NumPy forward function.
    """
    from torch import nn  # noqa: PLC0415

    if isinstance(module, nn.Sequential):
        layers = [_compile(layer) for layer in module]

        def sequential(value):
            for layer in layers:
                value = layer(value)
            return value

        return sequential
    if isinstance(module, nn.Linear):
        weight = module.weight.detach().cpu().numpy().astype(np.float64)
        bias = (
            module.bias.detach().cpu().numpy().astype(np.float64) if module.bias is not None else 0
        )
        return lambda value: value @ weight.T + bias
    if isinstance(module, nn.Conv2d):
        if module.groups != 1 or module.dilation != (1, 1) or module.padding_mode != "zeros":
            raise ValueError("Unsupported pinned convolution")
        weight = module.weight.detach().cpu().numpy().astype(np.float64)
        bias = (
            module.bias.detach().cpu().numpy().astype(np.float64) if module.bias is not None else 0
        )
        padding, stride = module.padding, module.stride
        kernel = module.kernel_size

        def convolution(value):
            value = np.pad(
                value, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1]))
            )
            windows = np.lib.stride_tricks.sliding_window_view(value, kernel, axis=(2, 3))
            windows = windows[:, :, :: stride[0], :: stride[1], :, :]
            batch, _, height, width, _, _ = windows.shape
            columns = windows.transpose(0, 2, 3, 1, 4, 5).reshape(batch * height * width, -1)
            result = columns @ weight.reshape(weight.shape[0], -1).T + bias
            return result.reshape(batch, height, width, -1).transpose(0, 3, 1, 2)

        return convolution
    if isinstance(module, nn.ReLU):
        return lambda value: np.maximum(value, 0)
    if isinstance(module, nn.Tanh):
        return np.tanh
    if isinstance(module, (nn.Dropout, nn.Identity)):
        return lambda value: value
    if isinstance(module, nn.Flatten) and module.start_dim == 1 and module.end_dim == -1:
        return lambda value: value.reshape(value.shape[0], -1)
    raise ValueError(f"Unsupported pinned actor layer: {type(module).__name__}")


class PinnedActor:
    """Deterministic float64 action mean for the admitted GridSocNav architecture."""

    def __init__(self, model):
        """Copy actor weights and reject unsupported policy architectures."""
        policy = model.policy
        extractor = policy.pi_features_extractor
        if (
            type(extractor).__name__ != "GridSocNavExtractor"
            or extractor._pedestrian_attn is not None
        ):
            raise ValueError("Pinned actor supports GridSocNav CNN/MLP without attention only")
        if policy.squash_output:
            raise ValueError("Pinned actor does not support squashed policies")
        self.space = model.observation_space
        self.action_space = model.action_space
        self.grid_key = extractor._grid_key
        self.keys = extractor._socnav_keys
        self.goal_vector = extractor._goal_vector_enabled
        self.grid = _compile(extractor.grid_extractor)
        self.social = _compile(extractor.socnav_mlp)
        self.policy = _compile(policy.mlp_extractor.policy_net)
        self.action = _compile(policy.action_net)

    def mean(self, observation):
        """Return unclipped actor means, with all network arithmetic in float64."""
        obs = {}
        for key, space in self.space.spaces.items():
            value = np.asarray(observation[key], dtype=np.float64)
            if value.shape == space.shape:
                value = value[None]
            if value.shape[1:] != space.shape:
                raise ValueError(f"Invalid pinned observation shape for {key}")
            obs[key] = value
        grid = self.grid(obs[self.grid_key])
        parts = [obs[key].reshape(grid.shape[0], -1) for key in self.keys]
        if self.goal_vector:
            delta = obs["goal_next"] - obs["robot_position"]
            heading = obs["robot_heading"].reshape(-1)
            cos_h, sin_h = np.cos(heading), np.sin(heading)
            parts.append(
                np.stack(
                    (
                        cos_h * delta[:, 0] + sin_h * delta[:, 1],
                        -sin_h * delta[:, 0] + cos_h * delta[:, 1],
                    ),
                    axis=1,
                )
            )
        social = self.social(np.concatenate(parts, axis=1))
        return self.action(self.policy(np.concatenate((grid, social), axis=1)))

    def predict(self, observation):
        """Preserve existing Box action clipping.

        Returns:
            Clipped float64 action.
        """
        return np.clip(
            self.mean(observation), self.action_space.low, self.action_space.high
        ).squeeze()
