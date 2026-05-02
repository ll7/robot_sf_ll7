"""Tests for PPO policy variants."""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

from robot_sf.feature_extractors.grid_socnav_extractor import GridSocNavExtractor
from robot_sf.training.ppo_policy import AsymmetricGridSocNavPolicy


def _make_policy_spaces() -> tuple[spaces.Dict, spaces.Box]:
    """Create a compact grid SocNav observation/action space for policy construction."""
    observation_space = spaces.Dict(
        {
            "occupancy_grid": spaces.Box(low=0.0, high=1.0, shape=(3, 32, 32), dtype=np.float32),
            "robot_position": spaces.Box(low=0.0, high=10.0, shape=(2,), dtype=np.float32),
            "robot_heading": spaces.Box(low=-3.14, high=3.14, shape=(1,), dtype=np.float32),
            "goal_next": spaces.Box(low=0.0, high=10.0, shape=(2,), dtype=np.float32),
            "critic_privileged_state": spaces.Box(
                low=-10.0, high=10.0, shape=(3,), dtype=np.float32
            ),
        }
    )
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    return observation_space, action_space


def _make_policy(*, asymmetric_critic: bool) -> AsymmetricGridSocNavPolicy:
    """Construct the policy with a small extractor/network for fast unit tests."""
    observation_space, action_space = _make_policy_spaces()
    return AsymmetricGridSocNavPolicy(
        observation_space=observation_space,
        action_space=action_space,
        lr_schedule=lambda _progress_remaining: 0.001,
        net_arch=[8],
        features_extractor_class=GridSocNavExtractor,
        features_extractor_kwargs={
            "grid_channels": [4],
            "grid_kernel_sizes": [3],
            "socnav_hidden_dims": [8],
        },
        asymmetric_critic=asymmetric_critic,
    )


def test_asymmetric_grid_socnav_policy_separates_actor_and_critic_extractors() -> None:
    """Asymmetric mode should hide privileged state from actor and expose it to critic."""
    policy = _make_policy(asymmetric_critic=True)

    assert policy.features_extractor is not policy.vf_features_extractor
    assert "critic_privileged_state" not in policy.features_extractor._socnav_keys
    assert "critic_privileged_state" in policy.vf_features_extractor._socnav_keys

    constructor_params = policy._get_constructor_parameters()
    assert constructor_params["asymmetric_critic"] is True
    assert constructor_params["critic_features_extractor_kwargs"] == {}


def test_grid_socnav_policy_uses_shared_extractor_without_asymmetric_critic() -> None:
    """Standard mode should keep SB3's shared actor-critic extractor behavior."""
    policy = _make_policy(asymmetric_critic=False)

    assert policy.features_extractor is policy.vf_features_extractor
    assert policy._get_constructor_parameters()["asymmetric_critic"] is False
