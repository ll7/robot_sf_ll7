"""Tests for the grid + SocNav feature extractor."""

from __future__ import annotations

import numpy as np
import torch as th
from gymnasium import spaces

from robot_sf.feature_extractors.grid_socnav_extractor import GridSocNavExtractor


def _make_obs_dict(space: spaces.Dict, batch: int = 2) -> dict:
    obs = {}
    for key, subspace in space.spaces.items():
        sample = subspace.sample()
        tensor = th.as_tensor(sample, dtype=th.float32)
        if tensor.ndim == 0:
            tensor = tensor.view(1)
        tensor = tensor.unsqueeze(0)
        obs[key] = tensor.repeat(batch, *([1] * (tensor.ndim - 1)))
    return obs


def test_grid_socnav_extractor_forward_shape() -> None:
    """GridSocNavExtractor produces a feature vector per batch entry."""
    obs_space = spaces.Dict(
        {
            "occupancy_grid": spaces.Box(low=0.0, high=1.0, shape=(3, 64, 64), dtype=np.float32),
            "robot_position": spaces.Box(low=0.0, high=10.0, shape=(2,), dtype=np.float32),
            "robot_heading": spaces.Box(low=-3.14, high=3.14, shape=(1,), dtype=np.float32),
            "goal_next": spaces.Box(low=0.0, high=10.0, shape=(2,), dtype=np.float32),
            "pedestrians_positions": spaces.Box(low=0.0, high=10.0, shape=(4, 2), dtype=np.float32),
            "occupancy_grid_meta_resolution": spaces.Box(
                low=0.0, high=1.0, shape=(1,), dtype=np.float32
            ),
        }
    )
    extractor = GridSocNavExtractor(obs_space)
    obs = _make_obs_dict(obs_space, batch=2)
    features = extractor(obs)
    assert features.shape[0] == 2
    assert features.shape[1] == extractor.features_dim


def test_grid_socnav_extractor_supports_explicit_socnav_key_subset() -> None:
    """The extractor should allow config-driven SocNav input ablations."""
    obs_space = spaces.Dict(
        {
            "occupancy_grid": spaces.Box(low=0.0, high=1.0, shape=(3, 64, 64), dtype=np.float32),
            "robot_position": spaces.Box(low=0.0, high=10.0, shape=(2,), dtype=np.float32),
            "robot_heading": spaces.Box(low=-3.14, high=3.14, shape=(1,), dtype=np.float32),
            "goal_next": spaces.Box(low=0.0, high=10.0, shape=(2,), dtype=np.float32),
            "robot_speed": spaces.Box(low=0.0, high=5.0, shape=(1,), dtype=np.float32),
            "pedestrians_positions": spaces.Box(low=0.0, high=10.0, shape=(4, 2), dtype=np.float32),
            "pedestrians_velocities": spaces.Box(
                low=-5.0, high=5.0, shape=(4, 2), dtype=np.float32
            ),
        }
    )
    extractor = GridSocNavExtractor(
        obs_space,
        include_socnav_keys=["robot_speed"],
        include_ego_goal_vector=True,
    )

    assert extractor._socnav_keys == ("robot_speed",)

    obs = _make_obs_dict(obs_space, batch=2)
    features = extractor(obs)
    assert features.shape[0] == 2
    assert features.shape[1] == extractor.features_dim


def test_grid_socnav_extractor_can_include_privileged_state_for_critic() -> None:
    """The extractor should optionally consume critic-only privileged state."""
    obs_space = spaces.Dict(
        {
            "occupancy_grid": spaces.Box(low=0.0, high=1.0, shape=(3, 32, 32), dtype=np.float32),
            "robot_position": spaces.Box(low=0.0, high=10.0, shape=(2,), dtype=np.float32),
            "robot_heading": spaces.Box(low=-3.14, high=3.14, shape=(1,), dtype=np.float32),
            "goal_next": spaces.Box(low=0.0, high=10.0, shape=(2,), dtype=np.float32),
            "critic_privileged_state": spaces.Box(
                low=-10.0, high=10.0, shape=(8,), dtype=np.float32
            ),
        }
    )

    actor_extractor = GridSocNavExtractor(
        obs_space,
        privileged_state_key="critic_privileged_state",
        include_privileged_state=False,
    )
    critic_extractor = GridSocNavExtractor(
        obs_space,
        privileged_state_key="critic_privileged_state",
        include_privileged_state=True,
    )

    assert "critic_privileged_state" not in actor_extractor._socnav_keys
    assert "critic_privileged_state" in critic_extractor._socnav_keys

    obs = _make_obs_dict(obs_space, batch=2)
    actor_features = actor_extractor(obs)
    critic_features = critic_extractor(obs)
    assert actor_features.shape == critic_features.shape


def _make_ped_obs_space(max_peds: int = 4) -> spaces.Dict:
    return spaces.Dict(
        {
            "occupancy_grid": spaces.Box(low=0.0, high=1.0, shape=(3, 32, 32), dtype=np.float32),
            "robot_position": spaces.Box(low=0.0, high=10.0, shape=(2,), dtype=np.float32),
            "robot_heading": spaces.Box(low=-3.14, high=3.14, shape=(1,), dtype=np.float32),
            "goal_next": spaces.Box(low=0.0, high=10.0, shape=(2,), dtype=np.float32),
            "pedestrians_positions": spaces.Box(
                low=0.0, high=50.0, shape=(max_peds, 2), dtype=np.float32
            ),
            "pedestrians_velocities": spaces.Box(
                low=-5.0, high=5.0, shape=(max_peds, 2), dtype=np.float32
            ),
            "pedestrians_count": spaces.Box(
                low=0.0, high=float(max_peds), shape=(1,), dtype=np.float32
            ),
        }
    )


def test_pedestrian_attention_head_produces_correct_output_shape() -> None:
    """PedestrianAttentionHead maps (batch, max_peds, slot_dim) -> (batch, output_dim)."""
    from robot_sf.feature_extractors.grid_socnav_extractor import PedestrianAttentionHead

    head = PedestrianAttentionHead(slot_input_dim=4, d_model=32, num_heads=4, output_dim=16)
    batch, max_peds = 3, 8
    slot_feats = th.rand(batch, max_peds, 4)
    count = th.tensor([[2.0], [5.0], [0.0]])
    out = head(slot_feats, count)
    assert out.shape == (batch, 16)
    assert th.isfinite(out).all()


def test_pedestrian_attention_head_masks_padding_slots() -> None:
    """Attention output should differ between count=1 and count=8 for the same slots."""
    from robot_sf.feature_extractors.grid_socnav_extractor import PedestrianAttentionHead

    th.manual_seed(0)
    head = PedestrianAttentionHead(slot_input_dim=4, d_model=32, num_heads=4, output_dim=16)
    head.eval()
    slot_feats = th.rand(1, 8, 4)
    out_few = head(slot_feats, th.tensor([[1.0]]))
    out_many = head(slot_feats, th.tensor([[8.0]]))
    # Masking changes the pooled result — outputs should not be identical
    assert not th.allclose(out_few, out_many)


def test_grid_socnav_extractor_with_pedestrian_attention_forward_shape() -> None:
    """GridSocNavExtractor with attention head produces correct feature dim."""
    obs_space = _make_ped_obs_space(max_peds=4)
    extractor = GridSocNavExtractor(
        obs_space,
        use_pedestrian_attention=True,
        attn_d_model=32,
        attn_num_heads=4,
        attn_output_dim=16,
    )
    obs = _make_obs_dict(obs_space, batch=2)
    features = extractor(obs)
    assert features.shape[0] == 2
    assert features.shape[1] == extractor.features_dim


def test_grid_socnav_extractor_attention_stays_finite_with_zero_pedestrians() -> None:
    """Zero-pedestrian batches must not yield NaNs in the attention-enabled extractor."""
    obs_space = _make_ped_obs_space(max_peds=4)
    extractor = GridSocNavExtractor(
        obs_space,
        use_pedestrian_attention=True,
        attn_d_model=32,
        attn_num_heads=4,
        attn_output_dim=16,
    )
    obs = _make_obs_dict(obs_space, batch=2)
    obs["pedestrians_positions"] = th.zeros_like(obs["pedestrians_positions"])
    obs["pedestrians_velocities"] = th.zeros_like(obs["pedestrians_velocities"])
    obs["pedestrians_count"] = th.zeros_like(obs["pedestrians_count"])

    features = extractor(obs)

    assert th.isfinite(features).all()


def test_grid_socnav_extractor_attention_removes_slot_keys_from_mlp_path() -> None:
    """Slot keys must not appear in _socnav_keys when attention is enabled."""
    obs_space = _make_ped_obs_space(max_peds=4)
    extractor = GridSocNavExtractor(obs_space, use_pedestrian_attention=True)
    assert "pedestrians_positions" not in extractor._socnav_keys
    assert "pedestrians_velocities" not in extractor._socnav_keys
    # Count stays in the flat MLP path
    assert "pedestrians_count" in extractor._socnav_keys


def test_grid_socnav_extractor_attention_fails_closed_without_slot_keys() -> None:
    """use_pedestrian_attention=True with no slot keys in obs space raises ValueError."""
    obs_space = spaces.Dict(
        {
            "occupancy_grid": spaces.Box(low=0.0, high=1.0, shape=(3, 32, 32), dtype=np.float32),
            "robot_position": spaces.Box(low=0.0, high=10.0, shape=(2,), dtype=np.float32),
            "robot_heading": spaces.Box(low=-3.14, high=3.14, shape=(1,), dtype=np.float32),
            "goal_next": spaces.Box(low=0.0, high=10.0, shape=(2,), dtype=np.float32),
        }
    )
    try:
        GridSocNavExtractor(obs_space, use_pedestrian_attention=True)
    except ValueError as exc:
        assert "pedestrian slot keys" in str(exc)
    else:
        raise AssertionError("Expected ValueError when slot keys are absent.")


def test_grid_socnav_extractor_rejects_unknown_included_socnav_keys() -> None:
    """Misconfigured include lists should fail closed at construction time."""
    obs_space = spaces.Dict(
        {
            "occupancy_grid": spaces.Box(low=0.0, high=1.0, shape=(3, 32, 32), dtype=np.float32),
            "robot_position": spaces.Box(low=0.0, high=10.0, shape=(2,), dtype=np.float32),
            "robot_heading": spaces.Box(low=-3.14, high=3.14, shape=(1,), dtype=np.float32),
            "goal_next": spaces.Box(low=0.0, high=10.0, shape=(2,), dtype=np.float32),
        }
    )

    try:
        GridSocNavExtractor(obs_space, include_socnav_keys=["missing_key"])
    except ValueError as exc:
        assert "unknown observation keys" in str(exc)
    else:
        raise AssertionError("Expected GridSocNavExtractor to reject unknown include keys.")
