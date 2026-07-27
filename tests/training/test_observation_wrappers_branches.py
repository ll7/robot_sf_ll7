"""Branch-coverage tests for ``robot_sf.training.observation_wrappers``.

These tests target the box reshape/repeat rules, drive/ray adapter selection,
dict-observation compatibility routes, stack inference, and env-policy space
synchronization without loading a real policy or starting training. They exercise
the documented branch contracts of the shared observation helpers using in-memory
numpy arrays, Gymnasium spaces, and lightweight fakes.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from gymnasium import spaces

from robot_sf.sensor.sensor_fusion import OBS_DRIVE_STATE, OBS_RAYS
from robot_sf.training.observation_wrappers import (
    _make_drive_state_adapter,
    _make_ray_obs_adapter,
    _reshape_box_obs,
    adapt_dict_observation_to_policy_space,
    resolve_policy_obs_adapter,
    resolve_policy_stack_steps,
    sync_policy_spaces,
)


def _box(*shape: int, low: float = 0.0, high: float = 1.0) -> spaces.Box:
    """Build a float32 Box space with the requested shape.

    Positional args are collected into the shape tuple, so ``_box(4)`` yields a
    1D space of shape ``(4,)`` and ``_box(4, 5)`` a 2D ``(4, 5)`` space.
    """
    return spaces.Box(low=low, high=high, shape=shape, dtype=np.float32)


def _identity_adapter(obs):
    """Return the observation unchanged; a stand-in fallback adapter for selection tests."""
    return obs


# --------------------------------------------------------------------------- #
# _reshape_box_obs: exact / squeeze / repeat / compatible / passthrough routes
# --------------------------------------------------------------------------- #


def test_reshape_box_obs_returns_same_array_on_exact_shape_match():
    """An observation whose shape already matches should be returned unchanged."""
    obs = np.arange(5, dtype=np.float32)

    result = _reshape_box_obs(obs, (5,))

    assert result is obs
    assert result.shape == (5,)


def test_reshape_box_obs_squeezes_leading_singleton_for_1d_expected():
    """A ``(1, features)`` input should be squeezed to ``(features,)`` for a 1D target."""
    obs = np.arange(1, 6, dtype=np.float32).reshape(1, 5)

    result = _reshape_box_obs(obs, (5,))

    assert result.shape == (5,)
    assert np.array_equal(result, np.arange(1, 6, dtype=np.float32))


def test_reshape_box_obs_1d_expected_passes_through_2d_multirow_unchanged():
    """A ``(rows, features)`` input with ``rows != 1`` is left unchanged for a 1D target."""
    obs = np.full((3, 5), 7.0, dtype=np.float32)

    result = _reshape_box_obs(obs, (5,))

    assert result is obs
    assert result.shape == (3, 5)


def test_reshape_box_obs_1d_expected_passes_through_wrong_length_1d_unchanged():
    """A 1D input with the wrong feature count is left unchanged for a 1D target."""
    obs = np.full((4,), 7.0, dtype=np.float32)

    result = _reshape_box_obs(obs, (5,))

    assert result is obs
    assert result.shape == (4,)


def test_reshape_box_obs_1d_expected_passes_through_wrong_feature_count_2d_unchanged():
    """A ``(1, features)`` input whose feature count mismatches is left unchanged."""
    obs = np.full((1, 4), 7.0, dtype=np.float32)

    result = _reshape_box_obs(obs, (5,))

    assert result is obs
    assert result.shape == (1, 4)


def test_reshape_box_obs_repeats_1d_input_to_2d_target():
    """A 1D ``(features,)`` input should be repeated along a new leading axis."""
    obs = np.arange(5, dtype=np.float32)

    result = _reshape_box_obs(obs, (4, 5))

    assert result.shape == (4, 5)
    assert np.array_equal(result, np.tile(obs, (4, 1)))


def test_reshape_box_obs_repeats_singleton_2d_input_to_2d_target():
    """A ``(1, features)`` input should be repeated to ``(stack, features)``."""
    obs = np.arange(1, 6, dtype=np.float32).reshape(1, 5)

    result = _reshape_box_obs(obs, (4, 5))

    assert result.shape == (4, 5)
    assert np.array_equal(result, np.tile(np.arange(1, 6, dtype=np.float32), (4, 1)))


def test_reshape_box_obs_2d_target_preserves_incompatible_multirow_input():
    """A ``(rows, features)`` input with ``rows != 1`` is preserved unchanged for a 2D target."""
    obs = np.full((2, 5), 3.0, dtype=np.float32)

    result = _reshape_box_obs(obs, (4, 5))

    assert result is obs
    assert result.shape == (2, 5)


def test_reshape_box_obs_2d_target_passes_through_wrong_feature_count_2d():
    """A ``(rows, features)`` input with mismatched feature count is left unchanged."""
    obs = np.full((1, 4), 2.0, dtype=np.float32)

    result = _reshape_box_obs(obs, (4, 5))

    assert result is obs
    assert result.shape == (1, 4)


def test_reshape_box_obs_higher_rank_target_passes_through_unchanged():
    """Targets whose rank is neither 1 nor 2 fall through unchanged."""
    obs = np.full((2, 3, 4), 1.0, dtype=np.float32)

    result = _reshape_box_obs(obs, (1, 3, 4))

    assert result is obs
    assert result.shape == (2, 3, 4)


# --------------------------------------------------------------------------- #
# resolve_policy_obs_adapter: drive/ray/dict selection + missing-space fallback
# --------------------------------------------------------------------------- #


def test_resolve_policy_obs_adapter_returns_none_for_none_model():
    """A missing policy model should resolve to no adapter."""
    assert resolve_policy_obs_adapter(None) is None


def test_resolve_policy_obs_adapter_missing_observation_space_returns_none_without_fallback():
    """A policy without an observation space and no fallback resolves to None."""
    assert resolve_policy_obs_adapter(SimpleNamespace(observation_space=None)) is None


def test_resolve_policy_obs_adapter_missing_observation_space_uses_fallback():
    """A missing observation space should delegate to the provided fallback adapter."""

    assert (
        resolve_policy_obs_adapter(
            SimpleNamespace(observation_space=None), fallback_adapter=_identity_adapter
        )
        is _identity_adapter
    )


def test_resolve_policy_obs_adapter_selects_drive_state_adapter():
    """A Box observation space whose last dim is 5 selects the drive-state adapter."""
    adapter = resolve_policy_obs_adapter(SimpleNamespace(observation_space=_box(4, 5)))

    assert adapter is not None
    adapted = adapter({OBS_DRIVE_STATE: np.ones(5, dtype=np.float32), OBS_RAYS: np.zeros(3)})
    assert adapted.shape == (4, 5)
    assert np.allclose(adapted, np.ones((4, 5), dtype=np.float32))


def test_resolve_policy_obs_adapter_drive_adapter_uses_exact_shape_when_present():
    """The drive adapter returns the drive_state unchanged when shapes already match."""
    adapter = resolve_policy_obs_adapter(SimpleNamespace(observation_space=_box(4, 5)))

    drive = np.full((4, 5), 2.0, dtype=np.float32)
    adapted = adapter({OBS_DRIVE_STATE: drive, OBS_RAYS: np.zeros(3)})

    assert np.array_equal(adapted, drive)


def test_resolve_policy_obs_adapter_selects_ray_adapter():
    """A Box observation space whose last dim is 272 selects the ray adapter."""
    adapter = resolve_policy_obs_adapter(SimpleNamespace(observation_space=_box(4, 272)))

    assert adapter is not None
    rays = np.full(272, 0.5, dtype=np.float32)
    adapted = adapter({OBS_DRIVE_STATE: np.zeros(3), OBS_RAYS: rays})
    assert adapted.shape == (4, 272)
    assert np.allclose(adapted, np.full((4, 272), 0.5, dtype=np.float32))


def test_resolve_policy_obs_adapter_unsupported_box_last_dim_uses_fallback():
    """A Box space whose last dim is neither 5 nor 272 delegates to the fallback."""

    assert (
        resolve_policy_obs_adapter(
            SimpleNamespace(observation_space=_box(4, 7)), fallback_adapter=_identity_adapter
        )
        is _identity_adapter
    )


def test_resolve_policy_obs_adapter_unsupported_box_last_dim_returns_none_without_fallback():
    """An unsupported Box space with no fallback resolves to None."""
    assert resolve_policy_obs_adapter(SimpleNamespace(observation_space=_box(4, 7))) is None


def test_resolve_policy_obs_adapter_unsupported_space_type_uses_fallback():
    """A non-Box/non-Dict space (e.g. Discrete) delegates to the fallback."""

    assert (
        resolve_policy_obs_adapter(
            SimpleNamespace(observation_space=spaces.Discrete(3)),
            fallback_adapter=_identity_adapter,
        )
        is _identity_adapter
    )


def test_resolve_policy_obs_adapter_dict_space_returns_alignment_adapter():
    """A Dict observation space should select the dict-key alignment adapter."""
    adapter = resolve_policy_obs_adapter(
        SimpleNamespace(observation_space=spaces.Dict({"robot_speed": _box(2)}))
    )

    assert adapter is not None
    adapted = adapter({"robot_velocity_xy": [0.3, -0.2], "robot_heading": [0.0]})
    assert set(adapted) == {"robot_speed"}
    assert np.allclose(adapted["robot_speed"], [0.3, -0.2])


# --------------------------------------------------------------------------- #
# Drive/ray adapter factories: direct reshape behavior
# --------------------------------------------------------------------------- #


def test_drive_state_adapter_squeezes_singleton_batch():
    """The drive adapter squeezes a ``(1, features)`` batch to the expected 1D shape."""
    adapter = _make_drive_state_adapter((5,))

    adapted = adapter({OBS_DRIVE_STATE: np.arange(1, 6, dtype=np.float32).reshape(1, 5)})

    assert adapted.shape == (5,)
    assert np.array_equal(adapted, np.arange(1, 6, dtype=np.float32))


def test_ray_adapter_repeats_single_timestep_to_stack():
    """The ray adapter repeats a single ray scan to fill the declared stack dimension."""
    adapter = _make_ray_obs_adapter((4, 272))

    adapted = adapter({OBS_RAYS: np.full(272, 0.25, dtype=np.float32)})

    assert adapted.shape == (4, 272)
    assert np.allclose(adapted, np.full((4, 272), 0.25, dtype=np.float32))


# --------------------------------------------------------------------------- #
# adapt_dict_observation_to_policy_space: passthrough, alias, reshape, errors
# --------------------------------------------------------------------------- #


def test_adapt_dict_observation_passthrough_when_policy_is_none():
    """A None policy model should return the observation payload unchanged."""
    obs = {"robot_speed": [0.1, 0.2]}

    assert adapt_dict_observation_to_policy_space(obs, None) is obs


def test_adapt_dict_observation_passthrough_when_policy_space_is_not_dict():
    """A non-Dict policy observation space should return the payload unchanged."""
    obs = {"robot_speed": [0.1, 0.2]}
    policy = SimpleNamespace(observation_space=_box(3))

    assert adapt_dict_observation_to_policy_space(obs, policy) is obs


def test_adapt_dict_observation_compatible_reshape_matches_target_shape():
    """A subspace-declared reshape should be applied when element counts agree."""
    policy = SimpleNamespace(observation_space=spaces.Dict({"goal": _box(2, 2)}))

    adapted = adapt_dict_observation_to_policy_space({"goal": [1.0, 2.0, 3.0, 4.0]}, policy)

    assert adapted["goal"].shape == (2, 2)
    assert np.array_equal(adapted["goal"], np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))


def test_adapt_dict_observation_shape_mismatch_raises_value_error():
    """An incompatible element count for a declared shape should raise ValueError."""
    policy = SimpleNamespace(observation_space=spaces.Dict({"goal": _box(2, 2)}))

    with pytest.raises(ValueError, match="shape mismatch"):
        adapt_dict_observation_to_policy_space({"goal": [1.0, 2.0, 3.0, 4.0, 5.0]}, policy)


def test_adapt_dict_observation_missing_key_raises_value_error():
    """A subspace key with no value and no alias should raise a missing-key ValueError."""
    policy = SimpleNamespace(observation_space=spaces.Dict({"a": _box(2), "b": _box(2)}))

    with pytest.raises(ValueError, match="Missing required dict observation keys"):
        adapt_dict_observation_to_policy_space({"a": [1.0, 2.0]}, policy)


def test_adapt_dict_observation_backfills_key_from_alias_with_subspace_dtype():
    """The compatibility alias should populate the declared key at the subspace dtype."""
    policy = SimpleNamespace(observation_space=spaces.Dict({"robot_speed": _box(2)}))

    adapted = adapt_dict_observation_to_policy_space({"robot_velocity_xy": [0.3, -0.2]}, policy)

    assert adapted["robot_speed"].dtype == np.float32
    assert np.allclose(adapted["robot_speed"], [0.3, -0.2])


def test_adapt_dict_observation_drops_keys_not_declared_by_policy():
    """Payload keys absent from the policy space should be filtered out."""
    policy = SimpleNamespace(
        observation_space=spaces.Dict({"robot_speed": _box(2), "goal": _box(2)})
    )

    adapted = adapt_dict_observation_to_policy_space(
        {"robot_speed": [0.1, 0.2], "goal": [1.0, 2.0], "extra": [9.9]}, policy
    )

    assert set(adapted) == {"robot_speed", "goal"}
    assert adapted["robot_speed"].dtype == np.float32


# --------------------------------------------------------------------------- #
# resolve_policy_stack_steps: Box and Dict candidates
# --------------------------------------------------------------------------- #


def test_resolve_policy_stack_steps_returns_none_for_none_model():
    """A None policy model resolves to no stack steps."""
    assert resolve_policy_stack_steps(None) is None


def test_resolve_policy_stack_steps_returns_none_when_observation_space_missing():
    """A policy without an observation space resolves to no stack steps."""
    assert resolve_policy_stack_steps(SimpleNamespace(observation_space=None)) is None


def test_resolve_policy_stack_steps_box_prefers_leading_dimension():
    """A multi-dimensional Box observation space uses its leading dimension as the stack."""
    assert resolve_policy_stack_steps(SimpleNamespace(observation_space=_box(4, 5))) == 4


def test_resolve_policy_stack_steps_box_1d_returns_none():
    """A 1D Box observation space carries no stack dimension."""
    assert resolve_policy_stack_steps(SimpleNamespace(observation_space=_box(5))) is None


def test_resolve_policy_stack_steps_dict_prefers_drive_state_then_rays():
    """A Dict space should prefer drive_state, then rays, for the stack dimension."""
    policy = SimpleNamespace(
        observation_space=spaces.Dict({OBS_DRIVE_STATE: _box(7, 5), OBS_RAYS: _box(3, 272)})
    )

    assert resolve_policy_stack_steps(policy) == 7


def test_resolve_policy_stack_steps_dict_falls_back_to_rays_without_drive_state():
    """When drive_state is absent, the rays subspace should provide the stack dimension."""
    policy = SimpleNamespace(observation_space=spaces.Dict({OBS_RAYS: _box(3, 272)}))

    assert resolve_policy_stack_steps(policy) == 3


def test_resolve_policy_stack_steps_dict_falls_through_scalar_priority_to_rays():
    """A scalar drive_state subspace yields no stack; rays should then be consulted."""
    policy = SimpleNamespace(
        observation_space=spaces.Dict({OBS_DRIVE_STATE: _box(), OBS_RAYS: _box(3, 272)})
    )

    assert resolve_policy_stack_steps(policy) == 3


def test_resolve_policy_stack_steps_dict_uses_any_subspace_when_priority_keys_absent():
    """A Dict space without priority keys should fall back to any subspace."""
    policy = SimpleNamespace(observation_space=spaces.Dict({"other": _box(9, 2)}))

    assert resolve_policy_stack_steps(policy) == 9


def test_resolve_policy_stack_steps_empty_dict_returns_none():
    """An empty Dict observation space resolves to no stack dimension."""
    assert resolve_policy_stack_steps(SimpleNamespace(observation_space=spaces.Dict({}))) is None


def test_resolve_policy_stack_steps_unsupported_space_returns_none():
    """A non-Box/non-Dict observation space resolves to no stack dimension."""
    assert resolve_policy_stack_steps(SimpleNamespace(observation_space=spaces.Discrete(3))) is None


# --------------------------------------------------------------------------- #
# sync_policy_spaces: missing/present observation and action spaces
# --------------------------------------------------------------------------- #


class _FakeEnv:
    """Minimal env stand-in exposing mutable observation/action space slots."""

    def __init__(self) -> None:
        """Initialize both space slots to None."""
        self.observation_space = None
        self.action_space = None


def test_sync_policy_spaces_noop_for_none_policy():
    """A None policy model should leave env spaces untouched."""
    env = _FakeEnv()
    env.observation_space = "keep-obs"
    env.action_space = "keep-action"

    sync_policy_spaces(env, None)

    assert env.observation_space == "keep-obs"
    assert env.action_space == "keep-action"


def test_sync_policy_spaces_assigns_both_spaces_when_present():
    """Both observation and action spaces should be propagated when declared on the policy."""
    env = _FakeEnv()
    policy = SimpleNamespace(observation_space="policy-obs", action_space="policy-action")

    sync_policy_spaces(env, policy)

    assert env.observation_space == "policy-obs"
    assert env.action_space == "policy-action"


def test_sync_policy_spaces_skips_missing_observation_and_action_spaces():
    """Absent spaces on the policy should leave the existing env slots untouched."""
    env = _FakeEnv()
    env.observation_space = "keep-obs"
    env.action_space = "keep-action"

    sync_policy_spaces(env, SimpleNamespace())

    assert env.observation_space == "keep-obs"
    assert env.action_space == "keep-action"


def test_sync_policy_spaces_assigns_observation_space_only_when_action_absent():
    """Only the observation space should be updated when the policy declares no action space."""
    env = _FakeEnv()
    env.action_space = "keep-action"
    policy = SimpleNamespace(observation_space="policy-obs")

    sync_policy_spaces(env, policy)

    assert env.observation_space == "policy-obs"
    assert env.action_space == "keep-action"


def test_sync_policy_spaces_assigns_action_space_only_when_observation_absent():
    """Only the action space should be updated when the policy declares no observation space."""
    env = _FakeEnv()
    env.observation_space = "keep-obs"
    policy = SimpleNamespace(action_space="policy-action")

    sync_policy_spaces(env, policy)

    assert env.observation_space == "keep-obs"
    assert env.action_space == "policy-action"
