"""Contract tests for the Open Dreamer-style latent dynamics model (issue #6318 Step 3 slice).

These tests pin the **Step 3 model module slice** of parent #6318's maintainer-authorized sequenced
plan. They verify that the clean-room, compute-free latent dynamics model under
:mod:`robot_sf.research.open_dreamer_dynamics`:

* consumes the merged Step 2 adapter's :class:`StructuredEpisode` **read-only** (the adapter module
  is never edited here);
* encodes a structured observation into a finite, bounded latent state;
* applies an action-conditioned latent transition that produces finite next-latent predictions;
* exposes finite reward and continuation heads, with continuation strictly inside ``(0, 1)``;
* imagines an episode-major open-loop rollout whose every output is finite;
* is deterministic for a fixed seed and imports no ML framework (pure NumPy, compute-free);
* and **fails closed** on non-finite or mis-shaped inputs/outputs, config/weight mismatches, and
  episodes whose observation width does not match the model.

This is diagnostic/contract evidence only (``evidence_tier: idea``): the tests make no benchmark,
metric, or policy claim, and the weights are untrained.
"""

from __future__ import annotations

import ast
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import robot_sf.research.open_dreamer_dynamics as dynamics_module
from robot_sf.research.open_dreamer_adapter import (
    DRIVE_STATE_LAYOUT,
    OPEN_DREAMER_OBSERVATION_CONTRACT,
    StructuredActionStep,
    StructuredEpisode,
    StructuredObservationStep,
)
from robot_sf.research.open_dreamer_dynamics import (
    ACTION_DIM,
    DEFAULT_LATENT_DIM,
    DYNAMICS_PROVENANCE_KEY,
    EVIDENCE_BOUNDARY,
    OPEN_DREAMER_DYNAMICS_VERSION,
    DynamicsConfig,
    DynamicsStep,
    DynamicsWeights,
    LatentDynamicsModel,
    LatentRollout,
    OpenDreamerDynamicsError,
)

_BASE_OBS_DIM = len(DRIVE_STATE_LAYOUT)


def _drive_state(index: int) -> np.ndarray:
    """Return a finite drive_state vector in the adapter's documented component order."""
    return np.asarray(
        [float(index), float(index + 1), 0.5 * float(index), 0.25 * float(index), 0.0],
        dtype=float,
    )


def _make_structured_episode(
    *,
    step_count: int = 3,
    with_rays: bool = False,
    ray_width: int = 3,
) -> StructuredEpisode:
    """Build a contract-valid structured episode directly from the adapter's public types.

    Args:
        step_count: Number of episode steps.
        with_rays: When true, give every step a fixed-width finite rays group and flip episode-wide
            ray availability on.
        ray_width: Width of the rays group when ``with_rays`` is true.

    Returns:
        A :class:`StructuredEpisode` the dynamics model can consume read-only.
    """
    observations = tuple(
        StructuredObservationStep(
            drive_state=_drive_state(index),
            rays=(
                np.asarray([0.5 + 0.1 * float(index)] * ray_width, dtype=float)
                if with_rays
                else np.asarray([], dtype=float)
            ),
            rays_available=with_rays,
        )
        for index in range(step_count)
    )
    actions = tuple(
        StructuredActionStep(raw=(0.5, 0.1 * float(index))) for index in range(step_count)
    )
    rewards = tuple(float(index + 1) for index in range(step_count))
    total = sum(rewards)
    return StructuredEpisode(
        dataset_id="issue_6394_dynamics_smoke",
        episode_id="classic_cross_trap_low:seed101:goal:000000",
        scenario_id="classic_cross_trap_low",
        seed=101,
        source_policy_id="goal",
        split="train",
        observations=observations,
        raw_observations=tuple({} for _ in range(step_count)),
        actions=actions,
        raw_actions=tuple((0.5, 0.1 * float(index)) for index in range(step_count)),
        rewards=rewards,
        return_to_go=tuple(total - sum(rewards[:index]) for index in range(step_count)),
        terminated=tuple(index == step_count - 1 for index in range(step_count)),
        truncated=tuple(False for _ in range(step_count)),
        pedestrians=tuple([] for _ in range(step_count)),
        robot_states=tuple({} for _ in range(step_count)),
        provenance={"source": "unit_test"},
        rays_available=with_rays,
    )


def _default_model(*, with_rays: bool = False) -> LatentDynamicsModel:
    """Build a model matched to the default fixture episode's observation width."""
    return LatentDynamicsModel.from_episode(
        _make_structured_episode(with_rays=with_rays),
        seed=7,
    )


# ----------------------------------------------------------------------------------------------
# Positive contract: finite, bounded, deterministic, episode-major, provenance.
# ----------------------------------------------------------------------------------------------


def test_config_to_dict_round_trips_dimensions() -> None:
    """The config records positive integer dimensions and the seed as a JSON-safe mapping."""
    config = DynamicsConfig(obs_dim=_BASE_OBS_DIM, latent_dim=8, seed=3)
    assert config.to_dict() == {
        "obs_dim": _BASE_OBS_DIM,
        "action_dim": ACTION_DIM,
        "latent_dim": 8,
        "seed": 3,
    }


def test_weights_from_config_are_finite_shaped_and_deterministic() -> None:
    """Deterministic initialization yields finite, correctly shaped weights for a fixed seed."""
    config = DynamicsConfig(obs_dim=_BASE_OBS_DIM, latent_dim=DEFAULT_LATENT_DIM, seed=11)
    weights_a = DynamicsWeights.from_config(config)
    weights_b = DynamicsWeights.from_config(config)

    assert weights_a.w_enc.shape == (DEFAULT_LATENT_DIM, _BASE_OBS_DIM)
    assert weights_a.b_enc.shape == (DEFAULT_LATENT_DIM,)
    assert weights_a.w_latent.shape == (DEFAULT_LATENT_DIM, DEFAULT_LATENT_DIM)
    assert weights_a.w_action.shape == (DEFAULT_LATENT_DIM, ACTION_DIM)
    assert weights_a.b_latent.shape == (DEFAULT_LATENT_DIM,)
    assert weights_a.w_reward.shape == (DEFAULT_LATENT_DIM,)
    assert weights_a.w_cont.shape == (DEFAULT_LATENT_DIM,)
    for array in (
        weights_a.w_enc,
        weights_a.b_enc,
        weights_a.w_latent,
        weights_a.w_action,
        weights_a.b_latent,
        weights_a.w_reward,
        weights_a.w_cont,
    ):
        assert np.all(np.isfinite(array))
    # Same seed -> identical bundle.
    np.testing.assert_array_equal(weights_a.w_enc, weights_b.w_enc)
    np.testing.assert_array_equal(weights_a.w_latent, weights_b.w_latent)


def test_encode_produces_finite_bounded_latent() -> None:
    """The encoder maps a finite observation to a finite latent strictly inside (-1, 1)."""
    model = _default_model()
    latent = model.encode(_drive_state(0))

    assert latent.shape == (DEFAULT_LATENT_DIM,)
    assert np.all(np.isfinite(latent))
    assert np.all(latent > -1.0) and np.all(latent < 1.0)


def test_reward_head_is_finite_scalar() -> None:
    """The reward head returns a finite scalar from a finite latent state."""
    model = _default_model()
    latent = model.encode(_drive_state(1))
    reward = model.reward_head(latent)

    assert isinstance(reward, float)
    assert np.isfinite(reward)


def test_continuation_head_is_strictly_inside_unit_interval() -> None:
    """The continuation head returns a finite probability strictly inside (0, 1)."""
    model = _default_model()
    latent = model.encode(_drive_state(2))
    continuation = model.continuation_head(latent)

    assert isinstance(continuation, float)
    assert np.isfinite(continuation)
    assert 0.0 < continuation < 1.0


@pytest.mark.parametrize("bias", [-1000.0, 1000.0])
def test_continuation_head_stays_open_interval_when_float_sigmoid_saturates(bias: float) -> None:
    """Finite extreme logits preserve the promised open probability interval."""
    config = DynamicsConfig(obs_dim=_BASE_OBS_DIM)
    weights = replace(DynamicsWeights.from_config(config), b_cont=bias)
    model = LatentDynamicsModel(config, weights)

    continuation = model.continuation_head(np.zeros(config.latent_dim, dtype=float))

    assert 0.0 < continuation < 1.0


def test_encoder_stays_open_interval_when_float_tanh_saturates() -> None:
    """Finite extreme pre-activations preserve the promised open latent interval."""
    config = DynamicsConfig(obs_dim=_BASE_OBS_DIM)
    weights = DynamicsWeights.from_config(config)
    saturated_weights = replace(
        weights,
        w_enc=np.zeros_like(weights.w_enc),
        b_enc=np.full(config.latent_dim, 1000.0, dtype=float),
    )
    model = LatentDynamicsModel(config, saturated_weights)

    latent = model.encode(np.zeros(config.obs_dim, dtype=float))

    assert np.all(latent > -1.0)
    assert np.all(latent < 1.0)


def test_step_produces_finite_next_latent_and_both_heads() -> None:
    """One action-conditioned transition yields a finite bounded next latent and finite heads."""
    model = _default_model()
    latent = model.encode(_drive_state(0))
    transition = model.step(latent, np.asarray([0.5, 0.1], dtype=float))

    assert isinstance(transition, DynamicsStep)
    assert transition.latent.shape == (DEFAULT_LATENT_DIM,)
    assert np.all(np.isfinite(transition.latent))
    assert np.all(transition.latent > -1.0) and np.all(transition.latent < 1.0)
    assert np.isfinite(transition.reward)
    assert 0.0 < transition.continuation < 1.0


def test_imagine_consumes_structured_episode_read_only() -> None:
    """Imagine a rollout over an episode without mutating its structured observation arrays."""
    episode = _make_structured_episode(step_count=4)
    before = np.stack([step.drive_state.copy() for step in episode.observations])
    model = LatentDynamicsModel.from_episode(episode, seed=7)

    rollout = model.imagine(episode)

    assert rollout.step_count == episode.step_count
    assert rollout.latents.shape == (episode.step_count + 1, DEFAULT_LATENT_DIM)
    assert rollout.rewards.shape == (episode.step_count,)
    assert rollout.continuations.shape == (episode.step_count,)
    # The episode's observation arrays are unchanged and remain read-only.
    after = np.stack([step.drive_state.copy() for step in episode.observations])
    np.testing.assert_array_equal(before, after)
    for step in episode.observations:
        assert not step.drive_state.flags.writeable


def test_imagine_rollout_is_everywhere_finite() -> None:
    """The full imagined rollout has no NaN or Inf in latents, rewards, or continuations."""
    episode = _make_structured_episode(step_count=6)
    model = LatentDynamicsModel.from_episode(episode, seed=7)

    rollout = model.imagine(episode)

    assert np.all(np.isfinite(rollout.latents))
    assert np.all(np.isfinite(rollout.rewards))
    assert np.all(np.isfinite(rollout.continuations))
    assert np.all(rollout.continuations > 0.0)
    assert np.all(rollout.continuations < 1.0)
    assert np.all(rollout.latents > -1.0)
    assert np.all(rollout.latents < 1.0)


def test_imagine_is_deterministic_for_fixed_seed() -> None:
    """Two models built from the same seed imagine identical rollouts over the same episode."""
    episode = _make_structured_episode(step_count=5)
    rollout_a = LatentDynamicsModel.from_episode(episode, seed=13).imagine(episode)
    rollout_b = LatentDynamicsModel.from_episode(episode, seed=13).imagine(episode)

    np.testing.assert_array_equal(rollout_a.latents, rollout_b.latents)
    np.testing.assert_array_equal(rollout_a.rewards, rollout_b.rewards)
    np.testing.assert_array_equal(rollout_a.continuations, rollout_b.continuations)


def test_from_episode_derives_observation_width_from_ray_availability() -> None:
    """The observation width is the drive_state width, plus the ray width when rays are present."""
    ray_free = LatentDynamicsModel.from_episode(_make_structured_episode(with_rays=False))
    ray_full = LatentDynamicsModel.from_episode(
        _make_structured_episode(with_rays=True, ray_width=4)
    )

    assert ray_free.config.obs_dim == _BASE_OBS_DIM
    assert ray_full.config.obs_dim == _BASE_OBS_DIM + 4


def test_imagine_with_rays_consumes_concatenated_observation() -> None:
    """A rays-available episode is imagined through the widened concatenated observation view."""
    episode = _make_structured_episode(step_count=3, with_rays=True, ray_width=3)
    model = LatentDynamicsModel.from_episode(episode, seed=7)

    rollout = model.imagine(episode)

    assert model.config.obs_dim == _BASE_OBS_DIM + 3
    assert rollout.step_count == episode.step_count
    assert np.all(np.isfinite(rollout.latents))


def test_imagine_from_arrays_matches_episode_bootstrap() -> None:
    """The array-level entry point reproduces the episode rollout for the same aligned arrays."""
    episode = _make_structured_episode(step_count=4)
    model = LatentDynamicsModel.from_episode(episode, seed=7)
    observations = np.stack([step.drive_state for step in episode.observations])
    actions = np.stack([np.asarray(step.raw, dtype=float) for step in episode.actions])

    from_episode = model.imagine(episode)
    from_arrays = model.imagine_from_arrays(observations, actions)

    np.testing.assert_array_equal(from_episode.latents, from_arrays.latents)
    np.testing.assert_array_equal(from_episode.rewards, from_arrays.rewards)
    np.testing.assert_array_equal(from_episode.continuations, from_arrays.continuations)


def test_rollout_provenance_records_clean_room_idea_boundary() -> None:
    """Rollout provenance records the clean-room route, idea tier, and consumed adapter contract."""
    episode = _make_structured_episode(step_count=2)
    model = LatentDynamicsModel.from_episode(episode, seed=7)

    rollout = model.imagine(episode)
    provenance = rollout.provenance

    assert provenance["source"] == "unit_test"
    dynamics_provenance = provenance[DYNAMICS_PROVENANCE_KEY]
    assert dynamics_provenance["dynamics_version"] == OPEN_DREAMER_DYNAMICS_VERSION
    assert dynamics_provenance["evidence_boundary"] == EVIDENCE_BOUNDARY == "idea"
    assert dynamics_provenance["route"] == "clean_room"
    assert dynamics_provenance["trained"] is False
    assert dynamics_provenance["compute_free"] is True
    assert dynamics_provenance["consumed_observation_contract"] == OPEN_DREAMER_OBSERVATION_CONTRACT
    assert dynamics_provenance["dataset_id"] == episode.dataset_id
    assert dynamics_provenance["episode_id"] == episode.episode_id
    assert dynamics_provenance["source_policy_id"] == episode.source_policy_id
    assert dynamics_provenance["split"] == episode.split
    assert dynamics_provenance["step_count"] == episode.step_count


def test_rollout_provenance_is_recursively_immutable() -> None:
    """Callers cannot mutate source or nested dynamics provenance after validation."""
    rollout = _default_model().imagine(_make_structured_episode())

    with pytest.raises(TypeError):
        rollout.provenance["source"] = "changed"  # type: ignore[index]
    with pytest.raises(TypeError):
        rollout.provenance[DYNAMICS_PROVENANCE_KEY]["trained"] = True  # type: ignore[index]


def test_rollout_refuses_to_overwrite_reserved_provenance_key() -> None:
    """A source provenance collision fails closed instead of hiding prior metadata."""
    episode = _make_structured_episode()
    colliding_episode = replace(
        episode,
        provenance={DYNAMICS_PROVENANCE_KEY: {"source": "preexisting"}},
    )
    model = LatentDynamicsModel.from_episode(colliding_episode)

    with pytest.raises(OpenDreamerDynamicsError, match="reserved key"):
        model.imagine(colliding_episode)


def test_rollout_to_dict_is_json_serializable() -> None:
    """The rollout summary serializes to JSON without error and preserves the step count."""
    episode = _make_structured_episode(step_count=3)
    model = LatentDynamicsModel.from_episode(episode, seed=7)

    payload = model.imagine(episode).to_dict()
    serialized = json.dumps(payload)

    assert isinstance(serialized, str)
    assert payload["step_count"] == episode.step_count
    assert payload["latent_dim"] == DEFAULT_LATENT_DIM


def test_module_is_compute_free_with_no_ml_framework_imports() -> None:
    """The dynamics module imports only NumPy and the adapter -- no GPU/ML framework."""
    source = Path(dynamics_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden = {"torch", "jax", "flax", "tensorflow", "keras"}
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    assert imported.isdisjoint(forbidden)
    assert "numpy" in imported


# ----------------------------------------------------------------------------------------------
# Fail-closed contract: non-finite, mis-shaped, mismatched, and non-episode inputs.
# ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"obs_dim": 0}, "obs_dim must be strictly positive"),
        ({"obs_dim": -1}, "obs_dim must be strictly positive"),
        ({"latent_dim": 0}, "latent_dim must be strictly positive"),
        ({"action_dim": 0}, "action_dim must be strictly positive"),
    ],
)
def test_config_rejects_non_positive_dimensions(overrides: dict, message: str) -> None:
    """Non-positive dimensions fail closed at the config boundary."""
    kwargs = {"obs_dim": _BASE_OBS_DIM}
    kwargs.update(overrides)
    with pytest.raises(OpenDreamerDynamicsError, match=message):
        DynamicsConfig(**kwargs)


@pytest.mark.parametrize("field_name", ["obs_dim", "latent_dim", "action_dim", "seed"])
def test_config_rejects_boolean_dimensions(field_name: str) -> None:
    """Boolean values are not accepted as integer dimensions or seeds."""
    kwargs = {"obs_dim": _BASE_OBS_DIM, field_name: True}
    with pytest.raises(OpenDreamerDynamicsError, match="must be"):
        DynamicsConfig(**kwargs)


def test_config_rejects_non_robot_sf_action_width() -> None:
    """The dynamics contract cannot silently accept a non-Robot-SF action space."""
    with pytest.raises(OpenDreamerDynamicsError, match="action_dim must equal"):
        DynamicsConfig(obs_dim=_BASE_OBS_DIM, action_dim=ACTION_DIM + 1)


def test_config_rejects_negative_seed_with_contract_error() -> None:
    """Invalid NumPy seeds fail through the module's declared error boundary."""
    with pytest.raises(OpenDreamerDynamicsError, match="seed must be non-negative"):
        DynamicsConfig(obs_dim=_BASE_OBS_DIM, seed=-1)


def test_weights_reject_non_finite_array() -> None:
    """A NaN in any weight array fails closed when the bundle is constructed."""
    config = DynamicsConfig(obs_dim=_BASE_OBS_DIM, latent_dim=4, seed=1)
    weights = DynamicsWeights.from_config(config)
    bad_enc = np.array(weights.w_enc, copy=True)
    bad_enc[0, 0] = np.nan

    with pytest.raises(OpenDreamerDynamicsError, match="w_enc must contain only finite values"):
        DynamicsWeights(
            w_enc=bad_enc,
            b_enc=weights.b_enc,
            w_latent=weights.w_latent,
            w_action=weights.w_action,
            b_latent=weights.b_latent,
            w_reward=weights.w_reward,
            b_reward=weights.b_reward,
            w_cont=weights.w_cont,
            b_cont=weights.b_cont,
        )


def test_weights_reject_non_array_encoder_with_contract_error() -> None:
    """Malformed weight containers cannot leak an AttributeError."""
    weights = DynamicsWeights.from_config(DynamicsConfig(obs_dim=_BASE_OBS_DIM))

    with pytest.raises(OpenDreamerDynamicsError, match="w_enc must be a 2D NumPy ndarray"):
        replace(weights, w_enc=[])  # type: ignore[arg-type]


def test_weights_reject_non_robot_sf_action_width() -> None:
    """A standalone weight bundle cannot encode an action space outside Robot SF's contract."""
    weights = DynamicsWeights.from_config(DynamicsConfig(obs_dim=_BASE_OBS_DIM, latent_dim=4))

    with pytest.raises(OpenDreamerDynamicsError, match="action_dim must equal"):
        replace(
            weights,
            w_action=np.zeros((weights.latent_dim, ACTION_DIM + 1), dtype=float),
        )


def test_weights_reject_infinite_bias_scalar() -> None:
    """A non-finite scalar bias fails closed when the bundle is constructed."""
    config = DynamicsConfig(obs_dim=_BASE_OBS_DIM, latent_dim=4, seed=1)
    weights = DynamicsWeights.from_config(config)

    with pytest.raises(OpenDreamerDynamicsError, match="b_reward must be finite"):
        DynamicsWeights(
            w_enc=weights.w_enc,
            b_enc=weights.b_enc,
            w_latent=weights.w_latent,
            w_action=weights.w_action,
            b_latent=weights.b_latent,
            w_reward=weights.w_reward,
            b_reward=float("inf"),
            w_cont=weights.w_cont,
            b_cont=weights.b_cont,
        )


def test_model_rejects_config_weight_shape_mismatch() -> None:
    """Pairing a config with a weight bundle of a different observation width fails closed."""
    config = DynamicsConfig(obs_dim=_BASE_OBS_DIM, latent_dim=4, seed=1)
    other_weights = DynamicsWeights.from_config(
        DynamicsConfig(obs_dim=_BASE_OBS_DIM + 2, latent_dim=4, seed=1)
    )

    with pytest.raises(OpenDreamerDynamicsError, match="does not match config obs_dim"):
        LatentDynamicsModel(config, other_weights)


@pytest.mark.parametrize("field_name", ["observations", "actions"])
def test_imagine_rejects_misaligned_episode_sequences(field_name: str) -> None:
    """An episode with missing observation or action steps fails before open-loop rollout."""
    episode = _make_structured_episode(step_count=3)
    broken = replace(episode, **{field_name: getattr(episode, field_name)[:2]})
    model = _default_model()

    with pytest.raises(OpenDreamerDynamicsError, match="has 2 steps; expected 3"):
        model.imagine(broken)


def test_imagine_rejects_empty_observation_sequence_with_contract_error() -> None:
    """An episode with steps but no observations fails with the dynamics error boundary."""
    episode = _make_structured_episode(step_count=3)
    broken = replace(episode, observations=())

    with pytest.raises(OpenDreamerDynamicsError, match="observations.*expected 3"):
        _default_model().imagine(broken)


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        (
            "observation_contract",
            "other_observation_contract.v1",
            "observation_contract does not match",
        ),
        ("drive_state_layout", ("wrong",), "drive_state_layout does not match"),
    ],
)
def test_from_episode_rejects_noncanonical_observation_contract(
    field_name: str,
    value: object,
    message: str,
) -> None:
    """The dynamics model accepts only the merged adapter's fixed observation contract."""
    episode = replace(_make_structured_episode(), **{field_name: value})

    with pytest.raises(OpenDreamerDynamicsError, match=message):
        LatentDynamicsModel.from_episode(episode)


def test_imagine_rejects_non_mapping_episode_provenance() -> None:
    """Malformed source provenance cannot bypass the rollout provenance boundary."""
    episode = replace(_make_structured_episode(), provenance=[])

    with pytest.raises(OpenDreamerDynamicsError, match="provenance must be a mapping"):
        _default_model().imagine(episode)


@pytest.mark.parametrize(
    ("observation", "message"),
    [
        (np.zeros(_BASE_OBS_DIM + 1, dtype=float), "observation must have shape"),
        (np.full(_BASE_OBS_DIM, np.nan, dtype=float), "observation must contain only finite"),
        (np.ones(_BASE_OBS_DIM, dtype=bool), "observation must not contain booleans"),
        ("not-a-vector", "observation must be a numeric vector"),
    ],
)
def test_encode_rejects_malformed_observation(observation: object, message: str) -> None:
    """Mis-shaped or non-finite observations fail closed at the encoder boundary."""
    model = _default_model()
    with pytest.raises(OpenDreamerDynamicsError, match=message):
        model.encode(observation)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("action", "message"),
    [
        (np.zeros(ACTION_DIM + 1, dtype=float), "action must have shape"),
        (np.full(ACTION_DIM, np.inf, dtype=float), "action must contain only finite"),
        (np.ones(ACTION_DIM, dtype=bool), "action must not contain booleans"),
    ],
)
def test_step_rejects_malformed_action(action: object, message: str) -> None:
    """Mis-shaped or non-finite actions fail closed at the transition boundary."""
    model = _default_model()
    latent = model.encode(_drive_state(0))
    with pytest.raises(OpenDreamerDynamicsError, match=message):
        model.step(latent, action)  # type: ignore[arg-type]


def test_continuation_head_rejects_non_finite_latent() -> None:
    """A non-finite latent fails closed at the continuation head boundary."""
    model = _default_model()
    with pytest.raises(OpenDreamerDynamicsError, match="latent must contain only finite"):
        model.continuation_head(np.full(DEFAULT_LATENT_DIM, np.nan, dtype=float))


def test_imagine_rejects_non_structured_episode() -> None:
    """Only the merged adapter's StructuredEpisode is accepted; other types fail closed."""
    model = _default_model()
    with pytest.raises(OpenDreamerDynamicsError, match="must be a StructuredEpisode"):
        model.imagine({"observations": []})  # type: ignore[arg-type]


def test_imagine_rejects_observation_width_mismatch() -> None:
    """A model built for ray-free episodes rejects a rays-available episode of a wider view."""
    model = _default_model(with_rays=False)
    rays_episode = _make_structured_episode(step_count=2, with_rays=True, ray_width=3)

    with pytest.raises(OpenDreamerDynamicsError, match="does not match model obs_dim"):
        model.imagine(rays_episode)


def test_dynamics_step_rejects_continuation_outside_unit_interval() -> None:
    """Direct construction cannot bypass the continuation (0, 1) contract."""
    with pytest.raises(OpenDreamerDynamicsError, match="continuation must lie strictly inside"):
        DynamicsStep(
            latent=np.zeros(DEFAULT_LATENT_DIM, dtype=float),
            reward=0.0,
            continuation=1.5,
        )


def test_dynamics_step_rejects_non_finite_latent() -> None:
    """Direct construction cannot bypass the finite-latent contract."""
    with pytest.raises(OpenDreamerDynamicsError, match="latent must contain only finite"):
        DynamicsStep(
            latent=np.full(DEFAULT_LATENT_DIM, np.inf, dtype=float),
            reward=0.0,
            continuation=0.5,
        )


def test_dynamics_step_rejects_latent_at_closed_interval_boundary() -> None:
    """Direct construction cannot bypass the open latent interval contract."""
    latent = np.zeros(DEFAULT_LATENT_DIM, dtype=float)
    latent[0] = 1.0

    with pytest.raises(OpenDreamerDynamicsError, match="strictly inside"):
        DynamicsStep(latent=latent, reward=0.0, continuation=0.5)


def test_latent_rollout_rejects_misaligned_arrays() -> None:
    """A rollout whose latent rows do not equal step_count + 1 fails closed."""
    with pytest.raises(OpenDreamerDynamicsError, match="latents must have shape"):
        LatentRollout(
            latents=np.zeros((2, DEFAULT_LATENT_DIM), dtype=float),
            rewards=np.zeros(3, dtype=float),
            continuations=np.full(3, 0.5, dtype=float),
        )


def test_latent_rollout_rejects_empty_rollout() -> None:
    """Direct construction cannot bypass the non-empty rollout contract."""
    with pytest.raises(OpenDreamerDynamicsError, match="at least one step"):
        LatentRollout(
            latents=np.zeros((1, DEFAULT_LATENT_DIM), dtype=float),
            rewards=np.zeros(0, dtype=float),
            continuations=np.zeros(0, dtype=float),
        )


def test_latent_rollout_rejects_continuation_out_of_range() -> None:
    """A rollout with a continuation outside (0, 1) fails closed."""
    with pytest.raises(OpenDreamerDynamicsError, match="strictly inside"):
        LatentRollout(
            latents=np.zeros((3, DEFAULT_LATENT_DIM), dtype=float),
            rewards=np.zeros(2, dtype=float),
            continuations=np.asarray([0.5, 1.0], dtype=float),
        )


@pytest.mark.parametrize(
    ("observations", "actions", "message"),
    [
        (
            np.zeros((2, _BASE_OBS_DIM), dtype=float),
            np.zeros((3, ACTION_DIM), dtype=float),
            "must be aligned",
        ),
        (
            np.zeros((0, _BASE_OBS_DIM), dtype=float),
            np.zeros((0, ACTION_DIM), dtype=float),
            "at least one step",
        ),
        (
            np.full((2, _BASE_OBS_DIM), np.nan, dtype=float),
            np.zeros((2, ACTION_DIM), dtype=float),
            "observations must contain only finite",
        ),
        (
            np.ones((2, _BASE_OBS_DIM), dtype=bool),
            np.zeros((2, ACTION_DIM), dtype=float),
            "observations must not contain booleans",
        ),
        (
            np.zeros((2, _BASE_OBS_DIM), dtype=float),
            np.ones((2, ACTION_DIM), dtype=bool),
            "actions must not contain booleans",
        ),
    ],
)
def test_imagine_from_arrays_rejects_malformed_inputs(
    observations: np.ndarray,
    actions: np.ndarray,
    message: str,
) -> None:
    """Mis-aligned, empty, or non-finite array inputs fail closed at the array boundary."""
    model = _default_model()
    with pytest.raises(OpenDreamerDynamicsError, match=message):
        model.imagine_from_arrays(observations, actions)
