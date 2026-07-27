"""Contract tests for the Open Dreamer-style structured-observation adapter (issue #6318 Step 2).

These tests pin the **Step 2 adapter contract** of parent #6318's maintainer-authorized sequenced
plan. They verify that the clean-room, episode-major adapter under
:mod:`robot_sf.research.open_dreamer_adapter`:

* consumes the existing ``RLTrajectoryDataset.v1`` / ``RLTrajectoryEpisode.v1`` contract read-only
  (the benchmark module is never edited here);
* preserves raw observations, reward, return_to_go, terminated, truncated, pedestrians,
  robot_states, scenario_id, seed, split, source_policy_id, and provenance for every episode;
* produces a leakage-safe structured-observation view with ``drive_state`` and ``rays`` groups;
* requires rays to be either unavailable for the whole episode or present at every step with one
  fixed vector width;
* exposes a bounded ``[-1, 1] -> (linear velocity, angular velocity)`` action mapping;
* stays **episode-major** (no flattening to transitions);
* and **fails closed** on missing fields, scenario/seed split leakage, non-finite outputs, and
  incompatible action spaces.

This is diagnostic/contract evidence only (``evidence_tier: idea``): the tests make no benchmark,
metric, or policy claim.
"""

from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from robot_sf.benchmark.map_runner_trace import _command_action_payload
from robot_sf.benchmark.rl_trajectory_dataset import (
    RL_TRAJECTORY_EPISODE_SCHEMA_VERSION,
    RLTrajectoryEpisode,
    assign_deterministic_split,
    load_rl_trajectory_dataset,
)
from robot_sf.research.open_dreamer_adapter import (
    ADAPTER_PROVENANCE_KEY,
    DRIVE_STATE_LAYOUT,
    EVIDENCE_BOUNDARY,
    EXPECTED_ACTION_DIM,
    OBSERVATION_NORMALIZATION,
    OPEN_DREAMER_ADAPTER_VERSION,
    OPEN_DREAMER_OBSERVATION_CONTRACT,
    ActionBounds,
    OpenDreamerAdapterError,
    adapt_episode,
    adapt_episodes,
    map_action_to_velocity,
    validate_split_leakage,
)

# Canonical bounds for the [-1, 1] -> (linear, angular) velocity mapping, matching the
# differential-drive defaults in robot_sf/robot/differential_drive.py. The adapter never hardcodes
# these; they are supplied by the caller and recorded in provenance.
_DEFAULT_BOUNDS = ActionBounds(max_linear_speed=2.0, max_angular_speed=1.0, min_linear_speed=0.0)

_EVIDENCE_PREVIEW = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "context"
    / "evidence"
    / "issue_4011_rl_trajectory_dataset_smoke_2026-07-02"
    / "issue_4011_smoke.preview.jsonl"
)

_RECORDER_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "benchmark"
    / "record_rl_trajectory_dataset.py"
)
_RECORDER_SPEC = importlib.util.spec_from_file_location(
    "record_rl_trajectory_dataset", _RECORDER_PATH
)
assert _RECORDER_SPEC is not None
assert _RECORDER_SPEC.loader is not None
record_rl_trajectory_dataset = importlib.util.module_from_spec(_RECORDER_SPEC)
_RECORDER_SPEC.loader.exec_module(record_rl_trajectory_dataset)


def _full_robot_state(index: int) -> dict:
    """Return a robot_states entry exposing position, heading, and velocity (real-data shape)."""
    return {
        "position": [float(index), float(index + 1)],
        "heading": 0.5 * float(index),
        "velocity": [0.25 * float(index), 0.0],
    }


def _full_observation(index: int, *, with_rays: bool = False) -> dict:
    """Return an observations entry matching the committed v1 recorder shape.

    Args:
        index: Step index used to vary the values.
        with_rays: When true, add a recognized ``rays`` key to exercise the rays group.

    Returns:
        A mapping with ``robot`` and ``pedestrians``, and optionally a ``rays`` vector.
    """
    obs: dict = {
        "robot": _full_robot_state(index),
        "pedestrians": [],
    }
    if with_rays:
        obs["rays"] = [1.0 - 0.1 * float(index), 0.8, 0.6]
    return obs


def _make_episode(
    *,
    scenario_id: str = "classic_cross_trap_low",
    seed: int = 101,
    step_count: int = 2,
    actions: tuple | None = None,
    rewards: tuple | None = None,
    robot_states: tuple | None = None,
    observations: tuple | None = None,
    extra: dict | None = None,
) -> RLTrajectoryEpisode:
    """Build a minimal but contract-valid v1 episode with the real-data robot/observation shape.

    Args:
        scenario_id: Scenario id for the episode.
        seed: Integer seed.
        step_count: Number of steps in the episode.
        actions: Optional override for the per-step actions.
        rewards: Optional override for the per-step rewards.
        robot_states: Optional override for the per-step robot_states.
        observations: Optional override for the per-step observations.
        extra: Optional overrides for rarely-varied fields (``split``, ``source_policy_id``,
            ``provenance``, ``return_to_go``). When ``split`` is absent it is derived from
            :func:`assign_deterministic_split`.

    Returns:
        A contract-valid ``RLTrajectoryEpisode``.
    """
    overrides = extra or {}
    split = overrides.get("split")
    if split is None:
        split = assign_deterministic_split(scenario_id, seed)
    source_policy_id = overrides.get("source_policy_id", "goal")
    provenance = overrides.get("provenance", {"source": "unit_test"})
    if actions is None:
        actions = tuple([0.5, 0.0] for _ in range(step_count))
    if rewards is None:
        rewards = tuple(float(step + 1) for step in range(step_count))
    return_to_go = overrides.get("return_to_go")
    if return_to_go is None:
        total = sum(rewards)
        return_to_go = tuple(total - sum(rewards[:step]) for step in range(step_count))
    if robot_states is None:
        robot_states = tuple(_full_robot_state(step) for step in range(step_count))
    if observations is None:
        observations = tuple(_full_observation(step) for step in range(step_count))
    # Place the single allowed terminal marker on the final step.
    terminated = tuple(step == step_count - 1 for step in range(step_count))
    truncated = tuple(False for _ in range(step_count))
    return RLTrajectoryEpisode(
        dataset_id="issue_6318_adapter_smoke",
        episode_id=f"{scenario_id}:seed{seed}:{source_policy_id}:000000",
        scenario_id=scenario_id,
        seed=seed,
        source_policy_id=source_policy_id,
        split=split,
        observations=observations,
        actions=actions,
        rewards=rewards,
        return_to_go=return_to_go,
        terminated=terminated,
        truncated=truncated,
        pedestrians=tuple([] for _ in range(step_count)),
        robot_states=robot_states,
        provenance=provenance,
    )


# ----------------------------------------------------------------------------------------------
# Positive contract: structured groups, preservation, episode-major, provenance.
# ----------------------------------------------------------------------------------------------


def test_drive_state_group_is_finite_fixed_layout_vector() -> None:
    """The drive_state group is a finite vector in the documented fixed component order."""
    episode = _make_episode(step_count=3)
    structured = adapt_episode(episode, action_bounds=_DEFAULT_BOUNDS)

    assert len(structured.observations) == 3
    for index, step in enumerate(structured.observations):
        assert step.drive_state.shape == (len(DRIVE_STATE_LAYOUT),)
        assert np.all(np.isfinite(step.drive_state))
        # [x, y, heading, vx, vy] derived directly from the recorded robot state.
        expected = np.asarray(
            [
                float(index),
                float(index + 1),
                0.5 * float(index),
                0.25 * float(index),
                0.0,
            ],
            dtype=float,
        )
        np.testing.assert_allclose(step.drive_state, expected)


def test_rays_group_is_unavailable_when_observation_has_no_ray_key() -> None:
    """The rays group is honestly empty/unavailable when the v1 observation records no rays."""
    episode = _make_episode(step_count=2)
    structured = adapt_episode(episode, action_bounds=_DEFAULT_BOUNDS)

    for step in structured.observations:
        assert step.rays_available is False
        assert step.rays.size == 0
    assert structured.rays_available is False


def test_rays_group_is_populated_when_observation_carries_ray_key() -> None:
    """A recognized ray-like key populates the finite rays group and flips episode availability."""
    observations = tuple(_full_observation(step, with_rays=True) for step in range(2))
    episode = _make_episode(step_count=2, observations=observations)
    structured = adapt_episode(episode, action_bounds=_DEFAULT_BOUNDS)

    assert structured.rays_available is True
    for index, step in enumerate(structured.observations):
        assert step.rays_available is True
        assert step.rays.shape == (3,)
        assert np.all(np.isfinite(step.rays))
        np.testing.assert_allclose(step.rays, [1.0 - 0.1 * float(index), 0.8, 0.6])


@pytest.mark.parametrize(
    ("ray_presence",),
    [((True, False),), ((False, True),)],
    ids=["missing-final-step", "missing-initial-step"],
)
def test_adapter_rejects_partial_ray_availability(ray_presence: tuple[bool, bool]) -> None:
    """A valid v1 episode cannot mix ray-bearing and ray-free steps."""
    observations = tuple(
        _full_observation(step, with_rays=has_rays) for step, has_rays in enumerate(ray_presence)
    )
    episode = _make_episode(step_count=2, observations=observations)

    with pytest.raises(OpenDreamerAdapterError, match="ray availability must be episode-wide"):
        adapt_episode(episode, action_bounds=_DEFAULT_BOUNDS)


def test_adapter_rejects_inconsistent_ray_vector_lengths() -> None:
    """A ray-bearing episode must keep one ray-vector width for sequence stacking."""
    observations = (
        {**_full_observation(0, with_rays=True), "rays": [1.0, 0.8]},
        {**_full_observation(1, with_rays=True), "rays": [0.9, 0.7, 0.6]},
    )
    episode = _make_episode(step_count=2, observations=observations)

    with pytest.raises(OpenDreamerAdapterError, match="ray vectors must have a consistent length"):
        adapt_episode(episode, action_bounds=_DEFAULT_BOUNDS)


def test_all_v1_fields_preserved_verbatim() -> None:
    """Every v1 field, including unrecognized raw observation keys, is retained verbatim."""
    observations = tuple(
        {
            **_full_observation(step),
            "future_model_feature": {"source": "preserve-me", "step": step},
        }
        for step in range(3)
    )
    episode = _make_episode(step_count=3, observations=observations)
    structured = adapt_episode(episode, action_bounds=_DEFAULT_BOUNDS)

    assert structured.dataset_id == episode.dataset_id
    assert structured.episode_id == episode.episode_id
    assert structured.scenario_id == episode.scenario_id
    assert structured.seed == episode.seed
    assert structured.source_policy_id == episode.source_policy_id
    assert structured.split == episode.split
    assert structured.raw_observations == episode.observations
    assert structured.raw_actions == episode.actions
    assert structured.raw_observations[0]["future_model_feature"] == {
        "source": "preserve-me",
        "step": 0,
    }
    assert structured.rewards == episode.rewards
    assert structured.return_to_go == episode.return_to_go
    assert structured.terminated == episode.terminated
    assert structured.truncated == episode.truncated
    assert structured.pedestrians == episode.pedestrians
    assert structured.robot_states == episode.robot_states
    assert structured.to_dict()["raw_observations"] == list(episode.observations)
    assert structured.to_dict()["raw_actions"] == list(episode.actions)


def test_raw_stored_action_preserved_verbatim() -> None:
    """The raw (linear, angular) action is preserved untouched in each structured action step."""
    actions = ([0.5, 0.0], [0.4, 0.1], [0.0, -0.3])
    episode = _make_episode(step_count=3, actions=tuple(actions))
    structured = adapt_episode(episode, action_bounds=_DEFAULT_BOUNDS)

    assert len(structured.actions) == 3
    assert structured.raw_actions == tuple(actions)
    for index, step in enumerate(structured.actions):
        assert step.raw == tuple(actions[index])
    assert structured.to_dict()["raw_actions"] == list(actions)


def test_adapter_accepts_numpy_real_scalars_from_programmatic_v1_producers() -> None:
    """Programmatic v1 producers can supply finite NumPy real scalars without coercion failures."""
    episode = _make_episode(
        step_count=1,
        actions=([np.float32(0.5), np.float32(-0.25)],),
        rewards=(np.float32(1.0),),
        robot_states=(
            {
                "position": [np.float32(0.0), np.float32(1.0)],
                "heading": np.float32(0.0),
                "velocity": [np.float32(0.5), np.float32(0.0)],
            },
        ),
        extra={"return_to_go": (np.float32(1.0),)},
    )

    structured = adapt_episode(episode, action_bounds=_DEFAULT_BOUNDS)

    assert structured.actions[0].raw == (0.5, -0.25)
    assert structured.rewards == (1.0,)
    np.testing.assert_allclose(structured.observations[0].drive_state, [0.0, 1.0, 0.0, 0.5, 0.0])


def test_adapter_accepts_current_simulation_trace_recorder_action_mapping(tmp_path: Path) -> None:
    """The recorder's selected_action mapping reaches the adapter as the same physical command."""
    source_record = {
        "episode_id": "classic_cross_trap_low:seed101:goal:000000",
        "scenario_id": "classic_cross_trap_low",
        "seed": 101,
        "algo": "goal",
        "algorithm_metadata": {
            "simulation_step_trace": {
                "schema_version": "simulation-step-trace.v1",
                "steps": [
                    {
                        "step": 0,
                        "robot": _full_robot_state(0),
                        "pedestrians": [],
                        "planner": {"selected_action": _command_action_payload([0.5, -0.25])},
                        "rl": {"reward": 1.0, "terminated": True, "truncated": False},
                    }
                ],
            }
        },
    }

    episode = record_rl_trajectory_dataset.convert_source_records(
        [source_record],
        dataset_id="issue_6318_adapter_smoke",
        source_jsonl=tmp_path / "simulation_steps.jsonl",
    )[0]
    structured = adapt_episode(episode, action_bounds=_DEFAULT_BOUNDS)

    assert episode.actions == ({"linear_velocity": 0.5, "angular_velocity": -0.25},)
    assert structured.actions[0].raw == (0.5, -0.25)


def test_episode_view_stays_episode_major() -> None:
    """The structured view never flattens to transitions: per-step lengths equal the step count."""
    episode = _make_episode(step_count=4)
    structured = adapt_episode(episode, action_bounds=_DEFAULT_BOUNDS)

    expected = 4
    assert structured.step_count == expected
    assert len(structured.observations) == expected
    assert len(structured.actions) == expected
    assert len(structured.rewards) == expected
    assert len(structured.return_to_go) == expected
    assert len(structured.terminated) == expected
    assert len(structured.truncated) == expected
    assert len(structured.pedestrians) == expected
    assert len(structured.robot_states) == expected


def test_provenance_preserved_and_augmented_with_adapter_metadata() -> None:
    """The original provenance is preserved and the adapter entry records version and bounds."""
    original = {"source": "unit_test", "return_convention": "undiscounted_future_return_to_go"}
    episode = _make_episode(step_count=2, extra={"provenance": original})
    structured = adapt_episode(episode, action_bounds=_DEFAULT_BOUNDS)

    provenance = structured.provenance
    # Original keys are preserved.
    for key, value in original.items():
        assert provenance[key] == value
    adapter_entry = provenance[ADAPTER_PROVENANCE_KEY]
    assert adapter_entry["adapter_version"] == OPEN_DREAMER_ADAPTER_VERSION
    assert adapter_entry["consumed_episode_schema"] == RL_TRAJECTORY_EPISODE_SCHEMA_VERSION
    assert adapter_entry["evidence_boundary"] == EVIDENCE_BOUNDARY
    assert adapter_entry["split_policy"] == "assign_deterministic_split"
    assert adapter_entry["observation_contract"] == OPEN_DREAMER_OBSERVATION_CONTRACT
    assert adapter_entry["observation_normalization"] == OBSERVATION_NORMALIZATION
    assert adapter_entry["action_bounds"] == _DEFAULT_BOUNDS.to_dict()
    assert "drive_state_layout" in adapter_entry
    assert structured.to_dict()["observation_contract"] == OPEN_DREAMER_OBSERVATION_CONTRACT


def test_adapter_rejects_reserved_provenance_collision_without_overwriting_source() -> None:
    """A source-owned adapter key fails closed instead of silently destroying provenance."""
    original_entry = {"source": "must-not-be-overwritten"}
    episode = _make_episode(
        step_count=1,
        extra={"provenance": {ADAPTER_PROVENANCE_KEY: original_entry}},
    )

    with pytest.raises(OpenDreamerAdapterError, match="refusing to overwrite source provenance"):
        adapt_episode(episode, action_bounds=_DEFAULT_BOUNDS)

    assert episode.provenance[ADAPTER_PROVENANCE_KEY] == original_entry


def test_to_dict_is_json_safe_for_accepted_numpy_producer_values() -> None:
    """The serialized view handles NumPy scalars accepted from programmatic v1 producers."""
    episode = _make_episode(
        step_count=1,
        actions=([np.float32(0.5), np.float32(-0.25)],),
        rewards=(np.float32(1.0),),
        robot_states=(
            {
                "position": np.asarray([0.0, 1.0], dtype=np.float32),
                "heading": np.float32(0.0),
                "velocity": np.asarray([0.5, 0.0], dtype=np.float32),
            },
        ),
        observations=(
            {
                "robot": {"quality": np.float32(0.75)},
                "pedestrians": [],
                "future_vector": np.asarray([1, 2], dtype=np.int64),
            },
        ),
        extra={
            "return_to_go": (np.float32(1.0),),
            "provenance": {"producer_counter": np.int64(3)},
        },
    )

    payload = adapt_episode(episode, action_bounds=_DEFAULT_BOUNDS).to_dict()

    json.dumps(payload)
    assert payload["raw_observations"][0]["future_vector"] == [1, 2]
    assert payload["provenance"]["producer_counter"] == 3


def test_adapt_episodes_preserves_input_order_and_count() -> None:
    """adapt_episodes returns structured episodes in input order without flattening or reordering."""
    episodes = [_make_episode(seed=seed, step_count=2) for seed in (101, 202, 303)]
    structured = adapt_episodes(episodes, action_bounds=_DEFAULT_BOUNDS)

    assert [item.episode_id for item in structured] == [item.episode_id for item in episodes]
    assert [item.seed for item in structured] == [101, 202, 303]


# ----------------------------------------------------------------------------------------------
# [-1, 1] -> (linear velocity, angular velocity) action mapping.
# ----------------------------------------------------------------------------------------------


def test_action_mapping_boundaries_are_respected() -> None:
    """The forward mapping saturates to the declared velocity envelope at the [-1, 1] corners."""
    bounds = ActionBounds(max_linear_speed=2.0, max_angular_speed=1.0, min_linear_speed=0.0)

    # Forward-only envelope: linear maps [-1,1] -> [0, 2], angular maps [-1,1] -> [-1, 1].
    assert map_action_to_velocity([1.0, 1.0], bounds) == (2.0, 1.0)
    assert map_action_to_velocity([-1.0, -1.0], bounds) == (0.0, -1.0)
    assert map_action_to_velocity([0.0, 0.0], bounds) == (1.0, 0.0)

    backwards = ActionBounds(max_linear_speed=2.0, max_angular_speed=1.0, min_linear_speed=-1.0)
    # Symmetric-ish envelope: linear maps [-1,1] -> [-1, 2].
    assert map_action_to_velocity([1.0, 1.0], backwards) == (2.0, 1.0)
    assert map_action_to_velocity([-1.0, 0.0], backwards) == (-1.0, 0.0)


def test_action_mapping_absorbs_float_roundoff_within_tolerance() -> None:
    """Accepted roundoff is clamped, so both mapped velocities stay in their envelopes."""
    bounds = ActionBounds(max_linear_speed=2.0, max_angular_speed=1.0, min_linear_speed=0.0)
    linear, angular = map_action_to_velocity([1.0 + 1e-12, -1.0 - 1e-12], bounds)
    assert linear == 2.0
    assert angular == -1.0
    assert bounds.min_linear_speed <= linear <= bounds.max_linear_speed
    assert -bounds.max_angular_speed <= angular <= bounds.max_angular_speed


def test_action_mapping_rejects_wrong_shape_nonfinite_and_far_out_of_domain() -> None:
    """The mapping fails closed on wrong-dimensionality, non-finite, or far-out-of-domain inputs."""
    bounds = ActionBounds(max_linear_speed=2.0, max_angular_speed=1.0)
    with pytest.raises(OpenDreamerAdapterError, match="shape"):
        map_action_to_velocity([1.0, 1.0, 1.0], bounds)  # type: ignore[list-item]
    with pytest.raises(OpenDreamerAdapterError, match="finite"):
        map_action_to_velocity([float("nan"), 0.0], bounds)
    with pytest.raises(OpenDreamerAdapterError, match=r"\[-1, 1\]"):
        map_action_to_velocity([5.0, 0.0], bounds)
    with pytest.raises(OpenDreamerAdapterError, match="numeric length-2"):
        map_action_to_velocity(["not-a-number", 0.0], bounds)  # type: ignore[list-item]
    with pytest.raises(OpenDreamerAdapterError, match="numeric length-2"):
        map_action_to_velocity(["1.0", 0.0], bounds)  # type: ignore[list-item]
    with pytest.raises(OpenDreamerAdapterError, match="numeric length-2"):
        map_action_to_velocity([True, False], bounds)  # type: ignore[list-item]


def test_action_bounds_validate_non_degenerate_envelope() -> None:
    """ActionBounds rejects non-positive, non-finite, or inconsistent speed bounds."""
    with pytest.raises(OpenDreamerAdapterError, match="max_linear_speed"):
        ActionBounds(max_linear_speed=0.0, max_angular_speed=1.0)
    with pytest.raises(OpenDreamerAdapterError, match="max_angular_speed"):
        ActionBounds(max_linear_speed=1.0, max_angular_speed=-1.0)
    with pytest.raises(OpenDreamerAdapterError, match="min_linear_speed"):
        ActionBounds(max_linear_speed=1.0, max_angular_speed=1.0, min_linear_speed=-2.0)
    with pytest.raises(OpenDreamerAdapterError, match="strictly less"):
        ActionBounds(max_linear_speed=1.0, max_angular_speed=1.0, min_linear_speed=1.0)
    with pytest.raises(OpenDreamerAdapterError, match="finite real number"):
        ActionBounds(max_linear_speed=True, max_angular_speed=1.0)  # type: ignore[arg-type]


def test_stored_actions_must_lie_within_declared_physical_bounds() -> None:
    """Stored physical commands accept envelope endpoints and reject out-of-range values."""
    boundary_actions = (
        {"linear_velocity": 0.0, "angular_velocity": -1.0},
        {"linear_velocity": 2.0, "angular_velocity": 1.0},
    )
    structured = adapt_episode(
        _make_episode(step_count=2, actions=boundary_actions), action_bounds=_DEFAULT_BOUNDS
    )
    assert [step.raw for step in structured.actions] == [(0.0, -1.0), (2.0, 1.0)]

    for action in (
        {"linear_velocity": -0.01, "angular_velocity": 0.0},
        {"linear_velocity": 2.01, "angular_velocity": 0.0},
        {"linear_velocity": 1.0, "angular_velocity": -1.01},
        {"linear_velocity": 1.0, "angular_velocity": 1.01},
    ):
        with pytest.raises(OpenDreamerAdapterError, match="supplied action bounds"):
            adapt_episode(
                _make_episode(step_count=1, actions=(action,)), action_bounds=_DEFAULT_BOUNDS
            )


# ----------------------------------------------------------------------------------------------
# Fail-closed contract: missing fields, non-finite outputs, incompatible action spaces.
# ----------------------------------------------------------------------------------------------


def test_fail_closed_on_missing_drive_state_components() -> None:
    """A robot_states entry missing heading/velocity fails closed rather than guessing a layout."""
    # The mixed key types also prove malformed metadata cannot escape through error formatting.
    robot_states = ({"position": [0.0, 0.0], 1: "unrecognized"},)
    episode = _make_episode(step_count=1, robot_states=robot_states)
    with pytest.raises(OpenDreamerAdapterError, match="position, heading, and velocity"):
        adapt_episode(episode, action_bounds=_DEFAULT_BOUNDS)


def test_fail_closed_on_non_finite_drive_state() -> None:
    """A non-finite drive_state component fails closed."""
    robot_states = ({"position": [0.0, float("inf")], "heading": 0.0, "velocity": [0.0, 0.0]},)
    episode = _make_episode(step_count=1, robot_states=robot_states)
    with pytest.raises(OpenDreamerAdapterError, match="finite"):
        adapt_episode(episode, action_bounds=_DEFAULT_BOUNDS)


def test_fail_closed_on_non_finite_rewards() -> None:
    """A non-finite reward fails closed so it cannot reach a future model."""
    rewards = (1.0, float("nan"))
    episode = _make_episode(step_count=2, rewards=rewards)
    with pytest.raises(OpenDreamerAdapterError, match="rewards must be finite"):
        adapt_episode(episode, action_bounds=_DEFAULT_BOUNDS)


def test_adapter_wraps_malformed_v1_field_types_in_its_fail_closed_error() -> None:
    """Malformed v1 field types do not leak a raw validator TypeError to adapter consumers."""
    malformed = replace(_make_episode(step_count=1), rewards=None)  # type: ignore[arg-type]

    with pytest.raises(OpenDreamerAdapterError, match="upstream RLTrajectoryEpisode.v1 validation"):
        adapt_episode(malformed, action_bounds=_DEFAULT_BOUNDS)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("terminated", ("False",)),
        ("truncated", (np.bool_(False),)),
    ],
)
def test_adapter_rejects_coercible_non_boolean_terminal_flags(
    field_name: str,
    value: tuple[object, ...],
) -> None:
    """Terminal flags must already be booleans so adaptation cannot change their meaning."""
    malformed = replace(_make_episode(step_count=1), **{field_name: value})

    with pytest.raises(OpenDreamerAdapterError, match=rf"{field_name}\[0\] must be a boolean"):
        adapt_episode(malformed, action_bounds=_DEFAULT_BOUNDS)


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("source_policy_id", "", "source_policy_id must be a non-empty string"),
        ("provenance", None, "provenance must be a mapping"),
        ("seed", True, "seed must be a non-boolean integer"),
    ],
)
def test_adapter_rejects_malformed_metadata_with_uniform_error(
    field_name: str,
    value: object,
    message: str,
) -> None:
    """Malformed programmatic metadata fails with the adapter's public error type."""
    malformed = replace(_make_episode(step_count=1), **{field_name: value})

    with pytest.raises(OpenDreamerAdapterError, match=message):
        adapt_episode(malformed, action_bounds=_DEFAULT_BOUNDS)


def test_batch_adapter_rejects_malformed_seed_with_uniform_error() -> None:
    """Batch leakage preflight cannot leak a raw conversion error for an invalid seed."""
    malformed = replace(_make_episode(step_count=1), seed=None)  # type: ignore[arg-type]

    with pytest.raises(OpenDreamerAdapterError, match="seed must be a non-boolean integer"):
        adapt_episodes([malformed], action_bounds=_DEFAULT_BOUNDS)


def test_fail_closed_on_non_finite_rays_when_present() -> None:
    """A present-but-non-finite ray-like field fails closed instead of silently masking it."""
    observations = (
        {"robot": _full_robot_state(0), "pedestrians": [], "rays": [0.5, float("inf")]},
    )
    episode = _make_episode(step_count=1, observations=observations)
    with pytest.raises(OpenDreamerAdapterError, match="ray-like key 'rays' must be finite"):
        adapt_episode(episode, action_bounds=_DEFAULT_BOUNDS)


@pytest.mark.parametrize(
    ("rays",),
    [([True, False],), (["0.5", "0.25"],)],
    ids=["booleans", "numeric-strings"],
)
def test_fail_closed_on_coercible_non_numeric_ray_values(rays: list[object]) -> None:
    """Ray values must already be real numbers rather than coercible strings or booleans."""
    observations = ({"robot": _full_robot_state(0), "pedestrians": [], "rays": rays},)
    episode = _make_episode(step_count=1, observations=observations)

    with pytest.raises(OpenDreamerAdapterError, match="finite real values"):
        adapt_episode(episode, action_bounds=_DEFAULT_BOUNDS)


def test_fail_closed_on_empty_rays_when_key_is_present() -> None:
    """An explicit but empty ray field is malformed, not an available zero-width sensor."""
    observations = ({"robot": _full_robot_state(0), "pedestrians": [], "rays": []},)
    episode = _make_episode(step_count=1, observations=observations)

    with pytest.raises(OpenDreamerAdapterError, match="must contain at least one range"):
        adapt_episode(episode, action_bounds=_DEFAULT_BOUNDS)


def test_fail_closed_on_incompatible_action_dimensionality() -> None:
    """A stored action that is not 2D (linear, angular) fails closed as an incompatible action space."""
    actions = ([0.5, 0.0, 0.0],)  # 3D command -- e.g. an upstream VPT-style container.
    episode = _make_episode(step_count=1, actions=actions)
    with pytest.raises(OpenDreamerAdapterError, match="incompatible action space"):
        adapt_episode(episode, action_bounds=_DEFAULT_BOUNDS)


def test_fail_closed_on_action_mapping_with_unrecognized_extra_dimension() -> None:
    """A mapping cannot smuggle an extra action dimension past the canonical two-key contract."""
    actions = ({"linear_velocity": 0.5, "angular_velocity": 0.0, "camera_pitch": 0.1},)
    episode = _make_episode(step_count=1, actions=actions)

    with pytest.raises(OpenDreamerAdapterError, match="only .*incompatible action space"):
        adapt_episode(episode, action_bounds=_DEFAULT_BOUNDS)


def test_fail_closed_on_non_numeric_action() -> None:
    """A non-numeric stored action fails closed."""
    actions = (["fwd", "left"],)  # type: ignore[list-item]
    episode = _make_episode(step_count=1, actions=actions)
    with pytest.raises(OpenDreamerAdapterError, match="must be numeric"):
        adapt_episode(episode, action_bounds=_DEFAULT_BOUNDS)


def test_fail_closed_on_non_finite_action() -> None:
    """A non-finite stored action fails closed."""
    actions = ([0.5, float("nan")],)
    episode = _make_episode(step_count=1, actions=actions)
    with pytest.raises(OpenDreamerAdapterError, match="finite"):
        adapt_episode(episode, action_bounds=_DEFAULT_BOUNDS)


# ----------------------------------------------------------------------------------------------
# Scenario/seed split leakage (must respect assign_deterministic_split).
# ----------------------------------------------------------------------------------------------


def test_validate_split_leakage_passes_for_deterministic_assignment() -> None:
    """A dataset split entirely via assign_deterministic_split has zero leakage by construction."""
    episodes = [
        _make_episode(scenario_id=f"map_{i}", seed=seed, step_count=1)
        for i in range(3)
        for seed in range(10)
    ]
    report = validate_split_leakage(episodes)
    assert report.ok is True
    assert report.leaked_keys == ()
    # Every split name is represented in the canonical names tuple.
    assert set(report.split_scenario_seed_keys).issubset({"train", "validation", "test"})


def test_validate_split_leakage_fails_closed_on_cross_split_key() -> None:
    """A (scenario_id, seed) key placed in two splits is reported as leakage and fails closed."""
    same_key_train = _make_episode(
        scenario_id="leaky_map", seed=42, step_count=1, extra={"split": "train"}
    )
    same_key_test = _make_episode(
        scenario_id="leaky_map", seed=42, step_count=1, extra={"split": "test"}
    )
    report = validate_split_leakage([same_key_train, same_key_test])

    assert report.ok is False
    assert "leaky_map:42" in report.leaked_keys


def test_adapt_episodes_rejects_cross_split_scenario_seed_leakage() -> None:
    """The public batch adapter rejects a scenario/seed key that crosses train and test splits."""
    same_key_train = _make_episode(
        scenario_id="leaky_map", seed=42, step_count=1, extra={"split": "train"}
    )
    same_key_test = _make_episode(
        scenario_id="leaky_map", seed=42, step_count=1, extra={"split": "test"}
    )

    with pytest.raises(OpenDreamerAdapterError, match="scenario/seed split leakage.*leaky_map:42"):
        adapt_episodes([same_key_train, same_key_test], action_bounds=_DEFAULT_BOUNDS)


def test_adapter_preserves_stored_split_without_reassigning() -> None:
    """The adapter preserves the dataset's recorded split and never silently rewrites it."""
    episode = _make_episode(scenario_id="classic_cross_trap_low", seed=202, extra={"split": "test"})
    structured = adapt_episode(episode, action_bounds=_DEFAULT_BOUNDS)

    assert structured.split == "test"
    # And the canonical deterministic policy agrees this key is not a train key.
    assert assign_deterministic_split("classic_cross_trap_low", 202) != "train"


def test_adapter_rejects_valid_but_noncanonical_stored_split() -> None:
    """A valid stored split must equal its deterministic assignment, preventing single-key leakage."""
    scenario_id = "canonical_split_guard"
    seed = 31337
    canonical = assign_deterministic_split(scenario_id, seed)
    noncanonical = next(split for split in ("train", "validation", "test") if split != canonical)
    episode = _make_episode(
        scenario_id=scenario_id,
        seed=seed,
        step_count=1,
        extra={"split": noncanonical},
    )

    with pytest.raises(
        OpenDreamerAdapterError, match="does not match canonical deterministic split"
    ):
        adapt_episode(episode, action_bounds=_DEFAULT_BOUNDS)


def test_adapter_rejects_invalid_split_name() -> None:
    """A split outside the canonical names fails closed at adaptation time."""
    episode = _make_episode(step_count=1, extra={"split": "holdout"})
    with pytest.raises(OpenDreamerAdapterError, match="split must be one of"):
        adapt_episode(episode, action_bounds=_DEFAULT_BOUNDS)


# ----------------------------------------------------------------------------------------------
# Integration against the committed RLTrajectoryDataset.v1 smoke preview (read-only).
# ----------------------------------------------------------------------------------------------


@pytest.mark.skipif(not _EVIDENCE_PREVIEW.is_file(), reason="committed smoke preview absent")
def test_adapter_consumes_committed_smoke_preview_read_only() -> None:
    """The adapter consumes the committed v1 smoke preview and preserves every field.

    This is the read-only integration smoke: the benchmark module is not edited, the committed
    ``RLTrajectoryDataset.v1`` preview is loaded through its canonical loader, and the adapter
    produces episode-major structured episodes that preserve all v1 fields and provenance. It is
    diagnostic/contract evidence only (evidence_tier: idea), not a benchmark or policy claim.
    """
    episodes = load_rl_trajectory_dataset(_EVIDENCE_PREVIEW)
    assert episodes, "committed smoke preview must contain at least one episode"

    structured = adapt_episodes(episodes, action_bounds=_DEFAULT_BOUNDS)

    assert len(structured) == len(episodes)
    assert validate_split_leakage(episodes).ok is True
    for original, adapted in zip(episodes, structured, strict=True):
        assert adapted.dataset_id == original.dataset_id
        assert adapted.episode_id == original.episode_id
        assert adapted.scenario_id == original.scenario_id
        assert adapted.seed == original.seed
        assert adapted.source_policy_id == original.source_policy_id
        assert adapted.split == original.split
        assert adapted.raw_observations == original.observations
        assert adapted.rewards == original.rewards
        assert adapted.return_to_go == original.return_to_go
        assert adapted.terminated == original.terminated
        assert adapted.truncated == original.truncated
        assert adapted.pedestrians == original.pedestrians
        assert adapted.robot_states == original.robot_states
        # Each step exposes a finite drive_state group; the preview records no rays.
        for step in adapted.observations:
            assert step.drive_state.shape == (len(DRIVE_STATE_LAYOUT),)
            assert np.all(np.isfinite(step.drive_state))
            assert step.rays_available is False
            assert step.rays.size == 0
        # Original provenance keys are preserved alongside the adapter entry.
        for key, value in original.provenance.items():
            assert adapted.provenance[key] == value
        assert ADAPTER_PROVENANCE_KEY in adapted.provenance


def test_expected_action_dim_is_two() -> None:
    """The adapter is pinned to the 2D (linear, angular) continuous action contract."""
    assert EXPECTED_ACTION_DIM == 2
