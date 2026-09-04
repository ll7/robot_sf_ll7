"""Diagnostic contract tests for the entity-ID-keyed ScenarioBelief snapshot projection.

These tests cover the additive interface only.  They do not claim planner
performance, identity-generation support, safety improvement, or benchmark
evidence.
"""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import robot_sf.planner.scenario_belief_adapter as adapter
from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.planner.scenario_belief_adapter import (
    BELIEF_AWARE_PLANNER_INPUT_SCHEMA_VERSION,
    SUPPORTED_BELIEF_AWARE_PLANNER_NAMES,
    SUPPORTED_PROJECTION_TARGETS,
    BeliefAwarePlannerInput,
    PlannerTrackBelief,
    project_belief_aware_planner_input,
    project_scenario_belief_for_belief_aware_planner,
    project_scenario_belief_for_planner,
)
from robot_sf.representation import VisibilityState, scenario_belief_from_simulator_oracle


def _belief_fixture():
    """Return a small public simulator-like ScenarioBelief fixture."""
    simulator = SimpleNamespace(
        ped_pos=np.array([[2.0, 0.0], [0.0, 3.0]], dtype=np.float32),
        ped_vel=np.array([[0.5, 0.0], [0.0, -0.25]], dtype=np.float32),
        robots=[
            SimpleNamespace(
                pose=((0.0, 0.0), 0.0),
                current_speed=np.array([0.1, 0.0], dtype=np.float32),
                config=SimpleNamespace(radius=0.4),
            )
        ],
        goal_pos=[np.array([5.0, 0.0], dtype=np.float32)],
        next_goal_pos=[None],
        map_def=SimpleNamespace(width=10.0, height=8.0, obstacles=[]),
        config=SimpleNamespace(time_per_step_in_secs=0.1),
    )
    belief = scenario_belief_from_simulator_oracle(
        simulator,
        env_config=RobotSimulationConfig(),
        max_pedestrians=4,
    )
    occluded = replace(
        belief.agents[1],
        visibility_state=VisibilityState.OCCLUDED,
        last_observed_age_s=0.25,
    )
    return replace(belief, sim_time_s=0.5, agents=(belief.agents[0], occluded))


def _standalone_track(**overrides):
    """Build a valid standalone planner track for validation-edge tests."""
    values = {
        "track_id": "ped_000",
        "mean_state": np.zeros(5),
        "covariance": np.eye(5),
        "confidence": 1.0,
        "existence_probability": 1.0,
        "visibility": True,
        "age_steps": 0,
        "source": "unit_test",
    }
    values.update(overrides)
    return PlannerTrackBelief(**values)


def test_projection_retains_visible_and_occluded_tracks_by_snapshot_entity_id() -> None:
    """Visible legacy rows and complete snapshot-keyed tracks stay distinct."""
    belief = _belief_fixture()

    projected = project_belief_aware_planner_input(
        belief,
        planner_name="BeliefGuidedLocalPlanner",
    )

    assert projected.schema_version == BELIEF_AWARE_PLANNER_INPUT_SCHEMA_VERSION
    assert projected.diagnostics["status"] == "projected"
    assert tuple(projected.tracks) == ("ped_000", "ped_001")
    assert projected.diagnostics["visible_track_count"] == 1
    assert projected.diagnostics["occluded_track_count"] == 1
    assert projected.diagnostics["stale_track_count"] == 1
    assert projected.diagnostics["projected_track_count"] == 2
    assert projected.diagnostics["supported_projection_target"] is True
    assert projected.ordered_track_ids() == ("ped_000", "ped_001")
    assert projected.tracks["ped_000"].visibility is True
    assert projected.tracks["ped_001"].visibility is False
    assert projected.tracks["ped_001"].age_steps == 3
    assert projected.legacy_observation["pedestrians"]["count"][0] == pytest.approx(1.0)


def test_projection_is_independent_of_scenario_agent_order() -> None:
    """Reordering source agents cannot exchange entity-ID-keyed snapshot metadata."""
    belief = _belief_fixture()
    reordered = replace(belief, agents=tuple(reversed(belief.agents)))

    first = project_belief_aware_planner_input(
        belief,
        planner_name="BeliefGuidedLocalPlanner",
    )
    second = project_belief_aware_planner_input(
        reordered,
        planner_name="BeliefGuidedLocalPlanner",
    )

    assert first.to_dict() == second.to_dict()
    assert first.tracks["ped_000"].confidence == second.tracks["ped_000"].confidence
    assert (
        first.tracks["ped_001"].covariance.tolist() == second.tracks["ped_001"].covariance.tolist()
    )


def test_reused_entity_id_exposes_no_cross_snapshot_continuity_token() -> None:
    """A reused snapshot ID has no fabricated generation or continuity token."""
    first_belief = _belief_fixture()
    first_agent = first_belief.agents[0]
    replacement_agent = replace(
        first_agent,
        position=replace(first_agent.position, mean_xy=(7.0, 7.0)),
        velocity=replace(first_agent.velocity, mean_xy=(-0.4, 0.2)),
    )
    second_belief = replace(
        first_belief,
        sim_time_s=0.6,
        agents=(replacement_agent, first_belief.agents[1]),
    )

    first = project_belief_aware_planner_input(
        first_belief,
        planner_name="BeliefGuidedLocalPlanner",
    )
    second = project_belief_aware_planner_input(
        second_belief,
        planner_name="BeliefGuidedLocalPlanner",
    )
    first_track = first.tracks["ped_000"]
    second_track = second.tracks["ped_000"]

    assert first_track.track_id == second_track.track_id == "ped_000"
    assert not np.array_equal(first_track.mean_state, second_track.mean_state)
    first_track_payload = first_track.to_dict()
    second_track_payload = second_track.to_dict()
    assert not hasattr(first_track, "identity_lifecycle_token")
    assert not hasattr(second_track, "identity_lifecycle_token")
    for payload in (first_track_payload, second_track_payload):
        assert not any(
            marker in key.lower()
            for key in payload
            for marker in ("token", "generation", "continuity")
        )
    assert "identity_lifecycle_tokens" not in first.diagnostics
    assert "identity_lifecycle_tokens" not in second.diagnostics
    assert first.diagnostics["identity_generation_available"] is False
    assert second.diagnostics["identity_generation_available"] is False


def test_projection_distinguishes_missing_empty_and_unsupported() -> None:
    """Missing belief, empty belief, and unsupported planner fallback are explicit."""
    belief = _belief_fixture()
    missing = project_belief_aware_planner_input(
        None,
        planner_name="BeliefGuidedLocalPlanner",
    )
    empty = project_belief_aware_planner_input(
        replace(belief, agents=()),
        planner_name="BeliefGuidedLocalPlanner",
    )
    unsupported = project_belief_aware_planner_input(
        belief,
        planner_name="stream_gap",
    )

    assert missing.diagnostics["status"] == "no_belief"
    assert missing.legacy_observation == {}
    assert empty.diagnostics["status"] == "empty_belief"
    assert empty.tracks == {}
    assert unsupported.diagnostics["status"] == "projection_target_not_supported"
    assert unsupported.diagnostics["fallback_reason"] == "projection_target_not_supported"
    assert unsupported.diagnostics["supported_projection_target"] is False
    assert unsupported.tracks == {}
    assert unsupported.legacy_observation["pedestrians"]["count"][0] == pytest.approx(1.0)
    assert SUPPORTED_BELIEF_AWARE_PLANNER_NAMES == frozenset({"BeliefGuidedLocalPlanner"})
    assert SUPPORTED_PROJECTION_TARGETS == SUPPORTED_BELIEF_AWARE_PLANNER_NAMES


def test_projection_counts_out_of_range_as_non_visible_not_occluded() -> None:
    """Visibility diagnostics distinguish explicit occlusion from other hidden states."""
    belief = _belief_fixture()
    out_of_range = replace(
        belief.agents[1],
        visibility_state=VisibilityState.OUT_OF_RANGE,
    )

    projected = project_belief_aware_planner_input(
        replace(belief, agents=(belief.agents[0], out_of_range)),
        planner_name="BeliefGuidedLocalPlanner",
    )

    assert projected.diagnostics["visible_track_count"] == 1
    assert projected.diagnostics["occluded_track_count"] == 0
    assert projected.diagnostics["projected_track_count"] == 2


def test_projection_rejects_malformed_track_and_keeps_safe_legacy_fallback() -> None:
    """A malformed maintained track rejects the complete typed projection."""
    belief = _belief_fixture()
    malformed = replace(belief.agents[0], radius=-1.0)
    rejected = project_belief_aware_planner_input(
        replace(belief, agents=(malformed, belief.agents[1])),
        planner_name="BeliefGuidedLocalPlanner",
    )

    assert rejected.diagnostics["status"] == "invalid_belief"
    assert "radius" in rejected.diagnostics["fallback_reason"]
    assert rejected.tracks == {}
    assert rejected.diagnostics["dropped_track_count"] == 0
    assert rejected.diagnostics["retired_track_count"] is None
    assert "pedestrians" in rejected.legacy_observation


def test_typed_records_own_arrays_and_export_deterministically() -> None:
    """Validated arrays are read-only copies and JSON export rejects non-finite data."""
    mean = np.array([1.0, 2.0, 0.1, 0.2, 0.3], dtype=np.float32)
    covariance = np.eye(5, dtype=np.float32)
    track = PlannerTrackBelief(
        track_id=2,
        mean_state=mean,
        covariance=covariance,
        confidence=0.8,
        existence_probability=0.7,
        visibility=True,
        age_steps=0,
        source="unit_test",
    )
    wrapper = BeliefAwarePlannerInput(
        legacy_observation={"array": mean},
        tracks={2: track},
        belief_step=4,
        diagnostics={"status": "projected"},
    )
    mean[0] = 99.0
    covariance[0, 0] = 99.0

    assert track.mean_state[0] == pytest.approx(1.0)
    assert track.covariance[0, 0] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="read-only"):
        track.mean_state[0] = 4.0
    payload = wrapper.to_dict()
    assert list(payload["tracks"]) == ["2"]
    assert json.loads(wrapper.to_json()) == payload
    assert payload["diagnostics"]["status"] == "projected"
    assert "identity_lifecycle_token" not in payload["tracks"]["2"]


def test_typed_record_rejects_non_psd_covariance_and_key_mismatch() -> None:
    """Standalone construction rejects unsafe covariance and mapping identity drift."""
    with pytest.raises(ValueError, match="positive semidefinite"):
        PlannerTrackBelief(
            track_id="ped_001",
            mean_state=np.zeros(5),
            covariance=np.diag([-1.0, 0.0, 0.0, 0.0, 0.0]),
            confidence=1.0,
            existence_probability=1.0,
            visibility=False,
            age_steps=1,
            source="unit_test",
        )
    with pytest.raises(ValueError, match="radius"):
        PlannerTrackBelief(
            track_id="ped_negative_radius",
            mean_state=np.array([0.0, 0.0, 0.0, 0.0, -0.1]),
            covariance=np.eye(5),
            confidence=1.0,
            existence_probability=1.0,
            visibility=True,
            age_steps=0,
            source="unit_test",
        )

    track = PlannerTrackBelief(
        track_id="ped_001",
        mean_state=np.zeros(5),
        covariance=np.eye(5),
        confidence=1.0,
        existence_probability=1.0,
        visibility=True,
        age_steps=0,
        source="unit_test",
    )
    with pytest.raises(ValueError, match="track key mismatch"):
        BeliefAwarePlannerInput(
            legacy_observation={},
            tracks={"ped_002": track},
            belief_step=0,
        )


def test_snapshot_identity_limitation_is_explicit() -> None:
    """The adapter exposes snapshot IDs without claiming lifecycle continuity."""
    projected = project_belief_aware_planner_input(
        _belief_fixture(),
        planner_name="BeliefGuidedLocalPlanner",
    )

    assert projected.diagnostics["identity_generation_available"] is False
    assert projected.diagnostics["identity_reuse_safe"] is False
    assert projected.diagnostics["lifecycle_reset_required"] is True
    assert projected.diagnostics["stateful_identity_admitted"] is False
    assert projected.diagnostics["retired_track_count"] is None
    assert projected.diagnostics["uncertainty_semantics"] == {
        "source": "adapter_derived",
        "aggregate_confidence": "min(position_confidence, velocity_confidence)",
        "state_covariance": "position_velocity_blocks_plus_zero_radius_block",
        "radius_uncertainty": "unavailable_as_modelled",
    }
    assert projected.diagnostics["retirement_tracking"] == (
        "unavailable_at_scenario_belief_boundary"
    )


def test_lazy_representation_import_fails_closed(monkeypatch) -> None:
    """Optional representation imports should report unavailable instead of leaking ImportError."""
    monkeypatch.setattr(adapter, "try_import", lambda name: None)
    assert adapter._load_scenario_belief_types() is None


def test_legacy_projection_fails_closed_for_malformed_inputs() -> None:
    """The legacy sidecar adapter distinguishes malformed observations and reports."""
    malformed_observation = SimpleNamespace(to_socnav_struct=lambda: {})
    result = project_scenario_belief_for_planner(
        malformed_observation,
        planner_key="stream_gap",
    )
    assert result.compatibility["reason"] == "malformed_legacy_observation"

    malformed_count = SimpleNamespace(
        to_socnav_struct=lambda: {"pedestrians": {"count": ["not-a-number"]}}
    )
    result = project_scenario_belief_for_planner(malformed_count, planner_key="stream_gap")
    assert result.compatibility["reason"] == "malformed_pedestrian_count"

    incomplete_report = SimpleNamespace(
        to_socnav_struct=lambda: {"pedestrians": {"count": np.asarray([1.0])}},
        to_uncertainty_report=lambda: {"agents": []},
    )
    result = project_scenario_belief_for_planner(incomplete_report, planner_key="stream_gap")
    assert result.compatibility["reason"] == "malformed_uncertainty_report"
    assert adapter._pedestrian_count({"pedestrians": []}) is None
    assert adapter._pedestrian_count({"pedestrians": {"count": []}}) is None


def test_projection_invalid_belief_fallbacks_and_alias(monkeypatch) -> None:
    """Typed projection failures retain an explicit status and safe legacy fallback."""
    belief = _belief_fixture()
    original_loader = adapter._load_scenario_belief_types
    monkeypatch.setattr(adapter, "_load_scenario_belief_types", lambda: None)
    unavailable = project_belief_aware_planner_input(
        belief,
        planner_name="BeliefGuidedLocalPlanner",
    )
    assert unavailable.diagnostics["fallback_reason"] == (
        "scenario_belief_representation_unavailable"
    )
    monkeypatch.setattr(adapter, "_load_scenario_belief_types", original_loader)

    unsupported = project_belief_aware_planner_input(
        object(),
        planner_name="BeliefGuidedLocalPlanner",
    )
    assert unsupported.diagnostics["fallback_reason"] == "belief_type_unsupported"

    bad_time = project_belief_aware_planner_input(
        replace(belief, sim_time_s=-0.1),
        planner_name="BeliefGuidedLocalPlanner",
    )
    assert bad_time.diagnostics["status"] == "invalid_belief"
    assert "sim_time_s" in bad_time.diagnostics["fallback_reason"]

    duplicate = replace(
        belief,
        agents=(belief.agents[0], replace(belief.agents[1], entity_id=belief.agents[0].entity_id)),
    )
    duplicate_result = project_belief_aware_planner_input(
        duplicate,
        planner_name="BeliefGuidedLocalPlanner",
    )
    assert duplicate_result.diagnostics["status"] == "invalid_belief"
    assert "duplicate track_id" in duplicate_result.diagnostics["fallback_reason"]

    alias = project_scenario_belief_for_belief_aware_planner(
        belief,
        planner_key="BeliefGuidedLocalPlanner",
    )
    assert (
        alias.to_dict()
        == project_belief_aware_planner_input(
            belief,
            planner_name="BeliefGuidedLocalPlanner",
        ).to_dict()
    )

    def fail_legacy(_belief):
        raise RuntimeError("legacy observation unavailable")

    monkeypatch.setattr(type(belief), "to_socnav_struct", fail_legacy)
    legacy_failure = project_belief_aware_planner_input(
        belief,
        planner_name="BeliefGuidedLocalPlanner",
    )
    assert legacy_failure.diagnostics["fallback_reason"] == "legacy_observation_unavailable"


def test_validation_helpers_reject_malformed_values() -> None:
    """Array, covariance, probability, integer, and ID validators fail closed."""

    class _BadArray:
        def __array__(self):
            raise TypeError("cannot convert")

    with pytest.raises(ValueError, match="numeric array"):
        adapter._readonly_float_array("state", _BadArray(), shape=(2,))
    with pytest.raises(ValueError, match="shape"):
        adapter._readonly_float_array("state", [1.0], shape=(2,))
    with pytest.raises(ValueError, match="numeric dtype"):
        adapter._readonly_float_array("state", ["x", "y"], shape=(2,))
    with pytest.raises(ValueError, match="finite"):
        adapter._readonly_float_array("state", [np.nan, 1.0], shape=(2,))

    with pytest.raises(ValueError, match="numeric"):
        adapter._readonly_covariance([[1.0], [1.0, 2.0]])
    assert adapter._readonly_covariance(np.eye(4)).shape == (5, 5)
    with pytest.raises(ValueError, match="numeric"):
        adapter._readonly_covariance(np.full((5, 5), "x", dtype=object))
    with pytest.raises(ValueError, match="shape"):
        adapter._readonly_covariance(np.eye(3))
    nonfinite_covariance = np.eye(5)
    nonfinite_covariance[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        adapter._readonly_covariance(nonfinite_covariance)
    asymmetric_covariance = np.eye(5)
    asymmetric_covariance[0, 1] = 1.0
    with pytest.raises(ValueError, match="symmetric"):
        adapter._readonly_covariance(asymmetric_covariance)

    with pytest.raises(ValueError, match="finite value"):
        adapter._validate_probability("probability", object())
    with pytest.raises(ValueError, match="finite value"):
        adapter._validate_probability("probability", 2.0)
    with pytest.raises(ValueError, match="finite value"):
        adapter._validate_probability("probability", np.nan)

    with pytest.raises(ValueError, match="non-negative integer"):
        adapter._validate_nonnegative_int("steps", object())
    with pytest.raises(ValueError, match="non-negative integer"):
        adapter._validate_nonnegative_int("steps", -1)
    with pytest.raises(ValueError, match="non-negative integer"):
        adapter._validate_nonnegative_int("steps", 1.5)
    with pytest.raises(ValueError, match="track_id"):
        adapter._validate_track_id(True)
    with pytest.raises(ValueError, match="track_id"):
        adapter._validate_track_id("")
    with pytest.raises(ValueError, match="track_id"):
        adapter._validate_track_id(1.5)


def test_typed_input_validation_and_json_edges() -> None:
    """Typed records own nested values and reject invalid mappings or JSON payloads."""
    track = _standalone_track()
    wrapper = BeliefAwarePlannerInput(
        legacy_observation={
            "scalar": np.float32(1.0),
            "nested": (np.asarray([2.0]), [np.asarray([3.0])]),
        },
        tracks={"ped_000": track},
        belief_step=0,
        diagnostics={"status": "projected"},
    )
    assert wrapper.projection == wrapper.diagnostics
    assert wrapper.legacy_observation["scalar"] == 1.0

    assert adapter._runtime_value_is_finite(np.asarray([1], dtype=np.int64))
    assert not adapter._runtime_value_is_finite(np.asarray([np.inf]))
    assert not adapter._runtime_value_is_finite(np.float32(np.nan))
    assert adapter._runtime_value_is_finite({"values": [1.0, (2.0,)]})
    assert not adapter._runtime_value_is_finite({"values": [float("nan")]})
    assert adapter._json_safe(np.float32(1.25)) == pytest.approx(1.25)
    with pytest.raises(ValueError, match="NaN or Inf"):
        adapter._json_safe(float("nan"))

    with pytest.raises(TypeError, match="legacy_observation"):
        BeliefAwarePlannerInput(legacy_observation=None, tracks={}, belief_step=0)
    with pytest.raises(TypeError, match="tracks"):
        BeliefAwarePlannerInput(legacy_observation={}, tracks=None, belief_step=0)
    with pytest.raises(ValueError, match="schema_version"):
        BeliefAwarePlannerInput(legacy_observation={}, tracks={}, belief_step=0, schema_version="")
    with pytest.raises(TypeError, match="diagnostics"):
        BeliefAwarePlannerInput(legacy_observation={}, tracks={}, belief_step=0, diagnostics=None)
    with pytest.raises(TypeError, match="PlannerTrackBelief"):
        BeliefAwarePlannerInput(legacy_observation={}, tracks={"bad": object()}, belief_step=0)

    numeric_track = _standalone_track(track_id=1)
    text_track = _standalone_track(track_id="1")
    colliding = BeliefAwarePlannerInput(
        legacy_observation={},
        tracks={1: numeric_track, "1": text_track},
        belief_step=0,
    )
    with pytest.raises(ValueError, match="collide"):
        colliding.to_dict()

    non_json = BeliefAwarePlannerInput(
        legacy_observation={},
        tracks={},
        belief_step=0,
        diagnostics={"unserializable": object()},
    )
    with pytest.raises(ValueError, match="JSON-safe"):
        non_json.to_dict()


def test_track_field_and_time_validation_edges() -> None:
    """Planner-track fields and canonical time metadata reject unsafe values."""
    for overrides, match in (
        ({"visibility": "yes"}, "visibility"),
        ({"source": ""}, "source"),
        ({"visibility_state": ""}, "visibility_state"),
    ):
        with pytest.raises(ValueError, match=match):
            _standalone_track(**overrides)

    with pytest.raises(ValueError, match="belief time metadata"):
        adapter._belief_step(SimpleNamespace(sim_time_s="bad", timestep_s=0.1))
    with pytest.raises(ValueError, match="sim_time_s"):
        adapter._belief_step(SimpleNamespace(sim_time_s=-1.0, timestep_s=0.1))
    with pytest.raises(ValueError, match="timestep_s"):
        adapter._belief_step(SimpleNamespace(sim_time_s=0.0, timestep_s=-1.0))
    with pytest.raises(ValueError, match="positive"):
        adapter._belief_step(SimpleNamespace(sim_time_s=1.0, timestep_s=0.0))
    with pytest.raises(ValueError, match="aligned"):
        adapter._belief_step(SimpleNamespace(sim_time_s=0.15, timestep_s=0.1))
    assert adapter._belief_step(SimpleNamespace(sim_time_s=0.0, timestep_s=0.0)) == 0
    assert adapter._belief_step(SimpleNamespace(sim_time_s=0.2, timestep_s=0.1)) == 2

    with pytest.raises(ValueError, match="last_observed_age_s must be numeric"):
        adapter._age_steps(object(), 0.1)
    with pytest.raises(ValueError, match="finite and non-negative"):
        adapter._age_steps(-1.0, 0.1)
    with pytest.raises(ValueError, match="positive observation age"):
        adapter._age_steps(1.0, 0.0)
    assert adapter._age_steps(0.0, 0.0) == 0
    assert adapter._age_steps(0.21, 0.1) == 3


def test_entity_projection_rejects_malformed_public_fields() -> None:
    """Entity-to-track projection validates every public state and provenance field."""
    agent = _belief_fixture().agents[0]

    def project(candidate):
        return adapter._planner_track_from_entity(
            candidate,
            timestep_s=0.1,
            visibility_type=VisibilityState,
        )

    with pytest.raises(ValueError, match="entity_id"):
        project(replace(agent, entity_id=object()))
    with pytest.raises(ValueError, match="visibility_state"):
        project(replace(agent, visibility_state="visible"))
    with pytest.raises(ValueError, match="state or covariance"):
        project(replace(agent, position=SimpleNamespace(mean_xy=(0.0, 0.0))))
    with pytest.raises(ValueError, match="two coordinates"):
        project(replace(agent, position=replace(agent.position, mean_xy=(0.0,))))
    with pytest.raises(ValueError, match="2x2"):
        project(
            replace(
                agent,
                position=replace(agent.position, covariance_xy=((1.0, 0.0, 0.0),) * 3),
            )
        )
    with pytest.raises(ValueError, match="radius"):
        project(replace(agent, radius="bad"))
    with pytest.raises(ValueError, match="source adapter"):
        project(replace(agent, source=replace(agent.source, adapter="")))
