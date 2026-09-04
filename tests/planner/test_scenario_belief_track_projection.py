"""Diagnostic contract tests for the identity-keyed ScenarioBelief projection.

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

from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.planner.scenario_belief_adapter import (
    BELIEF_AWARE_PLANNER_INPUT_SCHEMA_VERSION,
    SUPPORTED_BELIEF_AWARE_PLANNER_NAMES,
    BeliefAwarePlannerInput,
    PlannerTrackBelief,
    project_belief_aware_planner_input,
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


def test_projection_retains_visible_and_occluded_tracks_by_canonical_id() -> None:
    """Visible legacy rows and complete ID-keyed maintained tracks stay distinct."""
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
    assert projected.diagnostics["retained_track_count"] == 2
    assert projected.ordered_track_ids() == ("ped_000", "ped_001")
    assert projected.tracks["ped_000"].visibility is True
    assert projected.tracks["ped_001"].visibility is False
    assert projected.tracks["ped_001"].age_steps == 3
    assert projected.legacy_observation["pedestrians"]["count"][0] == pytest.approx(1.0)


def test_projection_is_independent_of_scenario_agent_order() -> None:
    """Reordering source agents cannot exchange ID-keyed uncertainty metadata."""
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
    assert unsupported.diagnostics["status"] == "unsupported_planner"
    assert unsupported.tracks == {}
    assert unsupported.legacy_observation["pedestrians"]["count"][0] == pytest.approx(1.0)
    assert SUPPORTED_BELIEF_AWARE_PLANNER_NAMES == frozenset({"BeliefGuidedLocalPlanner"})


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
    assert projected.diagnostics["retained_track_count"] == 2


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


def test_identity_lifecycle_limitation_is_explicit() -> None:
    """The adapter does not fabricate a generation for numeric-ID reuse."""
    projected = project_belief_aware_planner_input(
        _belief_fixture(),
        planner_name="BeliefGuidedLocalPlanner",
    )

    assert projected.diagnostics["identity_generation_available"] is False
    assert projected.diagnostics["identity_reuse_safe"] is False
    assert projected.diagnostics["lifecycle_reset_required"] is True
    assert projected.diagnostics["retirement_tracking"] == (
        "unavailable_at_scenario_belief_boundary"
    )
