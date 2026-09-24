"""Identity-keyed planner projection tests for maintained pedestrian tracks."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from robot_sf.gym_env.unified_config import ObservationVisibilitySettings, RobotSimulationConfig
from robot_sf.planner.scenario_belief_adapter import (
    SUPPORTED_IDENTITY_SAFE_PLANNER_NAMES,
    TRACK_EXISTENCE_PROBABILITY_SEMANTICS,
    BeliefAwarePlannerInput,
    project_identity_safe_scenario_belief,
    project_scenario_belief_for_planner,
)
from robot_sf.representation import (
    BeliefSource,
    EntityBelief,
    Estimate2D,
    ScenarioBelief,
    VisibilityState,
)
from robot_sf.representation.scenario_belief import GoalBelief
from robot_sf.sensor.pedestrian_tracking import (
    PedestrianObservationSnapshot,
    PedestrianTracker,
    PedestrianTrackingConfig,
    PedestrianTrackingResult,
    TrackStatus,
)
from robot_sf.sensor.socnav_observation import SocNavObservationFusion

_PLANNER = "BeliefGuidedLocalPlanner"


def _base_belief() -> ScenarioBelief:
    """Build an explicit observation envelope without simulator-owned agent state."""
    source = BeliefSource(adapter="test_observation")
    ego = EntityBelief(
        entity_id="robot_0",
        entity_type="ego_robot",
        position=Estimate2D.point([0.0, 0.0], confidence=1.0, variance=0.01),
        velocity=Estimate2D.point([0.0, 0.0], confidence=1.0, variance=0.01),
        radius=0.35,
        heading=0.0,
        existence_probability=1.0,
        visibility_state=VisibilityState.VISIBLE,
        source=source,
    )
    goal = Estimate2D.point([4.0, 0.0], confidence=1.0, variance=0.01)
    return ScenarioBelief(
        frame_id="map",
        sim_time_s=0.0,
        timestep_s=0.1,
        map_size=(10.0, 10.0),
        ego=ego,
        ego_speed=(0.0, 0.0),
        goals=(GoalBelief(current=goal, next=goal, source=source),),
        agents=(),
        source_summary=source,
        max_pedestrians=8,
        pedestrian_radius=0.3,
    )


def _snapshot(
    step: int,
    positions: list[list[float]] | np.ndarray,
    *,
    visible: list[bool] | None = None,
    valid: list[bool] | None = None,
    velocity_valid: list[bool] | None = None,
    position_covariances: np.ndarray | None = None,
) -> PedestrianObservationSnapshot:
    """Build an observation-only tracker input for one step."""
    positions_array = np.asarray(positions, dtype=float).reshape(-1, 2)
    row_count = len(positions_array)
    visible_mask = np.ones(row_count, dtype=bool) if visible is None else np.asarray(visible)
    valid_mask = np.ones(row_count, dtype=bool) if valid is None else np.asarray(valid)
    return PedestrianObservationSnapshot(
        timestamp_s=step * 0.1,
        step_index=step,
        coordinate_frame="global_xy",
        robot_pose_global=(0.0, 0.0, 0.0),
        positions=positions_array,
        velocities=np.zeros_like(positions_array),
        velocity_valid_mask=velocity_valid,
        valid_mask=valid_mask,
        visible_mask=visible_mask,
        position_covariances=position_covariances,
        radius=0.3,
    )


def _tracker(**overrides: object) -> PedestrianTracker:
    """Return a low-noise tracker with an explicit short retirement horizon."""
    values: dict[str, object] = {
        "enabled": True,
        "process_noise": 0.0,
        "confirmation_steps": 1,
        "max_missed_seconds": 10.0,
        "max_missed_steps": 3,
        "history_capacity": 4,
        "position_gate_threshold": 100.0,
        "velocity_gate_threshold": 100.0,
    }
    values.update(overrides)
    return PedestrianTracker(PedestrianTrackingConfig(**values))


def _project(
    result: PedestrianTrackingResult,
    *,
    namespace: str | None = "sensor-a",
    planner_name: str = _PLANNER,
    belief: ScenarioBelief | None = None,
    legacy_observation=None,
    belief_step: int | None = None,
):
    """Project one canonical result alongside its unchanged public observation envelope."""
    selected_belief = _base_belief() if belief is None else belief
    return project_identity_safe_scenario_belief(
        selected_belief,
        planner_name=planner_name,
        belief_step=result.step_index if belief_step is None else belief_step,
        tracking_result=result,
        tracker_namespace=namespace,
        legacy_observation=legacy_observation,
    )


def test_fusion_exposes_current_immutable_result_without_observation_schema_change() -> None:
    """The observation adapter exposes tracking state as a side channel only."""
    simulator = SimpleNamespace(
        ped_pos=np.array([[1.0, 0.0]], dtype=np.float32),
        ped_vel=np.array([[0.0, 0.0]], dtype=np.float32),
        robots=[
            SimpleNamespace(
                pose=((0.0, 0.0), 0.0),
                current_speed=np.array([0.0, 0.0], dtype=np.float32),
                config=SimpleNamespace(radius=0.35),
            )
        ],
        goal_pos=[np.array([4.0, 0.0], dtype=np.float32)],
        next_goal_pos=[None],
        map_def=SimpleNamespace(width=10.0, height=10.0, obstacles=[]),
        config=SimpleNamespace(time_per_step_in_secs=0.1),
    )
    config = RobotSimulationConfig()
    config.observation_visibility = ObservationVisibilitySettings(
        include_track_ids=True,
        tracking_config={"confirmation_steps": 1},
    )
    fusion = SocNavObservationFusion(simulator, config, max_pedestrians=4)

    observation = fusion.next_obs()
    result = fusion.current_tracking_result

    assert result is not None
    assert result.reset_epoch == 0
    assert set(observation["pedestrians"]) == {
        "positions",
        "velocities",
        "radius",
        "count",
        "track_id",
    }
    assert result.tracks[0].position_global_xy.flags.writeable is False
    fusion.reset_cache()
    assert fusion.current_tracking_result is None


def test_projection_is_track_keyed_and_keeps_coasted_tracks_without_row_join() -> None:
    """Current associations determine visibility while track IDs own all metadata."""
    tracker = _tracker()
    first = tracker.update(_snapshot(0, [[1.0, 0.0], [3.0, 0.0]]))
    second = tracker.update(
        _snapshot(
            1,
            [[3.1, 0.0], [0.9, 0.0]],
            position_covariances=np.array(
                [np.eye(2) * 0.7, np.eye(2) * 0.2],
                dtype=float,
            ),
            velocity_valid=[True, False],
        )
    )
    projection = _project(second)

    assert first.track(1) is not None
    first_projection = _project(first)
    assert first.associations == ()
    assert first.track(1).observed_this_step is True
    assert first_projection.tracks[1].visibility is True
    assert first_projection.tracks[2].visibility is True
    assert projection.diagnostics.status == "supported"
    assert projection.diagnostics.ordered_track_ids == (1, 2)
    assert projection.belief_step == second.step_index
    np.testing.assert_allclose(projection.tracks[1].mean_state[:2], [0.9, 0.0], atol=0.03)
    np.testing.assert_allclose(projection.tracks[2].mean_state[:2], [3.1, 0.0], atol=0.05)
    assert projection.tracks[1].confidence == second.track(1).association_confidence
    assert projection.tracks[2].confidence == second.track(2).association_confidence
    assert projection.tracks[1].confidence < projection.tracks[2].confidence
    assert projection.tracks[1].existence_probability == 1.0
    assert projection.tracks[2].existence_probability == 1.0
    assert projection.tracks[1].existence_probability_calibrated is False
    assert (
        projection.tracks[1].existence_probability_semantics
        == TRACK_EXISTENCE_PROBABILITY_SEMANTICS
    )
    assert projection.tracks[1].confidence < 1.0
    np.testing.assert_allclose(
        projection.tracks[1].covariance[:2, :2], second.track(1).position_covariance
    )
    np.testing.assert_allclose(
        projection.tracks[2].covariance[:2, :2], second.track(2).position_covariance
    )
    assert np.trace(projection.tracks[1].covariance[:2, :2]) < np.trace(
        projection.tracks[2].covariance[:2, :2]
    )
    assert projection.tracks[1].visibility is True
    assert projection.tracks[1].last_observed_step == second.step_index
    assert projection.diagnostics.identity_lifecycle_tokens == (
        '["sensor-a",0,1]',
        '["sensor-a",0,2]',
    )

    coasted = tracker.update(_snapshot(2, [[0.9, 0.0]], visible=[False]))
    coasted_projection = _project(coasted)
    assert coasted.track(1).last_observation_slot == 1
    assert all(association.track_id != 1 for association in coasted.associations)
    assert coasted_projection.tracks[1].visibility is False
    assert coasted.track(1).observed_this_step is False
    assert coasted_projection.tracks[1].missed_steps == 1
    assert coasted_projection.tracks[1].last_observed_step == 1
    assert coasted_projection.diagnostics.occluded_track_count >= 1
    assert coasted_projection.diagnostics.stale_track_count == 2


def test_reacquisition_preserves_token_and_reset_reuse_changes_generation() -> None:
    """A retained ID keeps its token; reset epoch distinguishes numeric-ID reuse."""
    tracker = _tracker()
    first = tracker.update(_snapshot(0, [[1.0, 0.0]]))
    first_projection = _project(first)
    token = first_projection.tracks[1].lifecycle_token

    tracker.update(_snapshot(1, [[1.0, 0.0]], visible=[False]))
    reacquired = tracker.update(_snapshot(2, [[1.1, 0.0]]))
    reacquired_projection = _project(reacquired)
    assert reacquired_projection.tracks[1].lifecycle_token == token
    assert reacquired_projection.tracks[1].visibility is True

    tracker.reset()
    reused = tracker.update(_snapshot(0, [[1.1, 0.0]]))
    reused_projection = _project(reused)
    assert reused.track(1) is not None
    assert reused.reset_epoch == first.reset_epoch + 1
    assert reused_projection.tracks[1].lifecycle_token != token


def test_retirement_is_reported_and_not_projected_as_a_maintained_track() -> None:
    """Retired IDs remain explicit in supported and failed mixed-result diagnostics."""
    tracker = _tracker(max_missed_steps=1)
    tracker.update(_snapshot(0, [[1.0, 0.0]]))
    tracker.update(_snapshot(1, [[0.0, 0.0]], visible=[False]))
    retired = tracker.update(
        _snapshot(
            2,
            [[0.0, 0.0], [30.0, 0.0]],
            valid=[False, True],
            visible=[False, True],
        )
    )
    projection = _project(retired)
    failed = project_identity_safe_scenario_belief(
        None,
        planner_name=_PLANNER,
        belief_step=retired.step_index,
        tracking_result=retired,
        tracker_namespace="sensor-a",
    )

    assert retired.track(1).status.value == "retired"
    assert retired.track(2).status.value in {"tentative", "confirmed"}
    assert tuple(projection.tracks) == (2,)
    assert projection.diagnostics.retired_track_count == 1
    assert projection.diagnostics.retired_track_ids == (1,)
    assert projection.diagnostics.identity_lifecycle_tokens == (
        '["sensor-a",0,1]',
        '["sensor-a",0,2]',
    )
    assert projection.diagnostics.dropped_track_count == 0
    assert failed.diagnostics.retired_track_count == 1
    assert failed.diagnostics.retired_track_ids == (1,)
    assert failed.diagnostics.dropped_track_count == 1
    assert failed.diagnostics.per_reason_drop_count == (("missing_scenario_belief", 1),)


def test_namespace_and_reset_epoch_are_required_for_identity_projection() -> None:
    """Missing lifecycle provenance fails closed and namespaces separate trackers."""
    result = _tracker().update(_snapshot(0, [[1.0, 0.0]]))
    other_namespace = _project(result, namespace="sensor-b")
    missing_namespace = _project(result, namespace=None)
    missing_epoch = _project(replace(result, reset_epoch=None))

    assert other_namespace.tracks[1].lifecycle_token != _project(result).tracks[1].lifecycle_token
    assert missing_namespace.diagnostics.status == "invalid"
    assert missing_namespace.diagnostics.fallback_reason == "missing_tracker_namespace"
    assert missing_namespace.diagnostics.dropped_track_count == 1
    assert missing_epoch.diagnostics.status == "invalid"
    assert missing_epoch.diagnostics.fallback_reason == "missing_reset_epoch"
    assert missing_epoch.diagnostics.per_reason_drop_count == (("missing_reset_epoch", 1),)


def test_missing_empty_unsupported_and_invalid_inputs_are_distinct() -> None:
    """No belief, a valid empty tracker result, unsupported planner, and bad radius differ."""
    nonempty = _tracker().update(_snapshot(0, [[1.0, 0.0]]))
    empty = _tracker().update(_snapshot(0, np.empty((0, 2))))
    missing = project_identity_safe_scenario_belief(
        None,
        planner_name=_PLANNER,
        belief_step=nonempty.step_index,
        tracking_result=nonempty,
        tracker_namespace="sensor-a",
        legacy_observation={"sentinel": np.array([3.0])},
    )
    empty_projection = _project(empty)
    unsupported = _project(
        nonempty,
        planner_name="unregistered_planner",
        legacy_observation={"sentinel": np.array([3.0])},
    )
    invalid = _project(empty, belief=replace(_base_belief(), pedestrian_radius=float("nan")))

    assert missing.diagnostics.status == "missing"
    assert missing.diagnostics.fallback_reason == "missing_scenario_belief"
    assert missing.diagnostics.dropped_track_count == 1
    assert missing.diagnostics.per_reason_drop_count == (("missing_scenario_belief", 1),)
    assert empty_projection.diagnostics.status == "empty"
    assert empty_projection.tracks == {}
    assert unsupported.diagnostics.status == "unsupported"
    assert unsupported.diagnostics.fallback_reason == "unsupported_planner"
    assert unsupported.diagnostics.dropped_track_count == 1
    assert unsupported.diagnostics.per_reason_drop_count == (("unsupported_planner", 1),)
    np.testing.assert_array_equal(unsupported.legacy_observation["sentinel"], [3.0])
    assert invalid.diagnostics.status == "invalid"
    assert invalid.diagnostics.dropped_track_count == 0
    assert SUPPORTED_IDENTITY_SAFE_PLANNER_NAMES == frozenset({_PLANNER})


def test_disabled_tracker_is_unavailable_and_projection_step_must_match() -> None:
    """Disabled tracking differs from enabled-empty, and step mismatch fails closed."""
    enabled_empty = _tracker().update(_snapshot(0, np.empty((0, 2))))
    disabled = _tracker(enabled=False).update(_snapshot(0, np.empty((0, 2))))
    enabled_nonempty = _tracker().update(_snapshot(0, [[1.0, 0.0]]))

    empty_projection = _project(enabled_empty, belief_step=0)
    disabled_projection = _project(disabled, belief_step=0)
    matched = _project(enabled_nonempty, belief_step=0)
    mismatch = _project(enabled_nonempty, belief_step=1)

    assert enabled_empty.diagnostics.enabled is True
    assert disabled.diagnostics.enabled is False
    assert empty_projection.diagnostics.status == "empty"
    assert disabled_projection.diagnostics.status == "unavailable"
    assert disabled_projection.diagnostics.fallback_reason == "tracking_disabled"
    assert matched.diagnostics.status == "supported"
    assert matched.belief_step == 0
    assert mismatch.diagnostics.status == "invalid"
    assert mismatch.diagnostics.fallback_reason == "belief_step_mismatch"
    assert mismatch.belief_step == 1
    assert mismatch.diagnostics.dropped_track_count == 1
    assert mismatch.diagnostics.per_reason_drop_count == (("belief_step_mismatch", 1),)


def test_current_association_to_lost_track_is_rejected() -> None:
    """The producer rejects a current association that conflicts with LOST status."""
    tracker = _tracker()
    tracker.update(_snapshot(0, [[1.0, 0.0]]))
    matched = tracker.update(_snapshot(1, [[1.0, 0.0]]))
    assert matched.associations[0].track_id == 1

    lost_track = replace(
        matched.track(1),
        status=TrackStatus.LOST,
        missed_steps=1,
        observed_this_step=False,
    )
    with pytest.raises(ValueError, match="associations conflict"):
        replace(matched, tracks=(lost_track,))


def test_projection_arrays_are_immutable_and_export_is_deterministic_json() -> None:
    """Projection arrays own their storage and JSON export is stable and finite-only."""
    source_positions = np.array([[1.0, 0.0]], dtype=float)
    tracker = _tracker()
    result = tracker.update(_snapshot(0, source_positions))
    projection = _project(result)
    source_positions[:] = 99.0

    assert projection.tracks[1].mean_state.flags.writeable is False
    assert projection.tracks[1].covariance.flags.writeable is False
    with pytest.raises(ValueError):
        projection.tracks[1].mean_state[0] = -1.0
    assert projection.to_json() == projection.to_json()
    exported = json.loads(projection.to_json())
    assert list(exported["tracks"]) == ["1"]
    assert exported["tracks"]["1"]["existence_probability_calibrated"] is False
    assert (
        exported["tracks"]["1"]["existence_probability_semantics"]
        == TRACK_EXISTENCE_PROBABILITY_SEMANTICS
    )
    assert "goal" not in exported["tracks"]["1"]
    assert "force" not in exported["tracks"]["1"]

    with pytest.raises(ValueError, match="lifecycle_token"):
        replace(projection.tracks[1], lifecycle_token=json.dumps(["other", 0, 1]))
    with pytest.raises(ValueError, match="wrapper belief_step"):
        BeliefAwarePlannerInput(
            legacy_observation=projection.legacy_observation,
            tracks=projection.tracks,
            belief_step=projection.belief_step + 1,
            schema_version=projection.schema_version,
            diagnostics=projection.diagnostics,
        )

    non_finite = _project(
        result,
        legacy_observation={"sentinel": np.array([np.inf])},
    )
    with pytest.raises(ValueError, match="non-finite"):
        non_finite.to_json()


def test_observation_rows_are_preserved_as_legacy_data_and_never_joined_by_index() -> None:
    """Swapping legacy rows leaves keyed track metadata tied to its producer track ID."""
    tracker = _tracker()
    tracker.update(_snapshot(0, [[1.0, 0.0], [3.0, 0.0]]))
    result = tracker.update(_snapshot(1, [[3.1, 0.0], [0.9, 0.0]]))
    legacy_a = _base_belief().to_socnav_struct()
    legacy_b = _base_belief().to_socnav_struct()
    legacy_a["pedestrians"]["positions"][:2] = [[0.9, 0.0], [3.1, 0.0]]
    legacy_a["pedestrians"]["velocities"][:2] = [[0.0, 0.0], [0.0, 0.0]]
    legacy_a["pedestrians"]["count"][0] = 2.0
    legacy_b["pedestrians"]["positions"][:2] = [[3.1, 0.0], [0.9, 0.0]]
    legacy_b["pedestrians"]["velocities"][:2] = [[0.0, 0.0], [0.0, 0.0]]
    legacy_b["pedestrians"]["count"][0] = 2.0

    projection_a = _project(result, legacy_observation=legacy_a)
    projection_b = _project(result, legacy_observation=legacy_b)

    np.testing.assert_allclose(projection_a.tracks[1].mean_state, projection_b.tracks[1].mean_state)
    np.testing.assert_allclose(projection_a.tracks[2].covariance, projection_b.tracks[2].covariance)
    np.testing.assert_array_equal(
        projection_a.legacy_observation["pedestrians"]["positions"],
        legacy_a["pedestrians"]["positions"],
    )
    np.testing.assert_array_equal(
        projection_b.legacy_observation["pedestrians"]["positions"],
        legacy_b["pedestrians"]["positions"],
    )


@pytest.mark.parametrize(
    (
        "planner_name",
        "belief_step",
        "tracking_result",
        "belief",
        "legacy_observation",
        "status",
        "reason",
    ),
    [
        (" ", 0, "valid", "valid", None, "invalid", "invalid_planner_name"),
        (_PLANNER, -1, "valid", "valid", None, "invalid", "invalid_belief_step"),
        (_PLANNER, 0, "valid", "missing_radius", None, "invalid", "invalid_scenario_belief"),
        (_PLANNER, 0, None, "valid", None, "missing", "missing_tracking_result"),
        (_PLANNER, 0, object(), "valid", None, "invalid", "invalid_tracking_result"),
        (_PLANNER, 0, "valid", "valid", [], "invalid", "invalid_legacy_observation"),
    ],
)
def test_projection_input_contract_fails_closed_with_specific_diagnostics(
    planner_name: str,
    belief_step: int,
    tracking_result: object,
    belief: object,
    legacy_observation: object,
    status: str,
    reason: str,
) -> None:
    """Malformed public inputs never produce keyed planner tracks."""
    result = _tracker().update(_snapshot(0, [[1.0, 0.0]]))
    selected_result = result if tracking_result == "valid" else tracking_result
    selected_belief = _base_belief() if belief == "valid" else belief
    if belief == "missing_radius":
        selected_belief = SimpleNamespace(to_socnav_struct=lambda: {})

    projection = project_identity_safe_scenario_belief(
        selected_belief,
        planner_name=planner_name,
        belief_step=belief_step,
        tracking_result=selected_result,
        tracker_namespace="sensor-a",
        legacy_observation=legacy_observation,
    )

    assert projection.diagnostics.status == status
    assert projection.diagnostics.fallback_reason == reason
    assert projection.tracks == {}


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"track_id": 0}, "track_id"),
        ({"belief_step": -1}, "belief_step"),
        ({"last_observed_step": 2}, "last_observed_step"),
        ({"age_steps": -1}, "age_steps"),
        ({"missed_steps": -1}, "missed_steps"),
        ({"reset_epoch": -1}, "reset_epoch"),
        ({"tracker_namespace": " "}, "tracker_namespace"),
        ({"status": "retired"}, "status"),
        ({"visibility": 1}, "visibility"),
        ({"confidence": float("nan")}, "confidence"),
        ({"existence_probability": 0.9}, "1.0 keep-alive"),
        ({"existence_probability_calibrated": True}, "uncalibrated"),
    ],
)
def test_projected_track_rejects_invalid_public_lifecycle_and_probability_fields(
    changes: dict[str, object], message: str
) -> None:
    """A planner sidecar cannot represent invalid identity, timing, or existence semantics."""
    result = _tracker().update(_snapshot(0, [[1.0, 0.0]]))
    track = _project(result).tracks[1]

    with pytest.raises((TypeError, ValueError), match=message):
        replace(track, **changes)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"mean_state": np.array([1.0, 0.0, 0.0, 0.0])}, "finite.*vector"),
        ({"mean_state": np.array([1.0, 0.0, 0.0, 0.0, np.inf])}, "finite.*vector"),
        ({"mean_state": np.array([1.0, 0.0, 0.0, 0.0, -0.1])}, "radius"),
        ({"covariance": np.zeros((3, 3))}, "finite 4x4"),
        ({"covariance": np.full((4, 4), np.inf)}, "finite 4x4"),
        (
            {
                "covariance": np.array(
                    [
                        [1.0, 0.5, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                )
            },
            "symmetric",
        ),
    ],
)
def test_projected_track_rejects_invalid_state_and_covariance_arrays(
    changes: dict[str, object], message: str
) -> None:
    """Projected state vectors and covariance matrices retain their finite symmetric contract."""
    result = _tracker().update(_snapshot(0, [[1.0, 0.0]]))
    track = _project(result).tracks[1]

    with pytest.raises(ValueError, match=message):
        replace(track, **changes)


def test_wrapper_rejects_schema_order_key_and_legacy_observation_mismatches() -> None:
    """The exported wrapper enforces stable schema, identity-key, and legacy mapping contracts."""
    tracker = _tracker()
    tracker.update(_snapshot(0, [[1.0, 0.0], [3.0, 0.0]]))
    result = tracker.update(_snapshot(1, [[1.0, 0.0], [3.0, 0.0]]))
    projection = _project(result)

    with pytest.raises(ValueError, match="schema_version"):
        replace(projection, schema_version="future-schema")
    with pytest.raises(ValueError, match="ascending track_id"):
        replace(projection, tracks=dict(reversed(tuple(projection.tracks.items()))))
    with pytest.raises(ValueError, match="keys must match"):
        replace(projection, tracks={99: projection.tracks[1]})
    with pytest.raises(TypeError, match="mapping or None"):
        replace(projection, legacy_observation=[])


@pytest.mark.parametrize(
    ("legacy_observation", "reason"),
    [
        ({"world": {}}, "malformed_legacy_observation"),
        ({"pedestrians": []}, "malformed_legacy_observation"),
        ({"pedestrians": {"count": []}}, "malformed_pedestrian_count"),
        ({"pedestrians": {"count": [float("nan")]}}, "malformed_pedestrian_count"),
        ({"pedestrians": {"count": ["not-a-count"]}}, "malformed_pedestrian_count"),
        ({"pedestrians": {"count": [1]}}, "malformed_uncertainty_report"),
    ],
)
def test_uncertainty_projection_reports_malformed_legacy_and_report_contracts(
    legacy_observation: dict[str, object], reason: str
) -> None:
    """The legacy uncertainty adapter fails closed when its observation or report is malformed."""
    belief = SimpleNamespace(
        to_socnav_struct=lambda: legacy_observation,
        to_uncertainty_report=lambda: {"agents": []},
    )

    projection = project_scenario_belief_for_planner(belief, planner_key="stream_gap")

    assert projection.compatibility["status"] == "fail_closed"
    assert projection.compatibility["reason"] == reason
    assert projection.compatibility["uncertainty_consumed"] is False
    assert "uncertainty" not in projection.observation.get("pedestrians", {})


def test_uncertainty_projection_marks_unsupported_planners_without_adding_sidecar() -> None:
    """Only a supported planner receives uncertainty rows; legacy rows remain available."""
    belief = _base_belief()
    projection = project_scenario_belief_for_planner(belief, planner_key="legacy_planner")

    assert projection.compatibility["status"] == "fail_closed"
    assert projection.compatibility["reason"] == "unsupported_uncertainty_planner"
    assert projection.compatibility["uncertainty_consumed"] is False
    assert "uncertainty" not in projection.observation["pedestrians"]
    assert projection.observation["pedestrians"]["uncertainty_compatibility"] == (
        projection.compatibility
    )


def test_uncertainty_projection_adds_only_report_rows_visible_in_legacy_observation() -> None:
    """A supported planner receives a copied sidecar aligned to the active legacy count."""
    legacy_observation = {"pedestrians": {"count": np.array([1.0]), "positions": [[2.0, 0.0]]}}
    report_rows = [
        {"entity_id": "pedestrian-a", "position_confidence": 0.8},
        {"entity_id": "outside-active-count", "position_confidence": 0.4},
    ]
    belief = SimpleNamespace(
        to_socnav_struct=lambda: legacy_observation,
        to_uncertainty_report=lambda: {"agents": report_rows},
    )

    projection = project_scenario_belief_for_planner(belief, planner_key="stream_gap")

    pedestrians = projection.observation["pedestrians"]
    assert pedestrians["uncertainty"] == report_rows[:1]
    assert pedestrians["uncertainty"][0] is not report_rows[0]
    assert projection.compatibility == pedestrians["uncertainty_compatibility"]
    assert projection.compatibility["status"] == "compatible"
    assert projection.compatibility["uncertainty_consumed"] is True
    assert projection.compatibility["consumed_agent_count"] == 1
    assert projection.compatibility["claim_boundary"] == "diagnostic_interface_smoke"


def test_json_export_normalizes_nested_numpy_values_and_rejects_unsupported_data() -> None:
    """Full wrapper export normalizes nested values and rejects lossy or non-finite data."""
    result = _tracker().update(_snapshot(0, [[1.0, 0.0]]))
    projection = _project(
        result,
        legacy_observation={
            "scalar": np.float32(0.25),
            "tuple": (np.int64(3), True),
            "integer_keys": {2: "second", 1: "first"},
        },
    )

    exported = json.loads(projection.to_json())["legacy_observation"]
    assert exported == {
        "integer_keys": {"1": "first", "2": "second"},
        "scalar": pytest.approx(0.25),
        "tuple": [3, True],
    }

    bad_key = _project(result, legacy_observation={("tuple",): "unsupported key"})
    unsupported_value = _project(result, legacy_observation={"value": object()})
    non_finite_scalar = _project(result, legacy_observation={"value": float("nan")})
    with pytest.raises(TypeError, match="mapping keys"):
        bad_key.to_json()
    with pytest.raises(TypeError, match="unsupported JSON-safe projection value"):
        unsupported_value.to_json()
    with pytest.raises(ValueError, match="NaN and Inf"):
        non_finite_scalar.to_json()


def test_unobserved_track_without_occlusion_marker_counts_as_stale_only() -> None:
    """Staleness is reported independently when an active track has no occlusion marker."""
    tracker = _tracker()
    tracker.update(_snapshot(0, [[1.0, 0.0]]))
    result = tracker.update(_snapshot(1, np.empty((0, 2))))
    projection = _project(result)

    assert projection.diagnostics.status == "supported"
    assert projection.diagnostics.visible_track_count == 0
    assert projection.diagnostics.occluded_track_count == 0
    assert projection.diagnostics.stale_track_count == 1
    assert projection.tracks[1].visibility is False
