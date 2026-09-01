"""Contract tests for observation-only public goal-candidate generation."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from robot_sf.prediction.goal_candidate_provider import (
    CandidatePathMode,
    CandidatePriorMode,
    GoalCandidateProvider,
    GoalCandidateProviderConfig,
    GoalCandidateSource,
    PublicGoalCandidateRecord,
    generate_goal_candidates,
    generate_goal_candidates_from_map,
    public_goal_map_inputs_from_definition,
)
from robot_sf.prediction.goal_intention import GoalCandidateRole


def _route_record(
    *,
    source_id: str = "route-a",
    final: tuple[float, float] = (10.0, 0.0),
    route_signature: str | None = "route-a",
    path: tuple[tuple[float, float], ...] = ((0.0, 0.0), (10.0, 0.0)),
    prior_weight: float | None = None,
) -> PublicGoalCandidateRecord:
    """Build a public route terminal fixture."""

    return PublicGoalCandidateRecord(
        source=GoalCandidateSource.PEDESTRIAN_ROUTE_TERMINAL,
        source_id=source_id,
        position=final,
        route_signature=route_signature,
        path_points=path,
        path_mode=CandidatePathMode.PLANNER_PATH,
        prior_weight=prior_weight,
        provenance_refs=("fixture:public-route",),
    )


def test_empty_public_map_emits_direction_only_rays_and_unknown() -> None:
    """An empty public map does not force a finite destination hypothesis."""

    result = generate_goal_candidates(())

    assert len(result.candidate_set.candidates) == 5
    rays = [
        candidate
        for candidate in result.candidate_set.candidates
        if candidate.role is GoalCandidateRole.OPEN_RAY
    ]
    assert len(rays) == 4
    assert all(candidate.position is None for candidate in rays)
    assert all(candidate.direction is not None for candidate in rays)
    assert all(math.isclose(math.hypot(*candidate.direction), 1.0) for candidate in rays)
    assert any(
        candidate.role is GoalCandidateRole.UNKNOWN for candidate in result.candidate_set.candidates
    )
    assert result.candidate_set_digest == generate_goal_candidates(()).candidate_set_digest


def test_source_merge_preserves_provenance_and_derives_path_tangent() -> None:
    """Equivalent zone and route sources merge while retaining an active waypoint."""

    records = (
        PublicGoalCandidateRecord(
            source=GoalCandidateSource.MAP_DESTINATION_ZONE,
            source_id="zone-east",
            position=(1.0, 1.0),
            provenance_refs=("fixture:zone",),
        ),
        PublicGoalCandidateRecord(
            source=GoalCandidateSource.PEDESTRIAN_ROUTE_TERMINAL,
            source_id="route-east",
            position=(1.0, 1.0),
            route_signature="east",
            path_points=((0.0, 0.0), (0.0, 1.0), (1.0, 1.0)),
            provenance_refs=("fixture:route",),
        ),
    )
    config = GoalCandidateProviderConfig(waypoint_lookahead_m=0.5)
    first = generate_goal_candidates(records, config=config, observed_position_global=(0.0, 0.0))
    second = generate_goal_candidates(
        tuple(reversed(records)), config=config, observed_position_global=(0.0, 0.0)
    )

    assert first.candidate_set_digest == second.candidate_set_digest
    assert [candidate.id for candidate in first.candidate_set.candidates] == [
        candidate.id for candidate in second.candidate_set.candidates
    ]
    final = next(
        candidate
        for candidate in first.candidate_set.candidates
        if candidate.role is GoalCandidateRole.FINAL_DESTINATION
    )
    assert final.source == GoalCandidateSource.MAP_DESTINATION_ZONE.value
    assert final.route_signature == "east"
    assert "source_id:zone-east" in final.provenance_refs
    assert "source_id:route-east" in final.provenance_refs
    assert final.path_mode == CandidatePathMode.PLANNER_PATH.value
    active = next(
        candidate
        for candidate in first.candidate_set.candidates
        if candidate.role is GoalCandidateRole.ACTIVE_WAYPOINT
    )
    assert active.parent_destination_id == final.id
    assert active.path_tangent == pytest.approx((0.0, 1.0))
    assert active.position == pytest.approx((0.0, 0.5))


def test_distinct_route_signatures_retain_two_homotopies() -> None:
    """Same endpoint with distinct public route signatures remains distinguishable."""

    records = (
        _route_record(
            source_id="left", route_signature="left", path=((0.0, 0.0), (0.0, 2.0), (10.0, 0.0))
        ),
        _route_record(
            source_id="right", route_signature="right", path=((0.0, 0.0), (0.0, -2.0), (10.0, 0.0))
        ),
        _route_record(
            source_id="third", route_signature="third", path=((0.0, 0.0), (2.0, 0.0), (10.0, 0.0))
        ),
    )
    result = generate_goal_candidates(
        records,
        observed_position_global=(0.0, 0.0),
        config=GoalCandidateProviderConfig(homotopy_count=2),
    )

    finals = [
        candidate
        for candidate in result.candidate_set.candidates
        if candidate.role is GoalCandidateRole.FINAL_DESTINATION
    ]
    assert len(finals) == 2
    assert {candidate.route_signature for candidate in finals} <= {"left", "right", "third"}
    active = [
        candidate
        for candidate in result.candidate_set.candidates
        if candidate.role is GoalCandidateRole.ACTIVE_WAYPOINT
    ]
    assert {candidate.parent_destination_id for candidate in active} == {
        candidate.id for candidate in finals
    }
    assert any(record.reason == "homotopy_cap" for record in result.rejected_records)


def test_obstacle_crossing_is_rejected_with_explicit_reason() -> None:
    """A path crossing a declared obstacle is never silently retained."""

    result = generate_goal_candidates(
        (_route_record(path=((0.0, 0.0), (10.0, 0.0))),),
        observed_position_global=(0.0, 0.0),
        obstacles=(((4.0, -1.0), (6.0, -1.0), (6.0, 1.0), (4.0, 1.0)),),
    )

    assert not any(
        candidate.role is GoalCandidateRole.FINAL_DESTINATION
        for candidate in result.candidate_set.candidates
    )
    assert any(
        record.reason == "path_intersects_declared_obstacle" for record in result.rejected_records
    )
    assert any(
        candidate.role is GoalCandidateRole.UNKNOWN for candidate in result.candidate_set.candidates
    )


def test_disabling_derived_waypoints_keeps_only_the_selected_source_role() -> None:
    """Source selection applies to derived active-waypoint candidates too."""

    result = generate_goal_candidates(
        (_route_record(),),
        observed_position_global=(0.0, 0.0),
        config=GoalCandidateProviderConfig(
            enabled_sources=(GoalCandidateSource.PEDESTRIAN_ROUTE_TERMINAL,)
        ),
    )

    assert all(
        candidate.role is not GoalCandidateRole.ACTIVE_WAYPOINT
        for candidate in result.candidate_set.candidates
    )
    assert any(
        candidate.role is GoalCandidateRole.FINAL_DESTINATION
        for candidate in result.candidate_set.candidates
    )


def test_public_prior_cap_and_input_order_are_deterministic() -> None:
    """Public-prior ranking is bounded and independent of source iteration order."""

    records = tuple(
        _route_record(
            source_id=f"route-{index}",
            final=(float(index + 1), 0.0),
            route_signature=f"route-{index}",
            path=((0.0, 0.0), (float(index + 1), 0.0)),
            prior_weight=float(index + 1),
        )
        for index in range(5)
    )
    config = GoalCandidateProviderConfig(
        prior_mode=CandidatePriorMode.PUBLIC,
        final_destination_cap=2,
        active_waypoint_cap=2,
    )
    first = generate_goal_candidates(records, config=config, observed_position_global=(0.0, 0.0))
    second = generate_goal_candidates(
        tuple(reversed(records)), config=config, observed_position_global=(0.0, 0.0)
    )

    assert first.candidate_set_digest == second.candidate_set_digest
    final_positions = {
        candidate.position
        for candidate in first.candidate_set.candidates
        if candidate.role is GoalCandidateRole.FINAL_DESTINATION
    }
    assert final_positions == {(5.0, 0.0), (4.0, 0.0)}
    assert (
        len(first.candidate_set.candidates)
        <= config.final_destination_cap + config.active_waypoint_cap + 1
    )


def test_open_ray_record_has_no_endpoint_and_unknown_is_unconditional() -> None:
    """Explicit open rays carry normalized directions without fake Cartesian points."""

    result = generate_goal_candidates(
        (
            PublicGoalCandidateRecord(
                source=GoalCandidateSource.OPEN_RAY,
                source_id="north-ray",
                role=GoalCandidateRole.OPEN_RAY,
                direction=(0.0, 5.0),
                angular_support_rad=0.2,
            ),
        )
    )

    ray = next(
        candidate
        for candidate in result.candidate_set.candidates
        if candidate.role is GoalCandidateRole.OPEN_RAY
    )
    assert ray.position is None
    assert ray.direction == pytest.approx((0.0, 1.0))
    assert ray.angular_support_rad == pytest.approx(0.2)
    assert any(
        candidate.role is GoalCandidateRole.UNKNOWN for candidate in result.candidate_set.candidates
    )


def test_forbidden_oracle_source_and_invalid_geometry_fail_closed() -> None:
    """The actor boundary rejects truth aliases and non-finite public geometry."""

    with pytest.raises(ValueError, match="forbidden oracle source"):
        PublicGoalCandidateRecord(source="true_goal", source_id="truth", position=(1.0, 0.0))
    with pytest.raises(ValueError, match="finite"):
        PublicGoalCandidateRecord(
            source=GoalCandidateSource.MAP_DESTINATION_ZONE,
            source_id="bad",
            position=(float("nan"), 0.0),
        )
    with pytest.raises(ValueError, match="forbidden oracle source"):
        GoalCandidateProviderConfig(enabled_sources=("scenario_assigned_route",))


def test_map_projection_does_not_touch_hidden_pedestrian_assignments() -> None:
    """Map projection narrows objects using public map fields only."""

    class PublicMapWithHiddenTruth:
        ped_goal_zones = [((8.0, 0.0), (10.0, 0.0), (10.0, 2.0))]
        ped_routes = []
        poi_positions = []
        poi_labels = {}
        obstacles = []

        @property
        def single_pedestrians(self):
            raise AssertionError("hidden pedestrian assignments must not be read")

    public_map = public_goal_map_inputs_from_definition(PublicMapWithHiddenTruth())
    result = generate_goal_candidates_from_map(public_map)

    assert len(result.candidate_set.candidates) >= 2
    assert all(
        candidate.source != "scenario_assigned_route"
        for candidate in result.candidate_set.candidates
    )


def test_map_projection_is_order_invariant_and_cache_invalidates() -> None:
    """Static map cache keys change only when public map/config inputs change."""

    class PublicMap:
        ped_goal_zones = [((8.0, 0.0), (10.0, 0.0), (10.0, 2.0))]
        ped_routes = []
        poi_positions = [(3.0, 3.0), (4.0, 4.0)]
        poi_labels = {"b": "B", "a": "A"}
        obstacles = []

    public_map = public_goal_map_inputs_from_definition(PublicMap())
    provider = GoalCandidateProvider()
    first = provider.generate(public_map.records, obstacles=public_map.obstacles)
    second = provider.generate(tuple(reversed(public_map.records)), obstacles=public_map.obstacles)
    changed = provider.generate(public_map.records, obstacles=public_map.obstacles)

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert first.map_digest == second.map_digest == changed.map_digest
    assert first.candidate_set_digest == second.candidate_set_digest

    moved = provider.generate(
        public_map.records,
        obstacles=public_map.obstacles,
        observed_position_global=(1.0, 1.0),
    )
    assert moved.cache_hit is True
    assert moved.cache_key == first.cache_key
    assert moved.map_digest == first.map_digest

    config = GoalCandidateProviderConfig(open_ray_count=6)
    changed_provider = GoalCandidateProvider(config)
    changed_result = changed_provider.generate(public_map.records, obstacles=public_map.obstacles)
    assert changed_result.cache_hit is False
    assert changed_result.cache_key != first.cache_key


def test_map_adapter_projects_public_routes_and_unlabeled_pois_without_order_leakage() -> None:
    """Canonical public route/POI fields yield stable records without actor state."""

    class PublicMap:
        ped_goal_zones = []
        ped_routes = [
            SimpleNamespace(
                waypoints=[(0.0, 0.0), (4.0, 0.0)],
                source_path_id="",
                source_label="",
            ),
            SimpleNamespace(
                waypoints=[(0.0, 0.0), (0.0, 4.0)],
                source_path_id="",
                source_label="",
            ),
        ]
        poi_positions = [(2.0, 2.0), (-2.0, 2.0)]
        poi_labels = {}
        obstacles = []

    reordered = SimpleNamespace(
        ped_goal_zones=[],
        ped_routes=list(reversed(PublicMap.ped_routes)),
        poi_positions=list(reversed(PublicMap.poi_positions)),
        poi_labels={},
        obstacles=[],
    )

    first = public_goal_map_inputs_from_definition(PublicMap())
    second = public_goal_map_inputs_from_definition(reordered)

    assert first.map_digest == second.map_digest
    assert (
        sum(
            record.source is GoalCandidateSource.PEDESTRIAN_ROUTE_TERMINAL
            for record in first.records
        )
        == 2
    )
    assert (
        sum(record.source is GoalCandidateSource.POINT_OF_INTEREST for record in first.records) == 2
    )
    assert all(record.source_id != "poi:0" for record in first.records)


def test_config_is_frozen_and_translation_preserves_relative_geometry() -> None:
    """Configuration is immutable and transformed inputs preserve candidate semantics."""

    config = GoalCandidateProviderConfig()
    with pytest.raises(FrozenInstanceError):
        config.open_ray_count = 8  # type: ignore[misc]

    first = generate_goal_candidates(
        (_route_record(final=(10.0, 0.0), path=((0.0, 0.0), (0.0, 2.0), (10.0, 0.0))),),
        observed_position_global=(0.0, 0.0),
    )
    translated = generate_goal_candidates(
        (_route_record(final=(17.0, 4.0), path=((7.0, 4.0), (7.0, 6.0), (17.0, 4.0))),),
        observed_position_global=(7.0, 4.0),
    )
    first_active = next(
        candidate
        for candidate in first.candidate_set.candidates
        if candidate.role is GoalCandidateRole.ACTIVE_WAYPOINT
    )
    translated_active = next(
        candidate
        for candidate in translated.candidate_set.candidates
        if candidate.role is GoalCandidateRole.ACTIVE_WAYPOINT
    )
    assert first_active.path_tangent == pytest.approx(translated_active.path_tangent)
    assert first_active.path_mode == translated_active.path_mode


def test_rotation_preserves_active_waypoint_and_tangent_geometry() -> None:
    """Rotating public geometry rotates the emitted active waypoint and tangent."""

    first = generate_goal_candidates(
        (_route_record(final=(10.0, 0.0), path=((0.0, 0.0), (0.0, 2.0), (10.0, 0.0))),),
        observed_position_global=(0.0, 0.0),
    )
    rotated = generate_goal_candidates(
        (_route_record(final=(0.0, 10.0), path=((0.0, 0.0), (-2.0, 0.0), (0.0, 10.0))),),
        observed_position_global=(0.0, 0.0),
    )

    first_active = next(
        candidate
        for candidate in first.candidate_set.candidates
        if candidate.role is GoalCandidateRole.ACTIVE_WAYPOINT
    )
    rotated_active = next(
        candidate
        for candidate in rotated.candidate_set.candidates
        if candidate.role is GoalCandidateRole.ACTIVE_WAYPOINT
    )
    assert rotated_active.position == pytest.approx(
        (-first_active.position[1], first_active.position[0])
    )
    assert rotated_active.path_tangent == pytest.approx(
        (-first_active.path_tangent[1], first_active.path_tangent[0])
    )
