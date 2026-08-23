"""Pure-data preparation probes for issue #7809.

These tests exercise existing immutable-overlay, manifest-feasibility, and runtime-plausibility
owners. They intentionally do not load a simulator, planner, optimizer, campaign, or benchmark.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from robot_sf.adversarial.config import MultiPedAdversarialConfig, MultiPedCandidateSpec, Pose2D
from robot_sf.adversarial.materialize import ImmutableScenarioOverlay
from robot_sf.adversarial.runtime import validate_multi_ped_runtime_plausibility
from robot_sf.adversarial.search_harness import (
    FiniteSearchSpaceManifest,
    prepare_baseline,
)
from robot_sf.nav.global_route import GlobalRoute
from robot_sf.nav.map_config import MapDefinition


@pytest.mark.parametrize(
    "patch",
    [
        {},
        {"scenario": {"speed_m_s": 1.0}},
    ],
    ids=["empty", "zero-value"],
)
def test_zero_or_empty_overlay_preserves_source_identity_and_immutability(
    patch: dict[str, Any],
) -> None:
    """An empty or zero-value patch keeps the frozen source content and digest unchanged."""
    source = {
        "scenario": {
            "speed_m_s": 1.0,
            "waypoints": [[1.0, 2.0], [4.0, 2.0]],
        },
        "metadata": {"authored": True},
    }
    source_before = deepcopy(source)
    patch_before = deepcopy(patch)

    overlay = ImmutableScenarioOverlay(
        source=source,
        patch=patch,
        candidate_id="issue-7809:zero",
        adapter_id="issue-7809.test.v1",
    )

    assert source == source_before
    assert patch == patch_before
    assert overlay.source == overlay.materialized
    assert overlay.source_digest == overlay.materialized_digest
    assert overlay.to_dict()["source"] == source_before
    assert overlay.to_dict()["materialized"] == source_before

    source["scenario"]["speed_m_s"] = 9.0
    patch.setdefault("scenario", {})["speed_m_s"] = 9.0
    assert overlay.to_dict()["materialized"] == source_before

    with pytest.raises(TypeError):
        overlay.source["changed"] = True  # type: ignore[index]
    with pytest.raises(TypeError):
        overlay.materialized["scenario"]["speed_m_s"] = 9.0  # type: ignore[index]


def test_pre_adapter_infeasibility_rejection_never_calls_adapter() -> None:
    """An unsatisfied manifest constraint is recorded before the adapter boundary."""
    manifest = FiniteSearchSpaceManifest.from_mapping(
        {
            "schema_version": "adversarial_search_harness.v1",
            "name": "issue_7809_infeasible_fixture",
            "variables": {
                "start_x": {"unit": "m", "bounds": {"min": 0.0, "max": 1.0}},
                "goal_x": {"unit": "m", "bounds": {"min": 0.0, "max": 1.0}},
            },
            "constraints": [
                {"name": "minimum_clearance", "expression": "goal_x - start_x >= 2.0"},
            ],
            "objective_vector": {
                "components": [{"name": "diagnostic", "direction": "maximize", "unit": "score"}]
            },
            "seed_policy": {"search_seed": 11, "held_out_replay_seeds": [12]},
            "rollout_budget": {"candidate_budget": 2, "max_steps": 4},
        }
    )

    class CountingAdapter:
        """Adapter fixture that fails if a manifest-rejected candidate reaches it."""

        adapter_id = "issue-7809.counting.v1"

        def __init__(self) -> None:
            self.validate_calls = 0
            self.materialize_calls = 0

        def validate(self, source_scenario: dict[str, Any], candidate: Any) -> tuple[str, ...]:
            del source_scenario, candidate
            self.validate_calls += 1
            return ()

        def materialize(self, source_scenario: dict[str, Any], candidate: Any) -> Any:
            del source_scenario, candidate
            self.materialize_calls += 1
            raise AssertionError("manifest-rejected candidates must not materialize")

    adapter = CountingAdapter()
    result = prepare_baseline(manifest, {"source": "unchanged"}, adapter, baseline="random")

    assert adapter.validate_calls == 0
    assert adapter.materialize_calls == 0
    assert result.prepared_count == 0
    assert result.rejected_count == manifest.rollout_budget.candidate_budget
    assert all(row.rejection is not None for row in result.candidates)
    assert all(row.rejection.stage == "manifest" for row in result.candidates if row.rejection)
    assert all(
        row.rejection.reasons == ("constraint:minimum_clearance:unsatisfied",)
        for row in result.candidates
        if row.rejection
    )
    assert all(
        row.rejection.to_dict()["simulation_executed"] is False
        for row in result.candidates
        if row.rejection
    )


def _plausibility_map() -> MapDefinition:
    """Build the smallest valid map object needed by the pure plausibility predicate."""
    robot_spawn_zones = [((0.5, 0.5), (1.0, 0.5), (1.0, 1.0))]
    robot_goal_zones = [((7.0, 5.0), (7.5, 5.0), (7.5, 5.5))]
    return MapDefinition(
        width=8.0,
        height=6.0,
        obstacles=[],
        robot_spawn_zones=robot_spawn_zones,
        ped_spawn_zones=[],
        robot_goal_zones=robot_goal_zones,
        bounds=[
            (0.0, 8.0, 0.0, 0.0),
            (0.0, 8.0, 6.0, 6.0),
            (0.0, 0.0, 0.0, 6.0),
            (8.0, 8.0, 0.0, 6.0),
        ],
        robot_routes=[
            GlobalRoute(
                spawn_id=0,
                goal_id=0,
                waypoints=[(0.75, 0.75), (7.25, 5.25)],
                spawn_zone=robot_spawn_zones[0],
                goal_zone=robot_goal_zones[0],
            )
        ],
        ped_goal_zones=[],
        ped_crowded_zones=[],
        ped_routes=[],
        single_pedestrians=[],
    )


def test_runtime_plausibility_rejects_speed_cap_without_simulator() -> None:
    """The independent runtime plausibility owner rejects an over-cap scripted speed."""
    config = MultiPedAdversarialConfig(
        family="issue_7809_plausibility_fixture",
        scenario_seed=11,
        pedestrians=[
            MultiPedCandidateSpec(
                id="p0",
                start=Pose2D(1.5, 2.0),
                goal=Pose2D(6.5, 2.0),
                speed_mps=3.0,
            )
        ],
    )

    errors = validate_multi_ped_runtime_plausibility(
        config,
        _plausibility_map(),
        max_speed_mps=2.5,
    )

    assert errors == [
        "pedestrians[0] speed_mps (3.000) exceeds max_speed_mps (2.500)",
    ]
