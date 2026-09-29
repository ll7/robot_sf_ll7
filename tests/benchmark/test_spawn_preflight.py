"""Fail-closed release matrix preflight contracts (issue #9731)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from shapely.geometry import LineString
from shapely.ops import unary_union

from robot_sf.benchmark import spawn_preflight
from robot_sf.benchmark.identity.hash_utils import sha256_file
from robot_sf.benchmark.release_protocol import load_release_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]
RELEASE_MANIFEST = REPO_ROOT / "configs/benchmarks/releases/benchmark_data_release_s30_h600.yaml"


@pytest.mark.parametrize(
    ("fixture_name", "clearance", "pedestrian_count", "reason"),
    (
        (
            "issue_9725_initial_pedestrian_overlap",
            {
                "overlap": True,
                "robot_obstacle_min_surface_clearance_m": 0.5,
                "robot_pedestrian_min_surface_clearance_m": -0.2,
            },
            1,
            "robot_pedestrian_below_clearance_margin",
        ),
        (
            "issue_9725_robot_start_inside_wall_clearance",
            {
                "overlap": True,
                "robot_obstacle_min_surface_clearance_m": -0.1,
                "robot_pedestrian_min_surface_clearance_m": None,
            },
            0,
            "robot_wall_below_clearance_margin",
        ),
    ),
)
def test_issue_9725_reset_fixtures_fail_the_configured_margin(
    fixture_name: str,
    clearance: dict,
    pedestrian_count: int,
    reason: str,
) -> None:
    """The original reset-overlap and wall-contact mechanisms produce stable reasons."""
    result = spawn_preflight._check_reset_clearance(
        clearance,
        margin_m=0.1,
        pedestrian_count=pedestrian_count,
    )
    assert result["status"] == "fail", fixture_name
    assert reason in result["reason"], fixture_name


def test_issue_9725_route_end_respawn_fixture_fails_stationary_window() -> None:
    """The #9725 fallback respawn event is a respawn-safety failure."""

    class FakeEnv:
        action_space = SimpleNamespace(shape=(2,), dtype=np.dtype(np.float32))

        def __init__(self) -> None:
            self.steps = 0
            behavior = SimpleNamespace(respawn_overlap_events=[])
            self.simulator = SimpleNamespace(
                ped_pos=[(1.0, 0.0)],
                peds_behaviors=[behavior],
                robots=[SimpleNamespace(pose=((0.0, 0.0), 0.0))],
            )

        def step(self, _action):
            self.steps += 1
            if self.steps == 3:
                self.simulator.peds_behaviors[0].respawn_overlap_events.append(
                    {"group_id": 1, "ped_rows": [0], "step": 2, "positions": [[0.1, 0.0]]}
                )
            return None, 0.0, False, False, {}

    env = FakeEnv()
    result = spawn_preflight._check_respawn_window(env, window_steps=5)

    assert env.steps == 5
    assert result["status"] == "fail"
    assert result["reason"] == "pedestrian_respawn_inside_robot_exclusion_radius"
    assert result["first_overlap_event"]["ped_rows"] == [0]


def test_respawn_window_ignores_behavior_types_without_route_respawn_ledgers() -> None:
    """Single-pedestrian controllers do not expose the route-group respawn ledger."""

    class FakeEnv:
        action_space = SimpleNamespace(shape=(2,), dtype=np.dtype(np.float32))

        def __init__(self) -> None:
            self.simulator = SimpleNamespace(
                ped_pos=[(1.0, 0.0)],
                peds_behaviors=[
                    SimpleNamespace(navigators={1: object()}, respawn_overlap_events=[]),
                    SimpleNamespace(single_pedestrians=[object()]),
                ],
                robots=[SimpleNamespace(pose=((0.0, 0.0), 0.0))],
            )

        def step(self, _action):
            return None, 0.0, False, False, {}

    result = spawn_preflight._check_respawn_window(FakeEnv(), window_steps=3)

    assert result["status"] == "pass"
    assert result["reason"] == "no_respawn_inside_robot_exclusion_radius"
    assert result["steps_checked"] == 3


def test_respawn_window_fails_closed_for_untracked_route_behavior() -> None:
    """A route behavior with active navigators must provide an overlap event ledger."""
    env = SimpleNamespace(
        simulator=SimpleNamespace(
            ped_pos=[(1.0, 0.0)],
            peds_behaviors=[SimpleNamespace(navigators={1: object()})],
        )
    )

    result = spawn_preflight._check_respawn_window(env, window_steps=3)

    assert result["status"] == "invalid"
    assert result["reason"] == "respawn_event_ledger_unavailable"


def _doorway_fixture(*, expected_outcome: str | None = None):
    """Build the 2.0 m #9728 doorway with a 1.0 m radius robot and 0.10 m margin."""
    occupancy = np.zeros((60, 60), dtype=bool)
    occupancy[0, :] = occupancy[-1, :] = True
    occupancy[:, 0] = occupancy[:, -1] = True
    occupancy[1:59, 30] = True
    occupancy[20:40, 30] = False
    inflated = occupancy.copy()
    inflated[20:40, 30] = True
    wall_geometry = unary_union(
        [
            LineString([(3.0, 0.0), (3.0, 2.0)]),
            LineString([(3.0, 4.0), (3.0, 6.0)]),
            LineString([(0.0, 0.0), (6.0, 0.0)]),
            LineString([(6.0, 0.0), (6.0, 6.0)]),
            LineString([(6.0, 6.0), (0.0, 6.0)]),
            LineString([(0.0, 6.0), (0.0, 0.0)]),
        ]
    )
    robot = SimpleNamespace(pose=((1.5, 3.0), 0.0), config=SimpleNamespace(radius=1.0))
    simulator = SimpleNamespace(
        robots=[robot],
        robot_navs=[SimpleNamespace(waypoints=[(4.5, 3.0)])],
    )
    env = SimpleNamespace(simulator=simulator)
    analysis = {
        "occupancy": occupancy,
        "inflated": inflated,
        "origin": (0.0, 0.0),
        "resolution": 0.1,
        "map_bounds": (0.0, 6.0, 0.0, 6.0),
        "wall_geometry": wall_geometry,
    }
    scenario = {"name": "francis2023_narrow_doorway"}
    if expected_outcome is not None:
        scenario["expected_outcome"] = expected_outcome
    return env, analysis, scenario


def test_issue_9728_undeclared_doorway_fails_reachability_and_width() -> None:
    """The 2.0 m doorway cannot fit the 2.2 m robot footprint plus margin."""
    env, analysis, scenario = _doorway_fixture()
    reachability, passage = spawn_preflight._check_footprint_path(
        env,
        analysis,
        scenario=scenario,
        margin_m=0.1,
    )

    assert reachability["status"] == "fail"
    assert reachability["reason"] == "no_collision_free_footprint_path"
    assert reachability["continuous_oracle"]["reason"] == (
        "ordered_route_segment_disconnected_in_continuous_free_space"
    )
    assert passage["status"] == "fail"
    assert 1.8 <= passage["minimum_opening_width_estimate_m"] < 2.2
    assert passage["required_opening_width_m"] == pytest.approx(2.2)


def test_safe_hold_requires_separate_probe_manifest() -> None:
    """A declaration cannot make an infeasible nominal route valid."""
    env, analysis, scenario = _doorway_fixture(expected_outcome="infeasible_safe_hold")
    reachability, passage = spawn_preflight._check_footprint_path(
        env,
        analysis,
        scenario=scenario,
        margin_m=0.1,
    )
    assert reachability["status"] == "invalid"
    assert passage["status"] == "invalid"
    assert reachability["reason"] == "infeasibility_probe_requires_separate_manifest"
    assert reachability["observed_status"] == "fail"
    assert passage["observed_status"] == "fail"
    assert passage["minimum_opening_width_estimate_m"] < 2.2

    reachability, passage = spawn_preflight._check_footprint_path(
        env,
        analysis,
        scenario=scenario,
        margin_m=0.1,
        probe_manifest=True,
    )
    assert reachability["status"] == "exempt_expected_outcome"
    assert passage["status"] == "exempt_expected_outcome"

    env, analysis, malformed = _doorway_fixture(expected_outcome="safe_hold")
    reachability, passage = spawn_preflight._check_footprint_path(
        env,
        analysis,
        scenario=malformed,
        margin_m=0.1,
    )
    assert reachability["status"] == "invalid"
    assert passage["status"] == "invalid"
    assert reachability["reason"] == "unsupported_expected_outcome"


def test_probe_declaration_requires_observed_infeasibility() -> None:
    env, analysis, scenario = _doorway_fixture(expected_outcome="infeasible_safe_hold")
    analysis["occupancy"] = np.zeros_like(analysis["occupancy"], dtype=bool)
    analysis["inflated"] = analysis["occupancy"]
    analysis["wall_geometry"] = LineString([(-100.0, -100.0), (-100.0, 100.0)])
    reachability, passage = spawn_preflight._check_footprint_path(
        env,
        analysis,
        scenario=scenario,
        margin_m=0.1,
        probe_manifest=True,
    )
    assert reachability["status"] == "invalid"
    assert passage["status"] == "invalid"
    assert reachability["reason"] == "declared_infeasibility_not_confirmed_by_continuous_oracle"


def test_footprint_path_checks_required_intermediate_waypoints_in_order() -> None:
    """A reachable final goal cannot hide an unreachable required route waypoint."""
    blocked = np.zeros((60, 60), dtype=bool)
    blocked[:, 30] = True
    robot = SimpleNamespace(pose=((1.0, 3.0), 0.0), config=SimpleNamespace(radius=0.1))
    navigator = SimpleNamespace(waypoints=[(5.0, 3.0), (2.0, 3.0)])
    env = SimpleNamespace(simulator=SimpleNamespace(robots=[robot], robot_navs=[navigator]))
    analysis = {
        "occupancy": blocked,
        "inflated": blocked,
        "origin": (0.0, 0.0),
        "resolution": 0.1,
        "wall_geometry": LineString([(3.0, 0.0), (3.0, 6.0)]),
    }

    reachability, passage = spawn_preflight._check_footprint_path(
        env, analysis, scenario={"name": "ordered_route"}, margin_m=0.1
    )

    assert reachability["status"] == "fail"
    assert reachability["first_blocked_segment_index"] == 0
    assert reachability["route_waypoint_count"] == 2
    assert passage["status"] == "fail"
    assert passage["first_blocked_segment_index"] == 0

    navigator.waypoints = [(2.0, 3.0), (1.0, 3.0)]
    reachability, passage = spawn_preflight._check_footprint_path(
        env, analysis, scenario={"name": "ordered_route"}, margin_m=0.1
    )
    assert reachability["status"] == "pass"
    assert reachability["path_length_m"] == pytest.approx(2.0)
    assert passage["status"] == "pass"


def test_continuous_margin_oracle_recovers_safe_grid_blocked_endpoint() -> None:
    """A conservative endpoint cell cannot block a continuously safe route."""
    occupancy = np.zeros((100, 100), dtype=bool)
    inflated = occupancy.copy()
    inflated[50, 88] = True
    robot = SimpleNamespace(pose=((2.0, 5.0), 0.0), config=SimpleNamespace(radius=1.0))
    env = SimpleNamespace(
        simulator=SimpleNamespace(
            robots=[robot], robot_navs=[SimpleNamespace(waypoints=[(8.85, 5.0)])]
        )
    )
    analysis = {
        "occupancy": occupancy,
        "inflated": inflated,
        "origin": (0.0, 0.0),
        "resolution": 0.1,
        "map_bounds": (0.0, 10.0, 0.0, 10.0),
        "wall_geometry": LineString([(0.0, 0.0), (0.0, 10.0)]),
    }

    reachability, passage = spawn_preflight._check_footprint_path(
        env, analysis, scenario={"name": "safe_endpoint"}, margin_m=0.1
    )

    assert reachability["status"] == passage["status"] == "pass"
    assert reachability["grid_status"] == "fail"
    assert reachability["continuous_oracle"]["status"] == "pass"
    assert reachability["path_length_m"] is None
    assert passage["measurement_path"] == "continuous_clearance_component"
    assert passage["minimum_opening_width_estimate_m"] is None
    assert passage["minimum_opening_width_certified_lower_bound_m"] == pytest.approx(2.2)

    reachability, passage = spawn_preflight._check_footprint_path(
        env,
        analysis,
        scenario={"name": "safe_endpoint", "expected_outcome": "infeasible_safe_hold"},
        margin_m=0.1,
        probe_manifest=True,
    )
    assert reachability["status"] == passage["status"] == "invalid"
    assert reachability["reason"] == "declared_infeasibility_not_confirmed_by_continuous_oracle"

    del analysis["map_bounds"]
    reachability, _passage = spawn_preflight._check_footprint_path(
        env, analysis, scenario={"name": "missing_geometry"}, margin_m=0.1
    )
    assert reachability["status"] == "fail"
    assert reachability["continuous_oracle"]["reason"] == "continuous_geometry_unavailable"


def test_continuous_margin_oracle_keeps_the_exact_boundary_conservative() -> None:
    """The buffer approximation guard cannot admit a point below the required margin."""
    analysis = {
        "map_bounds": (0.0, 10.0, 0.0, 10.0),
        "wall_geometry": LineString([(0.0, 0.0), (0.0, 10.0)]),
    }
    safe = spawn_preflight._continuous_margin_route(
        analysis, [(2.0, 5.0), (8.899, 5.0)], required_radius_m=1.1
    )
    unsafe = spawn_preflight._continuous_margin_route(
        analysis, [(2.0, 5.0), (8.901, 5.0)], required_radius_m=1.1
    )
    assert safe["status"] == "pass"
    assert unsafe["status"] == "fail"
    assert unsafe["first_unsafe_route_point_index"] == 1


def test_continuous_margin_oracle_rejects_unsafe_goal_and_disconnected_leg() -> None:
    """The fallback must keep exact margin failures and ordered route cuts blocked."""
    occupancy = np.zeros((100, 100), dtype=bool)
    inflated = occupancy.copy()
    inflated[50, 90] = True
    robot = SimpleNamespace(pose=((2.0, 5.0), 0.0), config=SimpleNamespace(radius=1.0))
    navigator = SimpleNamespace(waypoints=[(9.05, 5.0)])
    env = SimpleNamespace(simulator=SimpleNamespace(robots=[robot], robot_navs=[navigator]))
    analysis = {
        "occupancy": occupancy,
        "inflated": inflated,
        "origin": (0.0, 0.0),
        "resolution": 0.1,
        "map_bounds": (0.0, 10.0, 0.0, 10.0),
        "wall_geometry": LineString([(0.0, 0.0), (0.0, 10.0)]),
    }
    reachability, passage = spawn_preflight._check_footprint_path(
        env, analysis, scenario={"name": "unsafe_goal"}, margin_m=0.1
    )
    assert reachability["status"] == "fail"
    assert reachability["continuous_oracle"]["reason"] == (
        "required_route_point_below_continuous_margin"
    )
    assert reachability["continuous_oracle"]["first_unsafe_route_point_index"] == 1
    assert passage["status"] == "pass"

    occupancy[:, 50] = True
    inflated[:, 50] = True
    navigator.waypoints = [(8.0, 5.0), (2.0, 5.0)]
    analysis["wall_geometry"] = LineString([(5.0, 0.0), (5.0, 10.0)])
    reachability, passage = spawn_preflight._check_footprint_path(
        env, analysis, scenario={"name": "disconnected_waypoint"}, margin_m=0.1
    )
    assert reachability["status"] == passage["status"] == "fail"
    assert reachability["continuous_oracle"]["reason"] == (
        "ordered_route_segment_disconnected_in_continuous_free_space"
    )
    assert reachability["continuous_oracle"]["first_blocked_segment_index"] == 0


def test_known_unsafe_sampled_goals_stay_blocked_on_release_seeds() -> None:
    """The exact-margin fallback cannot admit the observed unsafe nominal goals."""
    matrix = (
        REPO_ROOT / "configs/scenarios/classic_interactions_francis2023_goal_zone_entry_v3.yaml"
    )
    scenario = next(
        dict(row)
        for row in spawn_preflight._load_matrix(matrix)
        if row["name"] == "classic_t_intersection_low"
    )
    unsafe_seeds = (111, 115, 118, 121, 122, 125, 134, 138)
    result = spawn_preflight._check_release_scenario(
        (scenario, str(matrix), unsafe_seeds, 0.1, 20, 0.1, False)
    )

    assert [row["seed"] for row in result["rows"]] == list(unsafe_seeds)
    for row in result["rows"]:
        assert row["overall_status"] == "blocked", row
        assert row["footprint_reachability"]["status"] == "fail", row
        assert row["footprint_reachability"]["continuous_oracle"]["reason"] == (
            "required_route_point_below_continuous_margin"
        ), row


def test_historical_narrow_doorway_probe_stays_blocked_on_all_release_seeds() -> None:
    """A probe declaration cannot admit the 2 m doorway to the nominal matrix."""
    matrix = (
        REPO_ROOT / "configs/scenarios/classic_interactions_francis2023_goal_zone_entry_v3.yaml"
    )
    scenario = next(
        dict(row)
        for row in spawn_preflight._load_matrix(matrix)
        if row["name"] == "francis2023_narrow_doorway"
    )
    scenario["expected_outcome"] = "infeasible_safe_hold"
    result = spawn_preflight._check_release_scenario(
        (scenario, str(matrix), tuple(range(111, 141)), 0.1, 20, 0.1, False)
    )

    assert len(result["rows"]) == 30
    for row in result["rows"]:
        assert row["overall_status"] == "blocked", row
        reachability = row["footprint_reachability"]
        assert reachability["status"] == "invalid", row
        assert reachability["reason"] == "infeasibility_probe_requires_separate_manifest", row
        assert reachability["observed_status"] == "fail", row
        assert row["passage_width"]["observed_status"] == "fail", row


def test_main_grid_doorway_probe_requires_pinned_map_and_oracle() -> None:
    matrix = REPO_ROOT / "configs/scenarios/classic_interactions_francis2023_release_0_0_8_v1.yaml"
    scenario = next(
        row
        for row in spawn_preflight._load_matrix(matrix)
        if row["name"] == "francis2023_narrow_doorway"
    )
    assert spawn_preflight._verified_main_grid_probe(scenario, matrix)
    changed = dict(scenario)
    changed["map_file"] = (
        "../../maps/successor_svg_maps/issue_9728_francis2023_narrow_doorway_feasible_3p60_v1.svg"
    )
    assert not spawn_preflight._verified_main_grid_probe(changed, matrix)
    result = spawn_preflight._check_release_scenario(
        (scenario, str(matrix), (111, 119), 0.1, 20, 0.1, True)
    )
    assert [row["overall_status"] for row in result["rows"]] == [
        "infeasibility_probe",
        "infeasibility_probe",
    ]
    assert all(
        row["footprint_reachability"]["continuous_oracle"]["status"] == "fail"
        for row in result["rows"]
    )


def test_station_platform_117_respawn_defect_is_removed_by_successor_map() -> None:
    matrix = REPO_ROOT / "configs/scenarios/classic_interactions_francis2023_release_0_0_8_v1.yaml"
    scenario = next(
        row
        for row in spawn_preflight._load_matrix(matrix)
        if row["name"] == "classic_station_platform_medium"
    )
    corrected = spawn_preflight._check_release_scenario(
        (scenario, str(matrix), (117,), 0.1, 20, 0.1, False)
    )["rows"][0]
    historical = dict(scenario)
    historical["map_file"] = "../../maps/svg_maps/classic_station_platform.svg"
    historical_result = spawn_preflight._check_release_scenario(
        (historical, str(matrix), (117,), 0.1, 20, 0.1, False)
    )["rows"][0]
    assert corrected["overall_status"] == "valid"
    assert corrected["respawn_safety"]["status"] == "pass"
    assert historical_result["overall_status"] == "blocked"
    assert historical_result["respawn_safety"]["reason"] == ("episode_ended_before_respawn_window")


def test_boundary_width_straight_route_requires_full_margin() -> None:
    _, analysis, _ = _doorway_fixture()
    analysis["wall_geometry"] = unary_union(
        [
            LineString([(3.0, 0.0), (3.0, 1.9)]),
            LineString([(3.0, 4.1), (3.0, 6.0)]),
        ]
    )
    route = [(1.5, 3.0), (4.5, 3.0)]
    assert (
        spawn_preflight._continuous_margin_route(analysis, route, required_radius_m=1.1)["status"]
        == "pass"
    )
    assert (
        spawn_preflight._continuous_margin_route(analysis, route, required_radius_m=1.100001)[
            "status"
        ]
        == "fail"
    )


def test_continuous_margin_oracle_replaces_narrow_raw_grid_witness() -> None:
    """A narrow sampled grid witness does not disprove a safe route around an obstacle."""
    occupancy = np.zeros((100, 100), dtype=bool)
    robot = SimpleNamespace(pose=((2.0, 5.0), 0.0), config=SimpleNamespace(radius=1.0))
    env = SimpleNamespace(
        simulator=SimpleNamespace(
            robots=[robot], robot_navs=[SimpleNamespace(waypoints=[(8.0, 5.0)])]
        )
    )
    analysis = {
        "occupancy": occupancy,
        "inflated": occupancy.copy(),
        "origin": (0.0, 0.0),
        "resolution": 0.1,
        "map_bounds": (0.0, 10.0, 0.0, 10.0),
        "wall_geometry": LineString([(5.0, 4.0), (5.0, 6.0)]),
    }
    reachability, passage = spawn_preflight._check_footprint_path(
        env, analysis, scenario={"name": "detour"}, margin_m=0.1
    )
    assert reachability["status"] == passage["status"] == "pass"
    assert passage["grid_status"] == "fail"
    assert passage["grid_minimum_opening_width_estimate_m"] < 2.2
    assert passage["minimum_opening_width_certified_lower_bound_m"] == pytest.approx(2.2)


def test_release_input_resolver_uses_manifest_matrix_and_seed_set() -> None:
    """The current release identities resolve to all 48 scenarios and seeds 111-140."""
    manifest = load_release_manifest(RELEASE_MANIFEST)
    identity, scenarios, seeds = spawn_preflight._release_manifest_inputs(manifest)

    assert Path(
        identity["scenario_matrix_path"]
    ) == manifest.scenario_matrix_path.resolve().relative_to(REPO_ROOT)
    assert identity["scenario_matrix_sha256"] == sha256_file(manifest.scenario_matrix_path)
    assert identity["seed_sets_sha256"] == manifest.seed_sets_sha256
    assert len(scenarios) == 48
    assert seeds == tuple(range(111, 141))
    assert identity["seed_set"] == "paper_eval_s30"


def _manifest_with_seed_policy(seed_policy: dict[str, object]):
    manifest = load_release_manifest(RELEASE_MANIFEST)
    fields = {name: getattr(manifest, name) for name in dir(manifest) if not name.startswith("_")}
    fields["seed_policy"] = seed_policy
    return SimpleNamespace(**fields)


def test_release_input_resolver_rejects_fixed_list_that_differs_from_resolved_seeds() -> None:
    """A fixed list of [111] cannot stand in for the resolved 111-140 evaluation seeds."""
    manifest = _manifest_with_seed_policy(  # seed-holdout: synthetic-fixture
        {"mode": "fixed-list", "seeds": [111]}  # seed-holdout: synthetic-fixture
    )

    with pytest.raises(ValueError, match="do not match seed_policy.seeds"):
        spawn_preflight._release_manifest_inputs(manifest)


def test_release_input_resolver_rejects_fixed_list_without_a_seed_list() -> None:
    manifest = _manifest_with_seed_policy({"mode": "fixed-list"})

    with pytest.raises(ValueError, match="requires seed_policy.seeds"):
        spawn_preflight._release_manifest_inputs(manifest)


@pytest.mark.parametrize("mode", ["seedset", "", None, "range"])
def test_release_input_resolver_rejects_unknown_seed_mode(mode: object) -> None:
    """An unknown mode must not skip the named seed-set check."""
    manifest = _manifest_with_seed_policy(
        # seed-holdout: synthetic-fixture begin
        {"mode": mode, "seed_set": "paper_eval_s30", "seeds": list(range(111, 141))}
        # seed-holdout: synthetic-fixture end
    )

    with pytest.raises(ValueError, match="unsupported seed_policy mode"):
        spawn_preflight._release_manifest_inputs(manifest)


def test_matrix_only_cli_input_cannot_be_mistaken_for_release_preflight(
    tmp_path: Path,
) -> None:
    """A matrix or seed override cannot produce a passing release preflight."""
    with pytest.raises(SystemExit):
        spawn_preflight.main(
            [
                "--matrix",
                str(REPO_ROOT / "configs/scenarios/classic_interactions.yaml"),
                "--seeds",
                "111",
                "--json-output",
                str(tmp_path / "report.json"),
                "--markdown-output",
                str(tmp_path / "report.md"),
            ]
        )


def test_preflight_report_writer_binds_both_file_hashes(tmp_path: Path) -> None:
    """JSON and Markdown preserve release identities and label evidence as diagnostic."""
    report = {
        "schema_version": "spawn_matrix_preflight.v1",
        "status": "blocked",
        "evidence_class": "preflight_diagnostic_only",
        "benchmark_success": None,
        "release_inputs": {
            "release_id": "fixture-release",
            "manifest_sha256": "a" * 64,
            "scenario_matrix_path": "configs/scenarios/fixture.yaml",
            "scenario_matrix_sha256": "b" * 64,
            "seed_set": "fixture-seeds",
            "seed_sets_sha256": "c" * 64,
            "resolved_seeds": [111],
        },
        "clearance_margin_m": 0.1,
        "respawn_window_steps": 20,
        "grid_resolution_m": 0.1,
        "scenario_count": 1,
        "seed_count": 1,
        "cell_count": 1,
        "blocked_cell_count": 1,
        "rows": [
            {
                "scenario": "fixture",
                "seed": 111,
                "overall_status": "blocked",
                "reset_clearance": {"status": "pass", "reason": "clearance_meets_margin"},
                "footprint_reachability": {"status": "fail", "reason": "no_path"},
                "passage_width": {"status": "fail", "reason": "too_narrow"},
                "respawn_safety": {"status": "pass", "reason": "no_respawn"},
            }
        ],
    }
    json_path = tmp_path / "preflight.json"
    markdown_path = tmp_path / "preflight.md"
    json_sha, markdown_sha = spawn_preflight.write_preflight_reports(
        report,
        json_path=json_path,
        markdown_path=markdown_path,
    )

    assert json_sha == sha256_file(json_path)
    assert markdown_sha == sha256_file(markdown_path)
    saved = json.loads(json_path.read_text(encoding="utf-8"))
    assert saved["benchmark_success"] is None
    assert "fixture-release" in markdown_path.read_text(encoding="utf-8")
    assert "preflight diagnostic output" in markdown_path.read_text(encoding="utf-8")


def test_reset_clearance_rejects_missing_nonfinite_and_overlap_measurements() -> None:
    result = spawn_preflight._check_reset_clearance(
        {
            "robot_obstacle_min_surface_clearance_m": float("nan"),
            "robot_pedestrian_min_surface_clearance_m": None,
            "overlap": True,
        },
        margin_m=0.1,
        pedestrian_count=1,
    )

    assert result["status"] == "fail"
    assert result["reason"] == (
        "missing_or_invalid_wall_clearance;missing_or_invalid_pedestrian_clearance;"
        "reset_footprint_overlap"
    )


def test_grid_path_fails_closed_for_bounds_barriers_and_diagonal_corner_cutting() -> None:
    open_grid = np.zeros((3, 3), dtype=bool)
    assert spawn_preflight._grid_path(
        open_grid, (0.2, 0.2), (0.2, 0.2), origin=(0.0, 0.0), resolution=1.0
    ) == [(0, 0)]
    assert (
        spawn_preflight._grid_path(
            open_grid, (-0.1, 0.2), (2.2, 2.2), origin=(0.0, 0.0), resolution=1.0
        )
        is None
    )

    blocked = np.zeros((3, 3), dtype=bool)
    blocked[:, 1] = True
    assert (
        spawn_preflight._grid_path(
            blocked, (0.2, 0.2), (2.2, 0.2), origin=(0.0, 0.0), resolution=1.0
        )
        is None
    )

    corner = np.zeros((2, 2), dtype=bool)
    corner[0, 1] = corner[1, 0] = True
    assert (
        spawn_preflight._grid_path(
            corner, (0.2, 0.2), (1.2, 1.2), origin=(0.0, 0.0), resolution=1.0
        )
        is None
    )


def test_occupancy_analysis_requires_canonical_static_geometry() -> None:
    map_def = SimpleNamespace(get_map_bounds=lambda: (0.0, 2.0, 0.0, 2.0))
    simulator = SimpleNamespace(map_def=map_def)
    base = {"simulator": simulator}
    args = {"robot_radius_m": 0.3, "margin_m": 0.1, "resolution_m": 0.2}

    with pytest.raises(ValueError, match="geometry is unavailable"):
        spawn_preflight._build_occupancy_analysis(SimpleNamespace(**base), **args)
    with pytest.raises(ValueError, match="contains no static occupancy-grid geometry"):
        spawn_preflight._build_occupancy_analysis(
            SimpleNamespace(**base, _get_static_grid_obstacles=lambda: ([], [])), **args
        )


def test_occupancy_analysis_inflates_static_obstacles_and_reports_world_origin() -> None:
    map_def = SimpleNamespace(get_map_bounds=lambda: (0.0, 4.0, 0.0, 4.0))
    env = SimpleNamespace(
        simulator=SimpleNamespace(map_def=map_def),
        _get_static_grid_obstacles=lambda: ([((2.0, 0.0), (2.0, 4.0))], []),
    )

    analysis = spawn_preflight._build_occupancy_analysis(
        env, robot_radius_m=0.3, margin_m=0.1, resolution_m=0.2
    )

    assert analysis["occupancy"].shape == analysis["inflated"].shape
    assert analysis["occupancy"].dtype == np.bool_
    assert np.count_nonzero(analysis["inflated"]) >= np.count_nonzero(analysis["occupancy"])
    assert analysis["origin"][0] < 0.0 and analysis["origin"][1] < 0.0
    assert analysis["resolution"] == pytest.approx(0.2)


def test_footprint_check_reports_invalid_robot_and_missing_goal_route() -> None:
    analysis = {"origin": (0.0, 0.0), "resolution": 0.1}
    missing_robot = SimpleNamespace(simulator=SimpleNamespace(robots=[], robot_navs=[]))
    assert spawn_preflight._check_footprint_path(
        missing_robot, analysis, scenario={}, margin_m=0.1
    ) == (
        {"status": "invalid", "reason": "robot_route_state_unavailable"},
        {"status": "invalid", "reason": "robot_route_state_unavailable"},
    )

    no_goal = SimpleNamespace(
        simulator=SimpleNamespace(
            robots=[SimpleNamespace(pose=((0.2, 0.2), 0), config=SimpleNamespace(radius=0.3))],
            robot_navs=[SimpleNamespace(waypoints=[])],
        )
    )
    reachability, passage = spawn_preflight._check_footprint_path(
        no_goal, analysis, scenario={}, margin_m=0.1
    )
    assert reachability["reason"] == "scenario_goal_route_unavailable"
    assert passage["reason"] == "scenario_goal_route_unavailable"


@pytest.mark.parametrize(
    ("pedestrians", "behaviors", "step_result", "move_robot", "expected"),
    (
        ([], [], None, False, "no_pedestrians_to_respawn"),
        ([(1.0, 0.0)], [], None, False, "no_route_end_respawn_groups"),
        (
            [(1.0, 0.0)],
            [SimpleNamespace(respawn_overlap_events=())],
            None,
            False,
            "respawn_event_ledger_unavailable",
        ),
        (
            [(1.0, 0.0)],
            [SimpleNamespace(respawn_overlap_events=[])],
            (None, 0.0, True, False, {}),
            False,
            "episode_ended_before_respawn_window",
        ),
        (
            [(1.0, 0.0)],
            [SimpleNamespace(respawn_overlap_events=[])],
            None,
            True,
            "robot_did_not_remain_stationary",
        ),
    ),
)
def test_respawn_window_validates_ledger_stationarity_and_episode_state(
    pedestrians, behaviors, step_result, move_robot, expected
) -> None:
    class FakeEnv:
        action_space = SimpleNamespace(shape=(2,), dtype=np.dtype(np.float32))

        def __init__(self):
            self.steps = 0
            self.simulator = SimpleNamespace(
                ped_pos=pedestrians,
                peds_behaviors=behaviors,
                robots=[SimpleNamespace(pose=((0.0, 0.0), 0.0))],
            )

        def step(self, action):
            self.steps += 1
            assert np.array_equal(action, np.zeros(2, dtype=np.float32))
            if move_robot:
                self.simulator.robots[0].pose = ((0.01, 0.0), 0.0)
            return step_result or (None, 0.0, False, False, {})

    result = spawn_preflight._check_respawn_window(FakeEnv(), window_steps=2)

    assert result["reason"] == expected
    if expected in {"no_pedestrians_to_respawn", "no_route_end_respawn_groups"}:
        assert result["status"] == "pass"
    elif expected == "respawn_event_ledger_unavailable":
        assert result["status"] == "invalid"
    else:
        assert result["status"] == "invalid"


@pytest.mark.parametrize("analysis_error", [False, True])
def test_release_scenario_keeps_rows_and_reuses_map_analysis(monkeypatch, analysis_error) -> None:
    class FakeEnv:
        def __init__(self):
            self.simulator = SimpleNamespace(
                robots=[SimpleNamespace(config=SimpleNamespace(radius=0.3))],
                config=SimpleNamespace(ped_radius=0.2),
                ped_pos=[(1.0, 1.0)],
                last_spawn_relocation=None,
            )
            self.closed = False

        def reset(self, seed):
            self.seed = seed

        def close(self):
            self.closed = True

    envs = []
    monkeypatch.setattr(
        spawn_preflight,
        "_scenario_with_episode_seed_defaults",
        lambda scenario, seed: {**scenario, "episode_seed": seed},
    )
    monkeypatch.setattr(spawn_preflight, "build_env_config", lambda *_a, **_kw: object())

    def make_env(**_kwargs):
        env = FakeEnv()
        envs.append(env)
        return env

    monkeypatch.setattr(spawn_preflight, "make_robot_env", make_env)
    monkeypatch.setattr(
        spawn_preflight,
        "reset_spawn_clearance",
        lambda _sim: {
            "overlap": False,
            "robot_obstacle_min_surface_clearance_m": 0.2,
            "robot_pedestrian_min_surface_clearance_m": 0.2,
        },
    )
    monkeypatch.setattr(spawn_preflight, "_static_map_warnings", lambda _sim: [])
    analysis_calls = []

    def build_analysis(*_args, **_kwargs):
        analysis_calls.append(1)
        if analysis_error:
            raise ValueError("fixture geometry unavailable")
        return {"fixture": True}

    monkeypatch.setattr(spawn_preflight, "_build_occupancy_analysis", build_analysis)
    monkeypatch.setattr(
        spawn_preflight,
        "_check_footprint_path",
        lambda *_a, **_kw: (
            {"status": "pass", "reason": "path"},
            {"status": "pass", "reason": "width"},
        ),
    )
    monkeypatch.setattr(
        spawn_preflight,
        "_check_respawn_window",
        lambda *_a, **_kw: {"status": "pass", "reason": "window"},
    )

    result = spawn_preflight._check_release_scenario(
        ({"name": "fixture"}, "matrix.yaml", (111, 112), 0.1, 20, 0.1, False)
    )

    assert [row["seed"] for row in result["rows"]] == [111, 112]
    expected_status = "blocked" if analysis_error else "valid"
    assert all(row["overall_status"] == expected_status for row in result["rows"])
    assert all(row["robot_radius_m"] == pytest.approx(0.3) for row in result["rows"])
    if analysis_error:
        assert all(row["footprint_reachability"]["status"] == "invalid" for row in result["rows"])
        assert "fixture geometry unavailable" in result["rows"][0]["passage_width"]["reason"]
    else:
        assert all(row["footprint_reachability"]["status"] == "pass" for row in result["rows"])
    assert len(analysis_calls) == 1
    assert all(env.closed for env in envs)


def test_release_scenario_keeps_initialization_errors_as_invalid_rows(monkeypatch) -> None:
    class FailingEnv:
        simulator = SimpleNamespace()
        closed = False

        def reset(self, seed):
            del seed
            raise RuntimeError("fixture reset failed")

        def close(self):
            self.closed = True

    env = FailingEnv()
    monkeypatch.setattr(spawn_preflight, "_scenario_with_episode_seed_defaults", lambda *a, **k: {})
    monkeypatch.setattr(spawn_preflight, "build_env_config", lambda *a, **k: object())
    monkeypatch.setattr(spawn_preflight, "make_robot_env", lambda **_kwargs: env)

    result = spawn_preflight._check_release_scenario(
        ({"scenario_id": "fixture"}, "matrix.yaml", (111,), 0.1, 20, 0.1, False)
    )
    row = result["rows"][0]

    assert row["overall_status"] == "blocked"
    assert "fixture reset failed" in row["cell_error"]
    assert row["footprint_reachability"]["status"] == "invalid"
    assert env.closed


def _minimal_manifest_files(tmp_path: Path):
    manifest_path = tmp_path / "release.yaml"
    matrix_path = tmp_path / "matrix.yaml"
    manifest_path.write_text("release: fixture\n", encoding="utf-8")
    matrix_path.write_text("scenarios: fixture\n", encoding="utf-8")
    manifest = SimpleNamespace(
        path=manifest_path,
        scenario_matrix_path=matrix_path,
        seed_policy={},
    )
    return manifest


def test_manifest_runner_returns_complete_diagnostic_for_small_valid_matrix(
    tmp_path: Path, monkeypatch
) -> None:
    manifest = _minimal_manifest_files(tmp_path)
    identity = {
        "manifest_path": "release.yaml",
        "manifest_sha256": sha256_file(manifest.path),
        "scenario_matrix_path": "matrix.yaml",
        "scenario_matrix_sha256": sha256_file(manifest.scenario_matrix_path),
        "seed_set": "fixture-seeds",
        "seed_sets_path": None,
        "seed_sets_sha256": "seed-digest",
    }
    monkeypatch.setattr(
        spawn_preflight,
        "_release_manifest_inputs",
        lambda _manifest: (identity, [{"name": "fixture"}], (111,)),
    )
    monkeypatch.setattr(
        spawn_preflight,
        "_check_release_scenario",
        lambda job: {
            "scenario": job[0]["name"],
            "map_warnings": [{"kind": "fixture-warning"}],
            "rows": [
                {
                    "scenario": job[0]["name"],
                    "seed": job[2][0],
                    "overall_status": "valid",
                    "reset_clearance": {"status": "pass"},
                    "respawn_safety": {"status": "pass"},
                }
            ],
        },
    )

    report = spawn_preflight.run_manifest_preflight(manifest, workers=1, source_commit="abc123")

    assert report["status"] == "valid"
    assert report["evidence_class"] == "preflight_diagnostic_only"
    assert report["benchmark_success"] is None
    assert report["source_commit"] == "abc123"
    assert report["expected_cell_count"] == report["cell_count"] == 1
    assert report["map_warnings"] == {"fixture": [{"kind": "fixture-warning"}]}
    assert report["rows"][0]["scenario_matrix_sha256"] == identity["scenario_matrix_sha256"]


def test_probe_manifest_is_labelled_diagnostic_and_never_passes_release_gate(
    tmp_path: Path, monkeypatch
) -> None:
    manifest = _minimal_manifest_files(tmp_path)
    manifest.release_kind = spawn_preflight.INFEASIBILITY_PROBE_RELEASE_KIND
    identity = {
        "manifest_path": "release.yaml",
        "manifest_sha256": sha256_file(manifest.path),
        "release_kind": manifest.release_kind,
        "scenario_matrix_path": "matrix.yaml",
        "scenario_matrix_sha256": sha256_file(manifest.scenario_matrix_path),
        "seed_set": "fixture-seeds",
        "seed_sets_path": None,
        "seed_sets_sha256": None,
    }
    monkeypatch.setattr(
        spawn_preflight,
        "_release_manifest_inputs",
        lambda _manifest: (identity, [{"name": "doorway"}], (111,)),
    )

    def check_probe_job(job):
        assert job[6] is True
        return {
            "scenario": "doorway",
            "map_warnings": [],
            "rows": [
                {
                    "scenario": "doorway",
                    "seed": 111,
                    "overall_status": "infeasibility_probe",
                    "footprint_reachability": {"status": "exempt_expected_outcome"},
                    "passage_width": {"status": "exempt_expected_outcome"},
                }
            ],
        }

    monkeypatch.setattr(spawn_preflight, "_check_release_scenario", check_probe_job)
    report = spawn_preflight.run_manifest_preflight(manifest)

    assert report["status"] == "blocked"
    assert report["blocked_cell_count"] == 1
    assert report["release_inputs"]["release_kind"] == manifest.release_kind
    assert report["rows"][0]["overall_status"] == "infeasibility_probe"

    monkeypatch.setattr(
        spawn_preflight,
        "_check_release_scenario",
        lambda job: {
            "scenario": job[0]["name"],
            "map_warnings": [],
            "rows": [{"scenario": "doorway", "seed": 111, "overall_status": "valid"}],
        },
    )
    feasible_probe = spawn_preflight.run_manifest_preflight(manifest)
    assert feasible_probe["blocked_cell_count"] == 0
    assert feasible_probe["status"] == "blocked"


def test_manifest_runner_preserves_cells_when_worker_fails(tmp_path: Path, monkeypatch) -> None:
    manifest = _minimal_manifest_files(tmp_path)
    identity = {
        "manifest_path": "release.yaml",
        "manifest_sha256": sha256_file(manifest.path),
        "scenario_matrix_path": "matrix.yaml",
        "scenario_matrix_sha256": sha256_file(manifest.scenario_matrix_path),
        "seed_set": "fixture-seeds",
        "seed_sets_path": None,
        "seed_sets_sha256": None,
    }
    monkeypatch.setattr(
        spawn_preflight,
        "_release_manifest_inputs",
        lambda _manifest: (identity, [{"scenario_id": "broken"}], (111, 112)),
    )
    monkeypatch.setattr(
        spawn_preflight,
        "_check_release_scenario",
        lambda _job: (_ for _ in ()).throw(RuntimeError("worker fixture failure")),
    )

    report = spawn_preflight.run_manifest_preflight(manifest, workers=1)

    assert report["status"] == "blocked"
    assert report["cell_count"] == report["expected_cell_count"] == 2
    assert report["input_error"].startswith("matrix_worker_failed: RuntimeError")
    assert all(row["overall_status"] == "blocked" for row in report["rows"])
    assert all(row["footprint_reachability"]["status"] == "invalid" for row in report["rows"])


def test_manifest_runner_marks_changed_inputs_and_invalid_options(
    tmp_path: Path, monkeypatch
) -> None:
    manifest = _minimal_manifest_files(tmp_path)
    identity = {
        "manifest_path": "release.yaml",
        "manifest_sha256": sha256_file(manifest.path),
        "scenario_matrix_path": "matrix.yaml",
        "scenario_matrix_sha256": sha256_file(manifest.scenario_matrix_path),
        "seed_set": None,
        "seed_sets_path": None,
        "seed_sets_sha256": None,
    }
    monkeypatch.setattr(
        spawn_preflight,
        "_release_manifest_inputs",
        lambda _manifest: (identity, [{"name": "fixture"}], (111,)),
    )
    monkeypatch.setattr(
        spawn_preflight,
        "_check_release_scenario",
        lambda _job: {
            "scenario": "fixture",
            "map_warnings": [],
            "rows": [{"scenario": "fixture", "seed": 111, "overall_status": "valid"}],
        },
    )
    with pytest.raises(ValueError, match="workers"):
        spawn_preflight.run_manifest_preflight(manifest, workers=9)

    with pytest.raises(ValueError, match="grid_resolution_m"):
        spawn_preflight.run_manifest_preflight(manifest, grid_resolution_m=0.0)

    manifest.path.write_text("release: changed\n", encoding="utf-8")
    report = spawn_preflight.run_manifest_preflight(manifest)
    assert report["status"] == "blocked"
    assert report["input_error"] == "release manifest changed while preflight was running"
