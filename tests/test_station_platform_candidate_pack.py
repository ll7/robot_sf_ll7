"""Contract tests for the exploratory station-platform candidate pack."""

from __future__ import annotations

from pathlib import Path

from robot_sf.training.scenario_loader import build_robot_config_from_scenario, load_scenarios

ROOT = Path(__file__).resolve().parent.parent
PACK_PATH = (
    ROOT / "configs" / "scenarios" / "sets" / "station_platform_candidate_pack_issue736.yaml"
)


def test_station_platform_candidate_pack_defines_four_ordered_variants() -> None:
    """The exploratory pack should stay small, ordered, and explicitly scoped."""
    scenarios = load_scenarios(PACK_PATH, base_dir=PACK_PATH)

    assert [scenario["name"] for scenario in scenarios] == [
        "station_platform_route_flow_low",
        "station_platform_static_furniture_low",
        "station_platform_waiting_passengers_medium",
        "station_platform_dense_stress_optional",
    ]

    for scenario in scenarios:
        metadata = scenario["metadata"]
        assert metadata["archetype"] == "station_platform"
        assert metadata["pack_id"] == "issue_736_station_platform_candidate_pack"
        assert metadata["evaluation_scope"] == "exploratory"
        assert len(scenario["seeds"]) == 3


def test_station_platform_candidate_pack_exposes_distinct_knobs() -> None:
    """The variants should change density and deterministic waiting behavior, not geometry only."""
    scenarios = {scenario["name"]: scenario for scenario in load_scenarios(PACK_PATH)}

    route_flow = scenarios["station_platform_route_flow_low"]
    static = scenarios["station_platform_static_furniture_low"]
    waiting = scenarios["station_platform_waiting_passengers_medium"]
    dense = scenarios["station_platform_dense_stress_optional"]

    assert route_flow["simulation_config"]["ped_density"] == 0.02
    assert route_flow["metadata"]["platform_variant"] == "route_flow_low"
    assert "single_pedestrians" not in route_flow

    assert static["simulation_config"]["ped_density"] == 0.02
    assert static["metadata"]["platform_variant"] == "static_waiting_baseline"
    assert all(
        ped["goal"] is None and ped["trajectory"] is None for ped in static["single_pedestrians"]
    )

    assert waiting["simulation_config"]["ped_density"] == 0.05
    assert waiting["metadata"]["platform_variant"] == "waiting_passengers"
    assert waiting["single_pedestrians"]

    assert dense["simulation_config"]["ped_density"] == 0.11
    assert dense["metadata"]["density_advisory"] == "high_density_stress"
    assert dense["metadata"]["optional"] is True


def test_station_platform_waiting_variant_builds_wait_rules() -> None:
    """The canonical waiting variant should exercise the existing single-pedestrian wait contract."""
    scenarios = load_scenarios(PACK_PATH, base_dir=PACK_PATH)
    waiting = next(
        scenario
        for scenario in scenarios
        if scenario["name"] == "station_platform_waiting_passengers_medium"
    )

    config = build_robot_config_from_scenario(waiting, scenario_path=PACK_PATH)
    map_def = next(iter(config.map_pool.map_defs.values()))
    waiters = [ped for ped in map_def.single_pedestrians if ped.wait_at]

    assert {ped.id for ped in waiters} == {"p1", "p3"}
    assert [rule.wait_s for ped in waiters for rule in ped.wait_at or []] == [6.0, 5.0]
