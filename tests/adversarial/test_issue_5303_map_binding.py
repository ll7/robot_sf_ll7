"""Regression coverage for the issue #5303 v2 map/template pedestrian binding."""

from __future__ import annotations

from pathlib import Path

from robot_sf.benchmark.issue_5303_search_promotion_preflight import (
    evaluate_preflight_from_files,
    load_template_scenario,
)
from robot_sf.nav.svg_map_parser import convert_map
from robot_sf.training.scenario_loader import apply_single_pedestrian_overrides

ROOT = Path(__file__).resolve().parents[2]
MAP_PATH = ROOT / "maps/svg_maps/classic_group_crossing.svg"
SEARCH_SPACE_PATH = ROOT / "configs/adversarial/issue_5303_search_promotion_space_v2.yaml"
TEMPLATE_PATH = ROOT / "configs/adversarial/issue_5303_classic_group_crossing_medium_v2.yaml"


def test_issue_5303_v2_template_binds_to_a_parser_visible_map_pedestrian() -> None:
    """The frozen v2 pair must bind its candidate identity before search execution."""
    map_definition = convert_map(str(MAP_PATH))
    assert map_definition is not None

    template_scenario = load_template_scenario(TEMPLATE_PATH)
    candidate_id = template_scenario["single_pedestrians"][0]["id"]
    search_space_result = evaluate_preflight_from_files(
        search_space_path=SEARCH_SPACE_PATH,
        scenario_template_path=TEMPLATE_PATH,
    )

    pedestrians_by_id = {
        pedestrian.id: pedestrian for pedestrian in map_definition.single_pedestrians
    }
    assert candidate_id in pedestrians_by_id

    apply_single_pedestrian_overrides(
        map_definition,
        [template_scenario["single_pedestrians"][0]],
    )
    bound_pedestrian = next(
        pedestrian
        for pedestrian in map_definition.single_pedestrians
        if pedestrian.id == candidate_id
    )
    assert bound_pedestrian.start == (2.0, 3.0)
    assert search_space_result.status == "promotion_timing_ready"
    assert search_space_result.materialized_pedestrian_id == candidate_id
    assert search_space_result.single_pedestrian_populated is True
    assert search_space_result.pedestrian_route_populated is True
