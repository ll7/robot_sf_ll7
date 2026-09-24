"""Regression contract for the bounded issue #9645 search space."""

from __future__ import annotations

from pathlib import Path
from random import Random

import yaml

from robot_sf.adversarial.bundle import build_candidate_payload
from robot_sf.adversarial.config import SearchSpaceConfig

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_issue_9645_space_varies_only_effective_dimensions_and_freezes_environment() -> None:
    """Keep the pilot's searched controls effective and its simulator seed fixed."""
    search_space_path = REPO_ROOT / "configs/adversarial/issue_9645_pilot_space.v1.yaml"
    template_path = REPO_ROOT / "configs/scenarios/templates/crossing_ttc.yaml"
    search_space = SearchSpaceConfig.from_file(search_space_path)
    template = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    template_scenario = template["scenarios"][0]

    assert search_space.pedestrian_id is None
    assert (search_space.spawn_time_s.min, search_space.spawn_time_s.max) == (0.0, 0.0)
    assert (search_space.pedestrian_delay_s.min, search_space.pedestrian_delay_s.max) == (0.0, 0.0)
    assert (search_space.scenario_seed.min, search_space.scenario_seed.max) == (123, 123)
    assert (search_space.start_x.min, search_space.start_x.max) == (2.5, 3.0)
    assert (search_space.goal_x.min, search_space.goal_x.max) == (7.0, 9.0)
    assert (search_space.goal_y.min, search_space.goal_y.max) == (2.5, 4.0)
    assert (search_space.pedestrian_speed_mps.min, search_space.pedestrian_speed_mps.max) == (
        0.8,
        1.4,
    )

    candidate = search_space.sample_candidate(Random(1101))
    specialized, route_payload = build_candidate_payload(
        candidate,
        index=0,
        template_scenario=template_scenario,
        pedestrian_id=search_space.pedestrian_id,
        pedestrian_route_mode=search_space.pedestrian_route_mode,
    )

    assert candidate.scenario_seed == 123
    assert candidate.spawn_time_s == candidate.pedestrian_delay_s == 0.0
    assert specialized["simulation_config"]["route_spawn_seed"] == 123
    assert specialized["simulation_config"]["peds_speed_mult"] == candidate.pedestrian_speed_mps
    assert route_payload["robot_routes"][0]["waypoints"] == [
        candidate.start.as_waypoint(),
        candidate.goal.as_waypoint(),
    ]
    assert route_payload["ped_routes"] == []
    assert "single_pedestrians" not in specialized
