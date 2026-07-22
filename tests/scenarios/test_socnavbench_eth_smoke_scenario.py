"""Test that the committed SocNavBench ETH map loads through the scenario pipeline.

This test proves the REAL committed ``socnavbench_eth.svg`` (PR #4693) parses into a
``MapDefinition`` with the expected structure. It does not require any external data root,
raw SocNavBench asset, or GPU.

Expected SVG structure (from maps/svg_maps/socnavbench/socnavbench_eth.svg):
  - 377 obstacle rectangles
  - 1 robot route (robot_route_0_0)
  - 1 pedestrian route (ped_route_0_0)
  - robot_spawn_zone_0, robot_goal_zone_0
  - ped_spawn_zone_0, ped_goal_zone_0
"""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree

from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.nav.map_config import MapDefinition
from robot_sf.training.scenario_loader import build_robot_config_from_scenario, load_scenarios

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENARIO_PATH = REPO_ROOT / "configs/scenarios/single/socnavbench_eth_smoke.yaml"
EXPECTED_OBSTACLE_COUNT = 377
EXPECTED_MAP_FILE = REPO_ROOT / "maps/svg_maps/socnavbench/socnavbench_eth.svg"
EXPECTED_ROBOT_ROUTE_ID = "robot_route_0_0"
EXPECTED_PED_ROUTE_ID = "ped_route_0_0"
EXPECTED_ZONE_IDS = {
    "robot_spawn_zone_0",
    "robot_goal_zone_0",
    "ped_spawn_zone_0",
    "ped_goal_zone_0",
}


def _load_eth_scenario_and_map() -> tuple[dict[str, object], MapDefinition]:
    """Load the configured ETH map so every assertion targets the intended committed SVG."""
    scenario = load_scenarios(SCENARIO_PATH)[0]
    configured_map_file = (SCENARIO_PATH.parent / str(scenario["map_file"])).resolve()
    assert configured_map_file == EXPECTED_MAP_FILE
    assert configured_map_file.name == "socnavbench_eth.svg"

    config = build_robot_config_from_scenario(scenario, scenario_path=SCENARIO_PATH)
    map_key = configured_map_file.stem
    assert map_key in config.map_pool.map_defs
    return scenario, config.map_pool.map_defs[map_key]


def test_socnavbench_eth_smoke_loads() -> None:
    """Smoke scenario YAML loads through the existing loader, protecting scenario wiring."""
    scenarios = load_scenarios(SCENARIO_PATH)
    assert len(scenarios) >= 1
    scenario = scenarios[0]
    assert scenario["name"] == "socnavbench_eth_smoke"
    assert "map_file" in scenario
    assert "simulation_config" in scenario


def test_socnavbench_eth_map_file_resolves_to_committed_svg() -> None:
    """Configured map resolves to the committed ETH SVG, preventing another pool entry from passing."""
    _scenario, map_def = _load_eth_scenario_and_map()
    svg_ids = {element.get("id") for element in ElementTree.parse(EXPECTED_MAP_FILE).iter()}
    assert map_def is not None
    assert {EXPECTED_ROBOT_ROUTE_ID, EXPECTED_PED_ROUTE_ID, *EXPECTED_ZONE_IDS} <= svg_ids


def test_socnavbench_eth_map_has_expected_obstacles() -> None:
    """Committed ETH SVG parses to 377 obstacles, catching an incomplete map conversion."""
    _scenario, map_def = _load_eth_scenario_and_map()
    assert len(map_def.obstacles) == EXPECTED_OBSTACLE_COUNT


def test_socnavbench_eth_map_has_robot_route_and_zones() -> None:
    """Robot route and zones retain ETH semantics, preventing a wrong route/zone pairing from passing."""
    _scenario, map_def = _load_eth_scenario_and_map()
    assert len(map_def.robot_routes) == 1
    assert len(map_def.robot_spawn_zones) == 1
    assert len(map_def.robot_goal_zones) == 1
    route = map_def.robot_routes[0]
    assert route.source_path_id == EXPECTED_ROBOT_ROUTE_ID
    assert route.spawn_zone == map_def.robot_spawn_zones[0]
    assert route.goal_zone == map_def.robot_goal_zones[0]


def test_socnavbench_eth_map_has_ped_route_and_zones() -> None:
    """Pedestrian route and zones retain ETH semantics, preventing a wrong route/zone pairing from passing."""
    _scenario, map_def = _load_eth_scenario_and_map()
    assert len(map_def.ped_routes) == 1
    assert len(map_def.ped_spawn_zones) == 1
    assert len(map_def.ped_goal_zones) == 1
    route = map_def.ped_routes[0]
    assert route.source_path_id == EXPECTED_PED_ROUTE_ID
    assert route.spawn_zone == map_def.ped_spawn_zones[0]
    assert route.goal_zone == map_def.ped_goal_zones[0]


def test_socnavbench_eth_map_has_four_bounds() -> None:
    """Map has four bounds, preserving the MapDefinition geometry contract at runtime."""
    _scenario, map_def = _load_eth_scenario_and_map()
    assert len(map_def.bounds) == 4


def test_socnavbench_eth_map_runs_headless_environment_smoke() -> None:
    """Committed ETH map survives headless reset and steps, protecting runtime compatibility."""
    scenario, _map_def = _load_eth_scenario_and_map()
    config = build_robot_config_from_scenario(scenario, scenario_path=SCENARIO_PATH)
    env = make_robot_env(config=config, seed=1134, debug=False)
    try:
        observation, _info = env.reset(seed=1134)
        assert env.observation_space.contains(observation)

        steps_taken = 0
        for _ in range(3):
            observation, _reward, terminated, truncated, info = env.step(env.action_space.sample())
            assert env.observation_space.contains(observation)
            steps_taken += 1
            if terminated or truncated:
                break

        assert steps_taken >= 1
        assert isinstance(info, dict)
    finally:
        env.close()
