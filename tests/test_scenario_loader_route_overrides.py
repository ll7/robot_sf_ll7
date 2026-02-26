"""Tests for scenario route override artifacts in scenario_loader."""

from __future__ import annotations

from pathlib import Path

import yaml

from robot_sf.training.scenario_loader import (
    _apply_route_overrides,
    build_robot_config_from_scenario,
    resolve_map_definition,
)


def test_apply_route_overrides_clones_map_definition(tmp_path: Path) -> None:
    """Applying route overrides should avoid mutating cached map definitions."""
    scenario_path = Path("configs/scenarios/classic_interactions.yaml").resolve()
    map_file = str(Path("maps/svg_maps/classic_overtaking.svg").resolve())
    map_def = resolve_map_definition(map_file, scenario_path=scenario_path)
    assert map_def is not None
    scenario = {"name": "demo", "map_file": map_file}
    config = build_robot_config_from_scenario(scenario, scenario_path=scenario_path)
    map_name, original_map = next(iter(config.map_pool.map_defs.items()))
    assert original_map is not None
    original_robot_waypoints = list(original_map.robot_routes[0].waypoints)

    override_path = tmp_path / "route_override.yaml"
    override_path.write_text(
        yaml.safe_dump(
            {
                "route_payload": {
                    "robot_routes": [
                        {
                            "spawn_id": 0,
                            "goal_id": 0,
                            "waypoints": [[5.0, 5.0], [10.0, 10.0], [15.0, 15.0]],
                        }
                    ],
                    "ped_routes": [
                        {
                            "spawn_id": 0,
                            "goal_id": 0,
                            "waypoints": [[6.0, 14.0], [12.0, 10.0], [18.0, 6.0]],
                        }
                    ],
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    _apply_route_overrides(config, str(override_path), scenario_path)

    updated_map = config.map_pool.map_defs[map_name]
    assert updated_map is not original_map
    assert original_map.robot_routes[0].waypoints == original_robot_waypoints
    assert updated_map.robot_routes[0].waypoints == [(5.0, 5.0), (10.0, 10.0), (15.0, 15.0)]
    assert updated_map.ped_routes[0].waypoints == [(6.0, 14.0), (12.0, 10.0), (18.0, 6.0)]


def test_build_robot_config_from_scenario_supports_route_overrides_file(tmp_path: Path) -> None:
    """Scenario configs should accept route_overrides_file and apply both entity route sets."""
    scenario_path = Path("configs/scenarios/classic_interactions.yaml").resolve()
    map_file = str(Path("maps/svg_maps/classic_overtaking.svg").resolve())
    override_path = tmp_path / "route_override.yaml"
    override_path.write_text(
        yaml.safe_dump(
            {
                "robot_routes": [
                    {
                        "spawn_id": 0,
                        "goal_id": 0,
                        "waypoints": [[4.0, 4.0], [8.0, 8.0], [12.0, 12.0]],
                    }
                ],
                "ped_routes": [
                    {
                        "spawn_id": 0,
                        "goal_id": 0,
                        "waypoints": [[5.0, 16.0], [10.0, 10.0], [15.0, 4.0]],
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    scenario = {
        "name": "demo",
        "map_file": map_file,
        "route_overrides_file": str(override_path),
    }

    config = build_robot_config_from_scenario(scenario, scenario_path=scenario_path)
    _map_name, updated_map = next(iter(config.map_pool.map_defs.items()))
    assert updated_map.robot_routes[0].waypoints == [(4.0, 4.0), (8.0, 8.0), (12.0, 12.0)]
    assert updated_map.ped_routes[0].waypoints == [(5.0, 16.0), (10.0, 10.0), (15.0, 4.0)]
