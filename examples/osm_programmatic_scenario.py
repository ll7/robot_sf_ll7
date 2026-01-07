"""Example: Programmatic scenario creation using OSM zones and routes (Phase 3, T042).

This example demonstrates the programmatic API for creating scenarios
without using the visual editor. Shows:
- Creating zones programmatically (spawn, goal, crowded)
- Creating routes programmatically
- Saving scenarios to YAML
- Loading scenarios back
- Using scenarios with robot environments

Workflow:
    1. Define zones and routes in Python
    2. Create OSMZonesConfig
    3. Save to YAML (for reproducibility)
    4. Use with environment factory
    5. Train or evaluate robot agent

Usage:
    uv run python examples/osm_programmatic_scenario.py
"""

from pathlib import Path

from loguru import logger

from robot_sf.maps.osm_zones_config import (
    create_config_with_zones_routes,
    create_crowded_zone,
    create_goal_zone,
    create_route,
    create_spawn_zone,
    load_scenario_config,
)
from robot_sf.maps.osm_zones_yaml import save_zones_yaml


def example_simple_scenario() -> None:
    """Create a simple 2-zone scenario (spawn → goal)."""
    logger.info("Example 1: Simple scenario (spawn → goal)")

    # Create zones
    spawn = create_spawn_zone(
        "spawn_start",
        polygon=[(5.0, 5.0), (15.0, 5.0), (15.0, 15.0), (5.0, 15.0)],
        priority=1,
        metadata={"description": "Robot starting area"},
    )

    goal = create_goal_zone(
        "goal_exit",
        polygon=[(85.0, 85.0), (95.0, 85.0), (95.0, 95.0), (85.0, 95.0)],
        metadata={"description": "Goal/exit area"},
    )

    # Create route (path from spawn to goal)
    route = create_route(
        "direct_path",
        waypoints=[(10.0, 10.0), (50.0, 50.0), (90.0, 90.0)],
        route_type="pedestrian",
        metadata={"description": "Direct path from spawn to goal"},
    )

    # Create config
    config = create_config_with_zones_routes(
        zones=[spawn, goal],
        routes=[route],
        metadata={"map_source": "simple.pbf", "description": "Simple crossing"},
    )

    # Save to YAML
    output_file = Path("output/scenarios/simple.yaml")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    save_zones_yaml(config, str(output_file))

    logger.info(f"✓ Saved simple scenario to {output_file}")


def example_intersection_scenario() -> None:
    """Create a realistic intersection scenario with multiple spawn/goal zones."""
    logger.info("Example 2: Intersection scenario (multi-agent)")

    zones = []

    # North spawn zone
    zones.append(
        create_spawn_zone(
            "spawn_north",
            polygon=[(45.0, 0.0), (55.0, 0.0), (55.0, 10.0)],
            priority=2,
        )
    )

    # South spawn zone
    zones.append(
        create_spawn_zone(
            "spawn_south",
            polygon=[(45.0, 90.0), (55.0, 90.0), (55.0, 100.0)],
            priority=2,
        )
    )

    # East spawn zone
    zones.append(
        create_spawn_zone(
            "spawn_east",
            polygon=[(90.0, 45.0), (100.0, 45.0), (100.0, 55.0)],
            priority=1,
        )
    )

    # Goal zones (opposite sides)
    zones.append(
        create_goal_zone(
            "goal_north",
            polygon=[(45.0, 90.0), (55.0, 90.0), (55.0, 100.0)],
        )
    )

    zones.append(
        create_goal_zone(
            "goal_south",
            polygon=[(45.0, 0.0), (55.0, 0.0), (55.0, 10.0)],
        )
    )

    zones.append(
        create_goal_zone(
            "goal_east",
            polygon=[(0.0, 45.0), (10.0, 45.0), (10.0, 55.0)],
        )
    )

    # Crowded zone at intersection center
    zones.append(
        create_crowded_zone(
            "intersection_center",
            polygon=[(40.0, 40.0), (60.0, 40.0), (60.0, 60.0), (40.0, 60.0)],
            density=2.5,
            metadata={"pedestrian_type": "mixed", "peak_hours": "8-18"},
        )
    )

    # Routes through intersection
    routes = [
        create_route(
            "north_to_south",
            waypoints=[(50.0, 5.0), (50.0, 50.0), (50.0, 95.0)],
            metadata={"description": "North to south crossing"},
        ),
        create_route(
            "south_to_north",
            waypoints=[(50.0, 95.0), (50.0, 50.0), (50.0, 5.0)],
            metadata={"description": "South to north crossing"},
        ),
        create_route(
            "east_to_west",
            waypoints=[(95.0, 50.0), (50.0, 50.0), (5.0, 50.0)],
            metadata={"description": "East to west crossing"},
        ),
    ]

    # Create config
    config = create_config_with_zones_routes(
        zones=zones,
        routes=routes,
        metadata={"map_source": "intersection.pbf", "description": "Urban intersection"},
    )

    # Save to YAML
    output_file = Path("output/scenarios/intersection.yaml")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    save_zones_yaml(config, str(output_file))

    logger.info(f"✓ Saved intersection scenario to {output_file}")


def example_variable_density_scenario() -> None:
    """Create scenario with varying crowd densities."""
    logger.info("Example 3: Variable density scenario")

    zones = []

    # Spawn and goal zones
    zones.append(
        create_spawn_zone(
            "spawn",
            polygon=[(5.0, 45.0), (15.0, 45.0), (15.0, 55.0), (5.0, 55.0)],
        )
    )

    zones.append(
        create_goal_zone(
            "goal",
            polygon=[(85.0, 45.0), (95.0, 45.0), (95.0, 55.0), (85.0, 55.0)],
        )
    )

    # Three crowd zones with increasing density
    zones.append(
        create_crowded_zone(
            "crowd_sparse",
            polygon=[(20.0, 40.0), (35.0, 40.0), (35.0, 60.0), (20.0, 60.0)],
            density=0.1,
            metadata={"zone": "entry", "difficulty": "easy"},
        )
    )

    zones.append(
        create_crowded_zone(
            "crowd_medium",
            polygon=[(40.0, 35.0), (60.0, 35.0), (60.0, 65.0), (40.0, 65.0)],
            density=1.5,
            metadata={"zone": "center", "difficulty": "medium"},
        )
    )

    zones.append(
        create_crowded_zone(
            "crowd_dense",
            polygon=[(65.0, 40.0), (80.0, 40.0), (80.0, 60.0), (65.0, 60.0)],
            density=4.0,
            metadata={"zone": "exit", "difficulty": "hard"},
        )
    )

    # Routes through varying densities
    routes = [
        create_route(
            "main_path",
            waypoints=[(10.0, 50.0), (50.0, 50.0), (90.0, 50.0)],
            metadata={"description": "Main corridor"},
        ),
        create_route(
            "upper_bypass",
            waypoints=[(10.0, 50.0), (50.0, 30.0), (90.0, 50.0)],
            metadata={"description": "Upper bypass route"},
        ),
        create_route(
            "lower_bypass",
            waypoints=[(10.0, 50.0), (50.0, 70.0), (90.0, 50.0)],
            metadata={"description": "Lower bypass route"},
        ),
    ]

    # Create config
    config = create_config_with_zones_routes(
        zones=zones,
        routes=routes,
        metadata={
            "map_source": "corridor.pbf",
            "description": "Variable density corridor",
        },
    )

    # Save to YAML
    output_file = Path("output/scenarios/variable_density.yaml")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    save_zones_yaml(config, str(output_file))

    logger.info(f"✓ Saved variable density scenario to {output_file}")


def example_load_and_verify() -> None:
    """Load a scenario and verify contents."""
    logger.info("Example 4: Load and verify scenario")

    # Load previously saved scenario
    scenario_file = Path("output/scenarios/intersection.yaml")
    if scenario_file.exists():
        config = load_scenario_config(str(scenario_file))

        logger.info(f"Loaded scenario with {len(config.zones)} zones, {len(config.routes)} routes")

        # Print zone summary
        logger.info("Zones:")
        for name, zone in config.zones.items():
            density_str = (
                f", density={zone.metadata.get('density', 'N/A')}" if zone.type == "crowded" else ""
            )
            logger.info(f"  - {name}: {zone.type}{density_str}")

        # Print route summary
        logger.info("Routes:")
        for name, route in config.routes.items():
            logger.info(f"  - {name}: {len(route.waypoints)} waypoints, type={route.route_type}")
    else:
        logger.warning(f"Scenario file not found: {scenario_file}")


def main() -> None:
    """Run all examples."""
    logger.info("=" * 80)
    logger.info("OSM Programmatic Scenario Creation Examples (Phase 3)")
    logger.info("=" * 80)

    example_simple_scenario()
    logger.info("")

    example_intersection_scenario()
    logger.info("")

    example_variable_density_scenario()
    logger.info("")

    example_load_and_verify()

    logger.info("=" * 80)
    logger.info("✅ All examples completed")
    logger.info("Scenarios saved to: output/scenarios/")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
