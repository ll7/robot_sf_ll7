"""Backward compatibility smoke tests for OSM integration (T035).

This test suite verifies that OSM-based map extraction and zones
integration does not break existing robot navigation environments.

Tests verify:
- OSM MapDefinition can be loaded and used
- Robot environments work with OSM-derived maps
- Environment reset/step loops execute without errors
- Basic metrics remain functional

These are smoke tests - they verify functionality without deep assertions.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Polygon, box

from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.maps.osm_zones_yaml import OSMZonesConfig, Route, Zone, save_zones_yaml
from robot_sf.nav.map_config import MapDefinition, MapDefinitionPool
from robot_sf.nav.osm_map_builder import osm_to_map_definition
from robot_sf.planner.visibility_planner import VisibilityPlanner


def _select_spawn_goal_polygons(allowed_areas: list[Polygon]) -> tuple[Polygon, Polygon]:
    """Pick polygons for spawn and goal zones from allowed areas.

    Args:
        allowed_areas: Candidate allowed-area polygons from the OSM MapDefinition.

    Returns:
        Tuple of (spawn_polygon, goal_polygon). Falls back to the same polygon
        when only one valid polygon is available.
    """
    candidates = [poly for poly in allowed_areas if not poly.is_empty]
    if not candidates:
        raise ValueError("No valid allowed_areas polygons available for zone selection.")
    candidates.sort(key=lambda poly: poly.area, reverse=True)
    spawn_poly = candidates[0]
    goal_poly = candidates[1] if len(candidates) > 1 else spawn_poly
    return spawn_poly, goal_poly


def _triangle_from_polygon(polygon: Polygon, size: float = 2.0) -> tuple[tuple[float, float], ...]:
    """Create a small triangle entirely contained within the given polygon.

    Args:
        polygon: Polygon to place the triangle within.
        size: Target triangle edge offset in meters before clamping.

    Returns:
        Triangle vertices as a tuple of (x, y) coordinates.
    """
    inner = polygon.buffer(-size)
    if inner.is_empty:
        inner = polygon

    point = inner.representative_point()
    minx, miny, maxx, maxy = inner.bounds
    span_x = (maxx - minx) / 6.0
    span_y = (maxy - miny) / 6.0
    span = max(min(size, span_x, span_y), 0.1)

    for factor in (1.0, 0.5, 0.25, 0.1):
        offset = span * factor
        triangle = (
            (point.x - offset, point.y - offset),
            (point.x + offset, point.y - offset),
            (point.x, point.y + offset),
        )
        if Polygon(triangle).within(polygon):
            return triangle

    # Last-resort tiny triangle around the point.
    return (
        (point.x, point.y),
        (point.x + 0.1, point.y),
        (point.x, point.y + 0.1),
    )


def _point_from_polygon(polygon: Polygon) -> tuple[float, float]:
    """Return a stable point inside the polygon for route waypoints.

    Args:
        polygon: Polygon to sample from.

    Returns:
        (x, y) coordinates for a waypoint inside the polygon.
    """
    point = polygon.representative_point()
    return (float(point.x), float(point.y))


class TestOSMBackwardCompat:
    """Backward compatibility tests for OSM integration (T035)."""

    @pytest.fixture
    def pbf_file(self) -> str:
        """Path to sample PBF file."""
        return "test_scenarios/osm_fixtures/sample_block.pbf"

    @pytest.fixture
    def osm_map_definition(self, pbf_file: str) -> MapDefinition:
        """Create MapDefinition from OSM PBF file with minimal routing."""
        from robot_sf.nav.global_route import GlobalRoute

        pbf_path = Path(pbf_file)
        if not pbf_path.exists():
            pytest.skip(f"PBF file not found: {pbf_file}")

        map_def = osm_to_map_definition(
            pbf_file=pbf_file,
            line_buffer_m=1.5,
        )

        if not map_def.allowed_areas:
            pytest.skip("OSM MapDefinition has no allowed_areas for zone placement.")

        spawn_poly, goal_poly = _select_spawn_goal_polygons(map_def.allowed_areas)
        spawn_zone = _triangle_from_polygon(spawn_poly)
        goal_zone = _triangle_from_polygon(goal_poly)
        waypoint = _point_from_polygon(spawn_poly)

        map_def.robot_spawn_zones = [spawn_zone]
        map_def.robot_goal_zones = [goal_zone]

        # Add minimal pedestrian zones to avoid warnings
        map_def.ped_spawn_zones = [spawn_zone]
        map_def.ped_goal_zones = [goal_zone]

        # Add minimal route from spawn to goal
        route = GlobalRoute(
            spawn_id=0,
            goal_id=0,
            waypoints=[waypoint],
            spawn_zone=spawn_zone,
            goal_zone=goal_zone,
        )
        map_def.robot_routes = [route]

        # Manually populate robot_routes_by_spawn_id since we modified after __post_init__
        map_def.robot_routes_by_spawn_id = {0: [route]}

        return map_def

    @pytest.fixture
    def map_def_with_allowed_areas(self) -> MapDefinition:
        """Load a sample map and inject allowed_areas to validate compatibility."""
        map_pool = MapDefinitionPool()
        map_name = sorted(map_pool.map_defs.keys())[0]
        map_def = map_pool.map_defs[map_name]
        map_def.allowed_areas = [box(0.0, 0.0, map_def.width, map_def.height)]
        return map_def

    @pytest.fixture
    def sample_zones_config(self) -> OSMZonesConfig:
        """Create sample zones configuration."""
        return OSMZonesConfig(
            zones={
                "spawn_zone": Zone(
                    name="spawn_zone",
                    type="spawn",
                    polygon=[(10, 10), (30, 10), (30, 30), (10, 30)],
                    priority=1,
                ),
                "goal_zone": Zone(
                    name="goal_zone",
                    type="goal",
                    polygon=[(70, 70), (90, 70), (90, 90), (70, 90)],
                    priority=1,
                ),
            },
            routes={
                "route_1": Route(
                    name="route_1",
                    waypoints=[(20, 20), (50, 50), (80, 80)],
                )
            },
        )

    def test_osm_map_definition_structure(self, osm_map_definition: MapDefinition):
        """Test that OSM-derived MapDefinition has expected structure."""
        # Verify MapDefinition has required attributes
        assert hasattr(osm_map_definition, "obstacles")
        assert hasattr(osm_map_definition, "allowed_areas")
        assert hasattr(osm_map_definition, "width")
        assert hasattr(osm_map_definition, "height")

        # Verify data types
        assert isinstance(osm_map_definition.obstacles, list)
        assert isinstance(osm_map_definition.width, (int, float))
        assert isinstance(osm_map_definition.height, (int, float))

        # Verify dimensions are reasonable (from sample_block.pbf)
        assert osm_map_definition.width > 0
        assert osm_map_definition.height > 0
        assert len(osm_map_definition.obstacles) > 0

    def test_osm_map_with_robot_environment(self, osm_map_definition: MapDefinition):
        """Test that robot environment works with OSM-derived MapDefinition.

        This is a critical smoke test verifying backward compatibility.
        """
        map_pool = MapDefinitionPool(
            maps_folder=".",
            map_defs={"osm_smoke": osm_map_definition},
        )
        config = RobotSimulationConfig(map_pool=map_pool)
        env = make_robot_env(config=config, debug=False)

        try:
            obs, info = env.reset()
            assert obs is not None
            assert isinstance(info, dict)

            action = env.action_space.sample()
            obs, _reward, _terminated, _truncated, _info = env.step(action)
            assert obs is not None
        finally:
            env.close()

    def test_environment_reset_step_loop(self):
        """Test environment reset and step loop with standard map.

        This verifies the complete integration works without errors.
        Uses default maps to verify environment behavior.
        """
        # Use default maps for this test
        config = RobotSimulationConfig()
        env = make_robot_env(config=config, debug=False)

        try:
            # Test reset
            obs, info = env.reset()
            assert obs is not None
            assert isinstance(info, dict)

            # Test multiple steps
            for _ in range(5):  # Small number for smoke test
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)

                # Verify output types
                assert obs is not None
                assert isinstance(reward, (int, float, np.number))
                assert isinstance(terminated, (bool, np.bool_))
                assert isinstance(truncated, (bool, np.bool_))
                assert isinstance(info, dict)

                if terminated or truncated:
                    obs, info = env.reset()

        finally:
            env.close()

    def test_zones_yaml_integration(self, sample_zones_config: OSMZonesConfig):
        """Test that zones YAML can be saved and loaded."""
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            yaml_path = f.name

        try:
            # Save zones
            save_zones_yaml(sample_zones_config, yaml_path)

            # Verify file created
            assert Path(yaml_path).exists()

            # Load zones (basic smoke test - actual loading tested elsewhere)
            from robot_sf.maps.osm_zones_yaml import load_zones_yaml

            loaded_config = load_zones_yaml(yaml_path)
            assert len(loaded_config.zones) == 2
            assert len(loaded_config.routes) == 1

        finally:
            # Cleanup
            if Path(yaml_path).exists():
                Path(yaml_path).unlink()

    def test_full_train_eval_cycle(self):
        """Test complete train/eval cycle (T035 acceptance test).

        This is the main acceptance test for T035, verifying that:
        1. Environment can be created
        2. Reset/step loop works without errors
        3. Basic metrics are collected
        4. No breaking changes to existing API after OSM integration

        Uses default maps since OSM integration is orthogonal to environment behavior.
        """
        # Use default maps - the key is that OSM integration didn't break existing behavior
        config = RobotSimulationConfig()
        env = make_robot_env(config=config, debug=False)

        try:
            # Simulate a minimal training/evaluation cycle
            episode_rewards = []
            num_episodes = 2  # Minimal for smoke test

            for episode in range(num_episodes):
                _obs, _info = env.reset()
                episode_reward = 0.0
                done = False
                steps = 0
                max_steps = 10  # Very short episodes for smoke test

                while not done and steps < max_steps:
                    action = env.action_space.sample()
                    _obs, reward, terminated, truncated, _info = env.step(action)

                    episode_reward += float(reward)
                    done = terminated or truncated
                    steps += 1

                episode_rewards.append(episode_reward)

            # Verify basic metrics were collected
            assert len(episode_rewards) == num_episodes
            assert all(isinstance(r, (int, float)) for r in episode_rewards)

            # Verify environment state is valid
            assert env.observation_space is not None
            assert env.action_space is not None

        finally:
            env.close()

    def test_pygame_visualization_unchanged(
        self, map_def_with_allowed_areas: MapDefinition
    ) -> None:
        """Ensure pygame rendering still works when allowed_areas is present to preserve UI workflows."""
        os.environ.setdefault("DISPLAY", "")
        os.environ.setdefault("MPLBACKEND", "Agg")
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

        pygame = pytest.importorskip("pygame")

        from robot_sf.render.sim_view import SimulationView, VisualizableSimState

        pygame.init()
        try:
            sim_view = SimulationView(map_def=map_def_with_allowed_areas, width=200, height=200)
            state = VisualizableSimState(
                timestep=0,
                robot_action=None,
                robot_pose=((1.0, 1.0), 0.0),
                pedestrian_positions=np.zeros((0, 2)),
                ray_vecs=np.zeros((0, 2, 2)),
                ped_actions=np.zeros((0, 2, 2)),
            )
            sim_view.render(state, target_fps=1)
        finally:
            pygame.quit()

    def test_sensor_suite_unchanged(self, map_def_with_allowed_areas: MapDefinition) -> None:
        """Verify sensor initialization still works with allowed_areas populated to protect observations."""
        map_pool = MapDefinitionPool(
            maps_folder=".",
            map_defs={"osm_allowed": map_def_with_allowed_areas},
        )
        config = RobotSimulationConfig(map_pool=map_pool)
        env = make_robot_env(config=config, debug=False)

        try:
            obs, info = env.reset()
            assert obs is not None
            assert isinstance(info, dict)

            action = env.action_space.sample()
            obs, _reward, _terminated, _truncated, _info = env.step(action)
            assert obs is not None
        finally:
            env.close()

    def test_planner_compatibility(self) -> None:
        """Confirm global planner still operates when allowed_areas is set to preserve routing behavior."""
        width = 10.0
        height = 10.0
        spawn_zone = ((0.0, 0.0), (width, 0.0), (width, height))
        map_def = MapDefinition(
            width=width,
            height=height,
            obstacles=[],
            robot_spawn_zones=[spawn_zone],
            ped_spawn_zones=[spawn_zone],
            robot_goal_zones=[spawn_zone],
            bounds=[
                (0.0, width, 0.0, 0.0),
                (0.0, width, height, height),
                (0.0, 0.0, 0.0, height),
                (width, width, 0.0, height),
            ],
            robot_routes=[],
            ped_goal_zones=[spawn_zone],
            ped_crowded_zones=[],
            ped_routes=[],
            allowed_areas=[box(0.0, 0.0, width, height)],
        )

        planner = VisibilityPlanner(map_def)
        path = planner.plan(start=(1.0, 1.0), goal=(9.0, 9.0))
        assert len(path) >= 2

    @pytest.mark.skipif(
        not Path("test_scenarios/osm_fixtures/sample_block.pbf").exists(),
        reason="Sample PBF file not available",
    )
    def test_osm_to_mapdef_preserves_api(self):
        """Test that osm_to_map_definition preserves MapDefinition API."""
        pbf_file = "test_scenarios/osm_fixtures/sample_block.pbf"

        # Call with minimal arguments
        map_def = osm_to_map_definition(pbf_file=pbf_file)

        # Verify returned type
        assert isinstance(map_def, MapDefinition)

        # Verify all expected MapDefinition attributes exist
        required_attrs = [
            "obstacles",
            "allowed_areas",
            "width",
            "height",
            "robot_routes",
            "robot_spawn_zones",
        ]

        for attr in required_attrs:
            assert hasattr(map_def, attr), f"Missing attribute: {attr}"

        # Verify obstacles structure
        assert isinstance(map_def.obstacles, list)
        if map_def.obstacles:
            # Each obstacle should be an Obstacle object (from robot_sf.nav.obstacle)
            obstacle = map_def.obstacles[0]
            from robot_sf.nav.obstacle import Obstacle

            assert isinstance(obstacle, Obstacle)
            assert hasattr(obstacle, "vertices")
            assert len(obstacle.vertices) >= 3  # Polygon needs at least 3 points

        # Verify allowed_areas structure (if present)
        if map_def.allowed_areas:
            assert isinstance(map_def.allowed_areas, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
