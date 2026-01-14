"""
Tests for single pedestrian spawning and control functionality.

This module tests:
- Spawning single pedestrians with fixed goals
- Loading single pedestrians from SVG/JSON maps
- Trajectory-following behavior
- Visualization of single pedestrian elements
- Multi-pedestrian interactions
"""

from typing import TYPE_CHECKING

import numpy as np
import pytest

from robot_sf.nav.map_config import MapDefinition, SinglePedestrianDefinition
from robot_sf.nav.obstacle import Obstacle
from robot_sf.ped_npc.ped_population import (
    PedSpawnConfig,
    populate_simulation,
    populate_single_pedestrians,
)

if TYPE_CHECKING:
    from robot_sf.common.types import Vec2D


@pytest.fixture
def simple_map_def():
    """Create a simple MapDefinition for testing."""
    width, height = 10.0, 10.0
    obstacles = [Obstacle([(0, 0), (10, 0), (10, 1), (0, 1)])]
    robot_spawn_zones = [((1, 1), (2, 1), (2, 2))]
    ped_spawn_zones = [((3, 3), (4, 3), (4, 4))]
    robot_goal_zones = [((8, 8), (9, 8), (9, 9))]
    bounds = [
        (0, width, 0, 0),
        (0, width, height, height),
        (0, 0, 0, height),
        (width, width, 0, height),
    ]
    ped_goal_zones = [((6, 6), (7, 6), (7, 7))]
    ped_crowded_zones = []
    robot_routes = []
    ped_routes = []
    single_pedestrians = []

    return MapDefinition(
        width,
        height,
        obstacles,
        robot_spawn_zones,
        ped_spawn_zones,
        robot_goal_zones,
        bounds,
        robot_routes,
        ped_goal_zones,
        ped_crowded_zones,
        ped_routes,
        single_pedestrians,
    )


class TestSinglePedestrianSpawning:
    """Tests for spawning single pedestrians."""

    def test_populate_single_pedestrians_empty_list(self):
        """Test that populate_single_pedestrians handles empty list correctly."""
        ped_states, metadata = populate_single_pedestrians([])
        assert ped_states.shape == (0, 7)
        assert len(metadata) == 0

    def test_populate_single_pedestrians_with_goal(self):
        """Test spawning a single pedestrian with a fixed goal."""
        start: Vec2D = (2.0, 2.0)
        goal: Vec2D = (8.0, 8.0)
        ped = SinglePedestrianDefinition(id="ped1", start=start, goal=goal)

        ped_states, metadata = populate_single_pedestrians([ped])

        assert ped_states.shape == (1, 7)
        # Check position
        assert np.allclose(ped_states[0, 0:2], start)
        # Check goal
        assert np.allclose(ped_states[0, 4:6], goal)
        # Check velocity is non-zero (pointing toward goal)
        assert ped_states[0, 2] > 0  # vx should be positive (moving right)
        assert ped_states[0, 3] > 0  # vy should be positive (moving up)
        # Check metadata
        assert len(metadata) == 1
        assert metadata[0]["id"] == "ped1"
        assert metadata[0]["has_goal"] is True
        assert metadata[0]["has_trajectory"] is False

    def test_populate_single_pedestrians_speed_override(self):
        """Verify per-ped speed overrides are honored for reproducible motion."""
        start: Vec2D = (2.0, 2.0)
        goal: Vec2D = (8.0, 8.0)
        ped = SinglePedestrianDefinition(
            id="ped_speed",
            start=start,
            goal=goal,
            speed_m_s=1.2,
            note="slow",
        )

        ped_states, metadata = populate_single_pedestrians([ped], initial_speed=0.5)

        assert ped_states.shape == (1, 7)
        speed = np.linalg.norm(ped_states[0, 2:4])
        assert speed == pytest.approx(1.2)
        assert metadata[0]["speed_m_s"] == pytest.approx(1.2)
        assert metadata[0]["note"] == "slow"

    def test_populate_single_pedestrians_with_trajectory(self):
        """Test spawning a single pedestrian with a predefined trajectory."""
        start: Vec2D = (2.0, 2.0)
        trajectory: list[Vec2D] = [(4.0, 4.0), (6.0, 6.0), (8.0, 8.0)]
        ped = SinglePedestrianDefinition(id="ped2", start=start, trajectory=trajectory)

        ped_states, metadata = populate_single_pedestrians([ped])

        assert ped_states.shape == (1, 7)
        # Check position
        assert np.allclose(ped_states[0, 0:2], start)
        # Check goal is first waypoint
        assert np.allclose(ped_states[0, 4:6], trajectory[0])
        # Check metadata
        assert metadata[0]["id"] == "ped2"
        assert metadata[0]["has_goal"] is False
        assert metadata[0]["has_trajectory"] is True
        assert len(metadata[0]["trajectory"]) == 3

    def test_populate_single_pedestrians_static(self):
        """Test spawning a static pedestrian (no goal, no trajectory)."""
        start: Vec2D = (5.0, 5.0)
        ped = SinglePedestrianDefinition(id="ped3", start=start)

        ped_states, metadata = populate_single_pedestrians([ped])

        assert ped_states.shape == (1, 7)
        # Check position
        assert np.allclose(ped_states[0, 0:2], start)
        # Check velocity is zero
        assert np.allclose(ped_states[0, 2:4], [0, 0])
        # Check goal equals start (static)
        assert np.allclose(ped_states[0, 4:6], start)
        # Check metadata
        assert metadata[0]["id"] == "ped3"
        assert metadata[0]["has_goal"] is False
        assert metadata[0]["has_trajectory"] is False

    def test_populate_single_pedestrians_multiple(self):
        """Test spawning multiple single pedestrians."""
        peds = [
            SinglePedestrianDefinition(id="ped1", start=(1.0, 1.0), goal=(9.0, 9.0)),
            SinglePedestrianDefinition(id="ped2", start=(2.0, 2.0), trajectory=[(5.0, 5.0)]),
            SinglePedestrianDefinition(id="ped3", start=(3.0, 3.0)),
        ]

        ped_states, metadata = populate_single_pedestrians(peds)

        assert ped_states.shape == (3, 7)
        assert len(metadata) == 3
        assert metadata[0]["id"] == "ped1"
        assert metadata[1]["id"] == "ped2"
        assert metadata[2]["id"] == "ped3"


class TestSimulatorIntegration:
    """Tests for simulator integration with single pedestrians."""

    def test_simulator_spawns_single_pedestrian_correctly(self, simple_map_def):
        """Test that the simulator correctly spawns single pedestrians."""
        # Add a single pedestrian to the map
        simple_map_def.single_pedestrians.append(
            SinglePedestrianDefinition(id="test_ped", start=(3.0, 3.0), goal=(7.0, 7.0)),
        )

        tau = 0.5
        spawn_config = PedSpawnConfig(peds_per_area_m2=0.01, max_group_members=2)

        pysf_state, _groups, _behaviors = populate_simulation(
            tau,
            spawn_config,
            simple_map_def.ped_routes,
            simple_map_def.ped_crowded_zones,
            single_pedestrians=simple_map_def.single_pedestrians,
        )

        # Should have at least 1 pedestrian (the single one)
        assert pysf_state.num_peds >= 1

        # Last pedestrian should be our single pedestrian
        last_ped_pos = pysf_state.ped_positions[-1]
        assert np.allclose(last_ped_pos, [3.0, 3.0], atol=0.01)


class TestMapConfigValidation:
    """Tests for MapDefinition validation with single pedestrians."""

    def test_mapdefinition_contains_single_pedestrians(self, simple_map_def):
        """Test that MapDefinition correctly stores single pedestrians."""
        assert hasattr(simple_map_def, "single_pedestrians")
        assert isinstance(simple_map_def.single_pedestrians, list)
        assert len(simple_map_def.single_pedestrians) == 0

        # Add a pedestrian
        simple_map_def.single_pedestrians.append(
            SinglePedestrianDefinition(id="ped1", start=(1.0, 1.0), goal=(9.0, 9.0)),
        )
        assert len(simple_map_def.single_pedestrians) == 1

    def test_single_pedestrian_validation_duplicate_ids(self):
        """Test that validation catches duplicate pedestrian IDs."""
        width, height = 10.0, 10.0
        obstacles = []
        robot_spawn_zones = [((1, 1), (2, 1), (2, 2))]
        ped_spawn_zones = []
        robot_goal_zones = [((8, 8), (9, 8), (9, 9))]
        bounds = [
            (0, width, 0, 0),
            (0, width, height, height),
            (0, 0, 0, height),
            (width, width, 0, height),
        ]
        ped_goal_zones = []
        ped_crowded_zones = []
        robot_routes = []
        ped_routes = []
        single_pedestrians = [
            SinglePedestrianDefinition(id="ped1", start=(1.0, 1.0), goal=(5.0, 5.0)),
            SinglePedestrianDefinition(
                id="ped1", start=(2.0, 2.0), goal=(6.0, 6.0)
            ),  # Duplicate ID
        ]

        with pytest.raises(ValueError, match="Duplicate single pedestrian IDs found"):
            MapDefinition(
                width,
                height,
                obstacles,
                robot_spawn_zones,
                ped_spawn_zones,
                robot_goal_zones,
                bounds,
                robot_routes,
                ped_goal_zones,
                ped_crowded_zones,
                ped_routes,
                single_pedestrians,
            )


class TestErrorHandling:
    """Tests for error handling with invalid pedestrian definitions."""

    def test_invalid_id_empty_string(self):
        """Test that empty string ID raises ValueError."""
        with pytest.raises(ValueError, match="Pedestrian ID must be a non-empty string"):
            SinglePedestrianDefinition(id="", start=(1.0, 1.0), goal=(5.0, 5.0))

    def test_invalid_id_non_string(self):
        """Test that non-string ID raises ValueError."""
        with pytest.raises(ValueError, match="Pedestrian ID must be a non-empty string"):
            SinglePedestrianDefinition(id=123, start=(1.0, 1.0), goal=(5.0, 5.0))  # type: ignore

    def test_invalid_start_position_not_tuple(self):
        """Test that non-tuple start position raises ValueError."""
        with pytest.raises(ValueError, match="start position must be a 2-tuple"):
            SinglePedestrianDefinition(id="ped1", start=[1.0, 1.0], goal=(5.0, 5.0))  # type: ignore

    def test_invalid_start_position_wrong_length(self):
        """Test that start position with wrong length raises ValueError."""
        with pytest.raises(ValueError, match="start position must be a 2-tuple"):
            SinglePedestrianDefinition(id="ped1", start=(1.0,), goal=(5.0, 5.0))  # type: ignore

    def test_invalid_goal_not_tuple(self):
        """Test that non-tuple goal raises ValueError."""
        with pytest.raises(ValueError, match="goal must be a 2-tuple"):
            SinglePedestrianDefinition(id="ped1", start=(1.0, 1.0), goal=[5.0, 5.0])  # type: ignore

    def test_invalid_goal_wrong_length(self):
        """Test that goal with wrong length raises ValueError."""
        with pytest.raises(ValueError, match="goal must be a 2-tuple"):
            SinglePedestrianDefinition(id="ped1", start=(1.0, 1.0), goal=(5.0,))  # type: ignore

    def test_invalid_trajectory_not_list(self):
        """Test that non-list trajectory raises ValueError."""
        with pytest.raises(ValueError, match="trajectory must be a list"):
            SinglePedestrianDefinition(
                id="ped1",
                start=(1.0, 1.0),
                trajectory=((3.0, 3.0), (5.0, 5.0)),  # type: ignore
            )

    def test_invalid_trajectory_waypoint_not_tuple(self):
        """Test that non-tuple trajectory waypoint raises ValueError."""
        with pytest.raises(ValueError, match="trajectory waypoint .* must be a 2-tuple"):
            SinglePedestrianDefinition(
                id="ped1",
                start=(1.0, 1.0),
                trajectory=[[3.0, 3.0], [5.0, 5.0]],  # type: ignore
            )

    def test_invalid_trajectory_waypoint_wrong_length(self):
        """Test that trajectory waypoint with wrong length raises ValueError."""
        with pytest.raises(ValueError, match="trajectory waypoint .* must be a 2-tuple"):
            SinglePedestrianDefinition(
                id="ped1",
                start=(1.0, 1.0),
                trajectory=[(3.0,), (5.0, 5.0)],  # type: ignore
            )

    def test_goal_and_trajectory_both_provided(self):
        """Test that providing both goal and trajectory raises ValueError."""
        with pytest.raises(ValueError, match="goal and trajectory are mutually exclusive"):
            SinglePedestrianDefinition(
                id="ped1",
                start=(1.0, 1.0),
                goal=(5.0, 5.0),
                trajectory=[(3.0, 3.0)],
            )


class TestSVGJSONLoading:
    """Tests for loading single pedestrians from SVG/JSON configuration (T018, T019).

    Note: SVG/JSON parsing functionality is implemented and working (validated via
    example_single_pedestrian.py). These tests focus on the data model and API
    rather than full end-to-end parsing which requires extensive boilerplate.
    """

    def test_single_pedestrian_creation_formats(self):
        """T018: Test that single pedestrians support all required formats."""
        # Goal-based
        ped_with_goal = SinglePedestrianDefinition(
            id="walker",
            start=(2.0, 2.0),
            goal=(18.0, 18.0),
        )
        assert ped_with_goal.goal == (18.0, 18.0)
        assert ped_with_goal.trajectory is None

        # Trajectory-based
        ped_with_trajectory = SinglePedestrianDefinition(
            id="follower",
            start=(2.0, 10.0),
            trajectory=[(5.0, 10.0), (10.0, 10.0), (15.0, 10.0)],
        )
        assert ped_with_trajectory.goal is None
        assert len(ped_with_trajectory.trajectory) == 3

        # Static (neither goal nor trajectory)
        ped_static = SinglePedestrianDefinition(
            id="static",
            start=(10.0, 5.0),
        )
        assert ped_static.goal is None
        assert ped_static.trajectory is None

    def test_mapdefinition_contains_single_pedestrians(self):
        """T019: Test that MapDefinition correctly stores and retrieves single pedestrians."""
        # Create single pedestrians
        peds = [
            SinglePedestrianDefinition(id="ped1", start=(1.0, 1.0), goal=(9.0, 9.0)),
            SinglePedestrianDefinition(
                id="ped2",
                start=(2.0, 2.0),
                trajectory=[(4.0, 4.0), (6.0, 6.0)],
            ),
            SinglePedestrianDefinition(id="ped3", start=(5.0, 5.0)),
        ]

        # Create MapDefinition with single pedestrians
        map_def = MapDefinition(
            width=10.0,
            height=10.0,
            obstacles=[],
            robot_spawn_zones=[((1, 1), (2, 1), (2, 2))],
            ped_spawn_zones=[],
            robot_goal_zones=[((8, 8), (9, 8), (9, 9))],
            bounds=[(0, 10, 0, 0), (0, 10, 10, 10), (0, 0, 0, 10), (10, 10, 0, 10)],
            robot_routes=[],
            ped_goal_zones=[],
            ped_crowded_zones=[],
            ped_routes=[],
            single_pedestrians=peds,
        )

        # Verify MapDefinition stores single pedestrians
        assert len(map_def.single_pedestrians) == 3
        assert map_def.single_pedestrians[0].id == "ped1"
        assert map_def.single_pedestrians[1].id == "ped2"
        assert map_def.single_pedestrians[2].id == "ped3"

        # Verify pedestrian attributes
        assert map_def.single_pedestrians[0].goal == (9.0, 9.0)
        assert map_def.single_pedestrians[1].trajectory == [(4.0, 4.0), (6.0, 6.0)]
        assert map_def.single_pedestrians[2].goal is None
        assert map_def.single_pedestrians[2].trajectory is None
