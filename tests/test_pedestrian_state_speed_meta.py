"""Tests for pedestrian speed metadata emitted by PedestrianState."""

import numpy as np

from robot_sf.ped_ego.pedestrian_state import PedestrianState


class _DummySensors:
    """Minimal sensor wrapper used by PedestrianState tests."""

    def __init__(self, speed: float, angular_speed: float = 0.0) -> None:
        self._speed = speed
        self._angular_speed = angular_speed

    def reset_cache(self) -> None:
        return None

    def next_obs(self) -> dict[str, float]:
        return {"ok": 1.0}

    def robot_speed_sensor(self) -> tuple[float, float]:
        return (self._speed, self._angular_speed)


class _DummyRobotOccupancy:
    """Occupancy stub exposing robot-facing flags used by PedestrianState."""

    is_robot_at_goal = False
    is_obstacle_collision = False
    is_pedestrian_collision = False

    def __init__(self) -> None:
        self._agent = (3.0, 0.0)
        self._heading = 0.0

    def get_agent_coords(self) -> tuple[float, float]:
        return self._agent

    def set_agent_coords(self, x: float, y: float) -> None:
        self._agent = (x, y)

    def set_heading(self, heading: float | None) -> None:
        self._heading = heading

    @property
    def obstacle_coords(self):
        return []

    @property
    def pedestrian_coords(self):
        return []

    @property
    def agent_heading(self) -> float | None:
        return self._heading


class _DummyEgoPedOccupancy:
    """Occupancy stub exposing ego-pedestrian-facing flags used by PedestrianState."""

    is_pedestrian_collision = False
    is_obstacle_collision = False
    is_agent_agent_collision = False
    distance_to_robot = 3.0

    def __init__(self) -> None:
        self._agent = (0.0, 0.0)
        self._enemy = (3.0, 0.0)

    def get_agent_coords(self) -> tuple[float, float]:
        return self._agent

    def set_agent_coords(self, x: float, y: float) -> None:
        self._agent = (x, y)

    def get_enemy_coords(self) -> tuple[float, float]:
        return self._enemy

    @property
    def obstacle_coords(self):
        return []

    @property
    def pedestrian_coords(self):
        return []


def test_pedestrian_state_meta_includes_ego_ped_speed() -> None:
    """PedestrianState metadata should include scalar ego pedestrian speed aliases."""
    state = PedestrianState(
        robot_occupancy=_DummyRobotOccupancy(),
        ego_ped_occupancy=_DummyEgoPedOccupancy(),
        sensors=_DummySensors(speed=0.42),
        d_t=0.1,
        sim_time_limit=10.0,
    )

    state.reset()
    state.step()
    meta = state.meta_dict()

    assert meta["ego_ped_speed"] == 0.42
    assert meta["collision_impact_angle_rad"] == 0.0
    assert meta["robot_ped_collision_zone"] == "none"


def test_pedestrian_state_uses_linear_speed_component_only() -> None:
    """Do not mix angular velocity into the reported ego-pedestrian speed metric."""
    state = PedestrianState(
        robot_occupancy=_DummyRobotOccupancy(),
        ego_ped_occupancy=_DummyEgoPedOccupancy(),
        sensors=_DummySensors(speed=1.0, angular_speed=0.5),
        d_t=0.1,
        sim_time_limit=10.0,
    )

    state.reset()
    state.step()

    assert state.ego_ped_speed == 1.0


def test_pedestrian_state_collision_meta_includes_impact_kinematics() -> None:
    """Robot-ped collision metadata should include impact speed and side-of-impact zone."""
    robot_occ = _DummyRobotOccupancy()
    robot_occ.is_pedestrian_collision = True
    ego_occ = _DummyEgoPedOccupancy()
    ego_occ.is_agent_agent_collision = True
    state = PedestrianState(
        robot_occupancy=robot_occ,
        ego_ped_occupancy=ego_occ,
        sensors=_DummySensors(speed=0.5),
        d_t=0.1,
        sim_time_limit=10.0,
    )

    state.reset()
    ego_occ.set_agent_coords(0.2, 0.0)
    state.step()
    meta = state.meta_dict()

    assert meta["robot_ped_collision_zone"] == "back"
    assert 2.9 <= meta["collision_impact_angle_rad"] <= 3.2


def test_pedestrian_state_reset_clears_flags_and_returns_initial_obs() -> None:
    """Reset should clear episode state and refresh the initial observation."""
    state = PedestrianState(
        robot_occupancy=_DummyRobotOccupancy(),
        ego_ped_occupancy=_DummyEgoPedOccupancy(),
        sensors=_DummySensors(speed=0.0),
        d_t=0.25,
        sim_time_limit=1.0,
    )
    state.is_collision_with_ped = True
    state.is_timeout = True
    state.distance_to_robot = 0.0
    state.robot_ped_collision_zone = "front"

    obs = state.reset()

    assert obs == {"ok": 1.0}
    assert state.episode == 1
    assert state.timestep == 0
    assert state.sim_time_elapsed == 0.0
    assert state.distance_to_robot == float("inf")
    assert state.robot_ped_collision_zone == "none"
    assert state.is_terminal is False


def test_pedestrian_state_max_steps_and_terminal_flags() -> None:
    """Expose max-step rounding and terminal evaluation across all terminal flags."""
    state = PedestrianState(
        robot_occupancy=_DummyRobotOccupancy(),
        ego_ped_occupancy=_DummyEgoPedOccupancy(),
        sensors=_DummySensors(speed=0.0),
        d_t=0.3,
        sim_time_limit=1.0,
    )

    assert state.max_sim_steps == 4

    terminal_flags = [
        "is_timeout",
        "is_collision_with_robot",
        "is_collision_with_ped",
        "is_collision_with_obst",
        "is_robot_at_goal",
        "is_collision_robot_with_obstacle",
        "is_collision_robot_with_pedestrian",
    ]
    for flag in terminal_flags:
        for other in terminal_flags:
            setattr(state, other, False)
        setattr(state, flag, True)
        assert state.is_terminal is True


def test_pedestrian_state_extract_linear_speed_supports_common_input_shapes() -> None:
    """Normalize scalar, list, and ndarray sensor outputs to translational speed."""
    assert PedestrianState._extract_linear_speed(np.array([-2.5, 9.0])) == 2.5
    assert PedestrianState._extract_linear_speed(np.array([])) == 0.0
    assert PedestrianState._extract_linear_speed([-3.5, 4.0]) == 3.5
    assert PedestrianState._extract_linear_speed([]) == 0.0
    assert PedestrianState._extract_linear_speed(-1.25) == 1.25


def test_pedestrian_state_impact_metrics_cover_front_side_back_and_unknown() -> None:
    """Classify collision geometry across all impact-angle branches."""
    robot_occ = _DummyRobotOccupancy()
    ego_occ = _DummyEgoPedOccupancy()
    state = PedestrianState(
        robot_occupancy=robot_occ,
        ego_ped_occupancy=ego_occ,
        sensors=_DummySensors(speed=0.0),
        d_t=0.1,
        sim_time_limit=10.0,
    )

    state.is_collision_with_robot = False
    assert state._compute_robot_ped_impact_metrics() == (0.0, "none")

    state.is_collision_with_robot = True
    robot_occ.set_agent_coords(0.0, 0.0)
    ego_occ.set_agent_coords(1.0, 0.0)
    robot_occ.set_heading(None)
    assert state._compute_robot_ped_impact_metrics() == (0.0, "unknown")

    robot_occ.set_heading(0.0)
    assert state._compute_robot_ped_impact_metrics() == (0.0, "front")

    ego_occ.set_agent_coords(0.0, 1.0)
    side_angle, side_zone = state._compute_robot_ped_impact_metrics()
    assert side_zone == "side"
    assert 1.5 <= side_angle <= 1.7

    ego_occ.set_agent_coords(-1.0, 0.0)
    back_angle, back_zone = state._compute_robot_ped_impact_metrics()
    assert back_zone == "back"
    assert 2.9 <= back_angle <= 3.2


def test_pedestrian_state_step_sets_timeout_and_resets_stale_collision_meta() -> None:
    """Step should update timeout state and clear stale impact metadata without collisions."""
    robot_occ = _DummyRobotOccupancy()
    ego_occ = _DummyEgoPedOccupancy()
    state = PedestrianState(
        robot_occupancy=robot_occ,
        ego_ped_occupancy=ego_occ,
        sensors=_DummySensors(speed=0.2),
        d_t=0.6,
        sim_time_limit=0.5,
    )
    state.reset()
    state.collision_impact_angle_rad = 1.23
    state.robot_ped_collision_zone = "side"

    obs = state.step()

    assert obs == {"ok": 1.0}
    assert state.is_timeout is True
    assert state.collision_impact_angle_rad == 0.0
    assert state.robot_ped_collision_zone == "none"
    assert state.meta_dict()["collision_impact_angle_deg"] == 0.0
