"""Tests for pedestrian speed metadata emitted by PedestrianState."""

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

    @property
    def obstacle_coords(self):
        return []

    @property
    def pedestrian_coords(self):
        return []

    @property
    def agent_heading(self) -> float:
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
