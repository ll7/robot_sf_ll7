"""Tests for pedestrian speed metadata emitted by PedestrianState."""

from robot_sf.ped_ego.pedestrian_state import PedestrianState


class _DummySensors:
    """Minimal sensor wrapper used by PedestrianState tests."""

    def __init__(self, speed: float) -> None:
        self._speed = speed

    def reset_cache(self) -> None:
        return None

    def next_obs(self) -> dict[str, float]:
        return {"ok": 1.0}

    def robot_speed_sensor(self) -> tuple[float, float]:
        return (self._speed, 0.0)


class _DummyRobotOccupancy:
    """Occupancy stub exposing robot-facing flags used by PedestrianState."""

    is_robot_at_goal = False
    is_obstacle_collision = False
    is_pedestrian_collision = False


class _DummyEgoPedOccupancy:
    """Occupancy stub exposing ego-pedestrian-facing flags used by PedestrianState."""

    is_pedestrian_collision = False
    is_obstacle_collision = False
    is_agent_agent_collision = False
    distance_to_robot = 3.0


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
