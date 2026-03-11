"""Tests for robot-robot collision propagation in RobotState."""

from types import SimpleNamespace

from robot_sf.robot.robot_state import RobotState


class _DummySensors:
    """Minimal sensor wrapper used by RobotState tests."""

    def reset_cache(self) -> None:
        return None

    def next_obs(self) -> dict[str, float]:
        return {"ok": 1.0}


class _DummyOccupancy:
    """Occupancy stub exposing collision flags used by RobotState."""

    is_pedestrian_collision = False
    is_obstacle_collision = False
    is_dynamic_collision = True
    is_robot_at_goal = False


def test_robot_state_sets_robot_collision_from_dynamic_collision() -> None:
    """RobotState.step should map occupancy dynamic collision to robot collision."""
    state = RobotState(
        nav=SimpleNamespace(reached_waypoint=False, reached_destination=False),
        occupancy=_DummyOccupancy(),
        sensors=_DummySensors(),
        d_t=0.1,
        sim_time_limit=10.0,
    )
    state.step()
    assert state.is_collision_with_robot is True
    assert state.meta_dict()["is_robot_collision"] is True
    assert state.meta_dict()["distance_to_goal"] == 0.0
    assert state.meta_dict()["prev_distance_to_goal"] == 0.0
    assert state.is_terminal is True


def test_robot_state_tracks_distance_to_goal_progress() -> None:
    """RobotState should emit previous/current distance-to-goal metadata."""
    nav = SimpleNamespace(
        reached_waypoint=False,
        reached_destination=False,
        waypoints=[(0.0, 0.0), (10.0, 0.0)],
        pos=(0.0, 0.0),
    )
    state = RobotState(
        nav=nav,
        occupancy=_DummyOccupancy(),
        sensors=_DummySensors(),
        d_t=0.1,
        sim_time_limit=10.0,
    )
    state.reset()
    nav.pos = (4.0, 0.0)
    state.step()
    meta = state.meta_dict()
    assert meta["prev_distance_to_goal"] == 10.0
    assert meta["distance_to_goal"] == 6.0
