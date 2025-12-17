"""Env factory and navigation integration tests (US7)."""
# ruff: noqa: D103

from robot_sf.gym_env.base_env import attach_planner_to_map
from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.nav.navigation import sample_route
from robot_sf.nav.svg_map_parser import convert_map
from robot_sf.planner import ClassicGlobalPlanner, GlobalPlanner, PlanningError


def test_sample_route_uses_planner_when_enabled(tmp_path):
    map_def = convert_map("tests/fixtures/test_maps/simple_corridor.svg")

    class FakePlanner:
        def __init__(self):
            self.called = False

        def plan(self, start, goal, *, via_pois=None):
            self.called = True
            return [("start",), ("goal",)]

    fake = FakePlanner()
    map_def._global_planner = fake
    map_def._use_planner = True

    route = sample_route(map_def)

    assert fake.called
    assert route == [("start",), ("goal",)]


def test_sample_route_falls_back_when_planner_disabled():
    map_def = convert_map("tests/fixtures/test_maps/simple_corridor.svg")

    class FakePlanner:
        def plan(self, *_args, **_kwargs):
            raise AssertionError("Planner should not be used when disabled")

    map_def._global_planner = FakePlanner()
    map_def._use_planner = False

    route = sample_route(map_def)

    assert len(route) >= 2


def test_sample_route_retries_planner_then_succeeds(tmp_path):
    map_def = convert_map("tests/fixtures/test_maps/simple_corridor.svg")

    class FlakyPlanner:
        def __init__(self):
            self.calls = 0

        def plan(self, *_args, **_kwargs):
            self.calls += 1
            if self.calls < 2:
                raise PlanningError("first call fails")
            return [("p0",), ("p1",)]

    planner = FlakyPlanner()
    map_def._global_planner = planner
    map_def._use_planner = True

    route = sample_route(map_def)

    assert planner.calls >= 2
    assert route == [("p0",), ("p1",)]


def test_attach_planner_to_map_sets_flags():
    config = RobotSimulationConfig()
    config.use_planner = True
    config.planner_clearance_margin = 0.2
    map_def = convert_map("tests/fixtures/test_maps/simple_corridor.svg")

    attach_planner_to_map(map_def, config)

    assert getattr(map_def, "_use_planner", False) is True
    assert isinstance(getattr(map_def, "_global_planner", None), GlobalPlanner)


def test_attach_planner_to_map_supports_classic_backend():
    config = RobotSimulationConfig(
        use_planner=True,
        planner_backend="classic",
    )
    map_def = convert_map("tests/fixtures/test_maps/simple_corridor.svg")

    attach_planner_to_map(map_def, config)

    assert getattr(map_def, "_use_planner", False) is True
    assert isinstance(getattr(map_def, "_global_planner", None), ClassicGlobalPlanner)
