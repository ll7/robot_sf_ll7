"""Env factory and navigation integration tests (US7)."""

from robot_sf.gym_env.base_env import attach_planner_to_map
from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.nav.navigation import sample_route
from robot_sf.nav.svg_map_parser import convert_map
from robot_sf.planner import ClassicGlobalPlanner, GlobalPlanner, PlanningError


def test_sample_route_uses_planner_when_enabled():
    """Verify route sampling delegates to the attached planner when enabled."""
    map_def = convert_map("tests/fixtures/test_maps/simple_corridor.svg")

    class FakePlanner:
        """Planner stub that records whether route planning was requested."""

        def __init__(self):
            self.called = False

        def plan(self, start, goal, *, via_pois=None):
            """Return a sentinel route and record planner usage."""
            self.called = True
            return [("start",), ("goal",)]

    fake = FakePlanner()
    map_def._global_planner = fake
    map_def._use_planner = True

    route = sample_route(map_def)

    assert fake.called
    assert route == [("start",), ("goal",)]


def test_sample_route_falls_back_when_planner_disabled():
    """Verify disabled planner routing keeps the parser-derived fallback route."""
    map_def = convert_map("tests/fixtures/test_maps/simple_corridor.svg")

    class FakePlanner:
        """Planner stub that fails if a disabled planner is invoked."""

        def plan(self, *_args, **_kwargs):
            """Raise when the route sampler incorrectly calls this planner."""
            raise AssertionError("Planner should not be used when disabled")

    map_def._global_planner = FakePlanner()
    map_def._use_planner = False

    route = sample_route(map_def)

    assert len(route) >= 2


def test_sample_route_retries_planner_then_succeeds():
    """Verify route sampling retries planner failures before falling back."""
    map_def = convert_map("tests/fixtures/test_maps/simple_corridor.svg")

    class FlakyPlanner:
        """Planner stub that fails once before returning a valid route."""

        def __init__(self):
            self.calls = 0

        def plan(self, *_args, **_kwargs):
            """Fail on the first call and succeed on the retry."""
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
    """Verify planner attachment stores the enabled flag and planner instance."""
    config = RobotSimulationConfig()
    config.use_planner = True
    config.planner_clearance_margin = 0.2
    map_def = convert_map("tests/fixtures/test_maps/simple_corridor.svg")

    attach_planner_to_map(map_def, config)

    assert getattr(map_def, "_use_planner", False) is True
    assert isinstance(
        getattr(map_def, "_global_planner", None),
        (GlobalPlanner, ClassicGlobalPlanner),
    )


def test_attach_planner_to_map_supports_classic_backend():
    """Verify planner attachment can select the classic backend explicitly."""
    config = RobotSimulationConfig(
        use_planner=True,
        planner_backend="classic",
    )
    map_def = convert_map("tests/fixtures/test_maps/simple_corridor.svg")

    attach_planner_to_map(map_def, config)

    assert getattr(map_def, "_use_planner", False) is True
    assert isinstance(getattr(map_def, "_global_planner", None), ClassicGlobalPlanner)
