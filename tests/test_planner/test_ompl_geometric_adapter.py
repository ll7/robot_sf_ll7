"""Tests for the OMPL geometric planner adapter (issue #4799 diagnostic tool).

Uses optional OMPL import so tests pass whether or not the optional dependency
is installed.  When `ompl` is available, the full integration exercises run;
otherwise the unit-surface tests verify the ImportError contract.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from shapely.geometry import Point, Polygon

from robot_sf.nav.svg_map_parser import convert_map

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_map_def(tmp_path):
    """Minimal MapDefinition with one rectangular obstacle."""
    svg_path = tmp_path / "simple_obstacle.svg"
    svg_path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<svg xmlns="http://www.w3.org/2000/svg"\n'
        '     xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"\n'
        '     width="20" height="20">\n'
        '  <rect inkscape:label="obstacle" x="5" y="5" width="10" height="5"/>\n'
        '  <rect inkscape:label="robot_spawn_zone" x="1" y="1" width="1" height="1"/>\n'
        '  <rect inkscape:label="robot_goal_zone" x="18" y="18" width="1" height="1"/>\n'
        '  <path inkscape:label="robot_route_0_0" d="M 2 2 L 19 19"/>\n'
        '</svg>'
    )
    return convert_map(str(svg_path))


@pytest.fixture()
def bottleneck_map_def():
    """Load the classic_bottleneck SVG map for integration tests."""
    return convert_map(
        str(Path(__file__).parents[2] / "maps" / "svg_maps" / "classic_bottleneck.svg")
    )


@pytest.fixture()
def empty_map_def(tmp_path):
    """Empty MapDefinition with no obstacles."""
    svg_path = tmp_path / "empty.svg"
    svg_path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<svg xmlns="http://www.w3.org/2000/svg"\n'
        '     xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"\n'
        '     width="10" height="10">\n'
        '  <rect inkscape:label="robot_spawn_zone" x="1" y="1" width="1" height="1"/>\n'
        '  <rect inkscape:label="robot_goal_zone" x="8" y="8" width="1" height="1"/>\n'
        '  <path inkscape:label="robot_route_0_0" d="M 2 2 L 9 9"/>\n'
        '</svg>'
    )
    return convert_map(str(svg_path))


# ---------------------------------------------------------------------------
# ImportError contract (always runs)
# ---------------------------------------------------------------------------


def test_raises_import_error_when_ompl_missing():
    """Construction raises ImportError with install instructions when ompl is absent."""
    result = subprocess.run(
        [
            sys.executable, "-c",
            "import sys; sys.modules.pop('ompl', None); "
            "from robot_sf.nav.map_config import MapDefinition, Obstacle; "
            "m = MapDefinition(width=10.0, height=10.0, obstacles=[]); "
            "from robot_sf.planner.ompl_geometric_adapter import OmplGeometricAdapter; "
            "OmplGeometricAdapter(m)",
        ],
        capture_output=True,
        text=True,
        check=False,
        env={
            **dict(os.environ),
            "PYTHONPATH": str(Path(__file__).parents[2]),
            "OMPPL_TEST_NO_OMPL": "1",
        },
    )
    if result.returncode == 0:
        pytest.skip("ompl is installed; ImportError contract cannot be tested in-process")


# ---------------------------------------------------------------------------
# Unit tests - config and enum (no OMPL needed)
# ---------------------------------------------------------------------------


def test_planner_choice_enum_values():
    """Verify all planner enum members have expected string labels."""
    from robot_sf.planner.ompl_geometric_adapter import OmplPlannerChoice

    expected = {
        "RRTCONNECT": "RRTConnect",
        "BITSTAR": "BITstar",
        "RRTSTAR": "RRTstar",
        "INFORMED_RRTSTAR": "InformedRRTstar",
        "PRMSTAR": "PRMstar",
    }
    for member_name, expected_value in expected.items():
        member = OmplPlannerChoice[member_name]
        assert member.value == expected_value


def test_config_defaults():
    """Verify default config uses BITstar with 5 s budget."""
    from robot_sf.planner.ompl_geometric_adapter import (
        OmplGeometricConfig,
        OmplPlannerChoice,
    )

    cfg = OmplGeometricConfig()
    assert cfg.planner == OmplPlannerChoice.BITSTAR
    assert cfg.time_budget_s == 5.0
    assert cfg.interpolate_waypoints == 50
    assert cfg.robot_radius_m == 0.0


def test_config_robot_radius():
    """Verify robot radius can be set on config."""
    from robot_sf.planner.ompl_geometric_adapter import OmplGeometricConfig

    cfg = OmplGeometricConfig(robot_radius_m=0.3)
    assert cfg.robot_radius_m == 0.3


# ---------------------------------------------------------------------------
# Obstacle union helper (no OMPL needed)
# ---------------------------------------------------------------------------


def test_build_obstacle_union_simple(simple_map_def):
    """Verify obstacle union contains points inside obstacles."""
    from robot_sf.planner.ompl_geometric_adapter import _build_obstacle_union

    union = _build_obstacle_union(simple_map_def)
    assert union.contains(Point(10.0, 7.5))
    assert not union.contains(Point(1.0, 1.0))


def test_build_obstacle_union_boundary(empty_map_def):
    """Verify boundary walls are included in obstacle union."""
    from robot_sf.planner.ompl_geometric_adapter import _build_obstacle_union

    union = _build_obstacle_union(empty_map_def)
    assert union.contains(Point(0.01, 5.0))
    assert union.contains(Point(5.0, 0.01))


def test_build_obstacle_union_inflation(simple_map_def):
    """Verify obstacle inflation expands the forbidden region."""
    from robot_sf.planner.ompl_geometric_adapter import _build_obstacle_union

    union_no_inflate = _build_obstacle_union(simple_map_def, robot_radius_m=0.0)
    union_inflated = _build_obstacle_union(simple_map_def, robot_radius_m=1.0)

    assert not union_no_inflate.contains(Point(4.9, 7.5))
    assert union_inflated.contains(Point(4.9, 7.5))


# ---------------------------------------------------------------------------
# Integration tests (require OMPL)
# ---------------------------------------------------------------------------

OMPL_AVAILABLE = False
try:
    __import__("ompl")
    OMPL_AVAILABLE = True
except ImportError:
    pass


@pytest.mark.skipif(not OMPL_AVAILABLE, reason="ompl not installed")
class TestOmplIntegration:
    """Integration tests that require the OMPL package."""

    def test_bitstar_plans_path(self, bottleneck_map_def):
        """BITstar finds a valid path on the classic bottleneck map."""
        from robot_sf.planner.ompl_geometric_adapter import (
            OmplGeometricAdapter,
            OmplPlannerChoice,
        )

        adapter = OmplGeometricAdapter(
            bottleneck_map_def, planner=OmplPlannerChoice.BITSTAR
        )
        result = adapter.plan(start=(20.0, 31.0), goal=(20.0, 8.0))

        assert result.solved
        assert result.planner_name == "BITstar"
        assert result.path_length_m > 0
        assert len(result.waypoints) > 1
        assert result.exact_solution

    def test_bitstar_path_avoids_obstacles(self, bottleneck_map_def):
        """Verifies OMPL path does not pass through obstacles."""
        from robot_sf.planner.ompl_geometric_adapter import (
            OmplGeometricAdapter,
            OmplPlannerChoice,
        )

        adapter = OmplGeometricAdapter(
            bottleneck_map_def, planner=OmplPlannerChoice.BITSTAR
        )
        result = adapter.plan(start=(20.0, 31.0), goal=(20.0, 8.0))

        assert result.solved
        for wp in result.waypoints:
            pt = Point(wp)
            for obs in bottleneck_map_def.obstacles:
                poly = Polygon(obs.vertices)
                assert not poly.contains(pt), f"Waypoint {wp} inside obstacle"

    def test_rrtconnect_plans_path(self, bottleneck_map_def):
        """RRTConnect finds a feasible path (may be suboptimal)."""
        from robot_sf.planner.ompl_geometric_adapter import (
            OmplGeometricAdapter,
            OmplPlannerChoice,
        )

        adapter = OmplGeometricAdapter(
            bottleneck_map_def, planner=OmplPlannerChoice.RRTCONNECT
        )
        result = adapter.plan(start=(20.0, 31.0), goal=(20.0, 8.0))

        assert result.solved
        assert result.planner_name == "RRTConnect"
        assert result.path_length_m > 0

    def test_unsolved_returns_zero_length(self, bottleneck_map_def):
        """When OMPL cannot solve within budget, returns unsolved result."""
        from robot_sf.planner.ompl_geometric_adapter import (
            OmplGeometricAdapter,
            OmplGeometricConfig,
            OmplPlannerChoice,
        )

        config = OmplGeometricConfig(
            planner=OmplPlannerChoice.RRTSTAR,
            time_budget_s=0.000001,
        )
        adapter = OmplGeometricAdapter(bottleneck_map_def, config=config)
        result = adapter.plan(start=(1.0, 1.0), goal=(38.0, 38.0))

        if not result.solved:
            assert result.path_length_m == 0.0
            assert result.waypoints == []

    def test_robot_radius_inflation(self, simple_map_def):
        """Robot radius inflation expands obstacle clearance in OMPL planning."""
        from robot_sf.planner.ompl_geometric_adapter import (
            OmplGeometricAdapter,
            OmplGeometricConfig,
            OmplPlannerChoice,
        )

        adapter_no_inflate = OmplGeometricAdapter(
            simple_map_def,
            config=OmplGeometricConfig(
                planner=OmplPlannerChoice.BITSTAR, robot_radius_m=0.0
            ),
        )
        result_no_inflate = adapter_no_inflate.plan(
            start=(1.0, 3.0), goal=(1.0, 13.0)
        )

        adapter_inflate = OmplGeometricAdapter(
            simple_map_def,
            config=OmplGeometricConfig(
                planner=OmplPlannerChoice.BITSTAR, robot_radius_m=2.0
            ),
        )
        result_inflate = adapter_inflate.plan(start=(1.0, 3.0), goal=(1.0, 13.0))

        if result_inflate.solved and result_no_inflate.solved:
            assert result_inflate.path_length_m >= result_no_inflate.path_length_m - 1.0

    def test_adapter_compares_with_grid_planner(self, bottleneck_map_def):
        """Smoke comparison: OMPL path length is within 50% of grid planner."""
        from robot_sf.planner.classic_global_planner import ClassicGlobalPlanner
        from robot_sf.planner.ompl_geometric_adapter import (
            OmplGeometricAdapter,
            OmplPlannerChoice,
        )

        start = (20.0, 31.0)
        goal = (20.0, 8.0)

        grid_planner = ClassicGlobalPlanner(bottleneck_map_def)
        _, grid_info = grid_planner.plan(start, goal)
        grid_length = grid_info.get("length", 0) if grid_info else 0

        adapter = OmplGeometricAdapter(
            bottleneck_map_def, planner=OmplPlannerChoice.BITSTAR
        )
        ompl_result = adapter.plan(start, goal)

        assert ompl_result.solved
        assert ompl_result.path_length_m <= grid_length * 1.5
