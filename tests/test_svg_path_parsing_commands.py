"""Unit tests for advanced SVG path command parsing."""

from __future__ import annotations

import pytest

from robot_sf.nav.svg_map_parser import SvgMapConverter


def test_parse_path_coordinates_supports_cubic_and_smooth_cubic() -> None:
    """Cubic commands C/S should parse into sampled waypoints with correct endpoint."""
    coords = SvgMapConverter._parse_path_coordinates("M 1 1 C 2 1 3 2 4 2 S 6 3 7 1")

    assert coords[0] == (1.0, 1.0)
    assert coords[-1] == pytest.approx((7.0, 1.0))
    assert len(coords) > 4


def test_parse_path_coordinates_supports_relative_cubic_and_smooth_cubic() -> None:
    """Relative cubic commands c/s should end at the expected absolute point."""
    coords = SvgMapConverter._parse_path_coordinates("M 1 1 c 1 0 2 1 3 1 s 2 1 3 -1")

    assert coords[0] == (1.0, 1.0)
    assert coords[-1] == pytest.approx((7.0, 1.0))
    assert len(coords) > 4


def test_parse_path_coordinates_supports_quadratic_and_smooth_quadratic() -> None:
    """Quadratic commands Q/T should parse and preserve the final endpoint."""
    coords = SvgMapConverter._parse_path_coordinates("M 1 1 Q 3 3 5 1 T 9 1")

    assert coords[0] == (1.0, 1.0)
    assert coords[-1] == pytest.approx((9.0, 1.0))
    assert len(coords) > 3


def test_parse_path_coordinates_supports_relative_quadratic_and_smooth_quadratic() -> None:
    """Relative quadratic commands q/t should parse and preserve the endpoint."""
    coords = SvgMapConverter._parse_path_coordinates("M 1 1 q 2 2 4 0 t 4 0")

    assert coords[0] == (1.0, 1.0)
    assert coords[-1] == pytest.approx((9.0, 1.0))
    assert len(coords) > 3


def test_parse_path_coordinates_supports_arc_absolute_and_relative() -> None:
    """Arc commands A/a should generate sampled points and end at expected coordinates."""
    abs_coords = SvgMapConverter._parse_path_coordinates("M 2 2 A 3 3 0 0 1 8 2")
    rel_coords = SvgMapConverter._parse_path_coordinates("M 2 2 a 3 3 0 0 1 6 0")

    assert abs_coords[0] == (2.0, 2.0)
    assert abs_coords[-1] == pytest.approx((8.0, 2.0))
    assert any(abs(y - 2.0) > 1e-6 for _x, y in abs_coords[1:-1])
    assert len(abs_coords) > 2

    assert rel_coords[0] == (2.0, 2.0)
    assert rel_coords[-1] == pytest.approx((8.0, 2.0))
    assert len(rel_coords) > 2


def test_parse_path_coordinates_supports_close_path() -> None:
    """Close-path commands should append the starting waypoint."""
    coords = SvgMapConverter._parse_path_coordinates("M 0 0 l 1 0 0 1 -1 0 z")

    assert coords[0] == (0.0, 0.0)
    assert coords[-1] == pytest.approx((0.0, 0.0))
    assert len(coords) >= 5
