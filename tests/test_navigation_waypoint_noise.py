"""Tests for waypoint-noise behavior in route sampling."""

from __future__ import annotations

import itertools

from robot_sf.nav.navigation import NavigationSettings, sample_route
from robot_sf.nav.svg_map_parser import convert_map


def _make_noise_test_map(tmp_path):
    svg = tmp_path / "waypoint_noise_map.svg"
    svg.write_text(
        """
<svg xmlns="http://www.w3.org/2000/svg" xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" width="12" height="3">
  <rect inkscape:label="robot_spawn_zone" x="0.0" y="0.0" width="1.0" height="1.0" />
  <rect inkscape:label="robot_goal_zone" x="10.0" y="0.0" width="1.0" height="1.0" />
  <path inkscape:label="robot_route_0_0" d="M 1.0 0.5 L 5.0 0.5 L 10.0 0.5" />
</svg>
        """.strip(),
        encoding="utf-8",
    )
    return convert_map(str(svg))


def test_sample_route_default_noise_disabled(tmp_path, monkeypatch) -> None:
    """Route sampling should keep waypoints unchanged by default."""
    map_def = _make_noise_test_map(tmp_path)

    monkeypatch.setattr("robot_sf.nav.navigation.sample", lambda seq, k: [seq[0]])

    sampled_points = iter(((0.5, 0.5), (10.5, 0.5)))
    monkeypatch.setattr(
        "robot_sf.nav.navigation.sample_zone",
        lambda zone, count, obstacle_polygons=None: [next(sampled_points)],
    )

    route = sample_route(map_def, spawn_id=0)
    assert route[0] == (0.5, 0.5)
    assert route[-1] == (10.5, 0.5)
    assert route[1:-1] == [(1.0, 0.5), (5.0, 0.5), (10.0, 0.5)]


def test_sample_route_applies_noise_only_to_intermediate_waypoints(
    tmp_path,
    monkeypatch,
) -> None:
    """Configured waypoint noise should not change sampled spawn/goal points."""
    map_def = _make_noise_test_map(tmp_path)
    map_def._navigation_settings = NavigationSettings(
        waypoint_noise_enabled=True,
        waypoint_noise_std=0.5,
    )

    monkeypatch.setattr("robot_sf.nav.navigation.sample", lambda seq, k: [seq[0]])

    sampled_points = iter(((0.5, 0.5), (10.5, 0.5)))
    monkeypatch.setattr(
        "robot_sf.nav.navigation.sample_zone",
        lambda zone, count, obstacle_polygons=None: [next(sampled_points)],
    )

    # 3 waypoints x (x,y) => 6 normal samples
    noise_values = itertools.cycle([0.1, -0.1, 0.2, -0.2, 0.3, -0.3])

    def _mock_normal(mean, std, size=None):
        del mean, std
        assert size == (3, 2)
        return [[next(noise_values), next(noise_values)] for _ in range(size[0])]

    monkeypatch.setattr("robot_sf.nav.navigation.np.random.normal", _mock_normal)

    route = sample_route(map_def, spawn_id=0)
    assert route[0] == (0.5, 0.5)
    assert route[-1] == (10.5, 0.5)
    assert route[1:-1] == [(1.1, 0.4), (5.2, 0.3), (10.3, 0.2)]
