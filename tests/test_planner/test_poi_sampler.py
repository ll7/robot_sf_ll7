"""Tests for POI sampling strategies (US3)."""
# ruff: noqa: D103

from pathlib import Path

import pytest

from robot_sf.nav.svg_map_parser import convert_map
from robot_sf.planner.poi_sampler import POISampler


def _make_poi_map(tmp_path: Path) -> str:
    svg = tmp_path / "poi_map.svg"
    svg.write_text(
        """
<svg xmlns="http://www.w3.org/2000/svg" xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" width="10" height="10">
  <rect inkscape:label="robot_spawn_zone" x="0.5" y="0.5" width="1" height="1" />
  <rect inkscape:label="robot_goal_zone" x="8.5" y="8.5" width="1" height="1" />
  <path inkscape:label="robot_route_0_0" d="M 0.5 0.5 L 9 9" />
  <circle class="poi" id="poi_west" cx="1.0" cy="5.0" r="0.2" inkscape:label="west" />
  <circle class="poi" id="poi_center" cx="5.0" cy="5.0" r="0.2" inkscape:label="center" />
  <circle class="poi" id="poi_east" cx="9.0" cy="5.0" r="0.2" inkscape:label="east" />
</svg>
        """.strip()
    )
    return str(svg)


def test_random_sampling_reproducible(tmp_path):
    map_def = convert_map(_make_poi_map(tmp_path))
    sampler = POISampler(map_def, seed=123)

    first = sampler.sample(count=2, strategy="random")
    second = sampler.sample(count=2, strategy="random")

    # Reset with the same seed to confirm determinism
    sampler2 = POISampler(map_def, seed=123)
    replay = sampler2.sample(count=2, strategy="random")

    assert first != second  # without replacement per call
    assert first == replay


def test_nearest_and_farthest(tmp_path):
    map_def = convert_map(_make_poi_map(tmp_path))
    sampler = POISampler(map_def, seed=42)
    start = (0.0, 5.0)

    nearest = sampler.sample(count=2, strategy="nearest", start=start)
    farthest = sampler.sample(count=2, strategy="farthest", start=start)

    nearest_labels = {
        map_def.poi_labels[pid]: pos
        for pid, pos in zip(map_def.poi_labels, map_def.poi_positions, strict=False)
    }
    # nearest should include west then center
    assert nearest[0] == nearest_labels["west"]
    assert nearest[1] == nearest_labels["center"]
    # farthest should include east then center
    assert farthest[0] == nearest_labels["east"]
    assert farthest[1] == nearest_labels["center"]


def test_sampler_requires_pois(tmp_path):
    svg = tmp_path / "no_poi.svg"
    svg.write_text(
        """
<svg xmlns="http://www.w3.org/2000/svg" xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" width="2" height="2">
  <rect inkscape:label="robot_spawn_zone" x="0.5" y="0.5" width="0.5" height="0.5" />
  <rect inkscape:label="robot_goal_zone" x="1.0" y="1.0" width="0.5" height="0.5" />
  <path inkscape:label="robot_route_0_0" d="M 0.5 0.5 L 1.5 1.5" />
</svg>
        """.strip()
    )
    map_def = convert_map(str(svg))

    with pytest.raises(ValueError):
        POISampler(map_def)
