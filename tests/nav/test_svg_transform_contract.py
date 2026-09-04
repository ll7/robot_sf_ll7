"""Geometry-contract tests for ancestor SVG transforms (issue #8314).

`SvgMapConverter` historically ignored ancestor `transform` attributes, silently
displacing every transformed element. The corrected geometry contract applies
nested `translate(...)` transforms and fails closed on anything else, while the
legacy contract reproduces the historical transform-ignoring coordinates exactly.
"""

from __future__ import annotations

import pickle
import re
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import pytest

from robot_sf.nav.map_config import MapDefinition
from robot_sf.nav.svg_map_parser import SvgMapConverter, convert_map

ROOT = Path(__file__).resolve().parents[2]
SVG_DIR = ROOT / "maps" / "svg_maps"
SVG_NS = {"svg": "http://www.w3.org/2000/svg"}
LABEL = "{http://www.inkscape.org/namespaces/inkscape}label"

SVG_HEADER = (
    '<svg xmlns="http://www.w3.org/2000/svg" '
    'xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" '
    'width="40" height="40" viewBox="0 0 40 40">'
)
SVG_FOOTER = "</svg>"
BASE_ELEMENTS = """
  <rect inkscape:label="robot_spawn_zone" x="1" y="4" width="1.5" height="1.5" />
  <rect inkscape:label="robot_goal_zone" x="17" y="4" width="1.5" height="1.5" />
  <rect inkscape:label="obstacle" x="9" y="2" width="2" height="6" />
  <path inkscape:label="robot_route_0_0" d="M 1 4.5 L 18 4.5" />
"""


def _write_svg(tmp_path: Path, name: str, inner: str) -> str:
    """Write a synthetic SVG map and return its path."""
    path = tmp_path / name
    path.write_text(SVG_HEADER + BASE_ELEMENTS + inner + SVG_FOOTER, encoding="utf-8")
    return str(path)


def _nested_group(inner: str) -> str:
    """Wrap elements in two nested translated groups (total offset 4, 6)."""
    return f'<g transform="translate(1, 2)"><g transform="translate(3,4)">{inner}</g></g>'


def _zone_bounds(zone) -> tuple[float, float, float, float]:
    """Return (min_x, min_y, max_x, max_y) for a zone tuple."""
    xs = [p[0] for p in zone]
    ys = [p[1] for p in zone]
    return min(xs), min(ys), max(xs), max(ys)


def test_nested_translate_rect_exact(tmp_path: Path):
    """Nested translations accumulate exactly for rectangles."""
    svg = _write_svg(
        tmp_path,
        "nested_rect.svg",
        _nested_group(
            '<rect inkscape:label="ped_spawn_zone" x="10" y="10" width="4" height="4" />'
        ),
    )
    md = SvgMapConverter(svg, geometry_contract="corrected").get_map_definition()
    assert _zone_bounds(md.ped_spawn_zones[0]) == pytest.approx((14.0, 16.0, 18.0, 20.0))


def test_nested_translate_path_exact(tmp_path: Path):
    """Nested translations accumulate exactly for path waypoints."""
    svg = _write_svg(
        tmp_path,
        "nested_path.svg",
        _nested_group('<path inkscape:label="ped_route_0_0" d="M 0 0 L 10 0" />')
        + '<rect inkscape:label="ped_spawn_zone" x="4" y="16" width="4" height="4" />'
        + '<rect inkscape:label="ped_goal_zone" x="14" y="16" width="4" height="4" />',
    )
    converter = SvgMapConverter(svg, geometry_contract="corrected")
    route = next(p for p in converter.path_info if p.label == "ped_route_0_0")
    flat = [c for point in route.coordinates for c in point]
    assert flat == pytest.approx([4.0, 6.0, 14.0, 6.0])


def test_nested_translate_circle_exact(tmp_path: Path):
    """Nested translations accumulate exactly for circle centers."""
    svg = _write_svg(
        tmp_path,
        "nested_circle.svg",
        _nested_group(
            '<circle inkscape:label="single_ped_a_start" cx="10" cy="10" r="0.5" />'
            '<circle inkscape:label="single_ped_a_goal" cx="20" cy="20" r="0.5" />'
        ),
    )
    converter = SvgMapConverter(svg, geometry_contract="corrected")
    starts = [c for c in converter.circle_info if c.label == "single_ped_a_start"]
    assert [(c.cx, c.cy) for c in starts] == pytest.approx([(14.0, 16.0)])


def test_corrected_parser_exercises_all_structured_debug_counts(tmp_path: Path):
    """The corrected parser visits every element-count logging branch."""
    svg = _write_svg(
        tmp_path,
        "structured_debug_counts.svg",
        _nested_group(
            '<path inkscape:label="ped_route_0_0" d="M 0 0 L 10 0" />'
            '<rect inkscape:label="ped_spawn_zone" x="10" y="10" width="4" height="4" />'
            '<circle inkscape:label="single_ped_a_start" cx="10" cy="10" r="0.5" />'
        ),
    )
    converter = SvgMapConverter(svg, geometry_contract="corrected")

    assert len(converter.path_info) == 2
    assert len(converter.rect_info) == 4
    assert len(converter.circle_info) == 1


def test_element_own_transform_applies(tmp_path: Path):
    """A transform directly on the element itself also applies."""
    svg = _write_svg(
        tmp_path,
        "own_transform.svg",
        '<rect inkscape:label="ped_spawn_zone" x="10" y="10" width="4" height="4" '
        'transform="translate(-2,-3)" />',
    )
    md = SvgMapConverter(svg, geometry_contract="corrected").get_map_definition()
    assert _zone_bounds(md.ped_spawn_zones[0]) == pytest.approx((8.0, 7.0, 12.0, 11.0))


def test_legacy_ignores_nested_translate(tmp_path: Path):
    """The legacy contract reproduces transform-ignoring coordinates exactly."""
    svg = _write_svg(
        tmp_path,
        "legacy_nested.svg",
        _nested_group(
            '<rect inkscape:label="ped_spawn_zone" x="10" y="10" width="4" height="4" />'
        ),
    )
    md = SvgMapConverter(svg, geometry_contract="legacy").get_map_definition()
    assert _zone_bounds(md.ped_spawn_zones[0]) == pytest.approx((10.0, 10.0, 14.0, 14.0))


def test_default_contract_is_legacy(tmp_path: Path):
    """Omitting the selector preserves historical behavior."""
    svg = _write_svg(
        tmp_path,
        "default_legacy.svg",
        _nested_group(
            '<rect inkscape:label="ped_spawn_zone" x="10" y="10" width="4" height="4" />'
        ),
    )
    md = SvgMapConverter(svg).get_map_definition()
    assert md.svg_geometry_contract == "legacy"
    assert _zone_bounds(md.ped_spawn_zones[0]) == pytest.approx((10.0, 10.0, 14.0, 14.0))


def test_legacy_pickle_without_geometry_contract_defaults_to_legacy(tmp_path: Path):
    """Maps pickled before the geometry contract field remain legacy maps."""

    svg = _write_svg(tmp_path, "old_pickle.svg", "")
    map_definition = SvgMapConverter(svg).get_map_definition()
    del map_definition.svg_geometry_contract

    restored = pickle.loads(pickle.dumps(map_definition))

    assert restored.svg_geometry_contract == "legacy"


def test_unknown_contract_rejected(tmp_path: Path):
    """Unknown geometry contract names fail closed."""
    svg = _write_svg(tmp_path, "unknown_contract.svg", "")
    with pytest.raises(ValueError, match="geometry_contract"):
        SvgMapConverter(svg, geometry_contract="translate-everything")


def test_map_definition_rejects_unknown_contract(tmp_path: Path):
    """Map definitions reject unsupported geometry provenance values directly."""

    svg = _write_svg(tmp_path, "unknown_map_contract.svg", "")
    map_definition = SvgMapConverter(svg).get_map_definition()

    with pytest.raises(ValueError, match="Unknown svg_geometry_contract"):
        replace(map_definition, svg_geometry_contract="translate-everything")


@pytest.mark.parametrize(
    "transform",
    [
        "scale(2)",
        "rotate(45)",
        "skewX(10)",
        "skewY(10)",
        "matrix(1,0,0,1,5,5)",
        "translate(foo)",
        "translate(1,foo)",
        "translate(foo,2)",
        "translate(1px,2)",
        "translate(1;2)",
        "translate(1,2,3)",
        "translate()",
        "translate(1,2) scale(3)",
        "translate(1 2",
        ",",
        ",translate(1,2)",
        "translate(1,2),",
        "translate(1,2),,translate(3,4)",
        "translate(1,2)translate(3,4)",
    ],
)
def test_unsupported_transform_fails_closed(tmp_path: Path, transform: str):
    """Every unsupported or malformed transform errors instead of being ignored."""
    svg = _write_svg(
        tmp_path,
        "rejected.svg",
        f'<g transform="{transform}">'
        '<rect inkscape:label="ped_spawn_zone" x="10" y="10" width="4" height="4" />'
        "</g>",
    )
    with pytest.raises(ValueError, match="transform"):
        SvgMapConverter(svg, geometry_contract="corrected")


def test_unsupported_transform_on_self_fails_closed(tmp_path: Path):
    """Unsupported transforms directly on the element also fail closed."""
    svg = _write_svg(
        tmp_path,
        "rejected_self.svg",
        '<rect inkscape:label="ped_spawn_zone" x="10" y="10" width="4" height="4" '
        'transform="rotate(90)" />',
    )
    with pytest.raises(ValueError, match="transform"):
        SvgMapConverter(svg, geometry_contract="corrected")


def test_legacy_tolerates_unsupported_transform(tmp_path: Path):
    """Legacy mode keeps byte-exact historical behavior, even for scale."""
    svg = _write_svg(
        tmp_path,
        "legacy_scale.svg",
        '<g transform="scale(2)">'
        '<rect inkscape:label="ped_spawn_zone" x="10" y="10" width="4" height="4" />'
        "</g>",
    )
    md = SvgMapConverter(svg, geometry_contract="legacy").get_map_definition()
    assert _zone_bounds(md.ped_spawn_zones[0]) == pytest.approx((10.0, 10.0, 14.0, 14.0))


def _independent_expected_zones(svg_path: Path) -> dict[str, tuple[float, float, float, float]]:
    """Recompute authored zone bounds straight from the SVG file, bypassing the converter."""
    root = ET.parse(str(svg_path)).getroot()
    expected: dict[str, tuple[float, float, float, float]] = {}

    def visit(element: ET.Element, dx: float, dy: float) -> None:
        transform = element.attrib.get("transform", "")
        for match in re.finditer(r"translate\(\s*([^)]+)\)", transform):
            numbers = [float(n) for n in re.findall(r"[+-]?(?:\d+\.\d+|\d+|\.\d+)", match.group(1))]
            dx += numbers[0]
            dy += numbers[1] if len(numbers) > 1 else 0.0
        if element.tag.endswith("rect"):
            label = element.attrib.get(LABEL)
            if label in {"ped_spawn_zone", "ped_goal_zone", "robot_goal_zone"}:
                x = float(element.attrib["x"]) + dx
                y = float(element.attrib["y"]) + dy
                w = float(element.attrib["width"])
                h = float(element.attrib["height"])
                expected[label] = (x, y, x + w, y + h)
        for child in element:
            visit(child, dx, dy)

    visit(root, 0.0, 0.0)
    return expected


TRANSFORMED_MAPS = [
    "classic_bottleneck.svg",
    "classic_bottleneck_medium.svg",
    "classic_bottleneck_high.svg",
    "classic_t_intersection.svg",
    "planner_test_simple.svg",
]


@pytest.mark.parametrize("map_name", TRANSFORMED_MAPS)
def test_corrected_matches_authored_geometry(map_name: str):
    """Corrected coordinates equal the authored file geometry for every transformed map."""
    md = SvgMapConverter(
        str(SVG_DIR / map_name), geometry_contract="corrected"
    ).get_map_definition()
    assert md.svg_geometry_contract == "corrected"
    expected = _independent_expected_zones(SVG_DIR / map_name)
    zones = {
        "ped_spawn_zone": md.ped_spawn_zones,
        "ped_goal_zone": md.ped_goal_zones,
        "robot_goal_zone": md.robot_goal_zones,
    }
    assert expected, f"no labeled zones recomputed for {map_name}"
    for label, bounds in expected.items():
        assert _zone_bounds(zones[label][0]) == pytest.approx(bounds)


@pytest.mark.parametrize("map_name", TRANSFORMED_MAPS)
def test_legacy_reproduces_historical_coordinates(map_name: str):
    """Legacy coordinates equal raw attributes, ignoring ancestor transforms."""
    root = ET.parse(str(SVG_DIR / map_name)).getroot()
    raw: dict[str, tuple[float, float, float, float]] = {}
    for rect in root.findall(".//svg:rect", SVG_NS):
        label = rect.attrib.get(LABEL)
        if label in {"ped_spawn_zone", "ped_goal_zone"}:
            x, y = float(rect.attrib["x"]), float(rect.attrib["y"])
            raw[label] = (x, y, x + float(rect.attrib["width"]), y + float(rect.attrib["height"]))
    md = SvgMapConverter(str(SVG_DIR / map_name), geometry_contract="legacy").get_map_definition()
    assert md.svg_geometry_contract == "legacy"
    zones = {"ped_spawn_zone": md.ped_spawn_zones, "ped_goal_zone": md.ped_goal_zones}
    for label, bounds in raw.items():
        assert _zone_bounds(zones[label][0]) == pytest.approx(bounds)


def test_corrected_bottleneck_removes_wall_and_spawn_overlaps():
    """Corrected bottleneck geometry matches the authored zones without overlaps."""
    md = SvgMapConverter(
        str(SVG_DIR / "classic_bottleneck.svg"), geometry_contract="corrected"
    ).get_map_definition()
    assert _zone_bounds(md.ped_spawn_zones[0]) == pytest.approx((18.0, 1.6348353, 22.0, 5.6348353))
    assert _zone_bounds(md.ped_goal_zones[0]) == pytest.approx(
        (17.801584, 33.7699133, 21.801584, 37.7699133)
    )
    spawn = _zone_bounds(md.ped_spawn_zones[0])
    robot_goal = _zone_bounds(md.robot_goal_zones[0])
    assert spawn[3] <= robot_goal[1], "pedestrian spawn still overlaps the robot goal"
    goal = _zone_bounds(md.ped_goal_zones[0])
    assert goal[3] <= 39.0, "pedestrian goal still reaches the northern boundary wall"
    assert goal[3] <= 40.0, "pedestrian goal still leaves the map viewBox"


def test_legacy_bottleneck_keeps_historical_overlaps():
    """Legacy bottleneck geometry keeps the reported as-run overlaps exactly."""
    md = SvgMapConverter(
        str(SVG_DIR / "classic_bottleneck.svg"), geometry_contract="legacy"
    ).get_map_definition()
    assert _zone_bounds(md.ped_spawn_zones[0]) == pytest.approx((18.0, 6.0, 22.0, 10.0))
    assert _zone_bounds(md.ped_goal_zones[0]) == pytest.approx(
        (17.801584, 38.135078, 21.801584, 42.135078)
    )


@pytest.mark.parametrize("geometry_contract", ["legacy", "corrected"])
def test_low_tier_bottleneck_authors_no_pedestrians(geometry_contract: str):
    """Transform support must not add pedestrians to the low-tier cell (diss#2144)."""
    md = SvgMapConverter(
        str(SVG_DIR / "classic_bottleneck.svg"), geometry_contract=geometry_contract
    ).get_map_definition()
    assert md.single_pedestrians == []


def test_convert_map_contract_passthrough(tmp_path: Path):
    """convert_map exposes the same geometry-contract selector."""
    svg = _write_svg(
        tmp_path,
        "passthrough.svg",
        _nested_group(
            '<rect inkscape:label="ped_spawn_zone" x="10" y="10" width="4" height="4" />'
        ),
    )
    legacy = convert_map(svg)
    assert isinstance(legacy, MapDefinition)
    assert legacy.svg_geometry_contract == "legacy"
    corrected = convert_map(svg, geometry_contract="corrected")
    assert isinstance(corrected, MapDefinition)
    assert corrected.svg_geometry_contract == "corrected"
    assert _zone_bounds(corrected.ped_spawn_zones[0]) == pytest.approx((14.0, 16.0, 18.0, 20.0))
