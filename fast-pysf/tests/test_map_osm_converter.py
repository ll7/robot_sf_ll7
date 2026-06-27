"""Regression tests for OSM SVG viewBox parsing in ``map_osm_converter``.

Covers issue #3708: comma-delimited (and mixed comma/whitespace) ``viewBox``
attributes from OSM SVG exports must parse without raising ``ValueError``.
"""

import xml.etree.ElementTree as ET

import pytest
from pysocialforce.map_osm_converter import (
    add_scale_bar_to_root,
    extract_buildings_as_obstacle,
    parse_viewbox,
)


@pytest.mark.parametrize(
    "viewbox_str",
    [
        "0 0 488.48 458.33",  # whitespace separated
        "0,0,488.48,458.33",  # comma separated (issue #3708)
        "0, 0, 488.48, 458.33",  # mixed comma + whitespace
        "  0 0 488.48 458.33  ",  # surrounding whitespace
    ],
)
def test_parse_viewbox_separator_variants(viewbox_str):
    """All valid SVG viewBox separator styles parse to the same floats."""
    assert parse_viewbox(viewbox_str) == [0.0, 0.0, 488.48, 458.33]


def _building_svg(viewbox_str: str) -> str:
    """Minimal OSM-style SVG with one building path and a given viewBox."""
    color = "rgb(85.098039%,81.568627%,78.823529%)"
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{viewbox_str}">'
        f'<path style="fill:{color};" d="M 1 1 L 2 2 Z"/>'
        "</svg>"
    )


@pytest.mark.parametrize(
    "viewbox_str",
    ["0 0 488.48 458.33", "0,0,488.48,458.33", "0, 0, 488.48, 458.33"],
)
def test_extract_buildings_handles_comma_viewbox(tmp_path, viewbox_str):
    """extract_buildings_as_obstacle parses comma/whitespace viewBoxes alike."""
    svg_path = tmp_path / "map.svg"
    svg_path.write_text(_building_svg(viewbox_str))

    new_root = extract_buildings_as_obstacle(str(svg_path))

    # viewBox is scaled by the composite factor; verify it parsed and re-emitted
    # four numeric components rather than raising on float conversion.
    assert "viewBox" in new_root.attrib
    assert len(parse_viewbox(new_root.attrib["viewBox"])) == 4


def test_add_scale_bar_handles_comma_viewbox():
    """add_scale_bar_to_root reads a comma-separated viewBox without error."""
    root = ET.Element("svg", attrib={"viewBox": "0,0,300,200"})
    add_scale_bar_to_root(root, line_length=100)
    # Scale bar lines were added across the parsed image width (300).
    assert any(child.tag == "line" for child in root)
