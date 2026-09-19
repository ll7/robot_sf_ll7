"""Fail-closed regressions for non-finite corrected SVG transform results."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from robot_sf.nav.svg_map_parser import SvgMapConverter

if TYPE_CHECKING:
    from pathlib import Path


SVG_HEADER = (
    '<svg xmlns="http://www.w3.org/2000/svg" '
    'xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" '
    'width="40" height="40" viewBox="0 0 40 40">'
)


def _write_svg(
    tmp_path: Path,
    name: str,
    inner: str,
    *,
    view_box: str = "0 0 40 40",
) -> str:
    """Write a minimal SVG fixture and return its path."""
    path = tmp_path / name
    header = SVG_HEADER.replace('viewBox="0 0 40 40"', f'viewBox="{view_box}"')
    path.write_text(header + inner + "</svg>", encoding="utf-8")
    return str(path)


def test_translate_list_rejects_nonfinite_accumulation() -> None:
    """A finite translate term must not make a list overflow silently."""
    with pytest.raises(ValueError, match="non-finite"):
        SvgMapConverter._parse_translate_offset(
            "translate(1e308) translate(1e308)",
            source="test",
        )


def test_corrected_rejects_nonfinite_nested_ancestor_translation(tmp_path: Path) -> None:
    """Nested finite ancestor translations must fail when their sum overflows."""
    svg = _write_svg(
        tmp_path,
        "nested_overflow.svg",
        '<g transform="translate(1e308)">'
        '<g transform="translate(1e308)">'
        '<path d="M 0 0 L 1 1" />'
        "</g></g>",
    )

    with pytest.raises(ValueError, match="non-finite"):
        SvgMapConverter(svg, geometry_contract="corrected")


def test_corrected_rejects_nonfinite_shifted_path_coordinate(tmp_path: Path) -> None:
    """A finite path coordinate must not overflow when an ancestor shift is applied."""
    svg = _write_svg(
        tmp_path,
        "path_coordinate_overflow.svg",
        '<g transform="translate(1e308)"><path d="M 1e308 0 L 1e308 1" /></g>',
    )

    with pytest.raises(ValueError, match="non-finite"):
        SvgMapConverter(svg, geometry_contract="corrected")


@pytest.mark.parametrize(
    ("element", "name"),
    [
        ('<rect x="1e308" y="0" width="1" height="1" />', "rect"),
        ('<circle cx="1e308" cy="0" r="1" />', "circle"),
    ],
)
def test_corrected_rejects_nonfinite_shifted_shape_coordinate(
    tmp_path: Path, element: str, name: str
) -> None:
    """Rectangles and circles also reject overflowing corrected shifts."""
    svg = _write_svg(
        tmp_path,
        f"{name}_coordinate_overflow.svg",
        f'<g transform="translate(1e308)">{element}</g>',
    )

    with pytest.raises(ValueError, match="non-finite"):
        SvgMapConverter(svg, geometry_contract="corrected")


@pytest.mark.parametrize(
    ("element", "name"),
    [
        ('<path d="M 1e308 0 L 1e308 1" />', "path"),
        ('<rect x="1e308" y="0" width="1" height="1" />', "rect"),
        ('<circle cx="1e308" cy="0" r="1" />', "circle"),
    ],
)
def test_corrected_rejects_nonfinite_viewbox_normalization(
    tmp_path: Path, element: str, name: str
) -> None:
    """Corrected viewBox normalization rejects finite coordinates that overflow."""
    svg = _write_svg(
        tmp_path,
        f"{name}_viewbox_overflow.svg",
        element,
        view_box="-1e308 0 40 40",
    )

    with pytest.raises(ValueError, match="non-finite corrected SVG coordinate"):
        SvgMapConverter(svg, geometry_contract="corrected")


def test_corrected_viewbox_normalization_preserves_finite_coordinates(tmp_path: Path) -> None:
    """Corrected viewBox normalization shifts finite geometry without changing its contract."""
    svg = _write_svg(
        tmp_path,
        "finite_viewbox.svg",
        '<path inkscape:label="robot_route_0_0" d="M 10 20 L 11 21" />',
        view_box="10 20 40 40",
    )

    converter = SvgMapConverter(svg, geometry_contract="corrected")

    assert converter.path_info[0].coordinates == ((0.0, 0.0), (1.0, 1.0))
    assert converter.get_map_definition().svg_geometry_contract == "corrected"


def test_legacy_ignores_overflowing_ancestor_translation(tmp_path: Path) -> None:
    """Legacy mode continues to ignore ancestor transforms, including huge values."""
    svg = _write_svg(
        tmp_path,
        "legacy_overflow.svg",
        '<g transform="translate(1e308)">'
        '<g transform="translate(1e308)">'
        '<path d="M 0 0 L 1 1" />'
        "</g></g>",
    )

    map_definition = SvgMapConverter(svg, geometry_contract="legacy").get_map_definition()

    assert map_definition.svg_geometry_contract == "legacy"
    assert map_definition.ped_routes == []
