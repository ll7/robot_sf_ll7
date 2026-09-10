"""Optional-extra public export contract for the GeoJSON map builder."""

from __future__ import annotations

import importlib
import inspect

import pytest

GEOJSON_MAP_BUILDER_ALL = [
    "build_parser",
    "geojson_to_map_definition",
    "geojson_to_map_structure",
    "load_geojson",
    "main",
    "write_segment_map",
]

UNEXPORTED_NAMES = ["_ROLE_KEYS", "_extract_zones", "argparse"]


def _load_geojson_module():
    """Import the optional module only when the optional lane is running."""
    return importlib.import_module("robot_sf.nav.geojson_map_builder")


def test_geojson_module_declares_the_reviewed_export_surface() -> None:
    """Verify the exact public export list for the optional GeoJSON module."""
    module = _load_geojson_module()
    assert module.__all__ == GEOJSON_MAP_BUILDER_ALL
    assert set(module.__all__) <= set(dir(module))


def test_geojson_module_all_names_resolve_on_pre_change_paths() -> None:
    """Verify exported GeoJSON names retain their reviewed module identity."""
    module = _load_geojson_module()
    expected_module = "robot_sf.nav.geojson_map_builder"

    for name in module.__all__:
        export = getattr(module, name)
        if inspect.isclass(export) or inspect.isfunction(export):
            assert export.__module__ == expected_module
            assert export.__qualname__ == name
        else:
            assert export is not None


@pytest.mark.parametrize("name", UNEXPORTED_NAMES)
def test_geojson_module_keeps_private_and_foreign_names_unexported(name: str) -> None:
    """Verify private, stale, or foreign names stay out of the optional public API."""
    module = _load_geojson_module()
    assert name not in module.__all__
