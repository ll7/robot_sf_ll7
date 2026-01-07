"""Tests for OSM PBF background rendering and affine transforms.

Tests verify:
- Background PNG and metadata JSON creation
- Affine transform round-trip accuracy for pixel↔world mapping
- Save/load of affine transform metadata for editor reuse
"""

import json
from pathlib import Path

import pytest

from robot_sf.maps.osm_background_renderer import (
    load_affine_transform,
    pixel_to_world,
    render_osm_background,
    save_affine_transform,
    validate_affine_transform,
    world_to_pixel,
)


@pytest.fixture
def pbf_fixture() -> str:
    """Path to the test PBF fixture."""
    return "test_scenarios/osm_fixtures/sample_block.pbf"


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    """Temporary output directory for renderer artifacts."""
    return tmp_path / "osm_output"


class TestRendering:
    """Renderer smoke tests for PNG and metadata output."""

    def test_render_osm_background_creates_png(self, pbf_fixture: str, output_dir: Path) -> None:
        """Ensure PNG output is produced for visual verification workflows."""
        output_dir.mkdir(parents=True, exist_ok=True)

        result = render_osm_background(
            str(pbf_fixture), str(output_dir), pixels_per_meter=0.25, dpi=50
        )

        png_file = Path(result["png_path"])
        assert png_file.exists()
        assert png_file.stat().st_size > 0

    def test_render_osm_background_creates_affine_json(
        self, pbf_fixture: str, output_dir: Path
    ) -> None:
        """Verify JSON metadata includes affine_transform for editor alignment."""
        output_dir.mkdir(parents=True, exist_ok=True)

        _ = render_osm_background(str(pbf_fixture), str(output_dir), pixels_per_meter=0.25, dpi=50)

        json_file = output_dir / "affine_transform.json"
        assert json_file.exists()

        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)
        assert "affine_transform" in data
        assert "pixel_per_meter" in data["affine_transform"]
        assert "bounds_meters" in data["affine_transform"]
        assert "pixel_dimensions" in data["affine_transform"]

    def test_render_osm_background_returns_affine_data(
        self, pbf_fixture: str, output_dir: Path
    ) -> None:
        """Confirm render returns affine transform data for immediate use."""
        output_dir.mkdir(parents=True, exist_ok=True)

        result = render_osm_background(str(pbf_fixture), str(output_dir), pixels_per_meter=0.25)

        assert isinstance(result, dict)
        assert "png_path" in result
        assert "affine_transform" in result
        affine = result["affine_transform"]
        assert "pixel_per_meter" in affine
        assert "bounds_meters" in affine
        assert "pixel_dimensions" in affine
        assert "dpi" in affine


class TestAffineTransform:
    """Test coordinate transformation round-trip behavior."""

    def test_pixel_to_world_conversion(self) -> None:
        """Pixel→world conversion respects origin and bounds for overlay alignment."""
        affine = {
            "pixel_per_meter": 2.0,
            "bounds_meters": [0.0, 0.0, 100.0, 100.0],
            "origin": "upper",
        }

        world = pixel_to_world((0, 0), affine)
        assert world == (0.0, 100.0)

        world = pixel_to_world((200, 200), affine)
        assert world == (100.0, 0.0)

    def test_world_to_pixel_conversion(self) -> None:
        """World→pixel conversion preserves bounds origin semantics."""
        affine = {
            "pixel_per_meter": 2.0,
            "bounds_meters": [0.0, 0.0, 100.0, 100.0],
            "origin": "upper",
        }

        pixel = world_to_pixel((0.0, 100.0), affine)
        assert pixel == (0.0, 0.0)

        pixel = world_to_pixel((100.0, 0.0), affine)
        assert pixel == (200.0, 200.0)

    def test_validate_affine_transform_round_trip(self) -> None:
        """Round-trip validation guards against drift in affine metadata."""
        affine = {
            "pixel_per_meter": 2.0,
            "bounds_meters": [0.0, 0.0, 100.0, 100.0],
            "origin": "upper",
        }

        test_points = [(0, 0), (50, 100), (200, 200)]
        for pt in test_points:
            is_valid = validate_affine_transform(affine, point_pixel=pt, tolerance_pixels=1.0)
            assert is_valid, f"Round-trip validation failed for {pt}"

    def test_save_load_affine_transform(self, tmp_path: Path) -> None:
        """Ensure affine metadata stays stable across serialization."""
        affine_orig = {
            "pixel_per_meter": 1.5,
            "bounds_meters": [10.0, 20.0, 110.0, 120.0],
            "pixel_dimensions": [400, 400],
            "dpi": 100,
            "origin": "upper",
        }

        json_file = tmp_path / "affine.json"
        save_affine_transform(affine_orig, str(json_file))

        assert json_file.exists()

        affine_loaded = load_affine_transform(str(json_file))
        assert affine_loaded == affine_orig


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
