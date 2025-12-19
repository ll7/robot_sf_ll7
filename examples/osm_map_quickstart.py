#!/usr/bin/env python3
"""OSM PBF to MapDefinition quickstart example.

End-to-end demonstration of the OSM-based map extraction pipeline:
1. Load OSM PBF file
2. Convert to MapDefinition with allowed_areas
3. Render PNG background with affine transform
4. Display results

Usage:
    uv run python examples/osm_map_quickstart.py
"""

import logging
from pathlib import Path

from robot_sf.maps.osm_background_renderer import render_osm_background
from robot_sf.nav.osm_map_builder import OSMTagFilters, osm_to_map_definition

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Run OSM to MapDefinition conversion and rendering pipeline."""
    # Paths
    pbf_file = "test_scenarios/osm_fixtures/sample_block.pbf"
    output_dir = "output/maps/osm_demo"

    pbf_path = Path(pbf_file)
    if not pbf_path.exists():
        logger.error(f"PBF file not found: {pbf_file}")
        logger.info("Download a sample from: https://extract.bbbike.org/")
        return

    logger.info(f"Starting OSM → MapDefinition pipeline with {pbf_file}")

    # Configure tag filters
    tag_filters = OSMTagFilters()
    logger.info(f"Driveable highways: {tag_filters.driveable_highways}")
    logger.info(f"Obstacle tags: {tag_filters.obstacle_tags}")

    # Convert PBF to MapDefinition
    logger.info("Converting PBF to MapDefinition...")
    map_def = osm_to_map_definition(
        pbf_file=pbf_file,
        line_buffer_m=1.5,
        tag_filters=tag_filters,
    )

    logger.info("✅ MapDefinition created:")
    logger.info(f"   Width: {map_def.width:.2f}m")
    logger.info(f"   Height: {map_def.height:.2f}m")
    logger.info(f"   Obstacles: {len(map_def.obstacles)}")
    logger.info(f"   Allowed areas: {len(map_def.allowed_areas) if map_def.allowed_areas else 0}")

    # Render PNG background
    logger.info("Rendering PNG background...")
    affine_data = render_osm_background(
        pbf_file=pbf_file,
        output_dir=output_dir,
        pixels_per_meter=0.5,
        dpi=100,
    )

    logger.info("✅ Background rendered:")
    logger.info(f"   Pixel per meter: {affine_data['pixel_per_meter']}")
    logger.info(f"   Pixel dimensions: {affine_data['pixel_dimensions']}")
    logger.info(f"   Bounds (meters): {affine_data['bounds_meters']}")

    # Verify outputs
    output_path = Path(output_dir)
    png_file = output_path / "background.png"
    json_file = output_path / "affine_transform.json"

    if png_file.exists():
        size_kb = png_file.stat().st_size / 1024
        logger.info(f"✅ PNG saved: {png_file} ({size_kb:.1f} KB)")
    else:
        logger.warning(f"❌ PNG file not found: {png_file}")

    if json_file.exists():
        logger.info(f"✅ Affine transform saved: {json_file}")
    else:
        logger.warning(f"❌ JSON file not found: {json_file}")

    logger.info("\n✅ OSM → MapDefinition pipeline complete!")
    logger.info(f"   MapDefinition: created with {len(map_def.obstacles)} obstacles")
    logger.info(f"   Output directory: {output_path}")


if __name__ == "__main__":
    main()
