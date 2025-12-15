"""export OSM SVG file as base svg with obstacle"""

from loguru import logger
from pysocialforce.map_osm_converter import (
    add_scale_bar_to_root,
    extract_buildings_as_obstacle,
    save_root_as_svg,
)


def import_svg_from_osm(input_osm_svg_file: str, output_svg_file: str, map_scale_factor: float):
    """Extract building obstacles from an OSM SVG file and save to a new SVG file.

    Args:
        input_osm_svg_file: Path to the input OSM SVG file.
        output_svg_file: Path to save the output SVG file with building obstacles.
        map_scale_factor: Scale factor for the map (e.g., pixels per meter).
    """
    logger.info(
        f"Extracting buildings from {input_osm_svg_file} with scale factor {map_scale_factor}"
    )

    # Extract the buildings from the map and save them to a new SVG file
    root = extract_buildings_as_obstacle(input_osm_svg_file, map_scale_factor=map_scale_factor)

    # Add a scale bar to the root for visualization
    scale_viz_root = add_scale_bar_to_root(root, line_length=50)

    # Save the resulting SVG with building obstacles
    save_root_as_svg(
        scale_viz_root,
        output_svg_file,
    )
    logger.info(f"Saved extracted building obstacles to {output_svg_file}")
