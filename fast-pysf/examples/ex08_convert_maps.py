"""
Take a open street map file and convert it to a svg file with the building only as obstacles

"""

import logging

from pysocialforce.map_osm_converter import (
    add_scale_bar_to_root,
    extract_buildings_as_obstacle,
    save_root_as_svg,
)

logging.basicConfig(level=logging.DEBUG)

# Extract the buildings from the map and save them to a new SVG file
root = extract_buildings_as_obstacle("maps/osm_maps/map3_1350.svg", map_scale_factor=1350)

# Add a scale bar to the root
scale_viz_root = add_scale_bar_to_root(root, line_length=50)

save_root_as_svg(
    scale_viz_root,
    "maps/osm_maps/converted_maps/map3_1350_buildings.svg",
)
