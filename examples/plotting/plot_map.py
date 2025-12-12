"""plot Map Definition"""

from robot_sf.maps import visualize_map_definition
from robot_sf.nav.svg_map_parser import convert_map

map_def = convert_map("maps/svg_maps/map3_1350_buildings_inkscape.svg")
visualize_map_definition(map_def, output_path="output/map.png", title="Map Overview")
