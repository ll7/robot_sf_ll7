"""example svg conversion"""

from robot_sf.common.logging import configure_logging
from robot_sf.maps.import_svg_from_osm import import_svg_from_osm

OSM_SVG_PATH = "maps/osm_svg_maps/uni_campus_1350.svg"
OUTPUT_SVG_PATH = "maps/obstacle_svg_maps/uni_campus_1350_obstacles.svg"


def main():
    configure_logging()
    import_svg_from_osm(OSM_SVG_PATH, OUTPUT_SVG_PATH, map_scale_factor=1350)


if __name__ == "__main__":
    main()
