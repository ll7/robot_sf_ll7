from loguru import logger

from robot_sf.nav.svg_map_parser import SvgMapConverter

SVG_FILE = "maps/svg_maps/02_simple_maps.svg"

logger.info("Converting SVG map to MapDefinition object.")
logger.info(f"SVG file: {SVG_FILE}")

converter = SvgMapConverter(SVG_FILE)

logger.info("MapDefinition object created.")
