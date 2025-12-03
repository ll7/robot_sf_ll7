"""
ex09_inkscape_svg_map.py
Inkscape SVG map used in in pysocialforce.
"""

import logging

import pysocialforce as pysf
from pysocialforce.map_loader_svg import path_info_to_mapdefintion, svg_path_info

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# extract svg information
path_info = svg_path_info("maps/osm_maps/converted_maps/map3_1350_buildings_inkscape.svg")

map_def = path_info_to_mapdefintion(path_info)

print(map_def.routes)

display = pysf.SimulationView(map_def=map_def, scaling=10)


def render_step(t, s):
    """TODO docstring. Document this function.

    Args:
        t: TODO docstring.
        s: TODO docstring.
    """
    return display.render(pysf.to_visualizable_state(t, s))


simulator = pysf.Simulator_v2(map_def, on_step=render_step)

logger.info("Running simulation")
display.show()
for step in range(10_000):
    simulator.step()
display.exit()
