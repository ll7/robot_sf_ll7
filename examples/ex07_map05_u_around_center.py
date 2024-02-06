"""
# Example 07: load a U shaped pedestrian map and simulate a route around an obstacle
"""

import pysocialforce as pysf

map_def = pysf.load_map("./maps/map05_u_around_center.json")
display = pysf.SimulationView(obstacles=map_def.obstacles, scaling=10)
render_step = lambda t, s: display.render(pysf.to_visualizable_state(t, s))
simulator = pysf.Simulator_v2(map_def, on_step=render_step)

display.show()
for step in range(10_000):
    simulator.step()
display.exit()
