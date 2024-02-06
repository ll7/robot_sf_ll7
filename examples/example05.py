"""
# Example 05: simulate map03
"""

import pysocialforce as pysf

map_def = pysf.load_map("./maps/map03.json")
display = pysf.SimulationView(map_def=map_def, scaling=10)
render_step = lambda t, s: display.render(pysf.to_visualizable_state(t, s))
simulator = pysf.Simulator_v2(map_def, on_step=render_step)

display.show()
for step in range(10_000):
    simulator.step()
display.exit()
