"""
# Example 03: simulate with visual rendering
"""

import pysocialforce as pysf

map_def = pysf.load_map("./maps/default_map.json")
display = pysf.SimulationView(map_def=map_def, scaling=10)


def render_step(t, s):
    """TODO docstring. Document this function.

    Args:
        t: TODO docstring.
        s: TODO docstring.
    """
    return display.render(pysf.to_visualizable_state(t, s))


simulator = pysf.Simulator_v2(map_def, on_step=render_step)

display.show()
for step in range(10_000):
    simulator.step()
display.exit()
