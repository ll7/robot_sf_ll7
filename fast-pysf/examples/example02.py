"""
# Example 02: test the configuration of the simulator
"""
import pysocialforce as pysf

map_def = pysf.load_map("./maps/default_map.json")
simulator = pysf.Simulator_v2(map_def)

for step in range(10):
    simulator.step()
    print(f"step {step}")
    print(simulator.states.ped_positions)
    print("=================")
