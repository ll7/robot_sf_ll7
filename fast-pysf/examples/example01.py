"""
# Example 01: basic import test
"""
import pysocialforce as pysf


simulator = pysf.Simulator_v2()

for step in range(10):
    simulator.step()
    print(f"step {step}")
    print(simulator.states.ped_positions)
    print("=================")
