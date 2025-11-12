#!/usr/bin/env python3
"""Inspect the fast-pysf pedestrian speed fix output.

Usage:
    uv run python examples/advanced/05_fast_pysf_speed_fix.py

Prerequisites:
    - None

Expected Output:
    - Console diff comparing pre- and post-fix pedestrian velocity vectors.

Limitations:
    - Informational only; does not launch a simulator.

References:
    - docs/performance_notes.md#fast-pysf-speed-fix
"""

import numpy as np

from robot_sf.render.sim_view import VisualizableSimState


def demonstrate_fix():
    """Demonstrate the difference before and after the fix."""

    # Example pedestrian data
    ped_positions = np.array([[1.0, 1.0], [3.0, 2.0]])
    ped_velocities = np.array([[0.5, 0.3], [0.2, 0.4]])

    print("=== Demonstration of 2x Speed Fix ===\n")

    print("Pedestrian positions:")
    print(ped_positions)
    print("\nPedestrian velocities:")
    print(ped_velocities)

    # Show what the old (incorrect) ped_actions would have been
    old_ped_actions = np.column_stack([ped_positions, ped_positions + ped_velocities * 2]).reshape(
        len(ped_positions),
        2,
        2,
    )

    print("\n--- BEFORE FIX (incorrect) ---")
    print("ped_actions with 2x multiplier:")
    for i, (start, end) in enumerate(old_ped_actions):
        velocity_vector = end - start
        print(f"Pedestrian {i}: {start} -> {end} (velocity: {velocity_vector})")

    # Show what the new (correct) ped_actions are
    new_ped_actions = np.column_stack([ped_positions, ped_positions + ped_velocities]).reshape(
        len(ped_positions),
        2,
        2,
    )

    print("\n--- AFTER FIX (correct) ---")
    print("ped_actions with actual velocities:")
    for i, (start, end) in enumerate(new_ped_actions):
        velocity_vector = end - start
        print(f"Pedestrian {i}: {start} -> {end} (velocity: {velocity_vector})")

    # Create a VisualizableSimState with the corrected data
    state = VisualizableSimState(
        timestep=0,
        robot_action=None,
        robot_pose=((0.0, 0.0), 0.0),
        pedestrian_positions=ped_positions,
        ray_vecs=np.array([[[0.0, 0.0], [1.0, 1.0]]]),
        ped_actions=new_ped_actions,
        time_per_step_in_secs=0.1,
    )

    print("\n--- VISUALIZABLE STATE CREATED ---")
    print(f"Timestep: {state.timestep}")
    print(f"Number of pedestrians: {len(state.pedestrian_positions)}")
    print(f"ped_actions shape: {state.ped_actions.shape}")
    print(f"time_per_step_in_secs: {state.time_per_step_in_secs}")

    print("\n=== Impact on Data Analysis ===")
    print("✅ Data analysts now get actual pedestrian speeds")
    print("✅ Saved recordings contain accurate velocity data")
    print("✅ Consistency with fast-pysf implementation")
    print("✅ Better foundation for research and analysis")


if __name__ == "__main__":
    demonstrate_fix()
