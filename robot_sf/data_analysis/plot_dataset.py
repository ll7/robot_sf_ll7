"""
This module provides basic functions to plot pedestrian data extracted from numpy arrays.

Key Features:
    - Plot all NPC pedestrian positions
    - Plot all NPC pedestrian velocities
    - Plot ego pedestrian acceleration
    - Plot ego pedestrian velocity

"""

import matplotlib.pyplot as plt
import numpy as np

from robot_sf.data_analysis.extract_json_from_pickle import (
    extract_key_from_json,
    extract_timestamp,
)
from robot_sf.data_analysis.plot_utils import save_plot
from robot_sf.nav.map_config import MapDefinition


def plot_all_npc_ped_positions(
    ped_positions_array: np.ndarray,
    interactive: bool = False,
    unique_id: str = None,
    map_def: MapDefinition = None,
):
    """
    Plot all NPC pedestrian positions from the given position array.

    Args:
        ped_position_array (np.ndarray): shape: (timesteps, num_pedestrians, 2)
        interactive (bool): If True, show the plot interactively.
        unique_id (str): Unique identifier for the plot filename, usually the timestamp
        map_def (MapDefinition, optional): Map definition to plot obstacles
    """
    # Create a figure and axes
    _fig, ax = plt.subplots(figsize=(10, 8))

    # E.g.: For only 100 timesteps x_vals = ped_positions_array[:100, :, 0]
    x_vals = ped_positions_array[:, :, 0]
    y_vals = ped_positions_array[:, :, 1]

    # colormap for better visibility
    colors = np.random.rand(x_vals.shape[0], x_vals.shape[1])

    ax.scatter(x_vals, y_vals, c=colors, alpha=0.5, s=1)

    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")

    # Plot map obstacles if map_def is provided
    if map_def is not None:
        map_def.plot_map_obstacles(ax)

    # Prepare filename
    if unique_id:
        filename = f"robot_sf/data_analysis/plots/all_npc_pedestrian_positions_{unique_id}.png"
    else:
        filename = "robot_sf/data_analysis/plots/all_npc_pedestrian_positions.png"

    # Save the plot
    save_plot(filename, "All recorded npc pedestrian positions", interactive)


def plot_all_npc_ped_velocities(
    ped_actions: list, raw: bool = True, interactive: bool = False, unique_id: str = None
):
    """
    Plot all NPC pedestrian velocities from the given list of actions.
    Based on the actions of the npc pedestrians.

    Args:
        ped_actions (list): List of pedestrian actions.
        raw (bool): If raw is True, dont use the applied scaling factor (for visual purpose).
        interactive (bool): If True, show the plot interactively.
        unique_id (str): Unique identifier for the plot filename, usually the timestamp
    """
    velocity_list = []
    for timestep, actions in enumerate(ped_actions):
        if actions:  # Check if positions list is not empty
            current_velocity = []
            for action in actions:
                vel_vector = np.array(action[1]) - np.array(action[0])
                velocity = np.linalg.norm(vel_vector)
                if raw:
                    velocity = velocity / 2  # See pedestrian_env.py -> ped_actions = ...
                current_velocity.append(velocity)
        velocity_list.append(current_velocity)

    for timestep, vels in enumerate(velocity_list):
        plt.scatter([timestep] * len(vels), vels, alpha=0.5, c="blue", s=1)

    plt.xlabel("Time Step")
    plt.ylabel("Velocity")

    # Prepare title and filename
    title = "Pedestrian Velocity over Time, raw" if raw else "Pedestrian Velocity over Time"
    plot_type = "all_npc_ped_velocities_raw" if raw else "all_npc_ped_velocities"

    if unique_id:
        filename = f"robot_sf/data_analysis/plots/{plot_type}_{unique_id}.png"
    else:
        filename = f"robot_sf/data_analysis/plots/{plot_type}.png"

    # Save the plot
    save_plot(filename, title, interactive)


def plot_ego_ped_acceleration(
    ego_ped_acceleration: list, interactive: bool = False, unique_id: str = None
):
    """
    Plot the acceleration of the ego pedestrian.

    Args:
        ego_ped_acceleration (list): List of ego pedestrian accelerations.
        interactive (bool): If True, show the plot interactively.
        unique_id (str): Unique identifier for the plot filename, usually the timestamp
    """
    plt.plot(ego_ped_acceleration, label="Acceleration")
    plt.xlabel("Timestep")
    plt.ylabel("Acceleration")
    plt.legend()

    # Prepare filename
    if unique_id:
        filename = f"robot_sf/data_analysis/plots/ego_ped_acc_{unique_id}.png"
    else:
        filename = "robot_sf/data_analysis/plots/ego_ped_acc.png"

    # Save the plot
    save_plot(filename, "Ego Ped Acceleration over Time", interactive)


def plot_ego_ped_velocity(
    ego_ped_acceleration: list, interactive: bool = False, unique_id: str = None
):
    """
    Plot the velocity of the ego pedestrian based on the acceleration.

    Args:
        ego_ped_acceleration (list): List of ego pedestrian accelerations.
        interactive (bool): If True, show the plot interactively.
        unique_id (str): Unique identifier for the plot filename, usually the timestamp
    """
    ego_ped_velocity = []
    cumulative_sum = 0.0
    for acc in ego_ped_acceleration:
        cumulative_sum += acc
        # Clip, because the ego pedestrian can't go faster than 3 m/s
        cumulative_sum = np.clip(cumulative_sum, None, 3)
        ego_ped_velocity.append(cumulative_sum)

    ego_ped_velocity = np.array(ego_ped_velocity)

    plt.plot(ego_ped_velocity, label="Velocity")
    plt.xlabel("Timestep")
    plt.ylabel("Velocity")
    plt.legend()

    # Prepare filename
    if unique_id:
        filename = f"robot_sf/data_analysis/plots/ego_ped_vel_{unique_id}.png"
    else:
        filename = "robot_sf/data_analysis/plots/ego_ped_vel.png"

    # Save the plot
    save_plot(filename, "Ego Ped Velocity over Time", interactive)


def main():
    # filename = "robot_sf/data_analysis/datasets/2025-02-06_10-24-12.json"
    filename = "robot_sf/data_analysis/datasets/2025-01-16_11-47-44.json"
    unique_id = extract_timestamp(filename)

    # ped_positions_array = extract_key_from_json_as_ndarray(filename, "pedestrian_positions")
    # ped_actions = extract_key_from_json(filename, "ped_actions")
    ego_ped_acceleration = [
        item["action"][0] for item in extract_key_from_json(filename, "ego_ped_action")
    ]

    # plot_all_npc_ped_positions(ped_positions_array)
    # plot_all_npc_ped_velocities(ped_actions, raw=True)
    plot_ego_ped_acceleration(ego_ped_acceleration, interactive=True, unique_id=unique_id)
    # plot_ego_ped_velocity(ego_ped_acceleration)


if __name__ == "__main__":
    main()
