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

from robot_sf.data_analysis.generate_dataset import (
    extract_key_from_json,
)


def plot_all_npc_ped_positions(ped_positions_array: np.ndarray):
    """
    Plot all NPC pedestrian positions from the given position array.

    Args:
        ped_position_array (np.ndarray): shape: (timesteps, num_pedestrians, 2)
    """
    # E.g.: For only 100 timesteps x_vals = ped_positions_array[:100, :, 0]
    x_vals = ped_positions_array[:, :, 0]
    y_vals = ped_positions_array[:, :, 1]

    # colormap for better visibility
    colors = np.random.rand(x_vals.shape[0], x_vals.shape[1])

    plt.scatter(x_vals, y_vals, c=colors, alpha=0.5, s=1)

    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("All recorded npc pedestrian positions")
    # plt.legend()
    plt.tight_layout()
    plt.savefig("robot_sf/data_analysis/plots/all_npc_pedestrian_positions.png")


def plot_all_npc_ped_velocities(ped_actions: list, raw: bool = True):
    """
    Plot all NPC pedestrian velocities from the given list of actions.
    Based on the actions of the npc pedestrians.

    Args:
        ped_actions (list): List of pedestrian actions.
        raw (bool): If raw is True, dont use the applied scaling factor (for visual purpose).
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
        # if timestep == 100:
        #     break

    for timestep, vels in enumerate(velocity_list):
        plt.scatter([timestep] * len(vels), vels, alpha=0.5, c="blue", s=1)

    plt.xlabel("Time Step")
    plt.ylabel("Velocity")
    plt.title("Pedestrian Velocity over Time, raw" if raw else "Pedestrian Velocity over Time")
    # plt.legend()
    plt.tight_layout()
    title = "all_npc_ped_velocities_raw" if raw else "all_npc_ped_velocities"
    plt.savefig(f"robot_sf/data_analysis/plots/{title}.png")


def plot_ego_ped_acceleration(ego_ped_acceleration: list):
    """
    Plot the acceleration of the ego pedestrian.

    Args:
        ego_ped_acceleration (list): List of ego pedestrian accelerations.
    """

    plt.plot(ego_ped_acceleration, label="Acceleration")
    plt.xlabel("Timestep")
    plt.ylabel("Acceleration")
    plt.title("Ego Ped Acceleration over Time")
    plt.legend()
    plt.savefig("robot_sf/data_analysis/plots/ego_ped_acc.png")


def plot_ego_ped_velocity(ego_ped_acceleration: list):
    """
    Plot the velocity of the ego pedestrian based on the acceleration.

    Args:
        ego_ped_acceleration (list): List of ego pedestrian accelerations.
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
    plt.title("Ego Ped Velocity over Time")
    plt.legend()
    plt.savefig("robot_sf/data_analysis/plots/ego_ped_vel.png")
    plt.show()


def main():
    # filename = "robot_sf/data_analysis/datasets/2025-02-06_10-24-12.json"
    filename = "robot_sf/data_analysis/datasets/2025-01-16_11-47-44.json"

    # ped_positions_array = extract_key_from_json_as_ndarray(filename, "pedestrian_positions")
    # ped_actions = extract_key_from_json(filename, "ped_actions")
    ego_ped_acceleration = [
        item["action"][0] for item in extract_key_from_json(filename, "ego_ped_action")
    ]

    # plot_all_npc_ped_positions(ped_positions_array)
    # plot_all_npc_ped_velocities(ped_actions, raw=True)
    plot_ego_ped_acceleration(ego_ped_acceleration)
    # plot_ego_ped_velocity(ego_ped_acceleration)


if __name__ == "__main__":
    main()
