"""
Module to analyze and visualize the trajectories, velocities, and accelerations of pedestrians
from numpy arrays.

Key Features:
    - Plot the trajectories of NPC pedestrians.
    - Split trajectories when the distance between two consecutive points is greater than normal and
        create a new trajectory segment.
    - Calculate and plot the velocity and acceleration of pedestrians.
    - Calculate and plot the probability distribution of the velocity and acceleration of
        the NPC pedestrians.
"""

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from scipy.stats import norm

from robot_sf.data_analysis.extract_json_from_pickle import (
    extract_key_from_json,
    extract_key_from_json_as_ndarray,
    extract_timestamp,
)
from robot_sf.data_analysis.plot_utils import save_plot
from robot_sf.nav.map_config import MapDefinition

TRAJECTORY_DISCONTINUITY_THRESHOLD = 2  # Threshold for abnormal distance

# Global flag to avoid log spam in calculate_velocity and calculate_acceleration
time_interval_warning_logged = False


def plot_single_splitted_traj(
    ped_positions_array: np.ndarray,
    ped_idx: int = 0,
    interactive: bool = False,
    unique_id: str = None,
    map_def: MapDefinition = None,
):
    """
    Plot from position_array from a single pedestrian id the multiple trajectories.
    Split when the distance between two consecutive points is greater than normal.

    Notes: A simulation pedestrian teleports to the spawn if the route is finished and starts a new
            route.

    Args:
        ped_position_array (np.ndarray): shape: (timesteps, num_pedestrians, 2)
        ped_idx (int): Which simulation pedestrian is inspected
        interactive (bool): If True, show the plot interactively
        unique_id (str): Unique identifier for the plot filename, usually the timestamp
        map_def (MapDefinition, optional): Map definition to plot obstacles
    """
    # Create figure and axes for better control
    fig, ax = plt.subplots(figsize=(10, 8))

    x_vals = ped_positions_array[:, ped_idx, 0]
    y_vals = ped_positions_array[:, ped_idx, 1]

    distances = np.sqrt(np.diff(x_vals) ** 2 + np.diff(y_vals) ** 2)
    start_idx = 0
    for i, dist in enumerate(distances):
        if dist > TRAJECTORY_DISCONTINUITY_THRESHOLD:
            ax.plot(
                x_vals[start_idx : i + 1],
                y_vals[start_idx : i + 1],
                label=f"Pedestrian {start_idx}",
            )
            # Start point
            ax.scatter(x_vals[start_idx], y_vals[start_idx], color="green", marker="o")
            # End point
            ax.scatter(x_vals[i], y_vals[i], color="red", marker="x")
            start_idx = i + 1

    # Plot the last segment
    ax.plot(x_vals[start_idx:], y_vals[start_idx:], label=f"Pedestrian {start_idx}")
    ax.scatter(x_vals[start_idx], y_vals[start_idx], color="green", marker="o")  # Start point
    ax.scatter(x_vals[-1], y_vals[-1], color="red", marker="x")  # End point

    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title(f"Pedestrian Trajectories: {x_vals.shape[0]} Steps")
    ax.invert_yaxis()

    # Plot map obstacles if map_def is provided
    if map_def is not None:
        map_def.plot_map_obstacles(ax)

    # Prepare filename and save plot
    if unique_id:
        filename = f"robot_sf/data_analysis/plots/single_splitted_npc{ped_idx}_traj_{unique_id}.png"
    else:
        filename = f"robot_sf/data_analysis/plots/single_splitted_npc{ped_idx}_traj.png"

    save_plot(filename, None, interactive)


def plot_all_splitted_traj(
    ped_positions_array: np.ndarray,
    interactive: bool = False,
    unique_id: str = None,
    map_def: MapDefinition = None,
):
    """
    Plot from position_array all npc pedestrian trajectories.
    Split when the distance between two consecutive points is greater than normal.

    Args:
        ped_position_array (np.ndarray): shape: (timesteps, num_pedestrians, 2)
        interactive (bool): If True, show the plot interactively
        unique_id (str): Unique identifier for the plot filename, usually the timestamp
        map_def (MapDefinition, optional): Map definition to plot obstacles
    """
    _, num_pedestrians, _ = ped_positions_array.shape

    # Create figure and axes for better control
    fig, ax = plt.subplots(figsize=(10, 8))

    for ped_idx in range(num_pedestrians):
        # Extract x and y for this ped_idx across all timesteps
        x_vals = ped_positions_array[:, ped_idx, 0]
        y_vals = ped_positions_array[:, ped_idx, 1]

        distances = np.sqrt(np.diff(x_vals) ** 2 + np.diff(y_vals) ** 2)
        start_idx = 0
        for i, dist in enumerate(distances):
            if dist > TRAJECTORY_DISCONTINUITY_THRESHOLD:
                ax.plot(
                    x_vals[start_idx : i + 1],
                    y_vals[start_idx : i + 1],
                    label=f"Pedestrian {ped_idx} Segment",
                )
                # Start point
                ax.scatter(x_vals[start_idx], y_vals[start_idx], color="green", marker="o")
                # End point
                ax.scatter(x_vals[i], y_vals[i], color="red", marker="x")
                start_idx = i + 1

        # Plot the last segment
        ax.plot(x_vals[start_idx:], y_vals[start_idx:], label=f"Pedestrian {ped_idx} Segment")
        ax.scatter(x_vals[start_idx], y_vals[start_idx], color="green", marker="o")
        ax.scatter(x_vals[-1], y_vals[-1], color="red", marker="x")

    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title("Pedestrian Trajectories")
    # plt.legend()  # Legend would be too crowded with many pedestrians
    ax.invert_yaxis()

    # Plot map obstacles if map_def is provided
    if map_def is not None:
        map_def.plot_map_obstacles(ax)

    # Prepare filename and save plot
    if unique_id:
        filename = f"robot_sf/data_analysis/plots/all_splitted_npc_traj_{unique_id}.png"
    else:
        filename = "robot_sf/data_analysis/plots/all_splitted_npc_traj.png"

    save_plot(filename, None, interactive)


def calculate_velocity(
    x_vals: np.ndarray, y_vals: np.ndarray, time_interval: float = None
) -> np.ndarray:
    """Calculate the velocity of a pedestrian given their x and y positions."""
    global time_interval_warning_logged
    if time_interval is None:
        if time_interval_warning_logged is False:
            time_interval_warning_logged = True
            logger.warning("Time interval not provided. Using default value of 0.1 seconds.")
        time_interval = 0.1

    # Calculate the differences between consecutive points
    dx = np.diff(x_vals)
    dy = np.diff(y_vals)

    # Calculate the Euclidean distance (velocity) between consecutive points
    distances = np.sqrt(dx**2 + dy**2)
    velocities = distances / time_interval

    return velocities


def calculate_acceleration(velocities: np.ndarray, time_interval: float = None) -> np.ndarray:
    """Calculate the acceleration of a pedestrian given their velocities."""
    # Calculate the differences between consecutive velocities
    global time_interval_warning_logged
    if time_interval is None:
        if time_interval_warning_logged is False:
            time_interval_warning_logged = True
            logger.warning("Time interval not provided. Using default value of 0.1 seconds.")
        time_interval = 0.1

    dv = np.diff(velocities)

    # Calculate the acceleration
    accelerations = dv / time_interval

    return accelerations


def subplot_single_splitted_traj_acc(
    ped_positions_array: np.ndarray,
    ped_idx: int = 0,
    interactive: bool = False,
    unique_id: str = None,
    map_def: MapDefinition = None,
):
    """
    Plot from position_array for a single pedestrian id trajectories, velocity and acceleration.

    Args:
        ped_position_array (np.ndarray): shape: (timesteps, num_pedestrians, 2)
        ped_idx (int): Which simulation pedestrian is inspected
        interactive (bool): If True, show the plot interactively
        unique_id (str): Unique identifier for the plot filename, usually the timestamp
        map_def (MapDefinition, optional): Map definition to plot obstacles
    """
    _, axes = plt.subplots(1, 3, figsize=(18, 6))

    x_vals = ped_positions_array[:, ped_idx, 0]
    y_vals = ped_positions_array[:, ped_idx, 1]

    distances = np.sqrt(np.diff(x_vals) ** 2 + np.diff(y_vals) ** 2)
    counter = 0
    start_idx = 0
    for i, dist in enumerate(distances):
        if dist > TRAJECTORY_DISCONTINUITY_THRESHOLD:
            axes[0].plot(
                x_vals[start_idx : i + 1],
                y_vals[start_idx : i + 1],
                label=f"Pedestrian {start_idx}",
            )
            axes[0].scatter(x_vals[start_idx], y_vals[start_idx], color="green", marker="o")
            axes[0].scatter(x_vals[i], y_vals[i], color="red", marker="x")

            velocities = calculate_velocity(x_vals[start_idx : i + 1], y_vals[start_idx : i + 1])
            accelerations = calculate_acceleration(velocities)
            axes[1].plot(range(len(velocities)), velocities, label=f"Pedestrian {start_idx}")
            axes[2].plot(range(len(accelerations)), accelerations, label=f"Pedestrian {start_idx}")

            counter += 1
            start_idx = i + 1
    # Plot the last segment
    axes[0].plot(x_vals[start_idx:], y_vals[start_idx:], label=f"Pedestrian {start_idx}")
    axes[0].scatter(x_vals[start_idx], y_vals[start_idx], color="green", marker="o")  # Start point
    axes[0].scatter(x_vals[-1], y_vals[-1], color="red", marker="x")  # End point

    if map_def:
        map_def.plot_map_obstacles(axes[0])

    velocities = calculate_velocity(x_vals[start_idx:], y_vals[start_idx:])
    accelerations = calculate_acceleration(velocities)
    axes[1].plot(range(len(velocities)), velocities, label=f"Pedestrian {start_idx}")
    axes[2].plot(range(len(accelerations)), accelerations, label=f"Pedestrian {start_idx}")

    axes[0].set_xlabel("X Position")
    axes[0].set_ylabel("Y Position")
    axes[0].set_title(f"Pedestrian Trajectories: {x_vals.shape[0]}")
    axes[0].invert_yaxis()
    axes[0].legend()

    axes[1].set_xlabel("Timestep")
    axes[1].set_ylabel("Velocity")
    axes[1].set_title("Pedestrian Velocities")
    axes[1].legend()

    axes[2].set_xlabel("Timestep")
    axes[2].set_ylabel("Acceleration")
    axes[2].set_title("Pedestrian Accelerations")
    axes[2].legend()

    # Prepare filename and save plot
    if unique_id:
        filename = f"robot_sf/data_analysis/plots/subplot_npc_{ped_idx}_{unique_id}.png"
    else:
        filename = f"robot_sf/data_analysis/plots/subplot_npc_{ped_idx}.png"

    save_plot(filename, None, interactive)


def plot_acceleration_distribution(
    ped_positions_array: np.ndarray, interactive: bool = False, unique_id: str = None
):
    """
    Calculate and plot the probability distribution of the acceleration of all pedestrians.

    Args:
        ped_position_array (np.ndarray): shape: (timesteps, num_pedestrians, 2)
        interactive (bool): If True, show the plot interactively
        unique_id (str): Unique identifier for the plot filename, usually the timestamp
    """
    _, num_pedestrians, _ = ped_positions_array.shape

    all_accelerations = []

    for ped_idx in range(num_pedestrians):
        x_vals = ped_positions_array[:, ped_idx, 0]
        y_vals = ped_positions_array[:, ped_idx, 1]

        distances = np.sqrt(np.diff(x_vals) ** 2 + np.diff(y_vals) ** 2)
        start_idx = 0
        for i, dist in enumerate(distances):
            if dist > TRAJECTORY_DISCONTINUITY_THRESHOLD:
                velocities = calculate_velocity(
                    x_vals[start_idx : i + 1], y_vals[start_idx : i + 1]
                )
                acceleration = calculate_acceleration(velocities)
                all_accelerations.extend(acceleration)
                start_idx = i + 1

        velocities = calculate_velocity(x_vals[start_idx:], y_vals[start_idx:])
        acceleration = calculate_acceleration(velocities)
        all_accelerations.extend(acceleration)

    max_acceleration = max(all_accelerations)
    logger.info(f"Maximum Acceleration: {max_acceleration}")

    # Plot the histogram of accelerations
    plt.hist(all_accelerations, bins=60, density=True, alpha=0.6, color="g")

    # Fit a normal distribution to the data
    mu, std = norm.fit(all_accelerations)

    # Plot the PDF
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, "k", linewidth=2)

    plt.xlabel("Acceleration")
    plt.ylabel("Probability Density")
    plt.title("Probability Distribution of Pedestrian Accelerations")

    # Prepare filename and save plot
    if unique_id:
        filename = f"robot_sf/data_analysis/plots/acceleration_distribution_{unique_id}.png"
    else:
        filename = "robot_sf/data_analysis/plots/acceleration_distribution.png"

    save_plot(filename, None, interactive)


def plot_velocity_distribution(
    ped_positions_array: np.ndarray, interactive: bool = False, unique_id: str = None
):
    """
    Calculate and plot the probability distribution of the velocity of all pedestrians.

    Args:
        ped_position_array (np.ndarray): shape: (timesteps, num_pedestrians, 2)
        interactive (bool): If True, show the plot interactively
        unique_id (str): Unique identifier for the plot filename, usually the timestamp
    """
    _, num_pedestrians, _ = ped_positions_array.shape

    all_velocities = []

    for ped_idx in range(num_pedestrians):
        x_vals = ped_positions_array[:, ped_idx, 0]
        y_vals = ped_positions_array[:, ped_idx, 1]

        distances = np.sqrt(np.diff(x_vals) ** 2 + np.diff(y_vals) ** 2)
        start_idx = 0
        for i, dist in enumerate(distances):
            if dist > TRAJECTORY_DISCONTINUITY_THRESHOLD:
                velocities = calculate_velocity(
                    x_vals[start_idx : i + 1], y_vals[start_idx : i + 1]
                )
                all_velocities.extend(velocities)
                start_idx = i + 1

        velocities = calculate_velocity(x_vals[start_idx:], y_vals[start_idx:])
        all_velocities.extend(velocities)

    max_velocity = max(all_velocities)
    logger.info(f"Maximum Velocity: {max_velocity}")

    # Plot the histogram of velocities
    plt.hist(all_velocities, bins=60, density=True, alpha=0.6, color="b")

    # Fit a normal distribution to the data
    mu, std = norm.fit(all_velocities)

    # Plot the PDF
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, "k", linewidth=2)

    plt.xlabel("Velocity")
    plt.ylabel("Probability Density")
    plt.title("Probability Distribution of Pedestrian Velocities")

    # Prepare filename and save plot
    if unique_id:
        filename = f"robot_sf/data_analysis/plots/velocity_distribution_{unique_id}.png"
    else:
        filename = "robot_sf/data_analysis/plots/velocity_distribution.png"

    save_plot(filename, None, interactive)


def subplot_velocity_distribution_with_ego_ped(
    ped_positions_array: np.ndarray,
    ego_positions: np.ndarray,
    interactive: bool = False,
    unique_id: str = None,
):
    """
    Calculate and plot the probability distribution of the velocity of all pedestrians
    in comparison to the ego pedestrian.

    Args:
        ped_position_array (np.ndarray): shape: (timesteps, num_pedestrians, 2)
        ego_positions (np.ndarray): shape: (timesteps, 2)
        interactive (bool): If True, show the plot interactively
        unique_id (str): Unique identifier for the plot filename, usually the timestamp
    """
    _, num_pedestrians, _ = ped_positions_array.shape

    all_npc_velocities = []
    ego_velocities = calculate_velocity(ego_positions[:, 0], ego_positions[:, 1])

    for ped_idx in range(num_pedestrians):
        x_vals = ped_positions_array[:, ped_idx, 0]
        y_vals = ped_positions_array[:, ped_idx, 1]

        distances = np.sqrt(np.diff(x_vals) ** 2 + np.diff(y_vals) ** 2)
        start_idx = 0
        for i, dist in enumerate(distances):
            if dist > TRAJECTORY_DISCONTINUITY_THRESHOLD:
                velocities = calculate_velocity(
                    x_vals[start_idx : i + 1], y_vals[start_idx : i + 1]
                )
                all_npc_velocities.extend(velocities)
                start_idx = i + 1

        velocities = calculate_velocity(x_vals[start_idx:], y_vals[start_idx:])
        all_npc_velocities.extend(velocities)

    max_ego = max(ego_velocities)
    max_npc = max(all_npc_velocities)
    logger.info(f"Maximum Velocity Ego: {max_ego}, NPC: {max_npc}")

    # Plot the histogram of velocities
    _, axes = plt.subplots(1, 2, figsize=(18, 6))

    axes[0].hist(all_npc_velocities, bins=60, density=True, alpha=0.6, color="b")
    mu_npc, std_npc = norm.fit(all_npc_velocities)
    x_npc = np.linspace(0, max_npc, 100)
    p_npc = norm.pdf(x_npc, mu_npc, std_npc)
    axes[0].plot(x_npc, p_npc, "k", linewidth=2)
    axes[0].set_title("Probability Distribution of NPC Pedestrian Velocities")
    axes[0].set_xlabel("Velocity")
    axes[0].set_ylabel("Probability Density")

    axes[1].hist(ego_velocities, bins=60, density=True, alpha=0.6, color="r")
    mu_ego, std_ego = norm.fit(ego_velocities)
    x_ego = np.linspace(0, max_ego, 100)
    p_ego = norm.pdf(x_ego, mu_ego, std_ego)
    axes[1].plot(x_ego, p_ego, "k", linewidth=2)
    axes[1].set_title("Probability Distribution of Ego Pedestrian Velocities")
    axes[1].set_xlabel("Velocity")
    axes[1].set_ylabel("Probability Density")

    plt.tight_layout()

    # Prepare filename and save plot
    if unique_id:
        filename = f"robot_sf/data_analysis/plots/velocity_distribution_comparison_{unique_id}.png"
    else:
        filename = "robot_sf/data_analysis/plots/velocity_distribution_comparison.png"

    save_plot(filename, None, interactive)


def subplot_acceleration_distribution(
    ped_positions_array: np.ndarray,
    ego_positions: np.ndarray,
    interactive: bool = False,
    unique_id: str = None,
):
    """
    Calculate and plot the probability distribution of the acceleration of all pedestrians.

    Args:
        ped_position_array (np.ndarray): shape: (timesteps, num_pedestrians, 2)
        ego_positions (np.ndarray): shape: (timesteps, 2)
        interactive (bool): If True, show the plot interactively
        unique_id (str): Unique identifier for the plot filename, usually the timestamp
    """
    _, num_pedestrians, _ = ped_positions_array.shape

    all_npc_accelerations = []
    ego_velocities = calculate_velocity(ego_positions[:, 0], ego_positions[:, 1])
    ego_accelerations = calculate_acceleration(ego_velocities)

    for ped_idx in range(num_pedestrians):
        x_vals = ped_positions_array[:, ped_idx, 0]
        y_vals = ped_positions_array[:, ped_idx, 1]

        distances = np.sqrt(np.diff(x_vals) ** 2 + np.diff(y_vals) ** 2)
        start_idx = 0
        for i, dist in enumerate(distances):
            if dist > TRAJECTORY_DISCONTINUITY_THRESHOLD:
                velocities = calculate_velocity(
                    x_vals[start_idx : i + 1], y_vals[start_idx : i + 1]
                )
                accelerations = calculate_acceleration(velocities)
                all_npc_accelerations.extend(accelerations)
                start_idx = i + 1

        velocities = calculate_velocity(x_vals[start_idx:], y_vals[start_idx:])
        accelerations = calculate_acceleration(velocities)
        all_npc_accelerations.extend(accelerations)

    max_ego = max(ego_accelerations)
    max_npc = max(all_npc_accelerations)
    min_ego = min(ego_accelerations)
    min_npc = min(all_npc_accelerations)
    logger.info(f"Maximum Acceleration Ego: {max_ego}, NPC: {max_npc}")
    logger.info(f"Minimum Acceleration Ego: {min_ego}, NPC: {min_npc}")

    # Plot the histogram of accelerations
    _, axes = plt.subplots(1, 2, figsize=(18, 6))

    axes[0].hist(all_npc_accelerations, bins=60, density=True, alpha=0.6, color="b")
    mu_npc, std_npc = norm.fit(all_npc_accelerations)
    x_npc = np.linspace(min_npc, max_npc, 100)
    p_npc = norm.pdf(x_npc, mu_npc, std_npc)
    axes[0].plot(x_npc, p_npc, "k", linewidth=2)
    axes[0].set_title("Probability Distribution of NPC Pedestrian Accelerations")
    axes[0].set_xlabel("Acceleration")
    axes[0].set_ylabel("Probability Density")

    axes[1].hist(ego_accelerations, bins=60, density=True, alpha=0.6, color="r")
    mu_ego, std_ego = norm.fit(ego_accelerations)
    x_ego = np.linspace(min_ego, max_ego, 100)
    p_ego = norm.pdf(x_ego, mu_ego, std_ego)
    axes[1].plot(x_ego, p_ego, "k", linewidth=2)
    axes[1].set_title("Probability Distribution of Ego Pedestrian Accelerations")
    axes[1].set_xlabel("Acceleration")
    axes[1].set_ylabel("Probability Density")

    plt.tight_layout()

    # Prepare filename and save plot
    if unique_id:
        filename = (
            f"robot_sf/data_analysis/plots/acceleration_distribution_comparison_{unique_id}.png"
        )
    else:
        filename = "robot_sf/data_analysis/plots/acceleration_distribution_comparison.png"

    save_plot(filename, None, interactive)


def subplot_velocity_distribution_with_positions(
    ped_positions_array: np.ndarray,
    interactive: bool = False,
    unique_id: str = None,
    map_def: MapDefinition = None,
):
    """
    Calculate and plot the probability distribution of the velocity of all pedestrians,
    and plot the positions of NPC pedestrians color-coded by their velocities.

    Args:
        ped_position_array (np.ndarray): shape: (timesteps, num_pedestrians, 2)
        interactive (bool): If True, show the plot interactively
        unique_id (str): Unique identifier for the plot filename, usually the timestamp
        map_def (MapDefinition, optional): Map definition to plot obstacles
    """
    _, num_pedestrians, _ = ped_positions_array.shape

    all_npc_velocities = []
    all_npc_positions = []

    for ped_idx in range(num_pedestrians):
        x_vals = ped_positions_array[:, ped_idx, 0]
        y_vals = ped_positions_array[:, ped_idx, 1]

        distances = np.sqrt(np.diff(x_vals) ** 2 + np.diff(y_vals) ** 2)
        start_idx = 0
        for i, dist in enumerate(distances):
            if dist > TRAJECTORY_DISCONTINUITY_THRESHOLD:
                velocities = calculate_velocity(
                    x_vals[start_idx : i + 1], y_vals[start_idx : i + 1]
                )
                all_npc_velocities.extend(velocities)
                all_npc_positions.extend(
                    zip(x_vals[start_idx + 1 : i + 1], y_vals[start_idx + 1 : i + 1])
                )
                start_idx = i + 1

        velocities = calculate_velocity(x_vals[start_idx:], y_vals[start_idx:])
        all_npc_velocities.extend(velocities)
        all_npc_positions.extend(zip(x_vals[start_idx + 1 :], y_vals[start_idx + 1 :]))

    max_velocity = max(all_npc_velocities)
    logger.info(f"Maximum Velocity: {max_velocity}")

    # Plot the histogram of velocities
    fig, axes = plt.subplots(1, 2, figsize=(27, 6))

    axes[0].hist(all_npc_velocities, bins=60, density=True, alpha=0.6, color="b")
    mu_npc, std_npc = norm.fit(all_npc_velocities)
    x_npc = np.linspace(0, max_velocity, 100)
    p_npc = norm.pdf(x_npc, mu_npc, std_npc)
    axes[0].plot(x_npc, p_npc, "k", linewidth=2)
    axes[0].set_title("Probability Distribution of NPC Pedestrian Velocities")
    axes[0].set_xlabel("Velocity")
    axes[0].set_ylabel("Probability Density")

    # Plot the positions of NPC pedestrians color-coded by their velocities
    all_npc_positions = np.array(all_npc_positions)
    scatter = axes[1].scatter(
        all_npc_positions[:, 0],
        all_npc_positions[:, 1],
        c=all_npc_velocities,
        cmap="viridis",
        alpha=0.6,
    )
    axes[1].set_title("NPC Pedestrian Positions Color-Coded by Velocity")
    axes[1].set_xlabel("X Position")
    axes[1].set_ylabel("Y Position")
    axes[1].invert_yaxis()
    if map_def:
        map_def.plot_map_obstacles(axes[1])
    fig.colorbar(scatter, ax=axes[1], label="Velocity")

    plt.tight_layout()

    # Prepare filename and save plot
    if unique_id:
        filename = (
            f"robot_sf/data_analysis/plots/velocity_distribution_with_positions_{unique_id}.png"
        )
    else:
        filename = "robot_sf/data_analysis/plots/velocity_distribution_with_positions.png"

    save_plot(filename, "NPC Pedestrian Positions Color-Coded by Velocity", interactive)


def main():
    # filename = "robot_sf/data_analysis/datasets/2025-02-06_10-24-12.json"
    # filename = "robot_sf/data_analysis/datasets/2025-01-16_11-47-44.json"
    filename = "robot_sf/data_analysis/datasets/2025-03-06_11-10-28.json"

    unique_id = extract_timestamp(filename)

    ped_positions_array = extract_key_from_json_as_ndarray(filename, "pedestrian_positions")

    # plot_single_splitted_traj(
    #     ped_positions_array, ped_idx=10, interactive=True, unique_id=unique_id
    # )
    plot_all_splitted_traj(ped_positions_array, interactive=True, unique_id=unique_id)
    # subplot_single_splitted_traj_acc(ped_positions_array, ped_idx=3)
    # plot_acceleration_distribution(ped_positions_array)
    # plot_velocity_distribution(ped_positions_array)

    ego_positions = np.array([item[0] for item in extract_key_from_json(filename, "ego_ped_pose")])

    subplot_velocity_distribution_with_ego_ped(ped_positions_array, ego_positions)
    # subplot_acceleration_distribution(ped_positions_array, ego_positions)
    # subplot_velocity_distribution_with_positions(ped_positions_array)


if __name__ == "__main__":
    main()
