"""
This module provides functions to perform Kernel Density Estimation (KDE) on pedestrian data
extracted from JSON datasets.

Key Features:
    - Plot KDE on the whole map
    - Plot KDE on X and Y axes only
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KernelDensity

from robot_sf.data_analysis.extract_json_from_pickle import (
    extract_key_from_json,
    extract_key_from_json_as_ndarray,
    extract_timestamp,
)
from robot_sf.data_analysis.plot_utils import save_plot
from robot_sf.nav.map_config import MapDefinition


def plot_kde_on_map(
    ped_positions_array: np.ndarray,
    bandwidth: float = 1.0,
    interactive: bool = False,
    unique_id: str = None,
    map_def: MapDefinition = None,
):
    """
    Plot the Kernel Density Estimation of pedestrian positions on a map.

    Args:
        ped_position_array (np.ndarray): shape: (timesteps, num_pedestrians, 2)
        bandwidth (float): The bandwidth of the kernel density estimator (Controls the smoothness).
        interactive (bool): If True, show the plot interactively.
        unique_id (str): Unique identifier for the plot filename, usually the timestamp
        map_def (MapDefinition, optional): Map definition to plot obstacles
    """
    peds_data = ped_positions_array.reshape(-1, 2)

    # Fit the KernelDensity model (with Gaussian kernel)
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)  # Adjust bandwidth as needed
    kde.fit(peds_data)

    # Create a grid of points to evaluate the density on
    x_min, x_max = peds_data[:, 0].min() - 1, peds_data[:, 0].max() + 1
    y_min, y_max = peds_data[:, 1].min() - 1, peds_data[:, 1].max() + 1
    x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    # Stack the grid points into a shape suitable for the model (n_samples, 2)
    grid_points = np.vstack([x_grid.ravel(), y_grid.ravel()]).T

    # Evaluate the density at the grid points
    log_density = kde.score_samples(grid_points)

    # Convert log density to actual density values
    density = np.exp(log_density).reshape(x_grid.shape)

    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(
        density, extent=(x_min, x_max, y_min, y_max), origin="lower", cmap="viridis", aspect="auto"
    )
    fig.colorbar(im, ax=ax, label="Density")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.invert_yaxis()

    # Plot map obstacles if map_def is provided
    if map_def is not None:
        map_def.plot_map_obstacles(ax)

    # Prepare filename
    if unique_id:
        filename = f"robot_sf/data_analysis/plots/kde_on_map_{unique_id}.png"
    else:
        filename = "robot_sf/data_analysis/plots/kde_on_map.png"

    # Save the plot
    save_plot(filename, "Kernel Density Estimation", interactive)


def perform_kde_on_axis(data: np.ndarray, bandwidth=0.1):
    """
    Perform Kernel Density Estimation on a 1D axis.

    Args:
        data (np.ndarray): shape: (n_samples, 1)
        bandwidth (float): The bandwidth of the kernel density estimator (Controls the smoothness).
    """
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    kde.fit(data)

    # Create a grid of points to evaluate the density on
    axis_min, axis_max = data.min() - 1, data.max() + 1
    axis_grid = np.linspace(axis_min, axis_max, 100).reshape(-1, 1)

    # Evaluate the density at the grid points
    log_density = kde.score_samples(axis_grid)

    # Convert log density to actual density values
    density = np.exp(log_density)

    return axis_grid, density


def plot_kde_in_x_y(
    ped_positions_array: np.ndarray,
    ego_data: np.ndarray,
    bandwidth: float = 0.1,
    interactive: bool = False,
    unique_id: str = None,
):
    """
    Plot the Kernel Density Estimation of npc and ego positions in X and Y axes.

    Args:
        ped_position_array (np.ndarray): shape: (timesteps, num_pedestrians, 2)
        ego_data (np.ndarray): shape: (timesteps, 2)
        bandwidth (float): The bandwidth of the kernel density estimator (Controls the smoothness).
        interactive (bool): If True, show the plot interactively.
        unique_id (str): Unique identifier for the plot filename, usually the timestamp
    """
    peds_data = ped_positions_array.reshape(-1, 2)
    # Perform KDE on npc data x positions
    x_positions = peds_data[:, 0].reshape(-1, 1)
    x_grid_npc, density_npc_x = perform_kde_on_axis(x_positions, bandwidth)

    # Perform KDE on ego_pedestrian x positions
    x_positions_ego = ego_data[:, 0].reshape(-1, 1)
    x_grid_ego, density_ego_x = perform_kde_on_axis(x_positions_ego, bandwidth)

    # Perform KDE on npc data y positions
    y_positions = peds_data[:, 1].reshape(-1, 1)
    y_grid_npc, density_npc_y = perform_kde_on_axis(y_positions, bandwidth)

    # Perform KDE on ego_pedestrian y positions
    y_positions_ego = ego_data[:, 1].reshape(-1, 1)
    y_grid_ego, density_ego_y = perform_kde_on_axis(y_positions_ego, bandwidth)

    # Plotting the results side by side
    _, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Plot npc data x density
    axes[0].plot(x_grid_npc, density_npc_x, label="Npc Data Density (X Position)")
    axes[0].plot(
        x_grid_ego, density_ego_x, label="Ego Pedestrian Density (X Position)", linestyle="--"
    )
    axes[0].set_title("Npc Data and Ego Pedestrian Density (X Position)")
    axes[0].set_xlabel("X Position")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    # Plot npc data y density
    axes[1].plot(y_grid_npc, density_npc_y, label="Npc Data Density (Y Position)", color="orange")
    axes[1].plot(
        y_grid_ego,
        density_ego_y,
        label="Ego Pedestrian Density (Y Position)",
        color="red",
        linestyle="--",
    )
    axes[1].set_title("Npc Data and Ego Pedestrian Density (Y Position)")
    axes[1].set_xlabel("Y Position")
    axes[1].set_ylabel("Density")
    axes[1].legend()

    # Prepare filename
    if unique_id:
        filename = f"robot_sf/data_analysis/plots/kde_xy_ego_npc_{unique_id}.png"
    else:
        filename = "robot_sf/data_analysis/plots/kde_xy_ego_npc.png"

    # Save the plot using our utility function
    save_plot(filename, None, interactive)


def main():
    # filename = "robot_sf/data_analysis/datasets/2025-02-06_10-24-12.json"
    filename = "robot_sf/data_analysis/datasets/2025-01-16_11-47-44.json"
    unique_id = extract_timestamp(filename)

    pedestrian_pos = extract_key_from_json_as_ndarray(filename, "pedestrian_positions")

    plot_kde_on_map(pedestrian_pos, interactive=True, unique_id=unique_id)

    ego_data = np.array([item[0] for item in extract_key_from_json(filename, "ego_ped_pose")])

    plot_kde_in_x_y(pedestrian_pos, ego_data, interactive=True, unique_id=unique_id)


if __name__ == "__main__":
    main()
