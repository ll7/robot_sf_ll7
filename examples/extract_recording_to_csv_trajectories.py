"""
Use the existing abilities to record the visualizable sim state.
Load a saved recording and visualize the gaussian_kde for the pedestrian position.
Afterwards, compare one pedestrian_postion with the gaussian_kde and use Kullback-Leibler divergence to compare the two.
"""

from collections import defaultdict
from robot_sf.render.playback_recording import load_states
from robot_sf.render.sim_view import VisualizableSimState
from robot_sf.nav.map_config import MapDefinition
import numpy as np
from loguru import logger
from typing import List
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import pandas as pd


def extract_pedestrian_positions(states: List[VisualizableSimState]) -> np.ndarray:
    """Extract pedestrian positions from recorded states."""
    pedestrian_positions = []

    for state in states:
        pedestrian_positions.extend(state.pedestrian_positions)

    # validate that pedestrian_positions has the shape (n, 2)
    if len(pedestrian_positions) == 0:
        logger.error("No pedestrian positions found in states")
        return np.array([])
    if not all(len(pos) == 2 for pos in pedestrian_positions):
        logger.error("Invalid pedestrian positions found in states")
        return np.array([])
    logger.info(f"Extracted {len(pedestrian_positions)} pedestrian positions")

    return np.array(pedestrian_positions)


def compute_kde(positions: np.ndarray) -> gaussian_kde:
    """Compute KDE for positions"""
    return gaussian_kde(positions.T)


def kl_divergence_kde(
    kde1: gaussian_kde, kde2: gaussian_kde, grid_points: np.ndarray
) -> float:
    """Calculate KL divergence between two KDEs"""
    p = kde1(grid_points)
    q = kde2(grid_points)
    p /= np.sum(p)
    q /= np.sum(q)
    return np.sum(p * np.log(p / q))


def compare_position_with_dataset(
    single_position: np.ndarray, dataset_positions: List[np.ndarray]
) -> float:
    """Compare single position with dataset using KL divergence"""
    # Setup evaluation grid
    grid_x, grid_y = np.linspace(-20, 20, 100), np.linspace(-20, 20, 100)
    grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)
    grid_points = np.vstack([grid_xx.ravel(), grid_yy.ravel()])

    # Compute KDEs
    single_kde = compute_kde(single_position)
    dataset_kde = compute_kde(np.concatenate(dataset_positions))

    # Visualize KDEs
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    kde_vals_single = single_kde(grid_points).reshape(grid_xx.shape)
    ax1.contourf(grid_xx, grid_yy, kde_vals_single, cmap="viridis")
    ax1.scatter(single_position[:, 0], single_position[:, 1], alpha=0.5)
    ax1.set_title("Single Position KDE")
    ax1.axis("equal")

    kde_vals_dataset = dataset_kde(grid_points).reshape(grid_xx.shape)
    ax2.contourf(grid_xx, grid_yy, kde_vals_dataset, cmap="viridis")
    for pos in dataset_positions:
        ax2.scatter(pos[:, 0], pos[:, 1], alpha=0.5)
    ax2.set_title("Dataset KDE")
    ax2.axis("equal")

    kl_div = kl_divergence_kde(single_kde, dataset_kde, grid_points)
    plt.suptitle(f"KL Divergence: {kl_div:.4f}")
    plt.tight_layout()
    plt.show()

    return kl_div


def get_map_bounds(bounds):
    """Extract minimum and maximum coordinates from map bounds array.

    Args:
        bounds (list): List of tuples containing map boundary coordinates.
                      Each tuple contains (x_start, x_end, y_start, y_end).

    Returns:
        tuple: Contains minimum and maximum coordinates as (x_min, x_max, y_min, y_max).
                - x_min (float): Minimum x coordinate
                - x_max (float): Maximum x coordinate
                - y_min (float): Minimum y coordinate
                - y_max (float): Maximum y coordinate
    """
    # Flatten list of tuples into separate x and y coordinates
    x_coords = []
    y_coords = []

    for x_start, x_end, y_start, y_end in bounds:
        x_coords.extend([x_start, x_end])
        y_coords.extend([y_start, y_end])

    # Get min/max values
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    return x_min, x_max, y_min, y_max


def kde_plot_grid_creation(
    x_min, x_max, y_min, y_max, number_of_grid_points: int = 100
):
    """
    Create a grid of points for Kernel Density Estimation (KDE) plotting.

    Parameters:
    x_min (float): Minimum value for the x-axis.
    x_max (float): Maximum value for the x-axis.
    y_min (float): Minimum value for the y-axis.
    y_max (float): Maximum value for the y-axis.
    number_of_grid_points (int, optional):
        Number of points along each axis for the grid. Default is 100.

    Returns:
    tuple: A tuple containing:
        - grid_xx (ndarray): 2D array of x coordinates for the grid.
        - grid_yy (ndarray): 2D array of y coordinates for the grid.
        - grid_points (ndarray): 2D array of grid points reshaped for KDE evaluation.
    """
    # Create 1D coordinate arrays (100 points each)
    grid_x = np.linspace(x_min, x_max, number_of_grid_points)  # [x1, x2, ..., x100]
    grid_y = np.linspace(y_min, y_max, number_of_grid_points)  # [y1, y2, ..., y100]

    # Create 2D coordinate grid (100x100 points)
    grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)
    # grid_xx shape: (100,100) - x coordinates
    # grid_yy shape: (100,100) - y coordinates

    # Reshape for KDE evaluation
    grid_points = np.vstack([grid_xx.ravel(), grid_yy.ravel()])
    # grid_points shape: (2, 10000)
    # - First row: all x coordinates
    # - Second row: all y coordinates

    return grid_xx, grid_yy, grid_points


def plot_map_obstacles(ax, map_def: MapDefinition):
    """Plot map obstacles on map"""
    for obstacle in map_def.obstacles:
        vertices = np.array(obstacle.vertices)
        ax.fill(vertices[:, 0], vertices[:, 1], "black")


def visualize_kde_of_pedestrians_on_map(
    pedestrian_positions: np.ndarray, map_def: MapDefinition
):
    """Visualize KDE for pedestrian positions on a map"""

    # Get map dimensions
    x_min, x_max, y_min, y_max = get_map_bounds(map_def.bounds)
    logger.info(f"Map bounds: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}]")

    # Calculate KDE
    # gaussian_kde expects shape (n_features, n_samples) but our data is (n_samples, n_features)
    # Therefore we transpose the array from shape (n_points, 2) to (2, n_points)
    pedestrian_kde = gaussian_kde(pedestrian_positions.T)

    # Create grid based on map bounds
    grid_xx, grid_yy, grid_points = kde_plot_grid_creation(x_min, x_max, y_min, y_max)

    _, ax = plt.subplots(1, 1, figsize=(6, 5))

    kde_vals = pedestrian_kde(grid_points).reshape(
        grid_xx.shape
    )  # 5. Reshape back to 2D for plotting
    ax.contourf(grid_xx, grid_yy, kde_vals, cmap="viridis")

    ax.scatter(
        pedestrian_positions[:, 0], pedestrian_positions[:, 1], alpha=1, s=10, c="black"
    )

    # Plot map obstacles
    plot_map_obstacles(ax, map_def)

    ax.set_title("Pedestrian Positions KDE")
    ax.axis("equal")
    ax.grid(True)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    plt.show()


def main():
    try:
        # Load pedestrian positions (implement according to your data source)
        states, map_def = load_states("examples/recordings/2024-12-06_15-39-44.pkl")

        pedestrian_positions = extract_pedestrian_positions(states)

        if len(pedestrian_positions) < 2:
            logger.error("Need at least 2 positions for comparison")
            return

        logger.info(f"Pedestrian positions shape: {pedestrian_positions.shape}")


        # Visualize KDE
        visualize_kde_of_pedestrians_on_map(pedestrian_positions, map_def)

        # # Split into dataset and single pedestrian position
        # dataset_pedestrian_positions = pedestrian_positions[:-1]
        # single_pedestrian_position = pedestrian_positions[-1]

        # # Compare pedestrian positions
        # kl_div = compare_position_with_dataset(
        #     single_pedestrian_position, dataset_pedestrian_positions
        # )
        # logger.info(f"KL Divergence: {kl_div:.4f}")

    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()
