"""
Anaylsis of the data recorded from the simulation.
"""

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from scipy.stats import gaussian_kde

from robot_sf.nav.map_config import MapDefinition
from robot_sf.render.sim_view import VisualizableSimState


def extract_pedestrian_positions(states: list[VisualizableSimState]) -> np.ndarray:
    """Extract pedestrian positions from recorded states.

    Args:
        states (List[VisualizableSimState]):
            List of simulation states containing pedestrian position data

    Returns:
        np.ndarray: Array of shape (n, 2) containing pedestrian positions,
            where n is the total number of
            pedestrian positions across all states. Returns empty array if no valid positions found.
            Each position is represented as [x, y] coordinates.

    Raises:
        None: Errors are logged but function returns empty array instead of raising exceptions

    Notes:
        - Function validates that all positions are 2D coordinates
        - If any validation fails, returns empty numpy array and logs error
        - Concatenates pedestrian positions from all input states into single array
    """
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


def kde_plot_grid_creation(x_min, x_max, y_min, y_max, number_of_grid_points: int = 100):
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


def visualize_kde_of_pedestrians_on_map(
    pedestrian_positions: np.ndarray,
    map_def: MapDefinition,
    kde_bandwith_method: str = "scott",
):
    """Visualize KDE for pedestrian positions on a map"""

    # Get map dimensions
    x_min, x_max, y_min, y_max = map_def.get_map_bounds()
    logger.info(f"Map bounds: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}]")

    # Calculate KDE
    # gaussian_kde expects shape (n_features, n_samples) but our data is (n_samples, n_features)
    # Therefore we transpose the array from shape (n_points, 2) to (2, n_points)
    pedestrian_kde = gaussian_kde(pedestrian_positions.T, bw_method=kde_bandwith_method)

    # Create grid based on map bounds
    grid_xx, grid_yy, grid_points = kde_plot_grid_creation(x_min, x_max, y_min, y_max)

    _, ax = plt.subplots(1, 1, figsize=(6, 5))

    kde_vals = pedestrian_kde(grid_points).reshape(
        grid_xx.shape,
    )  # 5. Reshape back to 2D for plotting

    # Normalize KDE values to probabilities
    kde_vals = kde_vals / kde_vals.sum()

    # Create contour plot with colorbar
    contour = ax.contourf(grid_xx, grid_yy, kde_vals, cmap="viridis", levels=20)
    colorbar = plt.colorbar(contour, ax=ax)
    colorbar.set_label("Probability Density")

    # ax.contourf(grid_xx, grid_yy, kde_vals, cmap="viridis")

    ax.scatter(pedestrian_positions[:, 0], pedestrian_positions[:, 1], alpha=1, s=1, c="red")

    # Plot map obstacles
    # plot_map_obstacles(ax, map_def)
    map_def.plot_map_obstacles(ax)

    ax.set_title("Pedestrian Positions KDE")
    ax.axis("equal")
    ax.grid(True)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    plt.show()
