"""
Anaylsis of the data recorded from the simulation.
"""

import numpy as np
from typing import List
from loguru import logger

from robot_sf.render.sim_view import VisualizableSimState


def extract_pedestrian_positions(states: List[VisualizableSimState]) -> np.ndarray:
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
