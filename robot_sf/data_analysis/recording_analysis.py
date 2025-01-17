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
