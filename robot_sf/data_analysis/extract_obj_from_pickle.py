"""
This module provides functions to extract objects from pickle files.

Key Features:
    - Extract specific objects from VisualizableSimStates
"""

import os
from typing import List

import numpy as np
from loguru import logger

from robot_sf.render.sim_view import VisualizableSimState


def extract_ped_positions(sim_states: List[VisualizableSimState]) -> np.ndarray:
    """
    Extract pedestrian positions from a list of simulation states.

    Args:
        states (List[Any]): List of VisualizableSimStates containing simulation data.

    Returns:
        np.ndarray: A numpy array with shape (timesteps, num_pedestrians, 2)
            containing all pedestrian positions

    Notes:
        - The returned array is formatted to be directly usable with plot_all_npc_ped_positions
        - The function handles varying numbers of pedestrians across timesteps
    """
    # Find the maximum number of pedestrians in any timestep
    max_peds = max(
        state.pedestrian_positions.shape[0] if hasattr(state, "pedestrian_positions") else 0
        for state in sim_states
    )

    # Create an array to store positions for all timesteps
    # Shape: (timesteps, max_pedestrians, 2)
    ped_positions = np.zeros((len(sim_states), max_peds, 2))

    # Fill in the pedestrian positions for each timestep
    for t, state in enumerate(sim_states):
        if hasattr(state, "pedestrian_positions") and state.pedestrian_positions is not None:
            # Get the number of pedestrians in this timestep
            num_peds = state.pedestrian_positions.shape[0]
            # Fill in the positions for this timestep
            ped_positions[t, :num_peds] = state.pedestrian_positions

    return ped_positions


def extract_ped_actions(sim_states: List[VisualizableSimState]) -> List[List]:
    """
    Extract pedestrian actions from a list of simulation states.

    Args:
        states (List[Any]): List of VisualizableSimStates containing simulation data.

    Returns:
        List[List]: A list of lists containing the pedestrian actions at each timestep.
            Each inner list contains action pairs (start_pos, end_pos).
    """
    # Extract ped_actions
    ped_actions = []
    for state in sim_states:
        if hasattr(state, "ped_actions") and state.ped_actions is not None:
            # Convert ndarray to list of tuples
            actions = [(tuple(action[0]), tuple(action[1])) for action in state.ped_actions]
            ped_actions.append(actions)
        else:
            ped_actions.append([])

    return ped_actions


def extract_ego_ped_acceleration(sim_states: List[VisualizableSimState]) -> List[float]:
    """
    Extract ego pedestrian acceleration from a list of simulation states.

    Args:
        sim_states (List[Any]): List of VisualizableSimStates containing simulation data.

    Returns:
        List[float]: A list containing the ego pedestrian acceleration at each timestep.
    """
    # Extract ego_ped_action.action[0] (acceleration)
    accelerations = []
    for state in sim_states:
        if (
            hasattr(state, "ego_ped_action")
            and state.ego_ped_action is not None
            and hasattr(state.ego_ped_action, "action")
        ):
            accelerations.append(state.ego_ped_action.action[0])
        else:
            accelerations.append(0.0)  # Default value when no acceleration data is available

    return accelerations


def ensure_dir_exists(directory):
    """
    Ensure that a directory exists, creating it if necessary.

    Args:
        directory (str): Path to the directory to check/create
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")
