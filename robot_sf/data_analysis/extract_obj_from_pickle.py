"""
This module provides functions to extract objects from pickle files.

Key Features:
    - Extract specific objects from VisualizableSimStates
"""

import os

import numpy as np
from loguru import logger

from robot_sf.data_analysis.plot_dataset import (
    plot_all_npc_ped_positions,
    plot_all_npc_ped_velocities,
    plot_ego_ped_acceleration,
    plot_ego_ped_velocity,
)
from robot_sf.data_analysis.plot_kernel_density import plot_kde_on_map
from robot_sf.data_analysis.plot_npc_trajectory import (
    plot_acceleration_distribution,
    plot_all_splitted_traj,
    plot_single_splitted_traj,
    plot_velocity_distribution,
    subplot_single_splitted_traj_acc,
    velocity_colorcoded_with_positions,
)
from robot_sf.nav.map_config import MapDefinition
from robot_sf.render.playback_recording import load_states
from robot_sf.render.sim_view import VisualizableSimState


def extract_ped_positions(sim_states: list[VisualizableSimState]) -> np.ndarray:
    """
    Extract pedestrian positions from a list of simulation states.

    Args:
        sim_states (List[VisualizableSimState]): Simulation states containing pedestrian data.

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


def extract_ped_actions(sim_states: list[VisualizableSimState]) -> list[list]:
    """
    Extract pedestrian actions from a list of simulation states.

    Args:
        sim_states (List[VisualizableSimState]): Simulation states containing pedestrian action data.

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


def extract_ego_ped_acceleration(sim_states: list[VisualizableSimState]) -> list[float]:
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


def plot_all_data_pkl(
    sim_states: list[VisualizableSimState],
    map_def: MapDefinition | None = None,
    unique_id: str | None = None,
    interactive: bool = True,
):
    """
    Plot all available data from simulation states extracted from pickle file.

    Args:
        sim_states (List[VisualizableSimState]): List of simulation states
        map_def (MapDefinition): Map definition for plotting obstacles
        unique_id (str): Unique identifier for plot filenames
        interactive (bool): Whether to display plots interactively
    Returns:
        None
    """

    # Extract and plot pedestrian positions
    ped_positions = extract_ped_positions(sim_states)
    plot_all_npc_ped_positions(
        ped_positions,
        interactive=interactive,
        unique_id=unique_id,
        map_def=map_def,
    )

    # Extract and plot pedestrian velocities
    ped_actions = extract_ped_actions(sim_states)
    plot_all_npc_ped_velocities(ped_actions, interactive=interactive, unique_id=unique_id)

    # Extract and plot ego pedestrian acceleration
    ego_ped_acceleration = extract_ego_ped_acceleration(sim_states)
    plot_ego_ped_acceleration(ego_ped_acceleration, interactive=interactive, unique_id=unique_id)

    # Plot ego pedestrian velocity based on acceleration
    plot_ego_ped_velocity(ego_ped_acceleration, interactive=interactive, unique_id=unique_id)

    # Extract and plot pedestrian positions using Kernel Density Estimation
    plot_kde_on_map(
        ped_positions_array=ped_positions,
        interactive=interactive,
        map_def=map_def,
        unique_id=unique_id,
    )

    # Extract and plot NPC trajectories
    plot_single_splitted_traj(
        ped_positions_array=ped_positions,
        interactive=interactive,
        unique_id=unique_id,
        map_def=map_def,
    )

    # Pass map_def to plot_all_splitted_traj
    plot_all_splitted_traj(
        ped_positions_array=ped_positions,
        interactive=interactive,
        unique_id=unique_id,
        map_def=map_def,
    )

    subplot_single_splitted_traj_acc(
        ped_positions_array=ped_positions,
        interactive=interactive,
        unique_id=unique_id,
        map_def=map_def,
    )

    plot_velocity_distribution(
        ped_positions_array=ped_positions,
        interactive=interactive,
        unique_id=unique_id,
    )

    plot_acceleration_distribution(
        ped_positions_array=ped_positions,
        interactive=interactive,
        unique_id=unique_id,
    )

    # Extract and plot NPC velocity distribution with positions
    velocity_colorcoded_with_positions(
        ped_positions_array=ped_positions,
        interactive=interactive,
        unique_id=unique_id,
        map_def=map_def,
    )

    logger.info("All data extracted and plotted successfully")


def ensure_dir_exists(directory):
    """
    Ensure that a directory exists, creating it if necessary.

    Args:
        directory (str): Path to the directory to check/create
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")


if __name__ == "__main__":
    # Example usage
    from pathlib import Path

    from robot_sf.data_analysis.extract_json_from_pickle import extract_timestamp

    # Ensure the plots directory exists
    PLOTS_DIR = "robot_sf/data_analysis/plots"
    ensure_dir_exists(PLOTS_DIR)

    # Find the most recent recording file
    recording_dir = Path("recordings")
    if recording_dir.exists():
        latest_file = max(recording_dir.glob("*.pkl"), key=os.path.getctime, default=None)

        if latest_file:
            # Extract timestamp for unique filenames
            unique_id = extract_timestamp(str(latest_file))

            # Load states and map definition
            states, map_def = load_states(latest_file)

            # Plot all available data
            plot_all_data_pkl(states, map_def, unique_id)

            logger.info(f"Successfully extracted and plotted data from {latest_file}")
            logger.info(f"Plots saved to {os.path.abspath(PLOTS_DIR)}")
        else:
            logger.error("No recording files found in the 'recordings' directory")
    else:
        logger.error("'recordings' directory not found")
