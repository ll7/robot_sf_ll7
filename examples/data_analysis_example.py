"""Perfom a showcase on how to use the data analysis module."""

import os
from pathlib import Path
from typing import List

import numpy as np
from loguru import logger

from robot_sf.data_analysis.extract_json_from_pickle import (
    extract_key_from_json,
    extract_key_from_json_as_ndarray,
    extract_timestamp,
    save_to_json,
)
from robot_sf.data_analysis.extract_obj_from_pickle import (
    ensure_dir_exists,
    extract_ego_ped_acceleration,
    extract_ped_actions,
    extract_ped_positions,
)
from robot_sf.data_analysis.plot_dataset import (
    plot_all_npc_ped_positions,
    plot_all_npc_ped_velocities,
    plot_ego_ped_acceleration,
    plot_ego_ped_velocity,
)
from robot_sf.data_analysis.plot_kernel_density import plot_kde_in_x_y, plot_kde_on_map
from robot_sf.data_analysis.plot_npc_trajectory import (
    plot_acceleration_distribution,
    plot_all_splitted_traj,
    plot_single_splitted_traj,
    plot_velocity_distribution,
    subplot_acceleration_distribution,
    subplot_single_splitted_traj_acc,
    subplot_velocity_distribution_with_ego_ped,
    subplot_velocity_distribution_with_positions,
)
from robot_sf.nav.map_config import MapDefinition
from robot_sf.render.playback_recording import load_states
from robot_sf.render.sim_view import VisualizableSimState


def plot_all_data_pkl(
    sim_states: List[VisualizableSimState],
    map_def: MapDefinition = None,
    unique_id: str = None,
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
        ped_positions, interactive=interactive, unique_id=unique_id, map_def=map_def
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
        ped_positions_array=ped_positions, interactive=interactive, unique_id=unique_id
    )

    plot_acceleration_distribution(
        ped_positions_array=ped_positions, interactive=interactive, unique_id=unique_id
    )

    # Extract and plot NPC velocity distribution with positions
    subplot_velocity_distribution_with_positions(
        ped_positions_array=ped_positions,
        interactive=interactive,
        unique_id=unique_id,
        map_def=map_def,
    )

    logger.info("All data extracted and plotted successfully")


def plot_all_data_json(
    filename: str,
    unique_id: str = None,
    interactive: bool = True,
):
    """
    Plot all available data from a JSON file.

    Args:
        filename (str): Path to the JSON file
        unique_id (str): Unique identifier for plot filenames
        interactive (bool): Whether to display plots interactively
    Returns:
        None
    """
    # Extract pedestrian positions
    ped_positions_array = extract_key_from_json_as_ndarray(filename, "pedestrian_positions")

    # Extract pedestrian actions as list
    ped_actions = extract_key_from_json(filename, "ped_actions")

    # Extract ego pedestrian acceleration
    ego_ped_acceleration = [
        item["action"][0] for item in extract_key_from_json(filename, "ego_ped_action")
    ]

    # Extract ego pedestrian positions
    ego_positions = np.array([item[0] for item in extract_key_from_json(filename, "ego_ped_pose")])

    plot_all_npc_ped_positions(ped_positions_array, interactive=True, unique_id=unique_id)
    plot_all_npc_ped_velocities(ped_actions, interactive=True, unique_id=unique_id)
    plot_ego_ped_acceleration(ego_ped_acceleration, interactive=True, unique_id=unique_id)
    plot_ego_ped_velocity(ego_ped_acceleration, interactive=True, unique_id=unique_id)

    plot_kde_on_map(ped_positions_array, interactive=True, unique_id=unique_id)

    plot_kde_in_x_y(ped_positions_array, ego_positions, interactive=True, unique_id=unique_id)

    # Choose id between 0 and ped_positions_array.shape[1] - 1
    ped_idx = 0

    plot_single_splitted_traj(
        ped_positions_array, ped_idx=ped_idx, interactive=interactive, unique_id=unique_id
    )

    plot_all_splitted_traj(ped_positions_array, interactive=interactive, unique_id=unique_id)

    subplot_single_splitted_traj_acc(
        ped_positions_array, ped_idx=ped_idx, interactive=interactive, unique_id=unique_id
    )

    plot_acceleration_distribution(
        ped_positions_array, interactive=interactive, unique_id=unique_id
    )

    plot_velocity_distribution(ped_positions_array, interactive=interactive, unique_id=unique_id)

    subplot_velocity_distribution_with_ego_ped(
        ped_positions_array, ego_positions, interactive=interactive, unique_id=unique_id
    )
    subplot_acceleration_distribution(
        ped_positions_array, ego_positions, interactive=interactive, unique_id=unique_id
    )
    subplot_velocity_distribution_with_positions(
        ped_positions_array, interactive=interactive, unique_id=unique_id
    )


def show_from_pkl(filename: str, unique_id: str):
    """
    Extract and plot data from a pickle file.

    Args:
        filename (str): Path to the pickle file
        unique_id (str): Unique identifier for plot filenames
    Returns:
        None
    """
    # Load states and map definition
    states, map_def = load_states(filename)

    # Plot all available data
    plot_all_data_pkl(states, map_def, unique_id)


def show_from_json(filename: str, unique_id: str):
    """
    Convert recording file into json and plot the data.

    Args:
        filename (str): Path to the JSON file
        unique_id (str): Unique identifier for plot filenames
    Returns:
        None
    """
    dataset_dir = Path("examples/datasets")
    if dataset_dir.exists():
        # Convert recording to json
        save_to_json(filename, f"{dataset_dir}/{unique_id}.json")

        latest_file = max(dataset_dir.glob("*.json"), key=os.path.getctime, default=None)

        if latest_file:
            # Plot all available data
            plot_all_data_json(str(latest_file), unique_id)

        else:
            logger.error("No json files found in the 'examples/datasets' directory")
    else:
        logger.error("'examples/datasets' directory not found")


if __name__ == "__main__":
    # Example usage
    # Ensure the plots directory exists
    PLOTS_DIR = "robot_sf/data_analysis/plots"
    ensure_dir_exists(PLOTS_DIR)

    # Find the most recent recording file
    recording_dir = Path("examples/recordings")
    if recording_dir.exists():
        latest_file = max(recording_dir.glob("*.pkl"), key=os.path.getctime, default=None)

        if latest_file:
            unique_id = extract_timestamp(str(latest_file))

            show_from_json(str(latest_file), unique_id)

            show_from_pkl(str(latest_file), unique_id)

            logger.info(f"Successfully extracted and plotted data from {latest_file}")
            logger.info(f"Plots saved in {os.path.abspath(PLOTS_DIR)}")
        else:
            logger.error("No recording files found in the 'examples/recordings' directory")
    else:
        logger.error("'examples/recordings' directory not found")
