"""
Module for converting the latest simulation recording from pickle to JSON format.

This module performs the following:
- Finds the most recent recording file in the 'recordings' folder.
- Loads the simulation states from that file.
- Converts the simulation states into a JSON serializable format.
- Saves the converted data as a JSON file named after the recording's timestamp.
"""

import json
import os
import re
from pathlib import Path

import numpy as np
from loguru import logger

from robot_sf.data_analysis.extract_obj_from_pickle import ensure_dir_exists
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
    velocity_colorcoded_with_positions,
)
from robot_sf.nav.map_config import MapDefinition
from robot_sf.nav.obstacle import Obstacle
from robot_sf.render.playback_recording import load_states
from robot_sf.render.sim_view import VisualizableAction, VisualizableSimState


def save_to_json(filename_pkl: str, filename_json: str | None = None):
    """
    Save simulation states from a pickle recording file to a JSON file.

    The JSON file is named based on the recording timestamp extracted from the filename.
    If no timestamp can be extracted, 'unknown' is used.

    Args:
        filename_pkl (str): The path to the recorded data in pickle format (*.pkl).
        filename_json (str): The path to save the converted data in JSON format (*.json).

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    # Verify that the file exists
    if not os.path.exists(filename_pkl):
        raise FileNotFoundError(f"File {filename_pkl} not found!")

    timestamp = extract_timestamp(filename_pkl)
    if not filename_json:
        logger.warning("No output filename provided. Using the timestamp as the filename.")
        filename_json = f"robot_sf/data_analysis/datasets/{timestamp}.json"

    # Load simulation states and map definition, which is not used
    states, map_def = load_states(filename_pkl)

    combined_data = {
        "states": convert_to_serializable(states),
        "map_def": convert_map_def_to_serializable(map_def),
    }

    with open(filename_json, "w", encoding="utf-8") as file:
        json.dump(combined_data, file, indent=4)

    logger.info(f"Saved data to {filename_json}")


def convert_to_serializable(obj):
    """
    Recursively convert non-JSON-serializable objects into serializable ones.

    Works with VisualizableSimState, VisualizableAction, numpy arrays, numpy scalar types,
    and nested lists or tuples. Custom conversion is applied to select attributes
    for simulation states and actions.

    Args:
        obj: The object to be converted.

    Returns:
        A JSON-serializable representation of the object.

    Raises:
        TypeError: If the object cannot be converted to a serializable format.
    Notes:
        keys_sim_state and keys_action are used to filter out unwanted keys.
        If you want them just add them to the list.
        Keys can be found in the VisualizableSimState and VisualizableAction classes.
    """
    # Only include desired keys for simulation states and actions
    keys_sim_state = [
        "timestep",
        "pedestrian_positions",
        "ped_actions",
        "ego_ped_pose",
        "ego_ped_action",
    ]
    keys_action = ["action"]

    if isinstance(obj, VisualizableSimState):
        return {
            key: convert_to_serializable(value)
            for key, value in obj.__dict__.items()
            if key in keys_sim_state
        }
    elif isinstance(obj, VisualizableAction):
        return {
            key: convert_to_serializable(value)
            for key, value in obj.__dict__.items()
            if key in keys_action
        }
    elif isinstance(obj, list | tuple):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, int | float | str | bool) or obj is None:
        return obj
    else:
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable!")


def convert_map_def_to_serializable(obj):
    """
    Recursively convert non-JSON-serializable objects into serializable ones.

    This is a extension to the convert_to_serializable() function and works with MapDefinition and
    Obstacle objects. Converting only the necessary vertices.

    Args:
        obj: The object to be converted.

    Returns:
        A JSON-serializable representation of the object.

    Raises:
        TypeError: If the object cannot be converted to a serializable format
    """
    if isinstance(obj, MapDefinition):
        return {
            key: convert_map_def_to_serializable(value)
            for key, value in obj.__dict__.items()
            if key == "obstacles"
        }
    elif isinstance(obj, list | tuple):
        return [convert_map_def_to_serializable(item) for item in obj]
    elif isinstance(obj, Obstacle):
        return {
            "obstacle_vertices": convert_to_serializable(obj.vertices),
        }
    else:
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable!")


def extract_key_from_json(filename: str, key: str) -> list:
    """
    Extract a list of values associated with a given key from a JSON file.

    Args:
        filename (str): Path to the JSON file.
        key (str): The key for which values should be extracted.

    Returns:
        list: A list of values corresponding to the key in each JSON object.
    """
    with open(filename, encoding="utf-8") as file:
        data = json.load(file)

    return [item[key] for item in data["states"]]


def extract_key_from_json_as_ndarray(filename: str, key: str) -> np.ndarray:
    """
    Extract values for a given key from a JSON file and return them as a NumPy array.

    Args:
        filename (str): Path to the JSON file.
        key (str): The key for which values should be extracted.

    Returns:
        np.ndarray: A NumPy array of values associated with the key.
    """
    data = extract_key_from_json(filename, key)
    # Convert the list of positions to a NumPy array.
    np_array = np.array(data)
    return np_array


def extract_map_def_from_json(filename: str) -> MapDefinition:
    """
    Extract the map definition from a JSON file.

    Args:
        filename (str): Path to the JSON file.

    Returns:
        MapDefinition: The map definition extracted from the JSON file.

    Notes:
        This does not recreate a valid map definition!
        Only the obstacles are extracted, which are needed for plotting.
    """
    with open(filename, encoding="utf-8") as file:
        json_data = json.load(file)

    obstacles = []
    for obstacle_data in json_data["map_def"]["obstacles"]:
        vertices = obstacle_data["obstacle_vertices"]
        obstacle = Obstacle(vertices=vertices)
        obstacles.append(obstacle)

    map_def = MapDefinition(
        width=1,
        height=1,
        obstacles=obstacles,
        robot_spawn_zones=[[(0, 0), (0, 0), (0, 0)]],
        ped_spawn_zones=[],
        robot_goal_zones=[[(0, 0), (0, 0), (0, 0)]],
        bounds=[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        robot_routes=[],
        ped_goal_zones=[],
        ped_crowded_zones=[],
        ped_routes=[],
    )

    return map_def


def get_latest_recording_file() -> Path:
    """
    Get the most recent recording file from the 'recordings' directory.

    Returns:
        Path: A Path object pointing to the latest file.
    """
    # Find the file with the most recent creation time in the recordings folder.
    filename = max(
        os.listdir("recordings"),
        key=lambda x: os.path.getctime(os.path.join("recordings", x)),
    )
    return Path("recordings", filename)


def extract_timestamp(filename: str) -> str:
    """
    Extract the timestamp from a filename.

    Args:
        filename (str): The filename from which to extract the timestamp.

    Returns:
        str: The extracted timestamp or 'unknown' if no timestamp is found.
    """
    match = re.search(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", filename)
    return match.group() if match else "unknown"


def plot_all_data_json(
    filename: str,
    unique_id: str | None = None,
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

    # Extract map definition
    map_def = extract_map_def_from_json(filename)

    # Extract ego pedestrian positions
    ego_positions = np.array([item[0] for item in extract_key_from_json(filename, "ego_ped_pose")])

    plot_all_npc_ped_positions(
        ped_positions_array,
        interactive=interactive,
        unique_id=unique_id,
        map_def=map_def,
    )
    plot_all_npc_ped_velocities(ped_actions, interactive=interactive, unique_id=unique_id)
    plot_ego_ped_acceleration(ego_ped_acceleration, interactive=interactive, unique_id=unique_id)
    plot_ego_ped_velocity(ego_ped_acceleration, interactive=interactive, unique_id=unique_id)

    plot_kde_on_map(
        ped_positions_array,
        interactive=interactive,
        unique_id=unique_id,
        map_def=map_def,
    )

    plot_kde_in_x_y(
        ped_positions_array,
        ego_positions,
        interactive=interactive,
        unique_id=unique_id,
    )

    # Choose id between 0 and ped_positions_array.shape[1] - 1
    ped_idx = 0

    plot_single_splitted_traj(
        ped_positions_array,
        ped_idx=ped_idx,
        interactive=interactive,
        unique_id=unique_id,
        map_def=map_def,
    )

    plot_all_splitted_traj(
        ped_positions_array,
        interactive=interactive,
        unique_id=unique_id,
        map_def=map_def,
    )

    subplot_single_splitted_traj_acc(
        ped_positions_array,
        ped_idx=ped_idx,
        interactive=interactive,
        unique_id=unique_id,
        map_def=map_def,
    )

    plot_acceleration_distribution(
        ped_positions_array,
        interactive=interactive,
        unique_id=unique_id,
    )

    plot_velocity_distribution(ped_positions_array, interactive=interactive, unique_id=unique_id)

    subplot_velocity_distribution_with_ego_ped(
        ped_positions_array,
        ego_positions,
        interactive=interactive,
        unique_id=unique_id,
    )
    subplot_acceleration_distribution(
        ped_positions_array,
        ego_positions,
        interactive=interactive,
        unique_id=unique_id,
    )
    velocity_colorcoded_with_positions(
        ped_positions_array,
        interactive=interactive,
        unique_id=unique_id,
        map_def=map_def,
    )


def show_from_json(filename: str, unique_id: str):
    """
    Convert recording file into json and plot the data.

    Args:
        filename (str): Path to the JSON file
        unique_id (str): Unique identifier for plot filenames
    Returns:
        None
    """
    dataset_dir = Path("robot_sf/data_analysis/datasets")
    if dataset_dir.exists():
        # Convert recording to json
        save_to_json(filename, f"{dataset_dir}/{unique_id}.json")

        latest_file = max(dataset_dir.glob("*.json"), key=os.path.getctime, default=None)

        if latest_file:
            # Plot all available data
            plot_all_data_json(str(latest_file), unique_id)

        else:
            logger.error(f"No json files found in the '{dataset_dir}' directory")
    else:
        logger.error(f"'{dataset_dir}' directory not found")


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
            show_from_json(str(latest_file), extract_timestamp(str(latest_file)))

            logger.info(f"Successfully extracted and plotted data from {latest_file}")
            logger.info(f"Plots saved to {os.path.abspath(PLOTS_DIR)}")
        else:
            logger.error("No recording files found in the 'recordings' directory")
    else:
        logger.error("'recordings' directory not found")
