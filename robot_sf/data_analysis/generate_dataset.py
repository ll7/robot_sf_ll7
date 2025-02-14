"""
Module for converting the latest simulation recording from pickle to JSON format.

This module performs the following:
- Finds the most recent recording file in the 'recordings' folder.
- Loads the simulation states from that file.
- Converts the simulation states into a JSON serializable format.
- Saves the converted data as a JSON file named after the recording's timestamp.
"""

import os
import json
import re
from pathlib import Path

import numpy as np
from loguru import logger

from robot_sf.render.playback_recording import load_states
from robot_sf.render.sim_view import VisualizableSimState, VisualizableAction


def run():
    """
    Main entry point of the module.

    Converts the most recent recording file from pickle format to a JSON dataset.
    """
    filename = get_file()
    save_to_json(filename)


def save_to_json(filename_pkl: str):
    """
    Save simulation states from a pickle recording file to a JSON file.

    The JSON file is named based on the recording timestamp extracted from the filename.
    If no timestamp can be extracted, 'unknown' is used.

    Args:
        filename_pkl (str): The path to the recorded data in pickle format (*.pkl).

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    # Verify that the file exists
    if not os.path.exists(filename_pkl):
        raise FileNotFoundError(f"File {filename_pkl} not found!")

    # Extract timestamp from filename using regular expression
    match = re.search(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", str(filename_pkl))
    timestamp = match.group(1) if match else "unknown"
    filename_json = f"robot_sf/data_analysis/datasets/{timestamp}.json"

    # Load simulation states and map definition, which is not used
    states, _ = load_states(filename_pkl)

    # Write the states into a JSON file using a custom serializer
    with open(filename_json, "w", encoding="utf-8") as file:
        json.dump(states, file, indent=4, default=convert_to_serializable)

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
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
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
    with open(filename, "r", encoding="utf-8") as file:
        data = json.load(file)

    return [item[key] for item in data]


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


def get_file() -> Path:
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


if __name__ == "__main__":
    run()
