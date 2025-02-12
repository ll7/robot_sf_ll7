import os
import json
import re
from pathlib import Path

import numpy as np
from loguru import logger

from robot_sf.render.playback_recording import load_states
from robot_sf.render.sim_view import VisualizableSimState, VisualizableAction


def main():
    filename = get_file()
    save_to_json(filename)


def save_to_json(filename_pkl: str):
    """
    Save the given data to a JSON file named after the recording timestamp.

    Args:
        filename_pkl (str): Filename of recorded data in pickle format. (*.pkl)

    """
    match = re.search(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", str(filename_pkl))
    if match:
        match = match.group(1)
    else:
        match = "unknown"
    filename_json = f"robot_sf/data_analysis/datasets/{match}.json"

    states, _ = load_states(filename_pkl)

    with open(filename_json, "w", encoding="utf-8") as file:
        json.dump(states, file, indent=4, default=convert_to_serializable)

    logger.info(f"Saved data to {filename_json}")


def convert_to_serializable(obj):
    """
    Recursively converts various types of objects into JSON serializable formats.

    Args:
        obj: The object to be converted. It can be an instance of VisualizableSimState, ...

    Returns:
        A JSON serializable representation of the input object.

    Raises:
        TypeError: If the object type is not supported for conversion.

    Notes:
        keys_sim_state and keys_action are used to filter out unwanted keys.
        If you want them just add them to the list.
        Keys can be found in the VisualizableSimState and VisualizableAction classes.
    """
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


def extract_key_from_json(filename: str, key: str):
    """
    Extract values associated with the given key from a JSON file.

    Args:
        filename (str): Path to the JSON file.
        key (str): The key to extract values for.

    Returns:
        list: A list of values associated with the given key.
    """
    with open(filename, "r", encoding="utf-8") as file:
        data = json.load(file)

    return [item[key] for item in data]


def get_file():
    """Get the latest recorded file."""

    filename = max(
        os.listdir("recordings"),
        key=lambda x: os.path.getctime(os.path.join("recordings", x)),
    )
    return Path("recordings", filename)


if __name__ == "__main__":
    main()
