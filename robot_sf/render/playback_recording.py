"""Robot Simulation State Playback Module

This module provides functionality to replay and visualize recorded robot simulation states.
It supports both interactive visualization and video recording of simulation playbacks.

Key Features:
    - Load simulation states from pickle files
    - Validate simulation state data
    - Visualize states interactively
    - Record simulation playback as video
    - Support for map definitions and robot states

Notes:
    The pickle files should contain a tuple of (states, map_def) where:
    - states: List[VisualizableSimState] - Sequence of simulation states
    - map_def: MapDefinition - Configuration of the simulation environment
"""

import os
import pickle
from typing import List

import loguru

from robot_sf.nav.map_config import MapDefinition
from robot_sf.render.sim_view import SimulationView, VisualizableSimState

logger = loguru.logger


def load_states(filename: str, return_dict: bool = False):
    """
    Load simulation states from a pickle file.

    This function supports both the new dictionary format and the legacy tuple format.

    Args:
        filename (str): Path to the pickle file containing the states
        return_dict (bool): If True, returns the complete data dictionary
        instead of just states and map_def

    Returns:
        If return_dict=False (default):
            Tuple[List[VisualizableSimState], MapDefinition]: A tuple containing states and map_def
        If return_dict=True:
            Dict: All data from the pickle file including states, map_def, metadata and rewards

    Raises:
        TypeError: If loaded states are not VisualizableSimState objects or map_def
            is not MapDefinition
    """
    # Check if the file is empty
    if os.path.getsize(filename) == 0:
        logger.error(f"File {filename} is empty")
        return ([], None) if not return_dict else {"states": [], "map_def": None}

    logger.info(f"Loading states from {filename}")

    with open(filename, "rb") as f:  # rb = read binary
        content = pickle.load(f)

    # Initialize the result dictionary
    result = {"states": [], "map_def": None, "metadata": {}, "rewards": None}

    # Handle dictionary format
    if isinstance(content, dict):
        logger.info("Detected dictionary format recording")
        result["states"] = content.get("states", [])
        result["map_def"] = content.get("map_def")
        result["metadata"] = content.get("metadata", {})
        result["rewards"] = content.get("rewards")

        # Log metadata if available
        if result["metadata"]:
            timestamp = result["metadata"].get("timestamp")
            num_states = len(result["states"])
            logger.info(f"Recording info: {num_states} states, created on {timestamp}")

    # Handle legacy tuple format
    elif isinstance(content, tuple):
        logger.info("Detected legacy tuple format recording")
        if len(content) >= 2:
            result["states"], result["map_def"] = content[0], content[1]
        else:
            logger.error(f"Invalid tuple format in {filename}")
            return ([], None) if not return_dict else {"states": [], "map_def": None}
    else:
        logger.error(f"Unknown format in {filename}")
        return ([], None) if not return_dict else {"states": [], "map_def": None}

    logger.info(f"Loaded {len(result['states'])} states")

    # Verify `states` is a list of VisualizableSimState
    if not all(isinstance(state, VisualizableSimState) for state in result["states"]):
        logger.error(f"Invalid states loaded from {filename}")
        raise TypeError(f"Invalid states loaded from {filename}")

    # Verify `map_def` is a MapDefinition
    if not isinstance(result["map_def"], MapDefinition):
        logger.error(f"Invalid map definition loaded from {filename}")
        logger.error(f"map_def: {type(result['map_def'])}")
        raise TypeError(f"Invalid map definition loaded from {filename}")

    # Return based on the return_dict flag
    return result if return_dict else (result["states"], result["map_def"])


def visualize_states(states: List[VisualizableSimState], map_def: MapDefinition):
    """
    use the SimulationView to render a list of states
    on the recorded map defintion
    """
    sim_view = SimulationView(map_def=map_def, caption="RobotSF Recording")
    for state in states:
        sim_view.render(state)

    sim_view.exit_simulation()  # to automatically close the window


def load_states_and_visualize(filename: str):
    """
    load a list of states from a file and visualize them
    """
    states, map_def = load_states(filename)
    visualize_states(states, map_def)


def load_states_and_record_video(state_file: str, video_save_path: str, video_fps: float = 10):
    """
    Load robot states from a file and create a video recording of the simulation.

    This function reads saved robot states from a file, initializes a simulation view,
    and records each state to create a video visualization of the robot's movement.

    Args:
        state_file (str): Path to the file containing saved robot states and map definition
        video_save_path (str): Path where the output video file should be saved
        video_fps (float, optional): Frames per second for the output video. Defaults to 10.

    Returns:
        None

    Note:
        The states file should contain both the robot states and map definition in a
            compatible format.
        The video will be written when the simulation view is closed via exit_simulation().

    Example:
        >>> load_states_and_record_video("states.pkl", "output.mp4", video_fps=30)
    """
    logger.info(f"Loading states from {state_file}")
    states, map_def = load_states(state_file)
    sim_view = SimulationView(
        map_def=map_def,
        caption="RobotSF Recording",
        record_video=True,
        video_path=video_save_path,
        video_fps=video_fps,
    )
    for state in states:
        sim_view.render(state)

    sim_view.exit_simulation()  # to write the video file
