"""
playback a recorded list of states
"""

import os
import pickle
from typing import List
import loguru
from robot_sf.render.sim_view import SimulationView, VisualizableSimState
from robot_sf.nav.map_config import MapDefinition

logger = loguru.logger


def load_states(filename: str) -> List[VisualizableSimState]:
    """
    Load a list of states from a pickle file.

    This function reads a pickle file containing simulation states and map definition,
    performs validation checks, and returns them if valid.

    Args:
        filename (str): Path to the pickle file containing the states

    Returns:
        Tuple[List[VisualizableSimState], MapDefinition]: A tuple containing:
            - List of VisualizableSimState objects representing simulation states
            - MapDefinition object containing the map information

    Raises:
        TypeError: If loaded states are not VisualizableSimState objects or map_def
            is not MapDefinition

    Notes:
        The pickle file must contain a tuple of (states, map_def) where:
        - states is a list of VisualizableSimState objects
        - map_def is a MapDefinition object
    """
    # Check if the file is empty
    if os.path.getsize(filename) == 0:
        logger.error(f"File {filename} is empty")
        return []

    logger.info(f"Loading states from {filename}")
    with open(filename, "rb") as f:  # rb = read binary
        states, map_def = pickle.load(f)
    logger.info(f"Loaded {len(states)} states")

    # Verify `states` is a list of VisualizableSimState
    if not all(isinstance(state, VisualizableSimState) for state in states):
        logger.error(f"Invalid states loaded from {filename}")
        raise TypeError(f"Invalid states loaded from {filename}")

    # Verify `map_def` is a MapDefinition
    if not isinstance(map_def, MapDefinition):
        logger.error(f"Invalid map definition loaded from {filename}")
        logger.error(f"map_def: {type(map_def)}")
        raise TypeError(f"Invalid map definition loaded from {filename}")

    return states, map_def


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


def load_states_and_record_video(
    state_file: str, video_save_path: str, video_fps: float = 10
):
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
