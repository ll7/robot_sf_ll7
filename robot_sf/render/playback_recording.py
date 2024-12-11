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
    load a list of states from a file with pickle `*.pkl` format
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


def load_states_and_record_video(state_file: str, video_save_path: str, video_fps: float = 10):
    """
    load a list of states from a file and record a video
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
