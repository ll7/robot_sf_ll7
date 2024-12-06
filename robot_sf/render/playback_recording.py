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
    load a list of states from a file with pickle
    """
    # Check if the file is empty
    if os.path.getsize(filename) == 0:
        logger.error(f"File {filename} is empty")
        return []

    logger.info(f"Loading states from {filename}")
    with open(filename, "rb") as f:  # rb = read binary
        states, map_def = pickle.load(f)
    logger.info(f"Loaded {len(states)} states")
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
