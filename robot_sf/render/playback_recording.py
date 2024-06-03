"""
playback a recorded list of states
"""
import pickle
from typing import List
import loguru
from robot_sf.render.sim_view import (
    SimulationView,
    VisualizableSimState
    )

logger = loguru.logger


def load_states(filename: str) -> List[VisualizableSimState]:
    """
    load a list of states from a file with pickle
    """
    logger.info(f"Loading states from {filename}")
    with open(filename) as f:
        states = pickle.load(f)
    logger.info(f"Loaded {len(states)} states")
    return states

def visualize_states(states: List[VisualizableSimState]):
    """
    use the SimulationView to visualize a list of states
    """
    sim_view = SimulationView()
    for state in states:
        sim_view.render(state)

def load_states_and_visualize(filename: str):
    """
    load a list of states from a file and visualize them
    """
    states = load_states(filename)
    visualize_states(states)


