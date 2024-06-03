import os
import pickle
import tempfile

from robot_sf.render.playback_recording import load_states, visualize_states
from robot_sf.render.sim_view import VisualizableSimState

def test_load_and_visualize_states():
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp:
        # Create a list of VisualizableSimState instances
        states = [VisualizableSimState() for _ in range(10)] # replace with your state creation logic

        # Save the states to the file
        pickle.dump(states, temp)

    # Load the states from the file
    loaded_states = load_states(temp.name)

    # Check that the loaded states match the original states
    assert loaded_states == states

    # Visualize the states
    visualize_states(loaded_states)  # this line is not necessary for the test, but it demonstrates how to use visualize_states

    # Delete the temporary file
    os.remove(temp.name)
