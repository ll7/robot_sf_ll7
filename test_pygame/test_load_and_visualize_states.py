"""
load a recording and play it back
"""
import loguru


from robot_sf.render.playback_recording import load_states_and_visualize

logger = loguru.logger


def test_load_and_visualize_states():
    logger.info("Testing load and visualize states")
    test_file = "test_pygame/recordings/2024-06-04_08-39-59.pkl"

    # Load the states from the file
    load_states_and_visualize(test_file)
