"""
Use a recording and plot the kde of the pedestrian postions on a map.
"""

from loguru import logger


from robot_sf.render.playback_recording import load_states
from robot_sf.data_analysis.recording_analysis import (
    extract_pedestrian_positions,
    visualize_kde_of_pedestrians_on_map,
)


def main():
    try:
        # Load pedestrian positions (implement according to your data source)
        states, map_def = load_states("examples/recordings/2024-12-06_15-39-44.pkl")

        pedestrian_positions = extract_pedestrian_positions(states)

        if len(pedestrian_positions) < 2:
            logger.error("Need at least 2 positions for comparison")
            return

        logger.info(f"Pedestrian positions shape: {pedestrian_positions.shape}")

        # Visualize KDE
        visualize_kde_of_pedestrians_on_map(pedestrian_positions, map_def)

    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()
