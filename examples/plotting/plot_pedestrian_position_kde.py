"""
Use a recording and plot the kde of the pedestrian positions on a map.
"""

import argparse

from loguru import logger

from robot_sf.data_analysis.recording_analysis import (
    extract_pedestrian_positions,
    visualize_kde_of_pedestrians_on_map,
)
from robot_sf.render.playback_recording import load_states


def main():
    parser = argparse.ArgumentParser(description="Plot KDE of pedestrian positions.")
    parser.add_argument(
        "--recording",
        type=str,
        default="examples/recordings/2024-12-06_15-39-44.pkl",
        help="Path to the recording file",
    )
    args = parser.parse_args()
    try:
        # Load pedestrian positions (implement according to your data source)
        states, map_def = load_states(args.recording)

        pedestrian_positions = extract_pedestrian_positions(states)

        if len(pedestrian_positions) < 2:
            logger.error("Need at least 2 positions for comparison")
            return

        logger.info(f"Pedestrian positions shape: {pedestrian_positions.shape}")

        # Visualize KDE
        visualize_kde_of_pedestrians_on_map(pedestrian_positions, map_def, "silverman")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e!s}")
        raise
    except ValueError as e:
        logger.error(f"Value error: {e!s}")
        raise
    except Exception as e:
        logger.error(f"Error during execution: {e!s}")
        raise


if __name__ == "__main__":
    main()
