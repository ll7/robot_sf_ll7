"""
Demo script to showcase trajectory visualization in interactive playback.

This script demonstrates the new trajectory visualization feature that allows
displaying the movement trails of entities (robot, pedestrians, ego pedestrian)
during playback of recorded simulation states.

Features demonstrated:
- Toggle trajectory display on/off (V key)
- Adjust trajectory trail length (B/C keys)
- Clear trajectory history (X key)
- Different colors for different entity types

Controls:
- V: Toggle trajectory display
- B: Increase trail length
- C: Decrease trail length
- X: Clear trajectories
- Space: Play/pause
- Period (.): Next frame
- Comma (,): Previous frame
"""

import sys
from pathlib import Path

from loguru import logger

from robot_sf.render.interactive_playback import InteractivePlayback, load_states


def demonstrate_trajectory_visualization(recording_file: str):
    """
    Load and demonstrate trajectory visualization with an interactive playback.

    Args:
        recording_file: Path to the pickle file containing recorded states
    """
    try:
        logger.info(f"Loading recording from: {recording_file}")
        states, map_def = load_states(recording_file)

        logger.info(f"Loaded {len(states)} states for trajectory demo")
        logger.info("Starting interactive playback with trajectory visualization...")
        logger.info("Press 'V' to toggle trajectory display")
        logger.info("Press 'B' or 'C' to adjust trail length")
        logger.info("Press 'X' to clear trajectories")
        logger.info("Press 'H' for full help")

        # Create interactive playback with trajectory visualization enabled
        playback = InteractivePlayback(states, map_def)

        # Enable trajectories by default for the demo
        playback.show_trajectories = True
        # Apply trail length using public API so existing deques are reconfigured
        playback.set_trail_length(50)  # Start with moderate trail length

        logger.info("Trajectory visualization enabled by default")
        logger.info(f"Initial trail length: {playback.max_trajectory_length}")

        # Run the interactive playback
        playback.run()

    except FileNotFoundError:
        logger.error(f"Recording file not found: {recording_file}")
        logger.info("Please provide a valid recording file path")
    except Exception as e:
        logger.error(f"Error during trajectory demo: {e}")


def main():
    """Main function for trajectory visualization demo."""
    if len(sys.argv) > 1:
        recording_file = sys.argv[1]
    else:
        # Try to find a recording file in common locations
        potential_files = [
            "examples/recordings/2024-12-06_15-39-44.pkl",
            "recordings/latest.pkl",
            "test_pygame/recordings/demo.pkl",
        ]

        recording_file = None
        for file_path in potential_files:
            if Path(file_path).exists():
                recording_file = file_path
                break

        if not recording_file:
            logger.error("No recording file specified and no default files found")
            logger.info("Usage: python trajectory_demo.py <recording_file.pkl>")
            logger.info("Available locations to check:")
            for file_path in potential_files:
                logger.info(f"  - {file_path}")
            return

    demonstrate_trajectory_visualization(recording_file)


if __name__ == "__main__":
    main()
