import os
from pathlib import Path

from loguru import logger

from robot_sf.render.playback_recording import load_states_and_visualize

logger.info("Play recordings")


def get_latest_file():
    """Get the latest recorded file."""

    filename = max(
        os.listdir("recordings"), key=lambda x: os.path.getctime(os.path.join("recordings", x))
    )
    return Path("recordings", filename)


def get_all_files():
    """Get a list of all recorded files sorted alphabetically."""
    return sorted(
        [
            Path("recordings", filename)
            for filename in os.listdir("recordings")
            if filename != "README.md"
        ]
    )


def get_specific_file(filename: str):
    """Get specific recorded file."""
    return Path("recordings", filename)


def main():
    # Load the states from the file and view the recording
    # load_states_and_visualize(get_latest_file())
    # View all files
    for file in get_all_files():
        load_states_and_visualize(file)


if __name__ == "__main__":
    main()
