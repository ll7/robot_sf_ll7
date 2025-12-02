"""Module play_recordings auto-generated docstring."""

from functools import lru_cache
from pathlib import Path

from loguru import logger

from robot_sf.common.artifact_paths import ensure_canonical_tree, get_artifact_category_path
from robot_sf.render.playback_recording import load_states_and_visualize

logger.info("Play recordings")


@lru_cache(maxsize=1)
def _recordings_dir() -> Path:
    """Return the canonical recordings directory."""

    ensure_canonical_tree(categories=("recordings",))
    return get_artifact_category_path("recordings")


def get_latest_file():
    """Get the latest recorded file."""

    directory = _recordings_dir()
    filename = max(directory.iterdir(), key=lambda path: path.stat().st_ctime)
    return filename


def get_all_files():
    """Get a list of all recorded files sorted alphabetically."""
    directory = _recordings_dir()
    return sorted(
        [path for path in directory.iterdir() if path.name != "README.md"],
        key=lambda path: path.name,
    )


def get_specific_file(filename: str):
    """Get specific recorded file."""
    return _recordings_dir() / filename


def main():
    """Main.

    Returns:
        Any: Auto-generated placeholder description.
    """
    # Load the states from the file and view the recording
    # load_states_and_visualize(get_latest_file())
    # View all files
    for file in get_all_files():
        load_states_and_visualize(file)


if __name__ == "__main__":
    main()
