"""Perform a showcase on how to use the data analysis module."""

import os
from pathlib import Path

from loguru import logger

from robot_sf.data_analysis.extract_json_from_pickle import (
    extract_timestamp,
    plot_all_data_json,
    save_to_json,
)
from robot_sf.data_analysis.extract_obj_from_pickle import (
    ensure_dir_exists,
    plot_all_data_pkl,
)
from robot_sf.render.playback_recording import load_states


def show_from_pkl(filename: str, unique_id: str):
    """
    Extract and plot data from a pickle file.

    Args:
        filename (str): Path to the pickle file
        unique_id (str): Unique identifier for plot filenames
    Returns:
        None
    """
    # Load states and map definition
    states, map_def = load_states(filename)

    # Plot all available data
    plot_all_data_pkl(states, map_def, unique_id)


def show_from_json(filename: str, unique_id: str):
    """
    Convert recording file into json and plot the data.

    Args:
        filename (str): Path to the JSON file
        unique_id (str): Unique identifier for plot filenames
    Returns:
        None
    """
    dataset_dir = Path("examples/datasets")
    if dataset_dir.exists():
        # Convert recording to json
        save_to_json(filename, f"{dataset_dir}/{unique_id}.json")

        latest_file = max(dataset_dir.glob("*.json"), key=os.path.getctime, default=None)

        if latest_file:
            # Plot all available data
            plot_all_data_json(str(latest_file), unique_id)

        else:
            logger.error("No json files found in the 'examples/datasets' directory")
    else:
        logger.error("'examples/datasets' directory not found")


if __name__ == "__main__":
    # Example usage
    # Ensure the plots directory exists
    PLOTS_DIR = "robot_sf/data_analysis/plots"
    ensure_dir_exists(PLOTS_DIR)

    # Find the most recent recording file
    recording_dir = Path("examples/recordings")
    if recording_dir.exists():
        latest_file = max(recording_dir.glob("*.pkl"), key=os.path.getctime, default=None)

        if latest_file:
            unique_id = extract_timestamp(str(latest_file))

            show_from_json(str(latest_file), unique_id)

            show_from_pkl(str(latest_file), unique_id)

            logger.info(f"Successfully extracted and plotted data from {latest_file}")
            logger.info(f"Plots saved in {os.path.abspath(PLOTS_DIR)}")
        else:
            logger.error("No recording files found in the 'examples/recordings' directory")
    else:
        logger.error("'examples/recordings' directory not found")
