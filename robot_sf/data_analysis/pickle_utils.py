"""Shared dependency-free helpers for the recording-extraction modules.

Extracted from ``robot_sf.data_analysis.extract_obj_from_pickle`` and
``robot_sf.data_analysis.extract_json_from_pickle`` to break the static import
cycle between them (issue #6455). Both legacy modules re-export these helpers,
so every existing public import path keeps working unchanged.
"""

import os
import re

from loguru import logger


def ensure_dir_exists(directory) -> None:
    """
    Ensure that a directory exists, creating it if necessary.

    Args:
        directory (str): Path to the directory to check/create
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info("Created directory: {}", directory)


def extract_timestamp(filename: str) -> str:
    """
    Extract the timestamp from a filename.

    Args:
        filename (str): The filename from which to extract the timestamp.

    Returns:
        str: The extracted timestamp or 'unknown' if no timestamp is found.
    """
    match = re.search(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", filename)
    return match.group() if match else "unknown"
