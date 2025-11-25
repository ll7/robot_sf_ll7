"""Logging helpers for map verification.

Centralizes Loguru configuration so the CLI and library code share
consistent formatting and levels.
"""

from __future__ import annotations

import sys

from loguru import logger

DEFAULT_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | <level>{message}</level>"
)
VERBOSE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> | "
    "<level>{message}</level>"
)


def configure_logging(verbose: bool = False) -> None:
    """Configure Loguru handlers for verification.

    Parameters
    ----------
    verbose : bool
        Enable DEBUG level and expanded format if True.
    """
    logger.remove()
    logger.add(
        sys.stderr,
        format=VERBOSE_FORMAT if verbose else DEFAULT_FORMAT,
        level="DEBUG" if verbose else "INFO",
    )
