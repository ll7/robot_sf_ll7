"""Logging configuration for the research reporting module.

This module provides centralized logging setup using Loguru as the
canonical logging facade (per Constitution Principle XII).

Key Features:
    - Structured logging with context fields
    - Conditional verbosity for debug mode
    - Per-module loggers with appropriate levels
    - Integration with existing robot_sf logging patterns

Usage:
    >>> from robot_sf.research.logging_config import get_logger
    >>> logger = get_logger(__name__)
    >>> logger.info("Report generation started", experiment="demo", seeds=3)
"""

from __future__ import annotations

import sys
from typing import Any

from loguru import logger


def configure_research_logging(level: str = "INFO", enable_debug: bool = False) -> None:
    """Configure logging for research module.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        enable_debug: If True, enable debug-level messages for research module

    Note:
        This function is idempotent - safe to call multiple times.
    """
    # Remove default logger if present
    logger.remove()

    # Add console handler with appropriate format
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    logger.add(
        sys.stderr,
        format=log_format,
        level="DEBUG" if enable_debug else level,
        colorize=True,
    )


def get_logger(name: str) -> Any:
    """Get a logger instance for the specified module.

    Args:
        name: Module name (typically __name__)

    Returns:
        Loguru logger bound to the module context

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing metrics", seed=42, variant_id="bc10_ds200")
    """
    return logger.bind(module=name)


# Default configuration on module import
configure_research_logging()
