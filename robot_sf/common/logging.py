"""Centralized logging configuration for robot_sf.

This module provides unified logging setup using Loguru as the canonical
logging facade (Constitution Principle XII). A single configure_logging()
function configures the global logger for all submodules.

Key Features:
    - Structured logging with context fields
    - File:line format for VS Code clickable links
    - Conditional verbosity for debug mode
    - Global configuration applied once at startup

Usage (in scripts/examples):
    >>> from robot_sf.common.logging import configure_logging
    >>> from loguru import logger
    >>> configure_logging(verbose=args.verbose)
    >>> logger.info("Processing started")

Usage (in modules):
    >>> from loguru import logger
    >>> logger.info("Processing", seed=42, variant="demo")
"""

from __future__ import annotations

import sys

from loguru import logger


def configure_logging(verbose: bool = False) -> None:
    """Configure the global loguru logger.

    # <https://loguru.readthedocs.io/en/stable/>

    Call this once at application startup to configure logging for all modules.
    After this, use `from loguru import logger` everywhere and the configuration
    will be applied automatically.

    Args:
        verbose: If True, enable DEBUG level; if False, use INFO level.

    Note:
        This function is idempotent—safe to call multiple times.
        File:line format is used for VS Code terminal link clickability.

    Example:
        >>> from robot_sf.common.logging import configure_logging
        >>> configure_logging(verbose=True)  # Enable debug output
    """
    logger.remove()  # Remove default handler

    log_format = (
        # "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <7}</level>| "
        # "<level>{level.icon} </level>"
        "<dim><cyan>{file}:{line}</cyan></dim> | "
        "<level>{message}</level>"
    )

    logger.add(
        sys.stderr,
        format=log_format,
        level="DEBUG" if verbose else "INFO",
        colorize=True,
        backtrace=True if verbose else False,
        diagnose=True if verbose else False,
    )

    # Configure log level colors
    logger.level("DEBUG", color="<dim><white>")
    logger.level("SUCCESS", color="<fg #00ff00><bold>", icon="✅")
    logger.level("ERROR", color="<fg #ff0000><bold>", icon="❌")
    logger.level("WARNING", color="<fg #ffff00><bold>", icon="⚠️")


def get_logger(name: str):
    """Get a logger instance bound to a module.

    This is a convenience wrapper for binding module context to loguru.
    After configure_logging() is called once at startup, this binding
    works globally.

    Args:
        name: Module name (typically __name__)

    Returns:
        Loguru logger bound to the module context

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing metrics", seed=42)
    """
    return logger.bind(module=name)
