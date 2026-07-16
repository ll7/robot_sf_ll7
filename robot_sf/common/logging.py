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
    >>> configure_logging(verbose=True)
    >>> logger.info("Processing started")

Usage (in modules):
    >>> from loguru import logger
    >>> logger.info("Processing", seed=42, variant="demo")
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Callable


def safe_sink(
    stream: Any,
    *,
    closed_message: str = "I/O operation on closed file",
) -> Callable[[str], None]:
    """Wrap a writable sink so writes after *stream* closes never raise.

    Loguru sinks that bind to a transient stream (a ``capsys``/``capfd``
    captured ``sys.stdout``/``sys.stderr``, or an explicitly ``close()``-d file
    object) raise ``ValueError: I/O operation on closed file`` when a record is
    flushed after pytest tears the capture stream down. Under ``pytest-xdist``
    that diagnostic is emitted by benchmark workers during teardown and is noisy
    without being actionable.

    Wrap the sink callable passed to ``logger.add`` with this helper to swallow
    the closed-stream write while preserving every ordinary log write, so the
    sink lifecycle is safe across teardown without hiding unrelated logging
    failures (e.g. real ``OSError`` from a live sink still propagates).

    Args:
        stream: The underlying writable object the sink writes to. Its ``closed``
            attribute (if present) is consulted before each write; the write is
            skipped once the stream is closed.
        closed_message: Substring matched against ``ValueError`` messages so only
            the benign "closed file" class is swallowed.

    Returns:
        A sink callable compatible with ``logger.add``.

    Example:
        >>> handle = logger.add(safe_sink(sys.stdout), format="{message}")
    """

    def _sink(message: str) -> None:
        closed = getattr(stream, "closed", False)
        if closed:
            return
        try:
            stream.write(message)
        except ValueError as exc:
            if closed_message and closed_message in str(exc):
                # The capture stream was closed during teardown; the log
                # record is intentionally dropped rather than raised.
                return
            raise

    return _sink


def configure_logging(verbose: bool = False) -> None:
    """Configure the global loguru logger.

    <https://loguru.readthedocs.io/en/stable/>

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
        "<level>{level: <7}</level>| "
        "<level>{message}</level> | "
        "<dim><cyan>{file}:{line}</cyan></dim> | "
        "<dim><cyan>{elapsed}</cyan></dim>"
    )

    logger.add(
        sys.stderr,
        format=log_format,
        level="DEBUG" if verbose else "INFO",
        colorize=True,
        backtrace=True if verbose else False,
        diagnose=True if verbose else False,
    )

    # Define a very low-importance level for high-volume debug chatter.
    try:
        logger.level("SPAM")
    except ValueError:
        logger.level("SPAM", no=5, color="<dim><white>")

    # Configure log level colors
    logger.level("DEBUG", color="<dim><white>")
    logger.level("SUCCESS", color="<fg #00dd00><bold>", icon="✅")
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
