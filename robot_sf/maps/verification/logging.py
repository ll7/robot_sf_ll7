"""Logging helpers for map verification (deprecated).

This module is deprecated. Use robot_sf.common.logging instead.

For backwards compatibility, this module re-exports the unified logging
function from robot_sf.common.logging.

New code should use:
    >>> from robot_sf.common.logging import configure_logging
"""

from __future__ import annotations

# Import the unified logging configuration function
from robot_sf.common.logging import configure_logging  # noqa: F401
