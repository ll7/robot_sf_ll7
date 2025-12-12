"""Logging configuration for the research module (deprecated).

This module is deprecated. Use robot_sf.common.logging instead.

For backwards compatibility, this module re-exports the unified logging
functions from robot_sf.common.logging.

New code should use:
    >>> from robot_sf.common.logging import configure_logging, get_logger
"""

from __future__ import annotations

from loguru import logger


def log_seed_failure(seed: int | str | None, policy_type: str | None, reason: str) -> None:
    """Emit a standardized warning for seed-level failures."""
    logger.warning(
        "Seed run failed or missing",
        seed=seed,
        policy_type=policy_type or "unknown",
        reason=reason,
    )
