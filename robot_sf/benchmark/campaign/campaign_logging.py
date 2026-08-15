"""Logging defaults shared by long-running benchmark campaign entry points."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from robot_sf.common.logging import configure_logging

if TYPE_CHECKING:
    import argparse

CAMPAIGN_LOG_LEVEL_ENV = "ROBOT_SF_CAMPAIGN_LOG_LEVEL"


def campaign_debug_default() -> bool:
    """Return whether the campaign log-level environment opts into DEBUG."""

    value = os.environ.get(CAMPAIGN_LOG_LEVEL_ENV, "INFO").strip().upper()
    if value not in {"INFO", "DEBUG"}:
        raise ValueError(f"{CAMPAIGN_LOG_LEVEL_ENV} must be INFO or DEBUG (got {value!r})")
    return value == "DEBUG"


def add_campaign_logging_argument(parser: argparse.ArgumentParser) -> None:
    """Add the common explicit DEBUG opt-in to a campaign argument parser."""

    parser.add_argument(
        "--debug",
        action="store_true",
        default=campaign_debug_default(),
        help=f"Enable DEBUG logs (default INFO; env: {CAMPAIGN_LOG_LEVEL_ENV}=DEBUG).",
    )


def configure_campaign_logging(*, debug: bool) -> None:
    """Set Loguru and standard-library logging to INFO unless DEBUG was requested."""

    configure_logging(verbose=debug)
    level = logging.DEBUG if debug else logging.INFO
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(level=level, format="%(levelname)s:%(name)s:%(message)s")
    root_logger.setLevel(level)
    for handler in root_logger.handlers:
        handler.setLevel(level)


__all__ = [
    "CAMPAIGN_LOG_LEVEL_ENV",
    "add_campaign_logging_argument",
    "campaign_debug_default",
    "configure_campaign_logging",
]
