"""Tests for global --quiet and --log-level flags in benchmark CLI."""

from __future__ import annotations

import logging

from robot_sf.benchmark import cli


def test_log_level_parsing_debug():
    """TODO docstring. Document this function."""
    parser = cli.get_parser()
    args = parser.parse_args(["list-algorithms", "--log-level", "DEBUG"])  # type: ignore[arg-type]
    assert args.log_level == "DEBUG"
    # configure logging and check effective level
    cli.configure_logging(False, args.log_level)
    assert logging.getLogger().level == logging.DEBUG


def test_quiet_overrides_log_level():
    """TODO docstring. Document this function."""
    parser = cli.get_parser()
    args = parser.parse_args(["list-algorithms", "--log-level", "DEBUG", "--quiet"])  # type: ignore[arg-type]
    cli.configure_logging(args.quiet, args.log_level)
    # Quiet should downgrade below DEBUG (WARNING expected)
    assert logging.getLogger().level == logging.WARNING
