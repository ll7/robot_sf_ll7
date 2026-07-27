"""Focused unit coverage for ``robot_sf/benchmark/campaign_logging.py``.

This module exercises the campaign logging public API directly (rather than through
the campaign CLI entry points) and prioritizes the branches that
``tests/benchmark/test_issue_5829_campaign_robustness.py`` does not cover: the
``ValueError`` invalid-environment branch of ``campaign_debug_default()``, the
whitespace/case normalization of the environment value, isolated
``argparse`` behavior for ``add_campaign_logging_argument()``, and the
``logging.basicConfig`` code path of ``configure_campaign_logging()`` taken when
the standard-library root logger has no pre-existing handlers.

Environment and standard-library root-logger state are snapshotted and restored
so no process-global configuration leaks into sibling tests.
"""

from __future__ import annotations

import argparse
import logging

import pytest

from robot_sf.benchmark import campaign_logging


def test_campaign_log_level_env_constant_value() -> None:
    """The exported environment-variable name matches the documented contract."""

    assert campaign_logging.CAMPAIGN_LOG_LEVEL_ENV == "ROBOT_SF_CAMPAIGN_LOG_LEVEL"


@pytest.mark.parametrize(
    ("raw_env", "expected"),
    [
        # Exact values.
        ("INFO", False),
        ("DEBUG", True),
        # ``.strip().upper()`` normalization (lower-case + surrounding whitespace).
        ("info", False),
        ("debug", True),
        ("  INFO  ", False),
        ("\tdebug\n", True),
    ],
)
def test_campaign_debug_default_parses_valid_env(
    monkeypatch: pytest.MonkeyPatch, raw_env: str, expected: bool
) -> None:
    """Valid INFO/DEBUG values resolve to the matching debug flag after normalization."""

    monkeypatch.setenv(campaign_logging.CAMPAIGN_LOG_LEVEL_ENV, raw_env)
    assert campaign_logging.campaign_debug_default() is expected


def test_campaign_debug_default_unset_env_defaults_to_info_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unset environment variable defaults to INFO and therefore returns ``False``."""

    monkeypatch.delenv(campaign_logging.CAMPAIGN_LOG_LEVEL_ENV, raising=False)
    assert campaign_logging.campaign_debug_default() is False


@pytest.mark.parametrize(
    ("raw_env", "normalized_in_message"),
    [
        ("WARNING", "WARNING"),
        ("trace", "TRACE"),
        ("verbose", "VERBOSE"),
        # Empty / whitespace-only values collapse to ``""`` after ``.strip()``.
        ("", ""),
        ("   ", ""),
    ],
)
def test_campaign_debug_default_rejects_invalid_env_with_actionable_message(
    monkeypatch: pytest.MonkeyPatch, raw_env: str, normalized_in_message: str
) -> None:
    """Any value outside INFO/DEBUG fails closed and names the variable and rejected value."""

    monkeypatch.setenv(campaign_logging.CAMPAIGN_LOG_LEVEL_ENV, raw_env)
    with pytest.raises(ValueError) as exc_info:
        campaign_logging.campaign_debug_default()

    message = str(exc_info.value)
    assert campaign_logging.CAMPAIGN_LOG_LEVEL_ENV in message
    assert repr(normalized_in_message) in message


def test_add_campaign_logging_argument_default_tracks_unset_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fresh parser without ``--debug`` defaults to the environment-derived value."""

    monkeypatch.delenv(campaign_logging.CAMPAIGN_LOG_LEVEL_ENV, raising=False)
    parser = argparse.ArgumentParser()
    campaign_logging.add_campaign_logging_argument(parser)

    parsed = parser.parse_args([])
    assert parsed.debug is campaign_logging.campaign_debug_default()
    assert parsed.debug is False


def test_add_campaign_logging_argument_default_tracks_debug_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ``--debug`` default becomes ``True`` when the environment opts into DEBUG."""

    monkeypatch.setenv(campaign_logging.CAMPAIGN_LOG_LEVEL_ENV, "DEBUG")
    parser = argparse.ArgumentParser()
    campaign_logging.add_campaign_logging_argument(parser)

    parsed = parser.parse_args([])
    assert parsed.debug is campaign_logging.campaign_debug_default()
    assert parsed.debug is True


def test_add_campaign_logging_argument_explicit_flag_overrides_info_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Passing ``--debug`` forces DEBUG even when the environment selects INFO."""

    monkeypatch.setenv(campaign_logging.CAMPAIGN_LOG_LEVEL_ENV, "INFO")
    parser = argparse.ArgumentParser()
    campaign_logging.add_campaign_logging_argument(parser)

    parsed = parser.parse_args(["--debug"])
    assert parsed.debug is True


@pytest.mark.parametrize(
    ("debug", "expected_level"),
    [(False, logging.INFO), (True, logging.DEBUG)],
)
def test_configure_campaign_logging_forwards_verbose_and_bootstraps_root_handlers(
    monkeypatch: pytest.MonkeyPatch, debug: bool, expected_level: int
) -> None:
    """Forwarding and the no-handlers ``basicConfig`` path apply one level everywhere.

    This complements ``test_campaign_logging_configures_loguru_and_stdlib_levels``
    in the issue-5829 regression suite, which exercises the path where root handlers
    already exist. Here the root logger is emptied first so the
    ``logging.basicConfig`` branch is taken, proving the campaign setup bootstraps a
    handler at the matching level rather than silently leaving logging unconfigured.
    """

    forwarded: list[bool] = []
    monkeypatch.setattr(
        campaign_logging,
        "configure_logging",
        lambda *, verbose: forwarded.append(verbose),
    )

    root = logging.getLogger()
    snapshot_level = root.level
    snapshot_handlers = [(handler, handler.level) for handler in root.handlers]
    try:
        for handler in list(root.handlers):
            root.removeHandler(handler)
        assert root.handlers == []

        campaign_logging.configure_campaign_logging(debug=debug)

        # The requested verbosity is forwarded to the shared logging surface.
        assert forwarded == [debug]
        # The standard-library root logger and its bootstrapped handler share one level.
        assert root.level == expected_level
        assert len(root.handlers) == 1
        assert all(handler.level == expected_level for handler in root.handlers)
    finally:
        for handler in list(root.handlers):
            root.removeHandler(handler)
        for handler, level in snapshot_handlers:
            root.addHandler(handler)
            handler.setLevel(level)
        root.setLevel(snapshot_level)
