"""Session-wide held-out seed enforcement and reason validation."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Enable before collection and propagate to spawned Python runners."""
    config.addinivalue_line(
        "markers",
        "heldout_seed_ok(reason): reference held-out seed data without executing a simulation",
    )
    config._seedguard_previous = os.environ.get("ROBOT_SF_PYTEST_SEED_GUARD")
    os.environ["ROBOT_SF_PYTEST_SEED_GUARD"] = "1"
    audit = os.environ.get("ROBOT_SF_PYTEST_SEED_AUDIT")
    if audit:
        worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
        Path(audit.replace("{worker}", worker)).touch(exist_ok=True)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Fail CI collection for every marker without a nonempty explicit reason."""
    for item in items:
        for marker in item.iter_markers("heldout_seed_ok"):
            reason = marker.kwargs.get("reason")
            if marker.args or not isinstance(reason, str) or not reason.strip():
                raise pytest.UsageError(f"{item.nodeid}: heldout_seed_ok requires reason='...'")
    # This marker documents static-only exceptions; it never disables real seeding checks.


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter) -> None:
    """Report intercepted attempts when an audit destination is requested."""
    audit = os.environ.get("ROBOT_SF_PYTEST_SEED_AUDIT")
    if audit:
        path = Path(audit.replace("{worker}", os.environ.get("PYTEST_XDIST_WORKER", "main")))
        count = len(path.read_text().splitlines()) if path.exists() else 0
        terminalreporter.write_line(f"SEEDGUARD: {count} held-out seeding attempts blocked")


def pytest_unconfigure(config: pytest.Config) -> None:
    """Restore the calling process's guard setting after the session."""
    from tests.support.seedguard_boundaries import uninstall

    uninstall()
    previous = getattr(config, "_seedguard_previous", None)
    if previous is None:
        os.environ.pop("ROBOT_SF_PYTEST_SEED_GUARD", None)
    else:
        os.environ["ROBOT_SF_PYTEST_SEED_GUARD"] = previous


@pytest.hookimpl(tryfirst=True)
def pytest_sessionstart(session: pytest.Session) -> None:
    """Guard legacy RNG seeds that simulator routes consume implicitly."""
    from tests.support.seedguard_boundaries import install

    install()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item):
    """Restore standalone RNG policy checks and validate dynamic marker use."""
    from tests.support.seedguard_boundaries import (
        restore_static_rngs,
        restore_unused_legacy_seeds,
        set_current_item,
    )

    set_current_item(item)
    try:
        yield
    finally:
        restore_static_rngs()
        restore_unused_legacy_seeds()
        set_current_item(None)
        pytest_collection_modifyitems([item])
