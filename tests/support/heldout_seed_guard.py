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
    patch = getattr(config, "_seedguard_rng_patch", None)
    if patch is not None:
        patch.undo()
    previous = getattr(config, "_seedguard_previous", None)
    if previous is None:
        os.environ.pop("ROBOT_SF_PYTEST_SEED_GUARD", None)
    else:
        os.environ["ROBOT_SF_PYTEST_SEED_GUARD"] = previous


@pytest.hookimpl(tryfirst=True)
def pytest_sessionstart(session: pytest.Session) -> None:
    """Guard legacy RNG seeds that simulator routes consume implicitly."""
    import inspect
    import subprocess
    from functools import wraps

    from tests.support.seedguard_boundaries import install

    install()
    patch = pytest.MonkeyPatch()
    session.config._seedguard_rng_patch = patch
    original_popen_init = subprocess.Popen.__init__
    popen_signature = inspect.signature(original_popen_init)
    root = Path(__file__).resolve().parents[2]
    bootstrap = root / "tests/support/seedguard_bootstrap"

    @wraps(original_popen_init)
    def guarded_popen_init(self, *args, **kwargs):
        bound = popen_signature.bind_partial(self, *args, **kwargs)
        supplied_env = bound.arguments.get("env")
        child_env = dict(os.environ if supplied_env is None else supplied_env)
        child_env["ROBOT_SF_PYTEST_SEED_GUARD"] = "1"
        paths = [str(bootstrap), str(root / "tests/support"), child_env.get("PYTHONPATH", "")]
        child_env["PYTHONPATH"] = os.pathsep.join(path for path in paths if path)
        for key in ("ROBOT_SF_PYTEST_SEED_AUDIT", "PYTEST_CURRENT_TEST", "PYTEST_XDIST_WORKER"):
            if key in os.environ:
                child_env[key] = os.environ[key]
        bound.arguments["env"] = child_env
        return original_popen_init(*bound.args, **bound.kwargs)

    patch.setattr(subprocess.Popen, "__init__", guarded_popen_init)
