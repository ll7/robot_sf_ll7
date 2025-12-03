"""Logging toggle test (T013 / FR-014).

Forward-looking test: expects the demo module to introduce a verbosity or logging
boolean constant (e.g., LOGGING_ENABLED) that suppresses non-essential per-episode
print lines when False.

TDD: The current implementation always prints. We simulate capturing stdout and
assert that when we patch LOGGING_ENABLED=False (if present) the output either becomes
empty or significantly reduced (heuristic). This will fail until the constant and
conditional logging logic (T023) are added.
"""

from __future__ import annotations

import importlib
import io
import sys


def test_logging_toggle_reduces_output():
    """TODO docstring. Document this function."""
    mod = importlib.import_module("examples.classic_interactions_pygame")
    # Force non-dry run
    original_dry = getattr(mod, "DRY_RUN", None)
    mod.DRY_RUN = False  # type: ignore
    # Provide fallback if constant not yet defined
    had_logging_flag = hasattr(mod, "LOGGING_ENABLED")
    if not had_logging_flag:
        # Inject a temporary attribute to mimic planned API
        mod.LOGGING_ENABLED = True  # type: ignore[attr-defined]
    original_logging = getattr(mod, "LOGGING_ENABLED", True)

    # First run with logging enabled
    buf_enabled = io.StringIO()
    sys_stdout_original = sys.stdout
    sys.stdout = buf_enabled  # type: ignore
    try:
        try:
            mod.LOGGING_ENABLED = True  # type: ignore[attr-defined]
            mod.run_demo()
        except Exception:
            pass  # ignore underlying failure; we only measure output volume (episodes empty ok)
    finally:
        sys.stdout = sys_stdout_original

    enabled_output = buf_enabled.getvalue()

    # Now run with logging disabled
    buf_disabled = io.StringIO()
    sys.stdout = buf_disabled  # type: ignore
    try:
        try:
            mod.LOGGING_ENABLED = False  # type: ignore[attr-defined]
            mod.run_demo()
        except Exception:
            pass
    finally:
        sys.stdout = sys_stdout_original

    disabled_output = buf_disabled.getvalue()

    # Restore originals
    mod.DRY_RUN = original_dry  # type: ignore
    mod.LOGGING_ENABLED = original_logging  # type: ignore[attr-defined]
    if not had_logging_flag:
        delattr(mod, "LOGGING_ENABLED")

    # TDD assertion: expect disabled output shorter than enabled output
    assert len(disabled_output) < len(enabled_output), (
        "Expected logging-disabled output to be shorter (TDD failing until logging toggle implemented)."
    )
