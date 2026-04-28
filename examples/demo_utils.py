"""Shared helpers for smoke-friendly example execution."""

import os


def fast_demo_enabled() -> bool:
    """Return True when examples should prefer smoke-test-friendly execution."""
    return os.environ.get("ROBOT_SF_FAST_DEMO", "0") == "1" or "PYTEST_CURRENT_TEST" in os.environ
