"""TODO docstring. Document this module."""

import pytest

from examples.classic_interactions_pygame import run_demo


def test_example_dry_run_minimal(monkeypatch: pytest.MonkeyPatch):
    """Run the classic_interactions demo in dry-run mode to ensure imports and basic validation succeed.

    This test must be lightweight: it only calls run_demo(dry_run=True) which should perform
    resource checks and return quickly without requiring heavy optional dependencies.
    """
    # Ensure environment is in a deterministic, headless state for the test
    monkeypatch.delenv("SDL_VIDEODRIVER", raising=False)
    monkeypatch.delenv("PYGAME_HIDE_SUPPORT_PROMPT", raising=False)

    # Call the demo in dry-run mode; must not raise
    result = run_demo(dry_run=True)
    assert result == []
