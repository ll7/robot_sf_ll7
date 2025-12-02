"""Module test_example_dry_run auto-generated docstring."""

import os

from examples.classic_interactions_pygame import run_demo


def test_example_dry_run_minimal():
    """Run the classic_interactions demo in dry-run mode to ensure imports and basic validation succeed.

    This test must be lightweight: it only calls run_demo(dry_run=True) which should perform
    resource checks and return quickly without requiring heavy optional dependencies.
    """
    # Ensure environment is in a deterministic, headless state for the test
    os.environ.pop("SDL_VIDEODRIVER", None)
    os.environ.pop("PYGAME_HIDE_SUPPORT_PROMPT", None)

    # Call the demo in dry-run mode; must not raise
    result = run_demo(dry_run=True)
    assert result == []
