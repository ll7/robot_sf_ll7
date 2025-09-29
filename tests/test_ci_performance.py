"""Integration test for CI linting performance."""

import subprocess
import time
from pathlib import Path


def test_ruff_linting_performance():
    """Test that ruff linting completes within performance budget."""
    start_time = time.time()

    # Run ruff check on the entire codebase
    result = subprocess.run(
        ["uv", "run", "ruff", "check", "."],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
        check=False,
    )

    end_time = time.time()
    duration = end_time - start_time

    # Assert completion within 60 seconds
    assert duration < 60.0, f"Ruff linting took {duration:.2f}s, exceeding 60s budget"

    # Should not crash (return code 0 or 1 is acceptable - 1 means issues found)
    assert result.returncode in [0, 1], f"Ruff failed with return code {result.returncode}"


def test_ruff_format_performance():
    """Test that ruff formatting completes within reasonable time."""
    start_time = time.time()

    # Run ruff format check (doesn't modify files)
    result = subprocess.run(
        ["uv", "run", "ruff", "format", "--check", "."],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
        check=False,
    )

    end_time = time.time()
    duration = end_time - start_time

    # Format should be fast - budget 30 seconds
    assert duration < 30.0, f"Ruff formatting took {duration:.2f}s, exceeding 30s budget"

    # Return code 0 = no changes needed, 1 = would format
    assert result.returncode in [0, 1], f"Ruff format failed with return code {result.returncode}"
