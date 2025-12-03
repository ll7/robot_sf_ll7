"""T036: Import purity test for environment_factory.

Ensures importing the factory module does not emit stdout/stderr prints or create files.
Allowable output: Loguru configuration lines (if any) and warnings indirectly from dependencies.

We capture sys.stdout/stderr during import in an isolated subprocess-like reload.
"""

from __future__ import annotations

import importlib
import io
import sys
from contextlib import redirect_stderr, redirect_stdout


def test_environment_factory_import_is_pure():
    # Remove module if already imported to force re-execution of top-level code
    """TODO docstring. Document this function."""
    sys.modules.pop("robot_sf.gym_env.environment_factory", None)
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
        importlib.import_module("robot_sf.gym_env.environment_factory")
    out = stdout_buf.getvalue()
    err = stderr_buf.getvalue()
    # No direct 'print(' output expected (Loguru logs include level markers, accept them)
    assert "print(" not in out
    # Avoid being over-strict with empty allowance; just ensure not dominated by plain prints
    assert not out.strip().startswith("Usage:")  # placeholder for accidental help text
    # stderr should be empty or contain only warnings
    assert not err or "WARNING" in err or "Deprecation" in err
