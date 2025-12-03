"""Test enforcement mode failure behavior (T021).

Strategy:
  - Invoke a subprocess running pytest on a trivial test that sleeps briefly.
  - Set env vars: ROBOT_SF_PERF_ENFORCE=1, ROBOT_SF_PERF_SOFT=0.01 to force a soft breach.
  - Expect non-zero exit code (pytest.ExitCode.TESTS_FAILED).

We do not recurse into running the whole suite; only a tiny inline test file is created
in a temp directory to keep runtime minimal.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


@pytest.mark.timeout(10)
def test_enforce_mode_escalates():
    # Create a small test file under repo tests/ so root conftest is loaded.
    """TODO docstring. Document this function."""
    repo_root = Path(__file__).resolve().parents[2]
    target_dir = repo_root / "tests" / "perf_utils" / "_enforce_tmp"
    target_dir.mkdir(parents=True, exist_ok=True)
    test_file = target_dir / "test_sleep_enforce.py"
    test_file.write_text(
        textwrap.dedent(
            """
        import time
        def test_sleep_short():
            time.sleep(0.05)
        """,
        ),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["ROBOT_SF_PERF_ENFORCE"] = "1"
    env["ROBOT_SF_PERF_SOFT"] = "0.01"  # ensure any test >10ms is soft breach
    env["ROBOT_SF_PERF_HARD"] = "0.015"  # treat >15ms as hard breach to exercise path
    # Ensure relax not set
    env.pop("ROBOT_SF_PERF_RELAX", None)
    # Run pytest from repository root so root-level conftest performance hooks are active.
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", str(test_file.resolve())],
            cwd=str(repo_root),
            env=env,
            check=False,
        )
        assert proc.returncode != 0, (
            "Enforce mode did not convert soft breach to failure (expected non-zero)"
        )
    finally:
        try:
            test_file.unlink(missing_ok=True)  # type: ignore[arg-type]
        except Exception:
            pass
