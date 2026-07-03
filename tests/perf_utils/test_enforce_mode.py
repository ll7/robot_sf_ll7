"""Test enforcement mode failure behavior (T021).

Strategy:
  - Invoke a subprocess running pytest on a trivial test that sleeps briefly.
  - Set env vars: ROBOT_SF_PERF_ENFORCE=1, ROBOT_SF_PERF_SOFT=0.01 to force a soft breach.
  - Expect non-zero exit code (pytest.ExitCode.TESTS_FAILED).

We do not recurse into running the whole suite; only a tiny inline test file is created
in a temp directory to keep runtime minimal.

The nested pytest child is bounded via ``run_bounded_subprocess`` so a wedged
child (e.g. a CUDA/interpreter teardown deadlock on a shared GPU/HPC node) is
reaped as a process group instead of blocking the parent suite forever after it
has already reached 100% -- the LiCCA post-guard teardown hang tracked in issue
#4216. Note: the ``@pytest.mark.timeout`` marker below is *not* enforced
(``pytest-timeout`` is not installed), so the explicit ``timeout_seconds`` bound is
what actually protects the parent process.
"""

from __future__ import annotations

import os
import shutil
import sys
import textwrap
from pathlib import Path

import pytest

from tests.support.process_teardown import NestedProcessTimeout, run_bounded_subprocess

# Generous default: the nested pytest imports the full robot_sf stack (torch, etc.),
# which is slow but bounded on a healthy host. A true teardown deadlock is reaped
# once this budget elapses. Overridable for constrained/faster hosts.
NESTED_PYTEST_TIMEOUT_SECONDS = float(
    os.environ.get("ROBOT_SF_NESTED_PYTEST_TIMEOUT", "180"),
)


@pytest.mark.timeout(10)
def test_enforce_mode_escalates():
    """Verify enforce mode converts soft breaches into failures."""
    # Create a small test file under tests/ so tests/conftest.py is loaded.
    repo_root = Path(__file__).resolve().parents[2]
    target_dir = Path(__file__).resolve().parent / f"_tmp_enforce_{os.getpid()}"
    target_dir.mkdir(parents=True, exist_ok=True)
    test_file = target_dir / "test_sleep_enforce.py"
    test_file.write_text(
        textwrap.dedent(
            """
        import time
        def test_sleep_short():
            time.sleep(0.02)
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
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    # Run pytest from repository root so root-level conftest performance hooks are active.
    # Bound the nested pytest child and reap its process group on timeout so a
    # teardown deadlock cannot hang the parent suite (issue #4216).
    try:
        try:
            proc = run_bounded_subprocess(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "-p",
                    "no:cov",
                    "--disable-warnings",
                    str(test_file.resolve()),
                ],
                cwd=str(repo_root),
                env=env,
                timeout_seconds=NESTED_PYTEST_TIMEOUT_SECONDS,
            )
        except NestedProcessTimeout as exc:
            pytest.fail(
                "Nested pytest child did not exit within "
                f"{NESTED_PYTEST_TIMEOUT_SECONDS:g}s and was reaped "
                f"({exc.cleanup_status}); suspected teardown deadlock (issue #4216).",
            )
        assert proc.returncode != 0, (
            "Enforce mode did not convert soft breach to failure (expected non-zero)"
        )
    finally:
        try:
            test_file.unlink(missing_ok=True)  # type: ignore[arg-type]
        except Exception:
            pass
        try:
            shutil.rmtree(target_dir, ignore_errors=True)
        except Exception:
            pass
