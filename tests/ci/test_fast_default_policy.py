"""New cheap test files must reach PR selection without registration edits."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_unregistered_test_runs_by_default():
    """Run a new test through the real PR marker selection."""
    target = ROOT / "tests/ci/test_temporary_fast_default.py"
    try:
        target.write_text(
            "import pytest\ndef test_new_contract(): assert True\n@pytest.mark.slow\ndef test_explicit_slow(): assert False\n"
        )
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(target), "-m", "not slow", "-q", "-o", "addopts="],
            cwd=ROOT,
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert result.returncode == 0 and "1 passed" in result.stdout, result.stdout + result.stderr
    finally:
        target.unlink(missing_ok=True)


def test_train_suite_refuses_partial_selectors():
    """A train-proof command cannot silently become a focused test run."""
    result = subprocess.run(
        ["bash", "scripts/dev/run_tests_parallel.sh", "--train-suite", "tests/ci"],
        cwd=ROOT,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "no pytest selectors" in result.stderr
