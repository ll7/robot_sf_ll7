"""New cheap test files must reach PR selection without registration edits."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_unregistered_test_runs_by_default(tmp_path):
    """Run a new test through the real PR marker selection."""
    target = tmp_path / "test_temporary_fast_default.py"
    try:
        target.write_text(
            "import pytest\ndef test_new_contract(): assert True\n@pytest.mark.slow\ndef test_explicit_slow(): assert False\n"
        )
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-p",
                "tests.conftest",
                str(target),
                "-m",
                "not slow",
                "-q",
                "-o",
                "addopts=",
            ],
            cwd=tmp_path,
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert result.returncode == 0 and "1 passed" in result.stdout, result.stdout + result.stderr
    finally:
        target.unlink(missing_ok=True)
