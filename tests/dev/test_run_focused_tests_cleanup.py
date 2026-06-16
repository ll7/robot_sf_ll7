"""Contract test for scripts/dev/run_focused_tests.sh coverage cleanup."""

import os
import subprocess
from pathlib import Path


def _run_focused(repo: Path, *, keep_coverage: bool = False) -> subprocess.CompletedProcess[str]:
    """Run the focused-test helper in *repo* and return the result."""
    env = {**os.environ, "PATH": f"{repo / 'bin'}{os.pathsep}{os.environ['PATH']}"}
    if keep_coverage:
        env["FOCUSED_TEST_KEEP_COVERAGE"] = "1"
    else:
        env.pop("FOCUSED_TEST_KEEP_COVERAGE", None)
    return subprocess.run(
        ["scripts/dev/run_focused_tests.sh", "tests/dev/test_tiny.py", "-q"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_focused_tests_removes_coverage_on_success(helper_repo: Path) -> None:
    """A successful focused run should delete output/coverage."""
    result = _run_focused(helper_repo)

    assert result.returncode == 0, f"Script failed: {result.stderr}"
    assert not (helper_repo / "output" / "coverage" / "generated.txt").exists(), (
        "generated coverage output should be removed after a successful focused run"
    )
    assert (helper_repo / "output" / "coverage" / "tracked.txt").exists(), (
        "tracked output files should remain after cleanup"
    )
    status = subprocess.run(
        ["git", "status", "--short"], cwd=helper_repo, check=True, text=True, capture_output=True
    ).stdout.strip()
    assert status == "", status


def test_focused_tests_preserves_coverage_when_keep_flag_set(helper_repo: Path) -> None:
    """FOCUSED_TEST_KEEP_COVERAGE=1 should leave output/coverage intact."""
    result = _run_focused(helper_repo, keep_coverage=True)

    assert result.returncode == 0, f"Script failed: {result.stderr}"
    assert (helper_repo / "output" / "coverage").exists(), (
        "output/coverage should be preserved when FOCUSED_TEST_KEEP_COVERAGE=1"
    )


def test_focused_tests_failure_prints_compact_summary_and_log_path(helper_repo: Path) -> None:
    """Failing focused tests should not stream raw pytest output into the parent thread."""
    (helper_repo / "bin" / "uv").write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "echo '============================= FAILURES ============================='",
                'for i in $(seq 1 120); do echo "FAILED tests/dev/test_tiny.py::test_$i - boom"; done',
                "exit 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (helper_repo / "bin" / "uv").chmod(0o755)

    result = _run_focused(helper_repo)

    assert result.returncode == 1
    assert "Focused pytest failed: exit 1" in result.stdout
    assert "Full log:" in result.stdout
    assert "FAILED tests/dev/test_tiny.py::test_1 - boom" in result.stdout
    assert "test_120" not in result.stdout
    assert "more matching lines omitted" in result.stdout
