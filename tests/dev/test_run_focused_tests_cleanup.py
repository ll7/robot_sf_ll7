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


def test_focused_tests_uses_portable_owned_temproot_and_removes_it(helper_repo: Path) -> None:
    """The wrapper should remove its exact macOS-style temporary root on exit."""
    capture = helper_repo / "captured-temproot.txt"
    (helper_repo / "bin" / "uv").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\nprintf '%s' \"$PYTEST_DEBUG_TEMPROOT\" > \"$CAPTURE_TEMPROOT\"\n",
        encoding="utf-8",
    )
    (helper_repo / "bin" / "uv").chmod(0o755)
    macos_temp = helper_repo / "macos-private-tmp"
    macos_temp.mkdir()
    env = {
        **os.environ,
        "PATH": f"{helper_repo / 'bin'}{os.pathsep}{os.environ['PATH']}",
        "TMPDIR": str(macos_temp),
        "CAPTURE_TEMPROOT": str(capture),
    }
    env.pop("PYTEST_DEBUG_TEMPROOT", None)

    result = subprocess.run(
        ["scripts/dev/run_focused_tests.sh", "tests/dev/test_tiny.py", "-q"],
        cwd=helper_repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    owned_root = Path(capture.read_text(encoding="utf-8"))
    assert owned_root.is_relative_to(macos_temp.resolve())
    assert not owned_root.exists()
    assert "sha256sum" not in (helper_repo / "scripts/dev/run_focused_tests.sh").read_text(
        encoding="utf-8"
    )
