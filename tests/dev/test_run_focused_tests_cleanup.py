"""Contract test for scripts/dev/run_focused_tests.sh coverage cleanup."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "dev" / "run_focused_tests.sh"


@pytest.fixture()
def helper_repo(tmp_path: Path) -> Path:
    """Return a tiny git repo with the focused-test helper and a fake uv binary."""
    repo = tmp_path / "repo"
    scripts_dir = repo / "scripts" / "dev"
    scripts_dir.mkdir(parents=True)
    shutil.copy2(SCRIPT, scripts_dir / "run_focused_tests.sh")
    shutil.copy2(REPO_ROOT / "scripts" / "dev" / "common_setup.sh", scripts_dir / "common_setup.sh")
    (repo / "bin").mkdir()
    (repo / "bin" / "uv").write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "mkdir -p output/coverage",
                "printf fake > output/coverage/generated.txt",
                "exit 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (repo / "bin" / "uv").chmod(0o755)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    return repo


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
    assert not (helper_repo / "output" / "coverage").exists(), (
        "output/coverage should be removed after a successful focused run"
    )


def test_focused_tests_preserves_coverage_when_keep_flag_set(helper_repo: Path) -> None:
    """FOCUSED_TEST_KEEP_COVERAGE=1 should leave output/coverage intact."""
    result = _run_focused(helper_repo, keep_coverage=True)

    assert result.returncode == 0, f"Script failed: {result.stderr}"
    assert (helper_repo / "output" / "coverage").exists(), (
        "output/coverage should be preserved when FOCUSED_TEST_KEEP_COVERAGE=1"
    )
