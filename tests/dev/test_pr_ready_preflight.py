"""Tests for the cheap shell preflight in scripts/dev/pr_ready_check.sh."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DEV = REPO_ROOT / "scripts" / "dev"

_POST_PREFLIGHT_SCRIPTS = [
    "ruff_fix_format.sh",
    "run_tests_parallel.sh",
    "check_changed_coverage.sh",
    "check_docstring_todos_diff.sh",
    "check_docstring_todos_ratchet.sh",
    "pr_ready_freshness.py",
]


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@test", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )


def _make_fake_bin(repo: Path, *, fail: bool = True) -> None:
    """Create fake ``python`` and ``uv`` in *repo*/bin that simulate missing modules."""
    bin_dir = repo / "bin"
    bin_dir.mkdir(exist_ok=True)

    fake = bin_dir / "python"
    if fail:
        fake.write_text(
            "#!/usr/bin/env bash\n"
            "has_stdin=0\n"
            'for arg in "$@"; do\n'
            '  if [[ "$arg" == "-" ]]; then\n'
            "    has_stdin=1\n"
            "    break\n"
            "  fi\n"
            "done\n"
            'if [[ "$has_stdin" -eq 1 ]]; then\n'
            '  payload="$* $(cat)"\n'
            "else\n"
            '  payload="$*"\n'
            "fi\n"
            "if [[ \"$payload\" == *'import importlib'* ]]; then\n"
            "  echo 'duckdb, pyarrow' >&2\n"
            "  exit 1\n"
            "fi\n"
            "exit 0\n",
            encoding="utf-8",
        )
    else:
        fake.write_text(
            "#!/usr/bin/env bash\nexit 0\n",
            encoding="utf-8",
        )
    fake.chmod(0o755)

    real_uv = shutil.which("uv") or "uv"
    fake_uv = bin_dir / "uv"
    fake_uv.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$1" == "run" ]]; then\n'
        "  shift\n"
        '  exec "$@"\n'
        "fi\n"
        f'exec "{real_uv}" "$@"\n',
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)


def _make_fake_scripts(repo: Path) -> None:
    """Create no-op stubs for every script ``pr_ready_check.sh`` calls after the preflight."""
    scripts_dir = repo / "scripts" / "dev"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    for name in _POST_PREFLIGHT_SCRIPTS:
        stub = scripts_dir / name
        if name.endswith(".py"):
            stub.write_text("import sys\nsys.exit(0)\n", encoding="utf-8")
        else:
            stub.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            stub.chmod(0o755)


@pytest.fixture()
def preflight_repo(tmp_path: Path) -> Path:
    """Return a committed git repo with the preflight and stub scripts."""
    repo = tmp_path / "repo"
    scripts_dir = repo / "scripts" / "dev"
    scripts_dir.mkdir(parents=True)
    shutil.copy2(SCRIPTS_DEV / "common_setup.sh", scripts_dir / "common_setup.sh")
    shutil.copy2(SCRIPTS_DEV / "pr_ready_check.sh", scripts_dir / "pr_ready_check.sh")
    _make_fake_scripts(repo)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")
    return repo


def _run_pr_ready(
    repo: Path,
    *,
    env_overrides: dict[str, str] | None = None,
    help_flag: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run ``pr_ready_check.sh`` and return the result."""
    cmd = ["scripts/dev/pr_ready_check.sh"]
    if help_flag:
        cmd.append("--help")
    env = {**os.environ, "PATH": f"{repo / 'bin'}{os.pathsep}{os.environ['PATH']}"}
    env.pop("PR_READY_FINAL", None)
    env.pop("PR_READY_MODE", None)
    env.pop("PR_READY_SKIP_PREFLIGHT", None)
    if env_overrides:
        env.update(env_overrides)
    return subprocess.run(
        cmd,
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_help_bypasses_preflight(preflight_repo: Path) -> None:
    """--help exits 0 before preflight runs, even with missing modules."""
    _make_fake_bin(preflight_repo, fail=True)
    result = _run_pr_ready(preflight_repo, help_flag=True)
    assert result.returncode == 0
    assert "Final PR readiness requires analytics dependencies" not in result.stderr


def test_preflight_helper_exits_nonzero_when_modules_are_missing() -> None:
    """The embedded Python must fail, not only print missing modules."""
    common_setup = (SCRIPTS_DEV / "common_setup.sh").read_text(encoding="utf-8")
    assert "raise SystemExit(1)" in common_setup


def test_preflight_fails_when_modules_missing(preflight_repo: Path) -> None:
    """Preflight should exit 2 with a concise error when modules are unavailable."""
    _make_fake_bin(preflight_repo, fail=True)
    _git(preflight_repo, "add", "-A")
    _git(preflight_repo, "commit", "-q", "-m", "fake tools")
    result = _run_pr_ready(
        preflight_repo,
        help_flag=False,
        env_overrides={"PR_READY_MODE": "final"},
    )
    assert result.returncode == 2, f"Expected exit 2, got {result.returncode}"
    assert "Final PR readiness requires analytics dependencies" in result.stderr
    assert "uv sync --all-extras" in result.stderr


def test_interim_mode_keeps_existing_non_preflight_path(preflight_repo: Path) -> None:
    """Default interim readiness should not run the final-only dependency preflight."""
    _make_fake_bin(preflight_repo, fail=True)
    result = _run_pr_ready(preflight_repo, help_flag=False)
    assert result.returncode == 0, f"Script failed: {result.stderr}"
    assert "Final PR readiness requires analytics dependencies" not in result.stderr


def test_preflight_passes_when_modules_available(preflight_repo: Path) -> None:
    """Preflight should pass silently when python reports no missing modules."""
    _make_fake_bin(preflight_repo, fail=False)
    _git(preflight_repo, "add", "-A")
    _git(preflight_repo, "commit", "-q", "-m", "fake tools")
    result = _run_pr_ready(
        preflight_repo,
        help_flag=False,
        env_overrides={"PR_READY_MODE": "final"},
    )
    assert result.returncode == 0, f"Script failed: {result.stderr}"
    assert "Final PR readiness requires analytics dependencies" not in result.stderr


def test_preflight_skip_env_var_bypasses_check(preflight_repo: Path) -> None:
    """PR_READY_SKIP_PREFLIGHT=1 should skip the preflight entirely."""
    _make_fake_bin(preflight_repo, fail=True)
    _git(preflight_repo, "add", "-A")
    _git(preflight_repo, "commit", "-q", "-m", "fake tools")
    result = _run_pr_ready(
        preflight_repo,
        help_flag=False,
        env_overrides={
            "PR_READY_SKIP_PREFLIGHT": "1",
            "PR_READY_MODE": "final",
        },
    )
    assert result.returncode == 0, f"Script failed: {result.stderr}"
    assert "Final PR readiness requires analytics dependencies" not in result.stderr
