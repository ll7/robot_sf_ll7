"""Tests for the cheap shell preflight in scripts/dev/pr_ready_check.sh."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DEV = REPO_ROOT / "scripts" / "dev"
OPTIONAL_ALLOWLIST = REPO_ROOT / "tests" / "support" / "optional_test_allowlist.txt"

_POST_PREFLIGHT_SCRIPTS = [
    "check_pr_followups.py",
    "check_perf_evidence.py",
    "check_fast_results_claim_map.py",
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
    # pr_ready_check.sh also invokes scripts/validation/check_broad_exceptions.py,
    # which lives outside scripts/dev/; stub it so the preflight lane reaches the
    # post-preflight scripts without a missing-file error.
    validation_dir = repo / "scripts" / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    (validation_dir / "check_broad_exceptions.py").write_text(
        "import sys\nsys.exit(0)\n", encoding="utf-8"
    )
    # PR #4865 made pr_ready_check.sh hard-require tests/support/optional_test_
    # allowlist.txt: is_optional_readiness_path reads it to classify each
    # changed test path (e.g. tests/planner/ -> optional, tests/unit/ -> core)
    # and exits 1 when it is absent. Copy the real file so the fake repo matches
    # production; the lane-split assertions below depend on its contents.
    support_dir = repo / "tests" / "support"
    support_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OPTIONAL_ALLOWLIST, support_dir / "optional_test_allowlist.txt")


def _write_lane_logging_stub(repo: Path) -> Path:
    """Replace the test wrapper with a logger that records each lane invocation."""
    stub = repo / "scripts" / "dev" / "run_tests_parallel.sh"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s %s\\n" "${ROBOT_SF_TEST_LANE:-unset}" "$*" >> "$PWD/lane.log"\n'
        "exit 0\n",
        encoding="utf-8",
    )
    stub.chmod(0o755)
    return repo / "lane.log"


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
    home = repo / ".home"
    home.mkdir(exist_ok=True)
    env["HOME"] = str(home)
    for key in (
        "BASE_REF",
        "GITHUB_ACTIONS",
        "GITHUB_BASE_REF",
        "GITHUB_HEAD_REF",
        "GITHUB_REF",
        "GITHUB_SHA",
        "GITHUB_WORKSPACE",
        "PR_READY_FINAL",
        "PR_READY_MODE",
        "PR_READY_SKIP_PREFLIGHT",
        "ROBOT_SF_TEST_ENV",
        "SLURM_CLUSTER_NAME",
        "SLURM_JOB_ID",
    ):
        env.pop(key, None)
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


def test_pr_ready_check_escalates_optional_changed_files_to_the_optional_lane(
    preflight_repo: Path,
) -> None:
    """Predictive or optional-path changes should trigger the optional lane."""
    lane_log = _write_lane_logging_stub(preflight_repo)

    changed_file = preflight_repo / "tests" / "planner" / "test_sonic_crowdnav.py"
    changed_file.parent.mkdir(parents=True, exist_ok=True)
    changed_file.write_text("print('optional lane')\n", encoding="utf-8")
    _git(preflight_repo, "add", "-A")
    _git(preflight_repo, "commit", "-q", "-m", "optional lane change")

    result = _run_pr_ready(
        preflight_repo,
        help_flag=False,
        env_overrides={
            "BASE_REF": "HEAD~1",
            "PR_READY_MODE": "interim",
        },
    )

    assert result.returncode == 0, result.stderr
    lane_lines = lane_log.read_text(encoding="utf-8").splitlines()
    assert lane_lines == [
        "core --lane core",
        "optional --lane optional",
    ]
    assert "Optional-extra changed files requiring the predictive lane" in result.stderr


def test_pr_ready_check_keeps_core_only_changes_on_the_core_lane(preflight_repo: Path) -> None:
    """Core-only changes should not schedule the optional lane."""
    lane_log = _write_lane_logging_stub(preflight_repo)

    changed_file = preflight_repo / "tests" / "unit" / "test_core_lane.py"
    changed_file.parent.mkdir(parents=True, exist_ok=True)
    changed_file.write_text("print('core lane')\n", encoding="utf-8")
    _git(preflight_repo, "add", "-A")
    _git(preflight_repo, "commit", "-q", "-m", "core lane change")

    result = _run_pr_ready(
        preflight_repo,
        help_flag=False,
        env_overrides={
            "BASE_REF": "HEAD~1",
            "PR_READY_MODE": "interim",
        },
    )

    assert result.returncode == 0, result.stderr
    lane_lines = lane_log.read_text(encoding="utf-8").splitlines()
    assert lane_lines == ["core --lane core"]
    assert "No committed changed files require the optional-extra lane." in result.stderr


def test_interim_mode_reports_dirty_paths_excluded_from_changed_file_gates(
    preflight_repo: Path,
) -> None:
    """Dirty paths are explicit when interim diff-scoped gates only inspect HEAD."""
    lane_log = _write_lane_logging_stub(preflight_repo)
    tracked_file = preflight_repo / "tests" / "unit" / "test_dirty_tracked.py"
    tracked_file.parent.mkdir(parents=True, exist_ok=True)
    tracked_file.write_text("print('baseline')\n", encoding="utf-8")
    _git(preflight_repo, "add", "-A")
    _git(preflight_repo, "commit", "-q", "-m", "tracked dirty fixture")
    tracked_file.write_text("print('dirty tracked path')\n", encoding="utf-8")

    dirty_file = preflight_repo / "tests" / "planner" / "test_dirty_optional.py"
    dirty_file.parent.mkdir(parents=True, exist_ok=True)
    dirty_file.write_text("print('dirty optional lane')\n", encoding="utf-8")

    result = _run_pr_ready(
        preflight_repo,
        help_flag=False,
        env_overrides={
            "BASE_REF": "HEAD",
            "PR_READY_MODE": "interim",
        },
    )

    assert result.returncode == 0, result.stderr
    assert lane_log.read_text(encoding="utf-8").splitlines() == ["core --lane core"]
    assert "Interim changed-file scope is committed HEAD vs HEAD." in result.stderr
    assert "Dirty paths excluded from diff-scoped gates:" in result.stderr
    assert "tests/unit/test_dirty_tracked.py" in result.stderr
    assert "tests/planner/test_dirty_optional.py" in result.stderr


def test_core_lane_collection_hook_skips_optional_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pytest collection hook should keep optional files out of the core lane."""
    import tests.conftest as test_conftest

    monkeypatch.setenv("ROBOT_SF_TEST_LANE", "core")
    assert (
        test_conftest.pytest_ignore_collect(Path("tests/planner/test_sonic_crowdnav.py"), None)
        is True
    )
    assert (
        test_conftest.pytest_ignore_collect(Path("tests/unit/test_config_validation.py"), None)
        is False
    )
    assert (
        test_conftest.pytest_ignore_collect(Path("tests/dev/test_pr_ready_preflight.py"), None)
        is False
    )
    assert (
        test_conftest._is_optional_readiness_test_path(
            "/tmp/tests-parent/repo/tests/planner/test_sonic_crowdnav.py::test_case"
        )
        is True
    )


def test_optional_lane_collection_hook_skips_core_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """The optional lane should collect only optional-extra test paths."""
    import tests.conftest as test_conftest

    monkeypatch.setenv("ROBOT_SF_TEST_LANE", "optional")
    assert (
        test_conftest.pytest_ignore_collect(Path("tests/planner/test_sonic_crowdnav.py"), None)
        is False
    )
    assert (
        test_conftest.pytest_ignore_collect(Path("tests/dev/test_pr_ready_preflight.py"), None)
        is True
    )


def test_missing_base_ref_falls_back_to_head_without_crashing(preflight_repo: Path) -> None:
    """A BASE_REF that does not resolve must fall back to HEAD, not leak a git fatal.

    Regression for issue #3702: on fresh checkouts the default origin/main is not
    always present. The unresolved ref previously leaked a raw
    ``fatal: ambiguous argument`` from the ``git diff "$BASE_REF...HEAD"`` call
    (silently swallowed by ``set -e`` inside the process substitution) and left an
    invalid base ref flowing to the downstream coverage/perf/freshness checks.
    """
    _make_fake_bin(preflight_repo, fail=True)
    result = _run_pr_ready(
        preflight_repo,
        help_flag=False,
        env_overrides={
            "BASE_REF": "non_existent_branch",
            "PR_READY_MODE": "interim",
        },
    )
    assert result.returncode == 0, f"Script failed: {result.stderr}"
    assert "Falling back to BASE_REF=HEAD" in result.stderr
    # The raw git error must not leak; the missing ref is handled gracefully.
    assert "fatal" not in result.stderr.lower()
    assert "unknown revision" not in result.stderr.lower()


def test_valid_base_ref_is_used_unchanged(preflight_repo: Path) -> None:
    """A BASE_REF that resolves must be used as-is, with no fallback or fetch.

    Guards against the issue #3702 fix degrading the normal path: a valid ref
    (here HEAD~1) should never trigger the HEAD fallback or a fetch attempt.
    """
    _make_fake_bin(preflight_repo, fail=True)
    extra = preflight_repo / "tests" / "unit" / "test_valid_base_ref.py"
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_text("print('valid base ref')\n", encoding="utf-8")
    _git(preflight_repo, "add", "-A")
    _git(preflight_repo, "commit", "-q", "-m", "valid base ref change")

    result = _run_pr_ready(
        preflight_repo,
        help_flag=False,
        env_overrides={
            "BASE_REF": "HEAD~1",
            "PR_READY_MODE": "interim",
        },
    )
    assert result.returncode == 0, f"Script failed: {result.stderr}"
    assert "Falling back to BASE_REF=HEAD" not in result.stderr
    assert "does not resolve to a local commit" not in result.stderr
    assert "Attempting git fetch" not in result.stderr


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
