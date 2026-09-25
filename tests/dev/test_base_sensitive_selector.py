"""Tests for the explicit base-sensitive changed-file selector."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from scripts.dev.base_sensitive_selector import (
    BASE_SENSITIVE,
    BASELINE_SCHEMA,
    ORDINARY,
    SELECTOR_VERSION,
    UNKNOWN,
    classify_changed_files,
    find_base_sensitive_test_files,
    main,
)

SENSITIVE = ["tests/test_snapshot.py", "tests/test_tuple_contract.py"]


def test_selector_classifies_marker_file_intersection() -> None:
    result = classify_changed_files(
        ["robot_sf/runtime.py", "tests/test_snapshot.py"],
        sensitive_files=SENSITIVE,
    )

    assert result["status"] == BASE_SENSITIVE
    assert result["selector"] == SELECTOR_VERSION
    assert result["changed_sensitive_files"] == ["tests/test_snapshot.py"]


def test_selector_classifies_unrelated_change_as_ordinary() -> None:
    result = classify_changed_files(["scripts/dev/helper.py"], sensitive_files=SENSITIVE)

    assert result["status"] == ORDINARY
    assert result["changed_sensitive_files"] == []


def test_selector_fails_closed_when_inventory_is_missing() -> None:
    result = classify_changed_files(None, sensitive_files=SENSITIVE)

    assert result["status"] == UNKNOWN
    assert result["reason"] == "changed_file_inventory_unavailable"


def test_path_normalization_preserves_hidden_directory_names() -> None:
    """Normalization removes only an explicit relative prefix."""
    result = classify_changed_files(["./.agents/skills/example.md"], sensitive_files=SENSITIVE)

    assert result["status"] == ORDINARY
    assert result["changed_files"] == [".agents/skills/example.md"]


def _init_fixture_repo(repo_root: Path) -> None:
    subprocess.run(["git", "init", "-q", str(repo_root)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "-c",
            "user.email=t@example.com",
            "-c",
            "user.name=t",
            "commit",
            "--allow-empty",
            "-q",
            "-m",
            "init",
        ],
        check=True,
    )


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_selection_is_repository_bound_and_ignores_nested_copies(tmp_path: Path) -> None:
    """Only Git-tracked test files are selected (issue #8025 regression)."""
    _init_fixture_repo(tmp_path)
    tracked = tmp_path / "tests" / "test_tracked_marker.py"
    _write(tracked, f"def test_x():\n    assert {BASE_SENSITIVE!r} in {BASE_SENSITIVE!r}\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "tests/test_tracked_marker.py"], check=True)

    ignored_copies = [
        tmp_path / ".emdash" / "wt-a" / "tests" / "test_copy.py",
        tmp_path / ".worktrees" / "wt-b" / "tests" / "test_copy.py",
        tmp_path / "output" / "cache" / "tests" / "test_copy.py",
    ]
    for copy in ignored_copies:
        _write(copy, "def test_copy():\n    assert True\n")

    selected = find_base_sensitive_test_files(tmp_path)

    assert selected == ["tests/test_tracked_marker.py"]


def test_untracked_nested_worktree_test_is_excluded_even_with_marker(tmp_path: Path) -> None:
    """An untracked copy carrying the marker text is still excluded."""
    _init_fixture_repo(tmp_path)
    ignored_marker_copy = tmp_path / ".worktrees" / "wt" / "tests" / "test_marker_copy.py"
    _write(ignored_marker_copy, "base_sensitive = True\n")

    assert find_base_sensitive_test_files(tmp_path) == []


def test_git_failure_fails_closed(tmp_path: Path) -> None:
    """A broken Git index must fail closed instead of widening the selection."""
    (tmp_path / "tests").mkdir(parents=True)
    (tmp_path / "tests" / "test_x.py").write_text("base_sensitive\n", encoding="utf-8")
    (tmp_path / ".git").write_text("not a repository\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="git ls-files failed"):
        find_base_sensitive_test_files(tmp_path)


def test_real_repository_selection_contains_no_ignored_copies() -> None:
    """The real checkout's selection must never include ignored nested copies."""
    repo_root = Path(__file__).resolve().parents[2]
    selected = find_base_sensitive_test_files(repo_root)
    forbidden_prefixes = (".emdash/", ".worktrees/", "output/")
    assert selected, "selector found no base-sensitive files in the real repository"
    assert not any(path.startswith(forbidden_prefixes) for path in selected)


def _commit_fixture_files(repo_root: Path) -> None:
    subprocess.run(["git", "-C", str(repo_root), "add", "--all"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "-c",
            "user.email=t@example.com",
            "-c",
            "user.name=t",
            "commit",
            "-q",
            "-m",
            "fixture",
        ],
        check=True,
    )


def _set_origin_main_to_head(repo_root: Path) -> str:
    current_sha = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(
        ["git", "-C", str(repo_root), "update-ref", "refs/remotes/origin/main", current_sha],
        check=True,
    )
    return current_sha


def test_baseline_cli_executes_selected_suite_and_reports_base_sha(tmp_path: Path) -> None:
    """The executable path runs a real marker-selected test and reports its commit."""
    _init_fixture_repo(tmp_path)
    _write(
        tmp_path / "tests" / "test_marker.py",
        "import pytest\n"
        "pytestmark = pytest.mark.base_sensitive\n\n"
        "def test_marker():\n    assert True\n",
    )
    _commit_fixture_files(tmp_path)
    expected_sha = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    script = Path(__file__).resolve().parents[2] / "scripts" / "dev" / "base_sensitive_selector.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--repo-root",
            str(tmp_path),
            "--base-ref",
            "HEAD",
            "--json",
            "--timeout-seconds",
            "30",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["schema"] == BASELINE_SCHEMA
    assert report["status"] == "passed"
    assert report["observed_base_sha"] == expected_sha
    assert report["selected_test_files"] == ["tests/test_marker.py"]
    assert report["test_result"]["test_count"] == 1
    assert report["test_result"]["passed"] == 1


def test_baseline_blocks_dirty_checkout_before_running_tests(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Dirty working-tree bytes cannot be presented as exact-base evidence."""
    _init_fixture_repo(tmp_path)
    test_path = tmp_path / "tests" / "test_marker.py"
    _write(
        test_path,
        "import pytest\n"
        "pytestmark = pytest.mark.base_sensitive\n\n"
        "def test_marker():\n    assert True\n",
    )
    _commit_fixture_files(tmp_path)
    _set_origin_main_to_head(tmp_path)
    test_path.write_text(
        test_path.read_text(encoding="utf-8").replace("assert True", "assert False")
    )

    exit_code = main(["--repo-root", str(tmp_path), "--json"])

    report = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert report["status"] == "blocked"
    assert report["reason"] == "checkout_dirty"
    assert report["observed_base_sha"] is None
    assert report["dirty_paths"] == ["tests/test_marker.py"]
    assert "pytest_command" not in report
    assert report["base_ref"] == "origin/main"
    assert report["branch"] is not None


def test_baseline_uses_committed_bytes_when_skip_worktree_hides_a_change(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A skip-worktree edit cannot replace the exact commit's failing test."""
    _init_fixture_repo(tmp_path)
    test_path = tmp_path / "tests" / "test_marker.py"
    _write(
        test_path,
        "import pytest\n"
        "pytestmark = pytest.mark.base_sensitive\n\n"
        "def test_marker():\n    assert False\n",
    )
    _commit_fixture_files(tmp_path)
    expected_sha = _set_origin_main_to_head(tmp_path)
    subprocess.run(
        ["git", "-C", str(tmp_path), "update-index", "--skip-worktree", "tests/test_marker.py"],
        check=True,
    )
    test_path.write_text(
        test_path.read_text(encoding="utf-8").replace("assert False", "assert True")
    )
    status = subprocess.run(
        ["git", "-C", str(tmp_path), "status", "--porcelain=v1", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert status.stdout == ""

    exit_code = main(["--repo-root", str(tmp_path), "--base-ref", "HEAD", "--json"])

    report = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert report["status"] == "failed"
    assert report["observed_base_sha"] == expected_sha
    assert report["test_source_sha"] == expected_sha
    assert report["test_result"]["failed"] == 1
    assert report["test_result"]["passed"] == 0


def test_baseline_does_not_import_ignored_helpers_from_caller_checkout(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An ignored helper in the caller checkout cannot alter exact-base test results."""
    _init_fixture_repo(tmp_path)
    _write(tmp_path / ".gitignore", "baseline_injected_helper.py\n")
    _write(
        tmp_path / "tests" / "test_marker.py",
        "import pytest\n"
        "import baseline_injected_helper\n"
        "pytestmark = pytest.mark.base_sensitive\n\n"
        "def test_marker():\n    assert baseline_injected_helper.VALUE == 'trusted'\n",
    )
    _commit_fixture_files(tmp_path)
    current_sha = _set_origin_main_to_head(tmp_path)
    _write(tmp_path / "baseline_injected_helper.py", "VALUE = 'trusted'\n")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    status = subprocess.run(
        ["git", "-C", str(tmp_path), "status", "--porcelain=v1", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert status.stdout == ""

    exit_code = main(["--repo-root", str(tmp_path), "--base-ref", "HEAD", "--json"])

    report = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert report["status"] == "failed"
    assert report["observed_base_sha"] == current_sha
    assert report["test_source_sha"] == current_sha
    assert report["test_result"]["errors"] == 1
    assert "baseline_injected_helper" in report["test_result"]["stdout"]


def test_baseline_reports_symlink_loop_in_pythonpath_as_blocked(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An invalid inherited import path returns structured unavailable evidence."""
    _init_fixture_repo(tmp_path)
    _write(
        tmp_path / "tests" / "test_marker.py",
        "import pytest\n"
        "pytestmark = pytest.mark.base_sensitive\n\n"
        "def test_marker():\n    assert True\n",
    )
    _commit_fixture_files(tmp_path)
    current_sha = _set_origin_main_to_head(tmp_path)
    with tempfile.TemporaryDirectory(prefix="baseline-pythonpath-loop-") as loop_dir:
        loop = Path(loop_dir) / "loop"
        loop.symlink_to(loop)
        monkeypatch.setenv("PYTHONPATH", str(loop))
        resolve = Path.resolve

        def reject_loop(path: Path, *args: object, **kwargs: object) -> Path:
            if path == loop:
                raise RuntimeError(f"Symlink loop from {path}")
            return resolve(path, *args, **kwargs)

        monkeypatch.setattr(Path, "resolve", reject_loop)

        exit_code = main(["--repo-root", str(tmp_path), "--base-ref", "HEAD", "--json"])

    output = capsys.readouterr()
    report = json.loads(output.out)
    assert exit_code == 2
    assert report["status"] == "blocked"
    assert report["reason"] == "pytest_unavailable"
    assert report["observed_base_sha"] == current_sha
    assert "Symlink loop" in report["error"]
    assert output.err == ""


def test_baseline_blocks_junit_failures_when_pytest_exit_is_forced_to_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """JUnit failures remain authoritative if a pytest hook forces exit code zero."""
    _init_fixture_repo(tmp_path)
    _write(
        tmp_path / "conftest.py",
        "def pytest_sessionfinish(session, exitstatus):\n"
        "    if exitstatus == 1:\n        session.exitstatus = 0\n",
    )
    _write(
        tmp_path / "tests" / "test_marker.py",
        "import pytest\n"
        "pytestmark = pytest.mark.base_sensitive\n\n"
        "def test_pass():\n    assert True\n\n"
        "def test_fail():\n    assert False\n",
    )
    _commit_fixture_files(tmp_path)

    exit_code = main(["--repo-root", str(tmp_path), "--base-ref", "HEAD", "--json"])

    report = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert report["status"] == "failed"
    assert report["reason"] == "selected_tests_failed"
    assert report["test_result"]["returncode"] == 0
    assert report["test_result"]["passed"] == 1
    assert report["test_result"]["failed"] == 1


def test_baseline_blocks_all_skipped_suite(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A successful pytest exit with no passing tests is not baseline evidence."""
    _init_fixture_repo(tmp_path)
    _write(
        tmp_path / "tests" / "test_marker.py",
        "import pytest\n"
        "pytestmark = [pytest.mark.base_sensitive, pytest.mark.skip(reason='not runnable')]\n\n"
        "def test_marker():\n    assert True\n",
    )
    _commit_fixture_files(tmp_path)
    current_sha = _set_origin_main_to_head(tmp_path)

    exit_code = main(["--repo-root", str(tmp_path), "--json"])

    report = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert report["status"] == "blocked"
    assert report["reason"] == "no_tests_passed"
    assert report["observed_base_sha"] == current_sha
    assert report["test_result"]["test_count"] == 1
    assert report["test_result"]["passed"] == 0
    assert report["test_result"]["skipped"] == 1


def test_baseline_blocks_suite_with_no_collected_tests(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A marker-like string cannot make an all-deselected suite look successful."""
    _init_fixture_repo(tmp_path)
    _write(
        tmp_path / "tests" / "test_marker.py",
        "# base_sensitive is intentionally only a comment, not a pytest mark.\n\n"
        "def test_marker():\n    assert True\n",
    )
    _commit_fixture_files(tmp_path)
    current_sha = _set_origin_main_to_head(tmp_path)

    exit_code = main(["--repo-root", str(tmp_path), "--json"])

    report = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert report["status"] == "blocked"
    assert report["reason"] == "no_tests_executed"
    assert report["observed_base_sha"] == current_sha
    assert report["test_result"]["test_count"] == 0


def test_baseline_fails_closed_on_empty_selection(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A valid checkout with no marker files cannot report a passing baseline."""
    _init_fixture_repo(tmp_path)
    _write(tmp_path / "tests" / "test_plain.py", "def test_plain():\n    assert True\n")
    _commit_fixture_files(tmp_path)
    current_sha = _set_origin_main_to_head(tmp_path)

    exit_code = main(["--repo-root", str(tmp_path), "--json"])

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert exit_code == 2
    assert report["base_ref"] == "origin/main"
    assert report["observed_base_sha"] == current_sha
    assert report["status"] == "blocked"
    assert report["reason"] == "selection_empty"
    assert report["selection_count"] == 0
    assert "pytest_command" not in report


def test_baseline_fails_closed_when_checkout_is_behind_requested_base(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A stale checkout cannot be reported as a passing current-main baseline."""
    _init_fixture_repo(tmp_path)
    base_sha = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _write(tmp_path / "tests" / "test_plain.py", "def test_plain():\n    assert True\n")
    _commit_fixture_files(tmp_path)

    exit_code = main(["--repo-root", str(tmp_path), "--base-ref", base_sha, "--json"])

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert exit_code == 2
    assert report["status"] == "blocked"
    assert report["reason"] == "checkout_not_at_base_ref"
    assert report["observed_base_sha"] == base_sha


def test_baseline_fails_closed_on_detached_head(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A detached commit is not accepted as a branch-attached main baseline."""
    _init_fixture_repo(tmp_path)
    subprocess.run(
        ["git", "-C", str(tmp_path), "checkout", "--detach", "--quiet", "HEAD"],
        check=True,
    )

    exit_code = main(["--repo-root", str(tmp_path), "--json"])

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert exit_code == 2
    assert report["status"] == "blocked"
    assert report["reason"] == "checkout_detached"
    assert report["branch"] is None


def test_baseline_fails_closed_on_unavailable_base_ref(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An unresolved baseline ref is an error, even before selection can run."""
    _init_fixture_repo(tmp_path)

    exit_code = main(["--repo-root", str(tmp_path), "--base-ref", "missing-ref", "--json"])

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert exit_code == 2
    assert report["status"] == "blocked"
    assert report["reason"] == "base_ref_unavailable"
    assert report["observed_base_sha"] is None
