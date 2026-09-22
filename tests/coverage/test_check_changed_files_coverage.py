"""Tests for changed-files coverage gate filtering and immutable verdicts."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.coverage.check_changed_files_coverage import (
    _changed_files,
    _changed_line_coverage_details,
    _changed_line_numbers,
    _coverage_for_changed_lines,
    _declaration_only_class_base_requirements,
    _declaration_proofs_from_test_source,
    _has_declaration_only_test_proof,
    _has_rename_only_reuse_proof,
    _is_doc_or_comment_only_python_change,
    _no_changed_files_message,
    _rename_only_callee_swaps,
    _resolve_comparison,
    _run_check,
)


def _git(repo: Path, *args: str) -> str:
    """Run one small fixture-repository Git command."""
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _fixture_repo(tmp_path: Path) -> tuple[Path, str, str]:
    """Create two exact commits for immutable base/head coverage checks."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "coverage@example.invalid")
    _git(repo, "config", "user.name", "coverage-fixture")
    source = repo / "robot_sf" / "feature.py"
    source.parent.mkdir()
    source.write_text("def answer() -> int:\n    return 1\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "base")
    base_sha = _git(repo, "rev-parse", "HEAD")
    source.write_text(
        "def answer() -> int:\n    return 1\n\ndef changed() -> int:\n    return 2\n",
        encoding="utf-8",
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "head")
    head_sha = _git(repo, "rev-parse", "HEAD")
    return repo, base_sha, head_sha


def test_empty_changed_file_message_names_committed_scope() -> None:
    """An empty HEAD diff must not hide dirty interim paths from the receipt."""
    assert _no_changed_files_message("origin/main") == "No committed changed files vs origin/main."


def test_doc_or_comment_only_python_change_detection() -> None:
    """Docstrings and comments should not require behavior coverage."""
    before = '''"""Old module docs."""

def answer() -> int:
    """Old helper docs."""
    return 1
'''
    after = '''"""New module docs."""

# Expanded context for maintainers.
def answer() -> int:
    """New helper docs."""
    # The behavior is unchanged.
    return 1
'''

    assert _is_doc_or_comment_only_python_change(before, after)
    assert not _is_doc_or_comment_only_python_change(before, after.replace("return 1", "return 2"))


def test_changed_line_coverage_uses_only_executable_changed_statements() -> None:
    """Large legacy files should not fail when the edited executable lines are covered."""
    file_data = {
        "executed_lines": [2, 5, 20],
        "missing_lines": [6, 21, 22],
    }

    coverage, scope = _coverage_for_changed_lines(
        file_data=file_data,
        changed_lines={1, 2, 5, 6},
    )

    assert coverage == 100.0 * 2 / 3
    assert scope == "changed executable lines 2/3"


def test_changed_line_coverage_treats_non_executable_edits_as_covered() -> None:
    """Comment-only diff lines that survive AST filtering should not create coverage debt."""
    coverage, scope = _coverage_for_changed_lines(
        file_data={"executed_lines": [10], "missing_lines": [20]},
        changed_lines={1, 2},
    )

    assert coverage == 100.0
    assert scope == "changed executable lines 0/0"


def test_empty_changed_lines_with_coverage_row_are_fully_covered() -> None:
    """Pure deletions have no new executable lines and therefore pass at 100 percent."""
    coverage, scope, changed, covered, missing = _changed_line_coverage_details(
        file_data={"executed_lines": [10], "missing_lines": [20]},
        changed_lines=set(),
    )

    assert coverage == 100.0
    assert scope == "changed executable lines 0/0"
    assert changed == []
    assert covered == []
    assert missing == []


def test_changed_line_coverage_fails_closed_on_malformed_file_data() -> None:
    """A coverage row without executable-line arrays cannot prove changed coverage."""
    coverage, scope, changed, covered, missing = _changed_line_coverage_details(
        file_data={"summary": {}},
        changed_lines={3},
    )

    assert coverage is None
    assert scope == "coverage malformed"
    assert changed == [3]
    assert covered is None
    assert missing is None


def test_explicit_base_and_head_shas_are_resolved_and_bound(tmp_path: Path) -> None:
    """Hosted-style inputs must resolve exact commits and reject checkout drift."""
    repo, base_sha, head_sha = _fixture_repo(tmp_path)
    args = SimpleNamespace(base="origin/main", base_sha=base_sha, head_sha=head_sha)

    comparison = _resolve_comparison(args, repo)

    assert comparison.base_sha == base_sha
    assert comparison.head_sha == head_sha
    with pytest.raises(RuntimeError, match="checked-out HEAD mismatch"):
        _resolve_comparison(
            SimpleNamespace(base="origin/main", base_sha=base_sha, head_sha=base_sha),
            repo,
        )


def test_machine_verdict_binds_changed_files_and_coverage_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The opt-in report carries exact identities, line evidence, and no-merge policy."""
    repo, base_sha, head_sha = _fixture_repo(tmp_path)
    coverage_path = repo / "coverage.json"
    coverage_path.write_text(
        json.dumps(
            {
                "files": {
                    "robot_sf/feature.py": {
                        "executed_lines": [1, 2, 4, 5],
                        "missing_lines": [],
                        "summary": {"percent_covered": 100.0},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    report_path = repo / "changed-coverage.json"
    monkeypatch.setattr("scripts.coverage.check_changed_files_coverage._repo_root", lambda: repo)
    args = SimpleNamespace(
        base="origin/main",
        base_sha=base_sha,
        head_sha=head_sha,
        event_name="pull_request",
        coverage=str(coverage_path),
        min=80.0,
        goal=100.0,
        include=[],
        exclude=[],
        show_skipped=False,
        json_output=report_path,
        json=False,
    )

    assert _run_check(args) == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["schema"] == "changed-coverage.v1"
    assert report["base_sha"] == base_sha
    assert report["head_sha"] == head_sha
    assert report["event"] == "pull_request"
    assert report["coverage_artifact"]["sha256"]
    assert report["verdict"] == "passed"
    assert report["no_merge"] is True
    assert report["files"][0]["missing_changed_lines"] == []


def test_declaration_only_base_change_accepts_parametrized_issubclass_proof() -> None:
    """A changed test can prove a safe class-base migration without a module reload."""
    before = '''from legacy import RuntimeErrorBase

class DatasetError(RuntimeErrorBase):
    """Dataset failure."""
'''
    after = '''from errors import RobotSfError
from legacy import RuntimeErrorBase

class DatasetError(RobotSfError, RuntimeErrorBase):
    """Dataset failure."""
'''
    proof_test = """import pytest

_ERRORS = (DatasetError,)

@pytest.mark.parametrize("error_type", _ERRORS)
def test_compatibility(error_type):
    assert issubclass(error_type, RobotSfError)
"""

    requirements = _declaration_only_class_base_requirements(before, after)

    assert requirements == {("DatasetError", "RobotSfError")}
    assert requirements <= _declaration_proofs_from_test_source(proof_test)


def test_declaration_only_base_change_accepts_direct_dual_catch_proof() -> None:
    """A direct shared-base catch is an alternative compatibility proof."""
    proof_test = """import pytest

def test_compatibility():
    with pytest.raises(RobotSfError):
        raise DatasetError("shared catch")
"""

    assert ("DatasetError", "RobotSfError") in _declaration_proofs_from_test_source(proof_test)


def test_declaration_only_base_change_rejects_a_class_body_change() -> None:
    """The exemption cannot hide changes to behavior inside the class body."""
    before = '''class DatasetError(RuntimeError):
    """Dataset failure."""
'''
    after = '''from errors import RobotSfError

class DatasetError(RobotSfError, RuntimeError):
    """Changed behavior."""
'''

    assert _declaration_only_class_base_requirements(before, after) is None


def test_declaration_only_base_change_rejects_runtime_body_changes() -> None:
    """The exemption cannot hide a changed function beside the class declaration."""
    before = '''class DatasetError(RuntimeError):
    """Dataset failure."""

def retry_count() -> int:
    return 1
'''
    after = '''from errors import RobotSfError

class DatasetError(RobotSfError, RuntimeError):
    """Dataset failure."""

def retry_count() -> int:
    return 2
'''

    assert _declaration_only_class_base_requirements(before, after) is None


def test_declaration_only_base_change_rejects_an_unrepresentable_added_base() -> None:
    """An added base without a direct proof name must fail closed."""
    before = "class DatasetError(RuntimeError):\n    pass\n"
    after = """from errors import RobotSfError

class DatasetError(RobotSfError, marker(), RuntimeError):
    pass
"""

    assert _declaration_only_class_base_requirements(before, after) is None


def test_changed_test_proof_enables_declaration_only_coverage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The coverage path requires a changed test, not just a matching source diff."""
    source_path = tmp_path / "robot_sf" / "errors.py"
    source_path.parent.mkdir()
    source_path.write_text(
        "from errors import RobotSfError\n\nclass DatasetError(RobotSfError, RuntimeError):\n    pass\n",
        encoding="utf-8",
    )
    test_path = tmp_path / "tests" / "test_errors.py"
    test_path.parent.mkdir()
    test_path.write_text(
        "from robot_sf.errors import DatasetError\n\n"
        "def test_compatibility():\n    assert issubclass(DatasetError, RobotSfError)\n",
        encoding="utf-8",
    )
    before = "class DatasetError(RuntimeError):\n    pass\n"
    monkeypatch.setattr(
        "scripts.coverage.check_changed_files_coverage._file_at_ref",
        lambda *args: before,
    )

    assert _has_declaration_only_test_proof(
        Path("robot_sf/errors.py"),
        "origin/main",
        tmp_path,
        [Path("robot_sf/errors.py"), Path("tests/test_errors.py")],
    )


def test_changed_test_proof_rejects_an_unrelated_same_named_class(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A proof must import the declaration from the changed source module."""
    source_path = tmp_path / "robot_sf" / "errors.py"
    source_path.parent.mkdir()
    source_path.write_text(
        "from errors import RobotSfError\n\nclass DatasetError(RobotSfError, RuntimeError):\n    pass\n",
        encoding="utf-8",
    )
    test_path = tmp_path / "tests" / "test_errors.py"
    test_path.parent.mkdir()
    test_path.write_text(
        "class DatasetError(RobotSfError):\n    pass\n\n"
        "def test_compatibility():\n    assert issubclass(DatasetError, RobotSfError)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "scripts.coverage.check_changed_files_coverage._file_at_ref",
        lambda *args: "class DatasetError(RuntimeError):\n    pass\n",
    )

    assert not _has_declaration_only_test_proof(
        Path("robot_sf/errors.py"),
        "origin/main",
        tmp_path,
        [Path("robot_sf/errors.py"), Path("tests/test_errors.py")],
    )


def test_changed_line_parser_ignores_no_newline_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unified diff metadata lines should not advance the new-file line counter."""

    def fake_run(*args, **kwargs):
        del args, kwargs
        return SimpleNamespace(
            returncode=0,
            stdout="\n".join(
                [
                    "diff --git a/demo.py b/demo.py",
                    "index 0000000..1111111 100644",
                    "--- a/demo.py",
                    "+++ b/demo.py",
                    "@@ -1 +1,2 @@",
                    "+first",
                    "\\ No newline at end of file",
                    "+second",
                ]
            ),
        )

    monkeypatch.setattr("scripts.coverage.check_changed_files_coverage.subprocess.run", fake_run)

    assert _changed_line_numbers("origin/main", Path("demo.py"), Path(".")) == {1, 2}


def test_pure_rename_reports_zero_changed_lines_and_100_percent_coverage(
    tmp_path: Path,
) -> None:
    """A pure ``git mv`` must not count every moved line as changed (issue #7552).

    With ``--find-renames=50%`` the rename is detected, so the moved file has no
    changed line numbers and the existing ``changed executable lines 0/0`` path
    reports 100.0 instead of gating the whole moved module.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "coverage@example.invalid")
    _git(repo, "config", "user.name", "coverage-fixture")
    source = repo / "robot_sf" / "old" / "feature.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "def answer() -> int:\n    return 1\n\ndef extra() -> int:\n    return 2\n",
        encoding="utf-8",
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "base")
    base_sha = _git(repo, "rev-parse", "HEAD")

    moved = repo / "robot_sf" / "new" / "feature.py"
    moved.parent.mkdir(parents=True)
    source.rename(moved)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "rename")
    head_sha = _git(repo, "rev-parse", "HEAD")

    # The renamed file must be reported (it is part of the diff as a rename), but
    # the per-file changed-line diff must yield no changed line numbers.
    moved_rel = Path("robot_sf/new/feature.py")
    assert moved_rel in _changed_files(base_sha, repo, head=head_sha)
    assert _changed_line_numbers(base_sha, moved_rel, repo, head=head_sha) == set()

    # End-to-end: the coverage row for the moved file is fully covered.
    details = _changed_line_coverage_details(
        file_data={"executed_lines": [1, 4], "missing_lines": []},
        changed_lines=_changed_line_numbers(base_sha, moved_rel, repo, head=head_sha),
    )
    assert details[0] == 100.0
    assert details[1] == "changed executable lines 0/0"


def test_renamed_and_substantially_modified_file_stays_gated(tmp_path: Path) -> None:
    """A move plus real content change must still report the new lines (issue #7552).

    Rename detection (``--find-renames=50%``) pairs the old and new paths but the
    added lines remain changed: only the truly new hunk lines count, not the
    whole moved file.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "coverage@example.invalid")
    _git(repo, "config", "user.name", "coverage-fixture")
    source = repo / "robot_sf" / "old" / "feature.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "def answer() -> int:\n    return 1\n\ndef extra() -> int:\n    return 2\n",
        encoding="utf-8",
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "base")
    base_sha = _git(repo, "rev-parse", "HEAD")

    moved = repo / "robot_sf" / "new" / "feature.py"
    moved.parent.mkdir(parents=True)
    source.rename(moved)
    moved.write_text(
        "def answer() -> int:\n    return 1\n\ndef extra() -> int:\n    return 2\n\n"
        "def new() -> int:\n    return 3\n",
        encoding="utf-8",
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "rename and modify")
    head_sha = _git(repo, "rev-parse", "HEAD")

    moved_rel = Path("robot_sf/new/feature.py")
    changed_lines = _changed_line_numbers(base_sha, moved_rel, repo, head=head_sha)
    assert changed_lines == {6, 7, 8}  # only the new lines count, not the whole file

    details = _changed_line_coverage_details(
        file_data={"executed_lines": [1, 4, 8], "missing_lines": []},
        changed_lines=changed_lines,
    )
    assert details[0] == 100.0
    # Only executable changed lines count toward the scope label (line 8 is the
    # single executable statement among the newly added lines 6-8).
    assert details[1] == "changed executable lines 1/1"


def _rename_fixture_repo(tmp_path: Path, *, after_text: str) -> tuple[Path, str, str, Path]:
    """Create a two-commit repo with a helper module and a call-site module."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "coverage@example.invalid")
    _git(repo, "config", "user.name", "coverage-fixture")
    helper = repo / "robot_sf" / "common" / "helpers.py"
    helper.parent.mkdir(parents=True)
    helper.write_text(
        "def legacy(left: float, right: float) -> float:\n    return abs(left - right)\n\n"
        "def point_distance(left: float, right: float) -> float:\n"
        "    gap = left - right\n"
        "    return abs(gap)\n",
        encoding="utf-8",
    )
    call_site = repo / "robot_sf" / "analysis" / "align.py"
    call_site.parent.mkdir(parents=True)
    call_site.write_text(
        "from robot_sf.common.helpers import legacy\n\n"
        "def delta(left: float, right: float) -> float:\n"
        "    return legacy(left, right)\n",
        encoding="utf-8",
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "base")
    base_sha = _git(repo, "rev-parse", "HEAD")
    call_site.write_text(after_text, encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "rename call site")
    head_sha = _git(repo, "rev-parse", "HEAD")
    return repo, base_sha, head_sha, Path("robot_sf/analysis/align.py")


_RENAME_AFTER = (
    "from robot_sf.common.helpers import point_distance\n\n"
    "def delta(left: float, right: float) -> float:\n"
    "    return point_distance(left, right)\n"
)


def test_rename_only_callee_swap_detects_pure_call_site_rename() -> None:
    """A pure callee swap plus its import change must report the rename map."""
    before = (
        "from robot_sf.common.helpers import legacy\n\n"
        "def delta(left: float, right: float) -> float:\n"
        "    return legacy(left, right)\n"
    )
    result = _rename_only_callee_swaps(before, _RENAME_AFTER)
    assert result is not None
    swaps, swapped_lines = result
    assert swaps == {"legacy": "point_distance"}
    assert swapped_lines == {4}


def test_rename_only_callee_swap_rejects_extra_logic_change() -> None:
    """Any non-rename executable change must fail the rename-only proof."""
    before = "def delta(left: float, right: float) -> float:\n    return legacy(left, right)\n"
    after = (
        "def delta(left: float, right: float) -> float:\n"
        "    return point_distance(left, right) + 1.0\n"
    )
    assert _rename_only_callee_swaps(before, after) is None


def test_rename_only_reuse_proof_accepts_covered_target(tmp_path: Path) -> None:
    """Missing lines on a pure rename onto a covered callee must prove out (issue #9238)."""
    repo, base_sha, head_sha, rel = _rename_fixture_repo(tmp_path, after_text=_RENAME_AFTER)
    missing = [4]
    coverage_data = {
        "robot_sf/common/helpers.py": {"executed_lines": [5, 6, 7], "missing_lines": [2]},
    }
    assert _has_rename_only_reuse_proof(rel, base_sha, repo, coverage_data, missing, head=head_sha)


def test_rename_only_reuse_proof_rejects_uncovered_target(tmp_path: Path) -> None:
    """A rename onto an untested callee must stay gated (issue #9238)."""
    repo, base_sha, head_sha, rel = _rename_fixture_repo(tmp_path, after_text=_RENAME_AFTER)
    coverage_data = {
        "robot_sf/common/helpers.py": {"executed_lines": [2], "missing_lines": [5, 6, 7]},
    }
    assert not _has_rename_only_reuse_proof(rel, base_sha, repo, coverage_data, [4], head=head_sha)


def test_rename_only_reuse_proof_rejects_unresolvable_import(tmp_path: Path) -> None:
    """A rename onto an unknown name must fail closed (issue #9238)."""
    repo, base_sha, head_sha, rel = _rename_fixture_repo(
        tmp_path,
        after_text=(
            "from robot_sf.common.helpers import missing_name\n\n"
            "def delta(left: float, right: float) -> float:\n"
            "    return missing_name(left, right)\n"
        ),
    )
    coverage_data = {
        "robot_sf/common/helpers.py": {"executed_lines": [2, 5, 6, 7], "missing_lines": []},
    }
    assert not _has_rename_only_reuse_proof(rel, base_sha, repo, coverage_data, [4], head=head_sha)


def test_rename_only_reuse_proof_passes_gate_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A pure rename onto a covered callee must pass the gate (issue #9238).

    Reproduces closed PR #9233: the call-site lines are uncovered in the
    artifact, but the change only renames onto an executed helper, so the row
    earns the auditable exemption instead of failing the minimum.
    """
    repo, base_sha, head_sha, rel = _rename_fixture_repo(tmp_path, after_text=_RENAME_AFTER)
    coverage_path = repo / "coverage.json"
    coverage_path.write_text(
        json.dumps(
            {
                "files": {
                    "robot_sf/analysis/align.py": {
                        "executed_lines": [1, 3],
                        "missing_lines": [4],
                        "summary": {"percent_covered": 50.0},
                    },
                    "robot_sf/common/helpers.py": {
                        "executed_lines": [5, 6, 7],
                        "missing_lines": [2],
                        "summary": {"percent_covered": 75.0},
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    report_path = repo / "changed-coverage.json"
    monkeypatch.setattr("scripts.coverage.check_changed_files_coverage._repo_root", lambda: repo)
    args = SimpleNamespace(
        base="origin/main",
        base_sha=base_sha,
        head_sha=head_sha,
        event_name="pull_request",
        coverage=str(coverage_path),
        min=80.0,
        goal=100.0,
        include=[],
        exclude=[],
        show_skipped=False,
        json_output=report_path,
        json=False,
    )

    assert _run_check(args) == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["verdict"] == "passed"
    rows = [row for row in report["files"] if row["file"] == rel.as_posix()]
    assert len(rows) == 1
    assert rows[0]["scope"] == "rename-only reuse proof"
    assert rows[0]["rename_only_proof"] is True
    assert rows[0]["missing_changed_lines"] == [4]


def test_rename_only_callee_swap_tolerates_removed_dead_helper() -> None:
    """Deleting the swapped-out helper's own def must not void the proof (issue #9238).

    Closed PR #9233 removed the local ``_distance`` body along with renaming its
    call sites; a deletion adds no coverage debt.
    """
    before = (
        "from robot_sf.common.helpers import legacy\n\n"
        "def delta(left: float, right: float) -> float:\n"
        "    return legacy(left, right)\n\n\n"
        "def legacy(left: float, right: float) -> float:\n"
        "    return abs(left - right)\n"
    )
    result = _rename_only_callee_swaps(before, _RENAME_AFTER)
    assert result is not None
    assert result[0] == {"legacy": "point_distance"}


def test_rename_only_callee_swap_rejects_live_helper_removal() -> None:
    """Removing a still-referenced helper must fail the proof (issue #9238)."""
    before = (
        "def delta(left: float, right: float) -> float:\n"
        "    return legacy(left, right) + kept(left)\n\n\n"
        "def legacy(left: float, right: float) -> float:\n"
        "    return abs(left - right)\n\n\n"
        "def kept(left: float) -> float:\n"
        "    return abs(left)\n"
    )
    after = (
        "def delta(left: float, right: float) -> float:\n"
        "    return point_distance(left, right) + kept(left)\n"
    )
    assert _rename_only_callee_swaps(before, after) is None
