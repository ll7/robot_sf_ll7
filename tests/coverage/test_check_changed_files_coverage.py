"""Tests for changed-files coverage gate filtering."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.coverage.check_changed_files_coverage import (
    _changed_line_numbers,
    _coverage_for_changed_lines,
    _declaration_only_class_base_requirements,
    _declaration_proofs_from_test_source,
    _has_declaration_only_test_proof,
    _is_doc_or_comment_only_python_change,
)


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
                    "@@ -1 +1,2 @@",
                    "+first",
                    "\\ No newline at end of file",
                    "+second",
                ]
            ),
        )

    monkeypatch.setattr("scripts.coverage.check_changed_files_coverage.subprocess.run", fake_run)

    assert _changed_line_numbers("origin/main", Path("demo.py"), Path(".")) == {1, 2}
