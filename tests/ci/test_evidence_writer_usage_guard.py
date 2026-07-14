"""Tests for the changed-file evidence-writer adoption guard."""

# evidence-writer-exempt: tests intentionally write temporary fixture files while testing the guard.

from __future__ import annotations

from typing import TYPE_CHECKING

from scripts.ci.check_evidence_writer_usage import check_changed_files, check_file

if TYPE_CHECKING:
    from pathlib import Path


def _write_fixture(tmp_path: Path, source: str, name: str = "fixture.py") -> Path:
    """Write a synthetic changed Python file for guard tests."""
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return path


def test_markerless_direct_writer_is_caught(tmp_path: Path) -> None:
    """A direct evidence-tree write must fail with an actionable message."""
    path = _write_fixture(
        tmp_path,
        """
from pathlib import Path
OUTPUT = Path('docs/context/evidence/example')
OUTPUT.joinpath('report.md').write_text('# report', encoding='utf-8')
""",
    )
    blockers = check_file(path)
    assert len(blockers) == 1
    assert "write_text" in blockers[0]
    assert "robot_sf.evidence.writers" in blockers[0]


def test_shared_writer_usage_passes(tmp_path: Path) -> None:
    """A generated evidence file written by the shared module passes."""
    path = _write_fixture(
        tmp_path,
        """
from pathlib import Path
from robot_sf.evidence.writers import write_json
OUTPUT = Path('docs/context/evidence/example')
write_json(OUTPUT / 'report.json', {'status': 'diagnostic-only'})
""",
    )
    assert check_file(path) == []


def test_exemption_text_in_string_does_not_bypass_guard(tmp_path: Path) -> None:
    """A string containing the exemption text is not a file-level comment."""
    path = _write_fixture(
        tmp_path,
        """
EXEMPTION_TEXT = "# evidence-writer-exempt: not a comment"
from pathlib import Path
OUTPUT = Path("docs/context/evidence/example")
OUTPUT.joinpath("report.md").write_text("# report", encoding="utf-8")
""",
    )
    blockers = check_file(path)
    assert len(blockers) == 1
    assert "write_text" in blockers[0]


def test_indented_exemption_comment_does_not_bypass_guard(tmp_path: Path) -> None:
    """An indented exemption comment is not a file-level exemption."""
    path = _write_fixture(
        tmp_path,
        """
if True:
    # evidence-writer-exempt: not file-level
    pass
from pathlib import Path
OUTPUT = Path("docs/context/evidence/example")
OUTPUT.joinpath("report.md").write_text("# report", encoding="utf-8")
""",
    )
    blockers = check_file(path)
    assert len(blockers) == 1
    assert "write_text" in blockers[0]


def test_evidence_path_read_does_not_classify_tmp_writes(tmp_path: Path) -> None:
    """Reading an evidence fixture does not make unrelated temporary writes blockers."""
    path = _write_fixture(
        tmp_path,
        """
from pathlib import Path
FIXTURE = Path("docs/context/evidence/example/report.json")
manifest_path = tmp_path / "manifest.json"
manifest_path.write_text("{}", encoding="utf-8")
FIXTURE.read_text(encoding="utf-8")
""",
    )
    assert check_file(path) == []


def test_handwritten_evidence_markdown_is_ignored(tmp_path: Path) -> None:
    """The Python-only guard does not classify a handwritten Markdown file."""
    path = tmp_path / "docs/context/evidence/README.md"
    path.parent.mkdir(parents=True)
    path.write_text("# Handwritten context\n", encoding="utf-8")
    assert check_changed_files([str(path)]) == []


def test_binary_sidecar_exemption_with_reason_passes(tmp_path: Path) -> None:
    """A justified binary sidecar exemption is accepted."""
    path = _write_fixture(
        tmp_path,
        """
# evidence-writer-exempt: binary PNG output cannot carry a text marker; SHA256SUMS is emitted by the shared writer.
from pathlib import Path
OUTPUT = Path('docs/context/evidence/example')
OUTPUT.joinpath('trace.png').write_bytes(b'png')
""",
    )
    assert check_file(path) == []


def test_exemption_without_reason_fails(tmp_path: Path) -> None:
    """An empty exemption cannot silence the guard."""
    path = _write_fixture(
        tmp_path,
        """
# evidence-writer-exempt:
from pathlib import Path
OUTPUT = Path('docs/context/evidence/example')
OUTPUT.joinpath('report.md').write_text('# report', encoding='utf-8')
""",
    )
    blockers = check_file(path)
    assert len(blockers) == 1
    assert "empty evidence-writer exemption reason" in blockers[0]


def test_pr_contract_check_reports_guard_violation(tmp_path: Path) -> None:
    """The parent PR contract check exposes the same guard blocker."""
    path = _write_fixture(
        tmp_path,
        """
from pathlib import Path
OUTPUT = Path('docs/context/evidence/example')
OUTPUT.joinpath('report.md').write_text('# report', encoding='utf-8')
""",
    )
    from scripts.ci.pr_contract_check import run_all_checks

    blockers, _, _ = run_all_checks("", "", [str(path)], "ll7/robot_sf_ll7", "origin/main", None)
    assert any("evidence-writer" in blocker for blocker in blockers)
