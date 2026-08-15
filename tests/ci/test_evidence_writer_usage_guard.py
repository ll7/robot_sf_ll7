"""Tests for the changed-file evidence-writer adoption guard."""

# evidence-writer-exempt: tests intentionally write temporary fixture files while testing the guard.

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

from scripts.ci.check_evidence_writer_usage import (
    EvidenceWriterInventoryFinding,
    _is_inventory_path,
    _run_inventory,
    check_changed_files,
    check_file,
    inventory_file,
)


def _write_fixture(tmp_path: Path, source: str, name: str = "fixture.py") -> Path:
    """Write a synthetic changed Python file for guard tests."""
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
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


def test_only_canonical_shared_writer_module_is_exempt(tmp_path: Path) -> None:
    """A sibling module under ``robot_sf.evidence`` cannot bypass the guard."""
    source = """
from pathlib import Path

OUTPUT = Path('docs/context/evidence/example/report.md')
OUTPUT.write_text('# report', encoding='utf-8')
"""
    canonical_writer = _write_fixture(
        tmp_path,
        source,
        name="robot_sf/evidence/writers.py",
    )
    sibling_module = _write_fixture(
        tmp_path,
        source,
        name="robot_sf/evidence/unchecked_writer.py",
    )

    assert check_file(canonical_writer) == []
    blockers = check_file(sibling_module)
    assert len(blockers) == 1
    assert "write_text" in blockers[0]


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


def test_issue_5903_prediction_mpc_factorial_fixture_writes_are_exempt(tmp_path: Path) -> None:
    """Issue #5903: the live regression must not flag merged PR #5880 fixture writes.

    PR #5880 introduced five ``write_text`` calls in
    ``tests/test_prediction_mpc_factorial.py`` (around lines 467, 469, 588, 619,
    and 623). Each writes a temporary config/registry fixture into ``tmp_path`` so
    the fail-closed readiness-gate readers can be exercised; none targets
    ``docs/context/evidence``, so no durable evidence can be produced. The file
    carries a justified ``# evidence-writer-exempt`` marker. This test locks that
    contract: the guard must classify the file as clean, while the separate
    ``test_markerless_direct_writer_is_caught`` test proves the same guard still
    fails a genuine evidence-tree write (the checker is not weakened globally).
    """
    repo_root = Path(__file__).resolve().parents[2]
    target = repo_root / "tests" / "test_prediction_mpc_factorial.py"
    assert target.is_file(), "expected tests/test_prediction_mpc_factorial.py to exist"
    source = target.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(target))
    tmp_path_names: set[str] = set()
    writer_receivers: list[ast.expr] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(child, ast.Name) and child.id == "tmp_path" for child in ast.walk(node.value)
        ):
            tmp_path_names.update(
                target_node.id for target_node in node.targets if isinstance(target_node, ast.Name)
            )
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"write_text", "write_bytes"}
        ):
            writer_receivers.append(node.func.value)

    assert writer_receivers, f"expected temporary fixture writes in {target}"
    assert all(
        isinstance(receiver, ast.Name) and receiver.id in tmp_path_names
        for receiver in writer_receivers
    ), "every direct writer in the exempt file must target a tmp_path-derived fixture"
    assert check_file(target) == []

    # Sanity: the guard still catches a genuine evidence-tree write replayed in a
    # sibling file, so the reconciliation did not weaken the global check.
    genuine = _write_fixture(
        tmp_path,
        """
from pathlib import Path
OUTPUT = Path('docs/context/evidence/example')
OUTPUT.joinpath('report.md').write_text('# report', encoding='utf-8')
""",
    )
    genuine_blockers = check_file(genuine)
    assert any("write_text" in blocker for blocker in genuine_blockers)


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


def test_private_helper_bypass_is_caught(tmp_path: Path) -> None:
    """A module-level ``_write`` helper called with an evidence path is caught.

    The private-helper bypass pattern forwards a path parameter to ``write_text`` /
    ``open`` / ``json.dump`` so the direct-write visitor cannot see the evidence
    target at the call site. The guard flags both the call (the evidence path is
    visible there) and the helper definition (the structural signal that the
    module wraps a raw write while producing evidence).
    """
    path = _write_fixture(
        tmp_path,
        """
from pathlib import Path
OUTPUT = Path('docs/context/evidence/example')


def _write(target, text):
    target.write_text(text)


_write(OUTPUT / 'report.md', '# report')
""",
    )
    blockers = check_file(path)
    assert blockers, "private helper bypass must be caught"
    assert any("_write()" in blocker and "evidence-tree" in blocker for blocker in blockers)
    assert any("defines _write()" in blocker for blocker in blockers)
    assert all("robot_sf.evidence.writers" in blocker for blocker in blockers)


def test_private_helper_path_alias_bypass_is_caught(tmp_path: Path) -> None:
    """A helper cannot hide its forwarded path behind a local alias."""
    path = _write_fixture(
        tmp_path,
        """
from pathlib import Path
OUTPUT = Path('docs/context/evidence/example')


def _write(target, text):
    resolved_target = target.resolve()
    resolved_target.write_text(text)


_write(OUTPUT / 'report.md', '# report')
""",
    )

    blockers = check_file(path)

    assert any("_write()" in blocker and "evidence-tree" in blocker for blocker in blockers)
    assert any("defines _write()" in blocker for blocker in blockers)


def test_cli_arg_bypass_is_caught(tmp_path: Path) -> None:
    """An ``args.output.write_text`` whose argparse default is evidence is caught.

    The CLI-arg bypass hides the evidence target behind an argparse destination
    whose default points at the evidence tree. The guard resolves the destination
    from ``add_argument(default=...)`` and treats ``args.output`` (and names
    derived from it) as evidence-mentioning.
    """
    path = _write_fixture(
        tmp_path,
        """
import argparse
from pathlib import Path

DEFAULT_OUTPUT = Path('docs/context/evidence/example/out.json')
parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
args = parser.parse_args()
output_path = Path(args.output)
output_path.write_text('{}', encoding='utf-8')
""",
    )
    blockers = check_file(path)
    assert len(blockers) == 1
    assert "write_text" in blockers[0]
    assert "robot_sf.evidence.writers" in blockers[0]


def test_cli_arg_short_and_long_options_use_argparse_destination(tmp_path: Path) -> None:
    """The first long option determines argparse's implicit destination name."""
    path = _write_fixture(
        tmp_path,
        """
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument(
    '-o',
    '--output',
    type=Path,
    default=Path('docs/context/evidence/example/out.json'),
)
args = parser.parse_args()
args.output.write_text('{}', encoding='utf-8')
""",
    )

    blockers = check_file(path)

    assert len(blockers) == 1
    assert "write_text" in blockers[0]


def test_cli_positional_arg_with_evidence_default_is_caught(tmp_path: Path) -> None:
    """A positional argparse destination can also default to evidence output."""
    path = _write_fixture(
        tmp_path,
        """
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument(
    'output',
    nargs='?',
    type=Path,
    default=Path('docs/context/evidence/example/out.json'),
)
args = parser.parse_args()
args.output.write_text('{}', encoding='utf-8')
""",
    )

    blockers = check_file(path)

    assert len(blockers) == 1
    assert "write_text" in blockers[0]


def test_path_built_from_separate_literal_components_is_caught(tmp_path: Path) -> None:
    """Split ``Path`` components cannot hide the evidence-tree destination."""
    path = _write_fixture(
        tmp_path,
        """
from pathlib import Path

OUTPUT = Path('docs') / 'context' / 'evidence' / 'example' / 'out.json'
OUTPUT.write_text('{}', encoding='utf-8')
""",
    )

    blockers = check_file(path)

    assert len(blockers) == 1
    assert "write_text" in blockers[0]


def test_keyword_only_private_helper_bypass_is_caught(tmp_path: Path) -> None:
    """A raw writer forwarded through a keyword-only path parameter is blocked."""
    path = _write_fixture(
        tmp_path,
        """
from pathlib import Path

OUTPUT = Path('docs/context/evidence/example/out.json')

def _write(*, target, text):
    target.write_text(text, encoding='utf-8')

_write(target=OUTPUT, text='{}')
""",
    )

    blockers = check_file(path)

    assert any("_write()" in blocker and "evidence-tree" in blocker for blocker in blockers)
    assert any("defines _write()" in blocker for blocker in blockers)


def test_interprocedural_helper_bypass_is_caught(tmp_path: Path) -> None:
    """A helper fed an evidence path through a keyword-only param is caught.

    Mirrors the canonical #4232 builder: the evidence path enters via a CLI dest,
    flows through a keyword-only ``output_dir`` parameter, and reaches a private
    ``_write`` helper whose call-site argument is the parameter (not a visible
    evidence literal). Only the structural helper-definition signal can catch
    this interprocedural bypass.
    """
    path = _write_fixture(
        tmp_path,
        """
import argparse
from pathlib import Path

DEFAULT_OUTPUT_DIR = Path('docs/context/evidence/example')


def _write(target, text):
    target.write_text(text)


def _emit(*, output_dir):
    _write(output_dir / 'report.md', '# report')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    _emit(output_dir=Path(args.output_dir))
""",
    )
    blockers = check_file(path)
    assert any("defines _write()" in blocker for blocker in blockers), (
        "interprocedural helper bypass must be caught by the structural signal"
    )


def test_shared_writer_routed_through_local_helper_passes(tmp_path: Path) -> None:
    """A helper that delegates to the shared writer is not a bypass."""
    path = _write_fixture(
        tmp_path,
        """
from pathlib import Path
from robot_sf.evidence.writers import write_json

OUTPUT = Path('docs/context/evidence/example')


def emit_report(target, payload):
    write_json(target, payload)


emit_report(OUTPUT / 'report.json', {'status': 'diagnostic-only'})
""",
    )
    assert check_file(path) == []


def test_non_evidence_module_with_tmp_helper_is_not_flagged(tmp_path: Path) -> None:
    """A private write helper for a non-evidence path is not a bypass.

    The structural helper-definition check only fires when the module produces
    evidence output. A helper writing a temporary debug file in a module with no
    evidence reference must stay clean so the guard does not over-flag ordinary
    code.
    """
    path = _write_fixture(
        tmp_path,
        """
from pathlib import Path


def _write_debug(target, text):
    target.write_text(text)


_write_debug(Path('/tmp/debug.log'), 'debug')
""",
    )
    assert check_file(path) == []


def test_argparse_dest_with_non_evidence_default_is_not_evidence(tmp_path: Path) -> None:
    """A CLI output whose default is not evidence is not flagged."""
    path = _write_fixture(
        tmp_path,
        """
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, default=Path('/tmp/out.json'))
args = parser.parse_args()
args.output.write_text('{}', encoding='utf-8')
""",
    )
    assert check_file(path) == []


def test_inventory_file_reports_raw_writer_with_exemption_status(tmp_path: Path) -> None:
    """Inventory mode reports residual bypasses instead of honoring exemptions."""
    path = _write_fixture(
        tmp_path,
        """
# evidence-writer-exempt: legacy JSONL byte contract awaiting migration.
from pathlib import Path

OUTPUT = Path("docs/context/evidence/example/out.json")
OUTPUT.write_text("{}", encoding="utf-8")
""",
    )

    findings = inventory_file(path, tmp_path)

    assert len(findings) == 1
    finding = findings[0]
    assert finding.path == "fixture.py"
    assert finding.line == 6
    assert finding.operation == ".write_text()"
    assert finding.exemption_status == "valid"
    assert finding.exemption_reason == "legacy JSONL byte contract awaiting migration."
    assert check_file(path) == []


def test_inventory_file_reports_unexempt_raw_writer(tmp_path: Path) -> None:
    """Inventory findings include the path, line, operation, and missing exemption."""
    path = _write_fixture(
        tmp_path,
        """
from pathlib import Path

OUTPUT = Path("docs/context/evidence/example/out.json")
OUTPUT.write_text("{}", encoding="utf-8")
""",
    )

    findings = inventory_file(path, tmp_path)

    assert len(findings) == 1
    assert findings[0].path == "fixture.py"
    assert findings[0].line == 5
    assert findings[0].operation == ".write_text()"
    assert findings[0].exemption_status == "none"


def test_inventory_file_reports_invalid_exemption_reason(tmp_path: Path) -> None:
    """Inventory mode preserves malformed exemption diagnostics instead of hiding them."""
    path = _write_fixture(
        tmp_path,
        """
# evidence-writer-exempt:
from pathlib import Path

OUTPUT = Path("docs/context/evidence/example/out.json")
OUTPUT.write_text("{}", encoding="utf-8")
""",
    )

    findings = inventory_file(path, tmp_path)

    assert len(findings) == 1
    assert findings[0].exemption_status == "invalid"
    assert "empty evidence-writer exemption reason" in findings[0].exemption_reason
    assert check_file(path)


def test_inventory_file_preserves_contiguous_exemption_reason(tmp_path: Path) -> None:
    """Inventory mode joins top-level continuation comments into the reason."""
    path = _write_fixture(
        tmp_path,
        """
# evidence-writer-exempt: first line of the immutable byte-contract reason
# second line explains why the shared marker cannot be added
from pathlib import Path

OUTPUT = Path("docs/context/evidence/example/out.json")
OUTPUT.write_text("{}", encoding="utf-8")
""",
    )

    findings = inventory_file(path, tmp_path)

    assert len(findings) == 1
    assert findings[0].exemption_reason == (
        "first line of the immutable byte-contract reason "
        "second line explains why the shared marker cannot be added"
    )


def test_inventory_file_reports_malformed_python_as_controlled_finding(tmp_path: Path) -> None:
    """Malformed tracked Python produces a deterministic error finding, not a traceback."""
    path = _write_fixture(tmp_path, "def broken(:\n    pass\n")

    findings = inventory_file(path, tmp_path)

    assert findings == [
        EvidenceWriterInventoryFinding(
            path="fixture.py",
            line=1,
            operation="parse",
            kind="error",
            exemption_status="invalid",
            exemption_reason="cannot parse source: invalid syntax",
        )
    ]


def test_inventory_json_returns_nonzero_for_scan_errors(monkeypatch, capsys) -> None:
    """Inventory JSON remains parseable and returns nonzero for scan errors."""
    error = EvidenceWriterInventoryFinding(
        path="broken.py",
        line=1,
        operation="parse",
        kind="error",
        exemption_status="invalid",
        exemption_reason="cannot parse source: invalid syntax",
    )
    monkeypatch.setattr(
        "scripts.ci.check_evidence_writer_usage.inventory_tracked_files",
        lambda: ([error], 1),
    )

    assert _run_inventory(json_output=True) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["count"] == 1
    assert payload["findings"][0]["kind"] == "error"


def test_inventory_path_filter_excludes_benchmark_owned_paths() -> None:
    """Inventory mode stays out of benchmark-owned Python paths."""
    assert not _is_inventory_path("scripts/benchmark/run_case.py")
    assert not _is_inventory_path("robot_sf/benchmark/metrics.py")
    assert not _is_inventory_path("tests/benchmark/test_case.py")
    assert not _is_inventory_path("tests/analysis/test_case.py")
    assert _is_inventory_path("scripts/analysis/build_case.py")
    assert _is_inventory_path("hooks/check_release.py")
    assert _is_inventory_path("robot_sf/evidence/case.py")
    assert not _is_inventory_path("scripts/analysis/README.md")


def test_cli_help_advertises_inventory_and_json() -> None:
    """The backward-compatible CLI documents the new optional modes."""
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "ci" / "check_evidence_writer_usage.py"

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--changed-files-file" in result.stdout
    assert "--inventory" in result.stdout
    assert "--json" in result.stdout


def test_cli_changed_files_json_output_preserves_guard_failure(tmp_path: Path) -> None:
    """JSON output does not change changed-file guard exit semantics."""
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "ci" / "check_evidence_writer_usage.py"
    fixture = _write_fixture(
        tmp_path,
        """
from pathlib import Path

OUTPUT = Path("docs/context/evidence/example/out.json")
OUTPUT.write_text("{}", encoding="utf-8")
""",
    )
    changed_files = tmp_path / "changed-files.txt"
    changed_files.write_text(f"{fixture}\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--changed-files-file",
            str(changed_files),
            "--base-ref",
            "missing-base-for-test",
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert result.returncode == 1
    assert payload["mode"] == "changed-files"
    assert payload["count"] == 1
    assert "write_text" in payload["blockers"][0]


def test_cli_inventory_json_smoke_current_checkout() -> None:
    """Inventory CLI emits deterministic JSON from a nested checkout directory."""
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "ci" / "check_evidence_writer_usage.py"

    result = subprocess.run(
        [sys.executable, str(script), "--inventory", "--json"],
        check=True,
        cwd=repo_root / "robot_sf",
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["mode"] == "inventory"
    assert payload["approved_prefixes"] == ["hooks/", "scripts/", "robot_sf/"]
    assert payload["excluded_prefixes"] == [
        "scripts/benchmark/",
        "robot_sf/benchmark/",
        "tests/benchmark/",
    ]
    assert isinstance(payload["scanned_paths"], int)
    assert isinstance(payload["findings"], list)
    assert payload["count"] == len(payload["findings"])
    paths = [finding["path"] for finding in payload["findings"]]
    assert paths == sorted(paths)
    assert all(_is_inventory_path(path) for path in paths)
