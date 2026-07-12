"""Tests for issue #5351's fail-closed successor-release input checker."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from robot_sf.benchmark.hierarchical_paired_release_inputs import (
    BLOCKED_MISSING_SUCCESSOR_ROWS,
    INPUTS_READY_ANALYSIS_NOT_RUN,
    HierarchicalPairedReleaseInputError,
    evaluate_hierarchical_paired_release_inputs,
    load_hierarchical_paired_release_input_manifest,
    validate_hierarchical_paired_release_input_manifest,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MANIFEST_PATH = (
    _REPO_ROOT / "configs/benchmarks/releases/hierarchical_paired_release_analysis_issue_5351.yaml"
)
_SCRIPT_PATH = _REPO_ROOT / "scripts/benchmark/check_hierarchical_paired_release_inputs.py"


def _manifest() -> dict[str, object]:
    """Load a fresh copy of the checked-in blocked manifest."""

    return copy.deepcopy(load_hierarchical_paired_release_input_manifest(_MANIFEST_PATH))


def test_checked_in_manifest_reports_missing_successor_rows_fail_closed() -> None:
    """The pre-release manifest cannot be mistaken for evidence or a completed analysis."""

    report = evaluate_hierarchical_paired_release_inputs(_manifest(), repo_root=_REPO_ROOT)

    assert report["status"] == BLOCKED_MISSING_SUCCESSOR_ROWS
    assert report["evidence_status"] == "not_benchmark_evidence"
    assert report["claim_gate"] == {
        "status": "blocked_analysis_not_run",
        "reason": "successor-release inputs are missing",
    }
    assert report["semantics"] == {
        "benchmark_metrics_changed": False,
        "analysis_executed": False,
        "claim_promotion": "none",
    }
    assert [row["status"] for row in report["protocol_conformance"]] == [
        BLOCKED_MISSING_SUCCESSOR_ROWS
    ] * 8


def test_present_durable_successor_rows_only_make_inputs_ready(tmp_path: Path) -> None:
    """Input presence still cannot promote an unrun analysis claim."""

    manifest = _manifest()
    successor_release = manifest["successor_release"]
    assert isinstance(successor_release, dict)
    successor_release.update(
        {
            "release_tag": "v2.0.0",
            "commit": "a" * 40,
            "typed_ledger_rows": "docs/context/evidence/release_successor/rows.jsonl",
        }
    )
    root = tmp_path
    rows_path = root / "docs/context/evidence/release_successor/rows.jsonl"
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    rows_path.write_text('{"schema_version":"EpisodeEventLedger.v2"}\n', encoding="utf-8")
    successor_release["typed_ledger_rows_sha256"] = hashlib.sha256(
        rows_path.read_bytes()
    ).hexdigest()
    report = evaluate_hierarchical_paired_release_inputs(manifest, repo_root=root)

    assert report["status"] == INPUTS_READY_ANALYSIS_NOT_RUN
    assert report["claim_gate"] == {
        "status": "blocked_analysis_not_run",
        "reason": "inputs are present but the hierarchical paired analysis has not run",
    }
    assert {row["status"] for row in report["protocol_conformance"]} == {
        "declared_pending_analysis"
    }


def test_output_or_parent_escaping_row_paths_fail_closed() -> None:
    """Temporary or unsafe paths cannot qualify as durable successor release rows."""

    manifest = _manifest()
    successor_release = manifest["successor_release"]
    assert isinstance(successor_release, dict)
    successor_release.update(
        {
            "release_tag": "v2.0.0",
            "commit": "b" * 40,
            "typed_ledger_rows": "output/release/rows.jsonl",
            "typed_ledger_rows_sha256": "c" * 64,
        }
    )

    report = evaluate_hierarchical_paired_release_inputs(manifest, repo_root=_REPO_ROOT)

    assert report["status"] == BLOCKED_MISSING_SUCCESSOR_ROWS
    assert report["blocking_prerequisites"] == [
        {
            "field": "successor_release.typed_ledger_rows",
            "reason": "typed-ledger rows must use a durable repository-relative non-output path",
        }
    ]


def test_symlinked_row_paths_fail_closed(tmp_path: Path) -> None:
    """A symlink cannot disguise a non-canonical source as durable release rows."""

    manifest = _manifest()
    successor_release = manifest["successor_release"]
    assert isinstance(successor_release, dict)
    target = tmp_path / "canonical-rows.jsonl"
    target.write_text('{"schema_version":"EpisodeEventLedger.v2"}\n', encoding="utf-8")
    rows_path = tmp_path / "docs/context/evidence/release_successor/rows.jsonl"
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    rows_path.symlink_to(target)
    successor_release.update(
        {
            "release_tag": "v2.0.0",
            "commit": "d" * 40,
            "typed_ledger_rows": "docs/context/evidence/release_successor/rows.jsonl",
            "typed_ledger_rows_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        }
    )

    report = evaluate_hierarchical_paired_release_inputs(manifest, repo_root=tmp_path)

    assert report["status"] == BLOCKED_MISSING_SUCCESSOR_ROWS
    assert report["blocking_prerequisites"] == [
        {
            "field": "successor_release.typed_ledger_rows",
            "reason": "typed-ledger rows must use a durable repository-relative non-output path",
        }
    ]


def test_manifest_requires_every_predeclared_protocol_delivery() -> None:
    """Dropping a protocol element cannot silently weaken the future conformance table."""

    manifest = _manifest()
    protocol = manifest["protocol"]
    assert isinstance(protocol, list)
    protocol.pop()

    with pytest.raises(HierarchicalPairedReleaseInputError, match="protocol ids must be"):
        validate_hierarchical_paired_release_input_manifest(manifest)


def test_cli_writes_blocked_machine_readable_report(tmp_path: Path) -> None:
    """The canonical checker exits nonzero while retaining a reviewable blocker report."""

    output = tmp_path / "input_report.json"
    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT_PATH),
            "--manifest",
            str(_MANIFEST_PATH),
            "--repo-root",
            str(_REPO_ROOT),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert BLOCKED_MISSING_SUCCESSOR_ROWS in result.stdout
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == BLOCKED_MISSING_SUCCESSOR_ROWS
