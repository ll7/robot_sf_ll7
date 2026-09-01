"""Tests for semantic post-anchor capability delta builder (issue #8046)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.analysis.build_post_anchor_capability_delta import (
    SCHEMA,
    SUMMARY_SCHEMA,
    VALID_STATUSES,
    build_all,
    check_all,
    get_canonical_capabilities,
)

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "analysis"
    / "build_post_anchor_capability_delta.py"
)


def test_canonical_capabilities_structure() -> None:
    """Verify canonical capabilities meet schema and taxonomy rules."""
    rows = get_canonical_capabilities()
    assert len(rows) >= 10
    for row in rows:
        assert row.status in VALID_STATUSES
        assert row.category in (
            "planner_and_policy",
            "diagnostic_method",
            "prototype_transfer_bridge",
            "operational_reproducibility",
        )
        assert len(row.owner_paths) >= 1
        assert len(row.linked_issues) >= 1
        assert len(row.missing_proof) >= 1
        assert len(row.strongest_permitted_statement) >= 10


def test_post_anchor_planners_not_release_evaluated() -> None:
    """Verify post-anchor planners are not classified as released benchmark evidence."""
    rows = get_canonical_capabilities()
    for row in rows:
        if row.category == "planner_and_policy" and row.status == "introduced_after_anchor":
            assert row.evidence_status in (
                "diagnostic_only",
                "synthetic_fixture",
                "unsupported_proxy",
            )
            assert row.release_relationship == "unreleased_prototype"
            assert (
                "not evaluated against benchmark release suites"
                in row.strongest_permitted_statement
                or "benchmark-grade evaluation unestablished" in row.strongest_permitted_statement
                or "full training campaign results unverified" in row.strongest_permitted_statement
            )


def test_build_all_creates_valid_json_and_markdown(tmp_path: Path) -> None:
    """Test generating JSON dataset and Markdown summary into temporary path."""
    json_path = tmp_path / "delta.json"
    summary_path = tmp_path / "summary.md"

    result = build_all(json_path, summary_path)
    assert result["schema"] == SUMMARY_SCHEMA
    assert result["row_count"] >= 10
    assert json_path.exists()
    assert summary_path.exists()

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["schema"] == SCHEMA
    assert data["row_count"] == result["row_count"]

    summary_text = summary_path.read_text(encoding="utf-8")
    assert "## 1. Evidence Anchor" in summary_text
    assert "## 2. Substantive Research / Method Additions" in summary_text
    assert "## 3. Future-Work Bridge Progress" in summary_text
    assert "## 4. Operational / Reproducibility Growth" in summary_text
    assert "## 5. Conflicts and Unknowns" in summary_text


def test_check_all_detects_drift(tmp_path: Path) -> None:
    """Test that check_all reports matching vs drifted state."""
    json_path = tmp_path / "delta.json"
    summary_path = tmp_path / "summary.md"

    build_all(json_path, summary_path)
    assert check_all(json_path, summary_path) is True

    # Mutate json
    data = json.loads(json_path.read_text(encoding="utf-8"))
    data["row_count"] = 999
    json_path.write_text(json.dumps(data), encoding="utf-8")

    assert check_all(json_path, summary_path) is False


def test_cli_check_and_json(tmp_path: Path) -> None:
    """Test CLI invocation with --check and --json."""
    json_path = tmp_path / "delta.json"
    summary_path = tmp_path / "summary.md"

    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--json-file",
            str(json_path),
            "--summary-file",
            str(summary_path),
            "--json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    res = json.loads(proc.stdout)
    assert res["schema"] == SUMMARY_SCHEMA

    proc_check = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--json-file",
            str(json_path),
            "--summary-file",
            str(summary_path),
            "--check",
            "--json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc_check.returncode == 0
    res_check = json.loads(proc_check.stdout)
    assert res_check["ok"] is True
