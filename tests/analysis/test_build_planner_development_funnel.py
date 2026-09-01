"""Tests for planner-development funnel and selection trace builder (issue #8045)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.analysis.build_planner_development_funnel import (
    CANONICAL_14_RELEASE_ROSTER,
    SCHEMA,
    SUMMARY_SCHEMA,
    VALID_RELATIONSHIPS,
    VALID_SELECTION_BIAS_NOTES,
    build_all,
    check_all,
    get_canonical_candidate_records,
)

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "analysis"
    / "build_planner_development_funnel.py"
)


def test_canonical_candidate_records_structure() -> None:
    """Verify candidate records meet schema and taxonomy rules."""
    records = get_canonical_candidate_records()
    assert len(records) >= 15
    for r in records:
        assert r.relationship_to_release in VALID_RELATIONSHIPS
        assert r.selection_bias_note in VALID_SELECTION_BIAS_NOTES
        assert len(r.candidate_id) >= 2
        assert len(r.display_name) >= 2
        assert len(r.family) >= 2
        assert len(r.evidence_pointer) >= 5
        assert len(r.strongest_permitted_statement) >= 10


def test_release_roster_exact_keys() -> None:
    """Verify release roster matches the exact 14 canonical release keys."""
    records = get_canonical_candidate_records()
    release_keys = tuple(
        r.candidate_id for r in records if r.relationship_to_release == "included_exact_key"
    )
    assert len(release_keys) == 14
    assert release_keys == CANONICAL_14_RELEASE_ROSTER


def test_post_anchor_planners_segregated() -> None:
    """Verify post-anchor planners are strictly segregated from release roster."""
    records = get_canonical_candidate_records()
    for r in records:
        if r.relationship_to_release == "post_anchor":
            assert r.candidate_id not in CANONICAL_14_RELEASE_ROSTER
            assert (
                "excluded from dissertation release roster"
                in r.strongest_permitted_statement.lower()
            )


def test_build_all_creates_valid_json_and_markdown(tmp_path: Path) -> None:
    """Test generating JSON dataset and Markdown summary into temporary path."""
    json_path = tmp_path / "funnel.json"
    summary_path = tmp_path / "summary.md"

    result = build_all(json_path, summary_path)
    assert result["schema"] == SUMMARY_SCHEMA
    assert result["release_roster_count"] == 14
    assert result["total_candidate_count"] >= 15
    assert json_path.exists()
    assert summary_path.exists()

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["schema"] == SCHEMA
    assert data["release_roster_count"] == 14

    summary_text = summary_path.read_text(encoding="utf-8")
    assert "## 1. Dissertation-Facing Compact Funnel View" in summary_text
    assert "## 2. Frozen 14-Arm Release Roster Trace" in summary_text
    assert "## 3. Exploratory and Diagnostic Candidates" in summary_text
    assert "## 4. Post-Anchor Candidate Demarcation" in summary_text
    assert "## 5. Methodological Separation Summary" in summary_text


def test_check_all_detects_drift(tmp_path: Path) -> None:
    """Test that check_all reports matching vs drifted state."""
    json_path = tmp_path / "funnel.json"
    summary_path = tmp_path / "summary.md"

    build_all(json_path, summary_path)
    assert check_all(json_path, summary_path) is True

    # Mutate json
    data = json.loads(json_path.read_text(encoding="utf-8"))
    data["release_roster_count"] = 999
    json_path.write_text(json.dumps(data), encoding="utf-8")

    assert check_all(json_path, summary_path) is False


def test_cli_check_and_json(tmp_path: Path) -> None:
    """Test CLI invocation with --check and --json."""
    json_path = tmp_path / "funnel.json"
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
