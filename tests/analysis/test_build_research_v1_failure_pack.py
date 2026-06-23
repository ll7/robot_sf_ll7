"""Tests for the issue #2159 research-v1 failure-case trace review pack builder."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.analysis.build_research_v1_failure_pack_issue_2159 import (
    CASE_INPUTS,
    build_case_report,
    build_manifest,
    build_readme,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_case_inputs_are_well_formed() -> None:
    """Each case input entry must have all required fields."""
    for case in CASE_INPUTS:
        assert "case_id" in case
        assert "evidence_dir" in case
        assert "slice_file" in case
        assert "report_title" in case
        assert "scenario_id" in case
        assert "planners" in case
        assert isinstance(case["planners"], list)
        assert "seeds" in case
        assert isinstance(case["seeds"], list)


def test_case_input_slice_files_exist() -> None:
    """Each case's trace slice file should exist in the repository."""
    for case in CASE_INPUTS:
        slice_path = REPO_ROOT / case["evidence_dir"] / case["slice_file"]
        assert slice_path.exists(), f"Missing: {slice_path}"


def test_case_report_generates_markdown() -> None:
    """build_case_report should return a non-empty markdown string."""
    for case in CASE_INPUTS:
        report = build_case_report(case)
        assert isinstance(report, str)
        assert len(report) > 100
        assert "# " in report  # has a Markdown heading
        assert case["case_id"] in report


def test_case_report_includes_claim_boundary() -> None:
    """Each case report should state its diagnostic claim boundary."""
    for case in CASE_INPUTS:
        report = build_case_report(case)
        assert "diagnostic" in report.lower()


def test_manifest_has_required_fields(tmp_path: Path) -> None:
    """The manifest should have the required schema fields."""
    input_artifacts = []
    for case in CASE_INPUTS:
        slice_path = REPO_ROOT / case["evidence_dir"] / case["slice_file"]
        input_artifacts.append(
            type("IA", (), {"name": f"trace_{case['case_id']}", "path": slice_path})()
        )
    manifest = build_manifest(
        generated_at="2026-06-23T00:00:00Z",
        input_artifacts=input_artifacts,
        case_inputs=CASE_INPUTS,
        output_dir=tmp_path,
    )
    assert manifest["schema_version"] == "research_v1_failure_pack_manifest.v1"
    assert manifest["paper_facing"] is False
    assert manifest["claim_boundary"] == (
        "diagnostic trace-review evidence only; not benchmark or paper evidence"
    )
    assert isinstance(manifest["input_artifacts"], list)
    assert isinstance(manifest["figure_catalog"], list)


def test_readme_includes_case_names() -> None:
    """The README should list each selected case."""
    manifest = {
        "claim_boundary": "diagnostic trace-review evidence only; not benchmark or paper evidence",
    }
    readme = build_readme(
        generated_at="2026-06-23T00:00:00Z",
        case_inputs=CASE_INPUTS,
        manifest=manifest,
    )
    for case in CASE_INPUTS:
        assert case["report_title"] in readme
        assert case["case_id"] in readme


def test_manifest_is_json_serializable(tmp_path: Path) -> None:
    """The manifest should be serializable to JSON without errors."""
    input_artifacts = []
    for case in CASE_INPUTS:
        slice_path = REPO_ROOT / case["evidence_dir"] / case["slice_file"]
        input_artifacts.append(
            type("IA", (), {"name": f"trace_{case['case_id']}", "path": slice_path})()
        )
    manifest = build_manifest(
        generated_at="2026-06-23T00:00:00Z",
        input_artifacts=input_artifacts,
        case_inputs=CASE_INPUTS,
        output_dir=tmp_path,
    )
    json.dumps(manifest)  # should not raise
