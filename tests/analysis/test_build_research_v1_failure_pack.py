"""Tests for the issue #2159 research-v1 failure-case trace review pack builder."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.analysis.build_research_v1_failure_pack_issue_2159 import (
    CASE_INPUTS,
    InputArtifact,
    build_case_report,
    build_manifest,
    build_readme,
    main,
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
        input_artifacts.append(InputArtifact(name=f"trace_{case['case_id']}", path=slice_path))
    manifest = build_manifest(
        generated_at="2026-06-23T00:00:00Z",
        input_artifacts=input_artifacts,
        case_inputs=CASE_INPUTS,
        output_dir=tmp_path,
        repo_root=REPO_ROOT,
    )
    assert manifest["schema_version"] == "research_v1_failure_pack_manifest.v1"
    assert manifest["paper_facing"] is False
    assert manifest["claim_boundary"] == (
        "diagnostic trace-review evidence only; not benchmark or paper evidence"
    )
    assert isinstance(manifest["input_artifacts"], list)
    assert isinstance(manifest["figure_catalog"], list)
    assert all(not artifact["path"].startswith("/") for artifact in manifest["input_artifacts"])


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
        input_artifacts.append(InputArtifact(name=f"trace_{case['case_id']}", path=slice_path))
    manifest = build_manifest(
        generated_at="2026-06-23T00:00:00Z",
        input_artifacts=input_artifacts,
        case_inputs=CASE_INPUTS,
        output_dir=tmp_path,
        repo_root=REPO_ROOT,
    )
    json.dumps(manifest)  # should not raise


def test_case_report_tolerates_null_optional_blocks(tmp_path: Path, monkeypatch) -> None:
    """Explicit JSON nulls in optional trace blocks should not crash report generation."""
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    (evidence_dir / "slice.json").write_text(
        json.dumps(
            {
                "planner_runs": None,
                "trace_pairs": [
                    {
                        "seed": 1,
                        "planner_key": "goal",
                        "no_op": None,
                        "perturbed": {"frame_range": None, "events": [None]},
                    }
                ],
                "pair_summary": {"clearance_delta_rows": None},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    report = build_case_report(
        {
            "case_id": "null_blocks",
            "claim_id": "test.claim",
            "evidence_dir": "evidence",
            "slice_file": "slice.json",
            "report_title": "Null Blocks",
            "scenario_id": "test_scenario",
            "planners": ["goal"],
            "seeds": [1],
        }
    )

    assert "Planner goal, seed 1" in report
    assert "No clearance deltas recorded" in report


def test_main_excludes_checksums_file_from_checksum_manifest(tmp_path: Path) -> None:
    """Rerunning the builder should not checksum the previous checksums file."""
    output_dir = tmp_path / "pack"

    main(["--output-dir", str(output_dir), "--generated-at", "2026-06-23T00:00:00Z"])
    main(["--output-dir", str(output_dir), "--generated-at", "2026-06-23T00:00:00Z"])

    checksums = (output_dir / "checksums.sha256").read_text(encoding="utf-8")
    assert "checksums.sha256" not in checksums
