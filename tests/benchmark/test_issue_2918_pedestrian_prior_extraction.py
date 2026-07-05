"""Tests fixture/local pedestrian-prior extraction pipeline for issue #2918."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot_sf.benchmark.pedestrian_prior_extraction import (
    PEDESTRIAN_PRIOR_EXTRACTION_REPORT_SCHEMA_VERSION,
    PedestrianPriorExtractionError,
    extract_pedestrian_prior_report,
    extract_pedestrian_prior_report_from_file,
    write_pedestrian_prior_extraction_report,
)
from robot_sf.benchmark.pedestrian_prior_extraction_manifest import (
    PRIOR_EXTRACTION_EVIDENCE_BOUNDARY,
    REQUIRED_PRIOR_PARAMETERS,
)
from scripts.tools.extract_pedestrian_prior import main as extract_cli_main

FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "issue_2918_pedestrian_prior_fixture.yaml"
)


def _summary_by_name(report) -> dict:
    return {summary.name: summary for summary in report.parameter_summaries}


def test_fixture_extraction_emits_all_required_prior_parameters() -> None:
    """A local fixture produces bounded summaries for every issue #2918 parameter."""

    report = extract_pedestrian_prior_report_from_file(FIXTURE_PATH)
    summaries = _summary_by_name(report)

    assert report.schema_version == PEDESTRIAN_PRIOR_EXTRACTION_REPORT_SCHEMA_VERSION
    assert report.source_id == "issue_2918_proxy_fixture"
    assert report.value_status == "proxy-placeholder"
    assert report.evidence_boundary == PRIOR_EXTRACTION_EVIDENCE_BOUNDARY
    assert tuple(summaries) == REQUIRED_PRIOR_PARAMETERS
    assert summaries["walking_speed"].minimum == 0.0
    assert summaries["walking_speed"].maximum == 1.0
    assert summaries["walking_speed"].mean == 0.75
    assert summaries["density"].minimum == 0.02
    assert summaries["density"].maximum == 0.02
    assert summaries["interaction_distance"].minimum == 2.0
    assert summaries["stop_yield_timing"].maximum == 1.0
    assert all(summary.value_status == "proxy-placeholder" for summary in summaries.values())


def test_report_serialization_excludes_raw_observations(tmp_path: Path) -> None:
    """Output report stores compact summaries and provenance, not raw trajectories."""

    report = extract_pedestrian_prior_report_from_file(FIXTURE_PATH)
    output_path = tmp_path / "prior_report.json"

    write_pedestrian_prior_extraction_report(report, output_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert "observations" not in payload
    assert "parameter_summaries" in payload
    assert payload["provenance"]["raw_trajectory_storage"] == "not_stored_in_git"


def test_cli_writes_report_and_prints_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """CLI supports fixture smoke extraction without campaigns or external data downloads."""

    output_path = tmp_path / "report.json"

    exit_code = extract_cli_main(["--input", str(FIXTURE_PATH), "--output", str(output_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert output_path.is_file()
    assert json.loads(captured.out)["source_id"] == "issue_2918_proxy_fixture"


def test_dataset_backed_status_is_only_a_stamp_not_manifest_admission() -> None:
    """Extraction can stamp staged values but does not bypass the manifest gate."""

    report = extract_pedestrian_prior_report_from_file(FIXTURE_PATH, value_status="dataset-backed")

    assert report.value_status == "dataset-backed"
    assert report.evidence_boundary == PRIOR_EXTRACTION_EVIDENCE_BOUNDARY
    assert report.provenance["claim_boundary"] == PRIOR_EXTRACTION_EVIDENCE_BOUNDARY


def test_missing_density_bounds_fail_closed() -> None:
    """Density is a required issue #2918 parameter, so missing bounds fail closed."""

    with pytest.raises(PedestrianPriorExtractionError, match="bounds are required"):
        extract_pedestrian_prior_report(
            {
                "source_id": "missing_bounds",
                "observations": [
                    {"pedestrian_id": "p1", "time": 0.0, "x": 0.0, "y": 0.0},
                    {"pedestrian_id": "p1", "time": 1.0, "x": 1.0, "y": 0.0},
                ],
            }
        )


def test_single_pedestrian_fixture_fails_interaction_distance_closed() -> None:
    """Interaction-distance extraction requires at least one two-pedestrian frame."""

    with pytest.raises(PedestrianPriorExtractionError, match="two pedestrians"):
        extract_pedestrian_prior_report(
            {
                "source_id": "single_pedestrian",
                "bounds": {"min_x": 0, "max_x": 10, "min_y": 0, "max_y": 10},
                "observations": [
                    {"pedestrian_id": "p1", "time": 0.0, "x": 0.0, "y": 0.0},
                    {"pedestrian_id": "p1", "time": 1.0, "x": 1.0, "y": 0.0},
                ],
            }
        )
