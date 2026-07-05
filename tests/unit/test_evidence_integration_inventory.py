"""Focused tests for the evidence-stream integration contract inventory.

All inputs are synthetic metadata; no real data is ingested and no calibration/safety claim is
made. Tests cover the inventory contract, presence-only checker, integration report, and CLI.
"""

from __future__ import annotations

import json

import pytest

from robot_sf.research.evidence_integration_inventory import (
    BASE_PROVENANCE_FIELDS,
    BASE_UNCERTAINTY_FIELDS,
    EvidenceCategory,
    FeasibilityStatus,
    build_integration_report,
    check_stream_metadata,
    get_stream,
    list_streams,
)
from scripts.tools.check_evidence_integration_inventory import main as cli_main


def _complete_metadata(stream_id: str) -> dict[str, object]:
    """Build a synthetic metadata record that satisfies the stream contract."""
    spec = get_stream(stream_id)
    provenance = {field: f"synthetic_{field}" for field in spec.required_provenance_fields}
    uncertainty = {field: f"synthetic_{field}" for field in spec.required_uncertainty_fields}
    uncertainty["calibration_status"] = "non_calibrated"
    return {"provenance": provenance, "uncertainty": uncertainty}


def test_inventory_is_non_empty_and_unique() -> None:
    """The inventory exposes issue #3293 stream ids exactly once."""
    streams = list_streams()
    assert streams, "inventory must not be empty"
    ids = [spec.stream_id for spec in streams]
    assert len(ids) == len(set(ids)), "stream ids must be unique"
    assert {
        "simulation_trace",
        "amv_command_response",
        "external_pedestrian_trajectory",
        "pilot_fleet_operational",
    } <= set(ids)


def test_categories_cover_calibration_benchmark_operational() -> None:
    """The inventory separates calibration, benchmark, and operational evidence."""
    categories = {spec.category for spec in list_streams()}
    assert EvidenceCategory.CALIBRATION in categories
    assert EvidenceCategory.BENCHMARK in categories
    assert EvidenceCategory.OPERATIONAL in categories


def test_external_streams_are_marked_blocked_or_partial() -> None:
    """Real/external data streams cannot silently become feasible in-repo evidence."""
    amv = get_stream("amv_command_response")
    assert amv.feasibility is FeasibilityStatus.BLOCKED_EXTERNAL
    assert amv.blocked_until

    external_traj = get_stream("external_pedestrian_trajectory")
    assert external_traj.feasibility is FeasibilityStatus.PARTIAL_EXTERNAL
    assert external_traj.blocked_until


def test_each_stream_requires_base_fields_and_calibration_status() -> None:
    """Every stream requires mandatory base provenance/uncertainty keys."""
    for spec in list_streams():
        assert set(BASE_PROVENANCE_FIELDS) <= set(spec.required_provenance_fields)
        assert set(BASE_UNCERTAINTY_FIELDS) <= set(spec.required_uncertainty_fields)
        assert "calibration_status" in spec.required_uncertainty_fields


def test_check_passes_on_complete_metadata() -> None:
    """A synthetic record with all keys passes presence check."""
    result = check_stream_metadata("simulation_trace", _complete_metadata("simulation_trace"))
    assert result.ok
    assert result.exit_code == 0
    assert not result.missing_provenance_fields
    assert not result.missing_uncertainty_fields


def test_check_reports_missing_fields() -> None:
    """Missing provenance and uncertainty keys are reported."""
    metadata = _complete_metadata("simulation_trace")
    provenance = metadata["provenance"]
    uncertainty = metadata["uncertainty"]
    assert isinstance(provenance, dict)
    assert isinstance(uncertainty, dict)
    provenance.pop("denominator")
    uncertainty.pop("sample_size")

    result = check_stream_metadata("simulation_trace", metadata)
    assert not result.ok
    assert result.exit_code == 1
    assert "denominator" in result.missing_provenance_fields
    assert "sample_size" in result.missing_uncertainty_fields


def test_check_accepts_top_level_fields() -> None:
    """Fields may live at top level instead of nested provenance/uncertainty blocks."""
    spec = get_stream("simulation_trace")
    flat: dict[str, object] = dict.fromkeys(spec.required_provenance_fields, "x")
    flat.update(dict.fromkeys(spec.required_uncertainty_fields, "x"))
    result = check_stream_metadata("simulation_trace", flat)
    assert result.ok


def test_check_handles_non_dict_metadata() -> None:
    """A non-dict record is treated as all-fields-missing, not a crash."""
    result = check_stream_metadata("simulation_trace", None)  # type: ignore[arg-type]
    assert not result.ok
    assert set(result.missing_provenance_fields) == set(
        get_stream("simulation_trace").required_provenance_fields
    )


def test_integration_report_preserves_blockers_and_denominator_rule() -> None:
    """The consolidation report exposes blockers and rejects denominator pooling."""
    report = build_integration_report()
    assert report["status"] == "design_stage_external_data_blocked"
    assert "not benchmark evidence" in report["claim_boundary"]
    assert report["categories"]["calibration"] == [
        "amv_command_response",
        "external_pedestrian_trajectory",
    ]
    blocked_ids = {blocker["stream_id"] for blocker in report["blockers_remaining"]}
    assert "amv_command_response" in blocked_ids
    assert "pilot_fleet_operational" in blocked_ids
    rules = {item["rule"] for item in report["invalid_combinations"]}
    assert "do_not_pool_denominators" in rules
    assert "amv_command_response_required_for_calibration" in rules
    assert report["next_empirical_action"]["status"] == "blocked_schema_or_data_prereq"
    assert "non-calibrated" in report["next_empirical_action"]["allowed_claim"]


def test_get_stream_unknown_raises() -> None:
    """Looking up an unknown stream id raises a clear KeyError."""
    with pytest.raises(KeyError):
        get_stream("does_not_exist")


def test_cli_list(capsys: pytest.CaptureFixture[str]) -> None:
    """--list prints inventory JSON and exits 0."""
    rc = cli_main(["--list"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    ids = {stream["stream_id"] for stream in payload["streams"]}
    assert "amv_command_response" in ids


def test_cli_report(capsys: pytest.CaptureFixture[str]) -> None:
    """--report prints integration-report JSON and exits 0."""
    rc = cli_main(["--report"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["issue"] == 3293
    assert payload["status"] == "design_stage_external_data_blocked"
    assert payload["next_empirical_action"]["status"] == "blocked_schema_or_data_prereq"


def test_cli_check_pass(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    """--check on complete synthetic record exits 0."""
    meta_path = tmp_path / "meta.json"
    meta_path.write_text(json.dumps(_complete_metadata("simulation_trace")), encoding="utf-8")
    rc = cli_main(["--check", "simulation_trace", "--metadata", str(meta_path)])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is True


def test_cli_check_fail(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    """--check on an incomplete record exits 1 and lists missing fields."""
    meta_path = tmp_path / "meta.json"
    meta_path.write_text(json.dumps({"provenance": {}, "uncertainty": {}}), encoding="utf-8")
    rc = cli_main(["--check", "simulation_trace", "--metadata", str(meta_path)])
    assert rc == 1
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is False
    assert out["missing_provenance_fields"]


def test_cli_check_requires_metadata(capsys: pytest.CaptureFixture[str]) -> None:
    """--check without --metadata is a usage error."""
    rc = cli_main(["--check", "simulation_trace"])
    assert rc == 2


def test_cli_check_unknown_stream(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    """--check on an unknown stream id is an input error."""
    meta_path = tmp_path / "meta.json"
    meta_path.write_text(json.dumps({}), encoding="utf-8")
    rc = cli_main(["--check", "nope", "--metadata", str(meta_path)])
    assert rc == 2
