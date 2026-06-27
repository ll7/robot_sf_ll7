"""Focused tests for the evidence-stream integration contract inventory (issue #3293).

All inputs are synthetic metadata; no real data is ingested and no calibration/safety claim
is made. Tests cover the inventory contract, the presence-only checker, and the CLI.
"""

from __future__ import annotations

import json

import pytest

from robot_sf.research.evidence_integration_inventory import (
    BASE_PROVENANCE_FIELDS,
    BASE_UNCERTAINTY_FIELDS,
    EvidenceCategory,
    FeasibilityStatus,
    check_stream_metadata,
    get_stream,
    list_streams,
)
from scripts.tools.check_evidence_integration_inventory import main as cli_main


def _complete_metadata(stream_id: str) -> dict[str, object]:
    """Build a synthetic metadata record that satisfies a stream's full contract."""
    spec = get_stream(stream_id)
    provenance = {field: f"synthetic_{field}" for field in spec.required_provenance_fields}
    uncertainty = {field: f"synthetic_{field}" for field in spec.required_uncertainty_fields}
    uncertainty["calibration_status"] = "non_calibrated"
    return {"provenance": provenance, "uncertainty": uncertainty}


def test_inventory_is_non_empty_and_unique() -> None:
    """The inventory exposes the issue's streams with unique ids."""
    streams = list_streams()
    assert streams, "inventory must not be empty"
    ids = [spec.stream_id for spec in streams]
    assert len(ids) == len(set(ids)), "stream ids must be unique"
    # The four evidence categories named in issue #3293 must all be representable.
    assert {"simulation_trace", "amv_command_response", "external_pedestrian_trajectory"} <= set(
        ids
    )


def test_categories_cover_calibration_benchmark_operational() -> None:
    """The inventory separates calibration, benchmark, and operational evidence."""
    categories = {spec.category for spec in list_streams()}
    assert EvidenceCategory.CALIBRATION in categories
    assert EvidenceCategory.BENCHMARK in categories
    assert EvidenceCategory.OPERATIONAL in categories


def test_blocked_streams_declare_unblock_condition() -> None:
    """Every externally-blocked stream names its blocked_until unblock condition."""
    for spec in list_streams():
        if spec.feasibility is FeasibilityStatus.BLOCKED_EXTERNAL:
            assert spec.blocked_until, f"{spec.stream_id} must declare blocked_until"


def test_amv_stream_is_blocked_and_calibration() -> None:
    """The AMV command-response stream reflects the maintainer hard-block decision."""
    spec = get_stream("amv_command_response")
    assert spec.feasibility is FeasibilityStatus.BLOCKED_EXTERNAL
    assert spec.category is EvidenceCategory.CALIBRATION
    assert spec.blocked_until


def test_required_fields_include_base_fields() -> None:
    """Each stream's required fields include the mandatory base provenance/uncertainty keys."""
    for spec in list_streams():
        assert set(BASE_PROVENANCE_FIELDS) <= set(spec.required_provenance_fields)
        assert set(BASE_UNCERTAINTY_FIELDS) <= set(spec.required_uncertainty_fields)
        # calibration_status is mandatory so synthetic envelopes cannot pass as calibrated.
        assert "calibration_status" in spec.required_uncertainty_fields


def test_check_passes_on_complete_metadata() -> None:
    """A synthetic record with all required keys passes the presence check."""
    result = check_stream_metadata("simulation_trace", _complete_metadata("simulation_trace"))
    assert result.ok
    assert result.exit_code == 0
    assert not result.missing_provenance_fields
    assert not result.missing_uncertainty_fields


def test_check_reports_missing_fields() -> None:
    """Missing provenance and uncertainty keys are reported, not silently accepted."""
    metadata = _complete_metadata("simulation_trace")
    metadata["provenance"].pop("denominator")
    metadata["uncertainty"].pop("sample_size")
    result = check_stream_metadata("simulation_trace", metadata)
    assert not result.ok
    assert result.exit_code == 1
    assert "denominator" in result.missing_provenance_fields
    assert "sample_size" in result.missing_uncertainty_fields


def test_check_accepts_top_level_fields() -> None:
    """Fields may live at the top level instead of nested provenance/uncertainty blocks."""
    spec = get_stream("simulation_trace")
    flat: dict[str, object] = dict.fromkeys(spec.required_provenance_fields, "x")
    flat.update(dict.fromkeys(spec.required_uncertainty_fields, "x"))
    result = check_stream_metadata("simulation_trace", flat)
    assert result.ok


def test_get_stream_unknown_raises() -> None:
    """Looking up an unknown stream id raises a clear KeyError."""
    with pytest.raises(KeyError):
        get_stream("does_not_exist")


def test_cli_list(capsys: pytest.CaptureFixture[str]) -> None:
    """--list prints the inventory as JSON and exits 0."""
    rc = cli_main(["--list"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    ids = {s["stream_id"] for s in payload["streams"]}
    assert "amv_command_response" in ids


def test_cli_check_pass(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    """--check on a complete synthetic record exits 0."""
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
    """--check without --metadata is a usage error (exit 2)."""
    rc = cli_main(["--check", "simulation_trace"])
    assert rc == 2


def test_cli_check_unknown_stream(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    """--check on an unknown stream id is an input error (exit 2)."""
    meta_path = tmp_path / "meta.json"
    meta_path.write_text(json.dumps({}), encoding="utf-8")
    rc = cli_main(["--check", "nope", "--metadata", str(meta_path)])
    assert rc == 2
