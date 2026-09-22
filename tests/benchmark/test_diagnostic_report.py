"""Focused tests for the deterministic diagnostic report adapters (issue #7387)."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from robot_sf.benchmark.diagnostic_report import (
    DIAGNOSTIC_SOURCE,
    SCHEMA_VERSION,
    TWO_PLANNER_FIELD_INVENTORY,
    DiagnosticReportError,
    adapt_nominal_social_force_diagnostics,
    adapt_orca_residual_row,
    canonical_digest,
    diagnose_diagnostic_rows,
)

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "diagnostic_report"


def _load_fixture(family: str, name: str) -> list[dict]:
    path = FIXTURE_ROOT / family / name
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _observation_ids(report: dict) -> list[str]:
    return [entry["observation_id"] for entry in report["observations"]]


def _symptom_ids(report: dict) -> list[str]:
    return [entry["symptom_id"] for entry in report["symptoms"]]


def test_two_planner_inventory_names_owners_and_unsupported_fields() -> None:
    """Inventory must name actual owners and distinguish unsupported fields."""
    orca = TWO_PLANNER_FIELD_INVENTORY["orca_residual"]
    assert "mechanism_trace" in orca["trace_owner"]
    assert "emit_orca_residual" in orca["emitter_owner"]
    assert orca["fields"]["candidate_table"]["support"] == "supported"
    nominal = TWO_PLANNER_FIELD_INVENTORY["nominal_social_force"]
    assert "fast_pysf_wrapper" in nominal["planner_owner"]
    unsupported = {
        name for name, spec in nominal["fields"].items() if spec["support"] == "unsupported"
    }
    assert {"candidate_table", "cost_terms_and_scale", "saturation_flag"} <= unsupported
    assert nominal["fields"]["collision_event"]["support"] == "post_hoc"


def test_stale_observation_present_reports_measured_mismatch() -> None:
    """Stale fixture must report the measured version mismatch as observation."""
    report = diagnose_diagnostic_rows(_load_fixture("stale_observation", "present.json"))
    assert "stale_observation_version" in _observation_ids(report)
    assert "stale_observation_consumed" in _symptom_ids(report)
    assert report["hypotheses"] == []
    finding = report["observations"][0]
    assert finding["rule_version"] == "obs_age_mismatch.v1"
    assert finding["source_digest"] == "digest-stale-present"
    assert finding["source_row"] == "stale_observation/present.json#0"
    assert finding["sim_step"] == 7


def test_cost_scale_present_reports_demonstrable_order_change() -> None:
    """Cost fixture must report only when scales demonstrably reverse order."""
    report = diagnose_diagnostic_rows(_load_fixture("cost_scaling", "present.json"))
    assert "cost_scale_order_change" in _observation_ids(report)
    assert "cost_scale_inconsistency" in _symptom_ids(report)
    assert report["hypotheses"] == []


def test_saturation_present_localizes_gap_with_recorded_flag() -> None:
    """Saturation fixture must localize the command gap with the recorded flag."""
    report = diagnose_diagnostic_rows(_load_fixture("controller_saturation", "present.json"))
    assert "command_execution_gap" in _observation_ids(report)
    assert "saturated_command_execution_gap" in _symptom_ids(report)
    assert report["hypotheses"] == []


def test_benign_controls_receive_no_mechanism_or_causal_labels() -> None:
    """Benign controls must not receive mechanism or causal labels."""
    for family in ("stale_observation", "cost_scaling", "controller_saturation"):
        report = diagnose_diagnostic_rows(_load_fixture(family, "benign.json"))
        assert report["observations"] == []
        assert report["symptoms"] == []
        assert report["hypotheses"] == []


def test_ambiguous_variants_reject_unsupported_narratives() -> None:
    """Ambiguous fixtures must abstain or qualify the unsupported explanation."""
    stale = diagnose_diagnostic_rows(_load_fixture("stale_observation", "ambiguous.json"))
    assert "stale_observation_version" in _observation_ids(stale)
    assert stale["hypotheses"] == []
    assert any(
        entry["reason"] == "unsupported_narrative_not_admitted" for entry in stale["abstentions"]
    )
    cost = diagnose_diagnostic_rows(_load_fixture("cost_scaling", "ambiguous.json"))
    assert "cost_scale_order_change" in _observation_ids(cost)
    assert cost["hypotheses"] == []
    saturated = diagnose_diagnostic_rows(_load_fixture("controller_saturation", "ambiguous.json"))
    assert saturated["observations"] == []
    assert saturated["hypotheses"] == []
    assert any(
        "saturation" in entry["reason"] or "unsupported" in entry["reason"]
        for entry in saturated["abstentions"]
    )


def test_missing_field_variants_abstain_without_invalidating_supported() -> None:
    """Missing fields must abstain while unrelated supported findings survive."""
    report = diagnose_diagnostic_rows(_load_fixture("stale_observation", "missing.json"))
    assert report["observations"] == []
    assert any(
        entry["reason"] == "cannot_assess_staleness_without_versions"
        for entry in report["abstentions"]
    )
    combined = _load_fixture("stale_observation", "present.json") + _load_fixture(
        "cost_scaling", "missing.json"
    )
    mixed = diagnose_diagnostic_rows(combined)
    assert "stale_observation_version" in _observation_ids(mixed)
    assert any(entry["missing_field"] == "candidates" for entry in mixed["missing_evidence"])


def test_collision_alone_never_triggers_mechanism() -> None:
    """A later collision alone must not trigger an invented mechanism."""
    rows = _load_fixture("stale_observation", "benign.json")
    rows[0]["collision_event"] = True
    report = diagnose_diagnostic_rows(rows)
    assert report["observations"] == []
    assert report["symptoms"] == []
    assert report["hypotheses"] == []


def test_truth_manifest_is_isolated_from_diagnostic_function() -> None:
    """Hidden truth must not leak into an allegedly online diagnosis."""
    import robot_sf.benchmark.diagnostic_report as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "truth_manifest" not in source
    assert "collision_event" in source  # outcome is referenced only as non-input
    rows = _load_fixture("cost_scaling", "present.json")
    report = diagnose_diagnostic_rows(rows)
    assert "mechanism_present" not in json.dumps(report)


def test_corrupt_provenance_malformed_units_nonfinite_fail_deterministically() -> None:
    """Corrupt inputs must fail with deterministic reasons, twice the same."""
    base = _load_fixture("stale_observation", "present.json")
    corrupt: list[dict] = copy.deepcopy(base)
    corrupt[0]["provenance"] = {"artifact_digest": "", "source_row": ""}
    with pytest.raises(DiagnosticReportError, match="corrupt provenance"):
        diagnose_diagnostic_rows(corrupt)
    malformed: list[dict] = copy.deepcopy(base)
    malformed[0]["observation_version"] = "v5"
    with pytest.raises(DiagnosticReportError, match="observation_version"):
        diagnose_diagnostic_rows(malformed)
    nonfinite: list[dict] = copy.deepcopy(base)
    nonfinite[0]["sim_time_s"] = float("nan")
    with pytest.raises(DiagnosticReportError, match="sim_time_s"):
        diagnose_diagnostic_rows(nonfinite)
    duplicate = base + copy.deepcopy(base)
    with pytest.raises(DiagnosticReportError, match="duplicate row_id"):
        diagnose_diagnostic_rows(duplicate)
    first = None
    second = None
    try:
        diagnose_diagnostic_rows(corrupt)
    except DiagnosticReportError as err:
        first = str(err)
    try:
        diagnose_diagnostic_rows(corrupt)
    except DiagnosticReportError as err:
        second = str(err)
    assert first == second


def test_determinism_reorder_invariance_and_temporal_preservation() -> None:
    """Same report twice; row-reorder invariance; temporal order preserved."""
    rows = _load_fixture("stale_observation", "present.json") + _load_fixture(
        "controller_saturation", "present.json"
    )
    first = diagnose_diagnostic_rows(rows)
    second = diagnose_diagnostic_rows(copy.deepcopy(rows))
    assert first == second
    assert first["report_digest"] == second["report_digest"]
    reversed_report = diagnose_diagnostic_rows(list(reversed(rows)))
    assert reversed_report["input_digest"] == first["input_digest"]
    assert reversed_report["observations"] == first["observations"]
    steps = [entry["sim_step"] for entry in first["observations"]]
    assert steps == sorted(steps)


def test_source_inputs_unchanged_and_text_is_data() -> None:
    """Inputs must be unchanged; text fields must be data, never execution."""
    rows = _load_fixture("cost_scaling", "ambiguous.json")
    rows[0]["unsupported_narrative"] = "__import__('os').system('echo pwned')"
    rows[0]["provenance"]["source_row"] = "x'; DROP TABLE findings; --"
    snapshot = copy.deepcopy(rows)
    report = diagnose_diagnostic_rows(rows)
    assert report["hypotheses"] == []
    assert rows == snapshot
    assert "os" not in json.dumps(report["hypotheses"])
    assert any(
        entry["reason"] == "unsupported_narrative_not_admitted" for entry in report["abstentions"]
    )


def test_report_binds_digest_row_time_rule_and_evidence_status() -> None:
    """Findings must bind digest, row, time, rule version, evidence status."""
    report = diagnose_diagnostic_rows(_load_fixture("controller_saturation", "present.json"))
    assert report["schema_version"] == SCHEMA_VERSION
    assert report["diagnostic_source"] == DIAGNOSTIC_SOURCE
    for finding in report["observations"] + report["symptoms"]:
        assert finding["source_digest"] == "digest-sat-present"
        assert finding["source_row"].endswith("#0")
        assert finding["rule_version"]
        assert finding["confidence"] in (
            "observed_mechanism",
            "supported_hypothesis",
            "weak_hypothesis",
            "unknown",
        )
        assert finding["evidence_mode"]


def test_existing_outputs_remain_compatible() -> None:
    """Mechanism-trace validation must still accept the existing example."""
    from robot_sf.benchmark.mechanism_trace import validate_mechanism_trace_payload

    example = Path(__file__).parent / "fixtures" / "mechanism_trace.v1.example.json"
    with example.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    assert validate_mechanism_trace_payload(payload)["schema_version"] == "mechanism_trace.v1"


def test_adapters_preserve_owner_meanings_and_leave_unknowns_missing() -> None:
    """Adapters must preserve schema meanings and never invent planner data."""
    mechanism_row = {
        "mechanism_id": "orca_residuals",
        "activation_step": 9,
        "selected_command": [0.4, -0.1],
        "command_source": "prior_residual_safe",
    }
    adapted = adapt_orca_residual_row(
        mechanism_row, source_digest="abc", source_row="trace#9", timing={"sim_time_s": 0.9}
    )
    assert adapted["commanded_control"] == [0.4, -0.1]
    assert adapted["candidates"] is None
    assert adapted["provenance"] == {"artifact_digest": "abc", "source_row": "trace#9"}
    sf_row = adapt_nominal_social_force_diagnostics(
        {"planner_type": "FastPysfWrapper", "fallback": False},
        source_digest="sf",
        source_row="sf#1",
        sim_step=1,
        sim_time_s=0.1,
        commanded_control=[0.2, 0.0],
        executed_control=[0.2, 0.0],
    )
    assert sf_row["planner_id"] == "nominal_social_force"
    assert sf_row["candidates"] is None
    assert sf_row["cost_scale"] is None
    assert sf_row["observation_version"] is None
    assert canonical_digest({"b": 1, "a": 2}) == canonical_digest({"a": 2, "b": 1})
