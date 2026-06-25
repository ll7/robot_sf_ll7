"""Tests issue #3484 feasibility diagnostic dry-run manifests."""

from __future__ import annotations

from robot_sf.scenario_certification.failure_cause import (
    DIAGNOSTIC_STATUS_NEEDS_EVIDENCE,
    DIAGNOSTIC_STATUS_NOT_RUN,
    FEASIBILITY_DIAGNOSTIC_MANIFEST_SCHEMA,
    INDETERMINATE,
    REQUIRED_DIAGNOSTIC_LANES,
    FeasibilityDiagnosticRow,
    build_feasibility_diagnostic_manifest,
    feasibility_diagnostic_row_to_dict,
)


def test_feasibility_diagnostic_row_is_not_run_until_evidence_exists() -> None:
    """Dry-run rows must not imply feasibility or benchmark evidence."""

    row = feasibility_diagnostic_row_to_dict(
        FeasibilityDiagnosticRow(
            scenario_id="classic_bottleneck_low",
            family_id="bottleneck",
            source="configs/scenarios/classic_interactions.yaml",
        )
    )

    assert row["diagnostic_status"] == DIAGNOSTIC_STATUS_NOT_RUN
    assert row["evidence_status"] == DIAGNOSTIC_STATUS_NEEDS_EVIDENCE
    assert row["required_diagnostics"] == list(REQUIRED_DIAGNOSTIC_LANES)
    assert row["failure_cause_verdict"]["cause"] == INDETERMINATE
    assert row["failure_cause_verdict"]["evidence_complete"] is False
    assert row["failure_cause_verdict"]["comparable_for_ranking"] is False


def test_build_feasibility_manifest_filters_candidate_family() -> None:
    """Manifest builder selects candidate families without classifying outcomes."""

    scenarios = [
        {"name": "classic_bottleneck_low", "metadata": {"archetype": "bottleneck"}},
        {"name": "classic_cross_trap_high", "metadata": {"archetype": "cross_trap"}},
    ]

    manifest = build_feasibility_diagnostic_manifest(
        scenarios,
        source="configs/scenarios/classic_interactions.yaml",
        family_ids=["bottleneck"],
    )

    assert manifest["schema_version"] == FEASIBILITY_DIAGNOSTIC_MANIFEST_SCHEMA
    assert manifest["claim_boundary"] == "diagnostic_only_not_benchmark_evidence"
    assert manifest["row_count"] == 1
    assert manifest["skipped_count"] == 1
    assert manifest["rows"][0]["scenario_id"] == "classic_bottleneck_low"
    assert manifest["rows"][0]["family_id"] == "bottleneck"
    assert manifest["rows"][0]["failure_cause_verdict"]["cause"] == INDETERMINATE
