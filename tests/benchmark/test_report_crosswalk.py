"""Tests for the report crosswalk (issue #6871).

Deterministic tests for the versioned crosswalk that maps failure-diagnosis and
execution-deviation fields into episode/campaign summaries.
"""

from __future__ import annotations

from typing import Any

import pytest

from robot_sf.analysis_workbench.trace_failure_predicates import TraceFailurePredicate
from robot_sf.benchmark.failure_diagnosis import (
    _NON_CAUSAL_CAVEAT,
    _OUT_OF_SCOPE_CAVEAT,
    _PAYLOAD_NON_CLAIM_CAVEAT,
    DIAGNOSIS_SOURCE,
    FAILURE_DIAGNOSIS_SCHEMA_VERSION,
    build_failure_diagnosis_payload,
    diagnose_from_trace_failure_predicate,
    validate_failure_diagnosis_payload,
)
from robot_sf.benchmark.report_crosswalk import (
    FIELD_PROVENANCE_STATES,
    FIELD_VALIDITY_STATES,
    REPORT_CROSSWALK_SCHEMA_VERSION,
    build_campaign_diagnostic_summary,
    build_crosswalk_example_fixture,
    build_episode_diagnostic_summary,
    export_crosswalk_example_fixture,
    validate_campaign_diagnostic_summary,
    validate_episode_diagnostic_summary,
)
from robot_sf.benchmark.trajectory_verifier import (
    EXECUTION_DEVIATION_CLAIM_BOUNDARY,
    EXECUTION_DEVIATION_SCHEMA,
    INTERVENTION_CONTINUE,
    INTERVENTION_WARN,
    ExecutionDeviationResult,
)


def _predicate(
    predicate_id: str = "collision",
    *,
    time_interval_s: list[float | None] | None = None,
    severity: str = "critical",
    validity_status: str = "valid",
) -> TraceFailurePredicate:
    """Build one synthetic trace failure predicate."""
    return TraceFailurePredicate(
        predicate_id=predicate_id,
        time_interval_s=list(time_interval_s if time_interval_s is not None else [1.0, 1.5]),
        steps=[10, 15],
        involved_actors=["robot", "ped_0"],
        scenario_family="crosswalk",
        planner_id="orca",
        evidence_fields={"min_clearance_m": 0.1},
        severity=severity,
        validity_status=validity_status,
    )


def _collision_diagnosis_payload() -> dict[str, Any]:
    """Build a minimal failure_diagnosis.v1 payload with one collision record."""
    record = diagnose_from_trace_failure_predicate(
        _predicate("collision", time_interval_s=[1.0, 1.5], severity="critical")
    ).to_dict()
    return {
        "schema_version": FAILURE_DIAGNOSIS_SCHEMA_VERSION,
        "diagnosis_source": DIAGNOSIS_SOURCE,
        "generated_at_utc": "2026-01-01T00:00:00+00:00",
        "records": [record],
        "failure_type_coverage": {
            "counts": {"collision": 1},
            "classification_source": DIAGNOSIS_SOURCE,
        },
        "caveats": [_PAYLOAD_NON_CLAIM_CAVEAT, _NON_CAUSAL_CAVEAT, _OUT_OF_SCOPE_CAVEAT],
    }


def _unknown_diagnosis_payload() -> dict[str, Any]:
    """Build a failure_diagnosis.v1 payload with one unknown-type record."""
    record = diagnose_from_trace_failure_predicate(
        _predicate(
            "oscillatory_local_control",
            time_interval_s=[2.0, 3.0],
            severity="medium",
        )
    ).to_dict()
    return {
        "schema_version": FAILURE_DIAGNOSIS_SCHEMA_VERSION,
        "diagnosis_source": DIAGNOSIS_SOURCE,
        "generated_at_utc": "2026-01-01T00:00:00+00:00",
        "records": [record],
        "failure_type_coverage": {
            "counts": {"unknown": 1},
            "classification_source": DIAGNOSIS_SOURCE,
        },
        "caveats": [_PAYLOAD_NON_CLAIM_CAVEAT, _NON_CAUSAL_CAVEAT, _OUT_OF_SCOPE_CAVEAT],
    }


def _status_diagnosis_payload(status: str) -> dict[str, Any]:
    """Build a canonical diagnosis payload for a non-valid source status."""
    record = diagnose_from_trace_failure_predicate(
        _predicate("collision", severity="critical", validity_status=status)
    )
    return build_failure_diagnosis_payload(
        [record],
        generated_at_utc="2026-01-01T00:00:00+00:00",
    )


def _warn_deviation_result() -> ExecutionDeviationResult:
    """Build an ExecutionDeviationResult with a warn intervention."""
    return ExecutionDeviationResult(
        intervention=INTERVENTION_WARN,
        deviation_score=0.6,
        component_deviations=(("robot_position", 0.6),),
        first_threshold_crossing_time_s=0.5,
        input_age_s=0.1,
        fail_closed=False,
        schema_version=EXECUTION_DEVIATION_SCHEMA,
        claim_boundary=EXECUTION_DEVIATION_CLAIM_BOUNDARY,
    )


def _fail_closed_deviation_result() -> ExecutionDeviationResult:
    """Build an ExecutionDeviationResult that came from the fail-closed path."""
    return ExecutionDeviationResult(
        intervention=INTERVENTION_WARN,
        deviation_score=None,
        component_deviations=(),
        first_threshold_crossing_time_s=None,
        input_age_s=None,
        fail_closed=True,
        schema_version=EXECUTION_DEVIATION_SCHEMA,
        claim_boundary=EXECUTION_DEVIATION_CLAIM_BOUNDARY,
    )


def _continue_deviation_result() -> ExecutionDeviationResult:
    """Build an ExecutionDeviationResult with a continue intervention."""
    return ExecutionDeviationResult(
        intervention=INTERVENTION_CONTINUE,
        deviation_score=0.1,
        component_deviations=(("robot_position", 0.1),),
        first_threshold_crossing_time_s=None,
        input_age_s=0.05,
        fail_closed=False,
        schema_version=EXECUTION_DEVIATION_SCHEMA,
        claim_boundary=EXECUTION_DEVIATION_CLAIM_BOUNDARY,
    )


# ---------------------------------------------------------------------------
# Episode-level mapping tests
# ---------------------------------------------------------------------------


class TestEpisodeDiagnosticSummaryMapping:
    """Tests for mapping diagnosis and execution-deviation fields to episode summaries."""

    def test_collision_diagnosis_maps_correctly(self) -> None:
        """A collision diagnosis record is mapped with correct type/severity counts."""
        payload = _collision_diagnosis_payload()
        summary = build_episode_diagnostic_summary(
            episode_id="ep_001",
            planner_id="orca",
            diagnosis_payload=payload,
            success=False,
            collision=True,
            comfort=None,
        )

        assert summary.episode_id == "ep_001"
        assert summary.planner_id == "orca"
        assert summary.diagnosis_available is True
        assert summary.diagnosis_record_count == 1
        assert summary.diagnosis_failure_type_counts == {"collision": 1}
        assert summary.diagnosis_severity_counts == {"critical": 1}
        assert summary.diagnosis_unknown_count == 0
        assert summary.diagnosis_validity_state == "available"
        assert summary.diagnosis_provenance == "complete"
        assert summary.diagnosis_validity_reason is None
        assert summary.success is False
        assert summary.collision is True
        assert summary.comfort is None

    def test_unknown_type_diagnosis_maps_correctly(self) -> None:
        """An unknown-type diagnosis record is counted correctly."""
        payload = _unknown_diagnosis_payload()
        summary = build_episode_diagnostic_summary(
            episode_id="ep_002",
            planner_id="orca",
            diagnosis_payload=payload,
            success=True,
            collision=False,
            comfort=0.8,
        )

        assert summary.diagnosis_available is True
        assert summary.diagnosis_record_count == 1
        assert summary.diagnosis_failure_type_counts == {"unknown": 1}
        assert summary.diagnosis_unknown_count == 1
        assert summary.success is True
        assert summary.collision is False
        assert summary.comfort == 0.8

    def test_execution_deviation_warn_maps_correctly(self) -> None:
        """A warn execution-deviation result is mapped with score and crossing time."""
        result = _warn_deviation_result()
        summary = build_episode_diagnostic_summary(
            episode_id="ep_003",
            planner_id="dwa",
            execution_deviation_result=result,
            success=True,
            collision=False,
            comfort=0.9,
        )

        assert summary.execution_deviation_available is True
        assert summary.execution_deviation_intervention == INTERVENTION_WARN
        assert summary.execution_deviation_score == 0.6
        assert summary.execution_deviation_fail_closed is False
        assert summary.execution_deviation_threshold_crossing_time_s == 0.5
        assert summary.execution_deviation_validity_state == "available"
        assert summary.execution_deviation_provenance == "complete"
        assert summary.execution_deviation_validity_reason is None
        assert summary.execution_deviation_claim_boundary == EXECUTION_DEVIATION_CLAIM_BOUNDARY

    def test_fail_closed_deviation_maps_correctly(self) -> None:
        """A fail-closed deviation result has unavailable validity with explicit reason."""
        result = _fail_closed_deviation_result()
        summary = build_episode_diagnostic_summary(
            episode_id="ep_004",
            planner_id="dwa",
            execution_deviation_result=result,
        )

        assert summary.execution_deviation_available is True
        assert summary.execution_deviation_fail_closed is True
        assert summary.execution_deviation_score is None
        assert summary.execution_deviation_validity_state == "unavailable"
        assert summary.execution_deviation_validity_reason is not None
        assert "fail_closed" in summary.execution_deviation_validity_reason

    def test_missing_diagnosis_payload_produces_unavailable(self) -> None:
        """A missing diagnosis payload yields unavailable validity with a reason."""
        summary = build_episode_diagnostic_summary(
            episode_id="ep_no_diag",
            planner_id="orca",
            diagnosis_payload=None,
        )

        assert summary.diagnosis_available is False
        assert summary.diagnosis_record_count == 0
        assert summary.diagnosis_failure_type_counts == {}
        assert summary.diagnosis_severity_counts == {}
        assert summary.diagnosis_unknown_count == 0
        assert summary.diagnosis_validity_state == "unavailable"
        assert summary.diagnosis_provenance == "unknown"
        assert summary.diagnosis_validity_reason == "diagnosis_payload_not_provided"

    def test_missing_deviation_result_produces_unavailable(self) -> None:
        """A missing deviation result yields unavailable validity with a reason."""
        summary = build_episode_diagnostic_summary(
            episode_id="ep_no_dev",
            planner_id="orca",
            execution_deviation_result=None,
        )

        assert summary.execution_deviation_available is False
        assert summary.execution_deviation_intervention is None
        assert summary.execution_deviation_score is None
        assert summary.execution_deviation_fail_closed is None
        assert summary.execution_deviation_validity_state == "unavailable"
        assert summary.execution_deviation_provenance == "unknown"
        assert (
            summary.execution_deviation_validity_reason == "execution_deviation_result_not_provided"
        )

    @pytest.mark.parametrize("status", ["fallback", "degraded"])
    def test_nonstandard_record_status_is_not_promoted_to_available(self, status: str) -> None:
        """Fallback and degraded records retain an explicit non-success state."""
        summary = build_episode_diagnostic_summary(
            episode_id=f"ep_{status}",
            planner_id="orca",
            diagnosis_payload=_status_diagnosis_payload(status),
        )

        assert summary.diagnosis_available is True
        assert summary.diagnosis_validity_state == status
        assert summary.diagnosis_provenance == "complete"
        assert summary.diagnosis_validity_reason == f"diagnosis_record_validity:{status}"

    def test_core_metrics_preserved_separately(self) -> None:
        """Core benchmark metrics are passed through unchanged alongside diagnostics."""
        payload = _collision_diagnosis_payload()
        summary = build_episode_diagnostic_summary(
            episode_id="ep_both",
            planner_id="orca",
            diagnosis_payload=payload,
            execution_deviation_result=_warn_deviation_result(),
            success=False,
            collision=True,
            comfort=0.5,
        )

        assert summary.success is False
        assert summary.collision is True
        assert summary.comfort == 0.5
        assert summary.diagnosis_available is True
        assert summary.execution_deviation_available is True

    def test_caveats_preserve_claim_boundary(self) -> None:
        """Every episode summary carries the non-causal and claim-boundary caveats."""
        summary = build_episode_diagnostic_summary(
            episode_id="ep_caveats",
            planner_id="orca",
        )

        caveats = summary.caveats
        assert any("causality" in c for c in caveats)
        assert any("diagnostic-only" in c for c in caveats)
        assert any("correction" in c for c in caveats)

    def test_to_dict_produces_valid_json_safe_structure(self) -> None:
        """The to_dict output is a flat dictionary with the expected sections."""
        summary = build_episode_diagnostic_summary(
            episode_id="ep_dict",
            planner_id="orca",
            diagnosis_payload=_collision_diagnosis_payload(),
            success=True,
        )
        result = summary.to_dict()

        assert result["schema_version"] == REPORT_CROSSWALK_SCHEMA_VERSION
        assert result["episode_id"] == "ep_dict"
        assert "diagnosis" in result
        assert "execution_deviation" in result
        assert "core_metrics" in result
        assert "caveats" in result
        assert result["core_metrics"]["success"] is True

    def test_validate_episode_summary_rejects_missing_fields(self) -> None:
        """Validation rejects summaries with missing required fields."""
        with pytest.raises(ValueError, match="episode_id"):
            validate_episode_diagnostic_summary({"schema_version": REPORT_CROSSWALK_SCHEMA_VERSION})

    def test_validate_episode_summary_rejects_wrong_schema_version(self) -> None:
        """Validation rejects summaries with the wrong schema version."""
        with pytest.raises(ValueError, match="schema_version"):
            validate_episode_diagnostic_summary(
                {
                    "schema_version": "wrong_version",
                    "episode_id": "ep",
                    "planner_id": "orca",
                    "diagnosis": {},
                    "execution_deviation": {},
                    "core_metrics": {},
                    "caveats": [],
                }
            )


# ---------------------------------------------------------------------------
# Unknown / unavailable handling tests
# ---------------------------------------------------------------------------


class TestUnknownUnavailableHandling:
    """Tests for explicit unavailable, invalid, and unknown handling."""

    def test_empty_records_produce_zero_counts(self) -> None:
        """An empty records list produces zero counts and zero unknown count."""
        payload = {
            "schema_version": FAILURE_DIAGNOSIS_SCHEMA_VERSION,
            "diagnosis_source": DIAGNOSIS_SOURCE,
            "generated_at_utc": "2026-01-01T00:00:00+00:00",
            "records": [],
            "failure_type_coverage": {
                "counts": {},
                "classification_source": DIAGNOSIS_SOURCE,
            },
            "caveats": [_PAYLOAD_NON_CLAIM_CAVEAT, _NON_CAUSAL_CAVEAT, _OUT_OF_SCOPE_CAVEAT],
        }
        summary = build_episode_diagnostic_summary(
            episode_id="ep_empty",
            planner_id="orca",
            diagnosis_payload=payload,
        )

        assert summary.diagnosis_record_count == 0
        assert summary.diagnosis_failure_type_counts == {}
        assert summary.diagnosis_severity_counts == {}
        assert summary.diagnosis_unknown_count == 0

    def test_mixed_known_and_unknown_records(self) -> None:
        """Records with both known and unknown failure types are counted correctly."""
        collision_record = diagnose_from_trace_failure_predicate(
            _predicate("collision", severity="critical")
        ).to_dict()
        unknown_record = diagnose_from_trace_failure_predicate(
            _predicate("oscillatory_local_control", severity="medium")
        ).to_dict()
        payload = {
            "schema_version": FAILURE_DIAGNOSIS_SCHEMA_VERSION,
            "diagnosis_source": DIAGNOSIS_SOURCE,
            "generated_at_utc": "2026-01-01T00:00:00+00:00",
            "records": [collision_record, unknown_record],
            "failure_type_coverage": {
                "counts": {"collision": 1, "unknown": 1},
                "classification_source": DIAGNOSIS_SOURCE,
            },
            "caveats": [_PAYLOAD_NON_CLAIM_CAVEAT, _NON_CAUSAL_CAVEAT, _OUT_OF_SCOPE_CAVEAT],
        }
        summary = build_episode_diagnostic_summary(
            episode_id="ep_mixed",
            planner_id="orca",
            diagnosis_payload=payload,
        )

        assert summary.diagnosis_record_count == 2
        assert summary.diagnosis_failure_type_counts == {"collision": 1, "unknown": 1}
        assert summary.diagnosis_unknown_count == 1

    def test_deviation_continue_result_available(self) -> None:
        """A continue deviation result is available with score 0.1."""
        result = _continue_deviation_result()
        summary = build_episode_diagnostic_summary(
            episode_id="ep_continue",
            planner_id="dwa",
            execution_deviation_result=result,
        )

        assert summary.execution_deviation_available is True
        assert summary.execution_deviation_intervention == INTERVENTION_CONTINUE
        assert summary.execution_deviation_score == 0.1
        assert summary.execution_deviation_validity_state == "available"


# ---------------------------------------------------------------------------
# Execution-deviation reporting case
# ---------------------------------------------------------------------------


class TestExecutionDeviationReporting:
    """Tests for execution-deviation crosswalk and campaign aggregation."""

    def test_campaign_aggregates_intervention_counts(self) -> None:
        """Campaign summary aggregates per-intervention counts across episodes."""
        summaries = [
            build_episode_diagnostic_summary(
                episode_id="ep1",
                planner_id="dwa",
                execution_deviation_result=_warn_deviation_result(),
            ),
            build_episode_diagnostic_summary(
                episode_id="ep2",
                planner_id="dwa",
                execution_deviation_result=_continue_deviation_result(),
            ),
            build_episode_diagnostic_summary(
                episode_id="ep3",
                planner_id="dwa",
                execution_deviation_result=_fail_closed_deviation_result(),
            ),
            build_episode_diagnostic_summary(
                episode_id="ep4",
                planner_id="dwa",
                execution_deviation_result=None,
            ),
        ]

        campaign = build_campaign_diagnostic_summary(
            campaign_id="test_campaign",
            episode_summaries=summaries,
        )

        assert campaign.episode_count == 4
        assert campaign.execution_deviation_available_count == 3
        assert campaign.execution_deviation_fail_closed_count == 1
        assert campaign.execution_deviation_intervention_counts[INTERVENTION_WARN] == 2
        assert campaign.execution_deviation_intervention_counts[INTERVENTION_CONTINUE] == 1
        assert campaign.execution_deviation_coverage_rate == pytest.approx(3.0 / 4.0)

    def test_campaign_aggregates_diagnosis_coverage(self) -> None:
        """Campaign summary counts diagnosis availability across episodes."""
        summaries = [
            build_episode_diagnostic_summary(
                episode_id="ep1",
                planner_id="orca",
                diagnosis_payload=_collision_diagnosis_payload(),
            ),
            build_episode_diagnostic_summary(
                episode_id="ep2",
                planner_id="orca",
                diagnosis_payload=_unknown_diagnosis_payload(),
            ),
            build_episode_diagnostic_summary(
                episode_id="ep3",
                planner_id="orca",
                diagnosis_payload=None,
            ),
        ]

        campaign = build_campaign_diagnostic_summary(
            campaign_id="diag_campaign",
            episode_summaries=summaries,
        )

        assert campaign.diagnosis_available_count == 2
        assert campaign.diagnosis_record_total == 2
        assert campaign.diagnosis_failure_type_totals == {"collision": 1, "unknown": 1}
        assert campaign.diagnosis_unknown_total == 1
        assert campaign.diagnosis_coverage_rate == pytest.approx(2.0 / 3.0)

    def test_campaign_aggregates_core_metrics_separately(self) -> None:
        """Success, collision, and comfort are aggregated independently of diagnostics."""
        summaries = [
            build_episode_diagnostic_summary(
                episode_id="ep1",
                planner_id="orca",
                success=True,
                collision=False,
                comfort=0.9,
            ),
            build_episode_diagnostic_summary(
                episode_id="ep2",
                planner_id="orca",
                success=False,
                collision=True,
                comfort=0.5,
            ),
            build_episode_diagnostic_summary(
                episode_id="ep3",
                planner_id="orca",
                success=True,
                collision=False,
                comfort=None,
            ),
        ]

        campaign = build_campaign_diagnostic_summary(
            campaign_id="metrics_campaign",
            episode_summaries=summaries,
        )

        assert campaign.success_rate == pytest.approx(2.0 / 3.0)
        assert campaign.collision_rate == pytest.approx(1.0 / 3.0)
        assert campaign.comfort_mean == pytest.approx(0.7)

    def test_empty_campaign_produces_none_rates(self) -> None:
        """An empty campaign produces None rates."""
        campaign = build_campaign_diagnostic_summary(
            campaign_id="empty",
            episode_summaries=[],
        )

        assert campaign.episode_count == 0
        assert campaign.diagnosis_coverage_rate is None
        assert campaign.execution_deviation_coverage_rate is None
        assert campaign.success_rate is None
        assert campaign.collision_rate is None
        assert campaign.comfort_mean is None

    def test_campaign_to_dict_structure(self) -> None:
        """Campaign summary to_dict has the expected nested structure."""
        summaries = [
            build_episode_diagnostic_summary(
                episode_id="ep1",
                planner_id="orca",
                diagnosis_payload=_collision_diagnosis_payload(),
                execution_deviation_result=_warn_deviation_result(),
                success=True,
            ),
        ]
        campaign = build_campaign_diagnostic_summary(
            campaign_id="struct_campaign",
            episode_summaries=summaries,
        )
        result = campaign.to_dict()

        assert result["schema_version"] == REPORT_CROSSWALK_SCHEMA_VERSION
        assert result["campaign_id"] == "struct_campaign"
        assert "diagnosis" in result
        assert "execution_deviation" in result
        assert "core_metrics" in result
        assert result["diagnosis"]["available_count"] == 1
        assert result["execution_deviation"]["available_count"] == 1
        assert result["core_metrics"]["success_rate"] == 1.0

    def test_campaign_caveats_preserve_claim_boundary(self) -> None:
        """Every campaign summary carries the claim-boundary caveats."""
        campaign = build_campaign_diagnostic_summary(
            campaign_id="caveat_campaign",
            episode_summaries=[],
        )
        caveats = campaign.caveats
        assert any("causality" in c for c in caveats)
        assert any("diagnostic-only" in c for c in caveats)
        assert any("correction" in c for c in caveats)


# ---------------------------------------------------------------------------
# Fixture / example tests
# ---------------------------------------------------------------------------


class TestCrosswalkExampleFixture:
    """Tests for the deterministic example fixture and export."""

    def test_fixture_schema_version(self) -> None:
        """The example fixture has the correct schema version."""
        fixture = build_crosswalk_example_fixture()
        assert fixture["schema_version"] == REPORT_CROSSWALK_SCHEMA_VERSION
        assert fixture["fixture_id"] == "report_crosswalk.example.v1"
        assert fixture["fixture_version"] == 1

    def test_fixture_has_four_episodes(self) -> None:
        """The example fixture contains four representative episodes."""
        fixture = build_crosswalk_example_fixture()
        assert len(fixture["episodes"]) == 4
        episode_ids = [ep["episode_id"] for ep in fixture["episodes"]]
        assert "ep_001_collision" in episode_ids
        assert "ep_002_unknown_type" in episode_ids
        assert "ep_003_deviation_warn" in episode_ids
        assert "ep_004_deviation_fail_closed" in episode_ids

    def test_export_produces_valid_summaries(self) -> None:
        """Export builds valid episode and campaign summaries from the fixture."""
        report = export_crosswalk_example_fixture()

        assert report["schema_version"] == REPORT_CROSSWALK_SCHEMA_VERSION
        assert len(report["episodes"]) == 4
        assert "campaign" in report
        campaign = report["campaign"]
        assert campaign["episode_count"] == 4
        assert campaign["diagnosis"]["available_count"] == 2
        assert campaign["execution_deviation"]["available_count"] == 2
        assert campaign["execution_deviation"]["fail_closed_count"] == 1

    def test_export_backward_compatible_core_metrics(self) -> None:
        """Export preserves core metrics alongside diagnostic fields."""
        report = export_crosswalk_example_fixture()

        episodes = {ep["episode_id"]: ep for ep in report["episodes"]}
        assert episodes["ep_001_collision"]["core_metrics"]["success"] is False
        assert episodes["ep_001_collision"]["core_metrics"]["collision"] is True
        assert episodes["ep_002_unknown_type"]["core_metrics"]["success"] is True
        assert episodes["ep_003_deviation_warn"]["core_metrics"]["comfort"] == 0.9
        assert episodes["ep_004_deviation_fail_closed"]["core_metrics"]["success"] is True

    def test_export_diagnostic_quality_separate_from_task_success(self) -> None:
        """Diagnostic counts are reported independently from task-success rates."""
        report = export_crosswalk_example_fixture()
        campaign = report["campaign"]

        # 2 out of 4 episodes have diagnosis (50% coverage)
        assert campaign["diagnosis"]["available_count"] == 2
        # 2 out of 4 episodes have deviation results (50% coverage)
        assert campaign["execution_deviation"]["available_count"] == 2
        # Success rate is 3/4 (75%) - computed from core_metrics, not diagnostics
        assert campaign["core_metrics"]["success_rate"] == pytest.approx(3.0 / 4.0)

    def test_fixture_caveats_include_correction_boundary(self) -> None:
        """The fixture caveats document the detection-vs-correction distinction."""
        fixture = build_crosswalk_example_fixture()
        caveats = fixture["caveats"]
        assert any("correction" in c.lower() for c in caveats)
        assert any("causality" in c.lower() for c in caveats)


# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------


class TestSchemaConstants:
    """Tests for crosswalk schema constants."""

    def test_schema_version_format(self) -> None:
        """Schema version follows the versioned convention."""
        assert REPORT_CROSSWALK_SCHEMA_VERSION == "report_crosswalk.v1"

    def test_validity_states_are_complete(self) -> None:
        """Validity states cover all expected values."""
        assert set(FIELD_VALIDITY_STATES) == {
            "available",
            "unavailable",
            "invalid",
            "fallback",
            "degraded",
        }

    def test_provenance_states_are_complete(self) -> None:
        """Provenance states cover all expected values."""
        assert set(FIELD_PROVENANCE_STATES) == {
            "complete",
            "incomplete",
            "unknown",
        }


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestValidation:
    """Tests for crosswalk validation functions."""

    def test_validate_episode_summary_accepts_valid(self) -> None:
        """Valid episode summary passes validation."""
        summary = build_episode_diagnostic_summary(
            episode_id="ep_valid",
            planner_id="orca",
        )
        result = validate_episode_diagnostic_summary(summary.to_dict())
        assert result["episode_id"] == "ep_valid"

    def test_validate_campaign_summary_accepts_valid(self) -> None:
        """Valid campaign summary passes validation."""
        campaign = build_campaign_diagnostic_summary(
            campaign_id="camp_valid",
            episode_summaries=[],
        )
        result = validate_campaign_diagnostic_summary(campaign.to_dict())
        assert result["campaign_id"] == "camp_valid"

    def test_validate_campaign_summary_rejects_missing_campaign_id(self) -> None:
        """Validation rejects campaign summaries without a campaign_id."""
        with pytest.raises(ValueError, match="campaign_id"):
            validate_campaign_diagnostic_summary(
                {
                    "schema_version": REPORT_CROSSWALK_SCHEMA_VERSION,
                    "episode_count": 0,
                    "diagnosis": {},
                    "execution_deviation": {},
                    "core_metrics": {},
                    "caveats": [],
                }
            )

    def test_validate_campaign_summary_rejects_non_int_episode_count(self) -> None:
        """Validation rejects non-integer episode counts."""
        with pytest.raises(ValueError, match="episode_count"):
            validate_campaign_diagnostic_summary(
                {
                    "schema_version": REPORT_CROSSWALK_SCHEMA_VERSION,
                    "campaign_id": "camp",
                    "episode_count": "not_int",
                    "diagnosis": {},
                    "execution_deviation": {},
                    "core_metrics": {},
                    "caveats": [],
                }
            )

    def test_wrong_payload_schema_fails_closed(self) -> None:
        """A source schema mismatch cannot become an available crosswalk field."""
        payload = _collision_diagnosis_payload()
        payload["schema_version"] = "failure_diagnosis.v0"
        summary = build_episode_diagnostic_summary(
            episode_id="ep_bad_payload_schema",
            planner_id="orca",
            diagnosis_payload=payload,
        )

        assert summary.diagnosis_available is False
        assert summary.diagnosis_validity_state == "invalid"
        assert summary.diagnosis_provenance == "incomplete"
        assert summary.diagnosis_record_count == 0
        assert "schema_version" in (summary.diagnosis_validity_reason or "")

    def test_wrong_record_schema_fails_closed(self) -> None:
        """A record schema mismatch is invalid even when the payload wrapper is valid."""
        payload = _collision_diagnosis_payload()
        payload["records"][0]["diagnosis_schema_version"] = "failure_diagnosis.v0"
        summary = build_episode_diagnostic_summary(
            episode_id="ep_bad_record_schema",
            planner_id="orca",
            diagnosis_payload=payload,
        )

        assert summary.diagnosis_available is False
        assert summary.diagnosis_validity_state == "invalid"
        assert summary.diagnosis_provenance == "incomplete"

    def test_wrong_execution_schema_fails_closed(self) -> None:
        """A deviation result from another schema cannot contribute score evidence."""
        result = ExecutionDeviationResult(
            intervention=INTERVENTION_WARN,
            deviation_score=0.6,
            component_deviations=(("robot_position", 0.6),),
            first_threshold_crossing_time_s=0.5,
            input_age_s=0.1,
            fail_closed=False,
            schema_version="execution_deviation.v0",
            claim_boundary=EXECUTION_DEVIATION_CLAIM_BOUNDARY,
        )
        summary = build_episode_diagnostic_summary(
            episode_id="ep_bad_deviation_schema",
            planner_id="dwa",
            execution_deviation_result=result,
        )

        assert summary.execution_deviation_available is False
        assert summary.execution_deviation_validity_state == "invalid"
        assert summary.execution_deviation_provenance == "incomplete"
        assert summary.execution_deviation_score is None

    def test_nested_validation_rejects_unknown_validity_state(self) -> None:
        """The report validator checks nested vocabulary, not only top-level keys."""
        payload = build_episode_diagnostic_summary(
            episode_id="ep_nested_validation",
            planner_id="orca",
        ).to_dict()
        payload["diagnosis"]["validity_state"] = "maybe"

        with pytest.raises(ValueError, match="validity_state"):
            validate_episode_diagnostic_summary(payload)

    def test_fixture_payloads_pass_upstream_validation(self) -> None:
        """The example fixture remains valid under the canonical source validator."""
        for episode in build_crosswalk_example_fixture()["episodes"]:
            payload = episode.get("diagnosis_payload")
            if payload is not None:
                validate_failure_diagnosis_payload(payload)
