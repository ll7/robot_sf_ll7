"""Tests for planner capability ledger and routing/handoff eligibility.

Evidence tier: schema/smoke only. These tests cover the five DoD cases from
the issue #6580 audit decision: valid assignment, unsupported assignment,
missing-evidence failure, valid handoff, and rejected handoff.
"""

from __future__ import annotations

import pytest

from robot_sf.benchmark.planner_capability import (
    PLANNER_CAPABILITY_SCHEMA_VERSION,
    EligibilityResult,
    MeasuredRange,
    PlannerCapabilityEntry,
    check_assignment_eligibility,
    check_handoff_eligibility,
    get_capability_entry,
)


class TestSchemaValidation:
    """Schema version and entry validation."""

    def test_schema_version_is_v1(self) -> None:
        assert PLANNER_CAPABILITY_SCHEMA_VERSION == "planner_capability.v1"

    def test_goal_entry_is_valid(self) -> None:
        entry = get_capability_entry("goal")
        assert entry is not None
        assert entry.validate() == ()

    def test_selector_entry_is_valid(self) -> None:
        entry = get_capability_entry("planner_selector_v2_diagnostic")
        assert entry is not None
        assert entry.validate() == ()

    def test_missing_evidence_refs_fails_validation(self) -> None:
        entry = PlannerCapabilityEntry(
            planner_id="hypothetical",
            evidence_refs=(),
        )
        errors = entry.validate()
        assert any("evidence_refs" in e for e in errors)

    def test_measured_range_without_evidence_fails_validation(self) -> None:
        entry = PlannerCapabilityEntry(
            planner_id="hypothetical",
            speed_range_mps=MeasuredRange(low=0.0, high=1.5, evidence_refs=()),
            evidence_refs=("some/file.py",),
        )
        errors = entry.validate()
        assert any("speed_range_mps" in e for e in errors)

    def test_partial_range_fails_validation(self) -> None:
        entry = PlannerCapabilityEntry(
            planner_id="hypothetical",
            speed_range_mps=MeasuredRange(low=0.0, evidence_refs=("some/file.py",)),
            evidence_refs=("some/file.py",),
        )
        errors = entry.validate()
        assert any("both low and high" in e for e in errors)

    def test_non_relative_evidence_ref_fails_validation(self) -> None:
        entry = PlannerCapabilityEntry(
            planner_id="hypothetical",
            speed_range_mps=MeasuredRange(
                low=0.0,
                high=1.5,
                evidence_refs=("/tmp/evidence.py",),
            ),
            evidence_refs=("https://example.invalid/evidence",),
        )
        errors = entry.validate()
        assert any("repository-relative" in e for e in errors)

    def test_unknown_range_passes_validation(self) -> None:
        entry = PlannerCapabilityEntry(
            planner_id="hypothetical",
            speed_range_mps=MeasuredRange(),
            evidence_refs=("some/file.py",),
        )
        errors = entry.validate()
        assert not any("speed_range_mps" in e for e in errors)

    def test_assumption_flag_on_measured_range(self) -> None:
        entry = get_capability_entry("goal")
        assert entry is not None
        assert entry.speed_range_mps.assumption is True

    def test_unknown_range_preserved_not_inferred(self) -> None:
        entry = get_capability_entry("goal")
        assert entry is not None
        assert entry.pedestrian_density_range.is_unknown


class TestAssignmentEligibility:
    """Assignment eligibility: valid, unsupported, and missing-evidence cases."""

    def test_valid_assignment(self) -> None:
        result = check_assignment_eligibility(
            planner_id="goal",
            scenario="open_space",
            preconditions_met={"goal_position_available": True},
        )
        assert result.eligible is True
        assert result.reasons == ()
        assert result.planner_id == "goal"
        assert len(result.evidence_refs) > 0
        assert "goal_position_available" in result.preconditions_checked

    def test_unsupported_assignment_scenario(self) -> None:
        result = check_assignment_eligibility(
            planner_id="goal",
            scenario="underwater_cave",
            preconditions_met={"goal_position_available": True},
        )
        assert result.eligible is False
        assert any("underwater_cave" in r for r in result.reasons)

    def test_unknown_planner_fails_closed(self) -> None:
        result = check_assignment_eligibility(
            planner_id="nonexistent_planner",
            scenario="open_space",
        )
        assert result.eligible is False
        assert any("not found" in r for r in result.reasons)

    def test_missing_evidence_fails_closed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        entry = PlannerCapabilityEntry(
            planner_id="unverified",
            speed_range_mps=MeasuredRange(low=0.0, high=1.0),
            evidence_refs=("robot_sf/benchmark/algorithm_metadata.py",),
        )
        monkeypatch.setattr(
            "robot_sf.benchmark.planner_capability.get_capability_entry",
            lambda planner_id: entry if planner_id == "unverified" else None,
        )
        result = check_assignment_eligibility(
            planner_id="unverified",
            scenario="open_space",
        )
        assert result.eligible is False
        assert any("speed_range_mps" in reason for reason in result.reasons)

    def test_unmet_precondition_fails_closed(self) -> None:
        result = check_assignment_eligibility(
            planner_id="goal",
            scenario="open_space",
            preconditions_met={},
        )
        assert result.eligible is False
        assert any("goal_position_available" in r for r in result.reasons)

    def test_unknown_supported_scenarios_fail_closed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        entry = PlannerCapabilityEntry(
            planner_id="unscoped",
            evidence_refs=("robot_sf/benchmark/algorithm_metadata.py",),
        )
        monkeypatch.setattr(
            "robot_sf.benchmark.planner_capability.get_capability_entry",
            lambda planner_id: entry if planner_id == "unscoped" else None,
        )
        result = check_assignment_eligibility(
            planner_id="unscoped",
            scenario="open_space",
        )
        assert result.eligible is False
        assert "supported_scenarios is unknown" in result.reasons

    def test_selector_requires_explicit_opt_in(self) -> None:
        result = check_assignment_eligibility(
            planner_id="planner_selector_v2_diagnostic",
            scenario="bottleneck",
            preconditions_met={"candidate_planners_available": True},
        )
        assert result.eligible is False
        assert any("explicit_opt_in" in r for r in result.reasons)


class TestHandoffEligibility:
    """Handoff eligibility: valid and rejected cases."""

    def test_valid_handoff(self) -> None:
        result = check_handoff_eligibility(
            from_planner_id="goal",
            to_planner_id="planner_selector_v2_diagnostic",
            scenario="bottleneck",
            preconditions_met={
                "explicit_opt_in": True,
                "candidate_planners_available": True,
            },
        )
        assert result.eligible is True
        assert result.reasons == ()
        assert result.prior_planner_id == "goal"
        assert result.planner_id == "planner_selector_v2_diagnostic"
        assert len(result.evidence_refs) > 0
        source_entry = get_capability_entry("goal")
        assert source_entry is not None
        assert set(source_entry.evidence_refs).issubset(result.evidence_refs)
        assert "explicit_opt_in" in result.preconditions_checked

    def test_valid_reverse_handoff_selector_to_goal(self) -> None:
        result = check_handoff_eligibility(
            from_planner_id="planner_selector_v2_diagnostic",
            to_planner_id="goal",
            scenario="open_space",
            preconditions_met={"goal_position_available": True},
        )
        assert result.eligible is True
        assert result.prior_planner_id == "planner_selector_v2_diagnostic"

    def test_rejected_handoff_unknown_source(self) -> None:
        result = check_handoff_eligibility(
            from_planner_id="nonexistent",
            to_planner_id="goal",
            scenario="open_space",
        )
        assert result.eligible is False
        assert any("not found" in r for r in result.reasons)
        assert result.prior_planner_id == "nonexistent"

    def test_rejected_handoff_unknown_target(self) -> None:
        result = check_handoff_eligibility(
            from_planner_id="goal",
            to_planner_id="nonexistent",
            scenario="open_space",
        )
        assert result.eligible is False
        assert any("not found" in r for r in result.reasons)
        assert result.prior_planner_id == "goal"

    def test_rejected_handoff_target_not_in_handoff_targets(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        isolated = PlannerCapabilityEntry(
            planner_id="isolated",
            supported_scenarios=("open_space",),
            handoff_targets=(),
            evidence_refs=("some/file.py",),
        )
        lookup = {"isolated": isolated, "goal": get_capability_entry("goal")}
        monkeypatch.setattr(
            "robot_sf.benchmark.planner_capability.get_capability_entry",
            lookup.get,
        )
        result = check_handoff_eligibility(
            from_planner_id="isolated",
            to_planner_id="goal",
            scenario="open_space",
            preconditions_met={"goal_position_available": True},
        )
        assert result.eligible is False
        assert any("handoff_targets" in r for r in result.reasons)
        assert result.prior_planner_id == "isolated"

    def test_handoff_records_prior_planner(self) -> None:
        result = check_handoff_eligibility(
            from_planner_id="goal",
            to_planner_id="planner_selector_v2_diagnostic",
            scenario="corridor",
            preconditions_met={
                "explicit_opt_in": True,
                "candidate_planners_available": True,
            },
        )
        assert isinstance(result, EligibilityResult)
        assert result.prior_planner_id == "goal"

    def test_handoff_fails_closed_on_unmet_target_precondition(self) -> None:
        result = check_handoff_eligibility(
            from_planner_id="goal",
            to_planner_id="planner_selector_v2_diagnostic",
            scenario="bottleneck",
            preconditions_met={},
        )
        assert result.eligible is False
        assert any("explicit_opt_in" in r for r in result.reasons)

    def test_handoff_fails_closed_on_invalid_source_evidence(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        source = PlannerCapabilityEntry(
            planner_id="unverified",
            speed_range_mps=MeasuredRange(low=0.0, high=1.0),
            handoff_targets=("goal",),
            evidence_refs=("robot_sf/benchmark/algorithm_metadata.py",),
        )
        lookup = {"unverified": source, "goal": get_capability_entry("goal")}
        monkeypatch.setattr(
            "robot_sf.benchmark.planner_capability.get_capability_entry",
            lookup.get,
        )
        result = check_handoff_eligibility(
            from_planner_id="unverified",
            to_planner_id="goal",
            scenario="open_space",
            preconditions_met={"goal_position_available": True},
        )
        assert result.eligible is False
        assert any("speed_range_mps" in reason for reason in result.reasons)
