"""Contract tests for the issue #7315 feasibility-first diagnostic."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot_sf.adversarial.feasibility_first import (
    CHECK_NAMES,
    FeasibilityCandidate,
    FeasibilityCheck,
    FeasibilityFirstError,
    build_comparison_report,
    build_fixture_candidates,
    rank_feasible_candidates,
    run_fixture_diagnostic,
    sample_risk_feedback,
    sample_seeded_uniform,
    validate_report,
)

CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "benchmarks"
    / "issue_7315_feasibility_first_smoke.yaml"
)


def test_fixture_has_explicit_four_way_feasibility_and_rejection_ledger() -> None:
    """Missing feasibility is not silently converted into a valid safety row."""
    candidates = build_fixture_candidates()
    assert all(
        tuple(check.name for check in candidate.checks) == CHECK_NAMES for candidate in candidates
    )
    assert sum(candidate.feasible for candidate in candidates) == 4
    assert sum(not candidate.feasible for candidate in candidates) == 4
    assert any(
        candidate.checks[-1].status == "unavailable" and not candidate.feasible
        for candidate in candidates
    )


def test_hierarchical_ranking_is_deterministic_and_excludes_rejected_candidates() -> None:
    """Risk-feature ordering is stable and never promotes a rejected candidate."""
    first = [
        candidate.candidate_id for candidate in rank_feasible_candidates(build_fixture_candidates())
    ]
    second = [
        candidate.candidate_id for candidate in rank_feasible_candidates(build_fixture_candidates())
    ]
    assert first == second
    assert first[:3] == ["crossing_high_risk", "blind_corner_diverse", "doorway_controlled"]
    assert "bottleneck_geometry_reject" not in first
    assert "crossing_simulator_unavailable" not in first


def test_seeded_uniform_sampling_is_reproducible() -> None:
    """The baseline draw is tied to one explicit seed and candidate order."""
    candidates = build_fixture_candidates()
    sample_a = sample_seeded_uniform(candidates, budget=4, seed=7315)
    sample_b = sample_seeded_uniform(candidates, budget=4, seed=7315)
    assert [candidate.candidate_id for candidate in sample_a] == [
        candidate.candidate_id for candidate in sample_b
    ]


def test_risk_feedback_rejects_budget_larger_than_feasible_pool() -> None:
    """The diagnostic fails closed instead of filling a risk sample with invalid rows."""
    with pytest.raises(FeasibilityFirstError, match="feasible candidates"):
        sample_risk_feedback(build_fixture_candidates(), budget=5)


def test_report_is_schema_valid_and_marks_safety_severity_unavailable(tmp_path: Path) -> None:
    """The report preserves provenance and the no-simulator evidence boundary."""
    output = tmp_path / "report.json"
    report = run_fixture_diagnostic(CONFIG_PATH, output_path=output)
    validate_report(report)
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted == report
    assert report["comparison"]["safety_event_severity"]["status"] == "unavailable"
    assert report["governance"]["simulator_executed"] is False
    assert report["feasibility"]["invalid_candidates_excluded_from_safety_denominators"] is True


def test_report_comparison_has_distinct_sampling_methods() -> None:
    """Uniform and hierarchical selection are reported as separate diagnostics."""
    report = build_comparison_report(
        build_fixture_candidates(),
        budget=4,
        seed=7315,
        config_sha256="a" * 64,
        criticality_threshold=0.6,
    )
    methods = report["comparison"]["methods"]
    assert set(methods) == {"seeded_uniform", "risk_feedback_hierarchical_value"}
    assert methods["risk_feedback_hierarchical_value"]["valid_scenario_rate"] == 1.0
    assert methods["seeded_uniform"]["rejected_count"] >= 0
    assert report["comparison"]["existing_adversarial_baseline"]["status"] == "not_executed"


def test_candidate_round_trip_rejects_contradictory_derived_fields() -> None:
    """A report cannot override the pass/fail decision derived from its checks."""
    payload = build_fixture_candidates()[4].to_dict()
    payload["feasible"] = True
    with pytest.raises(FeasibilityFirstError, match="contradicts checks"):
        FeasibilityCandidate.from_mapping(payload)


def test_candidate_round_trip_preserves_fixture_identity() -> None:
    """Candidate serialization keeps control identity and all check evidence."""
    original = build_fixture_candidates()[0]
    restored = FeasibilityCandidate.from_mapping(original.to_dict())
    assert restored == original


def test_passing_check_requires_explicit_evidence() -> None:
    """A pass without predicate evidence fails closed instead of entering the denominator."""
    with pytest.raises(FeasibilityFirstError, match="evidence"):
        FeasibilityCheck.from_mapping(
            {"status": "pass", "reason": "looks valid"},
            expected_name="kinematic_reachability",
        )
