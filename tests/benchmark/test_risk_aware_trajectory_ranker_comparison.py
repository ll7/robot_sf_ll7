"""Contract tests for the issue #6676 offline ranker comparison."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.benchmark.run_risk_aware_trajectory_ranker_comparison import (
    CLAIM_BOUNDARY,
    DEFAULT_CONFIG,
    REPORT_SCHEMA_VERSION,
    build_report,
    render_markdown,
)


def test_comparison_config_is_matched_held_out_and_provenance_bearing() -> None:
    """The committed diagnostic report keeps its split, gates, and provenance contract."""
    report = build_report(DEFAULT_CONFIG)

    assert report["schema_version"] == REPORT_SCHEMA_VERSION
    assert report["claim_boundary"] == CLAIM_BOUNDARY
    assert report["evidence_status"] == "smoke/diagnostic"
    assert report["matched_comparison"] == {
        "baseline": "deterministic_primitive",
        "candidate_generators": ["deterministic_primitive", "rbf"],
        "split_policy": "all fixture rows are valid held_out rows",
        "case_count": 4,
        "candidate_budget": 4,
        "same_start_states_local_goals_actor_predictions": True,
        "same_risk_estimator_config": True,
        "same_ranking_weights": True,
        "same_hard_gate_configs": True,
        "hard_gates": ["verify_trajectory", "evaluate_actuator_feasibility"],
        "default_planner_behavior_changed": False,
        "planner_loop_wiring": "not_run; intentionally out of scope",
    }

    provenance = report["provenance"]
    assert len(provenance["git_commit_sha"]) == 40
    assert provenance["seed"] == 6676
    assert provenance["config_sha256"]
    assert provenance["fixture_sha256"]

    for section in (
        "candidate_validity",
        "hard_gate_rejection",
        "selection_differences",
        "risk_score_reliability",
        "timing",
    ):
        assert section in report

    for generator in ("deterministic_primitive", "rbf"):
        validity = report["candidate_validity"]["by_generator"][generator]
        assert validity["candidate_count"] == 16
        assert validity["valid_count"] == 16
        assert validity["invalid_count"] == 0
        reliability = report["risk_score_reliability"]["by_generator"][generator]
        assert reliability["status"] == "pass"
        assert reliability["finite_risk_scores"] == 16
        assert reliability["in_range_risk_scores"] == 16
        assert reliability["complete_provenance_rows"] == 16
        assert reliability["repeatable_risk_scores"] == 16
        per_candidate = report["timing"]["per_candidate"][generator]
        assert len(per_candidate) == 4
        assert all(len(row["ranking_and_gate_ms"]) == 4 for row in per_candidate)


def test_comparison_markdown_leads_with_diagnostic_boundary() -> None:
    """The human-readable report does not turn a smoke into a planner claim."""
    report = build_report(DEFAULT_CONFIG)
    markdown = render_markdown(report)

    assert markdown.startswith("# Risk-aware trajectory ranker comparison")
    assert markdown.index("Claim boundary:") < markdown.index("## Matched comparison")
    assert "not calibration" in markdown.lower()
    assert "planner-loop wiring is not run" in markdown


def test_report_is_json_serializable() -> None:
    """The report can be persisted as the external artifact contract requires."""
    report = build_report(DEFAULT_CONFIG)
    encoded = json.dumps(report, sort_keys=True)
    assert encoded.startswith("{")
    assert (
        Path(report["provenance"]["fixture_path"]).name
        == "risk_aware_trajectory_ranker_comparison_v1.yaml"
    )
