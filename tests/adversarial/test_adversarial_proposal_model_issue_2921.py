"""Tests for adversarial proposal model and comparison script."""

from __future__ import annotations

import json
from pathlib import Path

from robot_sf.adversarial.config import CandidateSpec, Pose2D
from robot_sf.adversarial.proposal_model import FailureArchiveProposalModel
from robot_sf.adversarial.scenario_manifest import AdversarialScenarioManifest
from scripts.adversarial.run_proposal_vs_random_issue_2921 import (
    create_synthetic_archive,
    create_synthetic_search_space,
)


def _candidate(x: float, y: float, speed: float = 1.0) -> CandidateSpec:
    """Helper to construct a CandidateSpec."""
    return CandidateSpec(
        start=Pose2D(x=x, y=y, theta=0.0),
        goal=Pose2D(x=5.0, y=5.0, theta=0.0),
        spawn_time_s=1.0,
        pedestrian_speed_mps=speed,
        pedestrian_delay_s=0.0,
        scenario_seed=42,
    )


def test_proposal_model_initialization_and_blocked_state() -> None:
    """Test that missing/empty archives result in blocked state."""
    # Empty path
    model = FailureArchiveProposalModel(None)
    assert model.state == "blocked"

    # Empty dict
    model_empty_dict = FailureArchiveProposalModel({})
    assert model_empty_dict.state == "blocked"

    # Valid dict
    archive_data = create_synthetic_archive()
    model_active = FailureArchiveProposalModel(archive_data)
    assert model_active.state == "active"
    assert len(model_active.entries) == 2


def test_archive_path_loading_and_malformed_inputs(tmp_path: Path) -> None:
    """Cover path-based archive loading and malformed archive fail-closed states."""
    archive_path = tmp_path / "archive.json"
    archive_path.write_text(json.dumps(create_synthetic_archive()), encoding="utf-8")

    model_from_path = FailureArchiveProposalModel(archive_path)
    assert model_from_path.state == "active"
    assert len(model_from_path.entries) == 2

    missing_model = FailureArchiveProposalModel(tmp_path / "missing.json")
    assert missing_model.state == "blocked"

    empty_path = tmp_path / "empty.json"
    empty_path.write_text("", encoding="utf-8")
    assert FailureArchiveProposalModel(empty_path).state == "blocked"

    malformed_path = tmp_path / "malformed.json"
    malformed_path.write_text("{oops", encoding="utf-8")
    assert FailureArchiveProposalModel(malformed_path).state == "blocked"

    assert FailureArchiveProposalModel({"entries": []}).state == "blocked"


def test_tabular_view_and_scale_fallbacks() -> None:
    """Cover tabular feature extraction and scale fallback without search-space bounds."""
    archive_data = create_synthetic_archive()
    model = FailureArchiveProposalModel(archive_data)

    table = model.get_tabular_view()
    assert table[0]["archive_id"] == "failure_0000"
    assert table[0]["start_x"] == 2.0
    assert table[0]["goal_y"] == 8.0
    assert table[0]["primary_failure"] == "collision"
    assert table[0]["termination_reason"] == "collision"

    assert model._get_candidate_value(archive_data["entries"][0]["candidate"], "start_x") == 2.0
    assert model._get_candidate_value(archive_data["entries"][0]["candidate"], "goal_y") == 8.0
    assert (
        model._get_candidate_value(archive_data["entries"][0]["candidate"], "spawn_time_s") == 1.0
    )
    assert model._get_feature_scale("start_x") == 1.0
    assert model._get_feature_scale("missing_feature") == 1.0


def test_deterministic_ranking() -> None:
    """Test that candidates are ranked deterministically based on archive proximity."""
    archive_data = create_synthetic_archive()
    search_space = create_synthetic_search_space()
    model = FailureArchiveProposalModel(archive_data, search_space)

    # Candidate close to failure_0000 (which is at start_x=2.0, start_y=2.0)
    c_close = _candidate(2.1, 2.1)
    # Candidate far from both
    c_far = _candidate(9.0, 9.0)

    # Rank them
    candidates = [c_far, c_close]
    ranked = model.rank_candidates(candidates, strategy="nearest_neighbor")

    assert len(ranked) == 2
    # The closer candidate should have higher score (less negative distance) and be ranked first
    assert ranked[0][0] == c_close
    assert ranked[1][0] == c_far
    assert ranked[0][1] > ranked[1][1]


def test_score_strategies_and_empty_candidate_ranking() -> None:
    """Cover alternative strategy scoring, unknown strategy fallback, and empty inputs."""
    archive_data = create_synthetic_archive()
    search_space = create_synthetic_search_space()
    model = FailureArchiveProposalModel(archive_data, search_space)

    c_close = _candidate(2.1, 2.1)
    c_far = _candidate(9.0, 9.0)

    weighted_close = model.score_candidate(c_close, strategy="objective_weighted")
    weighted_far = model.score_candidate(c_far, strategy="objective_weighted")
    assert weighted_close > weighted_far

    fallback_score = model.score_candidate(c_close, strategy="unknown_strategy")
    nearest_score = model.score_candidate(c_close, strategy="nearest_neighbor")
    assert fallback_score == nearest_score

    assert model.rank_candidates([]) == []

    blocked = FailureArchiveProposalModel(None)
    blocked_ranked = blocked.rank_candidates([c_close])
    assert blocked_ranked == [(c_close, 0.0)]
    assert blocked.score_candidate(c_close) == 0.0


def test_manifest_emission_and_no_benchmark_promotion() -> None:
    """Test that emitted manifests are valid and contain diagnostic-only evidence boundary."""
    archive_data = create_synthetic_archive()
    search_space = create_synthetic_search_space()
    model = FailureArchiveProposalModel(archive_data, search_space)

    c = _candidate(2.0, 2.0)
    manifest = model.emit_manifest(c, generator_seed=123, candidate_index=5)

    assert isinstance(manifest, AdversarialScenarioManifest)
    assert manifest.generator is not None
    assert manifest.generator.family == "learned_proposal_model"
    assert manifest.generator.generator_id == "FailureArchiveProposalModel"
    assert manifest.generator.seed == 123
    assert manifest.generator.candidate_index == 5

    # Ensure no benchmark promotion: evidence_tier must be diagnostic-only
    assert manifest.evidence_tier == "diagnostic-only"
    assert "diagnostic-only" in manifest.evidence_boundary


def test_certification_status_handling() -> None:
    """Test that candidate certification handles not_available/passed status."""
    archive_data = create_synthetic_archive()
    model = FailureArchiveProposalModel(archive_data)

    c = _candidate(2.0, 2.0)
    dummy_yaml = Path("dummy_scenario.yaml")

    # Advisory mode
    status_advisory = model.certify_candidate(c, dummy_yaml, require_certification=False)
    assert status_advisory.status in ("passed", "failed")

    # Strict mode: fails closed as not_available if the package isn't present
    status_strict = model.certify_candidate(c, dummy_yaml, require_certification=True)
    assert status_strict.status in ("passed", "failed", "not_available")


def test_comparison_report_runs_and_outputs_correct_shape(tmp_path: Path) -> None:
    """Test that the run_proposal_vs_random script produces a valid JSON report."""
    import sys

    from scripts.adversarial.run_proposal_vs_random_issue_2921 import main as script_main

    output_json = tmp_path / "comparison_report.json"

    # Run the script via its main function
    args = ["--budget", "5", "--seed", "10", "--output", str(output_json)]

    # Mock sys.argv
    original_argv = sys.argv
    try:
        sys.argv = ["run_proposal_vs_random_issue_2921.py"] + args
        exit_code = script_main()
    finally:
        sys.argv = original_argv

    assert exit_code == 0
    assert output_json.exists()

    # Load and verify JSON shape
    with open(output_json, encoding="utf-8") as f:
        report = json.load(f)

    assert "state" in report
    assert report["state"] in ("diagnostic_only", "blocked")
    assert report["schema_version"] == "adversarial_proposal_comparison.v1"
    assert report["result_classification"] == "diagnostic_only"
    assert report["held_out_evidence"] is False
    assert report["benchmark_evidence"] is False
    assert report["planner_performance_claim"] is False
    assert report["synthetic_evidence"] is True
    assert "random_metrics" in report
    assert "proposal_metrics" in report
    assert "comparison" in report

    # Verify budget and seed match
    assert report["budget"] == 5
    assert report["seed"] == 10

    # Verify random_metrics and proposal_metrics shapes
    for key in ("mean_objective", "max_objective", "failure_count"):
        assert key in report["random_metrics"]
        assert key in report["proposal_metrics"]
        assert f"{key}_improvement" in report["comparison"]
