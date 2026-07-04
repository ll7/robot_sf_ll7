"""Tests for adversarial proposal model and comparison script."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot_sf.adversarial.config import CandidateSpec, Pose2D
from robot_sf.adversarial.proposal_model import FailureArchiveProposalModel
from robot_sf.adversarial.scenario_manifest import AdversarialScenarioManifest
from scripts.adversarial.run_proposal_vs_random_issue_2921 import (
    classify_issue_2921_stop_rule,
    create_synthetic_archive,
    create_synthetic_search_space,
)


def _held_out_comparison(*, mean_delta: float, failure_delta: int) -> dict[str, float | int | str]:
    """Build a comparison payload for the #2921 stop-rule classifier."""
    return {
        "interpretation": "independent_planner_execution_outcomes",
        "mean_objective_improvement": mean_delta,
        "max_objective_improvement": mean_delta,
        "failure_count_improvement": failure_delta,
    }


def test_classify_issue_2921_stop_rule_blocks_without_held_out_evidence() -> None:
    """The stop rule must fail closed to blocked when held-out evidence is unavailable."""
    decision = classify_issue_2921_stop_rule(
        held_out_evidence=False,
        held_out_status="not_available_no_disjoint_split",
        comparison=_held_out_comparison(mean_delta=5.0, failure_delta=5),
    )
    assert decision["status"] == "blocked"
    assert decision["reason"] == "not_available_no_disjoint_split"
    assert decision["evidence_tier"] == "analysis_only"


def test_classify_issue_2921_stop_rule_stop_on_negative_deltas() -> None:
    """Negative held-out deltas classify as stop (do not expand the proposal lane)."""
    decision = classify_issue_2921_stop_rule(
        held_out_evidence=True,
        held_out_status="eligible_held_out_diagnostic",
        comparison=_held_out_comparison(mean_delta=-0.5, failure_delta=-2),
    )
    assert decision["status"] == "stop"
    assert decision["evidence_tier"] == "diagnostic_only"


def test_classify_issue_2921_stop_rule_revise_on_neutral_deltas() -> None:
    """Neutral (zero) held-out deltas classify as revise before another empirical batch."""
    decision = classify_issue_2921_stop_rule(
        held_out_evidence=True,
        held_out_status="eligible_held_out_diagnostic",
        comparison=_held_out_comparison(mean_delta=0.0, failure_delta=0),
    )
    assert decision["status"] == "revise"
    assert decision["evidence_tier"] == "diagnostic_only"


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
    assert model_empty_dict.state_reason == "malformed_archive_payload"

    entries_only = FailureArchiveProposalModel({"entries": create_synthetic_archive()["entries"]})
    assert entries_only.state == "blocked"
    assert entries_only.state_reason.startswith("invalid_failure_archive_schema:")

    # Valid dict
    archive_data = create_synthetic_archive()
    model_active = FailureArchiveProposalModel(archive_data)
    assert model_active.state == "active"
    assert model_active.state_reason == "archive_loaded"
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
    assert FailureArchiveProposalModel({"entries": "not-a-list"}).state == "blocked"
    assert FailureArchiveProposalModel({"entries": ["not-a-dict"]}).state == "blocked"
    assert FailureArchiveProposalModel({"entries": [{"candidate": "not-a-dict"}]}).state == (
        "blocked"
    )


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
    assert status_advisory.status in ("passed", "failed", "not_available")

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
    assert report["result_classification"] == "plumbing_validation_only"
    assert report["held_out_evidence"] is False
    assert report["benchmark_evidence"] is False
    assert report["planner_performance_claim"] is False
    assert report["synthetic_archive"] is True
    assert report["synthetic_search_space"] is True
    assert report["archive_evaluation_provenance"]["disjointness_checks_passed"] is False
    assert report["comparison"]["interpretation"] == (
        "plumbing_only_circular_archive_nearness_objective"
    )
    assert report["null_tests"]["required_for_held_out_claim"] is True
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


def test_comparison_script_rejects_negative_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """Negative budgets should fail during argument parsing before sampling."""
    from scripts.adversarial.run_proposal_vs_random_issue_2921 import parse_args

    monkeypatch.setattr(
        "sys.argv",
        ["run_proposal_vs_random_issue_2921.py", "--budget", "-1"],
    )

    with pytest.raises(SystemExit) as exc_info:
        parse_args()

    assert exc_info.value.code == 2


def test_real_archive_without_search_space_fails_closed(tmp_path: Path, monkeypatch) -> None:
    """Real archive runs should not claim held-out evidence without a real search space."""
    from scripts.adversarial.run_proposal_vs_random_issue_2921 import main as script_main

    archive_path = tmp_path / "archive.json"
    output_json = tmp_path / "report.json"
    archive_path.write_text(json.dumps(create_synthetic_archive()), encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_proposal_vs_random_issue_2921.py",
            "--archive",
            archive_path.as_posix(),
            "--budget",
            "3",
            "--output",
            output_json.as_posix(),
        ],
    )

    assert script_main() == 0
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["state"] == "blocked"
    assert report["held_out_evidence"] is False
    assert report["synthetic_archive"] is False
    assert report["synthetic_search_space"] is True
    assert report["archive_evaluation_provenance"]["disjointness_checks_passed"] is False


_SEARCH_SPACE_YAML = """\
schema_version: adversarial-search-space.v1
variables:
  start_x: {min: 1.0, max: 3.0}
  start_y: {min: 2.0, max: 4.0}
  goal_x: {min: 7.0, max: 9.0}
  goal_y: {min: 2.0, max: 4.0}
  spawn_time_s: {min: 0.0, max: 2.0}
  pedestrian_speed_mps: {min: 0.8, max: 1.4}
  pedestrian_delay_s: {min: 0.0, max: 2.0}
  scenario_seed: {min: 100, max: 999}
constraints:
  min_start_goal_distance_m: 2.0
"""


def _two_family_archive() -> dict:
    """Build a small two-family archive with disjoint families/ids/seeds."""
    entries = []
    for family in ("goal_collision", "orca_collision"):
        for i in range(3):
            seed = (100 if family == "goal_collision" else 200) + i
            entries.append(
                {
                    "archive_id": f"{family}_{i}",
                    "cluster_key": family,
                    "candidate": {
                        "start": {"x": 2.0, "y": 3.0},
                        "goal": {"x": 8.0, "y": 3.0},
                        "spawn_time_s": 1.0,
                        "pedestrian_speed_mps": 1.0,
                        "pedestrian_delay_s": 0.5,
                        "scenario_seed": seed,
                    },
                    "failure_attribution": {"primary_failure": "collision"},
                    "objective_value": 8.0,
                    "normalized_perturbation": 0.1,
                }
            )
    return {"schema_version": "adversarial_failure_archive.v1", "entries": entries}


def _two_family_archive_with_seed_overlap() -> dict:
    """Build a two-family archive whose families share scenario seeds."""
    archive = _two_family_archive()
    for entry in archive["entries"]:
        suffix = int(entry["archive_id"].rsplit("_", maxsplit=1)[1])
        entry["candidate"]["scenario_seed"] = 100 + suffix
    return archive


def test_active_real_archive_computes_disjoint_provenance_but_fails_closed(
    tmp_path: Path, monkeypatch
) -> None:
    """An active real-archive run computes a real disjoint split yet stays fail-closed."""
    from scripts.adversarial.run_proposal_vs_random_issue_2921 import main as script_main

    archive_path = tmp_path / "archive.json"
    search_space_path = tmp_path / "search_space.yaml"
    output_json = tmp_path / "report.json"
    archive_path.write_text(json.dumps(_two_family_archive()), encoding="utf-8")
    search_space_path.write_text(_SEARCH_SPACE_YAML, encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_proposal_vs_random_issue_2921.py",
            "--archive",
            archive_path.as_posix(),
            "--search-space",
            search_space_path.as_posix(),
            "--budget",
            "3",
            "--seed",
            "7",
            "--output",
            output_json.as_posix(),
        ],
    )

    assert script_main() == 0
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["state"] == "active"
    assert report["synthetic_archive"] is False
    assert report["synthetic_search_space"] is False

    provenance = report["archive_evaluation_provenance"]
    assert provenance["split_policy"] == "disjoint_scenario_family"
    assert provenance["disjointness_checks_passed"] is True
    assert provenance["scenario_family_overlap"] == []
    assert provenance["archive_id_overlap"] == []
    assert provenance["fit_size"] > 0
    assert provenance["eval_size"] > 0

    # Held-out yield stays fail-closed: independent planner outcomes are not wired yet.
    assert report["held_out_evidence"] is False
    assert provenance["held_out_evidence_status"] == (
        "not_available_requires_independent_planner_outcomes"
    )


def test_real_archive_with_independent_outcomes_becomes_diagnostic_only(
    tmp_path: Path, monkeypatch
) -> None:
    """A valid independent outcome packet opens only the diagnostic held-out gate."""
    from scripts.adversarial.run_proposal_vs_random_issue_2921 import main as script_main

    archive_path = tmp_path / "archive.json"
    search_space_path = tmp_path / "search_space.yaml"
    outcomes_path = tmp_path / "outcomes.json"
    output_json = tmp_path / "report.json"
    archive = _two_family_archive()
    archive_path.write_text(json.dumps(archive), encoding="utf-8")
    search_space_path.write_text(_SEARCH_SPACE_YAML, encoding="utf-8")
    from robot_sf.adversarial.disjoint_evaluation import archive_sha256, disjoint_family_split

    split = disjoint_family_split(archive["entries"], eval_fraction=0.5, seed=7)
    outcomes_path.write_text(
        json.dumps(
            {
                "schema_version": "adversarial_independent_outcomes.v1",
                "source": "unit-test-fixture",
                "artifact": "docs/context/evidence/unit-test.json",
                "eval_archive_sha256": archive_sha256(split.eval_entries),
                "outcome_source": "planner_execution",
                "objective": "certified_failure_outcome",
                "proposal_outcomes": [10.0, 10.0, 10.0, 10.0],
                "random_outcomes": [0.0, 0.0, 0.0, 0.0],
                "ranked_outcomes": [10.0, 10.0, 10.0, 10.0, 0.0, 0.0, 0.0, 0.0],
                "certification_statuses": ["passed"] * 8,
                "row_statuses": ["success"] * 8,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_proposal_vs_random_issue_2921.py",
            "--archive",
            archive_path.as_posix(),
            "--search-space",
            search_space_path.as_posix(),
            "--evaluation-outcomes",
            outcomes_path.as_posix(),
            "--budget",
            "4",
            "--seed",
            "7",
            "--null-test-permutations",
            "200",
            "--output",
            output_json.as_posix(),
        ],
    )

    assert script_main() == 0
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["held_out_evidence"] is True
    assert report["benchmark_evidence"] is False
    assert report["planner_performance_claim"] is False
    assert report["result_classification"] == "held_out_diagnostic_only"
    assert report["comparison"]["interpretation"] == "independent_planner_execution_outcomes"
    assert report["issue_2921_stop_rule"] == {
        "status": "continue",
        "reason": "diagnostic held-out deltas are positive; run the next predeclared proof step",
        "evidence_tier": "diagnostic_only",
        "claim_boundary": (
            "issue #2921 stop-rule classification from held-out diagnostic evidence only; "
            "not benchmark, paper, or planner-performance evidence"
        ),
    }
    assert report["archive_evaluation_provenance"]["held_out_evidence_status"] == (
        "eligible_held_out_diagnostic"
    )
    assert report["independent_outcome_evaluation"]["certification_available"] is True
    assert report["independent_outcome_evaluation"]["null_tests_reject_null"] is True


def test_real_archive_seed_overlap_cannot_be_held_out_evidence(tmp_path: Path, monkeypatch) -> None:
    """Even independent outcomes cannot open the held-out gate with seed overlap."""
    from scripts.adversarial.run_proposal_vs_random_issue_2921 import main as script_main

    archive_path = tmp_path / "archive.json"
    search_space_path = tmp_path / "search_space.yaml"
    outcomes_path = tmp_path / "outcomes.json"
    output_json = tmp_path / "report.json"
    archive = _two_family_archive_with_seed_overlap()
    archive_path.write_text(json.dumps(archive), encoding="utf-8")
    search_space_path.write_text(_SEARCH_SPACE_YAML, encoding="utf-8")

    from robot_sf.adversarial.disjoint_evaluation import archive_sha256, disjoint_family_split

    split = disjoint_family_split(archive["entries"], eval_fraction=0.5, seed=7)
    outcomes_path.write_text(
        json.dumps(
            {
                "schema_version": "adversarial_independent_outcomes.v1",
                "source": "unit-test-fixture",
                "artifact": "docs/context/evidence/unit-test.json",
                "eval_archive_sha256": archive_sha256(split.eval_entries),
                "outcome_source": "planner_execution",
                "objective": "certified_failure_outcome",
                "proposal_outcomes": [10.0, 10.0, 10.0, 10.0],
                "random_outcomes": [0.0, 0.0, 0.0, 0.0],
                "ranked_outcomes": [10.0, 10.0, 10.0, 10.0, 0.0, 0.0, 0.0, 0.0],
                "certification_statuses": ["passed"] * 8,
                "row_statuses": ["success"] * 8,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_proposal_vs_random_issue_2921.py",
            "--archive",
            archive_path.as_posix(),
            "--search-space",
            search_space_path.as_posix(),
            "--evaluation-outcomes",
            outcomes_path.as_posix(),
            "--budget",
            "4",
            "--seed",
            "7",
            "--null-test-permutations",
            "200",
            "--output",
            output_json.as_posix(),
        ],
    )

    assert script_main() == 0
    report = json.loads(output_json.read_text(encoding="utf-8"))
    provenance = report["archive_evaluation_provenance"]
    assert report["held_out_evidence"] is False
    assert report["result_classification"] == "plumbing_validation_only"
    assert report["comparison"]["interpretation"] == (
        "independent_outcomes_rejected_by_held_out_gate"
    )
    assert report["issue_2921_stop_rule"]["status"] == "blocked"
    assert report["issue_2921_stop_rule"]["reason"] == "not_available_no_disjoint_split"
    assert provenance["held_out_evidence_status"] == "not_available_no_disjoint_split"
    assert provenance["disjointness_checks_passed"] is False
    assert provenance["seed_overlap"] == [100, 101, 102]
    assert provenance["seed_overlap_count"] == 3
    assert provenance["disjointness_failure_reasons"] == ["seed_overlap"]
    assert provenance["seed_overlap_invalidates_held_out_evidence"] is True
    assert report["independent_outcome_evaluation"]["independent_outcomes_available"] is True
    assert report["independent_outcome_evaluation"]["null_tests_reject_null"] is True


def test_real_archive_with_circular_outcomes_stays_fail_closed(tmp_path: Path, monkeypatch) -> None:
    """Archive-nearness outcome packets are rejected as circular."""
    from scripts.adversarial.run_proposal_vs_random_issue_2921 import main as script_main

    archive_path = tmp_path / "archive.json"
    search_space_path = tmp_path / "search_space.yaml"
    outcomes_path = tmp_path / "outcomes.json"
    output_json = tmp_path / "report.json"
    archive_path.write_text(json.dumps(_two_family_archive()), encoding="utf-8")
    search_space_path.write_text(_SEARCH_SPACE_YAML, encoding="utf-8")
    outcomes_path.write_text(
        json.dumps(
            {
                "outcome_source": "planner_execution",
                "objective": "archive_nearness",
                "proposal_outcomes": [10.0],
                "random_outcomes": [0.0],
                "ranked_outcomes": [10.0, 0.0],
                "certification_statuses": ["passed", "passed"],
                "row_statuses": ["success", "success"],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_proposal_vs_random_issue_2921.py",
            "--archive",
            archive_path.as_posix(),
            "--search-space",
            search_space_path.as_posix(),
            "--evaluation-outcomes",
            outcomes_path.as_posix(),
            "--budget",
            "1",
            "--seed",
            "7",
            "--output",
            output_json.as_posix(),
        ],
    )

    assert script_main() == 0
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["held_out_evidence"] is False
    assert report["archive_evaluation_provenance"]["held_out_evidence_status"] == (
        "not_available_requires_independent_planner_outcomes"
    )
    assert report["independent_outcome_evaluation"]["status"] == (
        "blocked_invalid_independent_outcomes"
    )
    assert report["issue_2921_stop_rule"]["status"] == "blocked"
    assert report["issue_2921_stop_rule"]["reason"] == (
        "not_available_requires_independent_planner_outcomes"
    )
