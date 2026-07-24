"""Tests for adversarial proposal model and comparison script."""

from __future__ import annotations

import hashlib
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


def test_classify_issue_2921_stop_rule_is_inconclusive_on_neutral_deltas() -> None:
    """Neutral (zero) held-out deltas classify as inconclusive without a new batch."""
    decision = classify_issue_2921_stop_rule(
        held_out_evidence=True,
        held_out_status="eligible_held_out_diagnostic",
        comparison=_held_out_comparison(mean_delta=0.0, failure_delta=0),
    )
    assert decision["status"] == "inconclusive"
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

    assert script_main() == 1
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

_TEST_PLANNER_CONFIG_SHA256 = "b" * 64
_TEST_MANIFEST_SHA256 = "a" * 64
_TEST_RECORD_SHA256 = "c" * 64


def _two_family_archive() -> dict:
    """Build a small two-family archive with disjoint families/ids/seeds."""
    entries = []
    for family in ("classic_group_crossing_medium", "classic_cross_trap_medium"):
        for i in range(3):
            seed = (100 if family == "classic_group_crossing_medium" else 200) + i
            entries.append(
                {
                    "archive_id": f"{family}_{i}",
                    "cluster_key": family,
                    "scenario_family": family,
                    "target_planner": "social_force",
                    "provenance": {
                        "target_planner": "social_force",
                        "config_sha256": _TEST_PLANNER_CONFIG_SHA256,
                    },
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


def _outcome_row(arm: str, rank: int, outcome: float) -> dict:
    """Build one contract-bound v2 independent-outcome row for runner integration tests."""
    return {
        "candidate_id": f"{arm}_{rank}",
        "manifest_sha256": _TEST_MANIFEST_SHA256,
        "selection_arm": arm,
        "rank": rank,
        "candidate_pool_seed": 42,
        "target_planner_id": "social_force",
        "planner_config_sha256": _TEST_PLANNER_CONFIG_SHA256,
        "scenario_family": "classic_cross_trap_medium",
        "scenario_seed": 1000 + rank,
        "execution_seed": 2000 + rank + (100 if arm == "random" else 0),
        "execution_commit": "ecf997d",
        "command_lineage": "uv run robot_sf_bench run --algo social_force",
        "execution_status": "native",
        "termination_reason": "collision" if outcome >= 8.0 else "goal_reached",
        "independent_failure_outcome": outcome,
        "scenario_certification_status": "passed",
        "candidate_certification_status": "passed",
        "replay_lineage": "replay.jsonl",
        "record_hash": _TEST_RECORD_SHA256,
        "exclusion_reason": None,
    }


def _write_frozen_contract_and_outcomes(
    tmp_path: Path,
    archive_path: Path,
    search_space_path: Path,
    proposal_outcomes: list[float],
    random_outcomes: list[float],
) -> tuple[Path, Path]:
    """Write a complete local frozen contract and matching v2 outcomes for runner tests."""
    from robot_sf.adversarial.disjoint_evaluation import archive_sha256, disjoint_family_split

    archive = json.loads(archive_path.read_text(encoding="utf-8"))
    split = disjoint_family_split(archive["entries"], eval_fraction=0.5, seed=7)
    proposal_first = sum(proposal_outcomes) >= sum(random_outcomes)
    proposal_rank_offset = 0 if proposal_first else len(random_outcomes)
    random_rank_offset = len(proposal_outcomes) if proposal_first else 0
    rows = [
        _outcome_row("proposal", proposal_rank_offset + rank, outcome)
        for rank, outcome in enumerate(proposal_outcomes)
    ] + [
        _outcome_row("random", random_rank_offset + rank, outcome)
        for rank, outcome in enumerate(random_outcomes)
    ]
    candidates = [
        {
            "candidate_id": row["candidate_id"],
            "selection_arm": row["selection_arm"],
            "rank": row["rank"],
            "candidate_pool_seed": row["candidate_pool_seed"],
            "scenario_seed": row["scenario_seed"],
            "execution_seeds": [row["execution_seed"]],
        }
        for row in rows
    ]
    fit_entries = [
        entry
        for entry in archive["entries"]
        if entry["scenario_family"] == "classic_group_crossing_medium"
    ]
    contract = {
        "study_id": "unit-test-same-planner-contract",
        "contract_status": "frozen",
        "external_prerequisites": [{"issue": 6139, "status": "satisfied"}],
        "tracked_archive_path": archive_path.as_posix(),
        "archive_raw_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        "archive_payload_sha256": archive_sha256(archive),
        "target_planner": "social_force",
        "target_planner_config_sha256": _TEST_PLANNER_CONFIG_SHA256,
        "fit_scenario_family": "classic_group_crossing_medium",
        "fit_entry_count": len(fit_entries),
        "fit_entry_ids": [entry["archive_id"] for entry in fit_entries],
        "fit_entries_payload_sha256": archive_sha256(fit_entries),
        "excluded_entry_count": 0,
        "excluded_entry_ids": [],
        "eval_scenario_family": "classic_cross_trap_medium",
        "search_space_path": search_space_path.as_posix(),
        "search_space_sha256": hashlib.sha256(search_space_path.read_bytes()).hexdigest(),
        "study_parameters": {
            "candidate_budget_per_arm": len(proposal_outcomes),
            "candidate_pool_size": 2 * len(proposal_outcomes),
            "confirmation_seeds_per_candidate": 1,
            "minimally_important_effect_margin": 0.1,
            "overlapping_candidate_policy": "deterministic_disjoint_assignment",
        },
        "power_sensitivity": {
            "status": "computed",
            "candidate_budget_per_arm": len(proposal_outcomes),
            "minimum_effect_margin": 0.1,
            "estimated_power": 0.8,
        },
        "outcome_admission": {
            "schema_version": "adversarial_independent_outcomes.v2",
            "execution_status": "native",
            "candidate_aggregation": "candidate_level_binary_yield",
            "fallback_degraded_policy": "exclude",
            "stable_failure_attribution_required": True,
            "deterministic_replay": {"exact_signature_match_required": True},
            "independent_seed_confirmation": {"minimum_confirmed_count": 1},
            "candidate_manifest": {
                "status": "frozen",
                "sha256": _TEST_MANIFEST_SHA256,
                "candidates": candidates,
            },
        },
        "decision_rule": {
            "positive_outcome": "continue",
            "negative_outcome": "stop",
            "neutral_outcome": "inconclusive",
            "invalid_or_fallback_outcome": "inconclusive",
        },
    }
    contract_path = tmp_path / "contract.json"
    outcomes_path = tmp_path / "outcomes.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    outcomes_path.write_text(
        json.dumps(
            {
                "schema_version": "adversarial_independent_outcomes.v2",
                "source": "unit-test-fixture",
                "artifact": "docs/context/evidence/unit-test.json",
                "eval_archive_sha256": archive_sha256(split.eval_entries),
                "outcome_source": "planner_execution",
                "objective": "certified_failure_outcome",
                "rows": rows,
            }
        ),
        encoding="utf-8",
    )
    return contract_path, outcomes_path


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
    output_json = tmp_path / "report.json"
    archive = _two_family_archive()
    archive_path.write_text(json.dumps(archive), encoding="utf-8")
    search_space_path.write_text(_SEARCH_SPACE_YAML, encoding="utf-8")
    contract_path, outcomes_path = _write_frozen_contract_and_outcomes(
        tmp_path,
        archive_path,
        search_space_path,
        [10.0, 10.0, 10.0, 10.0],
        [0.0, 0.0, 0.0, 0.0],
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_proposal_vs_random_issue_2921.py",
            "--archive",
            archive_path.as_posix(),
            "--search-space",
            search_space_path.as_posix(),
            "--contract",
            contract_path.as_posix(),
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
    output_json = tmp_path / "report.json"
    archive = _two_family_archive_with_seed_overlap()
    archive_path.write_text(json.dumps(archive), encoding="utf-8")
    search_space_path.write_text(_SEARCH_SPACE_YAML, encoding="utf-8")
    contract_path, outcomes_path = _write_frozen_contract_and_outcomes(
        tmp_path,
        archive_path,
        search_space_path,
        [10.0, 10.0, 10.0, 10.0],
        [0.0, 0.0, 0.0, 0.0],
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_proposal_vs_random_issue_2921.py",
            "--archive",
            archive_path.as_posix(),
            "--search-space",
            search_space_path.as_posix(),
            "--contract",
            contract_path.as_posix(),
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


def test_contract_verification_cli(tmp_path: Path, monkeypatch) -> None:
    """Test that side-effect-free contract verification CLI command passes."""
    from scripts.adversarial.run_proposal_vs_random_issue_2921 import main as script_main

    contract_path = Path("configs/adversarial/issue_3275_same_planner_contract.json")
    output_json = tmp_path / "verif.json"

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_proposal_vs_random_issue_2921.py",
            "--check-contract",
            contract_path.as_posix(),
            "--output",
            output_json.as_posix(),
        ],
    )

    assert script_main() == 1
    assert output_json.exists()
    verif = json.loads(output_json.read_text(encoding="utf-8"))
    assert verif["checks_passed"] is False
    assert verif["status"] == "failed"
    assert verif["fit_entry_count"] == 12
    assert verif["excluded_entry_count"] == 5
    assert verif["feature_semantics_audit"]["passed"] is False
    assert (
        "external_prerequisite_unsatisfied:issue=6139:status=pending_corrected_recertification"
        in (verif["blocking_reasons"])
    )


def test_normal_contract_run_refuses_provisional_contract_before_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Normal --contract runs stop before candidate selection when the contract is provisional."""
    from scripts.adversarial.run_proposal_vs_random_issue_2921 import main as script_main

    output_json = tmp_path / "blocked.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_proposal_vs_random_issue_2921.py",
            "--contract",
            "configs/adversarial/issue_3275_same_planner_contract.json",
            "--output",
            output_json.as_posix(),
        ],
    )

    assert script_main() == 1
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["state"] == "blocked"
    assert "no candidates were selected" in report["reason"]
    assert "proposal_metrics" not in report
    assert report["contract_verification"]["checks_passed"] is False


def test_train_only_ranking_isolation() -> None:
    """Excluded goal/cross-trap entries cannot alter fit model entries or candidate scores."""
    from robot_sf.adversarial.proposal_model import FailureArchiveProposalModel, isolate_fit_entries

    raw_archive = {
        "schema_version": "adversarial_failure_archive.v1",
        "entries": [
            {
                "archive_id": "sf_0",
                "target_planner": "social_force",
                "scenario_family": "classic_group_crossing_medium",
                "candidate": {
                    "start": {"x": 2.0, "y": 2.0},
                    "goal": {"x": 8.0, "y": 8.0},
                    "spawn_time_s": 1.0,
                    "pedestrian_speed_mps": 1.0,
                    "pedestrian_delay_s": 0.0,
                    "scenario_seed": 1,
                },
                "failure_attribution": {"primary_failure": "collision"},
            },
            {
                "archive_id": "goal_0",
                "target_planner": "goal",
                "scenario_family": "classic_cross_trap_medium",
                "candidate": {
                    "start": {"x": 9.0, "y": 9.0},
                    "goal": {"x": 1.0, "y": 1.0},
                    "spawn_time_s": 0.0,
                    "pedestrian_speed_mps": 2.0,
                    "pedestrian_delay_s": 1.0,
                    "scenario_seed": 2,
                },
                "failure_attribution": {"primary_failure": "timeout"},
            },
        ],
    }

    search_space = create_synthetic_search_space()
    fit_data = isolate_fit_entries(
        raw_archive, allowed_fit_ids={"sf_0"}, target_planner="social_force"
    )
    model = FailureArchiveProposalModel(fit_data, search_space)

    assert len(model.entries) == 1
    assert model.entries[0]["archive_id"] == "sf_0"

    cand = _candidate(2.0, 2.0)
    score_before = model.score_candidate(cand)

    # Mutate excluded entry in raw archive
    raw_archive["entries"][1]["candidate"]["start"] = {"x": 0.0, "y": 0.0}
    fit_data_mutated = isolate_fit_entries(
        raw_archive, allowed_fit_ids={"sf_0"}, target_planner="social_force"
    )
    model_mutated = FailureArchiveProposalModel(fit_data_mutated, search_space)

    assert model_mutated.score_candidate(cand) == score_before


def test_opposite_sign_execution_favors_random_stops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When archive-nearness favors proposal but independent execution favors random, final rule STOPS."""
    from scripts.adversarial.run_proposal_vs_random_issue_2921 import main as script_main

    archive_path = tmp_path / "archive.json"
    search_space_path = tmp_path / "search_space.yaml"
    outcomes_path = tmp_path / "outcomes.json"
    output_json = tmp_path / "report.json"

    archive = _two_family_archive()
    archive_path.write_text(json.dumps(archive), encoding="utf-8")
    search_space_path.write_text(_SEARCH_SPACE_YAML, encoding="utf-8")

    # Independent planner execution outcome favors RANDOM (random=10.0, proposal=0.0)
    contract_path, outcomes_path = _write_frozen_contract_and_outcomes(
        tmp_path,
        archive_path,
        search_space_path,
        [0.0, 0.0, 0.0, 0.0],
        [10.0, 10.0, 10.0, 10.0],
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_proposal_vs_random_issue_2921.py",
            "--archive",
            archive_path.as_posix(),
            "--search-space",
            search_space_path.as_posix(),
            "--contract",
            contract_path.as_posix(),
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
    assert report["comparison"]["interpretation"] == "independent_planner_execution_outcomes"
    # Independent execution shows proposal mean = 0.0, random mean = 10.0 -> improvement = -10.0
    assert report["comparison"]["mean_objective_improvement"] == -10.0
    assert report["issue_2921_stop_rule"]["status"] == "stop"


def test_opposite_sign_execution_favors_proposal_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When archive-nearness favors random but independent execution favors proposal, final rule CONTINUES."""
    from scripts.adversarial.run_proposal_vs_random_issue_2921 import main as script_main

    archive_path = tmp_path / "archive.json"
    search_space_path = tmp_path / "search_space.yaml"
    output_json = tmp_path / "report.json"

    archive = _two_family_archive()
    archive_path.write_text(json.dumps(archive), encoding="utf-8")
    search_space_path.write_text(_SEARCH_SPACE_YAML, encoding="utf-8")

    # Independent planner execution outcome favors PROPOSAL (proposal=10.0, random=0.0)
    contract_path, outcomes_path = _write_frozen_contract_and_outcomes(
        tmp_path,
        archive_path,
        search_space_path,
        [10.0, 10.0, 10.0, 10.0],
        [0.0, 0.0, 0.0, 0.0],
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_proposal_vs_random_issue_2921.py",
            "--archive",
            archive_path.as_posix(),
            "--search-space",
            search_space_path.as_posix(),
            "--contract",
            contract_path.as_posix(),
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
    assert report["comparison"]["interpretation"] == "independent_planner_execution_outcomes"
    # Independent execution shows proposal mean = 10.0, random mean = 0.0 -> improvement = +10.0
    assert report["comparison"]["mean_objective_improvement"] == 10.0
    assert report["issue_2921_stop_rule"]["status"] == "continue"
