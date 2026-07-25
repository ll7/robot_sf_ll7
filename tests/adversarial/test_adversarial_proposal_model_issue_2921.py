"""Tests for adversarial proposal model, frozen #3275 contract, and comparison script."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from robot_sf.adversarial.config import CandidateSpec, Pose2D
from robot_sf.adversarial.proposal_model import (
    FailureArchiveProposalModel,
    derive_fit_payload_from_recertification,
    load_issue_3275_contract,
)
from robot_sf.adversarial.scenario_manifest import AdversarialScenarioManifest
from scripts.adversarial.run_proposal_vs_random_issue_2921 import (
    ISSUE_3275_DECISION_VOCABULARY,
    _rank_pool_ids_by_candidate_identity,
    classify_issue_2921_stop_rule,
    create_synthetic_archive,
    create_synthetic_search_space,
    run_check_contract,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONTRACT = _REPO_ROOT / "configs/adversarial/issue_3275_same_planner_contract.json"
_RECERT = (
    _REPO_ROOT
    / "docs/context/evidence/issue_5305_certified_archive/recertification_issue_6139.json"
)
_ARCHIVE = _REPO_ROOT / "docs/context/evidence/issue_5305_certified_archive/archive.json"


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


# --- Frozen contract: fit-only construction and negative regression ------------


def _load_contract_payload():
    """Load and derive the frozen fit-only payload from the real artifacts."""
    contract = load_issue_3275_contract(_CONTRACT)
    recert = json.loads(_RECERT.read_text("utf-8"))
    archive = json.loads(_ARCHIVE.read_text("utf-8"))
    payload = derive_fit_payload_from_recertification(
        recert,
        archive,
        fit_family=contract["fit"]["scenario_family"],
        fit_planner=contract["fit"]["target_planner"],
        excluded_family=contract["exclusions"]["scenario_family"],
        required_benchmark_eligibility=contract["fit"]["required_benchmark_eligibility"],
        expected_count=contract["fit"]["count"],
        expected_ids_sha256=contract["fit"]["entry_ids_sha256"],
        expected_non_eligible_count=contract["fit"]["excluded_from_nominal_fit_count"],
        expected_non_eligible_ids_sha256=contract["fit"][
            "excluded_from_nominal_fit_entry_ids_sha256"
        ],
    )
    return contract, payload, archive


def test_check_contract_validates_frozen_contract() -> None:
    """The side-effect-free --check-contract command validates the frozen contract."""
    exit_code, verdict = run_check_contract(_CONTRACT, repo_root=_REPO_ROOT)
    assert exit_code == 0
    assert verdict["ok"] is True
    assert verdict["checks"]["fit_count"] == 6
    assert verdict["checks"]["model_entry_count"] == 6
    assert verdict["checks"]["fit_entry_ids_sha256_matches_contract"] is True
    assert verdict["checks"]["fit_entry_ids_match_contract"] is True
    assert verdict["checks"]["excluded_from_nominal_fit_count"] == 6
    assert verdict["checks"]["excluded_from_nominal_fit_ids_sha256_matches_contract"] is True
    assert verdict["checks"]["negative_regression_full_archive_same_fit_entries"] is True
    assert verdict["checks"]["negative_regression_non_fit_dropped_count"] == 11
    assert verdict["checks"]["negative_regression_held_out_dropped_count"] == 5
    assert verdict["checks"]["negative_regression_non_eligible_fit_dropped_count"] == 6
    assert verdict["checks"]["negative_regression_dropped_ids_match_contract"] is True
    assert verdict["checks"]["no_held_out_family_in_model"] is True
    assert verdict["failures"] == []


def test_fit_only_model_uses_exactly_the_six_nominally_eligible_fit_ids() -> None:
    """The fit-only model excludes every corrected stress-only fit-family record."""
    contract, payload, _ = _load_contract_payload()
    model = FailureArchiveProposalModel(
        payload.archive_payload, fit_entry_ids=payload.entry_ids, feature_view="family_invariant"
    )
    assert model.state == "active"
    assert len(model.entries) == 6
    model_ids = {entry["archive_id"] for entry in model.entries}
    assert model_ids == set(payload.entry_ids)
    assert model_ids == set(contract["fit"]["entry_ids"])
    assert set(payload.non_eligible_fit_entry_ids) == set(
        contract["fit"]["excluded_from_nominal_fit_entry_ids"]
    )
    assert model_ids.isdisjoint(payload.non_eligible_fit_entry_ids)
    # Every fit entry is group_crossing/social_force; no cross_trap leakage.
    assert all("classic_group_crossing_medium" in aid for aid in model_ids)
    assert not any("classic_cross_trap_medium" in aid for aid in model_ids)


def test_frozen_contract_factory_preserves_nominal_fit_and_exclusion_lineage() -> None:
    """The public factory enforces the six-anchor frozen contract end to end."""
    model, provenance = FailureArchiveProposalModel.from_frozen_contract(
        _CONTRACT, repo_root=_REPO_ROOT
    )

    assert model.state == "active"
    assert len(model.entries) == 6
    assert all("robot" in entry for entry in model.entries)
    assert provenance["fit_count"] == 6
    assert provenance["non_eligible_fit_count"] == 6
    assert provenance["excluded_count"] == 5
    assert provenance["fit_only_initialized"] is True
    assert provenance["model_entry_count"] == 6
    assert provenance["planner_drift"] == {}


def test_negative_regression_excluded_records_cannot_change_scores_or_ranks() -> None:
    """Feeding the full archive (incl. 5 excluded records) must not change scores/ranks."""
    _contract, payload, archive = _load_contract_payload()
    fit_ids = list(payload.entry_ids)

    model_fit_only = FailureArchiveProposalModel(
        payload.archive_payload, fit_entry_ids=fit_ids, feature_view="absolute"
    )
    model_full_archive = FailureArchiveProposalModel(
        archive, fit_entry_ids=fit_ids, feature_view="absolute"
    )
    # The full archive drops six ineligible fit-family records plus five held-out records.
    assert len(model_full_archive.excluded_entry_ids) == 11
    assert {e["archive_id"] for e in model_fit_only.entries} == {
        e["archive_id"] for e in model_full_archive.entries
    }

    candidates = [_candidate(2.5, 3.0), _candidate(8.0, 2.0), _candidate(3.0, 3.0)]
    ranks_a = model_fit_only.rank_candidates(candidates)
    ranks_b = model_full_archive.rank_candidates(candidates)
    assert [c for c, _ in ranks_a] == [c for c, _ in ranks_b]
    assert [s for _, s in ranks_a] == [s for _, s in ranks_b]


def test_fit_only_model_fails_closed_when_a_fit_id_is_missing() -> None:
    """A fit ID absent from the archive fails closed rather than silently shrinking."""
    _contract, payload, _ = _load_contract_payload()
    tampered_ids = list(payload.entry_ids) + ["issue5305_missing_record"]
    model = FailureArchiveProposalModel(
        payload.archive_payload, fit_entry_ids=tampered_ids, feature_view="absolute"
    )
    assert model.state == "blocked"
    assert model.state_reason.startswith("fit_entry_ids_missing_from_archive:")


# --- Decision rule (continue | stop | inconclusive) ---------------------------


def _independent_eval(*, available: bool, status: str = "complete", reason: str = "ok") -> dict:
    """Build a minimal independent-evaluation result for the stop-rule classifier."""
    return {
        "independent_outcomes_available": available,
        "status": status,
        "reason": reason,
        "decision": {
            "status": "continue" if available else "inconclusive",
            "reason": "proposal_beats_random" if available else reason,
            "claim_boundary": "diagnostic_only",
        },
    }


def test_classify_issue_2921_stop_rule_inconclusive_without_outcomes() -> None:
    """No independent outcomes -> inconclusive (vocabulary is frozen)."""
    decision = classify_issue_2921_stop_rule(
        independent_evaluation=_independent_eval(available=False, reason="not_available")
    )
    assert decision["status"] == "inconclusive"
    assert decision["vocabulary"] == list(ISSUE_3275_DECISION_VOCABULARY)
    assert "revise" not in decision["vocabulary"]


def test_classify_issue_2921_stop_rule_follows_independent_decision() -> None:
    """When independent outcomes are valid, the stop rule mirrors their decision."""
    decision = classify_issue_2921_stop_rule(
        independent_evaluation=_independent_eval(available=True)
    )
    assert decision["status"] == "continue"
    assert decision["vocabulary"] == list(ISSUE_3275_DECISION_VOCABULARY)


# --- Legacy proposal-model behavior (unchanged API) ---------------------------


def test_proposal_model_initialization_and_blocked_state() -> None:
    """Missing/empty archives result in blocked state."""
    model = FailureArchiveProposalModel(None)
    assert model.state == "blocked"

    model_empty_dict = FailureArchiveProposalModel({})
    assert model_empty_dict.state == "blocked"
    assert model_empty_dict.state_reason == "malformed_archive_payload"

    entries_only = FailureArchiveProposalModel({"entries": create_synthetic_archive()["entries"]})
    assert entries_only.state == "blocked"
    assert entries_only.state_reason.startswith("invalid_failure_archive_schema:")

    archive_data = create_synthetic_archive()
    model_active = FailureArchiveProposalModel(archive_data)
    assert model_active.state == "active"
    assert model_active.state_reason == "archive_loaded"
    assert len(model_active.entries) == 2


def test_archive_path_loading_and_malformed_inputs(tmp_path: Path) -> None:
    """Path-based archive loading and malformed fail-closed states."""
    archive_path = tmp_path / "archive.json"
    archive_path.write_text(json.dumps(create_synthetic_archive()), encoding="utf-8")

    model_from_path = FailureArchiveProposalModel(archive_path)
    assert model_from_path.state == "active"
    assert len(model_from_path.entries) == 2

    assert FailureArchiveProposalModel(tmp_path / "missing.json").state == "blocked"

    empty_path = tmp_path / "empty.json"
    empty_path.write_text("", encoding="utf-8")
    assert FailureArchiveProposalModel(empty_path).state == "blocked"

    malformed_path = tmp_path / "malformed.json"
    malformed_path.write_text("{oops", encoding="utf-8")
    assert FailureArchiveProposalModel(malformed_path).state == "blocked"

    assert FailureArchiveProposalModel({"entries": []}).state == "blocked"
    assert FailureArchiveProposalModel({"entries": "not-a-list"}).state == "blocked"
    assert FailureArchiveProposalModel({"entries": ["not-a-dict"]}).state == "blocked"
    assert (
        FailureArchiveProposalModel({"entries": [{"candidate": "not-a-dict"}]}).state == "blocked"
    )


def test_tabular_view_and_scale_fallbacks() -> None:
    """Tabular feature extraction and scale fallback without search-space bounds."""
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
    """Candidates are ranked deterministically by archive proximity."""
    archive_data = create_synthetic_archive()
    search_space = create_synthetic_search_space()
    model = FailureArchiveProposalModel(archive_data, search_space)

    c_close = _candidate(2.1, 2.1)
    c_far = _candidate(9.0, 9.0)

    ranked = model.rank_candidates([c_far, c_close], strategy="nearest_neighbor")
    assert len(ranked) == 2
    assert ranked[0][0] == c_close
    assert ranked[1][0] == c_far
    assert ranked[0][1] > ranked[1][1]


def test_runner_converts_ranked_candidates_to_stable_pool_ids_before_arm_assignment() -> None:
    """The random arm removes the actual ranked proposal candidates, not object reprs."""
    from robot_sf.adversarial.disjoint_evaluation import assign_arms_disjoint_by_candidate

    model = FailureArchiveProposalModel(create_synthetic_archive(), create_synthetic_search_space())
    pool = [_candidate(9.0, 9.0), _candidate(2.1, 2.1), _candidate(3.0, 3.0)]
    pool_ids = ["pool_far", "pool_close", "pool_mid"]

    ranked_ids = _rank_pool_ids_by_candidate_identity(model, pool, pool_ids)
    arms = assign_arms_disjoint_by_candidate(ranked_ids, pool_ids, budget_per_arm=1, rng_seed=7)

    assert ranked_ids[0] == "pool_close"
    assert set(ranked_ids) == set(pool_ids)
    assert set(arms.proposal_ids).isdisjoint(arms.random_ids)
    assert arms.proposal_ids == ["pool_close"]


def test_score_strategies_and_empty_candidate_ranking() -> None:
    """Alternative strategy scoring, unknown-strategy fallback, and empty inputs."""
    archive_data = create_synthetic_archive()
    search_space = create_synthetic_search_space()
    model = FailureArchiveProposalModel(archive_data, search_space)

    c_close = _candidate(2.1, 2.1)
    c_far = _candidate(9.0, 9.0)

    assert model.score_candidate(c_close, strategy="objective_weighted") > (
        model.score_candidate(c_far, strategy="objective_weighted")
    )
    assert model.score_candidate(c_close, strategy="unknown_strategy") == (
        model.score_candidate(c_close, strategy="nearest_neighbor")
    )
    assert model.rank_candidates([]) == []

    blocked = FailureArchiveProposalModel(None)
    assert blocked.rank_candidates([c_close]) == [(c_close, 0.0)]
    assert blocked.score_candidate(c_close) == 0.0


def test_manifest_emission_and_no_benchmark_promotion() -> None:
    """Emitted manifests are valid and carry a diagnostic-only evidence boundary."""
    archive_data = create_synthetic_archive()
    search_space = create_synthetic_search_space()
    model = FailureArchiveProposalModel(archive_data, search_space)

    manifest = model.emit_manifest(_candidate(2.0, 2.0), generator_seed=123, candidate_index=5)
    assert isinstance(manifest, AdversarialScenarioManifest)
    assert manifest.generator is not None
    assert manifest.generator.family == "learned_proposal_model"
    assert manifest.generator.generator_id == "FailureArchiveProposalModel"
    assert manifest.generator.seed == 123
    assert manifest.generator.candidate_index == 5
    assert manifest.evidence_tier == "diagnostic-only"
    assert "diagnostic-only" in manifest.evidence_boundary


def test_certification_status_handling() -> None:
    """Candidate certification handles not_available/passed status."""
    archive_data = create_synthetic_archive()
    model = FailureArchiveProposalModel(archive_data)
    dummy_yaml = Path("dummy_scenario.yaml")
    assert model.certify_candidate(
        _candidate(2.0, 2.0), dummy_yaml, require_certification=False
    ).status in (
        "passed",
        "failed",
        "not_available",
    )
    assert model.certify_candidate(
        _candidate(2.0, 2.0), dummy_yaml, require_certification=True
    ).status in (
        "passed",
        "failed",
        "not_available",
    )


# --- Comparison report plumbing ------------------------------------------------


def test_comparison_report_plumbing_only_shape(tmp_path: Path, monkeypatch) -> None:
    """The default (no archive, no outcomes) report is plumbing-only and fail-closed."""
    from scripts.adversarial.run_proposal_vs_random_issue_2921 import main as script_main

    output_json = tmp_path / "comparison_report.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_proposal_vs_random_issue_2921.py",
            "--budget",
            "5",
            "--seed",
            "10",
            "--output",
            str(output_json),
        ],
    )
    assert script_main() == 0
    report = json.loads(output_json.read_text("utf-8"))
    assert report["schema_version"] == "adversarial_proposal_comparison.v1"
    assert report["state"] in ("diagnostic_only", "blocked")
    assert report["result_classification"] == "plumbing_validation_only"
    assert report["held_out_evidence"] is False
    assert report["benchmark_evidence"] is False
    assert report["planner_performance_claim"] is False
    assert report["decision_vocabulary"] == list(ISSUE_3275_DECISION_VOCABULARY)
    assert (
        report["comparison_interpretation"] == "plumbing_only_circular_archive_nearness_objective"
    )
    # Archive-nearness lives under a diagnostic-only namespace.
    assert report["diagnostic_archive_nearness"]["comparison"]["namespace"] == (
        "archive_nearness_diagnostic_only_cannot_drive_verdict"
    )
    assert report["issue_2921_stop_rule"]["status"] == "inconclusive"
    assert report["budget_per_arm"] == 5


def test_comparison_script_rejects_negative_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """Negative budgets fail during argument parsing before sampling."""
    from scripts.adversarial.run_proposal_vs_random_issue_2921 import parse_args

    monkeypatch.setattr("sys.argv", ["run_proposal_vs_random_issue_2921.py", "--budget", "-1"])
    with pytest.raises(SystemExit) as exc_info:
        parse_args()
    assert exc_info.value.code == 2


def test_real_archive_without_search_space_fails_closed(tmp_path: Path, monkeypatch) -> None:
    """Real archive runs stay fail-closed without a real search space."""
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
    report = json.loads(output_json.read_text("utf-8"))
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


def _v2_outcome_packet(
    *, eval_archive_sha256: str, proposal_failures: int, random_failures: int, per_arm: int = 4
) -> dict:
    """Build a minimal v2 row-level outcome packet bound to the eval split hash."""
    rows: list[dict[str, Any]] = []
    for rank in range(per_arm):
        rows.append(
            _v2_row(f"prop_{rank}", f"prop_cand_{rank}", "proposal", rank, rank < proposal_failures)
        )
    for rank in range(per_arm):
        rows.append(
            _v2_row(f"rand_{rank}", f"rand_cand_{rank}", "random", rank, rank < random_failures)
        )
    return {
        "schema_version": "adversarial_independent_outcomes.v2",
        "source": "unit-test-fixture",
        "artifact": "docs/context/evidence/unit-test.json",
        "outcome_source": "planner_execution",
        "objective": "certified_failure_outcome",
        "target_planner_id": "social_force",
        "target_planner_config_sha256": "dfdebd497e19a046e41cb2b1e7d7a7f54cd592ac0a465e4149efff19efa16735",
        "eval_archive_sha256": eval_archive_sha256,
        "rows": rows,
    }


def _manifest_hash_binding(packet: dict[str, Any]) -> dict[str, Any]:
    """Build the separately supplied frozen arm-manifest binding fixture."""
    rows = packet["rows"]
    return {
        "schema_version": "adversarial_candidate_manifest_bindings.v2",
        "candidate_manifest_sha256_by_id": {
            row["candidate_manifest_id"]: row["candidate_manifest_sha256"] for row in rows
        },
        "candidate_manifest_ids_by_arm": {
            arm: [row["candidate_manifest_id"] for row in rows if row["selection_arm"] == arm]
            for arm in ("proposal", "random")
        },
        "execution_seeds_by_manifest_id": {
            row["candidate_manifest_id"]: [row["execution_seed"]] for row in rows
        },
        "candidate_pool_seed": 7,
    }


def _v2_row(row_id: str, manifest_id: str, arm: str, rank: int, failure: bool) -> dict[str, Any]:
    """Build one admissible v2 outcome row."""
    return {
        "row_id": row_id,
        "candidate_manifest_id": manifest_id,
        "candidate_manifest_sha256": f"manifest-{manifest_id}",
        "selection_arm": arm,
        "selection_rank": rank + 1,
        "candidate_pool_seed": 7,
        "candidate_pool_index": rank,
        "target_planner_id": "social_force",
        "target_planner_config_sha256": "dfdebd497e19a046e41cb2b1e7d7a7f54cd592ac0a465e4149efff19efa16735",
        "scenario_family": "classic_cross_trap_medium",
        "scenario_seed": 99001 + rank,
        "execution_seed": 70001 + rank,
        "execution_commit": "ecf997d392a4f2c1a4fb5a56e8101acb030b7e2f",
        "execution_command": ["python", "-m", "robot_sf.run_eval"],
        "execution_config_lineage": {"config": "eval.yaml"},
        "execution_mode": "native",
        "termination_reason": "collision" if failure else "goal_reached",
        "independent_failure_outcome": failure,
        "scenario_certification_status": "passed",
        "candidate_certification_status": "passed",
        "replay_lineage": {
            "exact_signature_match": True,
            "original_signature_sha256": "abc",
            "replay_signature_sha256": "abc",
        },
        "confirmation_lineage": {
            "confirmed_count": 3,
            "attempt_count": 5,
            "stable_attribution": True,
        },
        "record_sha256": f"rec-{manifest_id}",
        "admission_status": "admitted",
        "exclusion_reason": None,
    }


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
    report = json.loads(output_json.read_text("utf-8"))
    assert report["state"] == "active"
    assert report["synthetic_archive"] is False
    provenance = report["archive_evaluation_provenance"]
    assert provenance["split_policy"] == "disjoint_scenario_family"
    assert provenance["disjointness_checks_passed"] is True
    # No independent outcomes -> fail-closed held-out gate and inconclusive decision.
    assert report["held_out_evidence"] is False
    assert provenance["held_out_evidence_status"] == (
        "not_available_requires_independent_planner_outcomes"
    )
    assert report["issue_2921_stop_rule"]["status"] == "inconclusive"


def test_normal_contract_runner_uses_frozen_fit_factory_and_held_out_split(
    tmp_path: Path, monkeypatch
) -> None:
    """The normal --contract path cannot use the generic archive or random split."""
    from scripts.adversarial.run_proposal_vs_random_issue_2921 import main as script_main

    search_space_path = tmp_path / "search_space.yaml"
    output_json = tmp_path / "contract_report.json"
    search_space_path.write_text(_SEARCH_SPACE_YAML, encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_proposal_vs_random_issue_2921.py",
            "--contract",
            _CONTRACT.as_posix(),
            "--search-space",
            search_space_path.as_posix(),
            "--seed",
            "42",
            "--output",
            output_json.as_posix(),
        ],
    )

    assert script_main() == 0

    report = json.loads(output_json.read_text("utf-8"))
    assert report["state"] == "active"
    assert report["synthetic_archive"] is False
    assert report["budget_per_arm"] == 12
    assert len(report["arm_manifest_ids_by_arm"]["proposal"]) == 12
    assert len(report["arm_manifest_ids_by_arm"]["random"]) == 12
    assert set(report["arm_manifest_ids_by_arm"]["proposal"]).isdisjoint(
        report["arm_manifest_ids_by_arm"]["random"]
    )
    provenance = report["archive_evaluation_provenance"]
    assert provenance["split_policy"] == "frozen_same_planner_held_out_family"
    assert provenance["fit_size"] == 6
    assert provenance["eval_size"] == 5
    frozen_contract = provenance["frozen_contract"]
    assert frozen_contract["fit_entry_count"] == 6
    assert (
        frozen_contract["fit_entry_ids_sha256"]
        == load_issue_3275_contract(_CONTRACT)["fit"]["entry_ids_sha256"]
    )
    assert frozen_contract["model"]["model_entry_count"] == 6
    assert frozen_contract["model"]["feature_view"] == "family_invariant"
    assert frozen_contract["model"]["evaluation_robot_start"] == [3.0, 3.0]
    assert frozen_contract["model"]["evaluation_robot_goal"] == [17.0, 17.0]
    assert report["issue_2921_stop_rule"]["status"] == "inconclusive"


def test_contract_runner_rejects_a_budget_outside_the_frozen_contract(
    tmp_path: Path, monkeypatch
) -> None:
    """A larger packet budget cannot be requested through the --contract path."""
    from scripts.adversarial.run_proposal_vs_random_issue_2921 import main as script_main

    search_space_path = tmp_path / "search_space.yaml"
    search_space_path.write_text(_SEARCH_SPACE_YAML, encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_proposal_vs_random_issue_2921.py",
            "--contract",
            _CONTRACT.as_posix(),
            "--search-space",
            search_space_path.as_posix(),
            "--budget",
            "30",
        ],
    )

    assert script_main() == 2


def test_real_archive_with_independent_outcomes_follows_execution(
    tmp_path: Path, monkeypatch
) -> None:
    """Valid v2 outcomes make the comparison follow execution; archive-nearness is diagnostic."""
    from robot_sf.adversarial.disjoint_evaluation import archive_sha256, disjoint_family_split
    from scripts.adversarial.run_proposal_vs_random_issue_2921 import main as script_main

    archive_path = tmp_path / "archive.json"
    search_space_path = tmp_path / "search_space.yaml"
    outcomes_path = tmp_path / "outcomes.json"
    manifest_hashes_path = tmp_path / "manifest_hashes.json"
    output_json = tmp_path / "report.json"
    archive = _two_family_archive()
    archive_path.write_text(json.dumps(archive), encoding="utf-8")
    search_space_path.write_text(_SEARCH_SPACE_YAML, encoding="utf-8")
    split = disjoint_family_split(archive["entries"], eval_fraction=0.5, seed=7)
    # Proposal arm: 4/4 certified failures; random arm: 0/4 (execution favors proposal).
    packet = _v2_outcome_packet(
        eval_archive_sha256=archive_sha256(split.eval_entries),
        proposal_failures=4,
        random_failures=0,
        per_arm=4,
    )
    outcomes_path.write_text(json.dumps(packet), encoding="utf-8")
    manifest_hashes_path.write_text(json.dumps(_manifest_hash_binding(packet)), encoding="utf-8")
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
            "--expected-candidate-manifest-hashes",
            manifest_hashes_path.as_posix(),
            "--budget",
            "4",
            "--seed",
            "7",
            "--output",
            output_json.as_posix(),
        ],
    )
    assert script_main() == 0
    report = json.loads(output_json.read_text("utf-8"))
    assert report["comparison_interpretation"] == "independent_planner_execution_outcomes"
    assert report["independent_outcome_evaluation"]["independent_outcomes_available"] is True
    assert report["independent_outcome_evaluation"]["proposal_failure_yield"] == 1.0
    assert report["independent_outcome_evaluation"]["random_failure_yield"] == 0.0
    manifest_binding = report["independent_outcome_evaluation"]["candidate_manifest_binding"]
    assert manifest_binding["required"] is True
    assert manifest_binding["available"] is True
    assert manifest_binding["exact_arm_membership_required"] is True
    assert manifest_binding["execution_seed_lineage_required"] is True
    assert manifest_binding["reason"] == "ok"
    # k=4 is underpowered for delta=0.20 -> inconclusive (honest, not continue).
    assert report["issue_2921_stop_rule"]["status"] == "inconclusive"
    assert report["issue_2921_stop_rule"]["vocabulary"] == list(ISSUE_3275_DECISION_VOCABULARY)
    # Archive-nearness is still reported, but only as a diagnostic namespace.
    assert report["diagnostic_archive_nearness"]["comparison"]["namespace"] == (
        "archive_nearness_diagnostic_only_cannot_drive_verdict"
    )


def test_underpowered_execution_favors_random_is_inconclusive(tmp_path: Path, monkeypatch) -> None:
    """Underpowered random-favoring execution cannot stop the study."""
    from robot_sf.adversarial.disjoint_evaluation import archive_sha256, disjoint_family_split
    from scripts.adversarial.run_proposal_vs_random_issue_2921 import main as script_main

    archive_path = tmp_path / "archive.json"
    search_space_path = tmp_path / "search_space.yaml"
    outcomes_path = tmp_path / "outcomes.json"
    manifest_hashes_path = tmp_path / "manifest_hashes.json"
    output_json = tmp_path / "report.json"
    archive = _two_family_archive()
    archive_path.write_text(json.dumps(archive), encoding="utf-8")
    search_space_path.write_text(_SEARCH_SPACE_YAML, encoding="utf-8")
    split = disjoint_family_split(archive["entries"], eval_fraction=0.5, seed=7)
    # Execution favors random, but k=4 is underpowered for the frozen 0.20 effect.
    packet = _v2_outcome_packet(
        eval_archive_sha256=archive_sha256(split.eval_entries),
        proposal_failures=0,
        random_failures=4,
        per_arm=4,
    )
    outcomes_path.write_text(json.dumps(packet), encoding="utf-8")
    manifest_hashes_path.write_text(json.dumps(_manifest_hash_binding(packet)), encoding="utf-8")
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
            "--expected-candidate-manifest-hashes",
            manifest_hashes_path.as_posix(),
            "--budget",
            "4",
            "--seed",
            "7",
            "--output",
            output_json.as_posix(),
        ],
    )
    assert script_main() == 0
    report = json.loads(output_json.read_text("utf-8"))
    assert report["independent_outcome_evaluation"]["comparison"]["yield_improvement"] < 0.0
    assert report["issue_2921_stop_rule"]["status"] == "inconclusive"
    assert report["issue_2921_stop_rule"]["reason"] == "underpowered_for_minimally_important_effect"


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
                "schema_version": "adversarial_independent_outcomes.v2",
                "outcome_source": "planner_execution",
                "objective": "archive_nearness",
                "target_planner_id": "social_force",
                "rows": [],
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
    report = json.loads(output_json.read_text("utf-8"))
    assert report["held_out_evidence"] is False
    assert report["independent_outcome_evaluation"]["status"] == "blocked"
    assert "circular" in report["independent_outcome_evaluation"]["reason"]
    assert report["issue_2921_stop_rule"]["status"] == "inconclusive"
