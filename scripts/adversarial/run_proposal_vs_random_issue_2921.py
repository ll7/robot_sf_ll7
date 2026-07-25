#!/usr/bin/env python3
"""Run proposal model vs random candidate sampler under identical budget.

Side-effect-free contract check (issue #6103 / parent #3275):

    uv run python scripts/adversarial/run_proposal_vs_random_issue_2921.py \
        --check-contract configs/adversarial/issue_3275_same_planner_contract.json

The ``--check-contract`` command validates the frozen same-planner contract: it
derives the fit-only payload from the corrected recertification artifact,
asserts the frozen fit count/hash, planner, family, and exclusions, constructs
the fit-only FailureArchiveProposalModel, and verifies that excluded and
held-out-family records cannot influence scores or ranks. It executes no planner
and produces no new empirical outcome; its exact invocation is a required input
to the next sub-issue.

The comparison run keeps archive-nearness under an explicitly diagnostic
namespace. When valid independent native planner-execution outcomes (the frozen
``adversarial_independent_outcomes.v2`` row contract) are supplied, the
top-level comparison and the issue #2921 stop rule follow those outcomes
exclusively, and the decision vocabulary is exactly ``continue | stop |
inconclusive``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

CLAIM_BOUNDARY = (
    "plumbing_validation_only: proposal-vs-random deltas in this report exercise ranking and "
    "report plumbing only. The current objective is archive-nearness, so it is circular with "
    "archive-nearness ranking and is not held-out yield, benchmark evidence, planner-performance "
    "evidence, or evidence that learned proposals improve failure discovery."
)

HELD_OUT_DIAGNOSTIC_BOUNDARY = (
    "held_out_diagnostic_only: proposal-vs-random deltas use externally supplied independent "
    "planner-execution outcomes plus candidate certification and null-test checks. This is "
    "diagnostic evidence for issue #3275 only; it is not benchmark evidence, paper evidence, or "
    "a planner-performance claim without the durable execution artifacts named by the outcome "
    "payload."
)

#: Frozen decision vocabulary for the #3275 contract (no revise, no generic blocked).
ISSUE_3275_DECISION_VOCABULARY = ("continue", "stop", "inconclusive")


def classify_issue_2921_stop_rule(
    *,
    independent_evaluation: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return the issue #2921 ``continue | stop | inconclusive`` decision.

    The decision follows independent native planner-execution outcomes only. When
    no valid independent outcome evaluation is available, the decision is
    ``inconclusive``; it is never ``continue`` or ``stop`` on archive-nearness.
    The vocabulary is exactly :data:`ISSUE_3275_DECISION_VOCABULARY` (no
    ``revise``, no generic ``blocked``).
    """
    if not independent_evaluation or not independent_evaluation.get(
        "independent_outcomes_available"
    ):
        reason = independent_evaluation.get("reason") if independent_evaluation else "not_available"
        return {
            "status": "inconclusive",
            "reason": f"independent_outcomes_unavailable_or_fail_closed:{reason}",
            "vocabulary": list(ISSUE_3275_DECISION_VOCABULARY),
            "evidence_tier": "analysis_only",
            "claim_boundary": (
                "no continue/stop decision without independent planner-execution outcomes"
            ),
        }
    decision = independent_evaluation["decision"]
    return {
        "status": decision["status"],
        "reason": decision["reason"],
        "vocabulary": list(ISSUE_3275_DECISION_VOCABULARY),
        "evidence_tier": "diagnostic_only",
        "claim_boundary": decision["claim_boundary"],
    }


def create_synthetic_search_space() -> Any:
    """Create a default synthetic search space config for diagnostics."""
    from robot_sf.adversarial.config import RangeConfig, SearchSpaceConfig

    return SearchSpaceConfig(
        start_x=RangeConfig(min=0.0, max=10.0),
        start_y=RangeConfig(min=0.0, max=10.0),
        goal_x=RangeConfig(min=0.0, max=10.0),
        goal_y=RangeConfig(min=0.0, max=10.0),
        spawn_time_s=RangeConfig(min=0.0, max=5.0),
        pedestrian_speed_mps=RangeConfig(min=0.5, max=2.0),
        pedestrian_delay_s=RangeConfig(min=0.0, max=3.0),
        scenario_seed=RangeConfig(min=1.0, max=100.0),
    )


def create_synthetic_archive() -> dict[str, Any]:
    """Create a small synthetic archive of failures for testing/diagnostics."""
    return {
        "schema_version": "adversarial_failure_archive.v1",
        "entries": [
            {
                "archive_id": "failure_0000",
                "candidate": {
                    "start": {"x": 2.0, "y": 2.0},
                    "goal": {"x": 8.0, "y": 8.0},
                    "spawn_time_s": 1.0,
                    "pedestrian_speed_mps": 1.2,
                    "pedestrian_delay_s": 0.5,
                    "scenario_seed": 42,
                },
                "failure_attribution": {
                    "primary_failure": "collision",
                    "details": {"termination_reason": "collision"},
                },
                "objective_value": 8.5,
                "normalized_perturbation": 0.1,
            },
            {
                "archive_id": "failure_0001",
                "candidate": {
                    "start": {"x": 3.0, "y": 3.0},
                    "goal": {"x": 7.0, "y": 7.0},
                    "spawn_time_s": 2.0,
                    "pedestrian_speed_mps": 1.5,
                    "pedestrian_delay_s": 1.0,
                    "scenario_seed": 43,
                },
                "failure_attribution": {
                    "primary_failure": "timeout",
                    "details": {"termination_reason": "timeout"},
                },
                "objective_value": 6.0,
                "normalized_perturbation": 0.25,
            },
        ],
    }


def _payload_sha256(payload: dict[str, Any]) -> str:
    """Return a deterministic digest for JSON-like report provenance."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_search_space(path: Path | None) -> tuple[str, str, Any, bool]:
    """Load SearchSpaceConfig, returning state and synthetic fallback provenance."""
    from robot_sf.adversarial.config import SearchSpaceConfig

    if path is None:
        return (
            "diagnostic_only",
            "No search space path provided; using synthetic search-space fixture.",
            create_synthetic_search_space(),
            True,
        )
    if path.exists():
        try:
            return (
                "active",
                "Search space loaded successfully.",
                SearchSpaceConfig.from_file(path),
                False,
            )
        except (ValueError, TypeError, OSError) as exc:
            return (
                "blocked",
                f"Failed to load search space: {exc}; using synthetic fixture for plumbing only.",
                create_synthetic_search_space(),
                True,
            )
    return (
        "blocked",
        f"Search space path {path} does not exist; using synthetic fixture for plumbing only.",
        create_synthetic_search_space(),
        True,
    )


def load_archive(path: Path | None) -> tuple[str, str, dict[str, Any], bool]:
    """Load archive data or fallback to synthetic, returning (state, reason, archive_data, is_synthetic)."""
    if path is None:
        return (
            "diagnostic_only",
            "No archive path provided; using synthetic archive fixture.",
            create_synthetic_archive(),
            True,
        )
    if not path.exists():
        return (
            "diagnostic_only",
            f"Archive path {path} does not exist; using synthetic archive fixture.",
            create_synthetic_archive(),
            True,
        )
    if path.stat().st_size == 0:
        return (
            "blocked",
            f"Archive file {path} is empty; using synthetic archive fixture.",
            create_synthetic_archive(),
            True,
        )

    try:
        with open(path, encoding="utf-8") as f:
            archive_data = json.load(f)
        if not isinstance(archive_data, dict) or not archive_data.get("entries"):
            return (
                "blocked",
                "Loaded archive contains no entries or is malformed; using synthetic.",
                create_synthetic_archive(),
                True,
            )
        return "active", "Real archive loaded successfully.", archive_data, False
    except (ValueError, TypeError, json.JSONDecodeError, OSError) as exc:
        return (
            "blocked",
            f"Failed to load archive: {exc}; using synthetic.",
            create_synthetic_archive(),
            True,
        )


def compute_metrics(selection: list[Any], evaluate_fn: Any) -> dict[str, Any]:
    """Compute summary metrics for a candidate selection (archive-nearness diagnostic)."""
    objs = [evaluate_fn(c) for c in selection]
    return {
        "mean_objective": round(sum(objs) / len(objs), 4) if objs else 0.0,
        "max_objective": round(max(objs), 4) if objs else 0.0,
        "failure_count": sum(1 for o in objs if o >= 8.0),
    }


def build_archive_evaluation_provenance(
    archive_data: dict[str, Any],
    *,
    state: str,
    synthetic_archive: bool,
    split_seed: int,
    independent_evaluation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build archive-evaluation provenance for the comparison report.

    Archive-nearness is recorded here as a diagnostic-only namespace. It can
    never drive the held-out verdict: that requires independent planner-execution
    outcomes, disjointness, certification, and a rejected null.
    """
    independent_evaluation = independent_evaluation or {}
    provenance: dict[str, Any] = {
        "archive_nearness_namespace": "diagnostic_only_cannot_drive_verdict",
        "archive_sha256": _payload_sha256(archive_data),
        "evaluation_outcome_sha256": independent_evaluation.get("payload_sha256"),
        "split_policy": "none_plumbing_fixture",
        "scenario_family_overlap": "not_checked",
        "seed_overlap": "not_checked",
        "archive_id_overlap": "not_checked",
        "disjointness_checks_passed": False,
        "required_for_held_out_claim": True,
    }
    if state != "active" or synthetic_archive:
        provenance["held_out_evidence_status"] = "not_available_plumbing_fixture"
        return provenance

    from robot_sf.adversarial.disjoint_evaluation import (
        archive_sha256,
        classify_held_out_evidence,
        compute_overlap_provenance,
        disjoint_family_split,
    )

    entries = archive_data.get("entries", [])
    split = disjoint_family_split(entries, eval_fraction=0.5, seed=split_seed)
    overlap = compute_overlap_provenance(split.fit_entries, split.eval_entries)
    provenance.update(overlap)
    provenance["fit_archive_sha256"] = archive_sha256(split.fit_entries)
    provenance["eval_archive_sha256"] = archive_sha256(split.eval_entries)
    provenance["independent_outcome_evaluation"] = independent_evaluation.get(
        "status", "not_available_requires_planner_execution"
    )
    provenance["held_out_evidence_status"] = classify_held_out_evidence(
        disjointness_checks_passed=overlap["disjointness_checks_passed"],
        independent_outcomes_available=bool(
            independent_evaluation.get("independent_outcomes_available")
        ),
        certification_available=bool(independent_evaluation.get("certification_available")),
        null_tests_reject_null=bool(independent_evaluation.get("null_tests_reject_null")),
    )
    return provenance


def _negative_regression_checks(
    payload: Any, archive: dict[str, Any], excl_cfg: dict[str, Any]
) -> tuple[list[str], dict[str, Any]]:
    """Verify the full archive (incl. excluded records) yields the same fit entries."""
    from robot_sf.adversarial.proposal_model import FailureArchiveProposalModel

    failures: list[str] = []
    full_model = FailureArchiveProposalModel(
        archive, fit_entry_ids=payload.entry_ids, feature_view="absolute"
    )
    full_entry_ids = [entry.get("archive_id") for entry in full_model.entries]
    same = sorted(full_entry_ids) == sorted(payload.entry_ids)
    checks = {
        "negative_regression_full_archive_same_fit_entries": same,
        "negative_regression_excluded_dropped_count": len(full_model.excluded_entry_ids),
    }
    if not same:
        failures.append("negative regression failed: full archive changed fit entries")
    if len(full_model.excluded_entry_ids) != excl_cfg["count"]:
        failures.append(
            "negative regression failed: excluded drop count "
            f"{len(full_model.excluded_entry_ids)} != {excl_cfg['count']}"
        )
    return failures, checks


def _check_fit_only_model(
    payload: Any,
    archive: dict[str, Any],
    *,
    fit_cfg: dict[str, Any],
    excl_cfg: dict[str, Any],
    planner_cfg: dict[str, Any],
    drift: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    """Construct the fit-only model, run the negative regression, and collect checks."""
    from robot_sf.adversarial.proposal_model import FailureArchiveProposalModel

    failures: list[str] = []
    checks: dict[str, Any] = {
        "fit_count": payload.count,
        "fit_entry_ids_sha256": payload.entry_ids_sha256,
        "fit_entry_ids_sha256_matches_contract": (
            payload.entry_ids_sha256 == fit_cfg["entry_ids_sha256"]
        ),
        "excluded_count": len(payload.excluded_entry_ids),
        "planner_family_drift": drift,
    }
    if payload.count != fit_cfg["count"]:
        failures.append(f"fit_count drift: {payload.count} != {fit_cfg['count']}")
    if payload.entry_ids_sha256 != fit_cfg["entry_ids_sha256"]:
        failures.append("fit_entry_ids_sha256 does not match contract")
    if len(payload.excluded_entry_ids) != excl_cfg["count"]:
        failures.append(
            f"excluded count drift: {len(payload.excluded_entry_ids)} != {excl_cfg['count']}"
        )
    if drift:
        failures.append(f"planner/family drift: {drift}")

    model = FailureArchiveProposalModel(
        payload.archive_payload,
        fit_entry_ids=payload.entry_ids,
        feature_view="family_invariant",
    )
    fit_ids = set(payload.entry_ids)
    model_entry_ids = {entry.get("archive_id") for entry in model.entries}
    excluded_ids = set(payload.excluded_entry_ids)
    checks.update(
        model_state=model.state,
        model_entry_count=len(model.entries),
        model_entry_ids_match_fit=model_entry_ids == fit_ids,
        no_excluded_record_in_model=model_entry_ids.isdisjoint(excluded_ids),
        no_held_out_family_in_model=all(
            "classic_cross_trap_medium" not in str(aid) for aid in model_entry_ids
        ),
        all_fit_entries_are_group_crossing=all(
            "classic_group_crossing_medium" in str(aid) for aid in model_entry_ids
        ),
    )
    if model.state != "active":
        failures.append(f"model not active: state={model.state} reason={model.state_reason}")
    if model_entry_ids != fit_ids:
        failures.append("model entries do not equal the frozen fit IDs")
    if not model_entry_ids.isdisjoint(excluded_ids):
        failures.append("an excluded record entered the fit-only model")
    if not checks["no_held_out_family_in_model"]:
        failures.append("a held-out family record entered the fit-only model")

    neg_failures, neg_checks = _negative_regression_checks(payload, archive, excl_cfg)
    checks.update(neg_checks)
    failures.extend(neg_failures)
    return failures, checks


def run_check_contract(contract_path: Path, *, repo_root: Path | None = None) -> tuple[int, dict]:
    """Side-effect-free validation of the frozen #3275 contract.

    Derives the fit-only payload from the corrected recertification artifact,
    asserts the frozen fit count/hash/planner/family/exclusions, constructs the
    fit-only model, and verifies that excluded and held-out-family records cannot
    influence scores or ranks. Executes no planner and writes nothing.
    """
    from robot_sf.adversarial.proposal_model import (
        attach_robot_geometry,
        derive_fit_payload_from_recertification,
        load_issue_3275_contract,
        validate_fit_payload_integrity,
    )

    contract = load_issue_3275_contract(contract_path)
    root = repo_root if repo_root is not None else Path.cwd()
    source = contract["source_lineage"]
    recert = json.loads((root / source["corrected_recertification_path"]).read_text("utf-8"))
    archive = json.loads((root / source["pre_correction_archive_path"]).read_text("utf-8"))
    checks: dict[str, Any] = {
        "contract_schema_version": contract["schema_version"],
        "contract_path": str(contract_path),
        "recertification_sha256_expected": source["corrected_recertification_sha256"],
        "recertification_sha256_observed": recert.get("recertification_sha256"),
        "recertification_all_unchanged": (
            recert.get("counts", {}).get("before_after_status", {}).get("unchanged")
            == recert.get("counts", {}).get("record_count")
        ),
    }
    failures: list[str] = []
    if recert.get("recertification_sha256") != source["corrected_recertification_sha256"]:
        failures.append(
            "recertification_sha256_mismatch: "
            f"observed={recert.get('recertification_sha256')} "
            f"expected={source['corrected_recertification_sha256']}"
        )

    fit_cfg = contract["fit"]
    excl_cfg = contract["exclusions"]
    planner_cfg = contract["target_planner"]
    try:
        payload = derive_fit_payload_from_recertification(
            recert,
            archive,
            fit_family=fit_cfg["scenario_family"],
            fit_planner=fit_cfg["target_planner"],
            excluded_family=excl_cfg["scenario_family"],
            expected_count=fit_cfg["count"],
            expected_ids_sha256=fit_cfg["entry_ids_sha256"],
        )
        attach_robot_geometry(payload, recert)
        drift = validate_fit_payload_integrity(
            payload,
            expected_planner=planner_cfg["id"],
            expected_planner_config_sha256=planner_cfg["config_sha256"],
        )
    except ValueError as exc:
        checks["error"] = str(exc)
        return 1, {
            "ok": False,
            "checks": checks,
            "failures": [str(exc)],
            "claim_boundary": contract["claim_boundary"]["label"],
        }

    model_failures, model_checks = _check_fit_only_model(
        payload, archive, fit_cfg=fit_cfg, excl_cfg=excl_cfg, planner_cfg=planner_cfg, drift=drift
    )
    checks.update(model_checks)
    failures.extend(model_failures)
    return (0 if not failures else 1), {
        "ok": not failures,
        "checks": checks,
        "failures": failures,
        "claim_boundary": contract["claim_boundary"]["label"],
        "next_subissue_required_input": (
            "uv run python scripts/adversarial/run_proposal_vs_random_issue_2921.py "
            f"--check-contract {contract_path}"
        ),
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Compare proposal model vs random sampler under identical budget."
    )
    parser.add_argument(
        "--check-contract",
        type=Path,
        default=None,
        help="Side-effect-free validation of the frozen #3275 contract config.",
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=None,
        help="Optional frozen #3275 contract config supplying planner/family/minimally-important.",
    )
    parser.add_argument(
        "--archive", type=Path, default=None, help="Path to failure archive JSON file."
    )
    parser.add_argument(
        "--search-space", type=Path, default=None, help="Path to search space config YAML file."
    )
    parser.add_argument(
        "--budget", type=int, default=12, help="Candidate budget per arm (identical)."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for candidate pool.")
    parser.add_argument("--output", type=Path, default=None, help="Path to write the JSON report.")
    parser.add_argument(
        "--evaluation-outcomes",
        type=Path,
        default=None,
        help="Optional v2 row-level independent planner-execution outcome packet.",
    )
    parser.add_argument(
        "--null-test-permutations", type=int, default=1000, help="Permutations for diagnostic null."
    )
    parser.add_argument(
        "--minimally-important",
        type=float,
        default=0.20,
        help="Frozen minimally important absolute yield improvement.",
    )
    args = parser.parse_args()
    if args.budget < 0:
        parser.error("--budget must be >= 0")
    if args.null_test_permutations < 1:
        parser.error("--null-test-permutations must be >= 1")
    return args


def _contract_frozen_params(args: argparse.Namespace) -> dict[str, Any]:
    """Read optional frozen planner/family/minimally-important from a contract."""
    if args.contract is None:
        return {
            "expected_target_planner_id": "social_force",
            "expected_target_planner_config_sha256": None,
            "expected_eval_family": "classic_cross_trap_medium",
            "minimally_important": args.minimally_important,
            "confirmation_threshold": "3_of_5",
        }
    from robot_sf.adversarial.proposal_model import load_issue_3275_contract

    contract = load_issue_3275_contract(args.contract)
    return {
        "expected_target_planner_id": contract["target_planner"]["id"],
        "expected_target_planner_config_sha256": contract["target_planner"]["config_sha256"],
        "expected_eval_family": contract["evaluation"]["scenario_family"],
        "minimally_important": contract["power_sensitivity"][
            "minimally_important_absolute_yield_improvement"
        ],
        "confirmation_threshold": (
            "3_of_5" if not contract["failure_admission"]["four_of_five_required"] else "4_of_5"
        ),
    }


def _resolve_run_state(
    *,
    archive_state: str,
    archive_reason: str,
    search_space_state: str,
    search_space_reason: str,
    outcome_state: str,
    outcome_reason: str,
    synthetic_archive: bool,
    synthetic_search_space: bool,
) -> tuple[str, str]:
    """Compute the run-level state and human-readable reason."""
    state = archive_state
    reason_parts = [archive_reason, search_space_reason, outcome_reason]
    if not synthetic_archive and synthetic_search_space:
        state = "blocked"
        reason_parts.append("Real-archive runs require a real search-space config.")
    if search_space_state == "blocked" and state == "active":
        state = "blocked"
    if outcome_state == "blocked":
        state = "blocked"
    return state, " ".join(reason_parts)


def _diagnostic_archive_nearness(
    model: Any,
    proposal_selection: list[Any],
    random_selection: list[Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Compute diagnostic-only archive-nearness metrics (cannot drive the verdict)."""

    def evaluate_objective(candidate: Any) -> float:
        if not model.entries:
            return 0.0
        distances = [model._entry_distance(candidate, entry) for entry in model.entries]
        min_dist = min(distances) if distances else 999.0
        return max(0.0, 10.0 - min_dist)

    random_metrics = compute_metrics(random_selection, evaluate_objective)
    proposal_metrics = compute_metrics(proposal_selection, evaluate_objective)
    comparison = {
        "namespace": "archive_nearness_diagnostic_only_cannot_drive_verdict",
        "mean_objective_improvement": round(
            proposal_metrics["mean_objective"] - random_metrics["mean_objective"], 4
        ),
        "failure_count_improvement": (
            proposal_metrics["failure_count"] - random_metrics["failure_count"]
        ),
    }
    return random_metrics, proposal_metrics, comparison


def _assemble_report(  # noqa: PLR0913
    *,
    state: str,
    reason: str,
    synthetic_archive: bool,
    synthetic_search_space: bool,
    search_space_state: str,
    args: argparse.Namespace,
    arms: Any,
    diagnostic_random_metrics: dict[str, Any],
    diagnostic_proposal_metrics: dict[str, Any],
    diagnostic_comparison: dict[str, Any],
    independent_evaluation: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the comparison report from its computed pieces."""
    independent_available = bool(independent_evaluation.get("independent_outcomes_available"))
    if independent_available:
        comparison = independent_evaluation["comparison"]
        comparison_interpretation = "independent_planner_execution_outcomes"
    elif independent_evaluation.get("status", "").startswith("blocked"):
        comparison = diagnostic_comparison
        comparison_interpretation = "independent_outcomes_rejected_by_held_out_gate"
    else:
        comparison = diagnostic_comparison
        comparison_interpretation = "plumbing_only_circular_archive_nearness_objective"
    held_out_evidence = provenance.get("held_out_evidence_status") == "eligible_held_out_diagnostic"
    return {
        "schema_version": "adversarial_proposal_comparison.v1",
        "state": state,
        "reason": reason,
        "claim_boundary": (
            HELD_OUT_DIAGNOSTIC_BOUNDARY if independent_available else CLAIM_BOUNDARY
        ),
        "result_classification": (
            "held_out_diagnostic_only" if independent_available else "plumbing_validation_only"
        ),
        "held_out_evidence": held_out_evidence,
        "benchmark_evidence": False,
        "planner_performance_claim": False,
        "decision_vocabulary": list(ISSUE_3275_DECISION_VOCABULARY),
        "synthetic_archive": synthetic_archive,
        "synthetic_search_space": synthetic_search_space,
        "search_space_state": search_space_state,
        "budget_per_arm": args.budget,
        "arm_overlap_policy": arms.policy,
        "arm_overlap_ids": arms.overlap_ids,
        "seed": args.seed,
        "diagnostic_archive_nearness": {
            "random_metrics": diagnostic_random_metrics,
            "proposal_metrics": diagnostic_proposal_metrics,
            "comparison": diagnostic_comparison,
        },
        "comparison": comparison,
        "comparison_interpretation": comparison_interpretation,
        "issue_2921_stop_rule": classify_issue_2921_stop_rule(
            independent_evaluation=independent_evaluation
        ),
        "archive_evaluation_provenance": provenance,
        "independent_outcome_evaluation": independent_evaluation,
    }


def main() -> int:
    """Main execution function."""
    args = parse_args()

    if args.check_contract is not None:
        exit_code, verdict = run_check_contract(args.check_contract)
        print(json.dumps(verdict, indent=2, sort_keys=True))
        return exit_code

    frozen = _contract_frozen_params(args)
    search_space_state, search_space_reason, search_space, synthetic_search_space = (
        load_search_space(args.search_space)
    )
    archive_state, archive_reason, archive_data, synthetic_archive = load_archive(args.archive)
    from robot_sf.adversarial.independent_outcomes import (
        AdmissionSpec,
        build_independent_outcome_evaluation,
        load_independent_outcomes,
    )
    from robot_sf.adversarial.proposal_model import FailureArchiveProposalModel

    outcome_state, outcome_reason, outcome_data = load_independent_outcomes(
        args.evaluation_outcomes
    )
    state, reason = _resolve_run_state(
        archive_state=archive_state,
        archive_reason=archive_reason,
        search_space_state=search_space_state,
        search_space_reason=search_space_reason,
        outcome_state=outcome_state,
        outcome_reason=outcome_reason,
        synthetic_archive=synthetic_archive,
        synthetic_search_space=synthetic_search_space,
    )

    model = FailureArchiveProposalModel(archive_data, search_space)
    rng = random.Random(args.seed)
    pool = [search_space.sample_candidate(rng) for _ in range(max(args.budget * 5, 50))]
    pool_ids = [f"pool_{i}" for i in range(len(pool))]
    ranked_pool = model.rank_candidates(pool, strategy="nearest_neighbor")
    ranked_ids = [f"pool_{i}" for i, _ in ranked_pool]

    from robot_sf.adversarial.disjoint_evaluation import assign_arms_disjoint_by_candidate

    arms = assign_arms_disjoint_by_candidate(
        ranked_ids, pool_ids, budget_per_arm=args.budget, rng_seed=args.seed
    )
    proposal_set = set(arms.proposal_ids)
    random_set = set(arms.random_ids)
    proposal_selection = [
        ranked_pool[i][0] for i, cid in enumerate(ranked_ids) if cid in proposal_set
    ]
    if not proposal_selection:
        proposal_selection = [cand for cand, _ in ranked_pool[: args.budget]]
    random_selection = [
        pool[i] for i, cid in enumerate(pool_ids) if cid in random_set
    ] or rng.sample(pool, min(args.budget, len(pool)))

    diagnostic_random_metrics, diagnostic_proposal_metrics, diagnostic_comparison = (
        _diagnostic_archive_nearness(model, proposal_selection, random_selection)
    )
    provenance = build_archive_evaluation_provenance(
        archive_data, state=state, synthetic_archive=synthetic_archive, split_seed=args.seed
    )
    independent_evaluation = build_independent_outcome_evaluation(
        outcome_data,
        budget_per_arm=args.budget,
        minimally_important=frozen["minimally_important"],
        admission_spec=AdmissionSpec(
            expected_target_planner_id=frozen["expected_target_planner_id"],
            expected_eval_family=frozen["expected_eval_family"],
            confirmation_threshold=frozen["confirmation_threshold"],
            expected_target_planner_config_sha256=frozen["expected_target_planner_config_sha256"],
        ),
        expected_eval_archive_sha256=provenance.get("eval_archive_sha256"),
        n_permutations=args.null_test_permutations,
        seed=args.seed,
    )
    provenance = build_archive_evaluation_provenance(
        archive_data,
        state=state,
        synthetic_archive=synthetic_archive,
        split_seed=args.seed,
        independent_evaluation=independent_evaluation,
    )
    report = _assemble_report(
        state=state,
        reason=reason,
        synthetic_archive=synthetic_archive,
        synthetic_search_space=synthetic_search_space,
        search_space_state=search_space_state,
        args=args,
        arms=arms,
        diagnostic_random_metrics=diagnostic_random_metrics,
        diagnostic_proposal_metrics=diagnostic_proposal_metrics,
        diagnostic_comparison=diagnostic_comparison,
        independent_evaluation=independent_evaluation,
        provenance=provenance,
    )
    report_str = json.dumps(report, indent=2, sort_keys=True)
    print(report_str)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report_str + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
