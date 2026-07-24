#!/usr/bin/env python3
"""Run proposal model vs random candidate sampler under identical budget."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from robot_sf.adversarial.proposal_model import FailureArchiveProposalModel

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


def classify_issue_2921_stop_rule(
    *,
    held_out_evidence: bool,
    held_out_status: str | None,
    comparison: dict[str, float | int | str],
) -> dict[str, Any]:
    """Return the issue #2921 continue/stop/inconclusive decision without promoting claims.

    The issue #3275 rerun can only move into the #2921 stop-rule lane after the held-out
    diagnostic gate is open. Until then the stop rule is explicitly blocked so plumbing-only
    deltas cannot be mistaken for proposal-model evidence.
    """
    if not held_out_evidence:
        return {
            "status": "blocked",
            "reason": held_out_status or "held_out_evidence_not_available",
            "evidence_tier": "analysis_only",
            "claim_boundary": "no #2921 stop-rule decision without held-out diagnostic evidence",
        }

    mean_delta = float(comparison["mean_objective_improvement"])
    failure_delta = int(comparison["failure_count_improvement"])
    if mean_delta > 0.0 or failure_delta > 0:
        status = "continue"
        reason = "diagnostic held-out deltas are positive; run the next predeclared proof step"
    elif mean_delta < 0.0 or failure_delta < 0:
        status = "stop"
        reason = "diagnostic held-out deltas are negative; do not expand this proposal lane"
    else:
        status = "inconclusive"
        reason = "diagnostic held-out deltas are neutral; do not expand this proposal lane"

    return {
        "status": status,
        "reason": reason,
        "evidence_tier": "diagnostic_only",
        "claim_boundary": (
            "issue #2921 stop-rule classification from held-out diagnostic evidence only; "
            "not benchmark, paper, or planner-performance evidence"
        ),
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


def build_archive_evaluation_provenance(
    archive_data: dict[str, Any],
    *,
    state: str,
    synthetic_archive: bool,
    split_seed: int,
    independent_evaluation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build archive-evaluation provenance for the comparison report.

    For the diagnostic/blocked plumbing paths this preserves the historical
    placeholder provenance (no disjoint split). For an active real-archive run it
    computes a disjoint scenario-family split, real fit/eval overlap provenance
    (scenario-family / seed / archive-id), and a fail-closed held-out-evidence
    classification. The held-out claim stays fail-closed here: independent
    (non-archive-nearness) planner outcomes are not yet available, so eligibility
    resolves to a precise ``not_available`` reason.

    Returns:
        A JSON-safe provenance dict.
    """
    independent_evaluation = independent_evaluation or {}
    provenance: dict[str, Any] = {
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


def compute_metrics(selection: list[Any], evaluate_fn: Any) -> dict[str, Any]:
    """Compute summary metrics for a candidate selection."""
    objs = [evaluate_fn(c) for c in selection]
    return {
        "mean_objective": round(sum(objs) / len(objs), 4) if objs else 0.0,
        "max_objective": round(max(objs), 4) if objs else 0.0,
        "failure_count": sum(1 for o in objs if o >= 8.0),
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Compare proposal model vs random sampler under identical budget."
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=None,
        help="Path to failure archive JSON file. If missing/empty, runs diagnostic-only path.",
    )
    parser.add_argument(
        "--search-space",
        type=Path,
        default=None,
        help="Path to search space config YAML file.",
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=None,
        help="Path to frozen study contract JSON file.",
    )
    parser.add_argument(
        "--check-contract",
        nargs="?",
        const=True,
        default=False,
        help="Side-effect-free check-only command to verify frozen study contract.",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=10,
        help="Number of candidates to evaluate/rank.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for candidate pool generation.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write the comparison JSON report.",
    )
    parser.add_argument(
        "--evaluation-outcomes",
        type=Path,
        default=None,
        help=(
            "Optional JSON payload with independent planner-execution outcomes. "
            "Absent payload keeps held-out evidence fail-closed."
        ),
    )
    parser.add_argument(
        "--null-test-permutations",
        type=int,
        default=1000,
        help="Number of permutations for independent-outcome null tests.",
    )
    args = parser.parse_args()
    if args.budget < 0:
        parser.error("--budget must be >= 0")
    if args.null_test_permutations < 1:
        parser.error("--null-test-permutations must be >= 1")
    return args


def handle_check_contract(args: argparse.Namespace, contract_path: Path | None) -> int:
    """Handle side-effect-free contract verification command."""
    from robot_sf.adversarial.disjoint_evaluation import verify_same_planner_contract
    from robot_sf.adversarial.proposal_model import FailureArchiveProposalModel

    if contract_path is None:
        default_contract = Path("configs/adversarial/issue_3275_same_planner_contract.json")
        contract_path = default_contract if default_contract.exists() else None

    if contract_path is None or not contract_path.exists():
        msg = {"status": "failed", "reason": f"Contract file {contract_path} not found."}
        print(json.dumps(msg, indent=2))
        return 1

    contract_data = json.loads(contract_path.read_text(encoding="utf-8"))
    archive_path = (
        args.archive
        if args.archive is not None
        else Path(contract_data.get("tracked_archive_path", ""))
    )

    if not archive_path.exists():
        msg = {"status": "failed", "reason": f"Archive file {archive_path} not found."}
        print(json.dumps(msg, indent=2))
        return 1

    raw_bytes = archive_path.read_bytes()
    archive_data = json.loads(raw_bytes.decode("utf-8"))
    verif = verify_same_planner_contract(contract_data, archive_data, raw_bytes)

    fit_model = FailureArchiveProposalModel(
        archive_data,
        allowed_fit_ids=contract_data.get("fit_entry_ids"),
        target_planner=contract_data.get("target_planner"),
    )
    feature_audit = fit_model.audit_feature_semantics(
        target_family=contract_data.get("eval_scenario_family")
    )
    verif["feature_semantics_audit"] = feature_audit
    if not feature_audit.get("passed", False):
        verif["checks_passed"] = False
        verif.setdefault("blocking_reasons", []).append("feature_semantics_audit_failed")

    verif_str = json.dumps(verif, indent=2, sort_keys=True)
    print(verif_str)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(verif_str + "\n", encoding="utf-8")
    return 0 if verif["checks_passed"] else 1


def _select_and_score_candidates(
    model: FailureArchiveProposalModel,
    search_space: Any,
    budget: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Generate candidates and compute diagnostic archive-nearness metrics."""
    rng = random.Random(seed)
    pool_size = max(budget * 5, 50)
    pool = [search_space.sample_candidate(rng) for _ in range(pool_size)]

    random_selection = rng.sample(pool, min(budget, len(pool)))
    ranked_pool = model.rank_candidates(pool, strategy="nearest_neighbor")
    proposal_selection = [cand for cand, score in ranked_pool[:budget]]

    def evaluate_objective(candidate: Any) -> float:
        if not model.entries:
            return 0.0
        distances = [
            model._distance(candidate, entry.get("candidate", {})) for entry in model.entries
        ]
        min_dist = min(distances) if distances else 999.0
        return max(0.0, 10.0 - min_dist)

    diag_random_metrics = compute_metrics(random_selection, evaluate_objective)
    diag_proposal_metrics = compute_metrics(proposal_selection, evaluate_objective)

    dummy_yaml = Path("dummy_scenario.yaml")
    diag_proposal_metrics["certification_statuses_advisory"] = [
        model.certify_candidate(c, dummy_yaml, require_certification=False).status
        for c in proposal_selection
    ]
    diag_proposal_metrics["certification_statuses_strict"] = [
        model.certify_candidate(c, dummy_yaml, require_certification=True).status
        for c in proposal_selection
    ]

    return diag_random_metrics, diag_proposal_metrics


def _assemble_comparison_report(ctx: dict[str, Any]) -> dict[str, Any]:
    """Assemble the comparison report dict from context dictionary."""
    held_out_evidence = ctx["held_out_evidence"]
    independent_evaluation = ctx["independent_evaluation"]
    return {
        "schema_version": "adversarial_proposal_comparison.v1",
        "state": ctx["state"],
        "reason": ctx["reason"],
        "claim_boundary": HELD_OUT_DIAGNOSTIC_BOUNDARY if held_out_evidence else CLAIM_BOUNDARY,
        "result_classification": (
            "held_out_diagnostic_only" if held_out_evidence else "plumbing_validation_only"
        ),
        "held_out_evidence": held_out_evidence,
        "benchmark_evidence": False,
        "planner_performance_claim": False,
        "synthetic_archive": ctx["synthetic_archive"],
        "synthetic_search_space": ctx["synthetic_search_space"],
        "search_space_state": ctx["search_space_state"],
        "budget": ctx["budget"],
        "seed": ctx["seed"],
        "random_metrics": ctx["random_metrics"],
        "proposal_metrics": ctx["proposal_metrics"],
        "diagnostic_archive_nearness_metrics": ctx["diagnostic_archive_nearness_metrics"],
        "comparison": ctx["comparison"],
        "issue_2921_stop_rule": ctx["issue_2921_stop_rule"],
        "archive_evaluation_provenance": ctx["provenance"],
        "independent_outcome_evaluation": independent_evaluation,
        "contract_verification": ctx.get("contract_verification"),
        "null_tests": independent_evaluation.get(
            "null_tests",
            {
                "shuffled_archive_outcomes": "not_run_requires_real_certified_archive",
                "proposal_ranking_permutation": "not_run_requires_real_certified_archive",
                "required_for_held_out_claim": True,
            },
        ),
    }


def _write_json(args: argparse.Namespace, payload: dict[str, Any]) -> None:
    """Print a JSON payload and optionally persist it at the requested output path."""
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


def _load_contract(path: Path | None) -> tuple[dict[str, Any] | None, str | None]:
    """Load a normal-run contract without silently accepting malformed content."""
    if path is None:
        return None, None
    if not path.exists():
        return None, f"Contract file {path} not found."
    try:
        contract_data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return None, f"Failed to load contract: {exc}."
    if not isinstance(contract_data, dict):
        return None, "Contract payload must be a JSON object."
    return contract_data, None


def _contract_input_paths(
    args: argparse.Namespace, contract_data: dict[str, Any] | None
) -> tuple[Path | None, Path | None]:
    """Choose explicit CLI inputs first, then the contract's frozen input paths."""
    archive_path = args.archive
    search_space_path = args.search_space
    if contract_data is None:
        return archive_path, search_space_path
    if archive_path is None:
        archive_path = Path(contract_data.get("tracked_archive_path", ""))
    if search_space_path is None:
        search_space_path = Path(contract_data.get("search_space_path", ""))
    return archive_path, search_space_path


def _verify_normal_contract(
    contract_data: dict[str, Any] | None,
    archive_data: dict[str, Any],
    archive_path: Path | None,
    synthetic_archive: bool,
) -> dict[str, Any] | None:
    """Verify a supplied normal-run contract before any candidate selection."""
    if contract_data is None:
        return None
    if synthetic_archive or archive_path is None or not archive_path.is_file():
        return {
            "status": "failed",
            "checks_passed": False,
            "blocking_reasons": ["normal_contract_runs_require_a_real_readable_archive"],
        }
    from robot_sf.adversarial.disjoint_evaluation import verify_same_planner_contract

    return verify_same_planner_contract(contract_data, archive_data, archive_path.read_bytes())


def _resolve_run_state(
    archive_state: str,
    archive_reason: str,
    search_space_state: str,
    search_space_reason: str,
    outcome_state: str,
    outcome_reason: str,
    synthetic_archive: bool,
    synthetic_search_space: bool,
) -> tuple[str, str]:
    """Build an explicit blocked state whenever real inputs fall back to fixtures."""
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


def _resolve_comparison_metrics(
    held_out_evidence: bool,
    independent_evaluation: dict[str, Any],
    diag_proposal_metrics: dict[str, Any],
    diag_random_metrics: dict[str, Any],
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Keep diagnostic archive-nearness metrics out of an open outcome gate."""
    if held_out_evidence:
        return (
            "independent_planner_execution_outcomes",
            independent_evaluation["proposal_metrics"],
            independent_evaluation["random_metrics"],
        )
    if independent_evaluation.get("independent_outcomes_available"):
        return (
            "independent_outcomes_rejected_by_held_out_gate",
            diag_proposal_metrics,
            diag_random_metrics,
        )
    return (
        "plumbing_only_circular_archive_nearness_objective",
        diag_proposal_metrics,
        diag_random_metrics,
    )


@dataclass(frozen=True)
class _RunInputs:
    """Validated inputs passed to the side-effect-free comparison path."""

    contract_data: dict[str, Any] | None
    archive_data: dict[str, Any]
    archive_state: str
    archive_reason: str
    synthetic_archive: bool
    search_space: Any
    search_space_state: str
    search_space_reason: str
    synthetic_search_space: bool
    contract_verification: dict[str, Any] | None


def _run_comparison(
    args: argparse.Namespace,
    inputs: _RunInputs,
) -> int:
    """Run only the diagnostic comparison after contract verification has passed."""

    from robot_sf.adversarial.independent_outcomes import (
        build_independent_outcome_evaluation,
        load_independent_outcomes,
    )

    outcome_state, outcome_reason, outcome_data = load_independent_outcomes(
        args.evaluation_outcomes
    )
    state, reason = _resolve_run_state(
        inputs.archive_state,
        inputs.archive_reason,
        inputs.search_space_state,
        inputs.search_space_reason,
        outcome_state,
        outcome_reason,
        inputs.synthetic_archive,
        inputs.synthetic_search_space,
    )

    contract_fit_ids = (
        inputs.contract_data.get("fit_entry_ids") if inputs.contract_data is not None else None
    )

    from robot_sf.adversarial.proposal_model import FailureArchiveProposalModel, isolate_fit_entries

    fit_archive_data = (
        isolate_fit_entries(
            inputs.archive_data, allowed_fit_ids=contract_fit_ids, target_planner="social_force"
        )
        if not inputs.synthetic_archive and isinstance(inputs.archive_data, dict)
        else inputs.archive_data
    )

    model = FailureArchiveProposalModel(
        fit_archive_data, inputs.search_space, target_planner="social_force"
    )

    diag_random_metrics, diag_proposal_metrics = _select_and_score_candidates(
        model, inputs.search_space, args.budget, args.seed
    )

    provenance = build_archive_evaluation_provenance(
        inputs.archive_data,
        state=state,
        synthetic_archive=inputs.synthetic_archive,
        split_seed=args.seed,
    )
    independent_evaluation = build_independent_outcome_evaluation(
        outcome_data,
        budget=args.budget,
        n_permutations=args.null_test_permutations,
        seed=args.seed,
        expected_eval_archive_sha256=provenance.get("eval_archive_sha256"),
        outcome_contract=inputs.contract_data,
    )
    provenance = build_archive_evaluation_provenance(
        inputs.archive_data,
        state=state,
        synthetic_archive=inputs.synthetic_archive,
        split_seed=args.seed,
        independent_evaluation=independent_evaluation,
    )
    held_out_status = provenance.get("held_out_evidence_status")
    held_out_evidence = held_out_status == "eligible_held_out_diagnostic"

    comparison_interpretation, proposal_metrics, random_metrics = _resolve_comparison_metrics(
        held_out_evidence,
        independent_evaluation,
        diag_proposal_metrics,
        diag_random_metrics,
    )

    comparison = {
        "interpretation": comparison_interpretation,
        "mean_objective_improvement": round(
            proposal_metrics["mean_objective"] - random_metrics["mean_objective"], 4
        ),
        "max_objective_improvement": round(
            proposal_metrics["max_objective"] - random_metrics["max_objective"], 4
        ),
        "failure_count_improvement": (
            proposal_metrics["failure_count"] - random_metrics["failure_count"]
        ),
    }
    issue_2921_stop_rule = classify_issue_2921_stop_rule(
        held_out_evidence=held_out_evidence,
        held_out_status=held_out_status,
        comparison=comparison,
    )

    ctx = {
        "state": state,
        "reason": reason,
        "held_out_evidence": held_out_evidence,
        "synthetic_archive": inputs.synthetic_archive,
        "synthetic_search_space": inputs.synthetic_search_space,
        "search_space_state": inputs.search_space_state,
        "budget": args.budget,
        "seed": args.seed,
        "proposal_metrics": proposal_metrics,
        "random_metrics": random_metrics,
        "diagnostic_archive_nearness_metrics": {
            "proposal_metrics": diag_proposal_metrics,
            "random_metrics": diag_random_metrics,
        },
        "comparison": comparison,
        "issue_2921_stop_rule": issue_2921_stop_rule,
        "provenance": provenance,
        "independent_evaluation": independent_evaluation,
        "contract_verification": inputs.contract_verification,
    }
    report = _assemble_comparison_report(ctx)

    _write_json(args, report)
    return 1 if state == "blocked" else 0


def main() -> int:
    """Main execution function."""
    args = parse_args()
    contract_path = (
        Path(args.check_contract) if isinstance(args.check_contract, str) else args.contract
    )
    if args.check_contract:
        return handle_check_contract(args, contract_path)

    contract_data, contract_error = _load_contract(contract_path)
    if contract_error is not None:
        _write_json(args, {"state": "blocked", "reason": contract_error})
        return 1
    archive_path, search_space_path = _contract_input_paths(args, contract_data)
    search_state, search_reason, search_space, synthetic_search = load_search_space(
        search_space_path
    )
    archive_state, archive_reason, archive_data, synthetic_archive = load_archive(archive_path)
    verification = _verify_normal_contract(
        contract_data, archive_data, archive_path, synthetic_archive
    )
    if verification is not None and not verification["checks_passed"]:
        _write_json(
            args,
            {
                "state": "blocked",
                "reason": "Frozen study contract failed verification; no candidates were selected.",
                "contract_verification": verification,
            },
        )
        return 1
    return _run_comparison(
        args,
        _RunInputs(
            contract_data,
            archive_data,
            archive_state,
            archive_reason,
            synthetic_archive,
            search_space,
            search_state,
            search_reason,
            synthetic_search,
            verification,
        ),
    )


if __name__ == "__main__":
    import sys

    sys.exit(main())
