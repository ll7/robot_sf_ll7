#!/usr/bin/env python3
"""Run proposal model vs random candidate sampler under identical budget."""

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


def classify_issue_2921_stop_rule(
    *,
    held_out_evidence: bool,
    held_out_status: str | None,
    comparison: dict[str, float | int | str],
) -> dict[str, Any]:
    """Return the issue #2921 continue/revise/stop decision without promoting claims.

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
        status = "revise"
        reason = "diagnostic held-out deltas are neutral; revise before another empirical batch"

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


def main() -> int:
    """Main execution function."""
    args = parse_args()

    search_space_state, search_space_reason, search_space, synthetic_search_space = (
        load_search_space(args.search_space)
    )
    archive_state, archive_reason, archive_data, synthetic_archive = load_archive(args.archive)
    from robot_sf.adversarial.independent_outcomes import (
        build_independent_outcome_evaluation,
        load_independent_outcomes,
    )

    outcome_state, outcome_reason, outcome_data = load_independent_outcomes(
        args.evaluation_outcomes
    )
    state = archive_state
    reason_parts = [archive_reason, search_space_reason, outcome_reason]
    if not synthetic_archive and synthetic_search_space:
        state = "blocked"
        reason_parts.append("Real-archive runs require a real search-space config.")
    if search_space_state == "blocked" and state == "active":
        state = "blocked"
    if outcome_state == "blocked":
        state = "blocked"
    reason = " ".join(reason_parts)

    from robot_sf.adversarial.proposal_model import FailureArchiveProposalModel

    # Initialize proposal model
    model = FailureArchiveProposalModel(archive_data, search_space)

    # Generate a candidate pool deterministically using the seed
    rng = random.Random(args.seed)
    pool_size = max(args.budget * 5, 50)
    pool = [search_space.sample_candidate(rng) for _ in range(pool_size)]

    # Random Sampler Selection
    random_selection = rng.sample(pool, min(args.budget, len(pool)))

    # Proposal Model Selection
    ranked_pool = model.rank_candidates(pool, strategy="nearest_neighbor")
    proposal_selection = [cand for cand, score in ranked_pool[: args.budget]]

    # Define a synthetic objective function for evaluation
    def evaluate_objective(candidate: Any) -> float:
        if not model.entries:
            return 0.0
        distances = [
            model._distance(candidate, entry.get("candidate", {})) for entry in model.entries
        ]
        min_dist = min(distances) if distances else 999.0
        return max(0.0, 10.0 - min_dist)

    # Compute metrics
    random_metrics = compute_metrics(random_selection, evaluate_objective)
    proposal_metrics = compute_metrics(proposal_selection, evaluate_objective)

    # Certification check
    dummy_yaml = Path("dummy_scenario.yaml")
    cert_statuses_advisory = [
        model.certify_candidate(c, dummy_yaml, require_certification=False).status
        for c in proposal_selection
    ]
    cert_statuses_strict = [
        model.certify_candidate(c, dummy_yaml, require_certification=True).status
        for c in proposal_selection
    ]

    proposal_metrics["certification_statuses_advisory"] = cert_statuses_advisory
    proposal_metrics["certification_statuses_strict"] = cert_statuses_strict
    provenance = build_archive_evaluation_provenance(
        archive_data,
        state=state,
        synthetic_archive=synthetic_archive,
        split_seed=args.seed,
    )
    independent_evaluation = build_independent_outcome_evaluation(
        outcome_data,
        budget=args.budget,
        n_permutations=args.null_test_permutations,
        seed=args.seed,
        expected_eval_archive_sha256=provenance.get("eval_archive_sha256"),
    )
    provenance = build_archive_evaluation_provenance(
        archive_data,
        state=state,
        synthetic_archive=synthetic_archive,
        split_seed=args.seed,
        independent_evaluation=independent_evaluation,
    )
    held_out_status = provenance.get("held_out_evidence_status")
    held_out_evidence = held_out_status == "eligible_held_out_diagnostic"
    if held_out_evidence:
        comparison_interpretation = "independent_planner_execution_outcomes"
    elif independent_evaluation.get("independent_outcomes_available"):
        comparison_interpretation = "independent_outcomes_rejected_by_held_out_gate"
    else:
        comparison_interpretation = "plumbing_only_circular_archive_nearness_objective"

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

    report = {
        "schema_version": "adversarial_proposal_comparison.v1",
        "state": state,
        "reason": reason,
        "claim_boundary": HELD_OUT_DIAGNOSTIC_BOUNDARY if held_out_evidence else CLAIM_BOUNDARY,
        "result_classification": (
            "held_out_diagnostic_only" if held_out_evidence else "plumbing_validation_only"
        ),
        "held_out_evidence": held_out_evidence,
        "benchmark_evidence": False,
        "planner_performance_claim": False,
        "synthetic_archive": synthetic_archive,
        "synthetic_search_space": synthetic_search_space,
        "search_space_state": search_space_state,
        "budget": args.budget,
        "seed": args.seed,
        "random_metrics": random_metrics,
        "proposal_metrics": proposal_metrics,
        "comparison": comparison,
        "issue_2921_stop_rule": issue_2921_stop_rule,
        "archive_evaluation_provenance": provenance,
        "independent_outcome_evaluation": independent_evaluation,
        "null_tests": independent_evaluation.get(
            "null_tests",
            {
                "shuffled_archive_outcomes": "not_run_requires_real_certified_archive",
                "proposal_ranking_permutation": "not_run_requires_real_certified_archive",
                "required_for_held_out_claim": True,
            },
        ),
    }

    report_str = json.dumps(report, indent=2, sort_keys=True)
    print(report_str)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report_str + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
