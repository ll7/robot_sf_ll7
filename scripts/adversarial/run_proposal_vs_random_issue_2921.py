#!/usr/bin/env python3
"""Run proposal model vs random candidate sampler under identical budget."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

CLAIM_BOUNDARY = (
    "analysis_only_diagnostic: proposal-vs-random results from the synthetic fixture only "
    "exercise model plumbing and report shape. They are not held-out archive evidence, benchmark "
    "evidence, planner-performance evidence, or a claim that learned proposals improve yield."
)


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


def load_search_space(path: Path | None) -> Any:
    """Load SearchSpaceConfig from path, falling back to synthetic if not found/error."""
    from robot_sf.adversarial.config import SearchSpaceConfig

    if path and path.exists():
        try:
            return SearchSpaceConfig.from_file(path)
        except (ValueError, TypeError, OSError):
            pass
    return create_synthetic_search_space()


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
    args = parser.parse_args()
    if args.budget < 0:
        parser.error("--budget must be >= 0")
    return args


def main() -> int:
    """Main execution function."""
    args = parse_args()

    search_space = load_search_space(args.search_space)
    state, reason, archive_data, synthetic_evidence = load_archive(args.archive)

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

    report = {
        "schema_version": "adversarial_proposal_comparison.v1",
        "state": state,
        "reason": reason,
        "claim_boundary": CLAIM_BOUNDARY,
        "result_classification": "diagnostic_only" if synthetic_evidence else "analysis_only",
        "held_out_evidence": not synthetic_evidence,
        "benchmark_evidence": False,
        "planner_performance_claim": False,
        "synthetic_evidence": synthetic_evidence,
        "budget": args.budget,
        "seed": args.seed,
        "random_metrics": random_metrics,
        "proposal_metrics": proposal_metrics,
        "comparison": {
            "mean_objective_improvement": round(
                proposal_metrics["mean_objective"] - random_metrics["mean_objective"], 4
            ),
            "max_objective_improvement": round(
                proposal_metrics["max_objective"] - random_metrics["max_objective"], 4
            ),
            "failure_count_improvement": (
                proposal_metrics["failure_count"] - random_metrics["failure_count"]
            ),
        },
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
