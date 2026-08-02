#!/usr/bin/env python3
"""Issue #6474 social-compliance cross-planner report builder (preregistration stub).

This module freezes the **declared estimands** for the #6474 social-compliance
cross-planner comparison. It is a PREREGISTRATION STUB: it declares the analysis
contract -- paired planner-pair effects by metric family and scenario family,
declared support/denominators, bootstrap CI95, and Holm-Bonferroni control
across the exposed planner-pair-by-metric-family decisions -- but it does NOT
execute any campaign, load any campaign output, or compute any effect.

Production execution is separately gated via a follow-up SLURM child issue and
is not authorized by this preregistration alone. Running ``main`` only emits the
frozen estimand manifest so the analysis contract is auditable before any run.

Claim boundary (from the Domain-Aware Approval in #6474):
    Target claim: estimate within-simulator planner-pair differences for the
    versioned social-compliance metric families. No universal planner
    superiority, fairness, deployment-ethics, legibility, social-validity,
    safety, welfare, real-world, or composite-score claim. Fallback, degraded,
    missing, or invalid rows are excluded rather than zero-imputed.

Comparators and execution mode:
    goal (native), social_force (adapter), orca (adapter). Every row must record
    its execution mode; no fallback or degraded row may count as success evidence.

Usage (preregistration manifest only; no campaign is run):
    uv run python scripts/benchmark/build_social_compliance_cross_planner_report_issue_6474.py
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robot_sf.benchmark.social_compliance import (
    SOCIAL_COMPLIANCE_CLAIM_CLASS,
    SOCIAL_COMPLIANCE_SCHEMA_VERSION,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN_CONFIG_PATH = (
    REPO_ROOT / "configs/benchmarks/issue_6474_social_compliance_nominal_campaign.yaml"
)
SCENARIO_MATRIX_PATH = REPO_ROOT / "configs/scenarios/issue_6474_social_compliance_nominal.yaml"
PREREGISTRATION_PATH = (
    REPO_ROOT / "docs/context/evidence/issue_6474_social_compliance_preregistration.json"
)

# --- Frozen campaign design (mirrors the campaign config and preregistration) ---

PLANNERS = ("goal", "social_force", "orca")
PLANNER_EXECUTION_MODES = {
    "goal": "native",
    "social_force": "adapter",
    "orca": "adapter",
}
# Three unordered planner pairs; each pair's direction is recorded per estimand.
PLANNER_PAIRS = (
    ("goal", "social_force"),
    ("goal", "orca"),
    ("social_force", "orca"),
)
# Paired seeds 111-140 (30 paired seeds), matched across planners within each seed.
SEEDS = tuple(range(111, 141))
# Six frozen #6102 medium-band scenario identifiers, grouped by archetype family.
SCENARIOS = (
    "classic_head_on_corridor_medium",
    "classic_doorway_medium",
    "classic_group_crossing_medium",
    "classic_merging_medium",
    "classic_overtaking_medium",
    "classic_station_platform_medium",
)
SCENARIO_FAMILIES = (
    "head_on_corridor",
    "doorway",
    "group_crossing",
    "merging",
    "overtaking",
    "station_platform",
)
# Five versioned social-compliance metric families from
# configs/benchmarks/social_compliance_metric_contract_v1.yaml.
METRIC_FAMILIES = (
    "comfort_exposure",
    "distributional_inconvenience",
    "flow_disruption",
    "legibility_progress",
    "pedestrian_deviation",
)
METRIC_DIRECTION = "lower_is_better"
HORIZON_STEPS = 250
DT_SECONDS = 0.1
EXPECTED_CELL_COUNT = len(SCENARIOS) * len(PLANNERS) * len(SEEDS)  # 540
RESAMPLING_UNIT = "paired_seed_block"
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_CONFIDENCE = 0.95
FAMILYWISE_ALPHA = 0.05
MULTIPLICITY_FAMILY = "holm_bonferroni"
# Multiplicity is controlled across planner-pair-by-metric-family decisions.
NUM_DECISIONS = len(PLANNER_PAIRS) * len(METRIC_FAMILIES)  # 3 * 5 = 15


@dataclass(frozen=True)
class PairedEstimand:
    """One preregistered planner-pair-by-metric-family paired estimand.

    Attributes:
        decision_id: Stable identifier ``<pair>-<metric_family>``.
        comparator_a: First planner in the pair.
        comparator_b: Second planner in the pair.
        metric_family: Versioned social-compliance metric family.
        estimand: Estimand type (paired mean difference per episode).
        effect_definition: Plain-language definition of the paired effect.
        direction: Metric direction convention (lower_is_better).
        resampling_unit: Bootstrap resampling unit (paired seed block).
        ci_method: Confidence interval method (paired bootstrap percentile CI95).
        multiplicity_family: Multiplicity correction family (Holm-Bonferroni).
    """

    decision_id: str
    comparator_a: str
    comparator_b: str
    metric_family: str
    estimand: str
    effect_definition: str
    direction: str
    resampling_unit: str
    ci_method: str
    multiplicity_family: str


def declared_estimands() -> list[PairedEstimand]:
    """Return the frozen list of planner-pair-by-metric-family paired estimands.

    The list enumerates every exposed multiplicity decision: three planner pairs
    crossed with five metric families (15 decisions). Each decision estimates the
    paired mean per-episode difference between the two planners for that metric
    family, with uncertainty from a paired-seed-block bootstrap and Holm-
    Bonferroni control across the 15 decisions.
    """
    estimands: list[PairedEstimand] = []
    for comparator_a, comparator_b in PLANNER_PAIRS:
        for metric_family in METRIC_FAMILIES:
            estimands.append(
                PairedEstimand(
                    decision_id=f"{comparator_a}_vs_{comparator_b}-{metric_family}",
                    comparator_a=comparator_a,
                    comparator_b=comparator_b,
                    metric_family=metric_family,
                    estimand="paired_mean_difference_per_episode",
                    effect_definition=(
                        f"Mean over episodes of (per-episode {metric_family} value for "
                        f"{comparator_b} minus {comparator_a}), paired by seed and "
                        "restricted to rows where the metric family is available for both "
                        "planners in the pair."
                    ),
                    direction=METRIC_DIRECTION,
                    resampling_unit=RESAMPLING_UNIT,
                    ci_method="paired_bootstrap_percentile_ci95",
                    multiplicity_family=MULTIPLICITY_FAMILY,
                )
            )
    return estimands


def compute_paired_effects(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
    """STUB: compute paired mean effects per estimand from campaign output.

    Not implemented: this is a preregistration stub. The real implementation will
    load only native or declared-adapter rows, exclude fallback/degraded/missing/
    invalid rows (never zero-impute), group by metric family and scenario family,
    and return the paired mean per-episode difference for each estimand with
    declared support counts and denominators.
    """
    raise NotImplementedError(
        "Issue #6474 analysis is preregistration-only; no campaign output has been "
        "loaded or executed. Production execution is gated by a follow-up SLURM child issue."
    )


def bootstrap_ci95(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
    """STUB: paired-seed-block bootstrap percentile CI95 for one estimand.

    Not implemented: the real implementation will resample whole paired seed
    blocks (resampling unit declared above) with a fixed RNG seed, use the frozen
    number of replicates, and return the CI95 alongside support and denominator.
    """
    raise NotImplementedError(
        "Issue #6474 bootstrap CI95 is preregistration-only and not yet implemented."
    )


def holm_corrected_p_values(*_args: Any, **_kwargs: Any) -> dict[str, float]:
    """STUB: Holm-Bonferroni adjusted p-values across the 15 multiplicity decisions.

    Not implemented: the real implementation will apply stepwise Holm-Bonferroni
    adjustment across the planner-pair-by-metric-family decisions and return
    adjusted p-values keyed by decision_id.
    """
    raise NotImplementedError(
        "Issue #6474 Holm correction is preregistration-only and not yet implemented."
    )


def build_report(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
    """STUB: assemble the full cross-planner social-compliance report.

    Not implemented: the real implementation will compose paired effects,
    bootstrap CI95, Holm-corrected decisions, declared support/denominators, and
    the claim boundary into a single report. No composite social-compliance
    ranking is produced.
    """
    raise NotImplementedError(
        "Issue #6474 report build is preregistration-only; no campaign has run."
    )


def estimand_manifest() -> dict[str, Any]:
    """Return the preregistration-only estimand manifest without executing any campaign.

    The manifest declares the frozen campaign design, the multiplicity family,
    the claim boundary, and the full list of paired estimands. It is auditable
    evidence of the analysis contract frozen before any SLURM submission.
    """
    estimands = declared_estimands()
    return {
        "schema_version": "issue_6474_social_compliance_preregistration_manifest.v1",
        "review_marker": "AI-GENERATED NEEDS-REVIEW",
        "issue": 6474,
        "child_issue": 6638,
        "status": "preregistration_only",
        "evidence_status": "not_benchmark_evidence",
        "campaign_config": CAMPAIGN_CONFIG_PATH.relative_to(REPO_ROOT).as_posix(),
        "scenario_matrix": SCENARIO_MATRIX_PATH.relative_to(REPO_ROOT).as_posix(),
        "social_compliance_schema_version": SOCIAL_COMPLIANCE_SCHEMA_VERSION,
        "claim_class": SOCIAL_COMPLIANCE_CLAIM_CLASS,
        "planners": list(PLANNERS),
        "planner_execution_modes": dict(PLANNER_EXECUTION_MODES),
        "planner_pairs": [list(pair) for pair in PLANNER_PAIRS],
        "seeds": list(SEEDS),
        "scenario_families": list(SCENARIO_FAMILIES),
        "metric_families": list(METRIC_FAMILIES),
        "metric_direction": METRIC_DIRECTION,
        "horizon_steps": HORIZON_STEPS,
        "dt_seconds": DT_SECONDS,
        "expected_cell_count": EXPECTED_CELL_COUNT,
        "estimand": "paired_mean_difference_per_episode",
        "resampling_unit": RESAMPLING_UNIT,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_confidence": BOOTSTRAP_CONFIDENCE,
        "multiplicity_family": MULTIPLICITY_FAMILY,
        "familywise_alpha": FAMILYWISE_ALPHA,
        "num_multiplicity_decisions": NUM_DECISIONS,
        "claim_boundary": (
            "Estimate within-simulator planner-pair differences for the versioned "
            "social-compliance metric families. No universal planner superiority, "
            "fairness, deployment-ethics, legibility, social-validity, safety, welfare, "
            "real-world, or composite-score claim. Fallback, degraded, missing, or "
            "invalid rows are excluded rather than zero-imputed."
        ),
        "estimands": [asdict(estimand) for estimand in estimands],
        "execution_note": (
            "This manifest is declared only; no campaign has been executed and no "
            "campaign output has been loaded. Production execution is separately gated "
            "by a follow-up SLURM child issue."
        ),
    }


def _build_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the preregistration manifest emitter."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate that the declared estimand count matches the multiplicity contract and exit.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Emit the preregistration-only estimand manifest without executing any campaign.

    Returns:
        0 when the manifest is emitted and the multiplicity contract checks pass.
    """
    parser = _build_parser()
    _args = parser.parse_args(argv)
    manifest = estimand_manifest()
    # Defensive multiplicity contract check: the number of estimands must equal the
    # declared number of planner-pair-by-metric-family multiplicity decisions.
    if len(manifest["estimands"]) != manifest["num_multiplicity_decisions"]:
        print(
            "ERROR: estimand count does not match the multiplicity contract",
            file=sys.stderr,
        )
        return 1
    if _args.check_only:
        print(
            "OK: "
            f"{len(manifest['estimands'])} estimands / "
            f"{manifest['num_multiplicity_decisions']} multiplicity decisions "
            "(preregistration-only; no campaign executed)."
        )
        return 0
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
