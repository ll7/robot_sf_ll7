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
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from robot_sf.benchmark.camera_ready._config import load_campaign_config
from robot_sf.benchmark.runner import load_scenario_matrix
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
CLASSIC_SCENARIO_MATRIX_PATH = REPO_ROOT / "configs/scenarios/classic_interactions.yaml"
METRIC_CONTRACT_PATH = REPO_ROOT / "configs/benchmarks/social_compliance_metric_contract_v1.yaml"
PREFLIGHT_CONFIG_PATH = (
    REPO_ROOT / "configs/benchmarks/issue_6481_social_compliance_preflight_smoke.yaml"
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
METRIC_FAMILY_DENOMINATORS = {
    "comfort_exposure": "pedestrian_steps",
    "distributional_inconvenience": "pedestrians_with_delay_samples",
    "flow_disruption": "pedestrians_with_reference_arrival",
    "legibility_progress": "robot_steps_before_terminal",
    "pedestrian_deviation": "tracked_pedestrian_steps_with_baseline",
}
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
        metric_denominator: Declared denominator for the metric family.
        estimand: Estimand type (paired mean difference per episode).
        effect_definition: Plain-language definition of the paired effect.
        direction: Metric direction convention (lower_is_better).
        reporting_strata: Dimensions reported beside the primary effect.
        support_policy: Rule for available, unavailable, and invalid rows.
        resampling_unit: Bootstrap resampling unit (paired seed block).
        ci_method: Confidence interval method (paired bootstrap percentile CI95).
        multiplicity_family: Multiplicity correction family (Holm-Bonferroni).
    """

    decision_id: str
    comparator_a: str
    comparator_b: str
    metric_family: str
    metric_denominator: str
    estimand: str
    effect_definition: str
    direction: str
    reporting_strata: tuple[str, ...]
    support_policy: str
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
                    metric_denominator=METRIC_FAMILY_DENOMINATORS[metric_family],
                    estimand="paired_mean_difference_per_episode",
                    effect_definition=(
                        f"Mean over episodes of (per-episode {metric_family} value for "
                        f"{comparator_b} minus {comparator_a}), paired by seed and "
                        "restricted to rows where the metric family is available for both "
                        "planners in the pair."
                    ),
                    direction=METRIC_DIRECTION,
                    reporting_strata=("metric_family", "scenario_family"),
                    support_policy=(
                        "Report support and denominators; exclude fallback, degraded, missing, "
                        "invalid, or unavailable rows without zero-imputation."
                    ),
                    resampling_unit=RESAMPLING_UNIT,
                    ci_method="paired_bootstrap_percentile_ci95",
                    multiplicity_family=MULTIPLICITY_FAMILY,
                )
            )
    return estimands


def _require_contract(condition: bool, message: str) -> None:
    """Raise a clear error when a frozen preregistration field has drifted."""
    if not condition:
        raise ValueError(message)


def _file_sha256(path: Path) -> str:
    """Return the SHA-256 digest for a tracked preregistration surface."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _selected_scenario_content_sha256() -> str:
    """Return the canonical hash of the frozen #6102 scenario payloads."""
    scenarios = load_scenario_matrix(CLASSIC_SCENARIO_MATRIX_PATH)
    by_name = {str(row.get("name")): row for row in scenarios}
    missing = [name for name in SCENARIOS if name not in by_name]
    _require_contract(not missing, f"frozen scenarios missing from #6102 source: {missing}")
    payload = [by_name[name] for name in SCENARIOS]
    canonical_json = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical_json).hexdigest()


def _metric_contract_denominators() -> dict[str, str]:
    """Read the versioned metric contract as a family-to-denominator mapping."""
    payload = yaml.safe_load(METRIC_CONTRACT_PATH.read_text(encoding="utf-8"))
    _require_contract(isinstance(payload, dict), "metric contract must be a mapping")
    metrics = payload.get("metrics")
    _require_contract(isinstance(metrics, list), "metric contract must declare a metrics list")

    denominators: dict[str, str] = {}
    for metric in metrics:
        _require_contract(isinstance(metric, dict), "metric contract entries must be mappings")
        family = metric.get("family")
        denominator = metric.get("denominator")
        _require_contract(
            isinstance(family, str) and isinstance(denominator, str),
            "metric contract entries must declare string family and denominator values",
        )
        denominators[family] = denominator
    return denominators


def validate_preregistration_contract(manifest: dict[str, Any]) -> None:
    """Fail closed when the checked-in preregistration surfaces no longer agree.

    This intentionally reads only tracked configs and evidence. It does not load
    campaign output, execute a benchmark, or submit work.
    """
    campaign = load_campaign_config(CAMPAIGN_CONFIG_PATH)
    preflight = load_campaign_config(PREFLIGHT_CONFIG_PATH)
    matrix_rows = load_scenario_matrix(SCENARIO_MATRIX_PATH)
    matrix_names = tuple(str(row.get("name")) for row in matrix_rows)
    evidence = json.loads(PREREGISTRATION_PATH.read_text(encoding="utf-8"))
    _require_contract(isinstance(evidence, dict), "preregistration evidence must be an object")

    _require_contract(
        tuple(planner.key for planner in campaign.planners) == PLANNERS, "planner drift"
    )
    _require_contract(campaign.seed_policy.seeds == SEEDS, "paired seed policy drift")
    _require_contract(matrix_names == SCENARIOS, "scenario matrix selection or order drift")
    _require_contract(
        campaign.scenario_matrix_path == SCENARIO_MATRIX_PATH, "scenario matrix path drift"
    )
    _require_contract(
        (campaign.horizon, campaign.dt)
        == (preflight.horizon, preflight.dt)
        == (HORIZON_STEPS, DT_SECONDS),
        "horizon or dt drifted from the #6481 preflight",
    )
    _require_contract(
        (campaign.paper_facing, campaign.export_publication_bundle) == (False, False),
        "campaign must remain non-paper-facing and not export a publication bundle",
    )

    provenance = evidence.get("provenance")
    design = evidence.get("design")
    multiplicity = evidence.get("multiplicity")
    estimands = evidence.get("estimands")
    _require_contract(isinstance(provenance, dict), "preregistration provenance is missing")
    _require_contract(isinstance(design, dict), "preregistration design is missing")
    _require_contract(isinstance(multiplicity, dict), "preregistration multiplicity is missing")
    _require_contract(isinstance(estimands, dict), "preregistration estimands are missing")
    _require_contract(evidence.get("status") == "preregistration_only", "evidence status drift")
    _require_contract(
        evidence.get("evidence_status") == "not_benchmark_evidence", "evidence class drift"
    )
    _require_contract(
        provenance.get("config_sha256") == _file_sha256(CAMPAIGN_CONFIG_PATH), "config hash drift"
    )
    _require_contract(
        provenance.get("scenario_matrix_sha256") == _file_sha256(SCENARIO_MATRIX_PATH),
        "scenario matrix hash drift",
    )
    _require_contract(
        provenance.get("scenario_content_sha256") == _selected_scenario_content_sha256(),
        "selected scenario content hash drift",
    )
    _require_contract(tuple(design.get("planners", ())) == PLANNERS, "evidence planner drift")
    _require_contract(tuple(design.get("seeds", ())) == SEEDS, "evidence seed drift")
    _require_contract(tuple(design.get("scenarios", ())) == SCENARIOS, "evidence scenario drift")
    _require_contract(
        design.get("expected_cell_count") == EXPECTED_CELL_COUNT, "evidence cell count drift"
    )
    _require_contract(
        multiplicity.get("family") == MULTIPLICITY_FAMILY, "multiplicity family drift"
    )
    _require_contract(
        multiplicity.get("num_decisions") == NUM_DECISIONS, "multiplicity count drift"
    )
    _require_contract(
        estimands.get("support_and_denominators") is not None,
        "preregistration must declare support and denominator handling",
    )
    _require_contract(
        _metric_contract_denominators() == METRIC_FAMILY_DENOMINATORS,
        "metric-family denominator contract drift",
    )
    _require_contract(
        manifest["scenarios"] == list(SCENARIOS), "manifest scenario declaration drift"
    )
    _require_contract(
        manifest["metric_family_denominators"] == METRIC_FAMILY_DENOMINATORS,
        "manifest denominator declaration drift",
    )


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
        "scenarios": list(SCENARIOS),
        "scenario_families": list(SCENARIO_FAMILIES),
        "metric_families": list(METRIC_FAMILIES),
        "metric_family_denominators": dict(METRIC_FAMILY_DENOMINATORS),
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
        "reporting_strata": ["metric_family", "scenario_family"],
        "support_and_denominators": (
            "Report support counts and denominators beside every estimand. Exclude fallback, "
            "degraded, missing, invalid, or unavailable rows without zero-imputation."
        ),
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
        help=(
            "Validate the frozen config, scenario, metric, evidence, and multiplicity contracts "
            "without executing a campaign."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Emit the preregistration-only estimand manifest without executing any campaign.

    Returns:
        0 when the manifest is emitted or the frozen contract check passes.
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
        try:
            validate_preregistration_contract(manifest)
        except (KeyError, OSError, TypeError, ValueError, yaml.YAMLError) as error:
            print(f"ERROR: preregistration contract validation failed: {error}", file=sys.stderr)
            return 1
        print(
            "OK: "
            f"{len(manifest['estimands'])} estimands / "
            f"{manifest['num_multiplicity_decisions']} multiplicity decisions "
            "(frozen config, scenario, metric, and evidence contracts agree; no campaign executed)."
        )
        return 0
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
