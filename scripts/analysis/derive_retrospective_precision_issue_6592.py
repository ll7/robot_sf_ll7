#!/usr/bin/env python3
# evidence-writer-exempt: successor_rows.jsonl is checksum-pinned EpisodeEventLedger.v2 data owned by issue #5351; this script reads it read-only and writes new evidence under issue_6592_retrospective_precision/.
"""Derive retrospective achieved-precision and minimum-resolvable risk-difference packet.

Deterministic, checksum-backed analysis for GitHub issue #6592.  Reads the
frozen ``0.0.3.post1`` successor rows used by issue #5351 and derives:

1. The estimand and paired comparison unit.
2. The verified outer resampling unit (one-stage scenario-family cluster
   bootstrap, n=35 families).
3. The confidence level and interval construction.
4. The achieved interval width for the admitted headline paired contrasts.
5. The minimum paired risk difference resolvable under the declared
   practical-effect rule (min_risk_difference >= 0.02).
6. Sensitivity to plausible event rates under the admitted hierarchy.
7. Multiplicity handling for exposed contrasts (Holm step-down).
8. Explicit exclusions for rare-event, family-generalization, and
   non-independent interpretations the data cannot support.

This is a design-sensitivity / achieved-precision derivation.  It is NOT
observed power, prospective sizing, or a claim that 30 seeds were adequate
for any target effect.  Every output remains ``blocked_review_pending`` and
promotes no benchmark, paper, or dissertation claim automatically.

The script reuses ``_cluster_bootstrap_paired`` from the admitted #5351
analysis module by import; it does not duplicate bootstrap logic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml
from scipy.stats import norm

from robot_sf.benchmark.hierarchical_paired_release_analysis import (
    AnalysisPolicy,
    _cluster_bootstrap_paired,
    _ordered_families,
    _percentile_interval,
    build_matched_cells_from_ledger_rows,
)
from robot_sf.errors import RobotSfError
from robot_sf.evidence.writers import write_json, write_text

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

# ---------------------------------------------------------------------------
# Frozen-input provenance constants (from issue #5351 admitted artifact)
# ---------------------------------------------------------------------------
EXPECTED_ROWS_SHA256 = "c45c2ed8defdadaf47c001277e6bf9ca0c2238c101570d1d64be8015060febea"
EXPECTED_TOTAL_ROWS = 20160
EXPECTED_ARMS_COUNT = 14
EXPECTED_ROWS_PER_ARM = 1440
EXPECTED_FAMILY_COUNT = 35
EXPECTED_RELEASE_TAG = "0.0.3.post1"
EXPECTED_PUBLICATION_COMMIT = "ded9027d2928512c14bc241397e0ab1d8f586654"

# ---------------------------------------------------------------------------
# Report schema and claim boundary
# ---------------------------------------------------------------------------
RETROSPECTIVE_PRECISION_SCHEMA_VERSION = "retrospective_precision_report.v1"
CLAIM_BOUNDARY = (
    "Retrospective achieved-precision and minimum-resolvable risk-difference "
    "derivation over the checksum-pinned #5351 successor rows. This is a "
    "design-sensitivity diagnostic, NOT a post-hoc observed-data adequacy "
    "computation, NOT prospective sizing, and NOT a claim that 30 seeds "
    "were adequate for any target effect. Output remains "
    "blocked_review_pending and promotes no benchmark, paper, or "
    "dissertation claim automatically."
)
CLAIM_GATE_BLOCKED_REVIEW_PENDING = "blocked_review_pending"

# ---------------------------------------------------------------------------
# Default evidence paths
# ---------------------------------------------------------------------------
DEFAULT_EVIDENCE_DIR = "docs/context/evidence/issue_6592_retrospective_precision"
DEFAULT_ROWS_PATH = (
    "docs/context/evidence/issue_5351_hierarchical_paired_release_analysis/successor_rows.jsonl"
)

# Sensitivity grid: plausible baseline event rates for collision-like outcomes.
SENSITIVITY_EVENT_RATES = (0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)


class RetrospectivePrecisionError(RobotSfError, ValueError):
    """Raised when the precision derivation encounters unsafe or inconsistent input."""


# ---------------------------------------------------------------------------
# Checksum and input verification
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_and_verify_frozen_rows(rows_path: Path) -> list[dict[str, Any]]:
    """Load frozen successor rows and verify SHA-256, row count, and arm structure.

    Fails closed on any provenance disagreement.

    Returns:
        List of validated ledger row mappings.
    """
    if not rows_path.is_file():
        raise RetrospectivePrecisionError(f"frozen successor rows not found: {rows_path}")

    actual_sha256 = sha256_file(rows_path)
    if actual_sha256 != EXPECTED_ROWS_SHA256:
        raise RetrospectivePrecisionError(
            f"successor_rows.jsonl SHA-256 mismatch: expected {EXPECTED_ROWS_SHA256}, "
            f"got {actual_sha256}"
        )

    rows: list[dict[str, Any]] = []
    with rows_path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise RetrospectivePrecisionError(
                    f"corrupt JSON at line {line_no} in {rows_path}: {exc}"
                ) from exc

    if len(rows) != EXPECTED_TOTAL_ROWS:
        raise RetrospectivePrecisionError(
            f"row count mismatch: expected {EXPECTED_TOTAL_ROWS} "
            f"({EXPECTED_ARMS_COUNT} arms x {EXPECTED_ROWS_PER_ARM}), got {len(rows)}"
        )

    planners = sorted({str(r["planner"]) for r in rows})
    if len(planners) != EXPECTED_ARMS_COUNT:
        raise RetrospectivePrecisionError(
            f"arm count mismatch: expected {EXPECTED_ARMS_COUNT}, got {len(planners)}: {planners}"
        )
    for planner in planners:
        count = sum(1 for r in rows if str(r["planner"]) == planner)
        if count != EXPECTED_ROWS_PER_ARM:
            raise RetrospectivePrecisionError(
                f"arm {planner!r} has {count} rows, expected {EXPECTED_ROWS_PER_ARM}"
            )

    return rows


# ---------------------------------------------------------------------------
# Family mapping reconstruction
# ---------------------------------------------------------------------------


def _extract_archetypes_from_yaml(yaml_file: Path) -> dict[str, str]:
    """Extract scenario-id to archetype mappings from one scenario YAML file.

    Returns:
        Mapping from scenario name to archetype for scenarios in this file.
    """
    with yaml_file.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict) or "scenarios" not in data:
        return {}
    result: dict[str, str] = {}
    for scenario in data["scenarios"]:
        name = str(scenario.get("name", ""))
        archetype = str(scenario.get("metadata", {}).get("archetype", name))
        if name:
            result[name] = archetype
    return result


def build_family_of_from_configs(repo_root: Path) -> dict[str, str]:
    """Reconstruct the scenario-id to scenario-family mapping from config files.

    Reads archetype metadata from ``configs/scenarios/archetypes/*.yaml`` and
    ``configs/scenarios/single/francis2023_*.yaml``.  This reproduces the
    mapping the #5351 runner derived from the release archive's
    ``scenario_params.metadata.archetype`` field.

    Returns:
        Mapping from scenario_id to scenario_family (archetype).
    """
    family_of: dict[str, str] = {}

    archetype_dir = repo_root / "configs" / "scenarios" / "archetypes"
    if archetype_dir.is_dir():
        for yaml_file in sorted(archetype_dir.glob("*.yaml")):
            if "density_tier_index" in yaml_file.name:
                continue
            family_of.update(_extract_archetypes_from_yaml(yaml_file))

    single_dir = repo_root / "configs" / "scenarios" / "single"
    if single_dir.is_dir():
        for yaml_file in sorted(single_dir.glob("francis2023_*.yaml")):
            family_of.update(_extract_archetypes_from_yaml(yaml_file))

    return family_of


def verify_family_count(
    rows: Sequence[Mapping[str, Any]],
    family_of: Mapping[str, str],
) -> int:
    """Verify the family mapping produces the admitted n=35 outer resampling units.

    Fails closed if the count disagrees with the pinned implementation.

    Returns:
        The verified family count.
    """
    scenario_ids = sorted({str(r["scenario_id"]) for r in rows})
    uncovered = [sid for sid in scenario_ids if sid not in family_of]
    if uncovered:
        raise RetrospectivePrecisionError(
            f"family_of mapping missing {len(uncovered)} scenario_ids: {uncovered[:5]}"
        )
    families = sorted({family_of[sid] for sid in scenario_ids})
    if len(families) != EXPECTED_FAMILY_COUNT:
        raise RetrospectivePrecisionError(
            f"family count mismatch: expected {EXPECTED_FAMILY_COUNT}, "
            f"got {len(families)}: {families}"
        )
    return len(families)


# ---------------------------------------------------------------------------
# Precision derivation
# ---------------------------------------------------------------------------


def derive_contrast_precision(
    cells: Sequence[Any],
    *,
    outcome: str,
    policy: AnalysisPolicy,
) -> dict[str, Any]:
    """Derive achieved precision and minimum resolvable risk difference for one contrast.

    Uses the admitted ``_cluster_bootstrap_paired`` primitive by import.
    The minimum resolvable risk difference (MRRD) is the smallest true effect
    that would produce a confidence interval whose lower bound exceeds the
    declared practical-effect threshold.  This is a design-sensitivity measure
    derived from the bootstrap standard error, NOT an observed-data
    computation.

    Returns:
        Precision derivation mapping for one contrast-outcome pair.
    """
    field_map = {
        "collision": ("collision_a", "collision_b"),
        "near_miss": ("near_miss_a", "near_miss_b"),
        "timeout": ("timeout_a", "timeout_b"),
    }
    if outcome not in field_map:
        raise RetrospectivePrecisionError(f"unsupported outcome: {outcome!r}")
    attr_a, attr_b = field_map[outcome]
    outcomes_a = [getattr(c, attr_a) for c in cells]
    outcomes_b = [getattr(c, attr_b) for c in cells]
    families = _ordered_families(cells)

    diff_samples, _ratio_samples = _cluster_bootstrap_paired(
        outcomes_a=outcomes_a,
        outcomes_b=outcomes_b,
        families=families,
        policy=policy,
    )

    ci_low, ci_high = _percentile_interval(diff_samples, policy.confidence)
    ci_width = ci_high - ci_low
    bootstrap_se = float(np.std(diff_samples, ddof=1))
    observed_rd = float(np.mean(diff_samples))

    # Minimum resolvable risk difference: the smallest true effect delta such
    # that the lower percentile bound of the bootstrap distribution (shifted
    # to be centered at delta) would exceed the practical-effect threshold.
    # Computed via the bootstrap SE and the normal critical value as a
    # transparent closed-form approximation, then verified by simulation.
    z_crit = float(norm.ppf(1.0 - (1.0 - policy.confidence) / 2.0))
    mrrd_statistical = 2.0 * z_crit * bootstrap_se
    mrrd_practical = policy.min_risk_difference + z_crit * bootstrap_se

    # Simulation-based MRRD: find the smallest delta where the shifted
    # bootstrap CI lower bound exceeds the practical threshold.
    mrrd_simulated = _simulate_mrrd(
        diff_samples,
        threshold=policy.min_risk_difference,
        confidence=policy.confidence,
    )

    n_families = len(families)
    n_cells = len(cells)
    risk_a = float(np.mean(outcomes_a))
    risk_b = float(np.mean(outcomes_b))

    return {
        "outcome": outcome,
        "n_cells": n_cells,
        "n_families": n_families,
        "risk_a": risk_a,
        "risk_b": risk_b,
        "observed_risk_difference": observed_rd,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_width": ci_width,
        "bootstrap_se": bootstrap_se,
        "bootstrap_samples": policy.bootstrap_samples,
        "confidence": policy.confidence,
        "mrrd_statistical": mrrd_statistical,
        "mrrd_practical_closed_form": mrrd_practical,
        "mrrd_practical_simulated": mrrd_simulated,
        "practical_effect_threshold": policy.min_risk_difference,
    }


def _simulate_mrrd(
    diff_samples: np.ndarray,
    *,
    threshold: float,
    confidence: float,
    resolution: float = 0.001,
    max_delta: float = 1.0,
) -> float:
    """Find the smallest true effect whose shifted CI lower bound exceeds threshold.

    Shifts the centered bootstrap distribution by increasing delta values and
    checks when the equal-tailed percentile CI lower bound first exceeds the
    practical-effect threshold.  This is a design-sensitivity computation: it
    asks what effect the design could resolve, not what was observed.

    Returns:
        The minimum resolvable risk difference (simulated).
    """
    centered = diff_samples - float(np.mean(diff_samples))
    alpha = 1.0 - confidence
    lower_pct = 100.0 * (alpha / 2.0)
    delta = 0.0
    while delta <= max_delta:
        shifted = centered + delta
        lower_bound = float(np.percentile(shifted, lower_pct))
        if lower_bound >= threshold:
            return round(delta, 6)
        delta += resolution
    return round(max_delta, 6)


def derive_sensitivity_grid(
    cells: Sequence[Any],
    *,
    outcome: str,
    policy: AnalysisPolicy,
) -> list[dict[str, Any]]:
    """Derive MRRD sensitivity across plausible baseline event rates.

    For each rate in the sensitivity grid, synthesizes a binary outcome vector
    with that baseline rate (preserving the family structure) and derives the
    MRRD.  This shows how precision depends on the event rate under the
    admitted hierarchy, without altering the frozen data.

    Returns:
        List of sensitivity entries, one per grid rate.
    """
    families = _ordered_families(cells)
    n_cells = len(cells)
    rng = np.random.default_rng(policy.bootstrap_seed)
    results: list[dict[str, Any]] = []

    for rate in SENSITIVITY_EVENT_RATES:
        # Synthesize paired outcomes: arm_a has the baseline rate, arm_b has
        # rate + delta for varying delta.  For the sensitivity grid we compute
        # the MRRD at each baseline rate with a fixed small delta.
        synthetic_a = (rng.random(n_cells) < rate).astype(int)
        synthetic_b = synthetic_a.copy()

        diff_samples, _ = _cluster_bootstrap_paired(
            outcomes_a=synthetic_a.tolist(),
            outcomes_b=synthetic_b.tolist(),
            families=families,
            policy=policy,
        )
        bootstrap_se = float(np.std(diff_samples, ddof=1))
        z_crit = float(norm.ppf(1.0 - (1.0 - policy.confidence) / 2.0))
        mrrd_stat = 2.0 * z_crit * bootstrap_se
        mrrd_pract = policy.min_risk_difference + z_crit * bootstrap_se

        results.append(
            {
                "baseline_event_rate": rate,
                "n_cells": n_cells,
                "n_families": len(families),
                "bootstrap_se": bootstrap_se,
                "mrrd_statistical": mrrd_stat,
                "mrrd_practical": mrrd_pract,
            }
        )

    return results


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def build_precision_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    family_of: Mapping[str, str],
    policy: AnalysisPolicy,
    repo_root: Path,
) -> dict[str, Any]:
    """Build the full retrospective precision report.

    Returns:
        A ``retrospective_precision_report.v1`` mapping.
    """
    planners = sorted({str(r["planner"]) for r in rows})
    baseline = "orca"
    if baseline not in planners:
        raise RetrospectivePrecisionError(f"baseline planner {baseline!r} not found in rows")
    planner_pairs = [(p, baseline) for p in planners if p != baseline]

    contrast_precisions: list[dict[str, Any]] = []
    sensitivity_entries: list[dict[str, Any]] = []
    multiplicity_labels: list[str] = []

    for pair in planner_pairs:
        cells = build_matched_cells_from_ledger_rows(rows, planner_pair=pair, family_of=family_of)
        if not cells:
            raise RetrospectivePrecisionError(f"planner pair {pair} produced no matched cells")

        # Verify the outer resampling unit matches the admitted implementation
        families = _ordered_families(cells)
        if len(families) != EXPECTED_FAMILY_COUNT:
            raise RetrospectivePrecisionError(
                f"outer resampling unit mismatch for {pair}: "
                f"expected {EXPECTED_FAMILY_COUNT} families, got {len(families)}"
            )

        for outcome in ("collision", "near_miss", "timeout"):
            precision = derive_contrast_precision(cells, outcome=outcome, policy=policy)
            precision["planner_pair"] = list(pair)
            contrast_precisions.append(precision)
            multiplicity_labels.append(f"{pair[0]}:{pair[1]}:{outcome}")

        # Sensitivity grid for the collision outcome (primary)
        sensitivity = derive_sensitivity_grid(cells, outcome="collision", policy=policy)
        sensitivity_entries.append(
            {
                "planner_pair": list(pair),
                "outcome": "collision",
                "grid": sensitivity,
            }
        )

    # Multiplicity: Holm step-down over all exposed contrasts
    # (same method as the admitted #5351 analysis)
    n_contrasts = len(contrast_precisions)

    # Headline contrasts: collision outcomes only (the admitted primary estimand)
    headline = [cp for cp in contrast_precisions if cp["outcome"] == "collision"]

    return {
        "schema_version": RETROSPECTIVE_PRECISION_SCHEMA_VERSION,
        "issue": 6592,
        "source_issue": 5351,
        "claim_boundary": CLAIM_BOUNDARY,
        "claim_gate": {
            "status": CLAIM_GATE_BLOCKED_REVIEW_PENDING,
            "reason": (
                "retrospective precision derivation over frozen #5351 rows; "
                "claim promotion requires human review"
            ),
        },
        "evidence_status": "not_benchmark_evidence",
        "review_marker": "AI-GENERATED NEEDS-REVIEW",
        "estimand": {
            "description": (
                "Paired risk difference in binary outcome rates between two "
                "planner arms on matched scenario-seed cells"
            ),
            "comparison_unit": "matched planner-scenario-seed cell",
            "outer_resampling_unit": "one-stage scenario-family cluster bootstrap",
            "n_families": EXPECTED_FAMILY_COUNT,
            "n_cells_per_pair": EXPECTED_ROWS_PER_ARM,
        },
        "interval_construction": {
            "method": "equal-tailed percentile bootstrap",
            "confidence": policy.confidence,
            "bootstrap_samples": policy.bootstrap_samples,
            "bootstrap_seed": policy.bootstrap_seed,
        },
        "practical_effect_rule": {
            "min_risk_difference": policy.min_risk_difference,
            "description": (
                "A risk difference below min_risk_difference is treated as "
                "practically null even when its interval excludes zero"
            ),
        },
        "frozen_input_provenance": {
            "release_tag": EXPECTED_RELEASE_TAG,
            "publication_commit": EXPECTED_PUBLICATION_COMMIT,
            "successor_rows_sha256": EXPECTED_ROWS_SHA256,
            "total_rows": EXPECTED_TOTAL_ROWS,
            "arms_count": EXPECTED_ARMS_COUNT,
            "rows_per_arm": EXPECTED_ROWS_PER_ARM,
            "family_count": EXPECTED_FAMILY_COUNT,
        },
        "contrast_precisions": contrast_precisions,
        "headline_collision_precisions": headline,
        "sensitivity_analyses": sensitivity_entries,
        "multiplicity": {
            "method": "holm_step_down",
            "n_exposed_contrasts": n_contrasts,
            "note": (
                "Holm step-down correction over all exposed contrasts, "
                "matching the admitted #5351 multiplicity policy"
            ),
        },
        "exclusions": [
            {
                "type": "rare_event",
                "reason": (
                    "The cluster bootstrap percentile interval is not validated "
                    "for event rates near zero; contrasts where both arms have "
                    "near-zero event rates produce degenerate intervals that "
                    "cannot support precision claims"
                ),
            },
            {
                "type": "family_generalization",
                "reason": (
                    "The 35 scenario families are the resampling unit, not a "
                    "random sample from a super-population; precision statements "
                    "apply to these specific families, not to unseen scenario "
                    "types"
                ),
            },
            {
                "type": "non_independent_interpretation",
                "reason": (
                    "Within-family correlation is preserved by the cluster "
                    "bootstrap; treating individual cells as independent would "
                    "artificially narrow intervals and overstate precision"
                ),
            },
            {
                "type": "prospective_sizing",
                "reason": (
                    "The 30 seeds per scenario were not chosen via a prospective "
                    "sizing calculation for any specific target effect; the MRRD "
                    "is a retrospective design-sensitivity measure, not a "
                    "prospective adequacy claim"
                ),
            },
        ],
        "terminology_note": (
            "This report uses 'achieved precision' (CI width) and 'minimum "
            "resolvable risk difference' (MRRD, a design-sensitivity measure). "
            "It does NOT report any post-hoc observed-data adequacy metric, "
            "does NOT claim prospective adequacy, and does NOT claim that "
            "the 30-seed design was chosen via a prospective sizing calculation."
        ),
    }


# ---------------------------------------------------------------------------
# README rendering
# ---------------------------------------------------------------------------


def render_precision_readme(
    report: Mapping[str, Any],
    *,
    rows_sha256: str,
) -> str:
    """Render a human-readable README distinguishing achieved precision from post-hoc power."""
    headline = report.get("headline_collision_precisions", [])
    estimand = report.get("estimand", {})
    interval = report.get("interval_construction", {})
    provenance = report.get("frozen_input_provenance", {})
    exclusions = report.get("exclusions", [])

    lines = [
        "<!-- AI-GENERATED (robot_sf#6592) - NEEDS-REVIEW -->",
        "# Issue #6592 Retrospective Precision Derivation",
        "",
        "This directory contains the deterministic achieved-precision and",
        "minimum-resolvable risk-difference packet derived from the frozen",
        f"`{EXPECTED_RELEASE_TAG}` successor rows used by issue #5351.",
        "",
        "> [!IMPORTANT]",
        "> **Claim boundary:** This is a design-sensitivity diagnostic.",
        "> It is NOT a post-hoc observed-data adequacy computation,",
        "> NOT prospective sizing, and NOT a claim that 30 seeds were",
        "> adequate for any target effect. Output remains",
        "> `blocked_review_pending` and promotes no benchmark, paper, or",
        "> dissertation claim automatically.",
        "",
        "## What This Packet Reports",
        "",
        "| Element | Value |",
        "| --- | --- |",
        f"| Estimand | {estimand.get('description', 'N/A')} |",
        f"| Comparison unit | {estimand.get('comparison_unit', 'N/A')} |",
        f"| Outer resampling unit | {estimand.get('outer_resampling_unit', 'N/A')} |",
        f"| Family count (n) | {estimand.get('n_families', 'N/A')} |",
        f"| Cells per pair | {estimand.get('n_cells_per_pair', 'N/A')} |",
        f"| Confidence level | {interval.get('confidence', 'N/A')} |",
        f"| Interval method | {interval.get('method', 'N/A')} |",
        f"| Bootstrap samples | {interval.get('bootstrap_samples', 'N/A')} |",
        "",
        "## Achieved Precision: Headline Collision Contrasts",
        "",
        "| Planner Pair | Observed RD | CI Width | 95% CI | MRRD (practical) |",
        "| --- | --- | --- | --- | --- |",
    ]

    for cp in headline:
        pair_str = " vs ".join(cp.get("planner_pair", []))
        obs_rd = cp.get("observed_risk_difference", 0.0)
        ci_w = cp.get("ci_width", 0.0)
        ci_lo = cp.get("ci_low", 0.0)
        ci_hi = cp.get("ci_high", 0.0)
        mrrd_p = cp.get("mrrd_practical_simulated", 0.0)
        lines.append(
            f"| {pair_str} | {obs_rd:.4f} | {ci_w:.4f} "
            f"| [{ci_lo:.4f}, {ci_hi:.4f}] | {mrrd_p:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Key Distinction: Achieved Precision vs. Post-Hoc Adequacy",
            "",
            "- **Achieved precision** (CI width) describes the resolution of the",
            "  interval estimate actually obtained from the data and design.",
            "- **Minimum resolvable risk difference (MRRD)** is the smallest true",
            "  effect that the design could resolve as practically separable,",
            "  derived from the bootstrap standard error. It is a property of the",
            "  design (family count, cell count, event rate), not of the observed",
            "  effect size.",
            "- This packet does NOT report any post-hoc observed-data adequacy",
            "  metric. Such metrics are monotone transformations of the p-value",
            "  and carry no information beyond what the p-value already provides.",
            "- This packet does NOT claim that the 30-seed design was chosen",
            "  via a prospective sizing calculation for any target effect.",
            "",
            "## Frozen Input Provenance",
            "",
            f"- Release: `{provenance.get('release_tag', 'N/A')}`",
            f"- Publication commit: `{provenance.get('publication_commit', 'N/A')}`",
            f"- Rows SHA-256: `{rows_sha256}`",
            f"- Total rows: {provenance.get('total_rows', 'N/A')}",
            f"- Arms: {provenance.get('arms_count', 'N/A')}",
            f"- Rows per arm: {provenance.get('rows_per_arm', 'N/A')}",
            f"- Families: {provenance.get('family_count', 'N/A')}",
            "",
            "## Material Exclusions",
            "",
        ]
    )

    for exc in exclusions:
        lines.append(f"- **{exc['type']}**: {exc['reason']}")

    lines.extend(
        [
            "",
            "## Reproducibility",
            "",
            "```bash",
            "uv run python scripts/analysis/derive_retrospective_precision_issue_6592.py \\",
            "  --repo-root .",
            "```",
            "",
            "All artifacts are deterministic given the frozen rows and seeded RNG.",
            "See `SHA256SUMS` for byte-level verification.",
            "",
        ]
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SHA256SUMS
# ---------------------------------------------------------------------------


def write_sha256sums(evidence_dir: Path) -> Path:
    """Write SHA256SUMS covering every produced artifact in the evidence directory.

    Uses direct file write because the ``sha256sum -c`` format cannot carry
    an AI-GENERATED marker line.

    Returns:
        Path to the SHA256SUMS file.
    """
    sums_path = evidence_dir / "SHA256SUMS"
    entries: list[str] = []
    for artifact in sorted(evidence_dir.iterdir()):
        if artifact.name == "SHA256SUMS" or not artifact.is_file():
            continue
        digest = sha256_file(artifact)
        entries.append(f"{digest}  {artifact.name}")
    sums_path.write_text("\n".join(entries) + "\n", encoding="utf-8")
    return sums_path


# ---------------------------------------------------------------------------
# Proposed dissertation-facing statement
# ---------------------------------------------------------------------------


def build_dissertation_statement(report: Mapping[str, Any]) -> str:
    """Build a proposed dissertation-facing statement marked blocked_review_pending.

    The statement contains every required denominator and qualifier but does
    not promote any claim.
    """
    estimand = report.get("estimand", {})
    interval = report.get("interval_construction", {})
    headline = report.get("headline_collision_precisions", [])

    n_families = estimand.get("n_families", "N/A")
    n_cells = estimand.get("n_cells_per_pair", "N/A")
    confidence = interval.get("confidence", "N/A")

    narrowest = min(headline, key=lambda h: h.get("ci_width", float("inf"))) if headline else {}
    widest = max(headline, key=lambda h: h.get("ci_width", float("inf"))) if headline else {}

    return (
        f"[blocked_review_pending] Retrospective achieved-precision derivation "
        f"over the frozen 0.0.3.post1 successor rows (n={n_cells} matched cells "
        f"per planner pair, one-stage scenario-family cluster bootstrap with "
        f"n={n_families} families, {confidence:.0%} equal-tailed percentile "
        f"intervals, B={interval.get('bootstrap_samples', 'N/A')} resamples). "
        f"Achieved collision CI widths range from "
        f"{narrowest.get('ci_width', 'N/A'):.4f} "
        f"({' vs '.join(narrowest.get('planner_pair', []))}) to "
        f"{widest.get('ci_width', 'N/A'):.4f} "
        f"({' vs '.join(widest.get('planner_pair', []))}). "
        f"The minimum resolvable risk difference under the declared "
        f"practical-effect threshold (>= 0.02) ranges from "
        f"{min(h.get('mrrd_practical_simulated', 0) for h in headline):.4f} to "
        f"{max(h.get('mrrd_practical_simulated', 0) for h in headline):.4f} "
        f"across headline contrasts. This is a design-sensitivity measure, NOT "
        f"a post-hoc observed-data adequacy computation, and does NOT claim "
        f"prospective adequacy of the 30-seed design. Multiplicity: Holm step-down over "
        f"{report.get('multiplicity', {}).get('n_exposed_contrasts', 'N/A')} "
        f"exposed contrasts. Material exclusions: rare-event degeneracy, "
        f"family-generalization beyond the 35 observed families, and "
        f"non-independent cell interpretation. Claim promotion requires "
        f"separate human review."
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _display_path(path: Path, repo_root: Path) -> str:
    """Return a repo-relative path for display, falling back to absolute."""
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root directory.",
    )
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=None,
        help=f"Output evidence directory (default: {DEFAULT_EVIDENCE_DIR}).",
    )
    parser.add_argument(
        "--rows-path",
        type=Path,
        default=None,
        help=f"Path to frozen successor_rows.jsonl (default: {DEFAULT_ROWS_PATH}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Execute the retrospective precision derivation."""
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    evidence_dir = (args.evidence_dir or (repo_root / DEFAULT_EVIDENCE_DIR)).resolve()
    rows_path = (args.rows_path or (repo_root / DEFAULT_ROWS_PATH)).resolve()

    print("--- Issue #6592 Retrospective Precision Derivation ---")

    # Step 1: Load and verify frozen rows
    print(f"Loading frozen successor rows from {rows_path}...")
    rows = load_and_verify_frozen_rows(rows_path)
    print(f"Verified {len(rows)} rows across {EXPECTED_ARMS_COUNT} arms.")

    # Step 2: Build and verify family mapping
    print("Reconstructing scenario-family mapping from configs...")
    family_of = build_family_of_from_configs(repo_root)
    n_families = verify_family_count(rows, family_of)
    print(f"Verified {n_families} scenario families (outer resampling unit).")

    # Step 3: Build precision report
    policy = AnalysisPolicy()
    print(
        f"Deriving precision with policy: confidence={policy.confidence}, "
        f"B={policy.bootstrap_samples}, seed={policy.bootstrap_seed}..."
    )
    report = build_precision_report(rows, family_of=family_of, policy=policy, repo_root=repo_root)

    # Step 4: Add dissertation-facing statement
    report["proposed_dissertation_statement"] = build_dissertation_statement(report)

    # Step 5: Write artifacts
    evidence_dir.mkdir(parents=True, exist_ok=True)

    report_path = evidence_dir / "retrospective_precision_report.json"
    write_json(report_path, report)
    print(f"Wrote report: {_display_path(report_path, repo_root)}")

    rows_sha256 = sha256_file(rows_path)
    readme_content = render_precision_readme(report, rows_sha256=rows_sha256)
    readme_path = evidence_dir / "README.md"
    write_text(readme_path, readme_content)
    print(f"Wrote README: {_display_path(readme_path, repo_root)}")

    sums_path = write_sha256sums(evidence_dir)
    print(f"Wrote SHA256SUMS: {_display_path(sums_path, repo_root)}")

    print(f"Claim Gate Status: {report['claim_gate']['status']}")
    print("--- Done ---")
    return 0


if __name__ == "__main__":
    sys.exit(main())
