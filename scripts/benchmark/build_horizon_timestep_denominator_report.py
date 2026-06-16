#!/usr/bin/env python3
"""Build a denominator-health report for the horizon x timestep ablation.

The report is analysis-only: it inspects the cell-level unevaluable data from the
horizon/timestep ablation, classifies why each (trace, horizon, dt_s) cell is
missing, verifies the category totals sum to the expected matrix size, and
proposes the minimum fixture additions needed to evaluate at least 90% of the
matrix.  It does not change forecast defaults or claim navigation benefit.

Usage::

    uv run python scripts/benchmark/build_horizon_timestep_denominator_report.py \
        --output-dir docs/context/evidence/issue_2903_horizon_denominator_health_2026-06-16
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

# Reuse the ablation ladder and builder from the parent ablation report.
_ABLATION_SCRIPT = REPO_ROOT / "scripts/benchmark/build_horizon_timestep_ablation_report.py"

CLAIM_BOUNDARY = (
    "analysis_only_not_navigation_evidence: this report measures denominator coverage of the "
    "horizon x timestep ablation matrix. It does not change forecast defaults, prove navigation "
    "value, closed-loop benefit, safety improvement, or benchmark-strength predictor quality."
)

# Missingness categories required by the issue contract.
MISSINGNESS_CATEGORIES: tuple[str, ...] = (
    "trace_too_short",
    "no_pedestrian_motion",
    "metadata_missing",
    "actor_class_missing",
    "observation_tier_missing",
    "other_explicit_reason",
)

# Coverage target mandated by the issue contract.
COVERAGE_TARGET_FRACTION = 0.90


def _load_ablation_module() -> Any:
    """Import the parent ablation report builder as a module."""
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "build_horizon_timestep_ablation_report", _ABLATION_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load ablation script: {_ABLATION_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["build_horizon_timestep_ablation_report"] = mod
    spec.loader.exec_module(mod)
    return mod


_ablation_mod = _load_ablation_module()
HORIZON_LADDER_S: tuple[float, ...] = _ablation_mod.HORIZON_LADDER_S
DT_LADDER_S: tuple[float, ...] = _ablation_mod.DT_LADDER_S
TRACE_CANDIDATES: list[dict[str, Any]] = _ablation_mod.TRACE_CANDIDATES
build_ablation_report = _ablation_mod.build_ablation_report


def _classify_missingness(cell: dict[str, Any]) -> str:
    """Map an ablation cell status to a contract-required missingness reason."""
    status = cell.get("status", "")
    if status == "evaluated":
        return "evaluated"
    if status in {"horizon_longer_than_trace", "insufficient_frames"}:
        return "trace_too_short"
    if status == "limited_no_pedestrian_motion":
        return "no_pedestrian_motion"
    if status == "trace_file_missing":
        return "metadata_missing"
    if status == "actor_class_missing":
        return "actor_class_missing"
    if status == "observation_tier_missing":
        return "observation_tier_missing"
    return "other_explicit_reason"


def _reason_detail(cell: dict[str, Any]) -> str:
    """Return a human-readable detail string for a missing cell."""
    limitation = cell.get("limitation", "")
    status = cell.get("status", "")
    if status == "horizon_longer_than_trace":
        return limitation or "Requested horizon exceeds the resampled trace length."
    if status == "insufficient_frames":
        return limitation or "Resampled trace has fewer than 3 frames."
    if status == "limited_no_pedestrian_motion":
        return limitation or "Trace contains no pedestrian motion."
    if status == "trace_file_missing":
        return limitation or "Durable trace fixture is not present in the repository."
    return limitation or f"Status: {status}"


def _matrix_expected_total() -> int:
    """Total number of cells in the horizon x dt x trace matrix."""
    return len(HORIZON_LADDER_S) * len(DT_LADDER_S) * len(TRACE_CANDIDATES)


def _build_matrix_coverage(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Roll cells up to (horizon_s, dt_s) matrix rows with coverage fractions."""
    grouped: dict[tuple[float, float], list[dict[str, Any]]] = {}
    for row in rows:
        key = (round(row["horizon_s"], 6), round(row["requested_dt_s"], 6))
        grouped.setdefault(key, []).append(row)

    coverage_rows: list[dict[str, Any]] = []
    for (horizon_s, dt_s), cells in sorted(grouped.items()):
        by_reason: dict[str, int] = dict.fromkeys(MISSINGNESS_CATEGORIES, 0)
        by_reason["evaluated"] = 0
        for cell in cells:
            cat = _classify_missingness(cell)
            by_reason[cat] = by_reason.get(cat, 0) + 1
        total = len(cells)
        evaluated = by_reason.get("evaluated", 0)
        coverage_rows.append(
            {
                "horizon_s": horizon_s,
                "dt_s": dt_s,
                "total_cells": total,
                "evaluated_cells": evaluated,
                "evaluated_fraction": evaluated / total if total else 0.0,
                "missing_by_reason": {k: v for k, v in by_reason.items() if k != "evaluated"},
            }
        )
    return coverage_rows


def _build_category_totals(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    """Count missing cells by category across the whole matrix."""
    totals: dict[str, int] = dict.fromkeys(MISSINGNESS_CATEGORIES, 0)
    evaluated = 0
    for cell in rows:
        cat = _classify_missingness(cell)
        if cat == "evaluated":
            evaluated += 1
        else:
            totals[cat] = totals.get(cat, 0) + 1
    totals["evaluated"] = evaluated
    return totals


def _spot_check_missing_cells(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Pick one representative missing cell per category when possible."""
    checks: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    for cell in rows:
        cat = _classify_missingness(cell)
        if cat == "evaluated" or cat in seen:
            continue
        seen.add(cat)
        checks[cat] = {
            "family": cell["family"],
            "label": cell["label"],
            "scenario_id": cell.get("scenario_id", ""),
            "planner_id": cell.get("planner_id", ""),
            "horizon_s": cell["horizon_s"],
            "requested_dt_s": cell["requested_dt_s"],
            "actual_dt_s": cell["actual_dt_s"],
            "status": cell["status"],
            "reason": cat,
            "detail": _reason_detail(cell),
        }
    return checks


def _per_family_missingness(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Report missing-cell counts per trace family/label."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["family"], row["label"])
        grouped.setdefault(key, []).append(row)

    family_rows: list[dict[str, Any]] = []
    for (family, label), cells in sorted(grouped.items()):
        totals: dict[str, int] = dict.fromkeys(MISSINGNESS_CATEGORIES, 0)
        totals["evaluated"] = 0
        for cell in cells:
            cat = _classify_missingness(cell)
            totals[cat] = totals.get(cat, 0) + 1
        family_rows.append(
            {
                "family": family,
                "label": label,
                "total_cells": len(cells),
                "evaluated_cells": totals.pop("evaluated", 0),
                "missing_by_reason": totals,
            }
        )
    return family_rows


def _fixture_id(cell: dict[str, Any]) -> str:
    """Return a stable fixture identifier for repair planning."""
    return f"{cell['family']}/{cell['label']}"


def _compute_fixture_proposal(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Propose minimum fixture additions to reach 90% evaluated coverage.

    The proposal is computed from the observed gaps:

    * ``no_pedestrian_motion`` families need motion-rich replacements.
    * ``trace_too_short`` families need longer traces that cover the missing
      horizons at the requested output timesteps.

    The returned proposal states the count of fixtures to change, the expected
    cell gain, and the resulting coverage fraction.  It is intentionally a
    planning estimate, not a claim that the proposed fixtures already exist.
    """
    total = _matrix_expected_total()
    currently_evaluated = sum(1 for r in rows if _classify_missingness(r) == "evaluated")
    needed = max(0, int(total * COVERAGE_TARGET_FRACTION) - currently_evaluated)

    # Identify unique fixtures blocked by no pedestrian motion.
    no_motion_families: set[str] = set()
    no_motion_fixtures: set[str] = set()
    no_motion_cells = 0
    for cell in rows:
        if _classify_missingness(cell) == "no_pedestrian_motion":
            no_motion_families.add(cell["family"])
            no_motion_fixtures.add(_fixture_id(cell))
            no_motion_cells += 1

    # Identify unique motion-rich fixtures blocked by trace length.
    short_families: set[str] = set()
    short_fixtures: set[str] = set()
    short_cells = 0
    for cell in rows:
        if _classify_missingness(cell) == "trace_too_short":
            short_families.add(cell["family"])
            short_fixtures.add(_fixture_id(cell))
            short_cells += 1

    proposed_evaluated = currently_evaluated + no_motion_cells + short_cells

    # Minimum subset to reach 90%.
    # Heuristic: replace all no-motion fixtures, then extend short fixtures until
    # the target is met.  Report the smallest number of short fixtures that must
    # be extended.
    gain_from_replacements = no_motion_cells

    # Gains per extended short fixture are not uniform because each fixture misses
    # a different number of cells. Sort by descending number of missing cells.
    fixture_missing_counts: dict[str, int] = {}
    for cell in rows:
        if _classify_missingness(cell) == "trace_too_short":
            fixture = _fixture_id(cell)
            fixture_missing_counts[fixture] = fixture_missing_counts.get(fixture, 0) + 1
    sorted_short_fixtures = sorted(
        fixture_missing_counts.keys(), key=lambda f: (-fixture_missing_counts[f], f)
    )

    selected_short_fixtures: list[str] = []
    accumulated = gain_from_replacements
    for fixture in sorted_short_fixtures:
        if accumulated >= needed:
            break
        selected_short_fixtures.append(fixture)
        accumulated += fixture_missing_counts[fixture]

    minimum_additions = len(no_motion_fixtures) + len(selected_short_fixtures)
    estimated_gain = gain_from_replacements + sum(
        fixture_missing_counts[f] for f in selected_short_fixtures
    )
    estimated_coverage = (currently_evaluated + estimated_gain) / total

    return {
        "coverage_target_fraction": COVERAGE_TARGET_FRACTION,
        "current_evaluated_cells": currently_evaluated,
        "current_evaluated_fraction": currently_evaluated / total,
        "target_evaluated_cells": int(total * COVERAGE_TARGET_FRACTION),
        "needed_additional_cells": needed,
        "no_motion_families": sorted(no_motion_families),
        "no_motion_fixtures": sorted(no_motion_fixtures),
        "no_motion_blocked_cells": no_motion_cells,
        "short_families": sorted(short_families),
        "short_fixtures": sorted(short_fixtures),
        "short_blocked_cells": short_cells,
        "full_extension_evaluated_estimate": proposed_evaluated,
        "full_extension_fraction": proposed_evaluated / total,
        "minimum_fixture_additions": minimum_additions,
        "minimum_fixture_changes": {
            "replace_no_motion_families": sorted(no_motion_families),
            "replace_no_motion_fixtures": sorted(no_motion_fixtures),
            "extend_short_fixtures": selected_short_fixtures,
        },
        "minimum_additional_cells_estimate": estimated_gain,
        "minimum_coverage_estimate": estimated_coverage,
        "note": (
            "Estimates assume replacements are motion-rich and extensions cover the "
            "missing horizons. Actual coverage depends on the generated fixtures."
        ),
    }


def _generate_markdown(report: dict[str, Any]) -> str:
    """Render the denominator-health report as Markdown."""
    repro = report["reproducibility"]
    totals = report["category_totals"]
    matrix = report["matrix_coverage"]
    proposal = report["fixture_proposal"]

    lines = [
        "# Horizon x Timestep Denominator Health Report",
        "",
        "## Claim Boundary",
        "",
        f"**{CLAIM_BOUNDARY}**",
        "",
        "## Reproducibility",
        "",
        f"- **Issue:** #{report['issue']}",
        f"- **Parent ablation issue:** #{repro['parent_issue']}",
        f"- **Generated at (UTC):** {repro['generated_at_utc']}",
        f"- **Command:** `{repro['command']}`",
        f"- **Repo HEAD:** `{repro['repo_head']}`",
        f"- **Horizon ladder (s):** {repro['horizon_ladder_s']}",
        f"- **dt ladder (s):** {repro['dt_ladder_s']}",
        f"- **Trace families:** {repro['trace_family_count']}",
        f"- **Expected total cells:** {report['expected_total_cells']}",
        "",
        "## Category Totals",
        "",
        "| category | count | fraction |",
        "| --- | ---: | ---: |",
    ]
    for cat, count in totals.items():
        frac = count / report["expected_total_cells"]
        lines.append(f"| {cat} | {count} | {frac:.1%} |")

    lines.extend(
        [
            "",
            "## Matrix Coverage (horizon_s x dt_s)",
            "",
            "| horizon_s | dt_s | evaluated/total | fraction | trace_too_short | no_pedestrian_motion | "
            "metadata_missing | actor_class_missing | observation_tier_missing | other_explicit_reason |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in matrix:
        mb = row["missing_by_reason"]
        lines.append(
            f"| {row['horizon_s']:g} | {row['dt_s']:g} | "
            f"{row['evaluated_cells']}/{row['total_cells']} | "
            f"{row['evaluated_fraction']:.1%} | "
            f"{mb.get('trace_too_short', 0)} | "
            f"{mb.get('no_pedestrian_motion', 0)} | "
            f"{mb.get('metadata_missing', 0)} | "
            f"{mb.get('actor_class_missing', 0)} | "
            f"{mb.get('observation_tier_missing', 0)} | "
            f"{mb.get('other_explicit_reason', 0)} |"
        )

    lines.extend(
        [
            "",
            "## Per-Family Missingness",
            "",
            "| family | label | total | evaluated | trace_too_short | no_pedestrian_motion | "
            "metadata_missing | actor_class_missing | observation_tier_missing | other_explicit_reason |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in report["per_family_missingness"]:
        mb = row["missing_by_reason"]
        lines.append(
            f"| {row['family']} | {row['label']} | {row['total_cells']} | "
            f"{row['evaluated_cells']} | "
            f"{mb.get('trace_too_short', 0)} | "
            f"{mb.get('no_pedestrian_motion', 0)} | "
            f"{mb.get('metadata_missing', 0)} | "
            f"{mb.get('actor_class_missing', 0)} | "
            f"{mb.get('observation_tier_missing', 0)} | "
            f"{mb.get('other_explicit_reason', 0)} |"
        )

    lines.extend(
        [
            "",
            "## Spot Checks (one per missingness category)",
            "",
        ]
    )
    if report["spot_checks"]:
        lines.append("| reason | family | label | horizon_s | dt_s | status | detail |")
        lines.append("| --- | --- | --- | ---: | ---: | --- | --- |")
        for reason, cell in report["spot_checks"].items():
            lines.append(
                f"| {reason} | {cell['family']} | {cell['label']} | "
                f"{cell['horizon_s']:g} | {cell['actual_dt_s']:g} | {cell['status']} | "
                f"{cell['detail']} |"
            )
    else:
        lines.append("No missing cells to spot-check.")

    lines.extend(
        [
            "",
            "## Minimum Fixture Additions for 90% Coverage",
            "",
            f"- **Current coverage:** {proposal['current_evaluated_cells']}/{report['expected_total_cells']} "
            f"({proposal['current_evaluated_fraction']:.1%})",
            f"- **Target coverage:** {proposal['target_evaluated_cells']}/{report['expected_total_cells']} "
            f"({proposal['coverage_target_fraction']:.0%})",
            f"- **Additional cells needed:** {proposal['needed_additional_cells']}",
            f"- **No-motion families to replace:** {proposal['no_motion_families']} "
            f"({proposal['no_motion_blocked_cells']} blocked cells)",
            f"- **Short families to extend:** {proposal['short_families']} "
            f"({proposal['short_blocked_cells']} blocked cells)",
            f"- **No-motion fixtures to replace:** {proposal['no_motion_fixtures']}",
            f"- **Short fixtures observed:** {proposal['short_fixtures']}",
            "",
            "### Minimum Set Estimate",
            "",
            f"- **Fixture changes required:** {proposal['minimum_fixture_additions']}",
            f"- **Replace these no-motion fixtures:** {proposal['minimum_fixture_changes']['replace_no_motion_fixtures']}",
            f"- **Extend these short fixtures:** {proposal['minimum_fixture_changes']['extend_short_fixtures']}",
            f"- **Estimated additional cells:** {proposal['minimum_additional_cells_estimate']}",
            f"- **Estimated coverage:** {proposal['minimum_coverage_estimate']:.1%}",
            "",
            "### Full-Extension Estimate",
            "",
            f"- Replacing all no-motion families and extending all short families to cover the full "
            f"horizon ladder would yield approximately {proposal['full_extension_evaluated_estimate']} "
            f"evaluated cells ({proposal['full_extension_fraction']:.1%}).",
            "",
            f"> {proposal['note']}",
            "",
            "## Interpretation",
            "",
            "This report is a denominator-health audit for the horizon x timestep ablation.  "
            "Forecast defaults are explicitly not changed by this report alone.  "
            "The missing cells are dominated by short traces and by traces without pedestrian "
            "motion; no metadata, actor-class, or observation-tier gaps were observed in the "
            "current durable fixture set.  The proposed fixture additions are planning estimates "
            "and must be validated by actually generating or extending the relevant traces.",
        ]
    )
    return "\n".join(lines)


def build_denominator_report(
    parent_issue: int = 2837,
    issue: int = 2903,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    """Build the denominator-health report from the durable ablation fixtures."""
    repo_head = _ablation_mod._git_head()
    generated_at = generated_at_utc or datetime.datetime.now(datetime.UTC).isoformat()

    # Build the parent ablation report from durable fixtures.
    ablation_report = build_ablation_report(
        issue=parent_issue,
        generated_at_utc=generated_at,
    )
    rows = ablation_report["ablation_rows"]

    expected_total = _matrix_expected_total()
    category_totals = _build_category_totals(rows)
    matrix_coverage = _build_matrix_coverage(rows)
    spot_checks = _spot_check_missing_cells(rows)
    per_family = _per_family_missingness(rows)
    proposal = _compute_fixture_proposal(rows)

    # Verify category totals sum to the expected matrix size.
    total_from_categories = sum(category_totals.values())
    totals_valid = total_from_categories == expected_total

    command = (
        "uv run python scripts/benchmark/build_horizon_timestep_denominator_report.py "
        f"--issue {issue} --parent-issue {parent_issue}"
    )
    if generated_at_utc:
        command += f" --generated-at-utc {generated_at_utc}"

    repro = {
        "parent_issue": parent_issue,
        "issue": issue,
        "generated_at_utc": generated_at,
        "command": command,
        "repo_head": repo_head,
        "horizon_ladder_s": list(HORIZON_LADDER_S),
        "dt_ladder_s": list(DT_LADDER_S),
        "trace_family_count": len(TRACE_CANDIDATES),
    }

    return {
        "issue": issue,
        "parent_issue": parent_issue,
        "schema_version": "HorizonTimestepDenominatorHealth.v1",
        "claim_boundary": CLAIM_BOUNDARY,
        "forecast_defaults_unchanged": True,
        "reproducibility": repro,
        "expected_total_cells": expected_total,
        "category_totals": category_totals,
        "category_totals_valid": totals_valid,
        "matrix_coverage": matrix_coverage,
        "per_family_missingness": per_family,
        "spot_checks": spot_checks,
        "fixture_proposal": proposal,
        "ablation_rows": rows,
    }


def main() -> None:
    """Run the denominator-health audit and write evidence artifacts."""
    parser = argparse.ArgumentParser(
        description="Build horizon x timestep denominator-health report."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="docs/context/evidence/issue_2903_horizon_denominator_health_2026-06-16",
        help="Output directory for evidence artifacts.",
    )
    parser.add_argument(
        "--issue",
        type=int,
        default=2903,
        help="Issue number to record in generated evidence metadata.",
    )
    parser.add_argument(
        "--parent-issue",
        type=int,
        default=2837,
        help="Parent ablation issue number.",
    )
    parser.add_argument(
        "--generated-at-utc",
        help="Optional deterministic ISO-8601 generation timestamp for reviewable artifacts.",
    )
    args = parser.parse_args()

    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    report = build_denominator_report(
        parent_issue=args.parent_issue,
        issue=args.issue,
        generated_at_utc=args.generated_at_utc,
    )

    json_path = output_dir / "denominator_report.json"
    with open(json_path, "w") as fh:
        json.dump(report, fh, indent=2)

    md_path = output_dir / "denominator_report.md"
    with open(md_path, "w") as fh:
        fh.write(_generate_markdown(report))

    summary = {
        "issue": args.issue,
        "parent_issue": args.parent_issue,
        "schema_version": "HorizonTimestepDenominatorHealth.v1",
        "status": "diagnostic-only",
        "claim_boundary": CLAIM_BOUNDARY,
        "forecast_defaults_unchanged": True,
        "generated_at_utc": report["reproducibility"]["generated_at_utc"],
        "provenance": {
            "command": report["reproducibility"]["command"],
            "commit": report["reproducibility"]["repo_head"],
            "source": "durable repository trace fixtures via issue_2837 ablation",
        },
        "coverage": {
            "expected_total_cells": report["expected_total_cells"],
            "evaluated_cells": report["category_totals"]["evaluated"],
            "evaluated_fraction": report["category_totals"]["evaluated"]
            / report["expected_total_cells"],
            "category_totals_valid": report["category_totals_valid"],
            "minimum_fixture_additions": report["fixture_proposal"]["minimum_fixture_additions"],
            "minimum_coverage_estimate": report["fixture_proposal"]["minimum_coverage_estimate"],
        },
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {summary_path}")
    print(
        json.dumps(
            {
                "expected_total_cells": summary["coverage"]["expected_total_cells"],
                "evaluated_cells": summary["coverage"]["evaluated_cells"],
                "evaluated_fraction": summary["coverage"]["evaluated_fraction"],
                "category_totals_valid": summary["coverage"]["category_totals_valid"],
                "minimum_fixture_additions": summary["coverage"]["minimum_fixture_additions"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
