#!/usr/bin/env python3
"""Build the issue #3465 paired topology-gate decision and evidence packet.

This script loads the preregistration benchmark config and the campaign summary JSON.
It aggregates the metrics, computes the safety and efficiency improvements (enabled - disabled),
checks paired statistical significance, detects fallback/degraded rows, calls the near-parity
promotion gate classifier, and writes the required evidence artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robot_sf.benchmark.near_parity_promotion_gate import (
    NearParityComparison,
    classify_near_parity_promotion,
)
from robot_sf.benchmark.topology_gate_paired_preregistration import (
    load_topology_gate_paired_config,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = "configs/benchmarks/issue_3465_topology_gate_paired.yaml"
DEFAULT_OUTPUT_DIR = "docs/context/evidence/issue_3465_topology_gate_paired"

ACCEPTANCE_EVIDENCE: tuple[dict[str, str], ...] = (
    {
        "criterion": "Paired enabled/disabled config exists and is reproducible.",
        "evidence": (
            "PR #4246 added configs/benchmarks/issue_3465_topology_gate_paired.yaml "
            "and the fail-closed preregistration validator."
        ),
        "status": "met",
    },
    {
        "criterion": "Same scenario/seed/planner contract is used across arms.",
        "evidence": (
            "PR #4246 validates topology_gate_disabled and topology_gate_enabled differ "
            "only by near_parity_diversity_gate_enabled while preserving planner, scenario, "
            "seed, and horizon pairing."
        ),
        "status": "met",
    },
    {
        "criterion": "Corrective #3463 gate completed or waived before paired interpretation.",
        "evidence": (
            "PR #4465 integrated corrective packets #4388, #4411, #4426, #4435, and #4444 "
            "into the readiness contract and reports ready_for_paired_run."
        ),
        "status": "met",
    },
    {
        "criterion": "Degraded or fallback rows are excluded or caveated.",
        "evidence": (
            "PR #4487 added the paired decision builder, excludes fallback/degraded arms "
            "from promotion evidence, and defaults missing paired_significant to False in real mode."
        ),
        "status": "met",
    },
    {
        "criterion": "Result classification is recorded conservatively.",
        "evidence": (
            "PR #3602 added near_parity_promotion_gate.v1. This packet remains blocked until "
            "a real campaign_summary.json exists; then it writes summary.json, paired_deltas.csv, "
            "and README.md with the conservative decision."
        ),
        "status": "blocked_pending_campaign",
    },
)

_SAFETY_ALIASES = ("collisions_mean", "collision_rate", "collision_rate_mean", "collisions")
_EFFICIENCY_ALIASES = ("path_efficiency_mean", "path_efficiency", "efficiency", "efficiency_mean")
_STATUS_ALIASES = ("readiness_status", "status", "execution_mode", "availability_status")

BLOCKED_STATUSES = {"fallback", "degraded", "failed", "not_available", "blocked"}


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _set_criterion_status(
    evidence: list[dict[str, str]], criterion_prefix: str, status: str
) -> None:
    """Update one acceptance criterion's status in place.

    The acceptance-evidence template carries static default statuses; the actual
    decision state (corrective-gate completion, campaign availability) must be
    reflected dynamically so the generated report never claims a criterion is met
    or blocked in a state where that is untrue.
    """

    for item in evidence:
        if item["criterion"].startswith(criterion_prefix):
            item["status"] = status
            return


def _get_metric(row: Mapping[str, Any], aliases: tuple[str, ...]) -> float | None:
    for alias in aliases:
        if alias in row and row[alias] is not None:
            try:
                val = float(row[alias])
                import math

                if math.isfinite(val):
                    return val
            except (TypeError, ValueError):
                continue
    return None


def _get_status(row: Mapping[str, Any], aliases: tuple[str, ...]) -> str | None:
    for alias in aliases:
        if alias in row and row[alias] is not None:
            return str(row[alias]).strip().lower()
    return None


def build_decision_report(  # noqa: C901
    config_path: Path,
    campaign_dir: Path,
    mock: bool = False,
    mock_paired_significant: bool = True,
    mock_relies_on_fallback: bool = False,
    mock_safety_imp: float = 0.05,
    mock_efficiency_imp: float = 0.05,
) -> dict[str, Any]:
    """Build the issue #3465 decision report."""

    config = load_topology_gate_paired_config(config_path)
    acceptance_evidence = [dict(item) for item in ACCEPTANCE_EVIDENCE]
    corrective_complete = config.get("readiness", {}).get("corrective_complete", False)

    if not corrective_complete:
        comparison = NearParityComparison(
            safety_improvement=0.0,
            efficiency_improvement=0.0,
            paired_significant=False,
            relies_on_fallback=False,
            corrective_complete=False,
        )
        verdict = classify_near_parity_promotion(comparison)
        _set_criterion_status(acceptance_evidence, "Corrective #3463", "not_met")
        return {
            "status": "blocked",
            "reason": "corrective_incomplete",
            "verdict": verdict,
            "blocked_reasons": ["Corrective implementation (#3463) is not complete."],
            "acceptance_evidence": acceptance_evidence,
            "config_path": _display_path(config_path),
        }

    # Load campaign results
    campaign_summary_path = campaign_dir / "campaign_summary.json"
    blockers: list[str] = []

    if mock:
        # Mock campaign rows
        disabled_collisions = 0.10
        enabled_collisions = disabled_collisions - mock_safety_imp
        disabled_efficiency = 0.90
        enabled_efficiency = disabled_efficiency + mock_efficiency_imp

        disabled_status = "fallback" if mock_relies_on_fallback else "ok"
        enabled_status = "ok"

        rows = [
            {
                "planner_key": "topology_gate_disabled",
                "collisions_mean": disabled_collisions,
                "path_efficiency_mean": disabled_efficiency,
                "status": disabled_status,
            },
            {
                "planner_key": "topology_gate_enabled",
                "collisions_mean": enabled_collisions,
                "path_efficiency_mean": enabled_efficiency,
                "status": enabled_status,
            },
        ]
    else:
        if not campaign_summary_path.exists():
            return {
                "status": "blocked",
                "reason": "campaign_summary_missing",
                "blocked_reasons": [
                    f"Campaign summary JSON not found at {_display_path(campaign_summary_path)}."
                ],
                "acceptance_evidence": acceptance_evidence,
                "config_path": _display_path(config_path),
            }

        try:
            with open(campaign_summary_path, encoding="utf-8") as f:
                data = json.load(f)
            rows = data.get("planner_rows", [])
        except (OSError, ValueError) as e:
            return {
                "status": "blocked",
                "reason": "campaign_summary_invalid",
                "blocked_reasons": [
                    f"Failed to read/parse {_display_path(campaign_summary_path)}: {e}"
                ],
                "acceptance_evidence": acceptance_evidence,
                "config_path": _display_path(config_path),
            }

    disabled_row = None
    enabled_row = None
    for row in rows:
        pk = row.get("planner_key")
        if pk == "topology_gate_disabled":
            disabled_row = row
        elif pk == "topology_gate_enabled":
            enabled_row = row

    if not disabled_row or not enabled_row:
        blockers.append(
            "Campaign summary missing planner keys topology_gate_disabled and/or topology_gate_enabled"
        )
        return {
            "status": "blocked",
            "reason": "arms_not_found",
            "blocked_reasons": blockers,
            "acceptance_evidence": acceptance_evidence,
            "config_path": _display_path(config_path),
        }

    # Extract metrics
    disabled_collisions = _get_metric(disabled_row, _SAFETY_ALIASES)
    enabled_collisions = _get_metric(enabled_row, _SAFETY_ALIASES)
    disabled_efficiency = _get_metric(disabled_row, _EFFICIENCY_ALIASES)
    enabled_efficiency = _get_metric(enabled_row, _EFFICIENCY_ALIASES)

    disabled_status = _get_status(disabled_row, _STATUS_ALIASES)
    enabled_status = _get_status(enabled_row, _STATUS_ALIASES)

    if disabled_collisions is None or enabled_collisions is None:
        blockers.append("Missing collision/safety metrics in campaign rows")
    if disabled_efficiency is None or enabled_efficiency is None:
        blockers.append("Missing efficiency/progress metrics in campaign rows")

    if blockers:
        return {
            "status": "blocked",
            "reason": "metrics_missing",
            "blocked_reasons": blockers,
            "acceptance_evidence": acceptance_evidence,
            "config_path": _display_path(config_path),
        }

    # Paired calculations
    safety_imp = disabled_collisions - enabled_collisions  # Positive is better (fewer collisions)
    efficiency_imp = enabled_efficiency - disabled_efficiency  # Positive is better

    relies_on_fallback = disabled_status in BLOCKED_STATUSES or enabled_status in BLOCKED_STATUSES

    # In mock mode, use the mock parameter. In real mode read the explicit
    # ``campaign.paired_significant`` field; default to False (fail-closed) so a
    # campaign summary that omits paired-significance evidence can never yield a
    # promotable verdict.
    paired_sig = (
        mock_paired_significant
        if mock
        else bool(data.get("campaign", {}).get("paired_significant", False))
    )

    comparison = NearParityComparison(
        safety_improvement=safety_imp,
        efficiency_improvement=efficiency_imp,
        paired_significant=paired_sig,
        relies_on_fallback=relies_on_fallback,
        corrective_complete=True,
    )

    verdict = classify_near_parity_promotion(comparison)

    # Read config metadata for provenance
    config_data = config_path.read_bytes()
    config_sha256 = hashlib.sha256(config_data).hexdigest()

    # Campaign loaded and classified: the conservative-classification criterion is now met.
    _set_criterion_status(acceptance_evidence, "Result classification is recorded", "met")

    return {
        "status": "ready",
        "verdict": verdict,
        "acceptance_evidence": acceptance_evidence,
        "config_provenance": {
            "path": _display_path(config_path),
            "sha256": config_sha256,
        },
        "arms": {
            "topology_gate_disabled": {
                "collisions_mean": disabled_collisions,
                "path_efficiency_mean": disabled_efficiency,
                "status": disabled_status,
            },
            "topology_gate_enabled": {
                "collisions_mean": enabled_collisions,
                "path_efficiency_mean": enabled_efficiency,
                "status": enabled_status,
            },
        },
        "deltas": {
            "safety_improvement": safety_imp,
            "efficiency_improvement": efficiency_imp,
            "paired_significant": paired_sig,
            "relies_on_fallback": relies_on_fallback,
        },
        "blocked_reasons": blockers,
    }


def write_decision_artifacts(report: Mapping[str, Any], output_dir: Path) -> None:
    """Write the decision artifacts (JSON summary, CSV deltas, and README.md)."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # Write summary.json
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")

    # Write paired_deltas.csv
    csv_path = output_dir / "paired_deltas.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["metric", "disabled_val", "enabled_val", "delta"])
        if report.get("status") == "ready":
            deltas = report["deltas"]
            arms = report["arms"]
            disabled = arms["topology_gate_disabled"]
            enabled = arms["topology_gate_enabled"]
            writer.writerow(
                [
                    "collisions_mean",
                    disabled["collisions_mean"],
                    enabled["collisions_mean"],
                    -deltas["safety_improvement"],
                ]
            )
            writer.writerow(
                [
                    "path_efficiency_mean",
                    disabled["path_efficiency_mean"],
                    enabled["path_efficiency_mean"],
                    deltas["efficiency_improvement"],
                ]
            )

    # Write README.md
    readme_path = output_dir / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(format_markdown_report(report))


def format_markdown_report(report: Mapping[str, Any]) -> str:
    """Format the markdown report summary."""

    verdict_info = report.get("verdict", {})
    verdict = verdict_info.get("verdict", "blocked")
    rationale = verdict_info.get("rationale", "Campaign data is not available or checks failed.")

    lines = [
        "# Issue #3465 — Near-Parity Promotion Gate Decision Report",
        "",
        f"**Decision Status:** `{report.get('status', 'blocked')}`",
        f"**Verdict:** `{verdict}`",
        "",
        "## Rationale",
        "",
        f"{rationale}",
        "",
    ]

    if report.get("status") == "ready":
        deltas = report["deltas"]
        arms = report["arms"]
        disabled = arms["topology_gate_disabled"]
        enabled = arms["topology_gate_enabled"]

        lines.extend(
            [
                "## Paired Metrics Comparison",
                "",
                "| Metric | Disabled Arm | Enabled Arm | Delta (Enabled - Disabled) |",
                "| --- | ---: | ---: | ---: |",
                f"| Collisions Mean | {disabled['collisions_mean']:.4f} | {enabled['collisions_mean']:.4f} | {-deltas['safety_improvement']:.4f} |",
                f"| Path Efficiency Mean | {disabled['path_efficiency_mean']:.4f} | {enabled['path_efficiency_mean']:.4f} | {deltas['efficiency_improvement']:.4f} |",
                "",
                "## Decision Inputs",
                "",
                f"- Safety Improvement: `{deltas['safety_improvement']:.4f}` (disabled - enabled; positive is safer)",
                f"- Efficiency Improvement: `{deltas['efficiency_improvement']:.4f}` (enabled - disabled; positive is better)",
                f"- Paired Significant: `{deltas['paired_significant']}`",
                f"- Relies on Fallback/Degraded: `{deltas['relies_on_fallback']}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Acceptance Evidence",
            "",
            "| Criterion | Status | Evidence |",
            "| --- | --- | --- |",
        ]
    )
    for item in report.get("acceptance_evidence", []):
        lines.append(f"| {item['criterion']} | `{item['status']}` | {item['evidence']} |")
    if not report.get("acceptance_evidence"):
        lines.append("| none | `missing` | No acceptance evidence recorded. |")
    lines.append("")

    lines.append("## Blockers")
    lines.append("")
    blockers = report.get("blocked_reasons", [])
    for blocker in blockers:
        lines.append(f"- {blocker}")
    if not blockers:
        lines.append("- none")
    lines.append("")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Tracked preregistration YAML.")
    parser.add_argument(
        "--campaign-dir",
        default="output/benchmarks/issue_3465_topology_gate_paired",
        help="Campaign output directory containing campaign_summary.json.",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for decision artifacts.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Build a mock report for testing.",
    )
    parser.add_argument(
        "--mock-safety-imp",
        type=float,
        default=0.05,
        help="Mock safety improvement value.",
    )
    parser.add_argument(
        "--mock-efficiency-imp",
        type=float,
        default=0.05,
        help="Mock efficiency improvement value.",
    )
    parser.add_argument(
        "--mock-not-significant",
        action="store_true",
        help="Set mock significance to False.",
    )
    parser.add_argument(
        "--mock-fallback",
        action="store_true",
        help="Set mock relies_on_fallback to True.",
    )
    return parser.parse_args()


def main() -> int:
    """Run decision build and exit."""

    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path

    campaign_dir = Path(args.campaign_dir)
    if not campaign_dir.is_absolute():
        campaign_dir = REPO_ROOT / campaign_dir

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir

    report = build_decision_report(
        config_path=config_path,
        campaign_dir=campaign_dir,
        mock=args.mock,
        mock_paired_significant=not args.mock_not_significant,
        mock_relies_on_fallback=args.mock_fallback,
        mock_safety_imp=args.mock_safety_imp,
        mock_efficiency_imp=args.mock_efficiency_imp,
    )

    write_decision_artifacts(report, out_dir)
    print(f"Decision artifacts written to {out_dir}")
    print(f"Status: {report.get('status')}")
    if report.get("status") == "ready":
        print(f"Verdict: {report.get('verdict', {}).get('verdict')}")
    else:
        print(f"Blocked Reasons: {report.get('blocked_reasons')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
