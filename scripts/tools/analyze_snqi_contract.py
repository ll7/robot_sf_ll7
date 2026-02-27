#!/usr/bin/env python3
"""Analyze SNQI behavior for a completed camera-ready campaign."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from robot_sf.benchmark.snqi.campaign_contract import (
    SnqiContractThresholds,
    calibrate_weights,
    collect_episodes_from_campaign_runs,
    compute_baseline_stats_from_episodes,
    compute_component_dominance,
    evaluate_snqi_contract,
    resolve_weight_mapping,
    sanitize_baseline_stats,
)
from robot_sf.benchmark.utils import load_optional_json
from robot_sf.common.artifact_paths import get_repository_root


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--baseline", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--trials", type=int, default=3000)
    parser.add_argument("--rank-warn-threshold", type=float, default=0.5)
    parser.add_argument("--rank-fail-threshold", type=float, default=0.3)
    parser.add_argument("--outcome-warn-threshold", type=float, default=0.05)
    parser.add_argument("--outcome-fail-threshold", type=float, default=0.0)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.rank_warn_threshold <= args.rank_fail_threshold:
        parser.error("--rank-warn-threshold must be greater than --rank-fail-threshold")
    if args.outcome_warn_threshold <= args.outcome_fail_threshold:
        parser.error("--outcome-warn-threshold must be greater than --outcome-fail-threshold")
    return args


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    dominance = payload.get("component_dominance", {})
    lines = [
        "# SNQI Diagnostics",
        "",
        f"- Contract status: `{payload.get('contract_status', 'unknown')}`",
        f"- Rank alignment (Spearman): `{payload.get('rank_alignment_spearman', 0.0):.4f}`",
        f"- Outcome separation: `{payload.get('outcome_separation', 0.0):.4f}`",
        f"- Objective score: `{payload.get('objective_score', 0.0):.4f}`",
        "",
        "## Component Dominance",
        "",
        "| Component | Mean |",
        "|---|---:|",
    ]
    if isinstance(dominance, dict):
        for key, value in sorted(dominance.items()):
            lines.append(f"| {key} | {float(value):.6f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(path: Path, payload: dict[str, Any]) -> None:
    headers = ("component", "configured_weight", "calibrated_weight", "delta")
    configured = payload.get("configured_weights") if isinstance(payload, dict) else {}
    calibrated = payload.get("calibrated_weights") if isinstance(payload, dict) else {}
    rows = [",".join(headers)]
    for component in sorted((configured or {}).keys()):
        configured_value = float((configured or {}).get(component, 0.0))
        calibrated_value = float((calibrated or {}).get(component, configured_value))
        delta = calibrated_value - configured_value
        rows.append(f"{component},{configured_value:.10f},{calibrated_value:.10f},{delta:.10f}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Run SNQI diagnostics for one campaign and write report artifacts."""
    args = _parse_args(argv)
    campaign_root = args.campaign_root.resolve()
    summary_path = campaign_root / "reports" / "campaign_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing campaign summary: {summary_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    planner_rows = (
        summary.get("planner_rows") if isinstance(summary.get("planner_rows"), list) else []
    )
    run_entries = summary.get("runs") if isinstance(summary.get("runs"), list) else []

    raw_weights = (
        load_optional_json(str(args.weights.resolve())) if args.weights is not None else None
    )
    raw_baseline = (
        load_optional_json(str(args.baseline.resolve())) if args.baseline is not None else None
    )
    configured_weights = resolve_weight_mapping(raw_weights)
    episodes = collect_episodes_from_campaign_runs(run_entries, repo_root=get_repository_root())
    if raw_baseline is None:
        baseline, warnings = compute_baseline_stats_from_episodes(episodes)
        baseline_source = "derived_from_campaign_episodes"
        baseline_adjustments = len(warnings)
    else:
        baseline, warnings = sanitize_baseline_stats(raw_baseline)
        baseline_source = "input_file"
        baseline_adjustments = len(warnings)

    thresholds = SnqiContractThresholds(
        rank_alignment_warn=args.rank_warn_threshold,
        rank_alignment_fail=args.rank_fail_threshold,
        outcome_separation_warn=args.outcome_warn_threshold,
        outcome_separation_fail=args.outcome_fail_threshold,
    )
    evaluation = evaluate_snqi_contract(
        planner_rows,
        episodes,
        weights=configured_weights,
        baseline=baseline,
        thresholds=thresholds,
    )
    calibration = calibrate_weights(
        planner_rows,
        episodes,
        baseline=baseline,
        seed=args.seed,
        trials=args.trials,
    )
    dominance = compute_component_dominance(
        episodes,
        weights=configured_weights,
        baseline=baseline,
    )

    payload = {
        "schema_version": "benchmark-snqi-diagnostics.v1",
        "campaign_root": str(campaign_root),
        "contract_status": evaluation.status,
        "rank_alignment_spearman": evaluation.rank_alignment_spearman,
        "outcome_separation": evaluation.outcome_separation,
        "objective_score": evaluation.objective_score,
        "thresholds": {
            "rank_alignment_warn": args.rank_warn_threshold,
            "rank_alignment_fail": args.rank_fail_threshold,
            "outcome_separation_warn": args.outcome_warn_threshold,
            "outcome_separation_fail": args.outcome_fail_threshold,
        },
        "baseline_source": baseline_source,
        "baseline_adjustments": baseline_adjustments,
        "baseline": baseline,
        "configured_weights": configured_weights,
        "calibrated_weights": calibration.get("weights", {}),
        "calibration": calibration,
        "component_dominance": dominance,
    }

    reports_dir = campaign_root / "reports"
    output_json = (
        args.output_json.resolve() if args.output_json else reports_dir / "snqi_diagnostics.json"
    )
    output_md = args.output_md.resolve() if args.output_md else reports_dir / "snqi_diagnostics.md"
    output_csv = (
        args.output_csv.resolve() if args.output_csv else reports_dir / "snqi_sensitivity.csv"
    )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    output_json.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    _write_markdown(output_md, payload)
    _write_csv(output_csv, payload)

    print(
        json.dumps(
            {
                "snqi_diagnostics_json": str(output_json),
                "snqi_diagnostics_md": str(output_md),
                "snqi_sensitivity_csv": str(output_csv),
                "contract_status": evaluation.status,
                "rank_alignment_spearman": evaluation.rank_alignment_spearman,
                "outcome_separation": evaluation.outcome_separation,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
