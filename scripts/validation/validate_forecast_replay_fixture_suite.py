#!/usr/bin/env python3
"""Validate the issue #3146 scenario-diverse forecast replay fixture suite."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.live_forecast_replay_gate import (
    FORECAST_VARIANTS,
    LiveForecastReplayGateConfig,
    LiveForecastReplayGateError,
    load_trace_tolerant,
    run_live_forecast_replay_gate,
)
from robot_sf.common.artifact_paths import get_repository_root

DEFAULT_MANIFEST = (
    get_repository_root() / "configs/benchmarks/issue_3146_forecast_replay_fixture_suite.yaml"
)
DELTA_METRICS = (
    "collision",
    "near_miss",
    "min_distance_m",
    "progress_m",
    "false_positive_stops",
    "stop_yield_timing_steps",
)
SIGNATURE_METRICS = (
    "collision",
    "near_miss",
    "min_distance_m",
    "progress_m",
    "false_positive_stops",
    "stop_yield_timing_steps",
)


def _git_head() -> str | None:
    """Return the current repository HEAD short sha, or None when unavailable."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10.0,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        return None


def _load_manifest(path: Path) -> dict[str, Any]:
    """Load the YAML suite manifest."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("suite manifest must be a YAML mapping")
    return payload


def _repo_relative(path: Path, repo_root: Path) -> str:
    """Return a repository-relative path when possible."""

    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _metric_delta(value: Any, baseline: Any) -> Any:
    """Return a compact same-seed delta for comparable metric values."""

    if isinstance(value, bool) or isinstance(baseline, bool):
        return {"baseline": baseline, "value": value, "changed": value != baseline}
    if isinstance(value, int | float) and isinstance(baseline, int | float):
        return value - baseline
    return {"baseline": baseline, "value": value, "changed": value != baseline}


def _same_seed_deltas(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Compute per-variant closed-loop metric deltas against the none variant."""

    variant_results = report.get("variant_results", {})
    none_metrics = (
        variant_results.get("none", {}).get("closed_loop_metrics", {})
        if isinstance(variant_results, dict)
        else {}
    )
    deltas: dict[str, dict[str, Any]] = {}
    for variant, result in variant_results.items():
        if variant == "none":
            continue
        metrics = result.get("closed_loop_metrics", {}) if isinstance(result, dict) else {}
        deltas[variant] = {
            metric: _metric_delta(metrics.get(metric), none_metrics.get(metric))
            for metric in DELTA_METRICS
        }
    return deltas


def _non_none_closed_loop_signature_count(report: dict[str, Any]) -> int:
    """Count distinct non-none closed-loop metric signatures in one report."""

    signatures = set()
    for variant, result in report.get("variant_results", {}).items():
        if variant == "none":
            continue
        metrics = result.get("closed_loop_metrics", {}) if isinstance(result, dict) else {}
        signatures.add(tuple(metrics.get(metric) for metric in SIGNATURE_METRICS))
    return len(signatures)


def _summarize_fixture(
    fixture: dict[str, Any],
    report: dict[str, Any],
    trace_path: Path,
) -> dict[str, Any]:
    """Build one compact fixture row from a gate report."""

    provenance = report.get("provenance", {})
    classification = str(report.get("classification", "blocked"))
    return {
        "fixture_id": fixture["fixture_id"],
        "scenario_family": fixture["scenario_family"],
        "trace": trace_path.as_posix(),
        "trace_id": provenance.get("trace_id"),
        "scenario_id": provenance.get("scenario_id"),
        "seed": provenance.get("seed"),
        "execution_mode": classification,
        "row_classification": classification,
        "classification_reason": report.get("classification_reason"),
        "variant_results": {
            variant: {
                "closed_loop_metric_source": result.get("closed_loop_metric_source"),
                "forecast_metrics_status": result.get("forecast_metrics_status"),
                "closed_loop_replay_error": result.get("closed_loop_replay_error"),
                "replay_policy_params": result.get("replay_policy_params"),
                "closed_loop_metrics": result.get("closed_loop_metrics"),
            }
            for variant, result in report.get("variant_results", {}).items()
        },
        "same_seed_deltas": _same_seed_deltas(report),
        "non_none_closed_loop_signature_count": _non_none_closed_loop_signature_count(report),
        "caveats": list(report.get("limitations", [])),
    }


def _blocked_fixture_row(fixture: dict[str, Any], trace_path: Path, error: str) -> dict[str, Any]:
    """Build a fail-closed row for an unavailable fixture."""

    return {
        "fixture_id": fixture["fixture_id"],
        "scenario_family": fixture["scenario_family"],
        "trace": trace_path.as_posix(),
        "trace_id": None,
        "scenario_id": fixture.get("expected_scenario_id"),
        "seed": fixture.get("expected_seed"),
        "execution_mode": "blocked",
        "row_classification": "blocked",
        "classification_reason": error,
        "variant_results": {},
        "same_seed_deltas": {},
        "non_none_closed_loop_signature_count": 0,
        "caveats": [error],
    }


def _validate_summary(manifest: dict[str, Any], rows: list[dict[str, Any]]) -> list[str]:
    """Return manifest-contract violations for the suite summary."""

    contract = manifest.get("summary_contract", {})
    errors: list[str] = []
    if len(rows) < int(contract.get("required_fixture_count", 0)):
        errors.append("not enough fixture rows")

    families = {row["scenario_family"] for row in rows}
    if len(families) < int(contract.get("required_distinct_scenario_families", 0)):
        errors.append("not enough distinct scenario families")

    required_variant_count = int(contract.get("required_variant_count", 0))
    expected_modes = {
        fixture.get("fixture_id"): fixture.get("expected_execution_mode")
        for fixture in manifest.get("fixtures", [])
    }
    for row in rows:
        for field in contract.get("required_fields", []):
            if field not in row:
                errors.append(f"{row.get('fixture_id', '<unknown>')} missing field {field}")
        if (
            row["row_classification"] != "blocked"
            and len(row["variant_results"]) != required_variant_count
        ):
            errors.append(f"{row['fixture_id']} missing full variant matrix")
        expected_mode = expected_modes.get(row["fixture_id"])
        if expected_mode and row["execution_mode"] != expected_mode:
            errors.append(
                f"{row['fixture_id']} expected {expected_mode}, observed {row['execution_mode']}"
            )
    native_signature_counts = [
        row["non_none_closed_loop_signature_count"]
        for row in rows
        if row["row_classification"] == "native"
    ]
    required_signature_count = int(contract.get("required_native_non_none_signature_count", 0))
    if native_signature_counts and max(native_signature_counts) < required_signature_count:
        errors.append("native rows do not distinguish non-none forecast variants")
    return errors


def run_suite(
    manifest_path: Path,
    *,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    """Run the fixture suite and return a compact summary."""

    repo_root = get_repository_root()
    manifest = _load_manifest(manifest_path)
    variants = tuple(manifest.get("variants", FORECAST_VARIANTS))
    config = LiveForecastReplayGateConfig()
    rows: list[dict[str, Any]] = []

    for fixture in manifest.get("fixtures", []):
        trace_path = repo_root / fixture["trace"]
        if not trace_path.exists():
            rows.append(_blocked_fixture_row(fixture, Path(fixture["trace"]), "trace not found"))
            continue
        try:
            trace = load_trace_tolerant(trace_path)
            report = run_live_forecast_replay_gate(
                trace,
                config=config,
                variants=variants,
                repo_head=_git_head(),
                generated_at_utc=generated_at_utc,
            )
            rows.append(_summarize_fixture(fixture, report, Path(fixture["trace"])))
        except (LiveForecastReplayGateError, OSError, TypeError, ValueError) as exc:
            rows.append(_blocked_fixture_row(fixture, Path(fixture["trace"]), str(exc)))

    row_status_counts = Counter(row["row_classification"] for row in rows)
    summary = {
        "schema_version": "forecast-replay-fixture-suite-summary.v0.1",
        "issue": manifest.get("issue"),
        "manifest": _repo_relative(manifest_path, repo_root),
        "claim_boundary": manifest.get("claim_boundary"),
        "repo_head": _git_head(),
        "generated_at_utc": generated_at_utc,
        "fixture_count": len(rows),
        "scenario_family_count": len({row["scenario_family"] for row in rows}),
        "variants": list(variants),
        "row_status_summary": dict(sorted(row_status_counts.items())),
        "rows": rows,
        "validation_errors": _validate_summary(manifest, rows),
        "interpretation": (
            "Diagnostic smoke evidence only: the suite exercises full forecast variants across "
            "scenario-diverse replay fixtures and records native/degraded classifications, but "
            "does not establish benchmark-strength or paper-grade claims."
        ),
    }
    summary["status"] = "passed" if not summary["validation_errors"] else "failed"
    return summary


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    """Write a short Markdown companion for the JSON summary."""

    lines = [
        "# Issue #3146 Forecast Replay Fixture Suite",
        "",
        f"- Status: `{summary['status']}`",
        "- Evidence status: diagnostic smoke only",
        f"- Fixture count: {summary['fixture_count']}",
        f"- Scenario families: {summary['scenario_family_count']}",
        f"- Variants: {', '.join(summary['variants'])}",
        f"- Row status summary: `{json.dumps(summary['row_status_summary'], sort_keys=True)}`",
        "",
        "This summary does not claim planner superiority, safety improvement, or paper-grade evidence.",
        "",
        "## Rows",
        "",
    ]
    for row in summary["rows"]:
        lines.extend(
            [
                f"### {row['fixture_id']}",
                "",
                f"- Family: `{row['scenario_family']}`",
                f"- Scenario: `{row['scenario_id']}`",
                f"- Seed: `{row['seed']}`",
                f"- Classification: `{row['row_classification']}`",
                f"- Reason: {row['classification_reason']}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--generated-at-utc")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the suite and return a shell-friendly exit code."""

    args = build_arg_parser().parse_args(argv)
    summary = run_suite(args.manifest, generated_at_utc=args.generated_at_utc)
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_markdown(summary, args.output_dir / "README.md")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":  # pragma: no cover - CLI guard
    raise SystemExit(main())
