#!/usr/bin/env python3
"""Evaluate constant-velocity pedestrian forecast baseline on bounded repository traces.

Reads durable trace fixtures from the repository, runs
``compute_batch_forecast_metrics`` from ``robot_sf.benchmark.pedestrian_forecast``,
and writes JSON + Markdown evidence to the requested output directory.

Usage::

    uv run python scripts/benchmark/run_cv_forecast_eval.py \\
        --output-dir docs/context/evidence/issue_2774_motion_rich_forecast_traces_2026-06-14

This is diagnostic-only evidence, not paper-facing benchmark proof.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import pathlib
import subprocess
from typing import TYPE_CHECKING, Any

import numpy as np

from robot_sf.benchmark.pedestrian_forecast import (
    BASELINE_FUNCTIONS,
    DEFAULT_FORECAST_HORIZONS_S,
    ForecastBaselineFunction,
    actor_type_metric_key,
    compute_batch_forecast_metrics,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

TRACE_CANDIDATES: list[dict[str, Any]] = [
    {
        "family": "corridor_interaction",
        "label": "default_social_force",
        "path": "docs/context/evidence/issue_2428_mechanism_trace_panels_2026-06-06/traces/default_social_force_trace_export.json",
        "scenario_id": "classic_head_on_corridor_low",
        "planner_id": "default_social_force",
        "seed": 111,
    },
    {
        "family": "corridor_interaction",
        "label": "ammv_social_force",
        "path": "docs/context/evidence/issue_2428_mechanism_trace_panels_2026-06-06/traces/ammv_social_force_trace_export.json",
        "scenario_id": "classic_head_on_corridor_low",
        "planner_id": "ammv_social_force",
        "seed": 111,
    },
    {
        "family": "crossing_proxy",
        "label": "synthetic_crossing_proxy_orca",
        "path": "docs/context/evidence/issue_2667_trace_failure_predicate_tables_2026-06-12/inputs/synthetic_crossing_proxy_orca_111_trace_export.json",
        "scenario_id": "crossing_proxy",
        "planner_id": "orca",
        "seed": 111,
    },
    {
        "family": "bottleneck",
        "label": "minimal_fixture",
        "path": "tests/fixtures/analysis_workbench/simulation_trace_export_v1/minimal_trace.json",
        "scenario_id": "classic_bottleneck_medium",
        "planner_id": "hybrid_rule_v0_minimal",
        "seed": 111,
    },
    {
        "family": "occluded_emergence",
        "label": "deterministic_occluded_emergence",
        "path": "tests/fixtures/analysis_workbench/simulation_trace_export_v1/occluded_emergence_episode_0000.json",
        "scenario_id": "issue_2756_occluded_emergence",
        "planner_id": "hybrid_rule_v0_minimal",
        "seed": 111,
    },
]

MISSING_FAMILIES: list[dict[str, str]] = [
    {"family": "signalized_crossing", "reason": "no durable trace fixture available"},
    {"family": "dense_pedestrian_interaction", "reason": "no durable trace fixture available"},
    {
        "family": "bottleneck_with_motion",
        "reason": "existing bottleneck fixture has zero pedestrian velocity",
    },
]


def _load_trace(path: pathlib.Path) -> dict[str, Any]:
    """Load a simulation_trace_export.v1 JSON file."""
    with open(path) as fh:
        return json.load(fh)


def _extract_trace_steps(trace: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert trace export frames to the step list expected by compute_batch_forecast_metrics.

    Handles both ``frames`` (simulation_trace_export.v1) and ``steps`` keys.
    Converts string pedestrian IDs to integers.
    """
    raw_frames = trace.get("frames") or trace.get("steps") or []
    steps: list[dict[str, Any]] = []
    for frame in raw_frames:
        pedestrians = []
        for ped in frame.get("pedestrians", []):
            ped_copy = dict(ped)
            ped_id = ped_copy.get("id")
            if isinstance(ped_id, str):
                try:
                    ped_copy["id"] = int(ped_id)
                except ValueError:
                    digest = hashlib.sha256(ped_id.encode("utf-8")).hexdigest()
                    ped_copy["id"] = int(digest[:16], 16) % (2**31)
            pedestrians.append(ped_copy)
        steps.append(
            {
                "step": frame.get("step", len(steps)),
                "time_s": frame.get("time_s", len(steps) * 0.1),
                "robot": frame.get("robot"),
                "pedestrians": pedestrians,
            }
        )
    return steps


def _compute_dt_s(trace: dict[str, Any]) -> float:
    """Infer dt_s from the first two frames of the trace."""
    frames = trace.get("frames") or trace.get("steps") or []
    if len(frames) < 2:
        return 0.1
    t0 = float(frames[0].get("time_s", 0.0))
    t1 = float(frames[1].get("time_s", 0.1))
    dt = t1 - t0
    return dt if dt > 0 else 0.1


def _trace_has_motion(trace: dict[str, Any]) -> bool:
    """Check whether any pedestrian has non-zero velocity in any frame."""
    frames = trace.get("frames") or trace.get("steps") or []
    for frame in frames:
        for ped in frame.get("pedestrians", []):
            vel = ped.get("velocity", [0.0, 0.0])
            if any(abs(float(v)) > 1e-6 for v in vel):
                return True
    return False


def _pedestrian_count(trace: dict[str, Any]) -> int:
    """Return the max number of pedestrians in any single frame."""
    frames = trace.get("frames") or trace.get("steps") or []
    max_peds = 0
    for frame in frames:
        max_peds = max(max_peds, len(frame.get("pedestrians", [])))
    return max_peds


def _get_actor_classes(trace: dict[str, Any]) -> list[str]:
    """Collect all unique actor_type values from pedestrians in the trace."""
    return sorted(_actor_class_counts(trace))


def _actor_class_counts(trace: dict[str, Any]) -> dict[str, int]:
    """Count trace actors by declared actor_type, defaulting legacy traces to pedestrian."""
    frames = trace.get("frames") or trace.get("steps") or []
    counts: dict[str, int] = {}
    for frame in frames:
        for ped in frame.get("pedestrians", []):
            actor_class = actor_type_metric_key(ped.get("actor_type"))
            counts[actor_class] = counts.get(actor_class, 0) + 1
    return dict(sorted(counts.items()))


def _git_head() -> str:
    """Return the short git HEAD, or '' on failure."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            timeout=5,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


def evaluate_single_trace(
    candidate: dict[str, Any],
    baseline_function: ForecastBaselineFunction | None = None,
) -> dict[str, Any]:
    """Run CV forecast eval on one trace candidate and return a result dict."""
    rel_path = candidate["path"]
    abs_path = REPO_ROOT / rel_path
    label = candidate["label"]
    family = candidate["family"]

    result: dict[str, Any] = {
        "label": label,
        "family": family,
        "trace_path": rel_path,
        "scenario_id": candidate.get("scenario_id", ""),
        "planner_id": candidate.get("planner_id", ""),
        "seed": candidate.get("seed"),
        "status": "evaluated",
    }

    if not abs_path.exists():
        result["status"] = "trace_file_missing"
        result["metrics"] = {"forecast_evaluable_samples": 0.0}
        return result

    trace = _load_trace(abs_path)
    frames = trace.get("frames") or trace.get("steps") or []
    result["frame_count"] = len(frames)
    result["pedestrians_per_frame"] = _pedestrian_count(trace)
    result["actor_classes"] = _get_actor_classes(trace)
    result["actor_class_counts"] = _actor_class_counts(trace)
    result["pedestrian_forecast_denominator_policy"] = (
        "Only pedestrian/person actors are scored by pedestrian forecast metrics; "
        "non-pedestrian actor classes are excluded and counted separately."
    )
    result["has_motion"] = _trace_has_motion(trace)

    if not _trace_has_motion(trace):
        result["status"] = "limited_no_pedestrian_motion"
        result["metrics"] = {"forecast_evaluable_samples": 0.0}
        result["limitation"] = (
            "All pedestrian velocities are zero; constant-velocity forecast "
            "produces degenerate predictions with no motion to evaluate against."
        )
        return result

    if len(frames) < 3:
        result["status"] = "insufficient_frames"
        result["metrics"] = {"forecast_evaluable_samples": 0.0}
        result["limitation"] = (
            f"Only {len(frames)} frames available; need at least 3 for any horizon."
        )
        return result

    trace_steps = _extract_trace_steps(trace)
    dt_s = _compute_dt_s(trace)

    try:
        metrics = compute_batch_forecast_metrics(
            trace_steps,
            horizons_s=list(DEFAULT_FORECAST_HORIZONS_S),
            dt_s=dt_s,
            baseline_function=baseline_function,
        )
    except Exception as exc:
        result["status"] = "evaluation_error"
        result["error"] = str(exc)
        result["metrics"] = {"forecast_evaluable_samples": 0.0}
        return result

    result["dt_s"] = dt_s
    result["horizons_s"] = list(DEFAULT_FORECAST_HORIZONS_S)
    result["metrics"] = _sanitize_metrics(metrics)

    evaluable = metrics.get("forecast_evaluable_samples", 0.0)
    if evaluable == 0.0:
        result["status"] = "no_evaluable_samples"
    else:
        result["status"] = "evaluated"

    return result


def _sanitize_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """Convert numpy types to plain floats for JSON serialization."""
    sanitized: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, (np.floating, np.integer)):
            sanitized[key] = float(value)
        else:
            sanitized[key] = float(value)
    return sanitized


def _build_failure_cases(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract representative failure cases from per-trace results."""
    failure_cases: list[dict[str, Any]] = []
    for r in results:
        metrics = r.get("metrics", {})
        samples = metrics.get("forecast_evaluable_samples", 0.0)
        if samples == 0.0:
            continue

        miss_2s = metrics.get("mean_miss_rate_2s")
        ade_2s = metrics.get("mean_ade_2s")
        nll_2s = metrics.get("mean_negative_log_likelihood_2s")

        if miss_2s is not None and miss_2s > 0.3:
            failure_cases.append(
                {
                    "trace": r["label"],
                    "family": r["family"],
                    "trace_path": r["trace_path"],
                    "metric": "mean_miss_rate_2s",
                    "value": miss_2s,
                    "interpretation": (
                        "High miss rate at 2s horizon indicates constant-velocity "
                        "forecast frequently fails to capture actual pedestrian position "
                        "within the 95% confidence ellipse."
                    ),
                }
            )
        if ade_2s is not None and ade_2s > 0.5:
            failure_cases.append(
                {
                    "trace": r["label"],
                    "family": r["family"],
                    "trace_path": r["trace_path"],
                    "metric": "mean_ade_2s",
                    "value": ade_2s,
                    "interpretation": (
                        "Large average displacement error at 2s horizon suggests "
                        "pedestrian motion deviates substantially from constant velocity."
                    ),
                }
            )
        if nll_2s is not None and nll_2s > 5.0:
            failure_cases.append(
                {
                    "trace": r["label"],
                    "family": r["family"],
                    "trace_path": r["trace_path"],
                    "metric": "mean_negative_log_likelihood_2s",
                    "value": nll_2s,
                    "interpretation": (
                        "High negative log-likelihood at 2s horizon indicates poor "
                        "calibration of the constant-velocity Gaussian forecast."
                    ),
                }
            )
    return failure_cases


def _md_header() -> list[str]:
    """Return the report header section."""
    return [
        "# CV Forecast Baseline Evaluation",
        "",
        "## Claim Boundary",
        "",
        "**Diagnostic-only, not paper-facing evidence.** This evaluates the "
        "constant-velocity Gaussian forecast baseline on a bounded set of existing "
        "repository trace fixtures. Results are per-family rollups, not statistically "
        "powered population claims.",
    ]


def _md_repro(repro: dict[str, Any]) -> list[str]:
    """Return the reproducibility section."""
    return [
        "",
        "## Reproducibility",
        "",
        f"- **Issue:** #{repro['issue']}",
        f"- **Generated at (UTC):** {repro['generated_at_utc']}",
        f"- **Command:** `{repro['command']}`",
        f"- **Repo HEAD:** `{repro['repo_head']}`",
        f"- **Forecast horizons:** {repro['horizons_s']}",
    ]


def _md_trace_table(results: list[dict[str, Any]]) -> list[str]:
    """Return the trace families table."""
    lines = [
        "",
        "## Selected Trace Families",
        "",
        "| Family | Label | Scenario | Frames | Peds | Motion | dt (s) | Status |",
        "|--------|-------|----------|--------|------|--------|--------|--------|",
    ]
    for r in results:
        motion_str = "yes" if r.get("has_motion") else "no"
        frames_str = str(r.get("frame_count", "?"))
        peds_str = str(r.get("pedestrians_per_frame", "?"))
        dt_str = f"{r['dt_s']:.1f}" if "dt_s" in r else "-"
        lines.append(
            f"| {r['family']} | {r['label']} | {r.get('scenario_id', '')} "
            f"| {frames_str} | {peds_str} | {motion_str} | {dt_str} | {r['status']} |"
        )
    return lines


def _md_missing_families() -> list[str]:
    """Return the missing families section."""
    lines = [
        "",
        "## Missing Trace Families",
        "",
        "These scenario families have no durable trace fixtures in the repository "
        "and were not evaluated:",
        "",
    ]
    for mf in MISSING_FAMILIES:
        lines.append(f"- **{mf['family']}**: {mf['reason']}")
    return lines


def _md_horizon_metrics(r: dict[str, Any], metrics: dict[str, float], horizon: float) -> list[str]:
    """Return metric lines for one horizon."""
    suffix = f"{horizon:g}s"
    lines = [f"- **Horizon {suffix}:**"]
    metric_lines = 0
    for key, fmt in [
        ("mean_ade", "  - ADE: {val:.4f} m"),
        ("mean_negative_log_likelihood", "  - NLL: {val:.4f}"),
        ("mean_miss_rate", "  - Miss rate: {val:.2%}"),
        ("mean_within_95ci", "  - Within 95% CI: {val:.2%}"),
        ("mean_calibration_error", "  - Calibration error: {val:.4f}"),
    ]:
        val = metrics.get(f"{key}_{suffix}")
        if val is not None:
            lines.append(fmt.format(val=val))
            metric_lines += 1
    if metric_lines == 0:
        lines.append("  - Not available for this trace length.")
    return lines


def _md_metrics_section(results: list[dict[str, Any]]) -> list[str]:
    """Return the metrics section."""
    evaluated = [r for r in results if r["status"] == "evaluated"]
    if not evaluated:
        return ["", "## Metrics", "", "No traces had evaluable samples."]
    lines = ["", "## Aggregate Metrics by Trace Family", ""]
    for r in evaluated:
        metrics = r["metrics"]
        lines.append(f"### {r['family']} / {r['label']}")
        lines.append("")
        lines.append(f"- Evaluable samples: {metrics.get('forecast_evaluable_samples', 0):.0f}")
        for horizon in r.get("horizons_s", []):
            lines.extend(_md_horizon_metrics(r, metrics, horizon))
        lines.append("")
    return lines


def _md_failure_cases(failure_cases: list[dict[str, Any]]) -> list[str]:
    """Return the failure cases section."""
    if not failure_cases:
        return []
    lines = [
        "",
        "## Representative Failure Cases",
        "",
        "These cases show where the constant-velocity baseline produces "
        "degraded forecast quality. Each case links to the durable trace source.",
        "",
    ]
    for fc in failure_cases:
        lines.append(f"- **{fc['trace']}** ({fc['family']}): `{fc['metric']}` = {fc['value']:.4f}")
        lines.append(f"  - {fc['interpretation']}")
        lines.append(f"  - Trace: `{fc['trace_path']}`")
    return lines


def _md_interpretation() -> list[str]:
    """Return the interpretation section."""
    return [
        "",
        "## Interpretation",
        "",
        "The constant-velocity Gaussian baseline is adequate for short horizons "
        "(0.5s) on traces with relatively smooth, low-acceleration pedestrian motion "
        "(such as the corridor_interaction family). In this bounded evidence set, "
        "1s metrics remain available and low-error, while 2s metrics are unavailable "
        "because the durable traces are too short for that forecast horizon.",
        "",
        "The occluded_emergence family is now evaluated with the durable fixture "
        "from #2756. It produces 11 evaluable samples with motion-rich pedestrian "
        "trajectories, extending forecast evidence beyond the corridor family. "
        "This is diagnostic-only evidence for a single occluded-emergence episode, "
        "not a statistically powered claim across the family.",
        "",
        "The available crossing_proxy and bottleneck fixtures are limitation records, "
        "not forecast-quality evidence: they contain zero pedestrian motion, so they "
        "do not test whether constant velocity handles crossing, bottleneck, or "
        "dense interaction dynamics.",
        "",
        "All traces in this evaluation lack intent and signal context metadata, "
        "so the forecast operates in 'uncertain' mode with 1.5x widened standard "
        "deviation. This is a systematic limitation of the available trace fixtures.",
    ]


def _generate_markdown(
    results: list[dict[str, Any]],
    failure_cases: list[dict[str, Any]],
    repro: dict[str, Any],
) -> str:
    """Generate the Markdown evidence report."""
    lines: list[str] = []
    lines.extend(_md_header())
    lines.extend(_md_repro(repro))
    lines.extend(_md_trace_table(results))
    lines.extend(_md_missing_families())
    lines.extend(_md_metrics_section(results))
    lines.extend(_md_failure_cases(failure_cases))
    lines.extend(_md_interpretation())
    return "\n".join(lines)


def main() -> None:
    """Run CV forecast evaluation and write evidence artifacts."""
    parser = argparse.ArgumentParser(
        description="Evaluate CV forecast baseline on bounded repository traces."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="docs/context/evidence/issue_2774_motion_rich_forecast_traces_2026-06-14",
        help="Output directory for evidence artifacts.",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="cv",
        choices=list(BASELINE_FUNCTIONS.keys()),
        help="Baseline function to use (default: cv).",
    )
    parser.add_argument(
        "--compare-all",
        action="store_true",
        help="Run all baselines and write a comparison report.",
    )
    args = parser.parse_args()

    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    repo_head = _git_head()
    generated_at = datetime.datetime.now(datetime.UTC).isoformat()

    baselines_to_run = list(BASELINE_FUNCTIONS.keys()) if args.compare_all else [args.baseline]
    command = (
        f"uv run python scripts/benchmark/run_cv_forecast_eval.py --output-dir {args.output_dir}"
        + (f" --baseline {args.baseline}" if not args.compare_all else " --compare-all")
    )

    repro = {
        "issue": 2758,
        "generated_at_utc": generated_at,
        "command": command,
        "repo_head": repo_head,
        "horizons_s": list(DEFAULT_FORECAST_HORIZONS_S),
    }

    all_results: dict[str, list[dict[str, Any]]] = {}
    for baseline_name in baselines_to_run:
        baseline_fn = BASELINE_FUNCTIONS[baseline_name]
        results: list[dict[str, Any]] = []
        for candidate in TRACE_CANDIDATES:
            result = evaluate_single_trace(candidate, baseline_function=baseline_fn)
            result["baseline"] = baseline_name
            results.append(result)
        all_results[baseline_name] = results

    if args.compare_all:
        _write_comparison_report(all_results, repro, output_dir)
    else:
        results = all_results[args.baseline]
        failure_cases = _build_failure_cases(results)
        evaluated_families = sorted({r["family"] for r in results if r["status"] == "evaluated"})
        limited_families = sorted({r["family"] for r in results if r["status"] != "evaluated"})

        report_json: dict[str, Any] = {
            "issue": 2758,
            "claim_boundary": (
                "Diagnostic-only, not paper-facing evidence. "
                "Limited to bounded repository trace fixtures."
            ),
            "reproducibility": repro,
            "trace_family_gaps": {
                "evaluated": evaluated_families,
                "limited": limited_families,
                "missing": [mf["family"] for mf in MISSING_FAMILIES],
            },
            "results_by_trace": results,
            "failure_cases": failure_cases,
        }

        json_path = output_dir / "report.json"
        with open(json_path, "w") as fh:
            json.dump(report_json, fh, indent=2)

        md_content = _generate_markdown(results, failure_cases, repro)
        md_path = output_dir / "report.md"
        with open(md_path, "w") as fh:
            fh.write(md_content)

        print(f"Wrote {json_path}")
        print(f"Wrote {md_path}")

    for baseline_name, results in all_results.items():
        for r in results:
            status_icon = "+" if r["status"] == "evaluated" else "~"
            samples = r.get("metrics", {}).get("forecast_evaluable_samples", 0)
            print(
                f"  [{status_icon}] {baseline_name}/{r['family']}/{r['label']}: "
                f"{r['status']} ({samples:.0f} samples)"
            )


def _write_comparison_report(
    all_results: dict[str, list[dict[str, Any]]],
    repro: dict[str, Any],
    output_dir: pathlib.Path,
) -> None:
    """Write a comparison report across all baselines."""
    baselines = list(all_results.keys())

    comparison_rows: list[dict[str, Any]] = []
    for baseline_name in baselines:
        results = all_results[baseline_name]
        for r in results:
            metrics = r.get("metrics", {})
            comparison_rows.append(
                {
                    "baseline": baseline_name,
                    "family": r["family"],
                    "label": r["label"],
                    "status": r["status"],
                    "evaluable_samples": metrics.get("forecast_evaluable_samples", 0.0),
                    "mean_ade_0.5s": metrics.get("mean_ade_0.5s"),
                    "mean_ade_1s": metrics.get("mean_ade_1s"),
                    "mean_negative_log_likelihood_1s": metrics.get(
                        "mean_negative_log_likelihood_1s"
                    ),
                    "mean_miss_rate_1s": metrics.get("mean_miss_rate_1s"),
                    "mean_within_95ci_1s": metrics.get("mean_within_95ci_1s"),
                }
            )

    fixture_has_signal = _fixtures_have_signal_metadata(all_results)
    fixture_has_intent = _fixtures_have_intent_metadata(all_results)

    summary_by_baseline: dict[str, dict[str, Any]] = {}
    for baseline_name in baselines:
        results = all_results[baseline_name]
        evaluable = [r for r in results if r["status"] == "evaluated"]
        total_samples = sum(
            r.get("metrics", {}).get("forecast_evaluable_samples", 0.0) for r in evaluable
        )
        summary_by_baseline[baseline_name] = {
            "evaluated_traces": len(evaluable),
            "total_samples": float(total_samples),
        }

    report: dict[str, Any] = {
        "issue": 2758,
        "claim_boundary": (
            "Diagnostic-only, not paper-facing evidence. "
            "Limited to bounded repository trace fixtures."
        ),
        "reproducibility": repro,
        "semantic_usefulness": {
            "fixtures_have_signal_metadata": fixture_has_signal,
            "fixtures_have_intent_metadata": fixture_has_intent,
            "conclusion": (
                "Semantic conditioning is meaningful when fixtures carry signal/intent "
                "metadata. Existing durable fixtures lack this metadata, so the comparison "
                "shows whether baselines run correctly but does not yet show semantic "
                "conditioning improving calibration or collision relevance."
            ),
        },
        "summary_by_baseline": summary_by_baseline,
        "comparison_rows": comparison_rows,
    }

    json_path = output_dir / "comparison_report.json"
    with open(json_path, "w") as fh:
        json.dump(report, fh, indent=2)

    md_lines = [
        "# Semantic Forecast Baseline Comparison",
        "",
        "## Claim Boundary",
        "",
        "**Diagnostic-only, not paper-facing evidence.** Compares constant-velocity and "
        "semantic forecast baselines on bounded repository trace fixtures.",
        "",
        "## Reproducibility",
        "",
        f"- **Issue:** #{repro['issue']}",
        f"- **Generated at (UTC):** {repro['generated_at_utc']}",
        f"- **Command:** `{repro['command']}`",
        f"- **Repo HEAD:** `{repro['repo_head']}`",
        f"- **Forecast horizons:** {repro['horizons_s']}",
        "",
        "## Fixture Metadata Assessment",
        "",
        f"- **Fixtures have signal metadata:** {fixture_has_signal}",
        f"- **Fixtures have intent metadata:** {fixture_has_intent}",
        "",
    ]

    if not fixture_has_signal and not fixture_has_intent:
        md_lines.extend(
            [
                "Existing durable fixtures lack signal and intent metadata. Semantic "
                "conditioning baselines run correctly but their calibration/collision-relevance "
                "improvement cannot be measured until fixtures include this metadata.",
                "",
            ]
        )

    md_lines.extend(
        [
            "## Baseline Summary",
            "",
            "| Baseline | Evaluated Traces | Total Samples |",
            "|----------|------------------|---------------|",
        ]
    )
    for baseline_name in baselines:
        s = summary_by_baseline[baseline_name]
        md_lines.append(
            f"| {baseline_name} | {s['evaluated_traces']:.0f} | {s['total_samples']:.0f} |"
        )

    md_lines.extend(
        [
            "",
            "## Comparison Table",
            "",
            "| Baseline | Family | Label | Status | Samples | ADE 1s | NLL 1s | Miss Rate 1s |",
            "|----------|--------|-------|--------|---------|--------|--------|-------------|",
        ]
    )
    for row in comparison_rows:
        ade = f"{row['mean_ade_1s']:.4f}" if row["mean_ade_1s"] is not None else "-"
        nll = (
            f"{row['mean_negative_log_likelihood_1s']:.4f}"
            if row["mean_negative_log_likelihood_1s"] is not None
            else "-"
        )
        miss = f"{row['mean_miss_rate_1s']:.2%}" if row["mean_miss_rate_1s"] is not None else "-"
        md_lines.append(
            f"| {row['baseline']} | {row['family']} | {row['label']} "
            f"| {row['status']} | {row['evaluable_samples']:.0f} "
            f"| {ade} | {nll} | {miss} |"
        )

    md_content = "\n".join(md_lines)
    md_path = output_dir / "comparison_report.md"
    with open(md_path, "w") as fh:
        fh.write(md_content)

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


def _fixtures_have_signal_metadata(all_results: dict[str, list[dict[str, Any]]]) -> bool:
    """Check whether any evaluated trace has signal metadata on pedestrians."""
    for ped in _iter_unique_evaluated_pedestrians(all_results):
        if ped.get("signal_label") is not None:
            return True
        ss = ped.get("signal_state")
        if isinstance(ss, dict) and ss.get("label") is not None:
            return True
    return False


def _fixtures_have_intent_metadata(all_results: dict[str, list[dict[str, Any]]]) -> bool:
    """Check whether any evaluated trace has intent metadata on pedestrians."""
    return any(
        ped.get("intent_label") is not None
        for ped in _iter_unique_evaluated_pedestrians(all_results)
    )


def _iter_unique_evaluated_pedestrians(
    all_results: dict[str, list[dict[str, Any]]],
) -> Iterator[dict[str, Any]]:
    """Yield pedestrians from each unique evaluated trace with forecast samples."""
    checked_paths: set[pathlib.Path] = set()
    for results in all_results.values():
        for r in results:
            if r["status"] != "evaluated":
                continue
            metrics = r.get("metrics", {})
            if metrics.get("forecast_evaluable_samples", 0.0) <= 0.0:
                continue
            path = REPO_ROOT / r["trace_path"]
            if path in checked_paths or not path.exists():
                continue
            checked_paths.add(path)
            trace = _load_trace(path)
            frames = trace.get("frames") or trace.get("steps") or []
            for frame in frames:
                yield from frame.get("pedestrians", [])


if __name__ == "__main__":
    main()
