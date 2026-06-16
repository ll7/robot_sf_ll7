#!/usr/bin/env python3
"""Build an analysis-only horizon x timestep ablation report for forecast outputs.

Reads durable repository trace fixtures, evaluates a constant-velocity Gaussian
forecast baseline across a horizon ladder and output-dt_s ladder, and writes a
compact JSON + Markdown evidence bundle.  The report is diagnostic-only and does
not claim navigation benefit, closed-loop improvement, or benchmark-strength
predictor ranking.

Usage::

    uv run python scripts/benchmark/build_horizon_timestep_ablation_report.py \
        --output-dir docs/context/evidence/issue_2837_horizon_timestep_ablation_2026-06-15
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import pathlib
import subprocess
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from robot_sf.benchmark.pedestrian_forecast import (
    ForecastBaselineFunction,
    compute_batch_forecast_metrics,
    constant_velocity_gaussian_baseline,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

HORIZON_LADDER_S: tuple[float, ...] = (0.5, 1.0, 1.6, 2.0, 3.0)
DT_LADDER_S: tuple[float, ...] = (0.1, 0.2, 0.4, 0.5)

CLAIM_BOUNDARY = (
    "analysis_only_not_navigation_evidence: this report compares forecast horizon and "
    "output-timestep presets on open-loop trace fixtures. It does not prove navigation "
    "value, closed-loop benefit, safety improvement, or benchmark-strength predictor quality."
)

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
        "path": "tests/fixtures/analysis_workbench/simulation_trace_export_v1/issue_2937/crossing_proxy_motion_rich_fixture.json",
        "scenario_id": "crossing_proxy",
        "planner_id": "orca",
        "seed": 111,
    },
    {
        "family": "bottleneck",
        "label": "minimal_fixture",
        "path": "tests/fixtures/analysis_workbench/simulation_trace_export_v1/issue_2937/bottleneck_motion_rich_fixture.json",
        "scenario_id": "classic_bottleneck_medium",
        "planner_id": "hybrid_rule_v0_minimal",
        "seed": 111,
    },
    {
        "family": "occluded_emergence",
        "label": "deterministic_occluded_emergence",
        "path": "tests/fixtures/analysis_workbench/simulation_trace_export_v1/issue_2937/occluded_emergence_episode_extended.json",
        "scenario_id": "issue_2756_occluded_emergence",
        "planner_id": "hybrid_rule_v0_minimal",
        "seed": 111,
    },
    {
        "family": "signalized_crossing",
        "label": "signalized_crossing_semantic_metadata",
        "path": "tests/fixtures/analysis_workbench/simulation_trace_export_v1/issue_2937/issue_2868_signalized_crossing_fixture_extended.json",
        "scenario_id": "issue_2868_signalized_crossing",
        "planner_id": "hybrid_rule_v0_minimal",
        "seed": 2868,
    },
    {
        "family": "goal_directed_crossing",
        "label": "goal_directed_crossing_fixture",
        "path": "tests/fixtures/analysis_workbench/simulation_trace_export_v1/issue_2937/issue_2868_goal_directed_crossing_fixture_extended.json",
        "scenario_id": "issue_2868_goal_directed_crossing",
        "planner_id": "hybrid_rule_v0_minimal",
        "seed": 2868,
    },
    {
        "family": "waiting_with_intent_change",
        "label": "waiting_intent_change_fixture",
        "path": "tests/fixtures/analysis_workbench/simulation_trace_export_v1/issue_2937/issue_2868_waiting_intent_change_fixture_extended.json",
        "scenario_id": "issue_2868_waiting_intent_change",
        "planner_id": "hybrid_rule_v0_minimal",
        "seed": 2868,
    },
    {
        "family": "route_conflict_goal",
        "label": "route_conflict_goal_fixture",
        "path": "tests/fixtures/analysis_workbench/simulation_trace_export_v1/issue_2937/issue_2868_route_conflict_goal_fixture_extended.json",
        "scenario_id": "issue_2868_route_conflict_goal",
        "planner_id": "hybrid_rule_v0_minimal",
        "seed": 2868,
    },
]

MISSING_FAMILIES: list[dict[str, str]] = [
    {"family": "dense_pedestrian_interaction", "reason": "no durable trace fixture available"},
]


def _load_trace(path: pathlib.Path) -> dict[str, Any]:
    """Load a simulation_trace_export.v1 JSON file."""
    with open(path) as fh:
        return json.load(fh)


def _extract_trace_steps(trace: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert trace export frames to the step list expected by compute_batch_forecast_metrics."""
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


def _compute_native_dt_s(trace_steps: Sequence[dict[str, Any]]) -> float:
    """Infer the native timestep from the first two trace steps."""
    if len(trace_steps) < 2:
        return 0.1
    dt = float(trace_steps[1].get("time_s", 0.1)) - float(trace_steps[0].get("time_s", 0.0))
    return dt if dt > 0 else 0.1


def _resample_trace_steps(
    trace_steps: list[dict[str, Any]], target_dt_s: float
) -> tuple[list[dict[str, Any]], float]:
    """Resample trace steps to a coarser (or equal) output timestep.

    The native timestep is inferred from ``time_s``.  Frames are selected at
    integer multiples of ``target_dt_s`` rounded to the native grid, so the
    actual dt may differ slightly from the requested dt when the target does not
    divide the native grid.  The returned actual dt is always an integer
    multiple of the native dt.

    Returns:
        Tuple of (resampled steps, actual_dt_s).
    """
    if not trace_steps:
        return [], target_dt_s

    native_dt = _compute_native_dt_s(trace_steps)
    if target_dt_s <= native_dt + 1e-9:
        return trace_steps, native_dt

    stride = max(1, round(float(target_dt_s) / native_dt))
    actual_dt = stride * native_dt
    selected = trace_steps[::stride]

    resampled: list[dict[str, Any]] = []
    for new_index, step in enumerate(selected):
        step_copy = dict(step)
        step_copy["time_s"] = new_index * actual_dt
        step_copy["step"] = new_index
        resampled.append(step_copy)
    return resampled, actual_dt


def _trace_has_motion(trace_steps: Sequence[dict[str, Any]]) -> bool:
    """Check whether any pedestrian has non-zero velocity in any frame."""
    for step in trace_steps:
        for ped in step.get("pedestrians", []):
            vel = ped.get("velocity", [0.0, 0.0])
            if any(abs(float(v)) > 1e-6 for v in vel):
                return True
    return False


def _pedestrian_count(trace_steps: Sequence[dict[str, Any]]) -> int:
    """Return the max number of pedestrians in any single step."""
    return max((len(step.get("pedestrians", [])) for step in trace_steps), default=0)


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


def _sanitize_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """Convert numpy types to plain floats and keep only ablation-relevant keys.

    Per-cell metrics keep mean values and the overall evaluable-sample count.
    Raw count_* keys are omitted to keep the tracked evidence bundle compact.
    """
    keep_prefixes = ("mean_", "forecast_evaluable_samples")
    sanitized: dict[str, float] = {}
    for key, value in metrics.items():
        if not key.startswith(keep_prefixes):
            continue
        if isinstance(value, (np.floating, np.integer)):
            sanitized[key] = float(value)
        else:
            sanitized[key] = float(value)
    return sanitized


def _artifact_size_bytes(rel_path: str) -> int | None:
    """Return the on-disk size of a trace fixture, or None if missing."""
    path = REPO_ROOT / rel_path
    try:
        return path.stat().st_size if path.exists() else None
    except OSError:
        return None


def _memory_proxy(frame_count: int, peds_per_frame: int) -> dict[str, Any]:
    """Return a lightweight proxy for memory/artifact scale."""
    return {
        "frame_count": frame_count,
        "pedestrians_per_frame": peds_per_frame,
        "trace_scalar_proxy": frame_count * peds_per_frame,
    }


def _metric_at_horizon(metrics: dict[str, float], horizon_s: float, metric: str) -> float | None:
    """Extract a mean metric for a specific horizon, or None if unavailable."""
    suffix = f"{horizon_s:g}s"
    key = f"mean_{metric}_{suffix}"
    if key in metrics:
        return float(metrics[key])
    key = f"{metric}_{suffix}"
    if key in metrics:
        return float(metrics[key])
    return None


def evaluate_ablation_cell(
    candidate: dict[str, Any],
    horizon_s: float,
    requested_dt_s: float,
    baseline_function: ForecastBaselineFunction | None = None,
) -> dict[str, Any]:
    """Evaluate one (trace, horizon, dt_s) ablation cell.

    Returns a result dict with provenance, status, metrics, runtime, and
    limitation notes.  Missing fixtures or un-evaluable combos are reported as
    unavailable rather than fabricated.
    """
    rel_path = candidate["path"]
    abs_path = REPO_ROOT / rel_path

    cell: dict[str, Any] = {
        "family": candidate["family"],
        "label": candidate["label"],
        "scenario_id": candidate.get("scenario_id", ""),
        "planner_id": candidate.get("planner_id", ""),
        "seed": candidate.get("seed"),
        "trace_path": rel_path,
        "requested_dt_s": requested_dt_s,
        "actual_dt_s": requested_dt_s,
        "horizon_s": horizon_s,
        "status": "pending",
        "metrics": {},
        "runtime_s": 0.0,
        "artifact_size_bytes": _artifact_size_bytes(rel_path),
    }

    if not abs_path.exists():
        cell["status"] = "trace_file_missing"
        cell["limitation"] = "Durable trace fixture is not present in the repository."
        return cell

    trace = _load_trace(abs_path)
    trace_steps = _extract_trace_steps(trace)
    native_dt = _compute_native_dt_s(trace_steps)

    if not _trace_has_motion(trace_steps):
        cell["status"] = "limited_no_pedestrian_motion"
        cell["limitation"] = (
            "All pedestrian velocities are zero; constant-velocity forecast "
            "produces degenerate predictions with no motion to evaluate against."
        )
        cell["memory_proxy"] = _memory_proxy(len(trace_steps), _pedestrian_count(trace_steps))
        cell["native_dt_s"] = native_dt
        return cell

    resampled, actual_dt = _resample_trace_steps(trace_steps, requested_dt_s)
    cell["actual_dt_s"] = actual_dt
    cell["native_dt_s"] = native_dt
    cell["memory_proxy"] = _memory_proxy(len(resampled), _pedestrian_count(resampled))

    if len(resampled) < 3:
        cell["status"] = "insufficient_frames"
        cell["limitation"] = (
            f"Resampled trace has {len(resampled)} frames; "
            "need at least 3 for any horizon evaluation."
        )
        return cell

    future_step_offset = round(float(horizon_s) / actual_dt)
    if future_step_offset >= len(resampled):
        cell["status"] = "horizon_longer_than_trace"
        cell["limitation"] = (
            f"Horizon {horizon_s:g}s at dt {actual_dt:g}s requires {future_step_offset} steps, "
            f"but resampled trace has only {len(resampled)} frames."
        )
        return cell

    baseline_fn = baseline_function or constant_velocity_gaussian_baseline
    start_time = time.perf_counter()
    try:
        metrics = compute_batch_forecast_metrics(
            resampled,
            horizons_s=[float(horizon_s)],
            dt_s=actual_dt,
            baseline_function=baseline_fn,
        )
    except Exception as exc:
        cell["status"] = "evaluation_error"
        cell["limitation"] = str(exc)
        cell["runtime_s"] = time.perf_counter() - start_time
        return cell

    cell["runtime_s"] = time.perf_counter() - start_time
    cell["metrics"] = _sanitize_metrics(metrics)

    evaluable = metrics.get("forecast_evaluable_samples", 0.0)
    if evaluable == 0.0:
        cell["status"] = "no_evaluable_samples"
        cell["limitation"] = "Forecast pipeline produced zero evaluable samples."
    else:
        cell["status"] = "evaluated"

    return cell


def _cells_for_candidate(
    candidate: dict[str, Any],
    baseline_function: ForecastBaselineFunction | None = None,
) -> list[dict[str, Any]]:
    """Evaluate all horizon x dt_s cells for one trace candidate."""
    cells: list[dict[str, Any]] = []
    for horizon_s in HORIZON_LADDER_S:
        for requested_dt_s in DT_LADDER_S:
            cells.append(
                evaluate_ablation_cell(
                    candidate,
                    horizon_s=horizon_s,
                    requested_dt_s=requested_dt_s,
                    baseline_function=baseline_function,
                )
            )
    return cells


def _preset_recommendations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Recommend short/medium/long presets from evaluated cells.

    Presets are anchored to explicit (horizon, dt) targets and selected only
    when an evaluated cell matches.  If the target is unavailable because the
    fixtures are too short, the preset is reported as unavailable rather than
    promoted from a mismatched cell.  These are forecast-output presets for
    analysis, not navigation guarantees.
    """
    evaluated = [r for r in rows if r["status"] == "evaluated"]

    targets: list[tuple[str, float, float, str]] = [
        ("short", 0.5, 0.1, "near-term collision relevance; fine output granularity"),
        ("medium", 1.6, 0.2, "intent/goal horizon; moderate granularity"),
        ("long", 3.0, 0.4, "route-scale lookahead; coarse granularity where trace length permits"),
    ]

    def _match(name: str, horizon: float, target_dt: float) -> dict[str, Any] | None:
        exact = [
            c
            for c in evaluated
            if c["horizon_s"] == horizon and abs(c["actual_dt_s"] - target_dt) < 1e-9
        ]
        if not exact:
            return None

        # Prefer the cell with the most evaluable samples and lowest miss rate.
        def _score(cell: dict[str, Any]) -> tuple[float, float]:
            metrics = cell.get("metrics", {})
            samples = metrics.get("forecast_evaluable_samples", 0.0)
            miss = _metric_at_horizon(metrics, cell["horizon_s"], "miss_rate") or 1.0
            return (samples, 1.0 - miss)

        return max(exact, key=_score)

    presets: list[dict[str, Any]] = []
    for name, horizon, target_dt, intended_use in targets:
        cell = _match(name, horizon, target_dt)
        if cell is None:
            presets.append(
                {
                    "preset": name,
                    "status": "unavailable",
                    "intended_use": intended_use,
                    "reason": (
                        f"No evaluated cell matched horizon {horizon:g}s / "
                        f"dt {target_dt:g}s in this fixture set."
                    ),
                }
            )
        else:
            metrics = cell.get("metrics", {})
            presets.append(
                {
                    "preset": name,
                    "status": "recommended",
                    "horizon_s": cell["horizon_s"],
                    "requested_dt_s": cell["requested_dt_s"],
                    "actual_dt_s": cell["actual_dt_s"],
                    "intended_use": intended_use,
                    "evaluable_samples": metrics.get("forecast_evaluable_samples", 0.0),
                    "miss_rate": _metric_at_horizon(metrics, cell["horizon_s"], "miss_rate"),
                    "ade_m": _metric_at_horizon(metrics, cell["horizon_s"], "ade"),
                    "nll": _metric_at_horizon(
                        metrics, cell["horizon_s"], "negative_log_likelihood"
                    ),
                    "calibration_error": _metric_at_horizon(
                        metrics, cell["horizon_s"], "calibration_error"
                    ),
                    "collision_relevance_error": _metric_at_horizon(
                        metrics, cell["horizon_s"], "collision_relevance_error"
                    ),
                    "runtime_s": cell.get("runtime_s", 0.0),
                    "trace_family": cell["family"],
                    "limitation": "Diagnostic-only; no closed-loop or navigation claim.",
                }
            )
    return presets


def _build_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Roll ablation cells up to (horizon_s, dt_s) summary rows."""
    grouped: dict[tuple[float, float], list[dict[str, Any]]] = {}
    for row in rows:
        key = (round(row["horizon_s"], 6), round(row["actual_dt_s"], 6))
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for (horizon_s, dt_s), cells in sorted(grouped.items()):
        evaluated = [c for c in cells if c["status"] == "evaluated"]
        total_samples = sum(
            c.get("metrics", {}).get("forecast_evaluable_samples", 0.0) for c in evaluated
        )
        total_runtime = sum(c.get("runtime_s", 0.0) for c in cells)
        total_size = sum(
            c.get("artifact_size_bytes") or 0 for c in cells if c.get("artifact_size_bytes")
        )
        miss_rates = [
            _metric_at_horizon(c.get("metrics", {}), horizon_s, "miss_rate")
            for c in evaluated
            if _metric_at_horizon(c.get("metrics", {}), horizon_s, "miss_rate") is not None
        ]
        calibration_errors = [
            _metric_at_horizon(c.get("metrics", {}), horizon_s, "calibration_error")
            for c in evaluated
            if _metric_at_horizon(c.get("metrics", {}), horizon_s, "calibration_error") is not None
        ]
        collision_errors = [
            _metric_at_horizon(c.get("metrics", {}), horizon_s, "collision_relevance_error")
            for c in evaluated
            if _metric_at_horizon(c.get("metrics", {}), horizon_s, "collision_relevance_error")
            is not None
        ]

        summary_rows.append(
            {
                "horizon_s": horizon_s,
                "dt_s": dt_s,
                "evaluated_cells": len(evaluated),
                "total_cells": len(cells),
                "total_evaluable_samples": total_samples,
                "mean_miss_rate": float(np.mean(miss_rates)) if miss_rates else None,
                "mean_calibration_error": float(np.mean(calibration_errors))
                if calibration_errors
                else None,
                "mean_collision_relevance_error": float(np.mean(collision_errors))
                if collision_errors
                else None,
                "total_runtime_s": total_runtime,
                "total_artifact_size_bytes": total_size,
                "status": "evaluated" if evaluated else "unavailable",
            }
        )
    return summary_rows


def _generate_markdown(
    rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    presets: list[dict[str, Any]],
    repro: dict[str, Any],
) -> str:
    """Generate the Markdown ablation report."""
    lines = [
        "# Horizon and Timestep Ablation Report",
        "",
        "## Claim Boundary",
        "",
        f"**{CLAIM_BOUNDARY}**",
        "",
        "## Reproducibility",
        "",
        f"- **Issue:** #{repro['issue']}",
        f"- **Generated at (UTC):** {repro['generated_at_utc']}",
        f"- **Command:** `{repro['command']}`",
        f"- **Repo HEAD:** `{repro['repo_head']}`",
        f"- **Horizon ladder (s):** {repro['horizon_ladder_s']}",
        f"- **dt ladder (s):** {repro['dt_ladder_s']}",
        "",
        "## Ablation Summary (horizon x dt_s)",
        "",
        "| horizon_s | dt_s | evaluated/total | samples | mean miss rate | mean calib. error | "
        "mean coll. rel. error | runtime (s) | artifact size (B) | status |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary_rows:
        miss = f"{row['mean_miss_rate']:.2%}" if row["mean_miss_rate"] is not None else "NA"
        cal = (
            f"{row['mean_calibration_error']:.4f}"
            if row["mean_calibration_error"] is not None
            else "NA"
        )
        coll = (
            f"{row['mean_collision_relevance_error']:.4f}"
            if row["mean_collision_relevance_error"] is not None
            else "NA"
        )
        lines.append(
            f"| {row['horizon_s']:g} | {row['dt_s']:g} | "
            f"{row['evaluated_cells']}/{row['total_cells']} | "
            f"{row['total_evaluable_samples']:.0f} | {miss} | {cal} | {coll} | "
            f"{row['total_runtime_s']:.4f} | {row['total_artifact_size_bytes']} | {row['status']} |"
        )

    lines.extend(
        [
            "",
            "## Preset Recommendations",
            "",
            "| preset | horizon_s | dt_s | status | samples | miss rate | runtime (s) | intended use |",
            "| --- | ---: | ---: | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for preset in presets:
        if preset["status"] == "unavailable":
            lines.append(
                f"| {preset['preset']} | - | - | unavailable | - | - | - | {preset['intended_use']} |"
            )
        else:
            miss = f"{preset['miss_rate']:.2%}" if preset["miss_rate"] is not None else "NA"
            lines.append(
                f"| {preset['preset']} | {preset['horizon_s']:g} | {preset['actual_dt_s']:g} | "
                f"{preset['status']} | {preset['evaluable_samples']:.0f} | {miss} | "
                f"{preset['runtime_s']:.4f} | {preset['intended_use']} |"
            )

    lines.extend(
        [
            "",
            "## Per-Trace, Per-Cell Status",
            "",
            "| family | label | horizon_s | requested_dt_s | actual_dt_s | status | samples | "
            "ADE (m) | miss rate | NLL | calib. error | coll. rel. error | runtime (s) |",
            "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        metrics = row.get("metrics", {})
        horizon = row["horizon_s"]
        ade = _metric_at_horizon(metrics, horizon, "ade")
        miss = _metric_at_horizon(metrics, horizon, "miss_rate")
        nll = _metric_at_horizon(metrics, horizon, "negative_log_likelihood")
        cal = _metric_at_horizon(metrics, horizon, "calibration_error")
        coll = _metric_at_horizon(metrics, horizon, "collision_relevance_error")
        samples = metrics.get("forecast_evaluable_samples", 0.0)
        lines.append(
            f"| {row['family']} | {row['label']} | {horizon:g} | "
            f"{row['requested_dt_s']:g} | {row['actual_dt_s']:g} | {row['status']} | "
            f"{samples:.0f} | {ade if ade is None else f'{ade:.4f}'} | "
            f"{miss if miss is None else f'{miss:.2%}'} | "
            f"{nll if nll is None else f'{nll:.4f}'} | "
            f"{cal if cal is None else f'{cal:.4f}'} | "
            f"{coll if coll is None else f'{coll:.4f}'} | {row['runtime_s']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Missing Trace Families",
            "",
            "These scenario families have no durable trace fixtures and were not evaluated:",
            "",
        ]
    )
    for mf in MISSING_FAMILIES:
        lines.append(f"- **{mf['family']}**: {mf['reason']}")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This ablation varies only the forecast output horizon and output timestep on a "
            "bounded set of repository trace fixtures.  It does not change simulator physics "
            "step semantics.  Long-horizon and coarse-dt rows are frequently unavailable "
            "because the durable fixtures are short (1-2 s).  Preset recommendations are "
            "diagnostic suggestions for forecast-output configuration, not evidence that "
            "any preset improves navigation, safety, or closed-loop planner performance.",
        ]
    )
    return "\n".join(lines)


def build_ablation_report(
    baseline_function: ForecastBaselineFunction | None = None,
    issue: int = 2837,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    """Build the full ablation report from durable repository fixtures."""
    repo_head = _git_head()
    generated_at = generated_at_utc or datetime.datetime.now(datetime.UTC).isoformat()

    rows: list[dict[str, Any]] = []
    for candidate in TRACE_CANDIDATES:
        rows.extend(_cells_for_candidate(candidate, baseline_function=baseline_function))

    summary_rows = _build_summary_rows(rows)
    presets = _preset_recommendations(rows)

    command = (
        f"uv run python scripts/benchmark/build_horizon_timestep_ablation_report.py --issue {issue}"
    )
    if generated_at_utc:
        command += f" --generated-at-utc {generated_at_utc}"

    repro = {
        "issue": issue,
        "generated_at_utc": generated_at,
        "command": command,
        "repo_head": repo_head,
        "horizon_ladder_s": list(HORIZON_LADDER_S),
        "dt_ladder_s": list(DT_LADDER_S),
    }

    return {
        "issue": issue,
        "schema_version": "HorizonTimestepAblation.v1",
        "claim_boundary": CLAIM_BOUNDARY,
        "reproducibility": repro,
        "ablation_rows": rows,
        "summary_rows": summary_rows,
        "preset_recommendations": presets,
        "missing_families": MISSING_FAMILIES,
    }


def main() -> None:
    """Run the horizon/timestep ablation and write evidence artifacts."""
    parser = argparse.ArgumentParser(description="Build horizon x dt_s forecast ablation report.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="docs/context/evidence/issue_2837_horizon_timestep_ablation_2026-06-15",
        help="Output directory for evidence artifacts.",
    )
    parser.add_argument(
        "--issue",
        type=int,
        default=2837,
        help="Issue number to record in generated evidence metadata.",
    )
    parser.add_argument(
        "--generated-at-utc",
        help="Optional deterministic ISO-8601 generation timestamp for reviewable artifacts.",
    )
    args = parser.parse_args()

    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    report = build_ablation_report(
        issue=args.issue,
        generated_at_utc=args.generated_at_utc,
    )

    json_path = output_dir / "ablation_report.json"
    with open(json_path, "w") as fh:
        json.dump(report, fh, indent=2)

    md_content = _generate_markdown(
        report["ablation_rows"],
        report["summary_rows"],
        report["preset_recommendations"],
        report["reproducibility"],
    )
    md_path = output_dir / "ablation_report.md"
    with open(md_path, "w") as fh:
        fh.write(md_content)

    summary = {
        "issue": args.issue,
        "schema_version": "HorizonTimestepAblation.v1",
        "status": "diagnostic-only",
        "claim_boundary": CLAIM_BOUNDARY,
        "generated_at_utc": report["reproducibility"]["generated_at_utc"],
        "provenance": {
            "command": report["reproducibility"]["command"],
            "commit": report["reproducibility"]["repo_head"],
            "source": "durable repository trace fixtures",
        },
        "coverage": {
            "horizon_ladder_s": report["reproducibility"]["horizon_ladder_s"],
            "dt_ladder_s": report["reproducibility"]["dt_ladder_s"],
            "trace_families": sorted({r["family"] for r in report["ablation_rows"]}),
            "evaluated_cells": sum(
                1 for r in report["ablation_rows"] if r["status"] == "evaluated"
            ),
            "total_cells": len(report["ablation_rows"]),
            "preset_count": len(report["preset_recommendations"]),
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
                "evaluated_cells": summary["coverage"]["evaluated_cells"],
                "total_cells": summary["coverage"]["total_cells"],
                "presets": [p["preset"] for p in report["preset_recommendations"]],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
