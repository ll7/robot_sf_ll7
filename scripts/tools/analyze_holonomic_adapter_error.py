#!/usr/bin/env python3
"""Analyze holonomic adapter projection error and campaign-level impact.

This tool covers three related questions:

1. For a given unicycle command ``(v, omega)`` and current heading, what
   holonomic ``(vx, vy)`` does the current midpoint projection produce, and how
   far is that from exact constant-control integration over one step?
2. How does that projection error vary across heading and angular-rate ranges?
3. In an actual campaign, do the available aggregate metadata counters suggest
   that adapter projection is the main source of performance degradation, or is
   the regression more likely due to planner/dynamics mismatch?

The recorded benchmark schema does not store per-step command traces, so the
campaign analysis portion is limited to aggregate adapter-impact and kinematics
feasibility counters already emitted by the benchmark.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:  # Matplotlib is optional but preferred for visualization.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional dependency fallback.
    plt = None  # type: ignore[assignment]


EPS = 1e-12


@dataclass(frozen=True)
class CommandErrorSample:
    """One exact-vs-approximate projection comparison."""

    v: float
    omega: float
    heading_rad: float
    dt: float
    exact_vx: float
    exact_vy: float
    approx_vx: float
    approx_vy: float
    exact_dx: float
    exact_dy: float
    approx_dx: float
    approx_dy: float
    error_vx: float
    error_vy: float
    error_dx: float
    error_dy: float
    error_norm_v: float
    error_norm_d: float
    relative_speed_error: float


@dataclass(frozen=True)
class GridSummary:
    """Aggregate statistics over a sampled heading/angular-rate grid."""

    theta_samples: int
    omega_samples: int
    theta_min: float
    theta_max: float
    omega_min: float
    omega_max: float
    max_error_norm_d: float
    mean_error_norm_d: float
    p95_error_norm_d: float
    max_relative_speed_error: float
    mean_relative_speed_error: float


@dataclass(frozen=True)
class PlannerCampaignSummary:
    """Aggregate planner metadata extracted from a camera-ready campaign."""

    planner_key: str
    algo: str
    execution_mode: str
    adapter_name: str | None
    adapter_fraction: float | None
    projection_rate: float | None
    mean_abs_delta_linear: float | None
    mean_abs_delta_angular: float | None
    max_abs_delta_linear: float | None
    max_abs_delta_angular: float | None
    adapter_status: str | None
    success_mean: float | None
    collision_mean: float | None
    snqi_mean: float | None


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def _first_metric_value(payload: dict[str, Any] | None, *keys: str) -> float | None:
    """Return the first numeric metric value present under a set of aliases."""
    if not isinstance(payload, dict):
        return None
    for key in keys:
        value = _safe_float(payload.get(key))
        if value is not None:
            return value
    return None


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_summary(campaign_root: Path) -> dict[str, Any]:
    summary_path = campaign_root / "reports" / "campaign_summary.json"
    return _read_json(summary_path)


def _planner_rows_by_key(summary_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = summary_payload.get("planner_rows")
    if not isinstance(rows, list):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        planner_key = row.get("planner_key")
        if isinstance(planner_key, str) and planner_key:
            out[planner_key] = row
    return out


def _run_entries_by_key(summary_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    runs = summary_payload.get("runs")
    if not isinstance(runs, list):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for run in runs:
        if not isinstance(run, dict):
            continue
        planner = run.get("planner")
        if not isinstance(planner, dict):
            continue
        planner_key = planner.get("key")
        if isinstance(planner_key, str) and planner_key:
            out[planner_key] = run
    return out


def _sinc(x: np.ndarray | float) -> np.ndarray | float:
    """Return sin(x) / x with a stable zero limit."""
    if isinstance(x, np.ndarray):
        out = np.ones_like(x, dtype=float)
        mask = np.abs(x) > EPS
        out[mask] = np.sin(x[mask]) / x[mask]
        return out
    if abs(x) <= EPS:
        return 1.0
    return math.sin(x) / x


def _exact_average_velocity(
    v: float, omega: float, heading: float, dt: float
) -> tuple[float, float]:
    """Return exact average world-frame velocity over one constant-control step."""
    phase = omega * dt * 0.5
    scale = float(_sinc(phase))
    mid_heading = heading + phase
    return (v * scale * math.cos(mid_heading), v * scale * math.sin(mid_heading))


def _midpoint_velocity(v: float, omega: float, heading: float, dt: float) -> tuple[float, float]:
    """Return the current midpoint adapter's world-frame velocity estimate."""
    mid_heading = heading + omega * dt * 0.5
    return (v * math.cos(mid_heading), v * math.sin(mid_heading))


def _exact_displacement(v: float, omega: float, heading: float, dt: float) -> tuple[float, float]:
    """Return exact world-frame displacement over one constant-control step."""
    vx, vy = _exact_average_velocity(v, omega, heading, dt)
    return (vx * dt, vy * dt)


def _midpoint_displacement(
    v: float, omega: float, heading: float, dt: float
) -> tuple[float, float]:
    """Return the midpoint-projection displacement over one step."""
    vx, vy = _midpoint_velocity(v, omega, heading, dt)
    return (vx * dt, vy * dt)


def analyze_command(
    *,
    v: float,
    omega: float,
    heading: float,
    dt: float,
) -> CommandErrorSample:
    """Compare exact and midpoint projected motion for one command."""
    exact_vx, exact_vy = _exact_average_velocity(v, omega, heading, dt)
    approx_vx, approx_vy = _midpoint_velocity(v, omega, heading, dt)
    exact_dx, exact_dy = exact_vx * dt, exact_vy * dt
    approx_dx, approx_dy = approx_vx * dt, approx_vy * dt
    error_vx = exact_vx - approx_vx
    error_vy = exact_vy - approx_vy
    error_dx = exact_dx - approx_dx
    error_dy = exact_dy - approx_dy
    error_norm_v = float(math.hypot(error_vx, error_vy))
    error_norm_d = float(math.hypot(error_dx, error_dy))
    relative_speed_error = 0.0 if abs(v) <= EPS else error_norm_v / abs(v)
    return CommandErrorSample(
        v=float(v),
        omega=float(omega),
        heading_rad=float(heading),
        dt=float(dt),
        exact_vx=float(exact_vx),
        exact_vy=float(exact_vy),
        approx_vx=float(approx_vx),
        approx_vy=float(approx_vy),
        exact_dx=float(exact_dx),
        exact_dy=float(exact_dy),
        approx_dx=float(approx_dx),
        approx_dy=float(approx_dy),
        error_vx=float(error_vx),
        error_vy=float(error_vy),
        error_dx=float(error_dx),
        error_dy=float(error_dy),
        error_norm_v=error_norm_v,
        error_norm_d=error_norm_d,
        relative_speed_error=relative_speed_error,
    )


def sample_grid(
    *,
    v: float,
    dt: float,
    theta_min: float,
    theta_max: float,
    theta_samples: int,
    omega_min: float,
    omega_max: float,
    omega_samples: int,
) -> dict[str, Any]:
    """Sample a heading/rate grid and return vectorized error surfaces."""
    theta_values = np.linspace(theta_min, theta_max, int(theta_samples), dtype=float)
    omega_values = np.linspace(omega_min, omega_max, int(omega_samples), dtype=float)
    theta_grid, omega_grid = np.meshgrid(theta_values, omega_values, indexing="xy")
    phase = omega_grid * float(dt) * 0.5
    scale = _sinc(phase)
    mid_heading = theta_grid + phase
    exact_vx = float(v) * scale * np.cos(mid_heading)
    exact_vy = float(v) * scale * np.sin(mid_heading)
    approx_vx = float(v) * np.cos(mid_heading)
    approx_vy = float(v) * np.sin(mid_heading)
    exact_dx = exact_vx * float(dt)
    exact_dy = exact_vy * float(dt)
    approx_dx = approx_vx * float(dt)
    approx_dy = approx_vy * float(dt)
    error_dx = exact_dx - approx_dx
    error_dy = exact_dy - approx_dy
    error_norm_d = np.hypot(error_dx, error_dy)
    relative_speed_error = np.abs(1.0 - scale)
    return {
        "theta_values": theta_values,
        "omega_values": omega_values,
        "exact_vx": exact_vx,
        "exact_vy": exact_vy,
        "approx_vx": approx_vx,
        "approx_vy": approx_vy,
        "error_dx": error_dx,
        "error_dy": error_dy,
        "error_norm_d": error_norm_d,
        "relative_speed_error": relative_speed_error,
    }


def summarize_grid(grid: dict[str, Any]) -> GridSummary:
    """Summarize sampled grid error surfaces."""
    error_norm_d = np.asarray(grid["error_norm_d"], dtype=float)
    relative_speed_error = np.asarray(grid["relative_speed_error"], dtype=float)
    theta_values = np.asarray(grid["theta_values"], dtype=float)
    omega_values = np.asarray(grid["omega_values"], dtype=float)
    return GridSummary(
        theta_samples=int(theta_values.size),
        omega_samples=int(omega_values.size),
        theta_min=float(theta_values.min(initial=0.0)),
        theta_max=float(theta_values.max(initial=0.0)),
        omega_min=float(omega_values.min(initial=0.0)),
        omega_max=float(omega_values.max(initial=0.0)),
        max_error_norm_d=float(error_norm_d.max(initial=0.0)),
        mean_error_norm_d=float(error_norm_d.mean() if error_norm_d.size else 0.0),
        p95_error_norm_d=float(np.percentile(error_norm_d, 95) if error_norm_d.size else 0.0),
        max_relative_speed_error=float(relative_speed_error.max(initial=0.0)),
        mean_relative_speed_error=float(
            relative_speed_error.mean() if relative_speed_error.size else 0.0
        ),
    )


def _planner_summary(
    *,
    planner_key: str,
    row: dict[str, Any] | None,
    run_entry: dict[str, Any] | None,
) -> PlannerCampaignSummary:
    summary = (run_entry or {}).get("summary", {}) if isinstance(run_entry, dict) else {}
    contract = summary.get("algorithm_metadata_contract") or {}
    planner_contract = contract.get("planner_kinematics") or {}
    feasibility = contract.get("kinematics_feasibility") or {}
    adapter_impact = contract.get("adapter_impact") or {}
    return PlannerCampaignSummary(
        planner_key=planner_key,
        algo=str((run_entry or {}).get("planner", {}).get("algo", planner_key)),
        execution_mode=str(planner_contract.get("execution_mode", "unknown")),
        adapter_name=(
            str(planner_contract.get("adapter_name"))
            if planner_contract.get("adapter_name") is not None
            else None
        ),
        adapter_fraction=_safe_float(adapter_impact.get("adapter_fraction")),
        projection_rate=_safe_float(feasibility.get("projection_rate")),
        mean_abs_delta_linear=_safe_float(feasibility.get("mean_abs_delta_linear")),
        mean_abs_delta_angular=_safe_float(feasibility.get("mean_abs_delta_angular")),
        max_abs_delta_linear=_safe_float(feasibility.get("max_abs_delta_linear")),
        max_abs_delta_angular=_safe_float(feasibility.get("max_abs_delta_angular")),
        adapter_status=(
            str(adapter_impact.get("status")) if adapter_impact.get("status") is not None else None
        ),
        success_mean=_safe_float((row or {}).get("success_mean")),
        collision_mean=_first_metric_value(row, "collision_mean", "collisions_mean"),
        snqi_mean=_safe_float((row or {}).get("snqi_mean")),
    )


def _compare_campaigns(
    base_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    base_rows = _planner_rows_by_key(base_summary)
    candidate_rows = _planner_rows_by_key(candidate_summary)
    run_lookup = _run_entries_by_key(candidate_summary)
    common_planners = sorted(set(base_rows) & set(candidate_rows))
    output: list[dict[str, Any]] = []
    for planner_key in common_planners:
        base_row = base_rows[planner_key]
        candidate_row = candidate_rows[planner_key]
        run_entry = run_lookup.get(planner_key)
        deltas = {}
        for metric in ("success_mean", "collision_mean", "snqi_mean"):
            if metric == "collision_mean":
                base_value = _first_metric_value(base_row, "collision_mean", "collisions_mean")
                candidate_value = _first_metric_value(
                    candidate_row,
                    "collision_mean",
                    "collisions_mean",
                )
            else:
                base_value = _safe_float(base_row.get(metric))
                candidate_value = _safe_float(candidate_row.get(metric))
            if base_value is None or candidate_value is None:
                continue
            deltas[metric] = {
                "base": base_value,
                "candidate": candidate_value,
                "delta": candidate_value - base_value,
            }
        contract = ((run_entry or {}).get("summary") or {}).get("algorithm_metadata_contract") or {}
        planner_contract = contract.get("planner_kinematics") or {}
        feasibility = contract.get("kinematics_feasibility") or {}
        adapter_impact = contract.get("adapter_impact") or {}
        output.append(
            {
                "planner_key": planner_key,
                "execution_mode": str(planner_contract.get("execution_mode", "unknown")),
                "adapter_fraction": _safe_float(adapter_impact.get("adapter_fraction")),
                "projection_rate": _safe_float(feasibility.get("projection_rate")),
                "mean_abs_delta_linear": _safe_float(feasibility.get("mean_abs_delta_linear")),
                "mean_abs_delta_angular": _safe_float(feasibility.get("mean_abs_delta_angular")),
                "metrics": deltas,
            }
        )
    return output


def _write_grid_csv(grid: dict[str, Any], out_path: Path) -> Path:
    """Write the sampled grid error surfaces to CSV."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    theta_values = np.asarray(grid["theta_values"], dtype=float)
    omega_values = np.asarray(grid["omega_values"], dtype=float)
    error_dx = np.asarray(grid["error_dx"], dtype=float)
    error_dy = np.asarray(grid["error_dy"], dtype=float)
    error_norm_d = np.asarray(grid["error_norm_d"], dtype=float)
    relative_speed_error = np.asarray(grid["relative_speed_error"], dtype=float)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "theta_rad",
                "omega_rad_per_s",
                "error_dx_m",
                "error_dy_m",
                "error_norm_m",
                "relative_speed_error",
            ]
        )
        for j, omega in enumerate(omega_values):
            for i, theta in enumerate(theta_values):
                writer.writerow(
                    [
                        float(theta),
                        float(omega),
                        float(error_dx[j, i]),
                        float(error_dy[j, i]),
                        float(error_norm_d[j, i]),
                        float(relative_speed_error[j, i]),
                    ]
                )
    return out_path


def _render_heatmap_figure(grid: dict[str, Any], out_path: Path) -> Path | None:
    """Render a heatmap figure for the projection error surfaces."""
    if plt is None:
        return None
    theta_values = np.asarray(grid["theta_values"], dtype=float)
    omega_values = np.asarray(grid["omega_values"], dtype=float)
    error_dx = np.asarray(grid["error_dx"], dtype=float)
    error_dy = np.asarray(grid["error_dy"], dtype=float)
    error_norm_d = np.asarray(grid["error_norm_d"], dtype=float)
    relative_speed_error = np.asarray(grid["relative_speed_error"], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=160, constrained_layout=True)
    panels = [
        (axes[0, 0], error_dx, "endpoint error $x$ [m]", "coolwarm"),
        (axes[0, 1], error_dy, "endpoint error $y$ [m]", "coolwarm"),
        (axes[1, 0], error_norm_d, "endpoint error norm [m]", "magma"),
        (axes[1, 1], relative_speed_error, "relative speed error", "viridis"),
    ]
    extent = (
        float(theta_values.min()),
        float(theta_values.max()),
        float(omega_values.min()),
        float(omega_values.max()),
    )
    for ax, data, title, cmap in panels:
        im = ax.imshow(
            data,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap=cmap,
        )
        ax.set_title(title)
        ax.set_xlabel("heading $\\theta$ [rad]")
        ax.set_ylabel("$\\omega$ [rad/s]")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Holonomic midpoint projection error")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _render_markdown(payload: dict[str, Any]) -> str:
    """Render a human-readable markdown report."""
    command = payload["command_sample"]
    grid = payload["grid_summary"]
    lines = [
        "# Holonomic Adapter Error Analysis",
        "",
        "## Exact Projection Formula",
        "",
        "For a unicycle command `(v, omega)` at heading `theta` over step `dt`, the current "
        "holonomic bridge uses midpoint heading projection:",
        "",
        "```text",
        "vx_approx = v * cos(theta + omega * dt / 2)",
        "vy_approx = v * sin(theta + omega * dt / 2)",
        "```",
        "",
        "The exact constant-control average velocity over the same step is:",
        "",
        "```text",
        "vx_exact = v * sinc(omega * dt / 2) * cos(theta + omega * dt / 2)",
        "vy_exact = v * sinc(omega * dt / 2) * sin(theta + omega * dt / 2)",
        "```",
        "",
        "So the midpoint adapter changes only the magnitude, not the direction, and the"
        " relative shrink factor is `1 - sinc(omega * dt / 2)`.",
        "",
        "## Sample Command",
        "",
        f"- `v`: `{command['v']:.6f}`",
        f"- `omega`: `{command['omega']:.6f}` rad/s",
        f"- `heading`: `{command['heading_rad']:.6f}` rad",
        f"- `dt`: `{command['dt']:.6f}` s",
        f"- exact `(vx, vy)`: `({command['exact_vx']:.6f}, {command['exact_vy']:.6f})`",
        f"- midpoint `(vx, vy)`: `({command['approx_vx']:.6f}, {command['approx_vy']:.6f})`",
        f"- endpoint error norm: `{command['error_norm_d']:.8f}` m",
        f"- relative speed error: `{command['relative_speed_error']:.8f}`",
        "",
        "## Grid Summary",
        "",
        f"- theta samples: `{grid['theta_samples']}`",
        f"- omega samples: `{grid['omega_samples']}`",
        f"- theta range: `[{grid['theta_min']:.6f}, {grid['theta_max']:.6f}]`",
        f"- omega range: `[{grid['omega_min']:.6f}, {grid['omega_max']:.6f}]` rad/s",
        f"- max endpoint error: `{grid['max_error_norm_d']:.8f}` m",
        f"- mean endpoint error: `{grid['mean_error_norm_d']:.8f}` m",
        f"- p95 endpoint error: `{grid['p95_error_norm_d']:.8f}` m",
        f"- max relative speed error: `{grid['max_relative_speed_error']:.8f}`",
        f"- mean relative speed error: `{grid['mean_relative_speed_error']:.8f}`",
        "",
    ]
    campaign = payload.get("campaign")
    if isinstance(campaign, dict):
        lines.extend(
            [
                "## Campaign Analysis",
                "",
                f"- candidate campaign: `{campaign['candidate_campaign_id']}`",
                f"- candidate root: `{campaign['candidate_campaign_root']}`",
            ]
        )
        if campaign.get("baseline_campaign_id"):
            lines.append(f"- baseline campaign: `{campaign['baseline_campaign_id']}`")
            lines.append(f"- baseline root: `{campaign['baseline_campaign_root']}`")
        lines.extend(
            [
                "",
                "| planner | mode | adapter fraction | projection rate | mean |Δlinear| | mean |Δangular| | success | collision | SNQI | note |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for planner in campaign.get("planners", []):
            note = planner.get("note", "")
            lines.append(
                "| "
                f"{planner['planner_key']} | {planner['execution_mode']} | "
                f"{_format_optional(planner.get('adapter_fraction'))} | "
                f"{_format_optional(planner.get('projection_rate'))} | "
                f"{_format_optional(planner.get('mean_abs_delta_linear'))} | "
                f"{_format_optional(planner.get('mean_abs_delta_angular'))} | "
                f"{_format_optional(planner.get('success_mean'))} | "
                f"{_format_optional(planner.get('collision_mean'))} | "
                f"{_format_optional(planner.get('snqi_mean'))} | "
                f"{note} |"
            )
        lines.append("")
        if campaign.get("comparison"):
            lines.extend(
                [
                    "## Baseline Comparison",
                    "",
                    "| planner | metric | base | candidate | delta(candidate-base) |",
                    "|---|---|---:|---:|---:|",
                ]
            )
            for row in campaign["comparison"]:
                metrics = row.get("metrics", {})
                for metric_name, values in metrics.items():
                    lines.append(
                        "| "
                        f"{row['planner_key']} | {metric_name} | "
                        f"{values['base']:.6f} | {values['candidate']:.6f} | "
                        f"{values['delta']:.6f} |"
                    )
        lines.extend(["", "## Interpretation", ""])
        lines.extend(
            f"- {line}" for line in campaign.get("interpretation", ["No interpretation available."])
        )
        lines.append("")
    return "\n".join(lines) + "\n"


def _format_optional(value: Any) -> str:
    parsed = _safe_float(value)
    if parsed is None:
        return "N/A"
    return f"{parsed:.6f}"


def analyze(  # noqa: PLR0913
    *,
    v: float,
    omega: float,
    heading: float,
    dt: float,
    theta_min: float,
    theta_max: float,
    theta_samples: int,
    omega_min: float,
    omega_max: float,
    omega_samples: int,
    campaign_root: Path | None = None,
    baseline_campaign_root: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the full analysis pipeline."""
    sample = analyze_command(v=v, omega=omega, heading=heading, dt=dt)
    grid = sample_grid(
        v=v,
        dt=dt,
        theta_min=theta_min,
        theta_max=theta_max,
        theta_samples=theta_samples,
        omega_min=omega_min,
        omega_max=omega_max,
        omega_samples=omega_samples,
    )
    grid_summary = summarize_grid(grid)

    candidate_summary = _read_summary(campaign_root) if campaign_root is not None else None
    baseline_summary = (
        _read_summary(baseline_campaign_root) if baseline_campaign_root is not None else None
    )

    campaign_payload: dict[str, Any] | None = None
    if candidate_summary is not None:
        planner_rows = _planner_rows_by_key(candidate_summary)
        run_entries = _run_entries_by_key(candidate_summary)
        planners: list[dict[str, Any]] = []
        for planner_key in sorted(planner_rows):
            row = planner_rows[planner_key]
            run_entry = run_entries.get(planner_key)
            planner_summary = _planner_summary(
                planner_key=planner_key,
                row=row,
                run_entry=run_entry,
            )
            note = "adapter projection not observed"
            if (planner_summary.adapter_fraction or 0.0) > 0.0 or (
                planner_summary.projection_rate or 0.0
            ) > 0.0:
                note = "adapter projection observed"
            elif planner_summary.execution_mode == "native":
                note = "native execution"
            elif planner_summary.adapter_status == "disabled":
                note = "adapter impact not requested"
            planners.append({**asdict(planner_summary), "note": note})
        comparison = (
            _compare_campaigns(baseline_summary, candidate_summary)
            if baseline_summary is not None
            else []
        )
        interpretation = _build_interpretation(planners, comparison)
        campaign_payload = {
            "candidate_campaign_id": str(
                (candidate_summary.get("campaign") or {}).get("campaign_id", campaign_root.name)
                if campaign_root is not None
                else "unknown"
            ),
            "candidate_campaign_root": str(campaign_root) if campaign_root is not None else None,
            "baseline_campaign_id": str(
                (baseline_summary.get("campaign") or {}).get(
                    "campaign_id", baseline_campaign_root.name
                )
                if baseline_summary is not None and baseline_campaign_root is not None
                else ""
            )
            if baseline_summary is not None
            else None,
            "baseline_campaign_root": str(baseline_campaign_root)
            if baseline_campaign_root is not None
            else None,
            "planners": planners,
            "comparison": comparison,
            "interpretation": interpretation,
        }

    payload = {
        "command_sample": asdict(sample),
        "grid_summary": asdict(grid_summary),
        "grid": {
            "theta_min": theta_min,
            "theta_max": theta_max,
            "theta_samples": int(theta_samples),
            "omega_min": omega_min,
            "omega_max": omega_max,
            "omega_samples": int(omega_samples),
        },
    }
    if campaign_payload is not None:
        payload["campaign"] = campaign_payload

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_grid_csv(grid, output_dir / "grid_samples.csv")
        heatmap_path = _render_heatmap_figure(grid, output_dir / "error_heatmap.png")
        if heatmap_path is not None:
            payload["artifacts"] = {
                "grid_csv": str(output_dir / "grid_samples.csv"),
                "heatmap_png": str(heatmap_path),
            }
        else:
            payload["artifacts"] = {"grid_csv": str(output_dir / "grid_samples.csv")}
        (output_dir / "analysis.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output_dir / "analysis.md").write_text(_render_markdown(payload), encoding="utf-8")
    return payload


def _build_interpretation(
    planners: list[dict[str, Any]], comparison: list[dict[str, Any]]
) -> list[str]:
    lines: list[str] = []
    adapter_sensitive = [
        p
        for p in planners
        if (p.get("adapter_fraction") or 0.0) > 0.0 or (p.get("projection_rate") or 0.0) > 0.0
    ]
    native_only = [
        p
        for p in planners
        if (
            (p.get("adapter_fraction") or 0.0) == 0.0
            and (p.get("projection_rate") or 0.0) == 0.0
            and str(p.get("execution_mode", "")).strip().lower() == "native"
        )
    ]
    if adapter_sensitive:
        lines.append(
            "At least one planner records nonzero adapter projection; that planner family is "
            "eligible for adapter-induced motion error analysis."
        )
    else:
        lines.append(
            "No planner in the candidate campaign records nonzero adapter projection or "
            "projection-rate burden, so the midpoint adapter error is not the main measured "
            "source of degradation in this run."
        )
    if native_only:
        lines.append(
            f"{len(native_only)} planner(s) ran with native execution and zero recorded projection "
            "burden; regressions for those planners are more likely due to dynamics, policy "
            "quality, or scenario difficulty than to the midpoint adapter math."
        )
    if comparison:
        worse = []
        for row in comparison:
            metrics = row.get("metrics", {})
            success = metrics.get("success_mean")
            if isinstance(success, dict) and success.get("delta", 0.0) < 0.0:
                worse.append(row["planner_key"])
        if worse:
            lines.append(
                "The baseline comparison shows success-rate regressions for "
                + ", ".join(f"`{key}`" for key in worse)
                + "; these regressions persist even when adapter-projection burden is zero for "
                "some planners, which points away from the midpoint adapter as the sole cause."
            )
    if not lines:
        lines.append("No campaign data was provided.")
    return lines


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v", type=float, default=1.0, help="Command linear speed.")
    parser.add_argument("--omega", type=float, default=1.0, help="Command angular speed.")
    parser.add_argument(
        "--heading-rad",
        type=float,
        default=0.0,
        help="Current robot heading in radians for the sample command.",
    )
    parser.add_argument("--dt", type=float, default=0.1, help="Simulation step in seconds.")
    parser.add_argument(
        "--theta-min",
        type=float,
        default=-math.pi,
        help="Minimum heading value for the heatmap grid.",
    )
    parser.add_argument(
        "--theta-max",
        type=float,
        default=math.pi,
        help="Maximum heading value for the heatmap grid.",
    )
    parser.add_argument(
        "--theta-samples",
        type=int,
        default=181,
        help="Number of heading samples for the heatmap grid.",
    )
    parser.add_argument(
        "--omega-min",
        type=float,
        default=-4.0,
        help="Minimum angular-speed value for the heatmap grid.",
    )
    parser.add_argument(
        "--omega-max",
        type=float,
        default=4.0,
        help="Maximum angular-speed value for the heatmap grid.",
    )
    parser.add_argument(
        "--omega-samples",
        type=int,
        default=161,
        help="Number of angular-speed samples for the heatmap grid.",
    )
    parser.add_argument(
        "--campaign-root",
        type=Path,
        default=None,
        help="Optional camera-ready campaign root to summarize.",
    )
    parser.add_argument(
        "--baseline-campaign-root",
        type=Path,
        default=None,
        help="Optional baseline campaign root for a side-by-side comparison.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for analysis artifacts. Defaults to campaign-root/reports/holonomic_adapter_error.",
    )
    return parser


def main() -> int:
    """CLI entry point."""
    args = _build_parser().parse_args()
    campaign_root = args.campaign_root.resolve() if args.campaign_root is not None else None
    baseline_campaign_root = (
        args.baseline_campaign_root.resolve() if args.baseline_campaign_root is not None else None
    )
    output_dir = args.output_dir
    if output_dir is None and campaign_root is not None:
        output_dir = campaign_root / "reports" / "holonomic_adapter_error"
    if output_dir is not None:
        output_dir = output_dir.resolve()

    payload = analyze(
        v=float(args.v),
        omega=float(args.omega),
        heading=float(args.heading_rad),
        dt=float(args.dt),
        theta_min=float(args.theta_min),
        theta_max=float(args.theta_max),
        theta_samples=int(args.theta_samples),
        omega_min=float(args.omega_min),
        omega_max=float(args.omega_max),
        omega_samples=int(args.omega_samples),
        campaign_root=campaign_root,
        baseline_campaign_root=baseline_campaign_root,
        output_dir=output_dir,
    )

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
