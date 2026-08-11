"""Reduced, publication-facing case figure renderer.

The audit package remains the complete machine ledger.  This renderer consumes
only an admitted/proposed case trace and produces a deliberately small figure:
world panels above three absolute-time tracks below.
"""

# Matplotlib is deliberately optional at import time; this renderer is an
# artifact boundary rather than a simulator dependency.
# ruff: noqa: DOC201, PLC0415

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from robot_sf.common.optional_import import try_import


def render_publication_figure(
    package: str | Path,
    *,
    case_id: str | None = None,
    output: str | Path,
    output_format: str = "pdf",
) -> dict[str, Any]:
    """Render a deterministic reduced figure and sidecar metadata."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    package_path = Path(package)
    proposal = json.loads((package_path / "proposal.json").read_text(encoding="utf-8"))
    all_cases = [case for case in proposal.get("portfolio", []) if isinstance(case, dict)]
    cases = all_cases
    if case_id:
        selected = next(
            (case for case in all_cases if str(case.get("case_id")) == str(case_id)), None
        )
        if selected is None:
            cases = []
        else:
            pair_ids = selected.get("comparison_pair_ids")
            if isinstance(pair_ids, list) and pair_ids:
                pair_id_set = {str(value) for value in pair_ids}
                cases = [case for case in all_cases if str(case.get("case_id")) in pair_id_set]
            else:
                cases = [selected]
    if not cases:
        raise ValueError(f"no proposed case found in package: {case_id or '<any>'}")
    selected = cases[:2]
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7.0, 6.2), constrained_layout=False)
    grid = fig.add_gridspec(
        4,
        max(2, len(selected)),
        height_ratios=[1.35, 0.65, 0.65, 0.65],
        hspace=0.72,
        wspace=0.35,
    )
    world_axes = [fig.add_subplot(grid[0, index]) for index in range(max(2, len(selected)))]
    clearance_ax = fig.add_subplot(grid[1, :])
    speed_ax = fig.add_subplot(grid[2, :])
    turn_ax = fig.add_subplot(grid[3, :])
    fig.subplots_adjust(left=0.16, right=0.97, top=0.91, bottom=0.09)

    traces = []
    for index, case in enumerate(selected):
        trace = case.get("trace") if isinstance(case.get("trace"), dict) else None
        traces.append(trace)
    map_geometry = _resolve_map_geometry(selected)
    world_limits = _world_limits(traces, map_geometry)
    for index, case in enumerate(selected):
        _draw_world(world_axes[index], case, traces[index], map_geometry, world_limits)
    if len(selected) == 1:
        world_axes[1].set_axis_off()
        world_axes[1].text(
            0.5, 0.5, "second trace\nunavailable", ha="center", va="center", color="#777"
        )
    _draw_tracks(clearance_ax, speed_ax, turn_ax, selected, traces)
    clearance_ax.set_ylabel("surface clearance [m]")
    speed_ax.set_ylabel("applied speed [m/s]")
    turn_ax.set_ylabel("turn rate [rad/s]")
    turn_ax.set_xlabel("absolute time [s]")
    clearance_ax.axhline(0.0, color="#555", linewidth=0.7, linestyle="--")
    clearance_ax.grid(True, alpha=0.25)
    speed_ax.grid(True, alpha=0.25)
    turn_ax.grid(True, alpha=0.25)
    scenario_label = str(selected[0].get("scenario_id") or "scenario")
    fig.suptitle(f"Observed {scenario_label} case comparison", fontsize=11)
    metadata = {
        "Title": "RobotSF case workbench publication figure",
        "Creator": "robot_sf case-workbench.v1",
        "Subject": "Observed trajectories and applied controls",
        "CreationDate": None,
    }
    fig.savefig(output_path, format=output_format, metadata=metadata)
    plt.close(fig)
    sidecar = output_path.with_suffix(output_path.suffix + ".json")
    payload = {
        "schema_version": "case-publication-figure.v1",
        "case_ids": [case.get("case_id") for case in selected],
        "release_cell_counts": "see campaign-result-store.v2/cells.parquet",
        "evidence_grain": "exact recorded trace; proposal not author-admitted",
        "shared_prefix": False,
        "claim_boundary": "Observed result only; competing explanations remain open; no causal pivot.",
        "source_hashes": [case.get("provenance", {}).get("artifact_sha256") for case in selected],
        "map_hashes": [trace.get("map_digest") if trace else None for trace in traces],
        "observed_result": "Recorded world trajectories and applied controls for the selected cases.",
        "competing_explanation": "Seed, start-state, and other unrecorded factors remain possible explanations.",
        "generalization_limit": "Do not generalize beyond the exact scenario cell and trace package.",
        "panels": {
            "world": {
                "status": "available" if any(traces) else "unavailable",
                "shared_scale": world_limits is not None,
                "map_geometry": "available" if map_geometry is not None else "unavailable",
            },
            "surface_clearance": {
                "status": "available"
                if any(_series(trace, "clearance") for trace in traces)
                else "unavailable"
            },
            "applied_speed": {
                "status": "available"
                if any(_series(trace, "speed") for trace in traces)
                else "unavailable"
            },
            "applied_turn_rate": {
                "status": "available"
                if any(_series(trace, "turn") for trace in traces)
                else "unavailable"
            },
        },
        "forbidden_compositions": [
            "normalized_duration",
            "difference_curve",
            "first_divergence",
            "causal_pivot",
            "dual_y_axes",
        ],
    }
    sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _draw_world(
    axis: Any,
    case: dict[str, Any],
    trace: dict[str, Any] | None,
    map_geometry: Any | None,
    world_limits: tuple[tuple[float, float], tuple[float, float]] | None,
) -> None:
    """Draw a world-coordinate panel from the exact trace."""

    axis.set_title(f"{case.get('planner', 'planner')} · seed {case.get('seed')}", fontsize=9)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, alpha=0.2)
    if world_limits is not None:
        axis.set_xlim(*world_limits[0])
        axis.set_ylim(*world_limits[1])
    _draw_map_geometry(axis, map_geometry, world_limits)
    if not trace:
        axis.text(
            0.5,
            0.5,
            "trace unavailable",
            transform=axis.transAxes,
            ha="center",
            va="center",
            color="#777",
        )
        return
    steps = [step for step in trace.get("steps", []) if isinstance(step, dict)]
    robot_xy = [
        step.get("robot", {}).get("position")
        for step in steps
        if isinstance(step.get("robot"), dict)
    ]
    robot_xy = [xy for xy in robot_xy if isinstance(xy, list) and len(xy) >= 2]
    if robot_xy:
        xy = np.asarray(robot_xy, dtype=float)
        axis.plot(xy[:, 0], xy[:, 1], color="#1f77b4", linewidth=2.0, label="robot")
        axis.scatter(xy[0, 0], xy[0, 1], color="#1f77b4", s=18)
        axis.scatter(xy[-1, 0], xy[-1, 1], color="#1f77b4", s=28, marker="s")
    _draw_actor_series(axis, steps)
    _draw_critical_snapshot(axis, _critical_step(trace))
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")


def _draw_map_geometry(
    axis: Any,
    map_geometry: Any | None,
    world_limits: tuple[tuple[float, float], tuple[float, float]] | None,
) -> None:
    """Draw reusable static obstacles when the scenario map resolves."""

    if map_geometry is None or world_limits is None:
        return
    trace_scene = try_import("robot_sf.benchmark.trace_scene_figure")
    if trace_scene is not None:
        trace_scene._draw_obstacles(axis, map_geometry, world_limits)


def _draw_actor_series(axis: Any, steps: list[dict[str, Any]]) -> None:
    """Draw all pedestrian tracks while preserving stable actor identities."""

    actor_series: dict[str, list[list[float]]] = {}
    for step in steps:
        actors = step.get("pedestrians") if isinstance(step.get("pedestrians"), list) else []
        for actor in actors:
            if not isinstance(actor, dict):
                continue
            position = actor.get("position")
            if isinstance(position, list) and len(position) >= 2:
                actor_series.setdefault(str(actor.get("actor_id") or actor.get("id")), []).append(
                    position
                )
    for positions in actor_series.values():
        xy = np.asarray(positions, dtype=float)
        axis.plot(xy[:, 0], xy[:, 1], color="#d62728", linewidth=1.1, alpha=0.85)
        axis.scatter(xy[-1, 0], xy[-1, 1], color="#d62728", s=14)


def _draw_critical_snapshot(axis: Any, step: dict[str, Any] | None) -> None:
    """Mark robot and actors at the selected event/clearance state."""

    if step is None:
        return
    robot = step.get("robot")
    if isinstance(robot, dict):
        _scatter_position(axis, robot.get("position"), color="#111111", marker="*", size=65)
    actors = step.get("pedestrians") if isinstance(step.get("pedestrians"), list) else []
    for actor in actors:
        if isinstance(actor, dict):
            _scatter_position(
                axis,
                actor.get("position"),
                color="#f28e2b",
                marker="o",
                size=28,
            )


def _scatter_position(
    axis: Any,
    position: Any,
    *,
    color: str,
    marker: str,
    size: float,
) -> None:
    """Draw one critical-state position when it is a valid 2-D point."""

    if not isinstance(position, list) or len(position) < 2:
        return
    axis.scatter(
        position[0],
        position[1],
        color=color,
        edgecolor="#111111" if marker == "o" else "white",
        linewidth=0.6,
        marker=marker,
        s=size,
        zorder=8,
    )


def _resolve_map_geometry(cases: list[dict[str, Any]]) -> Any | None:
    """Resolve static map geometry for a single-scenario comparison."""

    scenario_ids = {
        str(case.get("scenario_id")) for case in cases if case.get("scenario_id") not in (None, "")
    }
    if len(scenario_ids) != 1:
        return None
    trace_scene = try_import("robot_sf.benchmark.trace_scene_figure")
    if trace_scene is None:
        return None
    try:
        return trace_scene._load_map_definition(next(iter(scenario_ids)))
    except (KeyError, OSError, ValueError):
        return None


def _world_limits(
    traces: list[dict[str, Any] | None],
    map_geometry: Any | None,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Return identical world limits for every panel, including map bounds."""

    points = _trace_points(traces)
    width = getattr(map_geometry, "width", None)
    height = getattr(map_geometry, "height", None)
    if isinstance(width, (int, float)) and isinstance(height, (int, float)):
        if float(width) > 0.0 and float(height) > 0.0:
            return ((0.0, float(width)), (0.0, float(height)))
    if not points:
        return None
    xs, ys = zip(*points, strict=True)
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    pad = max(0.5, 0.06 * max(max_x - min_x, max_y - min_y, 1.0))
    return ((min_x - pad, max_x + pad), (min_y - pad, max_y + pad))


def _trace_points(traces: list[dict[str, Any] | None]) -> list[tuple[float, float]]:
    """Collect valid robot and actor positions from all selected traces."""

    points: list[tuple[float, float]] = []
    for trace in traces:
        if not trace:
            continue
        for step in trace.get("steps", []):
            if not isinstance(step, dict):
                continue
            robot = step.get("robot")
            if isinstance(robot, dict):
                points.extend(_position_point(robot.get("position")))
            actors = step.get("pedestrians")
            if isinstance(actors, list):
                for actor in actors:
                    if isinstance(actor, dict):
                        points.extend(_position_point(actor.get("position")))
    return points


def _position_point(position: Any) -> list[tuple[float, float]]:
    """Convert one candidate position to a numeric point list."""

    if not isinstance(position, list) or len(position) < 2:
        return []
    return [(float(position[0]), float(position[1]))]


def _critical_step(trace: dict[str, Any]) -> dict[str, Any] | None:
    """Select the first recorded event state, otherwise the minimum-clearance state."""

    steps = [step for step in trace.get("steps", []) if isinstance(step, dict)]
    if not steps:
        return None
    event_times = _event_times(trace)
    if event_times:
        return min(
            steps,
            key=lambda step: abs(float(step.get("time_s", 0.0)) - event_times[0]),
        )
    clearances = _clearance_series(trace)
    finite = [index for index, value in enumerate(clearances) if np.isfinite(value)]
    if not finite:
        return steps[0]
    return steps[min(finite, key=lambda index: clearances[index])]


def _draw_tracks(
    clearance_ax: Any,
    speed_ax: Any,
    turn_ax: Any,
    cases: list[dict[str, Any]],
    traces: list[dict[str, Any] | None],
) -> None:
    """Draw three shared-absolute-time semantic tracks."""

    colors = ["#1f77b4", "#d62728"]
    all_times: list[float] = []
    for index, (case, trace) in enumerate(zip(cases, traces, strict=True)):
        if not trace:
            continue
        times = [
            float(step.get("time_s"))
            for step in trace.get("steps", [])
            if isinstance(step, dict) and isinstance(step.get("time_s"), (int, float))
        ]
        all_times.extend(times)
        clearances = _clearance_series(trace)
        speeds = _speed_series(trace)
        turns = _turn_series(trace)
        color = colors[index % len(colors)]
        label = f"seed {case.get('seed')}"
        if times and clearances:
            clearance_ax.plot(
                times[: len(clearances)], clearances, color=color, linewidth=1.5, label=label
            )
        if times and speeds:
            speed_ax.plot(times[: len(speeds)], speeds, color=color, linewidth=1.5, label=label)
        if times and turns:
            turn_ax.plot(times[: len(turns)], turns, color=color, linewidth=1.2, linestyle=":")
        for event_time in _event_times(trace):
            for axis in (clearance_ax, speed_ax, turn_ax):
                axis.axvline(event_time, color="#333333", linewidth=0.8, linestyle="--", alpha=0.65)
    clearance_ax.legend(loc="best", fontsize=8, frameon=False)
    speed_ax.legend(loc="best", fontsize=8, frameon=False)
    if all_times:
        lo, hi = min(all_times), max(all_times)
        pad = max(0.01, (hi - lo) * 0.04)
        for axis in (clearance_ax, speed_ax, turn_ax):
            axis.set_xlim(lo - pad, hi + pad)


def _series(trace: dict[str, Any] | None, kind: str) -> list[float]:
    """Return one plotted series by semantic name."""

    if not trace:
        return []
    if kind == "clearance":
        return _clearance_series(trace)
    if kind == "speed":
        return _speed_series(trace)
    return _turn_series(trace)


def _event_times(trace: dict[str, Any]) -> list[float]:
    """Return sorted semantic event times from the recorded trace."""

    times: set[float] = set()
    events = trace.get("events")
    if isinstance(events, list):
        for event in events:
            if isinstance(event, dict) and isinstance(event.get("time_s"), (int, float)):
                times.add(float(event["time_s"]))
    for step in trace.get("steps", []):
        if (
            not isinstance(step, dict)
            or not isinstance(step.get("events"), list)
            or not step["events"]
        ):
            continue
        if isinstance(step.get("time_s"), (int, float)):
            times.add(float(step["time_s"]))
    return sorted(times)


def _clearance_series(trace: dict[str, Any]) -> list[float]:
    """Compute surface clearance from recorded radii.

    Returns:
        One value per trace frame, with NaN where no actor is present.
    """

    values: list[float] = []
    for step in trace.get("steps", []):
        if not isinstance(step, dict) or not isinstance(step.get("robot"), dict):
            continue
        robot = step["robot"]
        rp = robot.get("position")
        if not isinstance(rp, list) or len(rp) < 2:
            continue
        rr = float(robot.get("radius_m") or 0.0)
        distances = []
        for actor in (
            step.get("pedestrians", []) if isinstance(step.get("pedestrians"), list) else []
        ):
            if not isinstance(actor, dict) or not isinstance(actor.get("position"), list):
                continue
            ap = actor["position"]
            if len(ap) < 2:
                continue
            distances.append(
                float(
                    np.linalg.norm(
                        np.asarray(rp[:2], dtype=float) - np.asarray(ap[:2], dtype=float)
                    )
                )
                - rr
                - float(actor.get("radius_m") or 0.0)
            )
        values.append(min(distances) if distances else float("nan"))
    return values


def _speed_series(trace: dict[str, Any]) -> list[float]:
    """Return the applied linear speed, with recorded velocity as a v1 fallback."""

    values: list[float] = []
    for step in trace.get("steps", []):
        if not isinstance(step, dict):
            continue
        controls = step.get("controls") if isinstance(step.get("controls"), dict) else {}
        applied = controls.get("applied") if isinstance(controls.get("applied"), dict) else {}
        applied_speed = applied.get("linear_m_s")
        if isinstance(applied_speed, (int, float)):
            values.append(float(applied_speed))
            continue
        velocity = (
            step.get("robot", {}).get("velocity") if isinstance(step.get("robot"), dict) else None
        )
        values.append(
            float(np.linalg.norm(np.asarray(velocity[:2], dtype=float)))
            if isinstance(velocity, list) and len(velocity) >= 2
            else float("nan")
        )
    return values


def _turn_series(trace: dict[str, Any]) -> list[float]:
    """Return applied turn rate when available."""

    values: list[float] = []
    for step in trace.get("steps", []):
        controls = (
            step.get("controls")
            if isinstance(step, dict) and isinstance(step.get("controls"), dict)
            else {}
        )
        applied = (
            controls.get("applied")
            if isinstance(controls, dict) and isinstance(controls.get("applied"), dict)
            else {}
        )
        value = applied.get("turn_rate_rad_s")
        values.append(float(value) if isinstance(value, (int, float)) else float("nan"))
    return values


__all__ = ["render_publication_figure"]
