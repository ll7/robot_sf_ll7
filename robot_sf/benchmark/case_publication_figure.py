"""Reduced, publication-facing case figure renderer.

The audit package remains the complete machine ledger.  This renderer consumes
only an admitted/proposed case trace and produces a deliberately small figure:
world panels above three absolute-time tracks below.
"""

# Matplotlib is deliberately optional at import time; this renderer is an
# artifact boundary rather than a simulator dependency.
# ruff: noqa: DOC201, PLC0415

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from robot_sf.benchmark.analysis_trace import canonical_json
from robot_sf.common.optional_import import try_import


def render_publication_figure(  # noqa: C901, PLR0915
    package: str | Path,
    *,
    case_id: str | None = None,
    output: str | Path,
    output_format: str = "pdf",
    _allow_unverified_preview: bool = False,
) -> dict[str, Any]:
    """Render a deterministic reduced figure and sidecar metadata."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    package_path = Path(package)
    if not _allow_unverified_preview:
        _verify_package_integrity(package_path)
        _verify_publication_gate(package_path)
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
    selected = _select_publication_cases(cases)
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
    }
    if output_format.lower() == "pdf":
        metadata.update(
            {
                "Subject": "Observed trajectories and applied controls",
                "CreationDate": None,
            }
        )
    fig.savefig(output_path, format=output_format, metadata=metadata)
    plt.close(fig)
    manifest = (
        _read_json_mapping(package_path / "manifest.json") if not _allow_unverified_preview else {}
    )
    proposal_sha = manifest.get("proposal_sha256") or _sha256_file(package_path / "proposal.json")
    config_sha = (
        (manifest.get("config") or {}).get("sha256")
        if isinstance(manifest.get("config"), dict)
        else None
    ) or _optional_sha256_file(package_path / "config.yaml")
    store_path = package_path / "campaign-result-store.v2"
    store_sha = (
        (manifest.get("source") or {}).get("sha256")
        if isinstance(manifest.get("source"), dict)
        else None
    ) or (_sha256_directory(store_path) if store_path.is_dir() else None)
    overlay_status = "proposed"
    overlay_path = package_path / "admission_overlay.json"
    if overlay_path.is_file():
        try:
            overlay_status = str(_read_json_mapping(overlay_path).get("status") or "proposed")
        except ValueError:
            overlay_status = "unavailable"
    sidecar = output_path.with_suffix(output_path.suffix + ".json")
    payload = {
        "schema_version": "case-publication-figure.v1",
        "case_ids": [case.get("case_id") for case in selected],
        "release_cell_counts": _release_cell_counts(package_path, selected),
        "evidence_grain": (
            "exact recorded trace; author-admitted package"
            if overlay_status.lower() == "admitted" and not _allow_unverified_preview
            else "exact recorded trace; diagnostic preview only; proposal not author-admitted"
        ),
        "shared_prefix": False,
        "claim_boundary": "Observed result only; competing explanations remain open; no causal pivot.",
        "source_hashes": [case.get("provenance", {}).get("artifact_sha256") for case in selected],
        "package_sha256": (
            _sha256_file(package_path / "SHA256SUMS")
            if (package_path / "SHA256SUMS").is_file()
            else None
        ),
        "proposal_sha256": proposal_sha,
        "config_sha256": config_sha,
        "store_sha256": store_sha,
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
                if any(_has_finite_series(_series(trace, "clearance")) for trace in traces)
                else "unavailable"
            },
            "applied_speed": {
                "status": "available"
                if any(_has_finite_series(_series(trace, "speed")) for trace in traces)
                else "unavailable"
            },
            "applied_turn_rate": {
                "status": "available"
                if any(_has_finite_series(_series(trace, "turn")) for trace in traces)
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


def _read_json_mapping(path: Path) -> dict[str, Any]:
    """Read a required JSON object at an artifact boundary."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"package provenance file is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"package provenance file must contain an object: {path}")
    return value


def _sha256_json(value: Any) -> str:
    """Hash canonical JSON for package-level semantic receipts."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _verify_package_integrity(package: Path) -> None:
    """Fail closed when a package manifest/checksum set does not verify."""

    manifest_path = package / "manifest.json"
    checksums_path = package / "SHA256SUMS"
    if not manifest_path.is_file() or not checksums_path.is_file():
        raise ValueError("case-workbench package is missing manifest.json or SHA256SUMS")
    _read_json_mapping(manifest_path)
    expected_files: set[str] = set()
    for line in checksums_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            expected, relative = line.split("  ", 1)
        except ValueError as exc:
            raise ValueError("case-workbench package checksum receipt is malformed") from exc
        if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
            raise ValueError(f"invalid package checksum entry: {relative}")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"invalid package checksum path: {relative}")
        target = package / relative_path
        if not target.is_file() or _sha256_file(target) != expected:
            raise ValueError(f"package checksum mismatch: {relative}")
        expected_files.add(relative)
    actual_files = {
        str(path.relative_to(package))
        for path in package.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    if actual_files != expected_files:
        missing = sorted(actual_files - expected_files)
        extra = sorted(expected_files - actual_files)
        raise ValueError(f"package checksum inventory mismatch: missing={missing}, extra={extra}")


def _verify_publication_gate(package: Path) -> None:
    """Require source restoration and author admission before publication output."""

    manifest = _read_json_mapping(package / "manifest.json")
    proposal_path = package / "proposal.json"
    proposal = _read_json_mapping(proposal_path)
    proposal_digest = _sha256_json(proposal)
    manifest_digest = manifest.get("proposal_sha256")
    if manifest_digest != proposal_digest:
        raise ValueError("publication proposal digest does not match manifest")
    gate = manifest.get("source_integrity_gate")
    from robot_sf.benchmark.case_workbench import _source_gate_is_trusted

    source_manifest = manifest.get("source")
    if (
        not isinstance(gate, dict)
        or not _source_gate_is_trusted(gate)
        or not isinstance(source_manifest, dict)
        or gate.get("source_sha256") != source_manifest.get("sha256")
    ):
        raise ValueError("publication rendering is blocked by the source-integrity gate")
    overlay = _read_json_mapping(package / "admission_overlay.json")
    if str(overlay.get("status") or "").lower() != "admitted":
        raise ValueError("publication rendering requires an admitted author overlay")
    machine_digest = manifest.get("machine_proposal_sha256")
    admission = proposal.get("author_admission")
    if not isinstance(machine_digest, str) or not isinstance(admission, dict):
        raise ValueError("publication package is missing machine proposal admission binding")
    if overlay.get("proposal_sha256") != machine_digest:
        raise ValueError("admission overlay is not bound to the machine proposal")
    if admission.get("machine_proposal_sha256") != machine_digest:
        raise ValueError("admitted proposal is not bound to the machine proposal")


def _select_publication_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Choose one case or one complete declared pair, never an arbitrary portfolio slice."""

    def compatible_pair(case: Mapping[str, Any], pair_ids: set[str]) -> bool:
        receipt = case.get("comparison_compatibility")
        return (
            isinstance(receipt, Mapping)
            and receipt.get("status") == "compatible"
            and {str(value) for value in receipt.get("pair_ids", [])} == pair_ids
            and receipt.get("shared_prefix") is False
        )

    if len(cases) == 1:
        return cases
    if len(cases) == 2:
        ids = {str(case.get("case_id")) for case in cases}
        if all(
            isinstance(case.get("comparison_pair_ids"), list)
            and ids.issubset({str(value) for value in case["comparison_pair_ids"]})
            and compatible_pair(case, ids)
            for case in cases
        ):
            return sorted(cases, key=lambda item: str(item.get("case_id")))
        raise ValueError("publication composition requires an explicit compatible case pair")
    for case in cases:
        pair_ids = case.get("comparison_pair_ids")
        if not isinstance(pair_ids, list) or len(pair_ids) != 2:
            continue
        ids = {str(value) for value in pair_ids}
        pair = [candidate for candidate in cases if str(candidate.get("case_id")) in ids]
        if len(pair) == 2 and all(compatible_pair(candidate, ids) for candidate in pair):
            return sorted(pair, key=lambda item: str(item.get("case_id")))
    raise ValueError("publication composition requires an explicit compatible case pair")


def _release_cell_counts(package: Path, cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Copy selected cell outcome context into the sidecar when the store exposes it."""

    store = package / "campaign-result-store.v2" / "cells.parquet"
    if not store.is_file() or not cases:
        return {"status": "unavailable", "reason": "cell_table_missing"}
    try:
        pyarrow = try_import("pyarrow.parquet")
        if pyarrow is None:
            return {"status": "unavailable", "reason": "pyarrow_missing"}
        rows = pyarrow.read_table(store).to_pylist()
    except (OSError, ValueError, AttributeError):
        return {"status": "unavailable", "reason": "cell_table_unreadable"}
    selected_keys = {
        (
            str(case.get("planner")),
            str(case.get("scenario_id")),
            str(case.get("config_hash") or ""),
            str(case.get("config_digest") or ""),
            str(case.get("scenario_digest") or ""),
            str(case.get("map_digest") or ""),
        )
        for case in cases
    }
    matching = [
        {
            "planner": row.get("planner"),
            "scenario_id": row.get("scenario_id"),
            "config_hash": row.get("config_hash"),
            "config_digest": row.get("config_digest"),
            "scenario_digest": row.get("scenario_digest"),
            "map_digest": row.get("map_digest"),
            "outcome_counts": _decode_json(row.get("outcome_counts_json")),
            "entropy": row.get("entropy"),
            "seed_count": row.get("seed_count"),
            "uncertainty": _decode_json(row.get("uncertainty_json")),
            "boundary_context": _decode_json(row.get("boundary_context_json")),
        }
        for row in rows
        if (
            str(row.get("planner")),
            str(row.get("scenario_id")),
            str(row.get("config_hash") or ""),
            str(row.get("config_digest") or ""),
            str(row.get("scenario_digest") or ""),
            str(row.get("map_digest") or ""),
        )
        in selected_keys
    ]
    return (
        {"status": "available", "cells": matching}
        if matching
        else {
            "status": "unavailable",
            "reason": "selected_cell_not_found",
        }
    )


def _decode_json(value: Any) -> Any:
    """Decode a JSON column for sidecar serialization."""

    if value is None:
        return None
    try:
        return json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return value


def _sha256_file(path: Path) -> str:
    """Hash one package file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optional_sha256_file(path: Path) -> str | None:
    """Hash an optional package file without turning absence into an exception."""

    return _sha256_file(path) if path.is_file() else None


def _sha256_directory(path: Path) -> str | None:
    """Hash a package directory by sorted relative names and bytes."""

    if not path.is_dir():
        return None
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(child.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(child.read_bytes())
    return digest.hexdigest()


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

    traces = [case.get("trace") for case in cases if isinstance(case.get("trace"), dict)]
    map_paths = {str(trace.get("map_file")) for trace in traces if trace.get("map_file")}
    map_hashes = {str(trace.get("map_digest")) for trace in traces if trace.get("map_digest")}
    if len(map_paths) != 1 or len(map_hashes) != 1:
        return None
    map_path = Path(next(iter(map_paths)))
    candidates = [map_path]
    if not map_path.is_absolute():
        candidates.extend([Path.cwd() / map_path, Path(__file__).resolve().parents[2] / map_path])
    resolved = next((candidate for candidate in candidates if candidate.is_file()), None)
    if resolved is None:
        return None
    try:
        digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
        if digest != next(iter(map_hashes)):
            return None
        svg_map = try_import("robot_sf.nav.svg_map_parser")
        if svg_map is None:
            return None
        return svg_map.SvgMapConverter(str(resolved)).map_definition
    except (KeyError, OSError, ValueError, AttributeError):
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
    if not all(isinstance(value, (int, float)) for value in position[:2]):
        return []
    point = (float(position[0]), float(position[1]))
    return [point] if all(np.isfinite(value) for value in point) else []


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


def _has_finite_series(values: list[float]) -> bool:
    """Return whether a plotted series contains at least one finite observation."""

    return any(np.isfinite(value) for value in values)


def _event_times(trace: dict[str, Any]) -> list[float]:
    """Return sorted semantic event times from the recorded trace."""

    times: set[float] = set()
    events = trace.get("events")
    if isinstance(events, list):
        for event in events:
            if isinstance(event, dict):
                value = event.get("time_s")
                if value is None:
                    value = event.get("collision_time")
                if isinstance(value, (int, float)):
                    times.add(float(value))
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
        rr = _finite_positive(robot.get("radius_m"))
        if rr is None or not all(
            isinstance(value, (int, float)) and np.isfinite(float(value)) for value in rp[:2]
        ):
            values.append(float("nan"))
            continue
        distances = []
        for actor in (
            step.get("pedestrians", []) if isinstance(step.get("pedestrians"), list) else []
        ):
            if not isinstance(actor, dict) or not isinstance(actor.get("position"), list):
                continue
            ap = actor["position"]
            if len(ap) < 2:
                continue
            actor_radius = _finite_positive(actor.get("radius_m"))
            if actor_radius is None or not all(
                isinstance(value, (int, float)) and np.isfinite(float(value)) for value in ap[:2]
            ):
                continue
            distances.append(
                float(
                    np.linalg.norm(
                        np.asarray(rp[:2], dtype=float) - np.asarray(ap[:2], dtype=float)
                    )
                )
                - rr
                - actor_radius
            )
        values.append(min(distances) if distances else float("nan"))
    return values


def _finite_positive(value: Any) -> float | None:
    """Return a finite positive geometry value, or typed unavailable."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if np.isfinite(numeric) and numeric > 0.0 else None


def _speed_series(trace: dict[str, Any]) -> list[float]:
    """Return only explicitly recorded applied linear speed controls."""

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
        values.append(float("nan"))
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
