"""Deterministic, evidence-gated Chapter 7 case dossier renderer.

The renderer composes validated Chapter 7 portfolio selections, mutually bound
``worked_example_process_trace.v1`` records, and ``campaign_atlas.v2`` cell
context.  It is a visualization-integrity surface only: rendering never admits
scientific evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from jsonschema import Draft202012Validator
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib.text import Text

from robot_sf.analysis_workbench.interaction_coordinates import (
    validate_worked_example_process_trace,
)
from robot_sf.benchmark.case_portfolio import validate_ch7_worked_example_portfolio
from robot_sf.benchmark.figure_qa import assert_clean

CASE_DOSSIER_INPUT_SCHEMA_VERSION = "case_dossier_input.v1"
CASE_DOSSIER_MANIFEST_SCHEMA_VERSION = "case_dossier_manifest.v1"
CASE_DOSSIER_RENDERER_VERSION = "case_dossier_figure.v1"
CASE_DOSSIER_STYLE_VERSION = "case_dossier_final_width.v1"
SYNTHETIC_FIXTURE_LABEL = "SYNTHETIC FIXTURE — RENDERER PROOF ONLY"
FINAL_WIDTH_IN = 426.79135 / 72.27
BASE_FONT_PT = 9.0
MINIMUM_VISIBLE_FONT_PT = 8.25
ROUTE_TITLE_BAND_IN = 0.20

PALETTE = {
    "left": "#0072B2",
    "right": "#D55E00",
    "focal": "#CC79A7",
    "context": "#7F7F7F",
    "collision": "#000000",
    "threshold": "#009E73",
}
LINESTYLES = {"left": "-", "right": "--"}
MARKERS = {"left": "o", "right": "s"}
_RC = {
    "font.family": "DejaVu Sans",
    "font.size": BASE_FONT_PT,
    "axes.titlesize": BASE_FONT_PT,
    "axes.labelsize": MINIMUM_VISIBLE_FONT_PT,
    "xtick.labelsize": MINIMUM_VISIBLE_FONT_PT,
    "ytick.labelsize": MINIMUM_VISIBLE_FONT_PT,
    "legend.fontsize": MINIMUM_VISIBLE_FONT_PT,
    "savefig.bbox": None,
    "svg.fonttype": "none",
    "svg.hashsalt": "case-dossier-figure-v1",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


class CaseDossierError(ValueError):
    """Fail-closed dossier input or rendering error with a stable code."""

    def __init__(self, code: str, detail: str):
        """Initialize an error with a stable machine-readable code."""

        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True, slots=True)
class CaseDossierBundle:
    """Paths and manifest produced by one dossier render."""

    svg_path: Path
    pdf_path: Path
    caption_path: Path
    sidecar_path: Path
    manifest_path: Path
    catalog_path: Path
    manifest: dict[str, Any]


def _schema_path(name: str) -> Path:
    return Path(__file__).with_name("schemas") / name


def _load_schema(name: str) -> dict[str, Any]:
    return json.loads(_schema_path(name).read_text(encoding="utf-8"))


def _schema_errors(payload: Any, schema_name: str) -> list[str]:
    validator = Draft202012Validator(_load_schema(schema_name))
    return [
        f"/{'/'.join(str(part) for part in error.absolute_path)}: {error.message}"
        for error in sorted(
            validator.iter_errors(payload), key=lambda item: list(item.absolute_path)
        )
    ]


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str:
    """Return the checkout commit used to generate the dossier metadata."""

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    commit = completed.stdout.strip()
    return commit if len(commit) >= 7 else "unknown"


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def _write_json(path: Path, payload: Any) -> None:
    _atomic_write(
        path,
        (json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def _resolve_ref(input_path: Path, ref: dict[str, Any]) -> Path:
    raw = Path(str(ref["path"]))
    path = raw if raw.is_absolute() else input_path.parent / raw
    path = path.resolve()
    if not path.is_file():
        raise CaseDossierError("source_file_unavailable", str(path))
    observed = _file_sha256(path)
    if observed != ref["sha256"]:
        raise CaseDossierError(
            "source_sha256_mismatch",
            f"{path}: expected {ref['sha256']}, observed {observed}",
        )
    return path


def _portable_source_path(input_path: Path, resolved_path: Path) -> str:
    """Represent a validated source relative to its dossier input tree.

    Returns:
        A POSIX path relative to the dossier input directory.
    """

    return Path(os.path.relpath(resolved_path, start=input_path.parent)).as_posix()


def _read_mapping(path: Path, *, code: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CaseDossierError(code, f"{path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CaseDossierError(code, f"{path}: expected object")
    return payload


def _selected_case(portfolio: dict[str, Any], case_id: str) -> dict[str, Any]:
    validation = validate_ch7_worked_example_portfolio(portfolio)
    if not validation.ok:
        raise CaseDossierError(
            "portfolio_manifest_invalid",
            "; ".join(validation.structural_violations),
        )
    selected = [item for item in portfolio["selected"] if item.get("case_id") == case_id]
    if len(selected) != 1:
        raise CaseDossierError("production_case_not_selected", case_id)
    return selected[0]


def _validate_input_semantics(payload: dict[str, Any]) -> None:
    layout = payload["layout"]
    if not math.isclose(float(layout["final_width_in"]), FINAL_WIDTH_IN, abs_tol=1e-9):
        raise CaseDossierError(
            "final_width_mismatch",
            f"expected {FINAL_WIDTH_IN:.9f}in",
        )
    if float(layout["minimum_font_pt"]) != MINIMUM_VISIBLE_FONT_PT:
        raise CaseDossierError("minimum_font_mismatch", f"expected {MINIMUM_VISIBLE_FONT_PT}pt")
    for key in ("time_range_s", "clearance_range_m", "speed_range_mps"):
        lower, upper = layout[key]
        if not lower < upper:
            raise CaseDossierError("invalid_shared_scale", key)
    xmin, xmax, ymin, ymax = layout["world_crop_m"]
    if not (xmin < xmax and ymin < ymax):
        raise CaseDossierError("invalid_shared_scale", "world_crop_m")
    roles = [item["role"] for item in payload["sources"]["process_traces"]]
    if sorted(roles) != ["left", "right"]:
        raise CaseDossierError("process_trace_roles_invalid", repr(roles))
    if payload["mode"] == "synthetic_fixture" and payload["scientific_admission"] is not False:
        raise CaseDossierError("synthetic_scientific_admission_forbidden", "must be false")
    options = payload["comparison_options"]
    if payload["comparison_grammar"] == "same_cell_seed_sensitivity" and (
        options["difference_curve"]
        or options["pivot_time_s"] is not None
        or options["causal_hinge"]
        or options["adjacent_seed_significance"]
    ):
        raise CaseDossierError(
            "no_shared_prefix_forbidden_mode",
            "difference curves, pivot time, causal hinge, and adjacent-seed significance are forbidden",
        )


def _prohibited_semantics(grammar: str) -> list[str]:
    if grammar != "same_cell_seed_sensitivity":
        return []
    return [
        "adjacent_seed_significance",
        "causal_hinge",
        "difference_curve",
        "divergence_point",
        "pivot_time",
    ]


def _validate_atlas(
    atlas: dict[str, Any],
    *,
    portfolio_hash: str,
    traces: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    if atlas.get("schema_version") != "campaign_atlas.v2":
        raise CaseDossierError("campaign_atlas_schema_invalid", repr(atlas.get("schema_version")))
    if atlas.get("selection_manifest_hash") != portfolio_hash:
        raise CaseDossierError("atlas_selection_hash_mismatch", "selection_manifest_hash")
    cells = atlas.get("cells")
    if not isinstance(cells, list):
        raise CaseDossierError("campaign_atlas_cells_invalid", "cells must be an array")
    sources = [trace["source_trace"]["source"] for trace in traces.values()]
    scenario = sources[0]["scenario_id"]
    planners = {source["planner_id"] for source in sources}
    matched = [
        cell
        for cell in cells
        if isinstance(cell, dict)
        and cell.get("scenario_family") == scenario
        and cell.get("planner") in planners
    ]
    if {cell.get("planner") for cell in matched} != planners:
        raise CaseDossierError("campaign_atlas_cell_unavailable", scenario)
    for cell in matched:
        counts = cell.get("outcome_counts")
        if not isinstance(counts, dict) or sum(counts.values()) != cell.get("n_total"):
            raise CaseDossierError("campaign_atlas_denominator_mismatch", str(cell.get("planner")))
    return sorted(
        matched, key=lambda item: (str(item.get("planner")), str(item.get("release_arm_id")))
    )


def _validate_pair(
    grammar: str,
    traces: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    left = traces["left"]
    right = traces["right"]
    left_pair = left["pair_compatibility"]
    right_pair = right["pair_compatibility"]
    if left_pair.get("right_source_trace", {}).get("content_sha256") != right["source_trace"].get(
        "content_sha256"
    ) or right_pair.get("right_source_trace", {}).get("content_sha256") != left["source_trace"].get(
        "content_sha256"
    ):
        raise CaseDossierError("process_trace_pair_binding_mismatch", "reciprocal source hashes")
    expected_grain = {
        "matched_start_planner": "matched_planner_pair",
        "same_cell_seed_sensitivity": "matched_realization_pair",
    }[grammar]
    for pair in (left_pair, right_pair):
        if pair.get("comparison_grain", {}).get("grain_id") != expected_grain:
            raise CaseDossierError("comparison_grammar_mismatch", expected_grain)
        if pair.get("status") != "available":
            raise CaseDossierError("pair_compatibility_unavailable", str(pair.get("status")))
    route_symmetric_fields = (
        "status",
        "initial_robot_separation_m",
        "max_initial_actor_separation_m",
        "scenario_id_equal",
        "scenario_provenance_compatible",
    )
    reciprocal_projections = []
    for pair in (left_pair, right_pair):
        route_spawn = pair.get("route_spawn_separation", {})
        reciprocal_projections.append(
            {
                "status": pair.get("status"),
                "comparison_grain": pair.get("comparison_grain"),
                "initial_state_equivalence": pair.get("initial_state_equivalence"),
                "shared_prefix": pair.get("shared_prefix"),
                "route_spawn_separation": {
                    key: route_spawn.get(key) for key in route_symmetric_fields
                },
                "divergence_interpretation": pair.get("divergence_interpretation"),
            }
        )
    if _canonical_sha256(reciprocal_projections[0]) != _canonical_sha256(reciprocal_projections[1]):
        raise CaseDossierError(
            "reciprocal_pair_contract_disagreement",
            "grain/status/start/prefix/separation/divergence",
        )
    left_source = left["source_trace"]["source"]
    right_source = right["source_trace"]["source"]
    if grammar == "matched_start_planner":
        if (
            left_source["seed"] != right_source["seed"]
            or left_source["planner_id"] == right_source["planner_id"]
            or left_pair.get("initial_state_equivalence", {}).get("equivalent") is not True
            or left_pair.get("shared_prefix", {}).get("shared_prefix") is not True
        ):
            raise CaseDossierError("matched_start_gate_failed", "planner/seed/start/prefix")
    elif (
        left_source["seed"] == right_source["seed"]
        or left_source["planner_id"] != right_source["planner_id"]
        or left_pair.get("shared_prefix", {}).get("shared_prefix") is not False
        or left_pair.get("divergence_interpretation", {}).get("allowed") is not False
    ):
        raise CaseDossierError("no_shared_prefix_gate_failed", "planner/seed/prefix")
    return left_pair


def _validate_shared_trace_contract(
    payload: dict[str, Any],
    traces: dict[str, dict[str, Any]],
) -> None:
    """Mechanically bind pair geometry, time range, and threshold profile."""

    time_min, time_max = (float(value) for value in payload["layout"]["time_range_s"])
    for role, trace in traces.items():
        observed = [float(frame["time_s"]) for frame in trace["frames"]]
        if min(observed) < time_min or max(observed) > time_max:
            raise CaseDossierError(
                "shared_time_range_excludes_trace",
                f"{role}: observed=[{min(observed)}, {max(observed)}]",
            )

    for geometry_kind in ("route", "conflict"):
        contracts = [trace["analysis_input_contract"][geometry_kind] for trace in traces.values()]
        if _canonical_sha256(contracts[0]) != _canonical_sha256(contracts[1]):
            raise CaseDossierError(
                "shared_geometry_contract_mismatch",
                geometry_kind,
            )

    expected_profile = payload["layout"]["threshold_profile"]
    threshold_profiles = [trace["profiles"]["threshold_profile"] for trace in traces.values()]
    if any(profile.get("profile_version") != expected_profile for profile in threshold_profiles):
        raise CaseDossierError("shared_threshold_profile_mismatch", expected_profile)
    thresholds = {
        float(profile["proxy_surface_clearance_threshold_m"]) for profile in threshold_profiles
    }
    if len(thresholds) != 1:
        raise CaseDossierError("shared_threshold_value_mismatch", repr(sorted(thresholds)))


def _load_bound_input(input_path: Path) -> dict[str, Any]:
    payload = _read_mapping(input_path, code="case_dossier_input_invalid")
    errors = _schema_errors(payload, "case_dossier_input.v1.json")
    if errors:
        raise CaseDossierError("case_dossier_input_invalid", "; ".join(errors))
    _validate_input_semantics(payload)
    sources = payload["sources"]
    portfolio_path = _resolve_ref(input_path, sources["portfolio"])
    portfolio = _read_mapping(portfolio_path, code="portfolio_manifest_invalid")
    selected_case = _selected_case(portfolio, payload["case_id"])
    if payload["mode"] == "synthetic_fixture" and not bool(
        selected_case.get("source_boundary", {}).get("synthetic_fixture")
    ):
        raise CaseDossierError("synthetic_portfolio_boundary_missing", payload["case_id"])
    if payload["mode"] == "production" and bool(
        selected_case.get("source_boundary", {}).get("synthetic_fixture")
    ):
        raise CaseDossierError("production_fixture_source_forbidden", payload["case_id"])

    trace_refs = {item["role"]: item for item in sources["process_traces"]}
    traces: dict[str, dict[str, Any]] = {}
    trace_paths: dict[str, Path] = {}
    for role in ("left", "right"):
        path = _resolve_ref(input_path, trace_refs[role])
        trace = _read_mapping(path, code="process_trace_invalid")
        try:
            validate_worked_example_process_trace(trace, source=path)
        except ValueError as exc:
            raise CaseDossierError("process_trace_invalid", str(exc)) from exc
        traces[role] = trace
        trace_paths[role] = path
    _validate_shared_trace_contract(payload, traces)
    pair = _validate_pair(payload["comparison_grammar"], traces)
    atlas_path = _resolve_ref(input_path, sources["campaign_atlas"])
    atlas = _read_mapping(atlas_path, code="campaign_atlas_invalid")
    cells = _validate_atlas(
        atlas,
        portfolio_hash=portfolio["content_sha256"],
        traces=traces,
    )
    return {
        "input": payload,
        "input_path": input_path,
        "portfolio": portfolio,
        "portfolio_path": portfolio_path,
        "selected_case": selected_case,
        "trace_refs": trace_refs,
        "trace_paths": trace_paths,
        "traces": traces,
        "pair": pair,
        "atlas": atlas,
        "atlas_path": atlas_path,
        "cells": cells,
    }


def _available_events(trace: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        event
        for event in trace["event_anchors"]
        if event.get("status") == "available"
        and event.get("visual_anchor_eligibility", {}).get("eligible") is True
    ]


_EVENT_LABELS = {
    "minimum_clearance": "minimum clearance",
    "first_material_deceleration": "deceleration",
    "first_material_turn_response": "turn response",
    "conflict_zone_entry": "zone entry",
    "exact_collision_event": "exact collision",
    "first_safety_predicate_breach": "safety breach",
    "proxy_overlap_event": "proxy overlap",
    "sustained_stall_onset": "stall onset",
    "recovery_onset": "recovery onset",
    "terminal_event": "terminal event",
}
_EVENT_LINESTYLES = ((0, (2, 2)), (0, (4, 2)), (0, (1, 1)), (0, (5, 1, 1, 1)))
_EVENT_MARKERS = ("o", "s", "D", "P", "^", "v")


def _event_groups(traces: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[float, set[str]] = {}
    for trace in traces.values():
        for event in _available_events(trace):
            time_s = round(float(event["time_s"]), 9)
            grouped.setdefault(time_s, set()).add(str(event["event_type"]))
    return [
        {
            "time_s": time_s,
            "event_types": sorted(event_types),
            "label": " / ".join(_EVENT_LABELS[item] for item in sorted(event_types)),
        }
        for time_s, event_types in sorted(grouped.items())
    ]


def _event_identity_caption(traces: dict[str, dict[str, Any]]) -> str:
    """Return a compact semantic key for the cursor styles outside plot data."""

    available_types = {
        str(event["event_type"]) for trace in traces.values() for event in _available_events(trace)
    }
    labels = [label for event_type, label in _EVENT_LABELS.items() if event_type in available_types]
    if not labels:
        return "Semantic cursors: none available"
    lines = ["Semantic cursors:"]
    for label in labels:
        separator = " " if lines[-1].endswith(":") else " · "
        candidate = f"{lines[-1]}{separator}{label}"
        if len(candidate) <= 52:
            lines[-1] = candidate
        else:
            lines.append(label)
    return "\n".join(lines)


def _draw_event_cursors(ax: Any, traces: dict[str, dict[str, Any]]) -> None:
    groups = _event_groups(traces)
    for index, group in enumerate(groups):
        collision = "exact_collision_event" in group["event_types"]
        ax.axvline(
            group["time_s"],
            color=PALETTE["collision"] if collision else PALETTE["context"],
            linestyle=_EVENT_LINESTYLES[index % len(_EVENT_LINESTYLES)],
            linewidth=1.0,
            alpha=0.55,
            zorder=0,
        )


def _draw_actor_event_anchor(ax: Any, actor: dict[str, Any], color: str) -> None:
    position = actor.get("position")
    radius = actor.get("radius_m")
    velocity = actor.get("velocity")
    if isinstance(position, list) and isinstance(radius, int | float):
        ax.add_patch(Circle(position, float(radius), edgecolor=color, facecolor="none", lw=0.9))
    if (
        isinstance(position, list)
        and isinstance(velocity, list)
        and len(velocity) == 2
        and all(
            isinstance(value, int | float) and math.isfinite(float(value)) for value in velocity
        )
    ):
        ax.quiver(
            [position[0]],
            [position[1]],
            [velocity[0]],
            [velocity[1]],
            angles="xy",
            scale_units="xy",
            scale=2.0,
            color=color,
            width=0.008,
        )


def _draw_world_event_markers(
    ax: Any,
    role: str,
    frames: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> int:
    groups = _event_groups({role: {"event_anchors": events}})
    for group_index, group in enumerate(groups):
        index = min(
            range(len(frames)),
            key=lambda idx: abs(frames[idx]["time_s"] - group["time_s"]),
        )
        world = frames[index]["world"]
        _draw_actor_event_anchor(ax, world.get("robot", {}), PALETTE[role])
        _draw_actor_event_anchor(ax, world.get("focal_actor", {}), PALETTE["focal"])
        position = world.get("robot", {}).get("position")
        if isinstance(position, list) and len(position) == 2:
            collision = "exact_collision_event" in group["event_types"]
            ax.scatter(
                [position[0]],
                [position[1]],
                marker="X" if collision else _EVENT_MARKERS[group_index % len(_EVENT_MARKERS)],
                s=38 if collision else 24,
                facecolors="none",
                edgecolors=PALETTE["collision"] if collision else PALETTE[role],
                linewidths=1.2 if collision else 0.9,
                zorder=5,
            )
    return len(groups)


def _draw_world(
    ax: Any, role: str, trace: dict[str, Any], crop: list[float], scale_m: float
) -> dict[str, Any]:
    frames = trace["frames"]
    worlds = [frame["world"] for frame in frames if frame["world"].get("status") == "available"]
    if not worlds:
        ax.text(
            0.5, 0.5, "WORLD VIEW\nUNAVAILABLE", ha="center", va="center", transform=ax.transAxes
        )
        return {
            "status": "unavailable",
            "reason": "source_trace_world_frame_unavailable",
            "semantic_event_count": 0,
            "semantic_event_anchor_count": 0,
        }
    robot_xy = [world["robot"]["position"] for world in worlds]
    focal_xy = [world["focal_actor"]["position"] for world in worlds]
    ax.plot(
        [point[0] for point in robot_xy],
        [point[1] for point in robot_xy],
        color=PALETTE[role],
        linestyle=LINESTYLES[role],
        marker=MARKERS[role],
        markevery=[0, len(robot_xy) - 1],
        linewidth=1.6,
        markersize=3.5,
        label=f"robot · {role}",
    )
    ax.plot(
        [point[0] for point in focal_xy],
        [point[1] for point in focal_xy],
        color=PALETTE["focal"],
        linestyle=":",
        linewidth=1.25,
        label="focal actor",
    )
    first_context = frames[0].get("source_coordinates", {}).get("contextual_actors", [])
    for actor in first_context:
        if actor.get("actor_id") == frames[0]["source_coordinates"].get("focal_actor_id"):
            continue
        position = actor.get("position")
        if isinstance(position, list) and len(position) == 2:
            ax.scatter(
                [position[0]],
                [position[1]],
                color=PALETTE["context"],
                marker="x",
                s=14,
                alpha=0.45,
                label="context actor",
            )
    conflict = trace["coordinate_frames"]["conflict"]
    geometry = conflict.get("geometry", {})
    if conflict.get("status") == "available" and geometry.get("type") == "circle":
        ax.add_patch(
            Circle(
                geometry["center"],
                geometry["radius_m"],
                edgecolor=PALETTE["threshold"],
                facecolor="none",
                linestyle=(0, (3, 2)),
                linewidth=1.0,
                label="registered conflict zone",
            )
        )
    events = _available_events(trace)
    rendered_anchor_count = _draw_world_event_markers(ax, role, frames, events)
    xmin, xmax, ymin, ymax = crop
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), xlabel="world x (m)", ylabel="world y (m)")
    ax.set_aspect("equal", adjustable="box")
    bar_y = ymin + 0.06 * (ymax - ymin)
    bar_x = xmin + 0.08 * (xmax - xmin)
    ax.plot([bar_x, bar_x + scale_m], [bar_y, bar_y], color="#222222", linewidth=2.0)
    ax.set_title(trace["source_trace"]["source"]["planner_id"])
    return {
        "status": "available",
        "reason": "source_trace_world_frame",
        "semantic_event_count": len(events),
        "semantic_event_anchor_count": len(events),
        "unique_anchor_frame_count": rendered_anchor_count,
        "anchor_treatment": "proxy_footprints_velocity_arrows_and_redundant_markers",
    }


def _draw_time_space(
    ax: Any, traces: dict[str, dict[str, Any]], time_range: list[float]
) -> dict[str, Any]:
    route_available = all(
        trace["coordinate_frames"]["route"].get("status") == "available"
        for trace in traces.values()
    )
    conflict_available = all(
        trace["coordinate_frames"]["conflict"].get("status") == "available"
        for trace in traces.values()
    )
    if not route_available and not conflict_available:
        ax.text(
            0.5,
            0.5,
            "ROUTE / CONFLICT TIME–SPACE\nUNAVAILABLE",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set(xlim=tuple(time_range), xlabel="absolute time (s)")
        return {
            "status": "unavailable",
            "reason": "route_and_conflict_projection_unavailable",
            "occupancy_ribbon": {
                "status": "unavailable",
                "reason": "route_and_conflict_projection_unavailable",
            },
        }
    frame_key = "route" if route_available else "conflict"
    robot_key = "s_m" if route_available else "robot_signed_distance_to_zone_m"
    actor_key = "focal_actor_s_m" if route_available else "focal_actor_signed_distance_to_zone_m"
    ribbons_available = True
    for role, trace in traces.items():
        frames = [
            frame for frame in trace["frames"] if frame[frame_key].get("status") == "available"
        ]
        times = [frame["time_s"] for frame in frames]
        robot_s = [frame[frame_key][robot_key] for frame in frames]
        actor_s = [frame[frame_key][actor_key] for frame in frames]
        robot_radii = [frame.get("world", {}).get("robot", {}).get("radius_m") for frame in frames]
        actor_radii = [
            frame.get("world", {}).get("focal_actor", {}).get("radius_m") for frame in frames
        ]
        radii_available = all(
            isinstance(radius, int | float)
            and math.isfinite(float(radius))
            and float(radius) >= 0.0
            for radius in (*robot_radii, *actor_radii)
        )
        if radii_available:
            ax.fill_between(
                times,
                [
                    float(value) - float(radius)
                    for value, radius in zip(robot_s, robot_radii, strict=True)
                ],
                [
                    float(value) + float(radius)
                    for value, radius in zip(robot_s, robot_radii, strict=True)
                ],
                color=PALETTE[role],
                alpha=0.10,
                linewidth=0.0,
                zorder=0,
            )
            ax.fill_between(
                times,
                [
                    float(value) - float(radius)
                    for value, radius in zip(actor_s, actor_radii, strict=True)
                ],
                [
                    float(value) + float(radius)
                    for value, radius in zip(actor_s, actor_radii, strict=True)
                ],
                color=PALETTE[role],
                alpha=0.05,
                linewidth=0.0,
                zorder=0,
            )
        else:
            ribbons_available = False
        ax.plot(
            times,
            robot_s,
            color=PALETTE[role],
            linestyle=LINESTYLES[role],
            marker=MARKERS[role],
            markevery=[0, len(times) - 1],
            markersize=2.8,
            linewidth=1.6,
            label=f"{role} robot",
        )
        ax.plot(
            times,
            actor_s,
            color=PALETTE[role],
            linestyle=":",
            marker=MARKERS[role],
            markevery=[0, len(times) - 1],
            markersize=2.4,
            linewidth=1.0,
            alpha=0.7,
            label=f"{role} focal actor",
        )
    _draw_event_cursors(ax, traces)
    ylabel = "route progress s (m)" if route_available else "signed distance to conflict zone (m)"
    title = "Route occupancy" if route_available else "Conflict-zone approach"
    ax.set(xlim=tuple(time_range), xlabel="absolute time (s)", ylabel=ylabel)
    ax.set_title(f"{title} · no duration normalization\n{_event_identity_caption(traces)}")
    return {
        "status": "available",
        "reason": "route_projection" if route_available else "conflict_projection",
        "semantic_event_cursors": _event_groups(traces),
        "occupancy_ribbon": {
            "status": "available" if ribbons_available else "unavailable",
            "reason": (
                "recorded_proxy_radius_envelope"
                if ribbons_available
                else "recorded_proxy_radius_unavailable"
            ),
        },
    }


def _draw_clearance(
    ax: Any,
    traces: dict[str, dict[str, Any]],
    time_range: list[float],
    clearance_range: list[float],
) -> dict[str, str]:
    available = False
    threshold_m = float(
        traces["left"]["profiles"]["threshold_profile"]["proxy_surface_clearance_threshold_m"]
    )
    for role, trace in traces.items():
        frames = trace["frames"]
        series = [
            (frame["time_s"], frame["relative_interaction"].get("proxy_surface_clearance_m"))
            for frame in frames
            if frame["relative_interaction"].get("status") == "available"
        ]
        points = [(time, value) for time, value in series if isinstance(value, int | float)]
        if points:
            available = True
            ax.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                color=PALETTE[role],
                linestyle=LINESTYLES[role],
                marker=MARKERS[role],
                markersize=2.8,
                linewidth=1.6,
                label=f"{role} surface clearance",
            )
        centers = [
            (frame["time_s"], frame["relative_interaction"].get("center_distance_m"))
            for frame in frames
            if frame["relative_interaction"].get("status") == "available"
        ]
        center_points = [(time, value) for time, value in centers if isinstance(value, int | float)]
        if center_points:
            ax.plot(
                [point[0] for point in center_points],
                [point[1] for point in center_points],
                color=PALETTE[role],
                linestyle=":",
                linewidth=0.9,
                alpha=0.65,
                label=f"{role} centre distance (secondary)",
            )
    if not available:
        ax.text(
            0.5,
            0.5,
            "SURFACE CLEARANCE\nUNAVAILABLE",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    if available:
        ax.axhline(
            threshold_m,
            color=PALETTE["threshold"],
            linestyle=(0, (3, 2)),
            linewidth=1.0,
            label="diagnostic threshold",
        )
        _draw_event_cursors(ax, traces)
    ax.set(
        xlim=tuple(time_range),
        ylim=tuple(clearance_range),
        xlabel="absolute time (s)",
        ylabel="distance (m)",
    )
    ax.set_title("Surface clearance · primary")
    return {
        "status": "available" if available else "unavailable",
        "reason": "proxy_surface_clearance" if available else "proxy_surface_clearance_unavailable",
    }


def _closest_approach_record(trace: dict[str, Any]) -> dict[str, Any]:
    anchor_time = trace.get("event_anchor_hierarchy", {}).get("anchor_time_s")
    if not isinstance(anchor_time, int | float) or not math.isfinite(float(anchor_time)):
        return {"status": "unavailable", "reason": "semantic_anchor_unavailable"}
    frame = min(
        trace["frames"],
        key=lambda item: abs(float(item["time_s"]) - float(anchor_time)),
    )
    diagnostic = frame.get("relative_interaction", {}).get("closest_approach")
    if not isinstance(diagnostic, dict) or diagnostic.get("status") != "available":
        return {
            "status": "unavailable",
            "reason": "closest_approach_diagnostic_unavailable",
        }
    time_to = diagnostic.get("time_to_closest_approach_s")
    clearance = diagnostic.get("proxy_surface_clearance_at_closest_approach_m")
    if not (
        isinstance(time_to, int | float)
        and math.isfinite(float(time_to))
        and isinstance(clearance, int | float)
        and math.isfinite(float(clearance))
    ):
        return {
            "status": "unavailable",
            "reason": "closest_approach_values_unavailable",
        }
    return {
        "status": "available",
        "reason": "valid_local_closest_approach_diagnostic",
        "source_time_s": float(frame["time_s"]),
        "time_to_closest_approach_s": float(time_to),
        "proxy_surface_clearance_at_closest_approach_m": float(clearance),
        "model": diagnostic.get("model", "unavailable"),
        "profile_version": diagnostic.get("profile_version", "unavailable"),
    }


def _draw_closing_speed(
    ax: Any,
    traces: dict[str, dict[str, Any]],
    time_range: list[float],
    speed_range: list[float],
) -> dict[str, Any]:
    available = False
    closest_records: dict[str, dict[str, Any]] = {}
    for role, trace in traces.items():
        points: list[tuple[float, float]] = []
        for frame in trace["frames"]:
            relative = frame.get("relative_interaction", {})
            value = relative.get("radial_closing_speed_mps")
            if isinstance(value, int | float) and math.isfinite(float(value)):
                points.append((float(frame["time_s"]), float(value)))
        closest_record = _closest_approach_record(trace)
        closest_records[role] = closest_record
        if points:
            available = True
            ax.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                color=PALETTE[role],
                linestyle=LINESTYLES[role],
                linewidth=1.5,
                label=f"{role} radial closing speed",
            )
            if closest_record["status"] == "available":
                source_time = float(closest_record["source_time_s"])
                source_point = min(points, key=lambda item: abs(item[0] - source_time))
                ax.scatter(
                    [source_point[0]],
                    [source_point[1]],
                    marker="D",
                    s=30,
                    facecolors="white",
                    edgecolors=PALETTE[role],
                    linewidths=1.0,
                    zorder=5,
                )
    if not available:
        ax.text(
            0.5,
            0.5,
            "RADIAL CLOSING SPEED\nUNAVAILABLE",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    if available:
        _draw_event_cursors(ax, traces)
    ax.set(
        xlim=tuple(time_range),
        ylim=tuple(speed_range),
        xlabel="absolute time (s)",
        ylabel="closing speed (m/s)",
    )
    cpa_available = any(record["status"] == "available" for record in closest_records.values())
    ax.set_title(
        "Radial closing speed\n"
        + ("◇ local CPA diagnostic" if cpa_available else "local CPA diagnostic unavailable")
    )
    return {
        "status": "available" if available else "unavailable",
        "reason": "relative_velocity" if available else "relative_velocity_unavailable",
        "closest_approach": closest_records,
    }


def _draw_speed(
    ax: Any,
    traces: dict[str, dict[str, Any]],
    time_range: list[float],
    speed_range: list[float],
) -> dict[str, str]:
    available = False
    for role, trace in traces.items():
        points: list[tuple[float, float]] = []
        executed: list[tuple[float, float]] = []
        for frame in trace["frames"]:
            command = frame.get("commands", {}).get("commanded")
            value = command.get("linear_velocity") if isinstance(command, dict) else None
            if isinstance(value, int | float):
                points.append((float(frame["time_s"]), float(value)))
            velocity = frame.get("world", {}).get("robot", {}).get("velocity")
            if (
                isinstance(velocity, list)
                and len(velocity) == 2
                and all(
                    isinstance(component, int | float) and math.isfinite(float(component))
                    for component in velocity
                )
            ):
                executed.append(
                    (
                        float(frame["time_s"]),
                        math.hypot(float(velocity[0]), float(velocity[1])),
                    )
                )
        if points:
            available = True
            ax.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                color=PALETTE[role],
                linestyle=LINESTYLES[role],
                linewidth=1.5,
                label=f"{role} commanded speed",
            )
        if executed:
            available = True
            ax.plot(
                [point[0] for point in executed],
                [point[1] for point in executed],
                color=PALETTE[role],
                linestyle=":",
                linewidth=1.0,
                alpha=0.7,
                label=f"{role} executed speed",
            )
    if not available:
        ax.text(
            0.5,
            0.5,
            "COMMAND / CONTROLLER STATE\nUNAVAILABLE",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    if available:
        _draw_event_cursors(ax, traces)
    ax.set(
        xlim=tuple(time_range),
        ylim=tuple(speed_range),
        xlabel="absolute time (s)",
        ylabel="speed (m/s)",
    )
    ax.set_title("Commanded and executed speed")
    return {
        "status": "available" if available else "unavailable",
        "reason": "commanded_speed" if available else "commanded_speed_unavailable",
    }


def _draw_progress(
    ax: Any,
    traces: dict[str, dict[str, Any]],
    time_range: list[float],
) -> dict[str, str]:
    available = False
    for role, trace in traces.items():
        points = [
            (frame["time_s"], frame["route"].get("progress_rate_mps"))
            for frame in trace["frames"]
            if frame["route"].get("status") == "available"
        ]
        finite = [
            (float(time), float(value))
            for time, value in points
            if isinstance(value, int | float) and math.isfinite(float(value))
        ]
        if finite:
            available = True
            ax.plot(
                [point[0] for point in finite],
                [point[1] for point in finite],
                color=PALETTE[role],
                linestyle=LINESTYLES[role],
                linewidth=1.5,
                label=f"{role} progress rate",
            )
    if not available:
        ax.text(
            0.5,
            0.5,
            "PROGRESS / STALL\nUNAVAILABLE",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    if available:
        _draw_event_cursors(ax, traces)
    ax.set(xlim=tuple(time_range), xlabel="absolute time (s)", ylabel="progress rate (m/s)")
    ax.set_title("Route progress rate · sustained stall")
    return {
        "status": "available" if available else "unavailable",
        "reason": "route_progress_rate" if available else "route_progress_unavailable",
    }


def _controller_state_status(traces: dict[str, dict[str, Any]]) -> dict[str, str]:
    signal_names = ("controller_state", "command_source", "guard_state", "fallback_state")
    for trace in traces.values():
        contract = trace["source_trace"]["content_receipt"]["content_contract"]
        for frame in contract["frames"]:
            planner = frame.get("planner", {})
            if any(planner.get(name) is not None for name in signal_names):
                return {"status": "available", "reason": "controller_state_signal"}
    return {"status": "unavailable", "reason": "controller_state_signal_absent"}


def _recorded_start_separation_text(pair: dict[str, Any]) -> str:
    value = pair.get("route_spawn_separation", {}).get("initial_robot_separation_m")
    if isinstance(value, int | float) and math.isfinite(float(value)):
        return f"{float(value):.3f} m"
    return "unavailable"


def _draw_cell_context(
    ax: Any,
    cells: list[dict[str, Any]],
    ensemble_context: dict[str, Any],
    selected_case: dict[str, Any],
) -> None:
    ax.set_title("Release-cell outcomes · release statistics")
    lines: list[str] = []
    for cell in cells:
        counts = ", ".join(
            f"{name} {count}" for name, count in sorted(cell["outcome_counts"].items())
        )
        lines.append(f"{cell['planner']}: {counts} (n={cell['n_total']})")
    lines.extend(
        (
            (
                f"Selected: {selected_case['primary_role']} · "
                f"{selected_case['claim']['grade']} evidence"
            ),
            "Uncertainty: atlas interval bounds consumed; not recomputed",
            (
                f"Ensemble: {ensemble_context['status'].upper()} — "
                f"{ensemble_context['reason'].replace('_', ' ')}"
            ),
            (
                "Trace inventory: "
                f"missing={','.join(ensemble_context['missing_trace_ids']) or 'none'} · "
                f"ineligible={','.join(ensemble_context['ineligible_trace_ids']) or 'none'} · "
                f"excluded={','.join(ensemble_context['excluded_trace_ids']) or 'none'}"
            ),
        )
    )
    ax.text(
        0.02,
        0.90,
        "\n".join(lines),
        transform=ax.transAxes,
        va="top",
        fontsize=MINIMUM_VISIBLE_FONT_PT,
        linespacing=1.0,
    )
    ax.set_xticks([])
    ax.set_yticks([])


def _assert_text_within_canvas(figure: Any) -> None:
    """Fail closed when any visible text extends beyond the physical canvas."""

    renderer = figure.canvas.get_renderer()
    canvas = figure.bbox
    violations: list[str] = []
    for artist in figure.findobj(match=Text):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        bounds = artist.get_window_extent(renderer=renderer)
        if (
            bounds.x0 < canvas.x0 - 0.5
            or bounds.y0 < canvas.y0 - 0.5
            or bounds.x1 > canvas.x1 + 0.5
            or bounds.y1 > canvas.y1 + 0.5
        ):
            violations.append(artist.get_text().replace("\n", " / "))
    if violations:
        raise CaseDossierError(
            "figure_text_outside_canvas",
            "; ".join(sorted(violations)),
        )


def _assert_cross_axes_text_separation(figure: Any) -> None:
    """Fail closed when structural text from adjacent axes overlaps."""

    renderer = figure.canvas.get_renderer()
    structural_text: list[tuple[int, Text, Any]] = []
    for axes_index, axes in enumerate(figure.axes):
        artists = (
            *axes.texts,
            axes.title,
            axes._left_title,
            axes._right_title,
            axes.xaxis.label,
            axes.yaxis.label,
        )
        for artist in artists:
            if artist.get_visible() and artist.get_text().strip():
                structural_text.append(
                    (axes_index, artist, artist.get_window_extent(renderer=renderer))
                )

    violations: list[str] = []
    for index, (left_axes, left_artist, left_bounds) in enumerate(structural_text):
        for right_axes, right_artist, right_bounds in structural_text[index + 1 :]:
            if left_axes == right_axes:
                continue
            overlap_width = min(left_bounds.x1, right_bounds.x1) - max(
                left_bounds.x0, right_bounds.x0
            )
            overlap_height = min(left_bounds.y1, right_bounds.y1) - max(
                left_bounds.y0, right_bounds.y0
            )
            if overlap_width > 0.5 and overlap_height > 0.5:
                left_text = left_artist.get_text().replace("\n", " / ")
                right_text = right_artist.get_text().replace("\n", " / ")
                violations.append(
                    f"axes[{left_axes}] {left_text!r} <> axes[{right_axes}] {right_text!r}"
                )
    if violations:
        raise CaseDossierError(
            "cross_axes_text_overlap",
            "; ".join(sorted(violations)),
        )


def _make_figure(bound: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    payload = bound["input"]
    layout = payload["layout"]
    selected = bound["selected_case"]
    sources = [bound["traces"][role]["source_trace"]["source"] for role in ("left", "right")]
    run_contract = bound["traces"]["left"]["source_trace"]["run_config_contract"]
    source_contract = bound["traces"]["left"]["source_trace"]["content_receipt"]["content_contract"]
    horizon = (
        source_contract["frames"][0]
        .get("planner", {})
        .get("run_config", {})
        .get("horizon", "unavailable")
    )
    with plt.rc_context(_RC):
        figure = plt.figure(
            figsize=(layout["final_width_in"], layout["final_height_in"]),
        )
        grid = figure.add_gridspec(
            6,
            2,
            height_ratios=(1.30, 1.85, 1.10, 1.10, 1.10, 1.30),
        )
        header = figure.add_subplot(grid[0, :])
        world_left = figure.add_subplot(grid[1, 0])
        world_right = figure.add_subplot(grid[1, 1])
        route = figure.add_subplot(grid[2, :])
        clearance = figure.add_subplot(grid[3, 0])
        closing = figure.add_subplot(grid[3, 1])
        speed = figure.add_subplot(grid[4, 0])
        progress = figure.add_subplot(grid[4, 1])
        context = figure.add_subplot(grid[5, :])
        header.axis("off")
        title = (
            f"{payload['case_id']}\n"
            f"{selected['grain']} · {selected['primary_role']}\n"
            f"{sources[0]['scenario_id']}\n"
            f"{sources[0]['planner_id']} / {sources[1]['planner_id']} · "
            f"seed {sources[0]['seed']} / {sources[1]['seed']} · "
            f"horizon {horizon} steps · Δt {run_contract.get('time_step_s', 'unavailable')} s"
        )
        header.text(
            0.0,
            0.98,
            title,
            ha="left",
            va="top",
            weight="bold",
            fontsize=MINIMUM_VISIBLE_FONT_PT,
        )
        figure.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    color=PALETTE[role],
                    linestyle=LINESTYLES[role],
                    marker=MARKERS[role],
                    label=bound["trace_refs"][role]["label"],
                )
                for role in ("left", "right")
            ],
            loc="center",
            bbox_to_anchor=(0.5, 0.84),
            ncol=2,
            frameon=False,
        )
        header.text(
            0.0,
            0.02,
            (
                f"{SYNTHETIC_FIXTURE_LABEL} · shared_prefix=false\n"
                "recorded start separation = "
                f"{_recorded_start_separation_text(bound['pair'])}"
                if payload["comparison_grammar"] == "same_cell_seed_sensitivity"
                else SYNTHETIC_FIXTURE_LABEL
                if payload["mode"] == "synthetic_fixture"
                else "RENDERING DOES NOT ADMIT SCIENTIFIC EVIDENCE"
            ),
            ha="left",
            va="bottom",
            color=PALETTE["collision"],
            weight="bold",
            fontsize=MINIMUM_VISIBLE_FONT_PT,
        )
        world_left_status = _draw_world(
            world_left,
            "left",
            bound["traces"]["left"],
            layout["world_crop_m"],
            layout["metre_scale_m"],
        )
        world_right_status = _draw_world(
            world_right,
            "right",
            bound["traces"]["right"],
            layout["world_crop_m"],
            layout["metre_scale_m"],
        )
        ensemble = payload["ensemble_context"]
        panel_status = {
            "world_left": world_left_status,
            "world_right": world_right_status,
            "time_space": _draw_time_space(route, bound["traces"], layout["time_range_s"]),
            "surface_clearance": _draw_clearance(
                clearance, bound["traces"], layout["time_range_s"], layout["clearance_range_m"]
            ),
            "radial_closing_speed": _draw_closing_speed(
                closing,
                bound["traces"],
                layout["time_range_s"],
                layout["speed_range_mps"],
            ),
            "commanded_speed": _draw_speed(
                speed, bound["traces"], layout["time_range_s"], layout["speed_range_mps"]
            ),
            "progress_stall": _draw_progress(progress, bound["traces"], layout["time_range_s"]),
            "controller_state": _controller_state_status(bound["traces"]),
            "cell_context": {"status": "available", "reason": "campaign_atlas_cell"},
            "ensemble_context": {
                "status": ensemble["status"],
                "reason": ensemble["reason"],
            },
        }
        _draw_cell_context(context, bound["cells"], ensemble, selected)
        for ax in (world_left, world_right, route, clearance, closing, speed, progress):
            ax.grid(True, color="#DDDDDD", linewidth=0.45)
        figure.subplots_adjust(
            left=0.10,
            right=0.97,
            bottom=0.045,
            top=0.985,
            hspace=0.85,
            wspace=0.46,
        )
        route_position = route.get_position()
        route_title_band = ROUTE_TITLE_BAND_IN / float(layout["final_height_in"])
        if route_position.height <= route_title_band:
            raise CaseDossierError(
                "route_title_band_unavailable",
                f"required={ROUTE_TITLE_BAND_IN:.3f}in",
            )
        route.set_position(
            (
                route_position.x0,
                route_position.y0,
                route_position.width,
                route_position.height - route_title_band,
            )
        )
        figure.canvas.draw()
        _assert_text_within_canvas(figure)
        _assert_cross_axes_text_separation(figure)
        assert_clean(figure)
    return figure, panel_status


def _save_figure(figure: Any, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp{path.suffix}")
    with plt.rc_context(_RC):
        if path.suffix == ".svg":
            figure.savefig(
                temporary,
                format="svg",
                metadata={"Date": None, "Creator": CASE_DOSSIER_RENDERER_VERSION},
            )
        elif path.suffix == ".pdf":
            figure.savefig(
                temporary,
                format="pdf",
                metadata={
                    "CreationDate": None,
                    "ModDate": None,
                    "Creator": CASE_DOSSIER_RENDERER_VERSION,
                },
            )
        else:
            raise CaseDossierError("output_format_invalid", path.suffix)
    os.replace(temporary, path)


def _caption(bound: dict[str, Any]) -> str:
    narrative = bound["input"]["narrative"]
    return (
        f"# {bound['input']['case_id']} case dossier\n\n"
        f"**Claim boundary:** Renderer-integrity proof only; scientific_admission=false.\n\n"
        f"- **observed_signature:** {narrative['observed_signature']}\n"
        f"- **competing_explanation:** {narrative['competing_explanation']}\n"
        f"- **causal_status:** {narrative['causal_status']}\n"
        f"- **generalization_limit:** {narrative['generalization_limit']}\n"
    )


def _file_ref_for_output(path: Path) -> dict[str, str]:
    return {"path": path.name, "sha256": _file_sha256(path)}


def _trace_run_metadata(trace: dict[str, Any]) -> dict[str, Any]:
    source = trace["source_trace"]["source"]
    content = trace["source_trace"]["content_receipt"]["content_contract"]
    run_config = content["frames"][0].get("planner", {}).get("run_config", {})
    return {
        "scenario_id": source["scenario_id"],
        "planner_id": source["planner_id"],
        "seed": source["seed"],
        "episode_id": source["episode_id"],
        "horizon_steps": run_config.get("horizon"),
        "time_step_s": trace["source_trace"]["run_config_contract"].get("time_step_s"),
    }


def _finalize_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    result = {**payload, "content_sha256": ""}
    result["content_sha256"] = _canonical_sha256(result)
    return result


def validate_case_dossier_manifest(payload: Any) -> list[str]:
    """Return deterministic schema/content-hash violations for a dossier manifest."""

    errors = _schema_errors(payload, "case_dossier_manifest.v1.json")
    if isinstance(payload, dict):
        expected = _canonical_sha256({**payload, "content_sha256": ""})
        if payload.get("content_sha256") != expected:
            errors.append("/content_sha256: content hash mismatch")
    return errors


def _artifact_catalog(
    bound: dict[str, Any],
    manifest_path: Path,
    outputs: dict[str, Path],
    *,
    generation_commit: str,
) -> dict[str, Any]:
    source_files = [
        _file_ref_for_output(manifest_path),
        _file_ref_for_output(outputs["sidecar"]),
    ]
    denominator = sum(cell["n_total"] for cell in bound["cells"])
    return {
        "schema_version": "artifact_catalog.v2",
        "catalog_id": f"{bound['input']['dossier_id']}.catalog",
        "artifacts": [
            {
                "artifact_id": "case_dossier",
                "artifact_kind": "figure",
                "source_kind": "case_dossier_input",
                "source_files": source_files,
                "outputs": {
                    "svg": _file_ref_for_output(outputs["svg"]),
                    "pdf": _file_ref_for_output(outputs["pdf"]),
                },
                "caption_file": _file_ref_for_output(outputs["caption"]),
                "generation_command": "scripts/analysis/render_case_dossier.py --input <input> --out-dir <output>",
                "generation_commit": generation_commit,
                "claim_boundary": "Rendering integrity only; scientific_admission=false.",
                "figure_semantics": {
                    "metric_id": "proxy_envelope_surface_clearance_diagnostic",
                    "unit": "m",
                    "desirability": "not_applicable",
                    "support": denominator,
                    "denominator": denominator,
                    "comparison": True,
                    "uncertainty_declared": True,
                    "uncertainty_method": "campaign_atlas_interval_bounds_consumed",
                    "tie_policy": "not_applicable_visualization",
                    "legend_series": [
                        item["label"] for item in bound["input"]["sources"]["process_traces"]
                    ],
                    "legend_complete": True,
                    "accessibility_palette_contract": "case_dossier_colorblind.v1_redundant_styles",
                },
            }
        ],
    }


def render_case_dossier(input_path: Path, out_dir: Path) -> CaseDossierBundle:
    """Render one deterministic dossier bundle from a versioned input manifest.

    Returns:
        Paths and validated metadata for the generated bundle.
    """

    input_path = Path(input_path).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    bound = _load_bound_input(input_path)
    stem = bound["input"]["dossier_id"]
    svg_path = out_dir / f"{stem}.svg"
    pdf_path = out_dir / f"{stem}.pdf"
    caption_path = out_dir / f"{stem}.caption.md"
    sidecar_path = out_dir / f"{stem}.sidecar.json"
    manifest_path = out_dir / f"{stem}.manifest.json"
    catalog_path = out_dir / f"{stem}.catalog.json"
    generation_commit = _git_commit()

    figure, panel_status = _make_figure(bound)
    try:
        _save_figure(figure, svg_path)
        _save_figure(figure, pdf_path)
    finally:
        plt.close(figure)
    _atomic_write(caption_path, _caption(bound).encode("utf-8"))
    pair = bound["pair"]
    sidecar = {
        "schema_version": "case_dossier_sidecar.v1",
        "dossier_id": stem,
        "case_id": bound["input"]["case_id"],
        "scientific_admission": False,
        "comparison_grammar": bound["input"]["comparison_grammar"],
        "comparison": {
            "shared_prefix": pair["shared_prefix"],
            "recorded_start_separation": pair["route_spawn_separation"],
            "divergence_interpretation": pair["divergence_interpretation"],
            "prohibited_semantics": _prohibited_semantics(bound["input"]["comparison_grammar"]),
        },
        "source_classes": {
            "release_statistics": "campaign_atlas.v2",
            "visualization_diagnostics": "visualization_only_rerun_diagnostics",
        },
        "panel_status": panel_status,
        "claim_fields": bound["input"]["narrative"],
    }
    _write_json(sidecar_path, sidecar)
    semantic_events = sorted(
        {
            (str(event["event_type"]), float(event["time_s"]))
            for trace in bound["traces"].values()
            for event in _available_events(trace)
        }
    )
    manifest = _finalize_manifest(
        {
            "schema_version": CASE_DOSSIER_MANIFEST_SCHEMA_VERSION,
            "dossier_id": stem,
            "case_id": bound["input"]["case_id"],
            "mode": bound["input"]["mode"],
            "comparison_grammar": bound["input"]["comparison_grammar"],
            "scientific_admission": False,
            "comparison": {
                "shared_prefix": pair["shared_prefix"],
                "recorded_start_separation": pair["route_spawn_separation"],
                "divergence_interpretation": pair["divergence_interpretation"],
                "prohibited_semantics": _prohibited_semantics(bound["input"]["comparison_grammar"]),
            },
            "selection": {
                "case_id": bound["selected_case"]["case_id"],
                "selected": True,
                "selection_manifest_sha256": bound["portfolio"]["content_sha256"],
                "grain": bound["selected_case"]["grain"],
                "primary_role": bound["selected_case"]["primary_role"],
                "allowed_claim": bound["selected_case"]["allowed_claim"],
                "claim_grade": bound["selected_case"]["claim"]["grade"],
                "evidence_grade": bound["selected_case"]["dimensions"]["evidence_grade"],
                "eligibility": bound["selected_case"]["eligibility"],
                "source_boundary": bound["selected_case"]["source_boundary"],
            },
            "renderer": {
                "renderer_version": CASE_DOSSIER_RENDERER_VERSION,
                "style_version": CASE_DOSSIER_STYLE_VERSION,
                "input_schema_version": CASE_DOSSIER_INPUT_SCHEMA_VERSION,
                "manifest_schema_version": CASE_DOSSIER_MANIFEST_SCHEMA_VERSION,
                "matplotlib_version": matplotlib.__version__,
                "final_width_in": FINAL_WIDTH_IN,
                "base_font_pt": BASE_FONT_PT,
                "minimum_visible_font_pt": MINIMUM_VISIBLE_FONT_PT,
                "svg_hashsalt": _RC["svg.hashsalt"],
                "generation_commit": generation_commit,
                "header_line_contract": "wrapped_left_aligned.v1",
                "canvas_text_bounds_checked": True,
                "cross_axes_text_overlap_checked": True,
                "route_title_band_in": ROUTE_TITLE_BAND_IN,
            },
            "source_bindings": {
                "portfolio": {
                    "path": _portable_source_path(bound["input_path"], bound["portfolio_path"]),
                    "sha256": _file_sha256(bound["portfolio_path"]),
                    "schema_version": "ch7_case_portfolio.v2",
                },
                "process_traces": [
                    {
                        "role": role,
                        "path": _portable_source_path(
                            bound["input_path"], bound["trace_paths"][role]
                        ),
                        "sha256": _file_sha256(bound["trace_paths"][role]),
                        "schema_version": "worked_example_process_trace.v1",
                        "source_content_sha256": bound["traces"][role]["source_trace"][
                            "content_sha256"
                        ],
                        "run": _trace_run_metadata(bound["traces"][role]),
                    }
                    for role in ("left", "right")
                ],
                "campaign_atlas": {
                    "path": _portable_source_path(bound["input_path"], bound["atlas_path"]),
                    "sha256": _file_sha256(bound["atlas_path"]),
                    "schema_version": "campaign_atlas.v2",
                    "release_cells": bound["cells"],
                },
                "source_classes": {
                    "release_statistics": "campaign_atlas.v2",
                    "visualization_diagnostics": "visualization_only_rerun_diagnostics",
                },
            },
            "shared_scales": bound["input"]["layout"],
            "panel_status": panel_status,
            "semantic_events": [
                {"event_type": event_type, "absolute_time_s": time_s}
                for event_type, time_s in semantic_events
            ],
            "claim_fields": bound["input"]["narrative"],
            "outputs": {
                "svg": _file_ref_for_output(svg_path),
                "pdf": _file_ref_for_output(pdf_path),
                "caption": _file_ref_for_output(caption_path),
                "sidecar": _file_ref_for_output(sidecar_path),
            },
        }
    )
    errors = validate_case_dossier_manifest(manifest)
    if errors:
        raise CaseDossierError("case_dossier_manifest_invalid", "; ".join(errors))
    _write_json(manifest_path, manifest)
    catalog = _artifact_catalog(
        bound,
        manifest_path,
        {
            "svg": svg_path,
            "pdf": pdf_path,
            "caption": caption_path,
            "sidecar": sidecar_path,
        },
        generation_commit=generation_commit,
    )
    _write_json(catalog_path, catalog)
    return CaseDossierBundle(
        svg_path=svg_path,
        pdf_path=pdf_path,
        caption_path=caption_path,
        sidecar_path=sidecar_path,
        manifest_path=manifest_path,
        catalog_path=catalog_path,
        manifest=manifest,
    )


__all__ = [
    "BASE_FONT_PT",
    "CASE_DOSSIER_INPUT_SCHEMA_VERSION",
    "CASE_DOSSIER_MANIFEST_SCHEMA_VERSION",
    "CASE_DOSSIER_RENDERER_VERSION",
    "FINAL_WIDTH_IN",
    "MINIMUM_VISIBLE_FONT_PT",
    "SYNTHETIC_FIXTURE_LABEL",
    "CaseDossierBundle",
    "CaseDossierError",
    "render_case_dossier",
    "validate_case_dossier_manifest",
]
