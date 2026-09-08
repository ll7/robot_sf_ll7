"""Batch publication views of the case workbench's selected recorded scenarios.

Selection/admission, metric semantics, map parsing and publication styling remain
owned by their existing modules. This module owns composition and packaging only.
No simulator is run; a diagnostic pack is never an author-admitted figure.
See docs/scenario_figure_pack.md for the config-first workflow.
"""

# Optional rendering imports remain lazy at this artifact boundary.
# ruff: noqa: DOC201, PLC0415, C901, PLR0912, PLR0915

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import math
import os
import shutil
import tempfile
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from robot_sf.benchmark.figures.profile import FigureProfile

SCHEMA = "scenario-figure-pack.v1"
EVIDENCE_STATUS = "diagnostic-only"
BOUNDARY = (
    "Exact recorded episode only. Selection is not prevalence, planner superiority, "
    "a causal explanation, or deployment-safety evidence."
)
VIEWS = ("trajectory", "snapshot", "clearance", "speed", "turn")


def _json(value: Any) -> str:
    """Serialize receipts without non-standard NaN or infinity literals."""
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"


def _sha(path: Path) -> str:
    """Hash bytes without materializing a large artifact twice."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object(path: Path) -> dict[str, Any]:
    """Read a JSON object, rejecting duplicate keys and non-finite literals."""

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid(value: str) -> None:
        raise ValueError(f"non-finite JSON literal: {value}")

    value = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=invalid
    )
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path.name}")
    return value


@dataclass(frozen=True)
class PackConfig:
    """Bounded export settings; defaults require the existing author-admission gate."""

    mode: str = "admitted"
    formats: tuple[str, ...] = ("pdf", "svg", "png")
    views: tuple[str, ...] = VIEWS
    max_cases: int = 12
    max_frames: int = 50000
    max_actors: int = 100
    max_points: int = 250000
    size: str = "double"

    def __post_init__(self) -> None:
        """Refuse misspellings and unbounded or ambiguous settings before any write."""
        if not isinstance(self.mode, str) or self.mode not in {"admitted", "diagnostic"}:
            raise ValueError("mode must be admitted or diagnostic")
        if not isinstance(self.size, str) or self.size not in {"single", "double"}:
            raise ValueError("size must be single or double")
        for name, allowed in (("formats", {"pdf", "svg", "png"}), ("views", set(VIEWS))):
            values = getattr(self, name)
            if (
                not isinstance(values, tuple)
                or not values
                or any(not isinstance(v, str) or v not in allowed for v in values)
            ):
                raise ValueError(f"invalid {name}")
            if len(values) != len(set(values)):
                raise ValueError(f"duplicate {name}")
        for name, ceiling in (
            ("max_cases", 100),
            ("max_frames", 1000000),
            ("max_actors", 1000),
            ("max_points", 5000000),
        ):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= ceiling:
                raise ValueError(f"{name} must be an integer in [1, {ceiling}]")

    @classmethod
    def from_file(cls, path: Path) -> PackConfig:
        """Read explicit configuration without silently ignoring unknown keys."""
        values = _object(path)
        if values.pop("schema_version", None) != SCHEMA:
            raise ValueError(f"config schema_version must be {SCHEMA}")
        if set(values) - set(cls.__dataclass_fields__):
            raise ValueError("unknown figure-pack configuration keys")
        for key in ("formats", "views"):
            if key in values:
                if not isinstance(values[key], list):
                    raise ValueError(f"{key} must be a JSON array")
                values[key] = tuple(values[key])
        return cls(**values)


def _owner() -> Any:
    """Load the canonical case renderer lazily, not through a simulator entry point."""
    return importlib.import_module("robot_sf.benchmark.case_publication_figure")


def _inventory(root: Path) -> dict[str, str]:
    """Reject links and record all source bytes for source-change detection."""
    if root.is_symlink() or not root.is_dir():
        raise ValueError("source must be a real package directory")
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("symlinks are not accepted in a figure source package")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = _sha(path)
        elif not path.is_dir():
            raise ValueError("non-regular file in source package")
    return result


def _verify_source(root: Path, mode: str) -> None:
    """Reuse source integrity and admission; diagnostic mode never bypasses integrity."""
    owner = _owner()
    owner._verify_package_integrity(root)
    if mode == "admitted":
        owner._verify_publication_gate(root)


def _source_admission_status(proposal: dict[str, Any]) -> str:
    """Report admission of the input package independently from export mode."""
    admission = proposal.get("author_admission")
    return (
        "admitted"
        if isinstance(admission, dict) and str(admission.get("status") or "").lower() == "admitted"
        else "not_admitted"
    )


def _trace_provenance(case: dict[str, Any], trace: dict[str, Any], mode: str) -> dict[str, Any]:
    """Validate canonical trace provenance before exposing it in a pack."""
    from robot_sf.benchmark.analysis_trace import trace_artifact_sha256, trace_coverage

    coverage = trace_coverage(
        {
            "scenario_id": case.get("scenario_id"),
            "planner": case.get("planner"),
            "algo": case.get("planner"),
            "provenance": case.get("provenance", {}),
            "algorithm_metadata": {"analysis_trace": trace},
        }
    )
    artifact_sha = trace.get("artifact_sha256")
    artifact_verified = isinstance(artifact_sha, str) and artifact_sha == trace_artifact_sha256(
        trace
    )
    coverage_complete = coverage.get("status") == "complete"
    if mode == "admitted" and not (artifact_verified and coverage_complete):
        reasons = list(coverage.get("reasons") or [])
        if not artifact_verified:
            reasons.append("artifact_hash")
        detail = ", ".join(dict.fromkeys(str(reason) for reason in reasons))
        raise ValueError(f"admitted source trace provenance is incomplete: {detail}")
    verified = artifact_verified and coverage_complete
    return {
        "status": "verified" if verified else "structural-only",
        "coverage_status": coverage.get("status"),
        "coverage_reasons": list(coverage.get("reasons") or []),
        "artifact_sha256": artifact_sha if verified else None,
        "map_digest": trace.get("map_digest") if verified else None,
        "git_hash": trace.get("git_hash") if verified else None,
    }


def select_cases(
    proposal: dict[str, Any], config: PackConfig, case_ids: tuple[str, ...] = ()
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Keep workbench order, with explicit accounting for every portfolio omission.

    This is a presentation budget, not a new relevance score or an admission
    operation. Each case is rendered separately; no pair is inferred.
    """
    portfolio = proposal.get("portfolio")
    if not isinstance(portfolio, list) or not portfolio:
        raise ValueError("proposal must have a nonempty portfolio")
    lookup: dict[str, dict[str, Any]] = {}
    for case in portfolio:
        if (
            not isinstance(case, dict)
            or not isinstance(case.get("case_id"), str)
            or not case["case_id"]
        ):
            raise ValueError("every portfolio case must have a nonempty case_id")
        if case["case_id"] in lookup:
            raise ValueError("duplicate portfolio case_id")
        lookup[case["case_id"]] = case
    if len(set(case_ids)) != len(case_ids) or any(key not in lookup for key in case_ids):
        raise ValueError("requested case IDs are duplicated or absent from the portfolio")
    selected: list[dict[str, Any]] = []
    omitted: list[dict[str, str]] = []
    for key, case in lookup.items():
        reason = "not_requested" if case_ids and key not in case_ids else None
        if reason is None and len(selected) >= config.max_cases:
            reason = "presentation_budget"
        if reason:
            omitted.append({"case_id": key, "reason": reason})
        else:
            selected.append(case)
    return selected, omitted


def _number(value: Any) -> bool:
    """Exclude booleans and non-finite coordinates/control samples."""
    return type(value) in (int, float) and math.isfinite(value)


def _position(value: Any) -> bool:
    """Require a finite, exactly two-dimensional recorded position."""
    return isinstance(value, list) and len(value) == 2 and all(_number(v) for v in value)


def prepare_case(case: dict[str, Any], config: PackConfig) -> dict[str, Any]:
    """Validate time/identity alignment and preserve missing actor frames as gaps."""
    trace = case.get("trace")
    if not isinstance(trace, dict):
        raise ValueError(f"trace unavailable for {case['case_id']}; restore the recorded trace")
    steps = trace.get("steps")
    if not isinstance(steps, list) or not 1 <= len(steps) <= config.max_frames:
        raise ValueError("trace is empty or exceeds max_frames")
    if trace.get("coordinate_frame") != "world":
        raise ValueError("trace coordinate_frame must explicitly be world")
    units = trace.get("units")
    if not isinstance(units, dict) or units.get("position") != "m" or units.get("time") != "s":
        raise ValueError("trace must explicitly declare position=m and time=s")
    if trace.get("schema_version") != "analysis-trace.v1":
        raise ValueError("expected analysis-trace.v1")
    times: list[float] = []
    robot_xy: list[list[float]] = []
    actor_ids: set[str] = set()
    actor_ids_by_frame: list[set[str]] = []
    for step in steps:
        if not isinstance(step, dict) or not _number(step.get("time_s")):
            raise ValueError("every frame requires a finite time_s")
        time = float(step["time_s"])
        if time < 0 or (times and time <= times[-1]):
            raise ValueError("trace times must be nonnegative and strictly increasing")
        times.append(time)
        robot = step.get("robot")
        if not isinstance(robot, dict) or not _position(robot.get("position")):
            raise ValueError("every frame requires a finite robot position")
        robot_xy.append(robot["position"])
        actors = step.get("pedestrians")
        if not isinstance(actors, list):
            raise ValueError("every frame requires an explicit pedestrians list")
        frame_ids: set[str] = set()
        for actor in actors:
            if not isinstance(actor, dict):
                raise ValueError("pedestrian must be an object")
            key = actor.get("actor_id")
            if not isinstance(key, str) or not key or key == "robot" or key in frame_ids:
                raise ValueError("pedestrians require unique stable actor_id strings")
            if not _position(actor.get("position")):
                raise ValueError("pedestrian position must be finite")
            frame_ids.add(key)
            actor_ids.add(key)
        actor_ids_by_frame.append(frame_ids)
        if len(actor_ids) > config.max_actors:
            raise ValueError("trace exceeds max_actors; no silent actor truncation")
        controls = step.get("controls")
        if controls is None:
            controls = {}
        if not isinstance(controls, dict):
            raise ValueError("controls must be an object")
        applied = controls.get("applied")
        if applied is None:
            applied = {}
        if not isinstance(applied, dict):
            raise ValueError("applied controls must be an object")
        for key in ("linear_m_s", "turn_rate_rad_s"):
            if applied.get(key) is not None and not _number(applied[key]):
                raise ValueError("recorded applied controls must be finite or unavailable")
    tracks = {key: [] for key in sorted(actor_ids)}
    for step in steps:
        actors = {actor["actor_id"]: actor["position"] for actor in step["pedestrians"]}
        for key, positions in tracks.items():
            positions.append(actors.get(key, [float("nan"), float("nan")]))
    owner = _owner()
    series = {key: owner._series(trace, key) for key in ("clearance", "speed", "turn")}
    if any(len(values) != len(times) for values in series.values()):
        raise ValueError("canonical metric series is not frame-aligned")
    # A minimum over only the actors with known radii is not a complete
    # robot-pedestrian clearance observation. Keep that frame unavailable,
    # rather than promoting a partial canonical result to a complete minimum.
    for index, step in enumerate(steps):
        actors = [step["robot"], *step["pedestrians"]]
        if actor_ids_by_frame[index] != actor_ids or any(
            not _number(actor.get("radius_m")) or actor["radius_m"] <= 0 for actor in actors
        ):
            series["clearance"][index] = float("nan")
    critical = owner._critical_step(trace)
    critical_index = next((i for i, step in enumerate(steps) if step is critical), None)
    if critical_index is None:
        raise ValueError("canonical critical frame is absent from trace")
    event_times = owner._event_times(trace)
    if not event_times:
        finite = [i for i, value in enumerate(series["clearance"]) if math.isfinite(value)]
        critical_index = min(finite, key=lambda i: series["clearance"][i]) if finite else 0
    if any(not _number(t) or t < times[0] or t > times[-1] for t in event_times):
        raise ValueError("recorded event is outside the trace time range")
    trace_provenance = _trace_provenance(case, trace, config.mode)
    return {
        "case": case,
        "trace": trace,
        "times": times,
        "robot": robot_xy,
        "actors": tracks,
        "series": series,
        "critical_index": critical_index,
        "event_times": event_times,
        "trace_provenance": trace_provenance,
        "snapshot_reason": (
            f"nearest recorded frame to event at {event_times[0]:g} s"
            if event_times
            else (
                "minimum trace-derived surface clearance"
                if any(math.isfinite(v) for v in series["clearance"])
                else "first recorded frame; no event or clearance available"
            )
        ),
    }


def _case_title(case: dict[str, Any], *, width: int = 76) -> str:
    """Wrap identifiers instead of shrinking all figure typography."""
    return textwrap.fill(
        f"{case.get('scenario_id', 'unknown scenario')} | {case.get('planner', 'unknown planner')} "
        f"| seed {case.get('seed', 'unavailable')}",
        width=width,
    )


def _series_status(view: str, values: list[float]) -> dict[str, Any]:
    """Classify finite, partial and unavailable recorded series explicitly."""
    total_samples = len(values)
    finite_samples = sum(math.isfinite(value) for value in values)
    missing_samples = total_samples - finite_samples
    reasons = {
        "clearance": (
            "an expected pedestrian identity or positive recorded body radius is missing"
        ),
        "speed": "recorded applied linear_m_s is missing",
        "turn": "recorded applied turn_rate_rad_s is missing",
    }
    return {
        "total_samples": total_samples,
        "finite_samples": finite_samples,
        "missing_samples": missing_samples,
        "missing_reason": reasons[view] if missing_samples else None,
        "status": (
            "unavailable"
            if finite_samples == 0
            else "partly_unavailable"
            if missing_samples
            else "available"
        ),
    }


def render_view(
    prepared: dict[str, Any],
    view: str,
    config: PackConfig,
    *,
    profile: FigureProfile | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Build one view under its complete final-size rendering profile."""
    import matplotlib

    selected_profile = profile or FigureProfile.builtin(config.size)
    with matplotlib.rc_context(selected_profile.rc_params()):
        return _render_view(prepared, view, config, selected_profile)


def _render_view(
    prepared: dict[str, Any],
    view: str,
    config: PackConfig,
    profile: FigureProfile,
) -> tuple[Any, dict[str, Any]]:
    """Compose one profiled single-axis view while preserving telemetry gaps."""
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Circle

    if view not in VIEWS:
        raise ValueError("unknown view")
    width, height = profile.figure_size()
    fig = Figure(figsize=(width, height), layout="constrained")
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    case, trace = prepared["case"], prepared["trace"]
    status: dict[str, Any] = {
        "view": view,
        "status": "available",
        "figure_profile_id": profile.profile_id,
        "figure_profile_sha256": profile.sha256(),
        "figure_size_in": [width, height],
    }
    mode_label = (
        "DIAGNOSTIC ONLY - not author admitted"
        if config.mode == "diagnostic"
        else "Author-admitted recorded case"
    )
    title = (
        f"{view.replace('_', ' ').title()}\n"
        f"{_case_title(case, width=76 if width > 4 else 35)}\n{mode_label}"
    )
    ax.set_title(title, fontsize=profile.axes_title_size_pt, pad=12)
    if view in {"trajectory", "snapshot"}:
        owner = _owner()
        geometry = owner._resolve_map_geometry([case])
        # The trace-derived extent always includes actors; optional maps cannot clip the data.
        limits = owner._world_limits([trace], None)
        if limits is None:
            raise ValueError("world limits unavailable")
        # Keep recorded body footprints inside the same extent as the paths.
        low_x, high_x = limits[0]
        low_y, high_y = limits[1]
        for step in trace["steps"]:
            for actor in [step["robot"], *step["pedestrians"]]:
                radius = actor.get("radius_m")
                if _number(radius) and radius > 0:
                    x, y = actor["position"]
                    low_x, high_x = min(low_x, x - radius), max(high_x, x + radius)
                    low_y, high_y = min(low_y, y - radius), max(high_y, y + radius)
        limits = ((low_x, high_x), (low_y, high_y))
        owner._draw_map_geometry(ax, geometry, limits)
        status["map_geometry"] = "available" if geometry is not None else "unavailable"
        if view == "trajectory":
            xy = np.asarray(prepared["robot"])
            ax.plot(xy[:, 0], xy[:, 1], linewidth=2, label="robot path")
            ax.scatter(*xy[0], marker="o", s=38, label="robot start", zorder=5)
            ax.scatter(*xy[-1], marker="s", s=38, label="robot end", zorder=5)
            for index, (key, positions) in enumerate(prepared["actors"].items()):
                points = np.asarray(positions)
                ax.plot(
                    points[:, 0],
                    points[:, 1],
                    linestyle="--",
                    linewidth=1,
                    alpha=0.7,
                    label="pedestrian paths" if index == 0 else None,
                )
        index = prepared["critical_index"]
        step = trace["steps"][index]
        expected_actor_ids = set(prepared["actors"])
        observed_actor_ids = {actor["actor_id"] for actor in step["pedestrians"]}
        missing_actor_ids = sorted(expected_actor_ids - observed_actor_ids)
        status["footprints"] = "partly_unavailable" if missing_actor_ids else "available"
        if missing_actor_ids:
            status.update(
                {
                    "status": "partly_unavailable",
                    "missing_actor_ids": missing_actor_ids,
                    "missing_reason": (
                        "expected pedestrian identity is absent at the selected frame"
                    ),
                }
            )
        for actor_index, actor in enumerate([step["robot"], *step["pedestrians"]]):
            x, y = actor["position"]
            radius = actor.get("radius_m")
            if _number(radius) and radius > 0:
                ax.add_patch(
                    Circle(
                        (x, y),
                        radius,
                        fill=False,
                        linewidth=1.4,
                        linestyle="-" if actor_index == 0 else "--",
                        zorder=6,
                    )
                )
            else:
                status["footprints"] = "partly_unavailable"
                status["status"] = "partly_unavailable"
            ax.scatter(
                x,
                y,
                marker="*" if actor_index == 0 else "+",
                s=60,
                zorder=7,
                label=(
                    "robot at selected time"
                    if actor_index == 0
                    else "pedestrians at selected time"
                    if actor_index == 1
                    else None
                ),
            )
        ax.set(xlim=limits[0], ylim=limits[1], xlabel="World x [m]", ylabel="World y [m]")
        ax.set_aspect("equal", adjustable="box")
        notes = (
            f"Selected frame: t={step['time_s']:g} s ({prepared['snapshot_reason']}).\n"
            f"Map: {status['map_geometry']}. Footprints: {status['footprints'].replace('_', ' ')}. "
            "Circles use recorded radii; no perception is inferred."
        )
        if missing_actor_ids:
            notes += f"\nMissing expected pedestrian identities: {', '.join(missing_actor_ids)}."
        status.update(
            {
                "snapshot_time_s": step["time_s"],
                "snapshot_reason": prepared["snapshot_reason"],
                "world_limits": limits,
            }
        )
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(0, -0.25),
            ncol=2 if width > 4 else 1,
            fontsize=profile.legend_size_pt,
        )
    else:
        style = importlib.import_module("robot_sf.benchmark.figures.style")
        metric_keys = {
            "clearance": "surface_clearance",
            "speed": "applied_linear_speed",
            "turn": "applied_turn_rate",
        }
        values = prepared["series"][view]
        status.update(_series_status(view, values))
        ax.set(
            xlabel=style.metric_label(
                "recorded_time", language=profile.language, strict=True
            ),
            ylabel=style.metric_label(
                metric_keys[view], language=profile.language, strict=True
            ),
        )
        if any(math.isfinite(v) for v in values):
            ax.plot(
                prepared["times"], values, linewidth=1.6, marker="." if len(values) < 15 else None
            )
        else:
            ax.text(
                0.5,
                0.5,
                "Recorded telemetry unavailable\nNot zero; not estimated",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
        if view == "clearance":
            ax.axhline(0, linewidth=0.8, linestyle=":")
        for event_time in prepared["event_times"]:
            ax.axvline(event_time, linewidth=0.8, linestyle="--", alpha=0.6)
        low, high = prepared["times"][0], prepared["times"][-1]
        pad = max(0.01, (high - low) * 0.025)
        ax.set_xlim(low - pad, high + pad)
        ax.grid(True, alpha=0.2)
        notes = "Dashed vertical lines: recorded event times. Missing samples remain gaps."
        if status["missing_samples"]:
            notes += (
                f"\n{status['missing_samples']} of {status['total_samples']} samples unavailable: "
                f"{status['missing_reason']}."
            )
        if view == "clearance":
            notes += "\nDisc-surface separation is not a recomputed benchmark collision label."
    fig.supxlabel(
        textwrap.fill(notes, width=100 if width > 4 else 48),
        fontsize=profile.annotation_size_pt,
    )
    status["caption"] = f"{view.title()}: {_case_title(case)}. {notes} {BOUNDARY}"
    return fig, status


def build_pack(
    package: Path,
    output: Path,
    *,
    config: PackConfig = PackConfig(),
    case_ids: tuple[str, ...] = (),
    profile: FigureProfile | None = None,
) -> dict[str, Any]:
    """Export a transactional, bounded pack without changing source or existing output.

    A sibling lock coordinates cooperating writers; exclusive directory creation
    prevents overwriting any existing target. The manifest is published last.
    """
    package, output = Path(package).absolute(), Path(output).absolute()
    if package.is_symlink() or output.is_symlink():
        raise ValueError("source and output must not be symlinks")
    source_root, target = package.resolve(), output.resolve()
    if target == source_root or source_root in target.parents or target in source_root.parents:
        raise ValueError("source and output directories must not overlap")
    profile = profile or FigureProfile.builtin(config.size)
    config_json = _json({"schema_version": SCHEMA, **asdict(config)})
    profile_json = profile.canonical_json()
    semantics = importlib.import_module("robot_sf.benchmark.figures.semantics")
    semantics_registry = semantics.default_registry()
    semantics_json = semantics_registry.canonical_json()
    before = _inventory(package)
    _verify_source(package, config.mode)
    proposal = _object(package / "proposal.json")
    source_admission_status = _source_admission_status(proposal)
    selected, omitted = select_cases(proposal, config, case_ids)
    point_count = 0
    for case in selected:
        trace = case.get("trace")
        steps = trace.get("steps") if isinstance(trace, dict) else None
        if not isinstance(steps, list):
            raise ValueError("recorded trace steps unavailable")
        for step in steps:
            actors = step.get("pedestrians") if isinstance(step, dict) else None
            point_count += 1 + (len(actors) if isinstance(actors, list) else 0)
            if point_count > config.max_points:
                raise ValueError("selected traces exceed max_points; no silent downsampling")
    prepared = [prepare_case(case, config) for case in selected]
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.parent / f".{output.name}.scenario-pack.lock"
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise ValueError("another figure-pack writer owns this output lock") from exc
    stage: Path | None = None
    try:
        if output.exists():
            raise ValueError("output already exists; choose a new directory")
        stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
        style = importlib.import_module("robot_sf.benchmark.figures.style")
        exporter = importlib.import_module("robot_sf.benchmark.figures.export")
        provenance = importlib.import_module("robot_sf.benchmark.figures.provenance")
        profile_module = importlib.import_module("robot_sf.benchmark.figures.profile")
        import matplotlib

        producer_files = {
            module.__name__: _sha(Path(inspect.getfile(module)))
            for module in (_owner(), style, exporter, provenance, profile_module, semantics)
        }
        producer_files["figure_semantics.v1.json"] = _sha(semantics.DEFAULT_REGISTRY)
        width, height = profile.figure_size()
        receipt: dict[str, Any] = {
            "schema_version": SCHEMA,
            "mode": config.mode,
            "evidence_status": EVIDENCE_STATUS,
            "source_admission_status": source_admission_status,
            "claim_boundary": BOUNDARY,
            "source_proposal_sha256": before["proposal.json"],
            "source_inventory_sha256": hashlib.sha256(_json(before).encode()).hexdigest(),
            "config_sha256": hashlib.sha256(config_json.encode()).hexdigest(),
            "figure_profile": {
                "schema_version": profile.payload()["schema_version"],
                "profile_id": profile.profile_id,
                "sha256": profile.sha256(),
                "language": profile.language,
                "target_width_in": width,
                "target_height_in": height,
                "requested_font_family": list(profile.font_family),
                "resolved_font_family": None,
            },
            "figure_semantics": {
                "schema_version": semantics.SCHEMA,
                "sha256": semantics_registry.sha256(),
                "metric_count": len(semantics_registry.metrics),
                "planner_count": len(semantics_registry.planners),
            },
            "recorded_world_points": point_count,
            "producer_sha256": _sha(Path(__file__)),
            "producer_dependencies": producer_files,
            "matplotlib_version": matplotlib.__version__,
            "reproduce": {
                "module": "robot_sf.benchmark.figures.scenario_pack",
                "config": "config.json",
                "figure_profile": "figure_profile.json",
                "figure_semantics": "figure_semantics.json",
                "case_ids": list(case_ids),
                "source": "restore the exact source_inventory_sha256 package separately",
                "output": "choose a new, non-existing directory",
            },
            "selection": {
                "policy": "existing workbench portfolio order; no new ranking",
                "portfolio_count": len(proposal["portfolio"]),
                "selected_count": len(selected),
                "requested_case_ids": list(case_ids),
                "omitted": omitted,
            },
            "cases": [],
            "artifacts": [],
        }
        (stage / "config.json").write_text(config_json, encoding="utf-8")
        (stage / "figure_profile.json").write_text(profile_json, encoding="utf-8")
        (stage / "figure_semantics.json").write_text(semantics_json, encoding="utf-8")
        with (
            style.publication_style(size=config.size),
            matplotlib.rc_context(
                {
                    **profile.rc_params(),
                    "text.usetex": False,
                    "text.parse_math": False,
                    "svg.hashsalt": receipt["config_sha256"],
                }
            ),
        ):
            receipt["figure_profile"]["resolved_font_family"] = profile.resolve_font_family()
            for item in prepared:
                case = item["case"]
                # Identifiers never become paths or executable TeX.
                stem = "case-" + hashlib.sha256(case["case_id"].encode()).hexdigest()[:24]
                case_record = {
                    key: case.get(key)
                    for key in ("case_id", "scenario_id", "planner", "seed", "role")
                }
                case_record["source_trace"] = {
                    **item["trace_provenance"],
                    "config_digest": item["trace"].get("config_digest")
                    if item["trace_provenance"]["status"] == "verified"
                    else None,
                }
                case_record["views"] = []
                for view in config.views:
                    figure, view_info = render_view(item, view, config, profile=profile)
                    view_info["stem"] = f"{stem}/{view}"
                    prov = {
                        "source_artifacts": [
                            {"path": "proposal.json", "sha256": before["proposal.json"]}
                        ],
                        "episode_ids": [str(case.get("episode_id") or case["case_id"])],
                        "seeds": [case["seed"]] if type(case.get("seed")) is int else [],
                        "generator_command": "python -m robot_sf.benchmark.figures.scenario_pack",
                        "figure_formats": list(config.formats),
                        "figure_profile_id": profile.profile_id,
                        "figure_profile_sha256": profile.sha256(),
                        "figure_semantics_sha256": semantics_registry.sha256(),
                        "resolved_font_family": receipt["figure_profile"]["resolved_font_family"],
                        "claim_boundary": BOUNDARY,
                        "evidence_status": EVIDENCE_STATUS,
                        "source_admission_status": receipt["source_admission_status"],
                        "source_trace_provenance_status": item["trace_provenance"]["status"],
                        "producer_sha256": receipt["producer_sha256"],
                        "config_hash": receipt["config_sha256"],
                        "source_repo_commit": item["trace_provenance"]["git_hash"],
                        "source_trace_sha256": item["trace_provenance"]["artifact_sha256"],
                        "map_sha256": item["trace_provenance"]["map_digest"],
                    }
                    caption = provenance.build_caption_fragment(
                        scenario_id=str(case.get("scenario_id", "unavailable")),
                        episode_ids=prov["episode_ids"],
                    )
                    try:
                        exporter.save_publication_figure(
                            figure,
                            stage / stem / view,
                            formats=config.formats,
                            provenance=prov,
                            caption_fragment=caption + " " + BOUNDARY,
                        )
                    finally:
                        figure.clear()
                    case_record["views"].append(view_info)
                receipt["cases"].append(case_record)
        # Fail if a campaign writer changed any source byte during the export.
        if _inventory(package) != before:
            raise ValueError("source package changed while rendering; export discarded")
        _verify_source(package, config.mode)
        catalog = [
            "# Scenario figure pack",
            "",
            f"Evidence status: **{EVIDENCE_STATUS}**",
            f"Source admission status: **{receipt['source_admission_status']}**",
            f"Figure profile: **{profile.profile_id}** (`{profile.sha256()}`)",
            f"Figure semantics: `{semantics_registry.sha256()}`",
            "",
            BOUNDARY,
            "",
            f"Selected {len(selected)} / {len(proposal['portfolio'])} workbench portfolio cases.",
            "No population denominator or new relevance ranking is inferred.",
            "",
        ]
        for case in receipt["cases"]:
            catalog.extend(["## " + str(case["case_id"]).replace("\n", " "), ""])
            for view in case["views"]:
                links = " | ".join(
                    f"[{fmt.upper()}]({view['stem']}.{fmt})" for fmt in config.formats
                )
                catalog.append(f"- {view['view']} ({view['status']}): {links}")
            catalog.append("")
        if omitted:
            catalog.extend(["## Omitted cases", "", "See manifest.json for every ID and reason."])
        (stage / "README.md").write_text("\n".join(catalog) + "\n", encoding="utf-8")
        for path in sorted(p for p in stage.rglob("*") if p.is_file()):
            receipt["artifacts"].append(
                {
                    "path": path.relative_to(stage).as_posix(),
                    "sha256": _sha(path),
                    "size_bytes": path.stat().st_size,
                }
            )
        (stage / "manifest.json").write_text(_json(receipt), encoding="utf-8")
        # mkdir is the atomic no-replace reservation (rename could replace an
        # empty directory created by another process between check and publish).
        try:
            output.mkdir()
        except FileExistsError as exc:
            raise ValueError("output appeared during rendering; export discarded") from exc
        marker = output / ".INCOMPLETE"
        marker.write_text(
            "Publication interrupted unless manifest.json is present.\n", encoding="utf-8"
        )
        for child in sorted(stage.iterdir()):
            if child.name != "manifest.json":
                os.rename(child, output / child.name)
        os.rename(stage / "manifest.json", output / "manifest.json")
        marker.unlink()
        return receipt
    finally:
        if stage is not None:
            shutil.rmtree(stage)
        lock.rmdir()


def main(argv: list[str] | None = None) -> int:
    """Run the config-first, offline figure pack CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--figure-profile", type=Path)
    parser.add_argument("--case-id", action="append", default=[])
    args = parser.parse_args(argv)
    try:
        receipt = build_pack(
            args.package,
            args.output,
            config=PackConfig.from_file(args.config),
            case_ids=tuple(args.case_id),
            profile=(FigureProfile.from_file(args.figure_profile) if args.figure_profile else None),
        )
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(2, f"scenario figure pack: {exc}\n")
    print(f"{receipt['selection']['selected_count']} cases -> {args.output}")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
