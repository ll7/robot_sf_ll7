"""Inspect one or two edge-case episodes on a synchronized Rerun timeline.

The viewer is intentionally diagnostic: it keeps the collision visible as an
outcome marker while the shared timeline and command/clearance tracks make it
possible to scrub back to the earlier trajectory divergence.

Examples:

    uv run --with rerun-sdk python scripts/tools/trace_viewer.py \
      output/exemplars/seed113 output/exemplars/seed114 --spawn

    uv run --with rerun-sdk python scripts/tools/trace_viewer.py \
      --episodes-jsonl output/benchmarks/example/episodes.jsonl \
      --seed 113 --seed 114 --save output/trace_viewer/seed113-vs-114.rrd

``rerun-sdk`` remains a disposable dependency; this script does not require a
project dependency change.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

# Running a script by path normally puts scripts/tools, rather than the checkout
# root, first on sys.path. Pin imports to this worktree so linked worktrees never
# silently import robot_sf or scripts.repro from the main checkout.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The hinge prototype imports Matplotlib. A temporary cache keeps this optional
# trace-viewer command usable in read-restricted worktrees without touching the
# user's global Matplotlib cache.
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "robot_sf_trace_viewer_matplotlib"),
)

from robot_sf.benchmark import trace_scene_figure as trace_scene  # noqa: E402
from robot_sf.benchmark.collision_definition_inventory import (  # noqa: E402
    DEFAULT_PED_RADIUS,
)
from robot_sf.benchmark.constants import NEAR_MISS_DIST  # noqa: E402
from robot_sf.common.robot_defaults import DEFAULT_ROBOT_RADIUS  # noqa: E402

APPLICATION_ID = "robot_sf_edge_case_trace_viewer"
TIMELINE_NAME = "sim_time"
STEP_TIMELINE_NAME = "step"

COLOR_A_HEX = "#0072B2"
COLOR_B_HEX = "#D55E00"
COLOR_A = (0, 114, 178)
COLOR_B = (213, 94, 0)
COLOR_MAP = (70, 70, 70)
COLOR_CONTEXT = (165, 165, 165)
COLOR_EVENT = (30, 30, 30)
COLOR_THRESHOLD = (100, 100, 100)

RADIUS_SUM_M = DEFAULT_ROBOT_RADIUS + DEFAULT_PED_RADIUS


class TraceViewerError(RuntimeError):
    """Raised when an input or recording cannot satisfy the viewer contract."""


@dataclass(frozen=True)
class EpisodeCase:
    """One episode loaded through the canonical scene and hinge loaders."""

    label: str
    root: str
    color: tuple[int, int, int]
    color_hex: str
    bundle_dir: Path
    scene_trace: trace_scene.EpisodeTrace
    hinge_trace: Any
    headline: str

    @property
    def frames(self) -> list[dict[str, Any]]:
        """Return raw frames already loaded by the hinge prototype."""
        frames = self.hinge_trace.payload.get("frames")
        if not isinstance(frames, list):
            raise TraceViewerError(f"{self.bundle_dir}/trace_series.json has no frames array")
        return frames

    @property
    def surface_clearance_m(self) -> tuple[float, ...]:
        """Return radius-aware pedestrian surface clearance for each step."""
        return tuple(
            float(center_distance) - RADIUS_SUM_M
            for center_distance in self.scene_trace.min_robot_ped_distance_m
        )

    @property
    def outcome(self) -> str:
        """Return the normalized episode outcome used in the headline."""
        metadata = self.scene_trace.metadata
        summary = metadata.get("summary", {})
        value = metadata.get(
            "episode_status",
            metadata.get("status", summary.get("episode_status", "unknown")),
        )
        return str(value).strip().lower().replace("_", " ")


@dataclass(frozen=True)
class FocalSelection:
    """Pedestrian IDs kept fixed while an episode is scrubbed."""

    by_label: Mapping[str, str]
    shared: bool
    note: str


@dataclass
class EntityAudit:
    """Observed logging calls for one Rerun entity."""

    static_logs: int = 0
    times_s: list[float] = field(default_factory=list)

    @property
    def time_range_s(self) -> tuple[float, float] | None:
        """Return the temporal extent of non-static logs."""
        if not self.times_s:
            return None
        return min(self.times_s), max(self.times_s)


@dataclass
class RecordingAudit:
    """Mirror entity paths and timestamps sent to one RecordingStream."""

    recording: Any
    entities: dict[str, EntityAudit] = field(default_factory=dict)
    current_time_s: float | None = None
    current_step: int | None = None

    def set_time(self, *, time_s: float, step: int) -> None:
        """Set the shared duration timeline and per-episode step timeline."""
        self.current_time_s = float(time_s)
        self.current_step = int(step)
        _set_recording_time(
            self.recording,
            timeline=TIMELINE_NAME,
            value=float(time_s),
            kind="duration",
        )
        _set_recording_time(
            self.recording,
            timeline=STEP_TIMELINE_NAME,
            value=int(step),
            kind="sequence",
        )

    def log(self, entity_path: str, archetype: Any, *, static: bool = False) -> None:
        """Log an archetype and retain a path/time audit for verification."""
        self.recording.log(entity_path, archetype, static=static)
        entity = self.entities.setdefault(entity_path, EntityAudit())
        if static:
            entity.static_logs += 1
            return
        if self.current_time_s is None:
            raise TraceViewerError(f"temporal entity logged before time was set: {entity_path}")
        entity.times_s.append(self.current_time_s)


@dataclass(frozen=True)
class VerificationReport:
    """Programmatic proof of emitted entities and their shared-time extents."""

    entity_paths: tuple[str, ...]
    time_ranges_s: Mapping[str, tuple[float, float]]
    rrd_path: Path | None
    rrd_size_bytes: int | None


def _hinge_module() -> Any:
    """Import the existing hinge metrics and loader lazily."""
    from scripts.repro import butterfly_hinge_figure_proto

    return butterfly_hinge_figure_proto


def _adapter_module() -> Any:
    """Import the existing raw-episode adapter lazily."""
    from scripts.repro import butterfly_reexport_to_trace_series

    return butterfly_reexport_to_trace_series


def _import_rerun() -> tuple[Any, Any]:
    """Import the optional Rerun SDK and blueprint module."""
    try:
        import rerun as rr  # type: ignore[import-not-found]
        import rerun.blueprint as rrb  # type: ignore[import-not-found]
    except ImportError as exc:
        raise TraceViewerError(
            "Rerun SDK is not installed. Run this tool with "
            "`uv run --with rerun-sdk python scripts/tools/trace_viewer.py ...`; "
            "do not add it to pyproject.toml."
        ) from exc
    return rr, rrb


def _headline(case_label: str, hinge_trace: Any) -> str:
    """Build the requested one-sentence per-episode case summary."""
    hinge = _hinge_module()
    closest = hinge.closest_approach(hinge_trace)
    near_miss_steps = hinge.count_near_miss_steps(hinge_trace)
    metadata = hinge_trace.metadata
    summary = metadata.get("summary", {})
    outcome = str(
        metadata.get(
            "episode_status",
            metadata.get("status", summary.get("episode_status", "unknown")),
        )
    ).replace("_", " ")
    step_count = len(hinge_trace.time_s)
    surface_clearance = float(closest["distance_m"]) - RADIUS_SUM_M
    return (
        f"Episode {case_label} ended in {outcome} after {step_count} steps; "
        f"its minimum pedestrian surface clearance was {surface_clearance:.2f} m "
        f"at t={float(closest['time_s']):.2f} s, with {near_miss_steps} near-miss steps."
    )


def load_episode_bundle(
    bundle_dir: Path,
    *,
    label: str,
    color: tuple[int, int, int],
    color_hex: str,
) -> EpisodeCase:
    """Load and cross-check one bundle through existing repository loaders."""
    bundle_dir = bundle_dir.resolve()
    scene_episode = trace_scene.load_episode(bundle_dir)
    hinge_episode = _hinge_module().load_episode(bundle_dir, label)

    if tuple(hinge_episode.time_s) != scene_episode.time_s:
        raise TraceViewerError(
            f"{bundle_dir}: hinge and canonical scene loaders disagree on timestamps"
        )
    if len(hinge_episode.robot_xy) != len(scene_episode.robot_xy):
        raise TraceViewerError(
            f"{bundle_dir}: hinge and canonical scene loaders disagree on step count"
        )
    frames = hinge_episode.payload.get("frames")
    if not isinstance(frames, list) or len(frames) != len(scene_episode.steps):
        raise TraceViewerError(f"{bundle_dir}: frames do not align with the validated trace")

    return EpisodeCase(
        label=label,
        root=f"episode_{label}",
        color=color,
        color_hex=color_hex,
        bundle_dir=bundle_dir,
        scene_trace=scene_episode,
        hinge_trace=hinge_episode,
        headline=_headline(label, hinge_episode),
    )


def load_episode_bundles(bundle_dirs: Sequence[Path]) -> list[EpisodeCase]:
    """Load one or two bundle directories as labeled A/B cases."""
    if not 1 <= len(bundle_dirs) <= 2:
        raise TraceViewerError("expected one or two episode bundle directories")
    styles = (("A", COLOR_A, COLOR_A_HEX), ("B", COLOR_B, COLOR_B_HEX))
    return [
        load_episode_bundle(path, label=label, color=color, color_hex=color_hex)
        for path, (label, color, color_hex) in zip(
            bundle_dirs,
            styles[: len(bundle_dirs)],
            strict=True,
        )
    ]


@contextmanager
def prepared_episode_dirs(
    *,
    bundle_dirs: Sequence[Path],
    episodes_jsonl: Path | None,
    seeds: Sequence[int],
) -> Iterator[list[Path]]:
    """Yield bundle directories, adapting raw JSONL rows in temporary storage."""
    if episodes_jsonl is None:
        if seeds:
            raise TraceViewerError("--seed requires --episodes-jsonl")
        if not 1 <= len(bundle_dirs) <= 2:
            raise TraceViewerError("provide one or two episode bundle directories")
        yield [path.resolve() for path in bundle_dirs]
        return

    if bundle_dirs:
        raise TraceViewerError("bundle directories and --episodes-jsonl are mutually exclusive")
    if not 1 <= len(seeds) <= 2:
        raise TraceViewerError("--episodes-jsonl requires one or two repeated --seed values")
    if len(set(seeds)) != len(seeds):
        raise TraceViewerError("--seed values must be unique")
    episodes_jsonl = episodes_jsonl.resolve()
    if not episodes_jsonl.is_file():
        raise TraceViewerError(f"episodes JSONL does not exist: {episodes_jsonl}")

    adapter = _adapter_module()
    with tempfile.TemporaryDirectory(prefix="robot_sf_trace_viewer_") as temp_dir:
        root = Path(temp_dir)
        adapted_dirs: list[Path] = []
        for label, seed in zip(("A", "B")[: len(seeds)], seeds, strict=True):
            bundle_dir = root / f"episode_{label}_seed_{seed}"
            adapter.build_bundle(episodes_jsonl, seed, bundle_dir)
            adapted_dirs.append(bundle_dir)
        yield adapted_dirs


def select_focal_pedestrians(cases: Sequence[EpisodeCase]) -> FocalSelection:
    """Lock one focal pedestrian per episode; share the ID when possible."""
    if len(cases) == 1:
        focal_id = trace_scene._focal_pedestrian_id(cases[0].scene_trace)
        if focal_id is None:
            raise TraceViewerError("episode has no pedestrian available for focal selection")
        return FocalSelection(
            by_label={cases[0].label: focal_id},
            shared=True,
            note=f"Locked focal pedestrian {focal_id} at global closest approach.",
        )

    hinge = _hinge_module()
    try:
        selection = hinge.select_focal_pedestrian(
            cases[0].bundle_dir,
            cases[1].hinge_trace,
        )
    except (ValueError, KeyError) as exc:
        focal_ids = {
            case.label: trace_scene._focal_pedestrian_id(case.scene_trace) for case in cases
        }
        if any(value is None for value in focal_ids.values()):
            raise TraceViewerError("pair has no usable focal pedestrian") from exc
        normalized_ids = {key: str(value) for key, value in focal_ids.items()}
        return FocalSelection(
            by_label=normalized_ids,
            shared=len(set(normalized_ids.values())) == 1,
            note=(
                "The episodes do not support one shared pedestrian ID; each panel locks "
                "its own closest-approach pedestrian."
            ),
        )

    focal_id = str(selection["ped_id"])
    return FocalSelection(
        by_label={case.label: focal_id for case in cases},
        shared=True,
        note=f"Both panels lock focal pedestrian {focal_id} selected from episode A.",
    )


def pair_contrast_headline(
    cases: Sequence[EpisodeCase],
    focal: FocalSelection,
) -> str | None:
    """Build a one-sentence A/B contrast from hinge-prototype metrics."""
    if len(cases) != 2:
        return None
    hinge = _hinge_module()
    focal_closest: list[dict[str, Any]] = []
    events: list[dict[str, dict[str, Any]]] = []
    for case in cases:
        focal_id = int(focal.by_label[case.label])
        focal_closest.append(hinge.closest_approach_to_focal_ped(case.hinge_trace, focal_id))
        events.append(hinge.compute_critical_events(case.hinge_trace))

    contrast = hinge.compute_contrast_gutter(
        cases[0].hinge_trace,
        cases[1].hinge_trace,
        focal_closest[0],
        focal_closest[1],
        events[0],
        events[1],
    )
    clearance = contrast["min_clearance_focal_m"]
    near_misses = contrast["near_miss_steps"]
    steps = contrast["steps_to_termination"]
    surface_a = float(clearance["episode_a"]) - RADIUS_SUM_M
    surface_b = float(clearance["episode_b"]) - RADIUS_SUM_M
    return (
        f"Contrast: A ended in {cases[0].outcome} after {steps['episode_a']} steps "
        f"while B ended in {cases[1].outcome} after {steps['episode_b']} steps; "
        f"minimum focal-pedestrian surface clearance was {surface_a:.2f} m versus "
        f"{surface_b:.2f} m, with {near_misses['episode_a']} versus "
        f"{near_misses['episode_b']} near-miss steps."
    )


def _entity_slug(value: object) -> str:
    """Return an entity-path-safe, readable identifier."""
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return slug or "unknown"


def _scenario_id(case: EpisodeCase) -> str:
    """Return the scenario identifier needed to resolve static map geometry."""
    metadata = case.scene_trace.metadata
    summary = metadata.get("summary", {})
    scenario_id = metadata.get("scenario_id", summary.get("scenario_id"))
    if not isinstance(scenario_id, str) or not scenario_id:
        raise TraceViewerError(f"{case.bundle_dir}: metadata has no scenario_id")
    return scenario_id


def _map_obstacle_strips(case: EpisodeCase) -> list[list[tuple[float, float]]]:
    """Resolve and close map obstacle polygons through the scene renderer."""
    scenario_id = _scenario_id(case)
    try:
        map_definition = trace_scene._load_map_definition(scenario_id)
    except (OSError, ValueError) as exc:
        raise TraceViewerError(
            f"cannot load static map obstacles for scenario '{scenario_id}': {exc}"
        ) from exc

    strips: list[list[tuple[float, float]]] = []
    for obstacle in map_definition.obstacles:
        vertices = trace_scene._obstacle_vertices(obstacle)
        if not vertices:
            continue
        strips.append([*vertices, vertices[0]])
    if not strips:
        raise TraceViewerError(f"scenario '{scenario_id}' has no map obstacles to log")
    return strips


def _trajectory_strips(
    track: Sequence[tuple[float, float, float]],
) -> list[list[tuple[float, float]]]:
    """Split pedestrian paths at simulator respawns using renderer logic."""
    xs = [sample[1] for sample in track]
    ys = [sample[2] for sample in track]
    strips = []
    for segment_xs, segment_ys in trace_scene._contiguous_segments(xs, ys):
        points = list(zip(segment_xs, segment_ys, strict=True))
        if len(points) == 1:
            points.append(points[0])
        strips.append(points)
    return strips


def _line_prefix(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """Return a LineStrips2D-compatible path, including for the first frame."""
    prefix = list(points)
    if len(prefix) == 1:
        prefix.append(prefix[0])
    return prefix


def _set_recording_time(
    recording: Any,
    *,
    timeline: str,
    value: float | int,
    kind: str,
) -> None:
    """Set time across supported Rerun RecordingStream API generations."""
    legacy_name = "set_time_seconds" if kind == "duration" else "set_time_sequence"
    legacy_method = getattr(recording, legacy_name, None)
    if callable(legacy_method):
        legacy_method(timeline, value)
        return
    modern_method = getattr(recording, "set_time", None)
    if not callable(modern_method):
        raise TraceViewerError("Rerun RecordingStream has no supported time-setting API")
    keyword = "duration" if kind == "duration" else "sequence"
    modern_method(timeline, **{keyword: value})


def _log_static_scene(
    audit: RecordingAudit,
    rr: Any,
    case: EpisodeCase,
    focal_id: str,
) -> None:
    """Log static map and pedestrian trajectory geometry."""
    audit.log(
        f"{case.root}/scene/map/obstacles",
        rr.LineStrips2D(
            strips=_map_obstacle_strips(case),
            colors=[COLOR_MAP],
            radii=0.05,
        ),
        static=True,
    )
    for ped_id, track in case.scene_trace.pedestrian_tracks.items():
        is_focal = ped_id == focal_id
        color = case.color if is_focal else COLOR_CONTEXT
        audit.log(
            f"{case.root}/scene/pedestrians/ped_{_entity_slug(ped_id)}/trajectory",
            rr.LineStrips2D(
                strips=_trajectory_strips(track),
                colors=[color],
                radii=0.055 if is_focal else 0.025,
            ),
            static=True,
        )


def _log_series_styles(audit: RecordingAudit, rr: Any, case: EpisodeCase) -> None:
    """Give A/B scalar tracks the same color identity as their scene."""
    styles = (
        ("min_pedestrian_clearance_m", f"{case.label} clearance", case.color, 2.0),
        ("speed_command_m_s", f"{case.label} speed command", case.color, 2.0),
        ("omega_command_rad_s", f"{case.label} omega command", case.color, 2.0),
        (
            "near_miss_threshold_m",
            f"{case.label} near-miss threshold",
            COLOR_THRESHOLD,
            1.0,
        ),
    )
    # rerun-sdk >=0.23 renamed SeriesLine -> SeriesLines (widths became plural too).
    series_cls = getattr(rr, "SeriesLines", None) or rr.SeriesLine
    plural = series_cls.__name__.endswith("s")
    for name, display_name, color, width in styles:
        kwargs = (
            {"colors": color, "names": display_name, "widths": width}
            if plural
            else {"color": color, "name": display_name, "width": width}
        )
        audit.log(
            f"{case.root}/metrics/{name}",
            series_cls(**kwargs),
            static=True,
        )


def _terminal_collision(case: EpisodeCase, index: int) -> bool:
    """Return whether this frame should carry a collision consequence marker."""
    if case.surface_clearance_m[index] < 0.0:
        return True
    return index == len(case.scene_trace.steps) - 1 and "collision" in case.outcome


def _log_frame(
    audit: RecordingAudit,
    rr: Any,
    case: EpisodeCase,
    *,
    index: int,
    focal_id: str,
) -> None:
    """Log one spatial and metric sample at the episode's simulation time."""
    scene = case.scene_trace
    frame = case.frames[index]
    step = scene.steps[index]
    time_s = scene.time_s[index]
    if frame.get("step") != step or not math.isclose(
        float(frame.get("time_s", math.nan)),
        time_s,
        abs_tol=1e-9,
    ):
        raise TraceViewerError(f"{case.bundle_dir}: raw frame {index} drifted from validated trace")

    audit.set_time(time_s=time_s, step=step)
    robot_points = list(scene.robot_xy[: index + 1])
    robot_xy = scene.robot_xy[index]
    audit.log(
        f"{case.root}/scene/robot/path",
        rr.LineStrips2D(
            strips=[_line_prefix(robot_points)],
            colors=[case.color],
            radii=0.065,
        ),
    )
    audit.log(
        f"{case.root}/scene/robot/position",
        rr.Points2D(
            positions=[robot_xy],
            colors=[case.color],
            radii=0.28,
            labels=[f"{case.label} robot"],
        ),
    )

    for pedestrian in frame.get("pedestrians", []):
        ped_id = str(pedestrian["id"])
        is_focal = ped_id == focal_id
        audit.log(
            f"{case.root}/scene/pedestrians/ped_{_entity_slug(ped_id)}/position",
            rr.Points2D(
                positions=[pedestrian["position"]],
                colors=[case.color if is_focal else COLOR_CONTEXT],
                radii=0.23 if is_focal else 0.14,
                labels=[f"focal p{ped_id}" if is_focal else f"p{ped_id}"],
            ),
        )

    clearance = case.surface_clearance_m[index]
    audit.log(
        f"{case.root}/metrics/min_pedestrian_clearance_m",
        rr.Scalars(clearance),
    )
    audit.log(
        f"{case.root}/metrics/speed_command_m_s",
        rr.Scalars(float(case.hinge_trace.cmd_v[index])),
    )
    audit.log(
        f"{case.root}/metrics/omega_command_rad_s",
        rr.Scalars(float(case.hinge_trace.cmd_omega[index])),
    )
    audit.log(
        f"{case.root}/metrics/near_miss_threshold_m",
        rr.Scalars(NEAR_MISS_DIST),
    )

    if 0.0 <= clearance < NEAR_MISS_DIST:
        audit.log(
            f"{case.root}/scene/events/near_miss",
            rr.Points2D(
                positions=[robot_xy],
                colors=[COLOR_EVENT],
                radii=0.22,
                labels=[f"near miss ({clearance:.2f} m)"],
            ),
        )
    if _terminal_collision(case, index):
        audit.log(
            f"{case.root}/scene/events/collision",
            rr.Points2D(
                positions=[robot_xy],
                colors=[COLOR_EVENT],
                radii=0.34,
                labels=["collision consequence"],
            ),
        )


def build_blueprint(rrb: Any, cases: Sequence[EpisodeCase]) -> Any:
    """Build a side-by-side scene over three shared metric tracks."""
    spatial_views = [
        rrb.Spatial2DView(
            origin=f"/{case.root}/scene",
            name=f"Episode {case.label} — {case.outcome} ({case.color_hex})",
        )
        for case in cases
    ]
    spatial_layout = (
        spatial_views[0]
        if len(spatial_views) == 1
        else rrb.Horizontal(*spatial_views, column_shares=[1, 1])
    )

    def metric_contents(metric_name: str) -> list[str]:
        return [f"/{case.root}/metrics/{metric_name}" for case in cases]

    clearance_contents = [
        *metric_contents("min_pedestrian_clearance_m"),
        *metric_contents("near_miss_threshold_m"),
    ]
    layout = rrb.Vertical(
        spatial_layout,
        rrb.TimeSeriesView(
            origin="/",
            name="Minimum pedestrian surface clearance (m)",
            contents=clearance_contents,
        ),
        rrb.TimeSeriesView(
            origin="/",
            name="Commanded linear speed (m/s)",
            contents=metric_contents("speed_command_m_s"),
        ),
        rrb.TimeSeriesView(
            origin="/",
            name="Commanded angular speed (rad/s)",
            contents=metric_contents("omega_command_rad_s"),
        ),
        row_shares=[4, 1, 1, 1],
    )
    return rrb.Blueprint(layout)


def _send_blueprint(recording: Any, rr: Any, blueprint: Any) -> None:
    """Send a blueprint through instance or compatibility global API."""
    method = getattr(recording, "send_blueprint", None)
    if callable(method):
        method(blueprint)
        return
    global_method = getattr(rr, "send_blueprint", None)
    if callable(global_method):
        global_method(blueprint, recording=recording)
        return
    raise TraceViewerError("installed Rerun SDK cannot send blueprints")


def log_cases(
    recording: Any,
    rr: Any,
    rrb: Any,
    cases: Sequence[EpisodeCase],
) -> tuple[RecordingAudit, FocalSelection, str | None]:
    """Log all cases and return the exact emitted-entity audit."""
    focal = select_focal_pedestrians(cases)
    contrast = pair_contrast_headline(cases, focal)
    _send_blueprint(recording, rr, build_blueprint(rrb, cases))

    audit = RecordingAudit(recording)
    for case in cases:
        _log_static_scene(audit, rr, case, focal.by_label[case.label])
        _log_series_styles(audit, rr, case)
        audit.set_time(time_s=case.scene_trace.time_s[0], step=case.scene_trace.steps[0])
        audit.log(f"{case.root}/summary/headline", rr.TextLog(case.headline))
        for index in range(len(case.scene_trace.steps)):
            _log_frame(
                audit,
                rr,
                case,
                index=index,
                focal_id=focal.by_label[case.label],
            )

    if contrast is not None:
        first_time = min(case.scene_trace.time_s[0] for case in cases)
        audit.set_time(time_s=first_time, step=0)
        audit.log("comparison/headline", rr.TextLog(contrast))
        audit.log(
            "comparison/evidence_boundary",
            rr.TextLog(
                "Shared-clock diagnostic contrast; different seeds are descriptive, "
                "not a counterfactual replay or a causal pivot validation."
            ),
        )
    return audit, focal, contrast


def _assert_time_range(
    audit: RecordingAudit,
    entity_path: str,
    expected: tuple[float, float],
) -> None:
    """Require an entity and exact shared-time extent in the emission audit."""
    entity = audit.entities.get(entity_path)
    if entity is None:
        raise TraceViewerError(f"recording is missing required entity: {entity_path}")
    observed = entity.time_range_s
    if observed is None or not all(
        math.isclose(actual, wanted, abs_tol=1e-9)
        for actual, wanted in zip(observed, expected, strict=True)
    ):
        raise TraceViewerError(
            f"{entity_path} time range {observed} does not match expected {expected}"
        )


def verify_recording_contract(  # noqa: C901 - explicit entity-by-entity proof checklist
    audit: RecordingAudit,
    cases: Sequence[EpisodeCase],
    *,
    rrd_path: Path | None = None,
) -> VerificationReport:
    """Verify required paths/time ranges sent to the saved RecordingStream."""
    for case in cases:
        map_path = f"{case.root}/scene/map/obstacles"
        map_audit = audit.entities.get(map_path)
        if map_audit is None or map_audit.static_logs < 1:
            raise TraceViewerError(f"recording is missing static map entity: {map_path}")

        expected_range = (case.scene_trace.time_s[0], case.scene_trace.time_s[-1])
        for suffix in (
            "scene/robot/path",
            "scene/robot/position",
            "metrics/min_pedestrian_clearance_m",
            "metrics/speed_command_m_s",
            "metrics/omega_command_rad_s",
            "metrics/near_miss_threshold_m",
        ):
            _assert_time_range(audit, f"{case.root}/{suffix}", expected_range)
        _assert_time_range(
            audit,
            f"{case.root}/summary/headline",
            (case.scene_trace.time_s[0], case.scene_trace.time_s[0]),
        )

        for ped_id, track in case.scene_trace.pedestrian_tracks.items():
            trajectory_path = f"{case.root}/scene/pedestrians/ped_{_entity_slug(ped_id)}/trajectory"
            trajectory = audit.entities.get(trajectory_path)
            if trajectory is None or trajectory.static_logs < 1:
                raise TraceViewerError(
                    f"recording is missing static pedestrian trajectory: {trajectory_path}"
                )
            _assert_time_range(
                audit,
                f"{case.root}/scene/pedestrians/ped_{_entity_slug(ped_id)}/position",
                (track[0][0], track[-1][0]),
            )

        near_miss_indices = [
            index
            for index, clearance in enumerate(case.surface_clearance_m)
            if 0.0 <= clearance < NEAR_MISS_DIST
        ]
        if near_miss_indices:
            _assert_time_range(
                audit,
                f"{case.root}/scene/events/near_miss",
                (
                    case.scene_trace.time_s[near_miss_indices[0]],
                    case.scene_trace.time_s[near_miss_indices[-1]],
                ),
            )
        if "collision" in case.outcome:
            collision = audit.entities.get(f"{case.root}/scene/events/collision")
            if collision is None or case.scene_trace.time_s[-1] not in collision.times_s:
                raise TraceViewerError(
                    f"{case.root} lacks a terminal collision marker at "
                    f"t={case.scene_trace.time_s[-1]}"
                )

    if len(cases) == 2:
        first_time = min(case.scene_trace.time_s[0] for case in cases)
        _assert_time_range(audit, "comparison/headline", (first_time, first_time))
        _assert_time_range(audit, "comparison/evidence_boundary", (first_time, first_time))

    rrd_size: int | None = None
    resolved_rrd: Path | None = None
    if rrd_path is not None:
        resolved_rrd = rrd_path.resolve()
        if not resolved_rrd.is_file():
            raise TraceViewerError(f"Rerun recording was not created: {resolved_rrd}")
        rrd_size = resolved_rrd.stat().st_size
        if rrd_size <= 0:
            raise TraceViewerError(f"Rerun recording is empty: {resolved_rrd}")

    return VerificationReport(
        entity_paths=tuple(sorted(audit.entities)),
        time_ranges_s={
            path: entity.time_range_s
            for path, entity in sorted(audit.entities.items())
            if entity.time_range_s is not None
        },
        rrd_path=resolved_rrd,
        rrd_size_bytes=rrd_size,
    )


def _configure_recording_sink(
    recording: Any,
    rr: Any,
    *,
    save: Path | None,
    spawn: bool,
    serve: bool,
) -> object | None:
    """Configure exactly one headless or interactive Rerun sink."""
    if save is not None:
        save.parent.mkdir(parents=True, exist_ok=True)
        recording.save(str(save))
        return None
    if spawn:
        method = getattr(recording, "spawn", None)
        if callable(method):
            method()
            return None
        rr.spawn(recording=recording)
        return None
    if serve:
        for owner in (recording, rr):
            for method_name in ("serve_web", "serve_grpc"):
                method = getattr(owner, method_name, None)
                if callable(method):
                    try:
                        return method(open_browser=True)
                    except TypeError:
                        return method()
        raise TraceViewerError("installed Rerun SDK has no supported serve API")
    raise TraceViewerError("select one of --save, --spawn, or --serve")


def _flush(recording: Any) -> None:
    """Flush buffered messages across supported RecordingStream versions."""
    flush = getattr(recording, "flush", None)
    if not callable(flush):
        return
    try:
        flush(blocking=True)
    except TypeError:
        flush()


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "bundle_dirs",
        nargs="*",
        type=Path,
        help="One or two directories containing trace_series.json and metadata.json.",
    )
    parser.add_argument(
        "--episodes-jsonl",
        type=Path,
        help="Raw episodes.jsonl input; requires one or two repeated --seed values.",
    )
    parser.add_argument(
        "--seed",
        action="append",
        default=[],
        type=int,
        help="Episode seed to adapt from --episodes-jsonl; repeat for an A/B pair.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--save", type=Path, metavar="OUT.RRD", help="Write a headless RRD recording."
    )
    mode.add_argument("--spawn", action="store_true", help="Spawn the native interactive viewer.")
    mode.add_argument(
        "--serve",
        action="store_true",
        help="Serve an interactive viewer until interrupted.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the trace-viewer CLI."""
    args = _build_parser().parse_args(argv)
    try:
        rr, rrb = _import_rerun()
        with prepared_episode_dirs(
            bundle_dirs=args.bundle_dirs,
            episodes_jsonl=args.episodes_jsonl,
            seeds=args.seed,
        ) as bundle_dirs:
            cases = load_episode_bundles(bundle_dirs)
            recording = rr.RecordingStream(APPLICATION_ID)
            server_guard = _configure_recording_sink(
                recording,
                rr,
                save=args.save,
                spawn=args.spawn,
                serve=args.serve,
            )
            audit, focal, contrast = log_cases(recording, rr, rrb, cases)
            _flush(recording)
            report = verify_recording_contract(audit, cases, rrd_path=args.save)

            for case in cases:
                print(case.headline)
            if contrast is not None:
                print(contrast)
            print(focal.note)
            ranges = ", ".join(
                f"{case.label}={case.scene_trace.time_s[0]:.2f}..{case.scene_trace.time_s[-1]:.2f}s"
                for case in cases
            )
            print(f"Verified {len(report.entity_paths)} entity paths on {TIMELINE_NAME}: {ranges}.")
            if report.rrd_path is not None:
                print(
                    f"Saved {report.rrd_path} "
                    f"({report.rrd_size_bytes} bytes) after recording-contract verification."
                )
            if args.serve:
                print("Rerun server is active; press Ctrl-C to stop.")
                # Keep any returned server guard alive for the duration of the wait.
                _ = server_guard
                try:
                    threading.Event().wait()
                except KeyboardInterrupt:
                    print("Rerun server stopped.")
    except (OSError, ValueError, KeyError, TraceViewerError) as exc:
        print(f"trace viewer error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
