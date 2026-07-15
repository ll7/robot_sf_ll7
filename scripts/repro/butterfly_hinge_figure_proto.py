"""Prototype: matched success-vs-failure "hinge figure" (the butterfly worked example).

STATUS: first-cut prototype for Stage 5-6 of the "butterfly worked-example pipeline" plan
(diss repo: docs/context/plan/2026-07-15_butterfly_worked_example_pipeline.md), built on the
adopted UI spec (diss repo: docs/context/research/2026-07-15_visualization_scenarios.md,
"The butterfly moment in one still"). Builds on the Stage 0-2 prototype in this worktree
(``scripts/repro/butterfly_trace_to_video_proto.py``): reuses ``trace_series_to_jsonl``,
``_render_scene_frames``, and ``compute_trace_metrics`` from that module rather than
re-implementing trace loading / rendering.

Chosen matched pair (see the diss-side report for the full selection rationale): SAME
scenario (``classic_head_on_corridor_medium``) + SAME seed (24), DIFFERENT planner --
``orca`` (success) vs ``social_force`` (failure, non-completion/timeout). Both episodes
start from an (almost) identical robot pose and an *exactly* identical 4-pedestrian layout
(same seed), so the shared prefix is a genuine "same start, different outcome" butterfly,
not an artifact of re-seeding.

What this script does NOT implement (first-cut scope, see the diss report for the gap list):
- The full ``D(t)`` normalized joint-state divergence algorithm from the design spec.
  Divergence is detected with a simple threshold-on-separation instead (first index where
  robot-to-robot position separation exceeds ``DEFAULT_ROBOT_RADIUS`` and stays above it for
  the remainder of the shared time window) -- explicitly permitted as a first cut.
- The central delta gutter / 2x3 contact sheet layout.
- Routing the still through ``robot_sf.benchmark.trace_scene_figure`` / the shipped
  ``figure_qa`` catalog pipeline (this is a standalone prototype renderer that borrows the
  same color/style conventions and runs the same collision linter, ``figure_qa.lint_figure``,
  as a smoke check).

Usage::

    uv run python scripts/repro/butterfly_hinge_figure_proto.py \\
        --episode-a docs/context/evidence/issue_4891_head_on_corridor_exemplars_2026-07/orca/classic_head_on_corridor_medium_seed24_best \\
        --episode-b docs/context/evidence/issue_4891_head_on_corridor_exemplars_2026-07/social_force/classic_head_on_corridor_medium_seed24_worst \\
        --label-a "orca -- success" \\
        --label-b "social_force -- non-completion (timeout)" \\
        --out-dir output/butterfly_hinge_proto
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Must be set before any robot_sf.render module imports pygame.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from butterfly_trace_to_video_proto import (
    compute_trace_metrics,
    trace_series_to_jsonl,
)

from robot_sf.benchmark import figure_qa
from robot_sf.benchmark.critical_intervals import (
    extract_critical_intervals,
    load_config,
)
from robot_sf.benchmark.trace_scene_figure import _contiguous_segments
from robot_sf.common.robot_defaults import DEFAULT_ROBOT_RADIUS

# ---------------------------------------------------------------------------
# Color / style spec (Okabe-Ito, redundant with shape+line style; no red/green
# success-failure semantics; see the adopted design-spec doc referenced above).
# ---------------------------------------------------------------------------

COLOR_A = "#0072B2"  # Okabe-Ito blue
COLOR_B = "#D55E00"  # Okabe-Ito vermilion
COLOR_COMMON_PREFIX = "#777777"
COLOR_PED_CONTEXT = "#BBBBBB"
COLOR_PED_FOCAL_OUTLINE = "#222222"
COLOR_COLLISION = "#000000"

MARKER_A = "o"  # filled circle
MARKER_B = "s"  # open square

#: Shared translucent-white backing box for direct labels, so text stays legible where it
#: crosses a trajectory line in this narrow-corridor layout (figure_qa's line-overlap lint
#: still flags the geometric bbox intersection; this is a human-legibility mitigation, not
#: a fix for that lint signal -- see the "gap to full spec" note in the module docstring).
_LABEL_BBOX = {"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.5}

#: Divergence threshold: first-cut proxy for the design spec's D(t) pivot detector.
#: "> robot radius" per the task brief; sourced from the repo's own default (1.0 m,
#: robot_sf/common/robot_defaults.py), not re-invented here.
DIVERGENCE_THRESHOLD_M: float = DEFAULT_ROBOT_RADIUS

#: Minimum sustained separation window (design spec: 0.3-0.5 s hold before calling a
#: divergence persistent). At dt=0.1 s this is 3-5 steps; we use "stays above threshold
#: for the rest of the shared trace window" instead, which is strictly stronger and holds
#: exactly for this pair (see the diss report -- separation is monotonically increasing
#: after the crossing, no re-convergence).
COMMAND_DIFF_OMEGA_THRESHOLD_RAD_S: float = 0.5


@dataclass(frozen=True)
class EpisodeTrace:
    """One loaded exemplar bundle: raw trace payload + derived per-step arrays."""

    label: str
    bundle_dir: Path
    payload: dict[str, Any]
    metadata: dict[str, Any]
    robot_xy: np.ndarray  # (T, 2)
    robot_vel: np.ndarray  # (T, 2)
    time_s: np.ndarray  # (T,)
    ped_ids: list[int]
    ped_xy: np.ndarray  # (T, N, 2), column order == ped_ids
    cmd_v: np.ndarray  # (T,) commanded linear velocity
    cmd_omega: np.ndarray  # (T,) commanded angular velocity
    metrics: dict[str, np.ndarray]  # from compute_trace_metrics: speed/clearance/nearest id


def load_episode(bundle_dir: Path, label: str, *, max_steps: int | None = None) -> EpisodeTrace:
    """Load one exemplar bundle and derive the columnar arrays this script needs.

    ``max_steps`` truncates the *story window* (for the "worst" / non-completion episode,
    which runs far longer than the interaction of interest) without touching the source
    trace file.

    Returns:
        The loaded, optionally truncated episode.
    """
    trace_series_path = bundle_dir / "trace_series.json"
    metadata_path = bundle_dir / "metadata.json"
    with trace_series_path.open(encoding="utf-8") as f:
        payload = json.load(f)
    with metadata_path.open(encoding="utf-8") as f:
        metadata = json.load(f)

    frames = payload["frames"]
    if max_steps is not None:
        frames = frames[:max_steps]
    ped_ids = sorted({p["id"] for fr in frames for p in fr.get("pedestrians", [])})
    id_to_col = {pid: i for i, pid in enumerate(ped_ids)}

    T = len(frames)
    N = len(ped_ids)
    robot_xy = np.zeros((T, 2))
    robot_vel = np.zeros((T, 2))
    time_s = np.zeros(T)
    ped_xy = np.full((T, N, 2), np.nan)
    cmd_v = np.zeros(T)
    cmd_omega = np.zeros(T)

    for t, fr in enumerate(frames):
        robot_xy[t] = fr["robot"]["position"]
        robot_vel[t] = fr["robot"]["velocity"]
        time_s[t] = fr["time_s"]
        for p in fr.get("pedestrians", []):
            ped_xy[t, id_to_col[p["id"]]] = p["position"]
        action = fr.get("planner", {}).get("selected_action", {})
        cmd_v[t] = action.get("linear_velocity", np.nan)
        cmd_omega[t] = action.get("angular_velocity", np.nan)

    # compute_trace_metrics reads the *file*, not our (possibly truncated) in-memory
    # frames; truncate its output to match so every array in this dataclass shares T.
    metrics = compute_trace_metrics(trace_series_path)
    metrics = {k: v[:T] for k, v in metrics.items()}

    return EpisodeTrace(
        label=label,
        bundle_dir=bundle_dir,
        payload=payload,
        metadata=metadata,
        robot_xy=robot_xy,
        robot_vel=robot_vel,
        time_s=time_s,
        ped_ids=ped_ids,
        ped_xy=ped_xy,
        cmd_v=cmd_v,
        cmd_omega=cmd_omega,
        metrics=metrics,
    )


def to_critical_intervals_trace(ep: EpisodeTrace) -> dict[str, Any]:
    """Reshape an :class:`EpisodeTrace` into the columnar dict ``extract_critical_intervals``
    expects: ``robot_pos`` (T,2), ``peds_pos`` (T,N,2), ``robot_vel`` (T,2), ``dt`` (scalar).

    This is the "shape adapter" flagged as missing (S1-glue-scope new code, not a blocker)
    in the plan doc's Stage-0/1 prototype-outcome note; ``extract_critical_intervals`` itself
    is untouched, reused as-is.

    Returns:
        Trace dict ready for ``robot_sf.benchmark.critical_intervals.extract_critical_intervals``.
    """
    dt = float(ep.time_s[1] - ep.time_s[0]) if len(ep.time_s) > 1 else 0.1
    return {
        "robot_pos": ep.robot_xy.tolist(),
        "peds_pos": ep.ped_xy.tolist(),
        "robot_vel": ep.robot_vel.tolist(),
        "dt": dt,
    }


CRITICAL_INTERVAL_CONFIG = {
    "schema_version": "critical-intervals.v1",
    "critical_intervals": {
        "closest_approach": {"enabled": True, "before_s": 1.0, "after_s": 1.0},
        "first_braking_event": {
            "enabled": True,
            "deceleration_threshold_mps2": 0.75,
            "before_s": 0.5,
            "after_s": 0.5,
        },
        "collision_or_near_miss": {"enabled": True, "before_s": 1.0, "after_s": 1.0},
    },
}


def compute_critical_events(ep: EpisodeTrace) -> dict[str, dict[str, Any]]:
    """Run ``extract_critical_intervals`` on one episode and collect resolved anchor steps.

    Returns:
        Mapping ``anchor -> {"step": int, "time_s": float}`` for anchors with
        ``status == "available"``. Missing anchors are omitted (not fabricated), matching
        the critical_intervals module's own "missing anchors reported, not invented" policy.
    """
    trace = to_critical_intervals_trace(ep)
    cfg = load_config(config_dict=CRITICAL_INTERVAL_CONFIG)
    intervals = extract_critical_intervals(trace, cfg)
    events: dict[str, dict[str, Any]] = {}
    for interval in intervals:
        if interval.status == "available" and interval.anchor_step is not None:
            step = interval.anchor_step
            events[interval.anchor] = {
                "step": step,
                "time_s": float(ep.time_s[step]) if step < len(ep.time_s) else None,
            }
    return events


def compute_divergence(
    ep_a: EpisodeTrace, ep_b: EpisodeTrace, threshold_m: float
) -> dict[str, Any]:
    """First-cut pivot detector: first step where robot-robot separation exceeds
    ``threshold_m`` and stays above it for the rest of the shared trace window.

    Returns:
        Dict with ``step``, ``time_s``, ``separation_m`` at the divergence step, or
        ``step: None`` if no persistent divergence is found in the shared window.
    """
    n = min(len(ep_a.robot_xy), len(ep_b.robot_xy))
    sep = np.linalg.norm(ep_a.robot_xy[:n] - ep_b.robot_xy[:n], axis=1)
    for i in range(n):
        if np.all(sep[i:] > threshold_m):
            return {
                "step": i,
                "time_s": float(ep_a.time_s[i]),
                "separation_m": float(sep[i]),
                "shared_window_steps": n,
                "monotonic_after_crossing": bool(np.all(np.diff(sep[i:]) >= -1e-9)),
            }
    return {"step": None, "time_s": None, "separation_m": None, "shared_window_steps": n}


def find_first_command_separator(
    ep_a: EpisodeTrace,
    ep_b: EpisodeTrace,
    divergence_step: int,
    *,
    search_back_s: float = 1.0,
    jump_threshold: float = COMMAND_DIFF_OMEGA_THRESHOLD_RAD_S,
) -> dict[str, Any]:
    """Search backward from the geometric divergence step for the first *discrete* command
    change (a step-to-step jump in commanded angular velocity in either trace), the
    design spec's "first differing mode / first materially different command" heuristic.

    NOTE on the simplification: the two planners' commanded omega already differ by a
    near-constant ~0.6-0.8 rad/s from step 0 (different control laws, not a discrete
    "separator" event) -- a plain ``|omega_A - omega_B| > threshold`` test would fire
    trivially at step 0 and say nothing useful. This function instead looks for a *jump*
    (``|omega[i] - omega[i-1]| > jump_threshold``) in either agent's own command sequence,
    which captures a genuine maneuver onset (e.g. a hard-turn command kicking in) rather
    than the planners' baseline behavioral gap.

    Returns:
        Dict with ``step``, ``time_s``, ``omega_a``, ``omega_b``, and ``jump_in`` (``"A"``
        or ``"B"``) at the separator step (falls back to the divergence step itself if no
        jump is found within the search window).
    """
    dt = float(ep_a.time_s[1] - ep_a.time_s[0]) if len(ep_a.time_s) > 1 else 0.1
    back_steps = max(1, round(search_back_s / dt))
    start = max(1, divergence_step - back_steps)
    for i in range(start, divergence_step + 1):
        jump_a = abs(ep_a.cmd_omega[i] - ep_a.cmd_omega[i - 1])
        jump_b = abs(ep_b.cmd_omega[i] - ep_b.cmd_omega[i - 1])
        if max(jump_a, jump_b) > jump_threshold:
            return {
                "step": i,
                "time_s": float(ep_a.time_s[i]),
                "omega_a": float(ep_a.cmd_omega[i]),
                "omega_b": float(ep_b.cmd_omega[i]),
                "jump_in": "A" if jump_a >= jump_b else "B",
                "jump_rad_s": float(max(jump_a, jump_b)),
            }
    i = divergence_step
    return {
        "step": i,
        "time_s": float(ep_a.time_s[i]),
        "omega_a": float(ep_a.cmd_omega[i]),
        "omega_b": float(ep_b.cmd_omega[i]),
        "jump_in": None,
        "jump_rad_s": 0.0,
    }


def closest_approach(ep: EpisodeTrace) -> dict[str, Any]:
    """Global closest robot-pedestrian approach within the (possibly truncated) episode.

    Returns:
        Dict with ``step``, ``time_s``, ``distance_m``, ``ped_id``, ``robot_xy``, ``ped_xy``.
    """
    clearance = ep.metrics["clearance_m"]
    step = int(np.nanargmin(clearance))
    ped_id = ep.metrics["nearest_pedestrian_id"][step]
    ped_col = ep.ped_ids.index(int(ped_id)) if not math.isnan(ped_id) else None
    ped_xy = ep.ped_xy[step, ped_col] if ped_col is not None else None
    return {
        "step": step,
        "time_s": float(ep.time_s[step]),
        "distance_m": float(clearance[step]),
        "ped_id": int(ped_id) if not math.isnan(ped_id) else None,
        "robot_xy": ep.robot_xy[step].tolist(),
        "ped_xy": ped_xy.tolist() if ped_xy is not None else None,
    }


def shared_world_bounds(
    ep_a: EpisodeTrace, ep_b: EpisodeTrace, *, margin_m: float = 1.5
) -> tuple[float, float, float, float]:
    """Bounding box covering both robots' full trajectories and all pedestrians in both
    episodes (identical for both panels -- the "identical viewport, scale, orientation"
    requirement).

    Returns:
        ``(xmin, xmax, ymin, ymax)`` in world meters, padded by ``margin_m``.
    """
    xs = [ep_a.robot_xy[:, 0], ep_b.robot_xy[:, 0]]
    ys = [ep_a.robot_xy[:, 1], ep_b.robot_xy[:, 1]]
    for ep in (ep_a, ep_b):
        flat = ep.ped_xy.reshape(-1, 2)
        xs.append(flat[:, 0])
        ys.append(flat[:, 1])
    all_x = np.concatenate(xs)
    all_y = np.concatenate(ys)
    return (
        float(np.nanmin(all_x) - margin_m),
        float(np.nanmax(all_x) + margin_m),
        float(np.nanmin(all_y) - margin_m),
        float(np.nanmax(all_y) + margin_m),
    )


# ---------------------------------------------------------------------------
# Static hinge figure
# ---------------------------------------------------------------------------


def _time_dot_indices(time_s: np.ndarray, interval_s: float = 0.5) -> list[int]:
    """Indices closest to every ``interval_s`` multiple within ``time_s``'s range.

    Returns:
        Ascending, de-duplicated frame indices.
    """
    if len(time_s) == 0:
        return []
    t0, t1 = float(time_s[0]), float(time_s[-1])
    marks = np.arange(t0, t1 + 1e-9, interval_s)
    indices = sorted({int(np.argmin(np.abs(time_s - m))) for m in marks})
    return indices


def _draw_panel(  # noqa: C901, PLR0913 - one-panel figure assembly, mirrors
    # robot_sf.benchmark.trace_scene_figure._draw_scene_panel's own noqa'd complexity for
    # the same reason (many independent drawing layers, not a single indivisible piece of
    # logic worth further splitting for a first-cut prototype).
    ax: plt.Axes,
    ep: EpisodeTrace,
    *,
    color: str,
    marker: str,
    linestyle: str,
    label: str,
    divergence_step: int | None,
    common_prefix_end: int,
    focal_ped_id: int,
    critical_events: dict[str, dict[str, Any]],
    closest: dict[str, Any],
    collision_or_near_miss_step: int | None,
    outcome_kind: str,  # "success" | "collision" | "near_miss" | "non_completion"
    bounds: tuple[float, float, float, float],
) -> None:
    """Draw one spatial panel of the hinge figure. Mutates ``ax`` in place."""
    xmin, xmax, ymin, ymax = bounds

    # -- context (non-focal) pedestrians, dimmed, stable IDs --------------------------
    # Pedestrians respawn under the same id at their goal (a ~25 m same-step position
    # jump); break the polyline at those jumps instead of drawing a spurious teleport
    # line, reusing the repo's own established fix for this exact artifact
    # (robot_sf.benchmark.trace_scene_figure._contiguous_segments / _TELEPORT_STEP_M).
    for col, pid in enumerate(ep.ped_ids):
        xy = ep.ped_xy[:, col, :]
        if pid == focal_ped_id:
            continue
        for seg_x, seg_y in _contiguous_segments(list(xy[:, 0]), list(xy[:, 1])):
            ax.plot(seg_x, seg_y, color=COLOR_PED_CONTEXT, linewidth=0.6, alpha=0.55, zorder=1)
        ax.annotate(
            f"p{pid}",
            xy[0],
            fontsize=6.5,
            color=COLOR_PED_CONTEXT,
            alpha=0.8,
            xytext=(8, 2),
            textcoords="offset points",
            bbox=_LABEL_BBOX,
        )

    # -- focal pedestrian, heavier outline, stable ID (teleport-broken, see above) ------
    focal_col = ep.ped_ids.index(focal_ped_id)
    focal_xy = ep.ped_xy[:, focal_col, :]
    for seg_x, seg_y in _contiguous_segments(list(focal_xy[:, 0]), list(focal_xy[:, 1])):
        ax.plot(
            seg_x,
            seg_y,
            color=COLOR_PED_FOCAL_OUTLINE,
            linewidth=1.1,
            alpha=0.9,
            zorder=2,
            solid_capstyle="round",
        )
    ax.annotate(
        f"focal p{focal_ped_id}",
        focal_xy[0],
        fontsize=7,
        color=COLOR_PED_FOCAL_OUTLINE,
        xytext=(-8, 10),
        textcoords="offset points",
        weight="bold",
        bbox=_LABEL_BBOX,
    )

    # -- robot trajectory: common prefix gray, post-pivot in planner style ------------
    xy = ep.robot_xy
    prefix_end = min(common_prefix_end + 1, len(xy))
    ax.plot(
        xy[:prefix_end, 0],
        xy[:prefix_end, 1],
        color=COLOR_COMMON_PREFIX,
        linewidth=1.6,
        zorder=3,
        solid_capstyle="round",
    )
    if divergence_step is not None and divergence_step < len(xy) - 1:
        ax.plot(
            xy[divergence_step:, 0],
            xy[divergence_step:, 1],
            color=color,
            linewidth=1.8,
            linestyle=linestyle,
            zorder=4,
            solid_capstyle="round",
        )

    # -- time dots every 0.5s ----------------------------------------------------------
    dot_idx = _time_dot_indices(ep.time_s, 0.5)
    ax.scatter(
        xy[dot_idx, 0],
        xy[dot_idx, 1],
        s=10,
        facecolor=color,
        edgecolor="white",
        linewidth=0.4,
        zorder=5,
    )

    # -- divergence ring -----------------------------------------------------------------
    if divergence_step is not None and divergence_step < len(xy):
        px, py = xy[divergence_step]
        ax.scatter(
            [px],
            [py],
            s=170,
            facecolor="none",
            edgecolor="black",
            linewidth=1.4,
            zorder=6,
        )
        ax.annotate(
            "divergence",
            (px, py),
            fontsize=6.5,
            xytext=(10, -14),
            textcoords="offset points",
            zorder=6,
            bbox=_LABEL_BBOX,
        )

    # -- critical-interval markers (closest_approach / first_braking_event) ------------
    for anchor, info in critical_events.items():
        step = info["step"]
        if step >= len(xy):
            continue
        px, py = xy[step]
        if anchor == "closest_approach":
            continue  # drawn explicitly below with the labelled clearance line
        if anchor == "first_braking_event":
            ax.scatter(
                [px],
                [py],
                marker="v",
                s=32,
                color=color,
                edgecolor="black",
                linewidth=0.4,
                zorder=6,
            )
            ax.annotate(
                "braking",
                (px, py),
                fontsize=6,
                xytext=(8, 4),
                textcoords="offset points",
                bbox=_LABEL_BBOX,
            )

    # -- closest-approach line, labelled with clearance ----------------------------------
    if closest["ped_xy"] is not None:
        rx, ry = closest["robot_xy"]
        pxg, pyg = closest["ped_xy"]
        ax.plot([rx, pxg], [ry, pyg], color=color, linewidth=0.8, linestyle=":", zorder=5)
        ax.annotate(
            f"clearance {closest['distance_m']:.2f} m\n@t={closest['time_s']:.1f}s",
            (pxg, pyg),
            fontsize=6.5,
            color=color,
            xytext=(-9, -22),
            textcoords="offset points",
            zorder=9,
            bbox=_LABEL_BBOX,
        )

    # -- outcome marker: collision (x) / near-miss (triangle) --------------------------
    if collision_or_near_miss_step is not None and collision_or_near_miss_step < len(xy):
        px, py = xy[collision_or_near_miss_step]
        if outcome_kind == "collision":
            ax.scatter([px], [py], marker="x", s=90, color=COLOR_COLLISION, linewidth=2.2, zorder=7)
            ax.annotate(
                "collision",
                (px, py),
                fontsize=7,
                xytext=(8, 8),
                textcoords="offset points",
                weight="bold",
                bbox=_LABEL_BBOX,
            )
        elif outcome_kind == "near_miss":
            ax.scatter(
                [px],
                [py],
                marker="^",
                s=80,
                facecolor="none",
                edgecolor=COLOR_COLLISION,
                linewidth=1.6,
                zorder=7,
            )
            ax.annotate(
                "near miss",
                (px, py),
                fontsize=7,
                xytext=(8, 8),
                textcoords="offset points",
                weight="bold",
                bbox=_LABEL_BBOX,
            )

    # -- start/end glyphs (large outlined) ----------------------------------------------
    ax.scatter(
        [xy[0, 0]],
        [xy[0, 1]],
        marker="D",
        s=55,
        facecolor="white",
        edgecolor="black",
        linewidth=1.1,
        zorder=8,
    )
    ax.scatter(
        [xy[-1, 0]],
        [xy[-1, 1]],
        marker=marker,
        s=75,
        facecolor=color,
        edgecolor="black",
        linewidth=1.1,
        zorder=8,
    )

    # -- outcome annotation for non-completion (B has no collision/near-miss marker) ----
    if outcome_kind == "non_completion":
        ax.annotate(
            "episode continues past this crop\n(non-completion / timeout, not shown)",
            (xy[-1, 0], xy[-1, 1]),
            fontsize=6,
            color=color,
            xytext=(-9, 10),
            textcoords="offset points",
            bbox=_LABEL_BBOX,
        )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(label, fontsize=10.5, loc="left", color=color, weight="bold")
    ax.tick_params(labelsize=7.5)
    ax.set_xlabel("x (m)", fontsize=8)
    ax.grid(alpha=0.15, linewidth=0.5)


def render_hinge_figure(  # noqa: PLR0913 - top-level figure assembly; each argument is a
    # distinct precomputed fact (event dicts, labels, outcome kinds) the two panels need.
    ep_a: EpisodeTrace,
    ep_b: EpisodeTrace,
    *,
    label_a: str,
    label_b: str,
    divergence: dict[str, Any],
    separator: dict[str, Any],
    focal_ped_id: int,
    events_a: dict[str, dict[str, Any]],
    events_b: dict[str, dict[str, Any]],
    closest_a: dict[str, Any],
    closest_b: dict[str, Any],
    outcome_a: str,
    outcome_b: str,
    b_outcome_step: int | None,
    headline: str,
    out_pdf: Path,
    out_png: Path,
) -> None:
    """Render + export the two-panel hinge figure (vector PDF + PNG preview)."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8.5,
            "pdf.fonttype": 42,  # embed as real (editable) glyphs, not Type 3 bitmaps
            "ps.fonttype": 42,
        }
    )
    bounds = shared_world_bounds(ep_a, ep_b)
    _, _, ymin, ymax = bounds
    xmin, xmax, _, _ = bounds
    aspect = (ymax - ymin) / (xmax - xmin)
    panel_w = 4.4
    fig_h = panel_w * aspect + 1.0
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(panel_w * 2 + 0.6, fig_h))

    common_prefix_end = divergence["step"] if divergence["step"] is not None else 0

    _draw_panel(
        ax_a,
        ep_a,
        color=COLOR_A,
        marker=MARKER_A,
        linestyle="-",
        label=f"A -- {label_a}",
        divergence_step=divergence["step"],
        common_prefix_end=common_prefix_end,
        focal_ped_id=focal_ped_id,
        critical_events=events_a,
        closest=closest_a,
        collision_or_near_miss_step=None,
        outcome_kind=outcome_a,
        bounds=bounds,
    )
    _draw_panel(
        ax_b,
        ep_b,
        color=COLOR_B,
        marker=MARKER_B,
        linestyle="--",
        label=f"B -- {label_b}",
        divergence_step=divergence["step"],
        common_prefix_end=common_prefix_end,
        focal_ped_id=focal_ped_id,
        critical_events=events_b,
        closest=closest_b,
        collision_or_near_miss_step=b_outcome_step,
        outcome_kind=outcome_b,
        bounds=bounds,
    )
    ax_b.set_ylabel("")
    ax_a.set_ylabel("y (m)", fontsize=8)

    legend_elements = [
        Line2D([0], [0], color=COLOR_COMMON_PREFIX, lw=1.6, label="common prefix (both)"),
        Line2D(
            [0],
            [0],
            color=COLOR_A,
            lw=1.8,
            marker=MARKER_A,
            markersize=5,
            label="A post-divergence",
        ),
        Line2D(
            [0],
            [0],
            color=COLOR_B,
            lw=1.8,
            linestyle="--",
            marker=MARKER_B,
            markersize=5,
            markerfacecolor="none",
            label="B post-divergence",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markeredgecolor="black",
            markerfacecolor="none",
            markersize=11,
            label="divergence",
        ),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=4,
        fontsize=7,
        frameon=False,
        bbox_to_anchor=(0.5, -0.005),
    )

    fig.suptitle(headline, fontsize=9, wrap=True, y=0.995)
    fig.tight_layout(rect=(0, 0.035, 1, 0.975))

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    # Pin CreationDate so re-renders with identical inputs are byte-identical (matplotlib
    # otherwise stamps the wall-clock time into the PDF, the only source of nondeterminism
    # observed in this script -- geometry/text content is already deterministic).
    fig.savefig(out_pdf, metadata={"CreationDate": None})
    fig.savefig(out_png, dpi=200)

    # QA smoke check: reuse the repo's own text/marker-collision linter.
    defects = figure_qa.lint_figure(fig)
    errors = [d for d in defects if getattr(d, "severity", "") == "error"]
    if errors:
        print(f"figure_qa.lint_figure: {len(errors)} error-severity defect(s):", file=sys.stderr)
        for d in errors:
            print(f"  - {d}", file=sys.stderr)
    else:
        print(f"figure_qa.lint_figure: clean ({len(defects)} non-error defect(s))")

    plt.close(fig)


# ---------------------------------------------------------------------------
# Side-by-side A/B video (secondary deliverable)
# ---------------------------------------------------------------------------


def render_ab_video(
    ep_a_bundle: Path,
    ep_b_bundle: Path,
    *,
    out_path: Path,
    b_max_steps: int,
    fps: int = 10,
    panel_width: int = 640,
    panel_height: int = 900,
    scaling: float = 18.0,
) -> dict[str, Any]:
    """Render a side-by-side A|B playback video with a *world-fixed* shared camera.

    Deviates from ``butterfly_trace_to_video_proto.render_episode_to_video`` /
    ``render_selected_frames`` on purpose: those construct ``SimulationView`` with its
    defaults, which means ``focus_on_robot=True`` (camera tracks the robot every frame --
    exactly what the adopted design spec forbids: "world-fixed, north-up, NEVER
    rotate/track the robot"). This function instead builds ``SimulationView`` directly with
    ``focus_on_robot=False`` and a fixed pixel ``offset`` computed once from the shared A/B
    world bounding box, so both panels share one static camera for the whole clip. No
    robot_sf source is modified; this only changes how *this script* constructs the
    (already-parameterized) SimulationView.

    Returns:
        Summary dict (frame counts per side, output path).
    """
    import imageio.v3 as iio

    from robot_sf.render.jsonl_playback import JSONLPlaybackLoader
    from robot_sf.render.sim_view import SimulationView

    ep_a = load_episode(ep_a_bundle, "A")
    ep_b = load_episode(ep_b_bundle, "B", max_steps=b_max_steps)
    xmin, xmax, ymin, ymax = shared_world_bounds(ep_a, ep_b, margin_m=1.5)
    cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
    fit_scaling = min(panel_width / (xmax - xmin), panel_height / (ymax - ymin))
    used_scaling = min(scaling, fit_scaling)
    offset = (
        panel_width / 2.0 - cx * used_scaling,
        panel_height / 2.0 - cy * used_scaling,
    )

    def render_fixed_camera(bundle_dir: Path, max_steps: int | None) -> list[np.ndarray]:
        with tempfile.TemporaryDirectory() as tmp_dir:
            trace_series_path = bundle_dir / "trace_series.json"
            src_payload = json.loads(trace_series_path.read_text(encoding="utf-8"))
            if max_steps is not None:
                src_payload["frames"] = src_payload["frames"][:max_steps]
            sliced_path = Path(tmp_dir) / "trace_series.json"
            sliced_path.write_text(json.dumps(src_payload), encoding="utf-8")
            jsonl_path = Path(tmp_dir) / "episode.jsonl"
            trace_series_to_jsonl(sliced_path, jsonl_path)

            loader = JSONLPlaybackLoader()
            episode, map_def = loader.load_single_episode(jsonl_path)
            states = episode.states
            sim_view = SimulationView(
                width=panel_width,
                height=panel_height,
                scaling=used_scaling,
                map_def=map_def,
                caption="butterfly hinge A/B",
                record_video=True,
                video_path=None,
                display_text=False,
                focus_on_robot=False,
                focus_on_ego_ped=False,
                manual_view_mode="fixed_map",
            )
            sim_view.offset[0] = offset[0]
            sim_view.offset[1] = offset[1]
            try:
                for state in states:
                    # SimulationView._move_camera runs before drawing and would otherwise
                    # reset offset each frame only when focus_on_robot/focus_on_ego_ped are
                    # set -- both are False here, so the fixed offset set above is preserved.
                    sim_view.render(state)
                frames = sim_view.exit_simulation(return_frames=True) or []
            finally:
                if not sim_view.is_exit_requested:
                    sim_view.exit_simulation()
            return frames

    frames_a = render_fixed_camera(ep_a_bundle, None)
    frames_b = render_fixed_camera(ep_b_bundle, b_max_steps)

    n = max(len(frames_a), len(frames_b))
    if len(frames_a) < n:
        frames_a = frames_a + [frames_a[-1]] * (n - len(frames_a))
    if len(frames_b) < n:
        frames_b = frames_b + [frames_b[-1]] * (n - len(frames_b))

    gutter = np.full((panel_height, 6, 3), 255, dtype=np.uint8)
    composed = [
        np.hstack([np.asarray(fa, dtype=np.uint8), gutter, np.asarray(fb, dtype=np.uint8)])
        for fa, fb in zip(frames_a, frames_b, strict=True)
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out_path, composed, fps=fps)
    return {
        "n_frames_a": len(frames_a),
        "n_frames_b": len(frames_b),
        "n_frames_composed": len(composed),
        "used_scaling": used_scaling,
        "offset": offset,
        "bounds": (xmin, xmax, ymin, ymax),
        "out_path": str(out_path),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _sha256_of_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episode-a", type=Path, required=True, help="Bundle dir for episode A (success)."
    )
    parser.add_argument(
        "--episode-b", type=Path, required=True, help="Bundle dir for episode B (failure)."
    )
    parser.add_argument("--label-a", default="A", help="Direct-label text for panel A.")
    parser.add_argument("--label-b", default="B", help="Direct-label text for panel B.")
    parser.add_argument(
        "--out-dir", type=Path, required=True, help="Output directory for figure + video."
    )
    parser.add_argument(
        "--b-story-steps",
        type=int,
        default=220,
        help="Truncate episode B's story window to this many steps (default: 220, ~22s at "
        "dt=0.1s -- covers B's own closest approach without dragging its full timeout tail).",
    )
    parser.add_argument("--no-video", action="store_true", help="Skip the side-by-side A/B video.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Returns:
        Process exit code.
    """
    args = _build_parser().parse_args(argv)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ep_a = load_episode(args.episode_a, args.label_a)
    ep_b = load_episode(args.episode_b, args.label_b, max_steps=args.b_story_steps)

    divergence = compute_divergence(ep_a, ep_b, DIVERGENCE_THRESHOLD_M)
    if divergence["step"] is None:
        print("error: no persistent divergence found in the shared trace window", file=sys.stderr)
        return 1
    separator = find_first_command_separator(ep_a, ep_b, divergence["step"])

    events_a = compute_critical_events(ep_a)
    events_b = compute_critical_events(ep_b)
    closest_a = closest_approach(ep_a)
    closest_b = closest_approach(ep_b)

    focal_ped_id = int(ep_a.metrics["nearest_pedestrian_id"][0])

    outcome_a = "success" if ep_a.metadata["episode_status"] == "success" else "unknown"
    b_status = ep_b.metadata["episode_status"]
    if b_status == "collision":
        outcome_b = "collision"
        b_outcome_step_full = ep_b.metadata["summary"]["global_min_distance_step"]
    elif b_status == "failure":
        outcome_b = "non_completion"
        b_outcome_step_full = None
    else:
        outcome_b = "unknown"
        b_outcome_step_full = None
    b_outcome_step = (
        b_outcome_step_full
        if b_outcome_step_full is not None and b_outcome_step_full < args.b_story_steps
        else None
    )

    d_clearance = closest_a["distance_m"] - closest_b["distance_m"]
    if separator["jump_in"] is not None:
        who = "A" if separator["jump_in"] == "A" else "B"
        separator_clause = (
            f"First separator at t={separator['time_s']:.2f} s: {who} commands a "
            f"{separator['jump_rad_s']:.2f} rad/s turn-rate jump "
            f"(omega_A={separator['omega_a']:.2f}, omega_B={separator['omega_b']:.2f} rad/s) "
        )
    else:
        separator_clause = (
            f"No discrete command jump found in the 1 s window before divergence "
            f"(omega_A={separator['omega_a']:.2f}, omega_B={separator['omega_b']:.2f} rad/s at "
            f"t={separator['time_s']:.2f} s) "
        )
    headline = (
        f"{separator_clause}"
        f"while the other continues near-unchanged; positions diverge beyond "
        f"{DIVERGENCE_THRESHOLD_M:.1f} m by t={divergence['time_s']:.2f} s. "
        f"B's subsequent minimum clearance ({closest_b['distance_m']:.2f} m at t={closest_b['time_s']:.1f} s) "
        f"is {d_clearance:.2f} m lower than A's ({closest_a['distance_m']:.2f} m at t={closest_a['time_s']:.1f} s)."
    )

    out_pdf = out_dir / "butterfly_hinge_figure_proto.pdf"
    out_png = out_dir / "butterfly_hinge_figure_proto.png"
    render_hinge_figure(
        ep_a,
        ep_b,
        label_a=args.label_a,
        label_b=args.label_b,
        divergence=divergence,
        separator=separator,
        focal_ped_id=focal_ped_id,
        events_a=events_a,
        events_b=events_b,
        closest_a=closest_a,
        closest_b=closest_b,
        outcome_a=outcome_a,
        outcome_b=outcome_b,
        b_outcome_step=b_outcome_step,
        headline=headline,
        out_pdf=out_pdf,
        out_png=out_png,
    )

    report: dict[str, Any] = {
        "episode_a": {
            "bundle": str(args.episode_a),
            "planner": ep_a.metadata["planner"],
            "episode_status": ep_a.metadata["episode_status"],
            "n_steps_used": len(ep_a.robot_xy),
        },
        "episode_b": {
            "bundle": str(args.episode_b),
            "planner": ep_b.metadata["planner"],
            "episode_status": ep_b.metadata["episode_status"],
            "n_steps_used": len(ep_b.robot_xy),
            "n_steps_full_trace": len(
                json.loads((args.episode_b / "trace_series.json").read_text())["frames"]
            ),
        },
        "divergence": divergence,
        "first_command_separator": separator,
        "focal_pedestrian_id": focal_ped_id,
        "critical_events_a": events_a,
        "critical_events_b": events_b,
        "closest_approach_a": closest_a,
        "closest_approach_b": closest_b,
        "headline": headline,
        "figure_pdf": str(out_pdf),
        "figure_png": str(out_png),
        "figure_pdf_sha256": _sha256_of_file(out_pdf),
    }

    if not args.no_video:
        video_out = out_dir / "butterfly_hinge_ab_proto.mp4"
        video_summary = render_ab_video(
            args.episode_a, args.episode_b, out_path=video_out, b_max_steps=args.b_story_steps
        )
        report["video"] = video_summary

    report_path = out_dir / "butterfly_hinge_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
