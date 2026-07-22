"""Prototype: matched success-vs-failure hinge figure for benchmark trace review.

STATUS: second-cut trace-tooling prototype for an A/B episode comparison pipeline.
Builds on the Stage 0-2 prototype in this worktree
(``scripts/repro/butterfly_trace_to_video_proto.py``):
reuses ``trace_series_to_jsonl``, ``_render_scene_frames``, and ``compute_trace_metrics``
from that module, plus ``robot_sf.benchmark.trace_scene_figure``'s ``_focal_pedestrian_id``
(focal-pedestrian selection), ``_contiguous_segments`` (teleport-safe polylines),
``_load_map_definition`` / ``_draw_obstacles`` (map geometry), and
``robot_sf.benchmark.figure_qa.lint_figure`` (the QA gate) rather than re-implementing any
of them.

The reference matched pair uses the SAME
scenario (``classic_head_on_corridor_medium``) + SAME seed (24), DIFFERENT planner --
``orca`` (success) vs ``social_force`` (failure, non-completion/timeout). Both episodes
start from an (almost) identical robot pose and an *exactly* identical 4-pedestrian layout
(same seed), so the shared prefix is a genuine "same start, different outcome" butterfly,
not an artifact of re-seeding.

Implemented since the first cut:
- ``select_focal_pedestrian``: locks ONE focal pedestrian, shared by both panels, reusing
  ``trace_scene_figure._focal_pedestrian_id`` -- fixes the first cut's "focal p3 floats
  disconnected from the interaction" bug (that bug was ``nearest_pedestrian_id[step=0]``).
- ``compute_joint_state_divergence`` / ``find_persistence_onset``: the design spec's D(t)
  normalized joint-state divergence detector (robot pose + commanded v/omega + clearance to
  the locked focal pedestrian, physically-scaled), replacing the first cut's threshold on
  raw robot-robot separation.
- ``find_separator``: the design spec's tiered backward search (mode / command jump /
  braking onset / largest risk rise) for the pivot, with an honest "unavailable" report for
  the mode tier (this trace schema has no per-step categorical planner-mode field).
- ``compute_delta_gutter`` / ``_draw_delta_gutter``: the design spec's central ``A | Delta |
  B`` gutter (Delta t_brake, Delta v_cmd at pivot, min clearance over the following horizon,
  first differing mode).
- Map obstacles reused from ``trace_scene_figure`` (item 4); for this pair's tight
  trajectory-based crop the scenario's only obstacles are far-field corridor boundary walls
  outside the crop, so nothing additional renders -- a genuine finding, not a wiring bug.
- ``figure_qa.lint_figure`` error-severity defects: 4 (first cut) -> 0 (this render); see
  ``_place_clear_label`` for the closed-loop, render-and-check label placement that got
  there (three prior heuristic attempts are documented in its docstring as a paper trail).
- ``build_provenance_sidecar``: episode ids/planners/seeds/source commits, trace file
  hashes, script commit, detector config, and figure hash, written to
  ``butterfly_hinge_provenance.json`` alongside the analysis-dump ``butterfly_hinge_report.json``.

Still NOT implemented:
- The 2x3 contact sheet layout; this geometry-dominant pair currently uses two panels.
- Counterfactual-replay validation (would upgrade the pivot label from "explanatory pivot
  candidate" to "counterfactually validated pivot"); no checkpointed replay is available.
- Rendering directly through ``trace_scene_figure.render_comparison`` (it lacks the hinge
  grammar -- common-prefix/post-pivot A/B styling, pivot ring, delta gutter). This module
  remains a standalone hinge composer while reusing the shared trace loaders and map
  helpers; its role-color constants (robot, focal pedestrian) now match
  ``trace_scene_figure``'s (``INK``/``ORANGE``) rather than diverging into a separate
  per-episode palette -- see the constants block below.

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
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

# Must be set before any robot_sf.render module imports pygame.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from butterfly_trace_to_video_proto import (
    compute_trace_metrics,
    trace_series_to_jsonl,
)

from robot_sf.analysis_workbench.trace_failure_predicates import (
    DEFAULT_CLEARANCE_THRESHOLD_M,
)
from robot_sf.benchmark import figure_qa
from robot_sf.benchmark import trace_scene_figure as tsf
from robot_sf.benchmark.collision_definition_inventory import DEFAULT_PED_RADIUS
from robot_sf.benchmark.constants import NEAR_MISS_DIST
from robot_sf.benchmark.critical_intervals import (
    extract_critical_intervals,
    load_config,
)
from robot_sf.benchmark.trace_scene_figure import _contiguous_segments
from robot_sf.common.robot_defaults import DEFAULT_ROBOT_RADIUS
from robot_sf.robot.differential_drive import DifferentialDriveSettings

# ---------------------------------------------------------------------------
# Color / style spec: corpus-wide role palette (author ruling), NOT a
# per-episode Okabe-Ito palette. Robot color is fixed regardless of which
# panel/episode it is in, matching the constants trace_scene_figure.py
# already uses for the other trace-comparison figures in this corpus; the
# focal pedestrian is likewise a fixed, single color across panels. Episode
# identity (A vs B) is instead carried entirely by a redundant channel this
# module already draws for its own legend: MARKER_A/MARKER_B (filled circle
# vs open square) plus linestyle (solid vs dashed, see the render call
# sites). Reusing tsf.INK/tsf.ORANGE directly (rather than re-declaring the
# hex values here) keeps the two modules' role-color definitions from
# drifting apart.
# ---------------------------------------------------------------------------

COLOR_A = tsf.INK  # robot, both episode panels (was episode-keyed Okabe-Ito blue)
COLOR_B = tsf.INK  # robot, both episode panels (was episode-keyed Okabe-Ito vermilion)
COLOR_COMMON_PREFIX = "#777777"
COLOR_PED_CONTEXT = "#BBBBBB"
COLOR_PED_FOCAL_OUTLINE = tsf.ORANGE  # was a fixed near-black outline
COLOR_COLLISION = "#000000"

MARKER_A = "o"  # filled circle
MARKER_B = "s"  # open square

#: Shared translucent-white backing box for direct labels, so text stays legible where it
#: crosses a trajectory line in this narrow-corridor layout (figure_qa's line-overlap lint
#: still flags the geometric bbox intersection; this is a human-legibility mitigation, not
#: a fix for that lint signal -- see the "gap to full spec" note in the module docstring).
_LABEL_BBOX = {"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.5}

#: --------------------------------------------------------------------------------------
#: Per-layout panel styling. "screen" reproduces the original prototype values exactly.
#: "print" designs at FINAL size: the figure is PRINT_FIG_WIDTH_IN wide and vendored at
#: \\textwidth with no LaTeX rescaling, so matplotlib pt == on-page pt. Print constraints (author
#: feedback 2026-07-16): minimum rendered font 7 pt; 8 pt for ticks/annotations; 9-10 pt
#: panel titles; declutter -- drop the per-step "braking" marker+label and the dimmed
#: context-pedestrian id labels (their roles move to the legend), keep collision marker,
#: min-clearance point+value (compact "x.xx m" text), focal-pedestrian highlight, and
#: start/end glyphs.
#: --------------------------------------------------------------------------------------
_SCREEN_PANEL_STYLE: dict[str, Any] = {
    "title_fs": 10.5,
    "tick_fs": 7.5,
    "axis_fs": 8.0,
    "annot_fs": 7.0,  # focal label, collision/near-miss labels
    "small_fs": 6.5,  # pivot + clearance labels, non-completion note
    "ped_fs": 6.5,  # context-pedestrian id labels
    "brake_fs": 6.0,
    "show_braking": True,
    "show_ped_labels": True,
    "clearance_compact": False,
    "outcome_label_in_legend": False,
    #: radial search distances for _place_clear_label -- the original defaults
    "label_radii": (52.0, 68.0, 86.0, 106.0, 128.0),
}
_PRINT_PANEL_STYLE: dict[str, Any] = {
    "title_fs": 9.0,
    "tick_fs": 8.0,
    "axis_fs": 8.0,
    "annot_fs": 8.0,
    "small_fs": 8.0,
    "ped_fs": 7.0,  # unused while show_ped_labels is False; floor kept >= 7 pt anyway
    "brake_fs": 7.0,  # unused while show_braking is False
    "show_braking": False,
    "show_ped_labels": False,
    "clearance_compact": True,
    #: at ~2.6 in panel width the congestion around the doorway leaves no clear spot
    #: near the outcome marker: the radial search parks the "collision" text next to
    #: the (unrelated) trace-start diamond, which misattributes it. The X marker stays
    #: in-panel; its name moves to the legend instead.
    "outcome_label_in_legend": True,
    #: closer-first radial search: at ~2.6 in panel width the screen default's 52 pt
    #: minimum offset is ~28% of the panel -- labels drift too far from their anchors
    #: to read as attached (no leader lines are drawn; see _place_clear_label's
    #: docstring for why leaders are off).
    "label_radii": (16.0, 26.0, 38.0, 52.0, 68.0),
}

#: Print figure width in inches: included at \\textwidth with NO rescaling, so this is the
#: exact on-page size and every pt above is the exact on-page pt. 5.906 in is the true
#: \\textwidth measured from the build log (NOT 6.3 in, an earlier stale estimate).
PRINT_FIG_WIDTH_IN: float = 5.906

#: --------------------------------------------------------------------------------------
#: D(t) normalized joint-state divergence -- physical scales.
#: The divergence detector uses physically meaningful repository scales rather than
#: values tuned to one trace pair:
#:   - pose separation   -> DEFAULT_ROBOT_RADIUS (robot_sf/common/robot_defaults.py)
#:   - commanded v        -> DifferentialDriveSettings.max_linear_speed (robot/differential_drive.py)
#:   - commanded omega     -> DifferentialDriveSettings.max_angular_speed (robot/differential_drive.py)
#:   - clearance-to-focal-ped -> DEFAULT_CLEARANCE_THRESHOLD_M (analysis_workbench/trace_failure_predicates.py)
#: --------------------------------------------------------------------------------------
_ROBOT_DEFAULTS = DifferentialDriveSettings()

POSE_SCALE_M: float = DEFAULT_ROBOT_RADIUS
NOMINAL_LINEAR_SPEED_MPS: float = _ROBOT_DEFAULTS.max_linear_speed
NOMINAL_ANGULAR_SPEED_RAD_S: float = _ROBOT_DEFAULTS.max_angular_speed
CLEARANCE_SCALE_M: float = DEFAULT_CLEARANCE_THRESHOLD_M

#: D(t) component weights. Equal weighting (w_k=1 for all four terms) -- the simplest,
#: least hand-tuned choice; not adjusted to force any particular t_d. See
#: ``compute_joint_state_divergence`` docstring for the per-component breakdown that makes
#: this choice auditable rather than opaque.
DIVERGENCE_WEIGHTS: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)

#: Persistence hold duration: design spec's own range is "0.3-0.5 s"; 0.4 s is the midpoint,
#: not tuned to hit a particular t_d for this pair.
DIVERGENCE_HOLD_S: float = 0.4

#: Threshold *margin* over the empirically observed pre-divergence noise floor of D(t)
#: (see ``find_persistence_onset``): the two planners' commanded turn-rate already differs
#: at t=0 (different control laws for the same "turn toward goal" behavior, not evidence of
#: an interaction-driven divergence), so a literal small threshold (e.g. the design spec's
#: own D(t) *value* is left unspecified -- only the 0.3-0.5 s *hold duration* is given) would
#: fire trivially at step 0. Instead the threshold is derived from the trace itself: 1.25x
#: the maximum D(t) observed over the first second (the "control-law-only" floor), so any
#: crossing is provably above ordinary cross-planner baseline noise, not hand-picked to land
#: on a particular timestamp.
DIVERGENCE_THRESHOLD_MARGIN: float = 1.25
DIVERGENCE_FLOOR_WINDOW_S: float = 1.0

#: Backward search window for the separator (design spec: "search approximately one second
#: backward").
SEPARATOR_SEARCH_BACK_S: float = 1.0

#: "Materially different command" jump thresholds (backward-search tier 2): a *jump*
#: (step-to-step change), not a raw cross-episode difference -- see
#: ``find_separator`` docstring for why raw differences are the wrong test here (the two
#: planners' commands differ from t=0 by construction; only a discrete jump is evidence of
#: a new decision). omega threshold reused from the first-cut script unchanged; the v
#: threshold is new, chosen the same way (50% of NOMINAL_ANGULAR_SPEED_RAD_S vs.
#: 15% of NOMINAL_LINEAR_SPEED_MPS -- see module-level note in ``find_separator``).
COMMAND_DIFF_OMEGA_THRESHOLD_RAD_S: float = 0.5
COMMAND_DIFF_LINEAR_THRESHOLD_MPS: float = 0.15 * NOMINAL_LINEAR_SPEED_MPS


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


def select_focal_pedestrian(episode_a_bundle: Path, ep_b: EpisodeTrace) -> dict[str, Any]:
    """Lock ONE focal pedestrian id, shared by both panels (design spec: "Lock the focal
    pedestrian at conflict entry -- never switch to whichever is currently nearest").

    Fixes the first-cut bug: the previous script set ``focal_ped_id =
    nearest_pedestrian_id[step=0]``, i.e. whichever pedestrian happened to be closest at
    the very first trace step. In this exemplar pair that pedestrian (p3) starts near the
    robot's spawn point but is never actually close (min separation ~4.2 m, well above both
    the near-miss and collision thresholds) before it teleports away to join the far
    pedestrian cluster the robot actually interacts with (respawn-at-goal artifact, see
    ``_draw_panel``'s teleport-segmenting note) -- so the "focal" label rendered on a
    trajectory segment with no real interaction, while the clearance line (correctly)
    pointed at whichever pedestrian the *global closest approach* used instead. Hence:
    "focal p3 renders as a floating segment disconnected from the interaction."

    Fix: reuse ``robot_sf.benchmark.trace_scene_figure._focal_pedestrian_id`` -- the
    shipped, tested selection rule already used by the single-episode paper-figure
    renderer: the pedestrian nearest the robot at the episode's global-minimum-distance
    step (``metadata.summary.global_min_distance_step``). We reuse episode A (the
    "success" / reference trajectory both episodes share an identical start with) as the
    anchor so both panels label the SAME pedestrian id, then verify that id is actually
    present -- and a genuine, non-trivial close approach -- in episode B too, so the
    labelled focal pedestrian is provably the one every clearance line in the figure
    measures to.

    Returns:
        Dict with ``ped_id`` (int), ``source`` (how it was picked), and a
        ``verification`` sub-dict comparing the locked pedestrian's closest approach in B
        against B's own true (any-pedestrian) global closest approach, so a bad lock would
        be visible in the report rather than silently accepted.
    """
    ep_a_tsf = tsf.load_episode(episode_a_bundle)
    focal_str = tsf._focal_pedestrian_id(ep_a_tsf)
    if focal_str is None:
        raise RuntimeError(
            f"trace_scene_figure._focal_pedestrian_id found no pedestrian for {episode_a_bundle}"
        )
    focal_id = int(focal_str)
    if focal_id not in ep_b.ped_ids:
        raise RuntimeError(
            f"focal pedestrian {focal_id} (locked from episode A's closest approach) is not "
            f"present in episode B's pedestrian ids {ep_b.ped_ids} -- cannot lock a shared "
            "focal pedestrian across this pair; this would be a genuine data limitation, not "
            "papered over."
        )
    dist_b_to_focal = clearance_to_ped(ep_b, focal_id)
    b_locked_min_step = int(np.argmin(dist_b_to_focal))
    b_true_min_step = int(np.nanargmin(ep_b.metrics["clearance_m"]))
    b_true_min_ped = ep_b.metrics["nearest_pedestrian_id"][b_true_min_step]
    return {
        "ped_id": focal_id,
        "source": "trace_scene_figure._focal_pedestrian_id(episode_a) -- pedestrian nearest "
        "the robot at episode A's metadata.summary.global_min_distance_step",
        "verification": {
            "episode_b_locked_focal_min_distance_m": float(dist_b_to_focal[b_locked_min_step]),
            "episode_b_locked_focal_min_step": b_locked_min_step,
            "episode_b_true_global_min_distance_m": float(
                ep_b.metrics["clearance_m"][b_true_min_step]
            ),
            "episode_b_true_global_min_ped_id": (
                int(b_true_min_ped) if not math.isnan(b_true_min_ped) else None
            ),
            "note": (
                "the locked focal pedestrian's own closest approach in B is compared "
                "against B's true (any-pedestrian) global closest approach; a large gap "
                "here would mean the locked pedestrian is NOT part of B's real interaction "
                "and the lock should be treated as suspect"
            ),
        },
    }


def clearance_to_ped(ep: EpisodeTrace, ped_id: int) -> np.ndarray:
    """Per-step robot-to-``ped_id`` center-to-center distance (not "nearest of any
    pedestrian" -- a fixed single pedestrian's distance, used to lock the clearance line to
    the same pedestrian the figure labels as focal).

    Returns:
        ``(T,)`` array of distances in meters.
    """
    col = ep.ped_ids.index(ped_id)
    return np.linalg.norm(ep.robot_xy - ep.ped_xy[:, col, :], axis=1)


def compute_joint_state_divergence(
    ep_a: EpisodeTrace,
    ep_b: EpisodeTrace,
    focal_ped_id: int,
    *,
    scales: tuple[float, float, float, float] = (
        POSE_SCALE_M,
        NOMINAL_LINEAR_SPEED_MPS,
        NOMINAL_ANGULAR_SPEED_RAD_S,
        CLEARANCE_SCALE_M,
    ),
    weights: tuple[float, float, float, float] = DIVERGENCE_WEIGHTS,
) -> dict[str, Any]:
    """Design spec's ``D(t)`` normalized joint-state divergence (replaces the first cut's
    threshold-on-robot-robot-separation): a physically-scaled combination of robot pose,
    commanded linear/angular velocity, and clearance-to-the-locked-focal-pedestrian.

        D(t) = sqrt( sum_k w_k * ((z_k^A(t) - z_k^B(t)) / s_k)^2 )

    with z_k in {robot pose separation, commanded v, commanded omega, clearance-to-focal-ped}
    and s_k the matching physical scale (module-level ``POSE_SCALE_M`` /
    ``NOMINAL_LINEAR_SPEED_MPS`` / ``NOMINAL_ANGULAR_SPEED_RAD_S`` / ``CLEARANCE_SCALE_M``).
    Equal weights (``DIVERGENCE_WEIGHTS``) -- not tuned per-component.

    Note on scope: this trace schema (issue-4891-exemplar-trace.v1) carries no per-step
    categorical "planner mode" field (only ``planner.selected_action`` {linear_velocity,
    angular_velocity} and, for social_force, an auxiliary ``ammv.pedestrian_force_vectors``
    array) -- so the design spec's "planner-mode mismatch" ``z_k`` term is not available for
    this pair and is honestly omitted (also see ``find_separator`` tier 1, which reports
    this same limitation explicitly rather than fabricating a mode signal).

    Returns:
        Dict with the ``D`` array plus every per-component array (``pose_sep_m``,
        ``dv_cmd_mps``, ``domega_cmd_rad_s``, ``dclearance_m``) so the eventual pivot
        selection stays explainable (design spec: "preserve their individual values").
    """
    n = min(len(ep_a.robot_xy), len(ep_b.robot_xy))
    pose_sep = np.linalg.norm(ep_a.robot_xy[:n] - ep_b.robot_xy[:n], axis=1)
    dv = np.abs(ep_a.cmd_v[:n] - ep_b.cmd_v[:n])
    domega = np.abs(ep_a.cmd_omega[:n] - ep_b.cmd_omega[:n])
    clear_a = clearance_to_ped(ep_a, focal_ped_id)[:n]
    clear_b = clearance_to_ped(ep_b, focal_ped_id)[:n]
    dclear = np.abs(clear_a - clear_b)

    s_pose, s_v, s_omega, s_clear = scales
    w_pose, w_v, w_omega, w_clear = weights
    d_squared = (
        w_pose * (pose_sep / s_pose) ** 2
        + w_v * (dv / s_v) ** 2
        + w_omega * (domega / s_omega) ** 2
        + w_clear * (dclear / s_clear) ** 2
    )
    return {
        "D": np.sqrt(d_squared),
        "n": n,
        "pose_sep_m": pose_sep,
        "dv_cmd_mps": dv,
        "domega_cmd_rad_s": domega,
        "dclearance_m": dclear,
        "scales": {"pose_m": s_pose, "v_mps": s_v, "omega_rad_s": s_omega, "clearance_m": s_clear},
        "weights": {"pose": w_pose, "v": w_v, "omega": w_omega, "clearance": w_clear},
    }


def find_persistence_onset(
    divergence: dict[str, Any],
    dt: float,
    *,
    hold_s: float = DIVERGENCE_HOLD_S,
    floor_window_s: float = DIVERGENCE_FLOOR_WINDOW_S,
    margin: float = DIVERGENCE_THRESHOLD_MARGIN,
) -> dict[str, Any]:
    """``t_d``: first time ``D(t)`` remains above threshold for ``hold_s`` seconds
    (design spec, step 2: "Define t_d as the first time D(t) remains above a threshold for
    0.3-0.5 s").

    Threshold derivation (data-driven, not hand-picked -- see the module-level
    ``DIVERGENCE_THRESHOLD_MARGIN`` docstring): ``margin`` times the maximum D(t) observed
    over the first ``floor_window_s`` seconds, i.e. the "two different control laws, no
    interaction yet" baseline noise floor for this pair.

    Returns:
        Dict with ``step`` (``None`` if no persistent crossing is found), ``time_s``,
        ``threshold``, ``floor``, ``hold_steps``, and ``floor_window_steps`` -- every input
        to the decision, so it is auditable from the report alone.
    """
    d = divergence["D"]
    n = len(d)
    floor_steps = max(1, round(floor_window_s / dt))
    floor = float(np.max(d[: min(floor_steps, n)]))
    threshold = margin * floor
    hold_steps = max(1, round(hold_s / dt))
    for i in range(n):
        if i + hold_steps <= n and np.all(d[i : i + hold_steps] > threshold):
            return {
                "step": i,
                "time_s": None,  # filled in by the caller (needs ep.time_s)
                "threshold": threshold,
                "floor": floor,
                "hold_steps": hold_steps,
                "floor_window_steps": floor_steps,
            }
    return {
        "step": None,
        "time_s": None,
        "threshold": threshold,
        "floor": floor,
        "hold_steps": hold_steps,
        "floor_window_steps": floor_steps,
    }


def find_separator(  # noqa: PLR0913 - every argument is a distinct input to the tiered search
    ep_a: EpisodeTrace,
    ep_b: EpisodeTrace,
    divergence: dict[str, Any],
    t_d_step: int,
    events_a: dict[str, dict[str, Any]],
    events_b: dict[str, dict[str, Any]],
    *,
    search_back_s: float = SEPARATOR_SEARCH_BACK_S,
    jump_threshold_omega: float = COMMAND_DIFF_OMEGA_THRESHOLD_RAD_S,
    jump_threshold_v: float = COMMAND_DIFF_LINEAR_THRESHOLD_MPS,
) -> dict[str, Any]:
    """Backward search from ``t_d`` for the separator (design spec, step 2): "search
    approximately one second backward for: 1. the first differing mode; 2. the first
    materially different command; 3. braking or yielding onset; 4. otherwise, the largest
    rise in risk."

    Tier 1 (planner mode) is unavailable for this trace schema (see
    ``compute_joint_state_divergence`` docstring) and is reported as such, not skipped
    silently. Tier 2 tests for a *jump* (``|cmd[i] - cmd[i-1]| > threshold``) in either
    agent's own command sequence, not a raw cross-episode difference -- the two planners'
    commands differ from t=0 by construction (different control laws for the same
    "turn/accelerate toward goal" behavior), so a raw-difference test would fire trivially
    at step 0 and explain nothing (this is exactly the failure mode the first cut's
    ``find_first_command_separator`` already worked around for omega; this version applies
    the same jump test to both v and omega and makes it tier-ordered per the design spec).
    Tier 3 checks whether either episode's ``first_braking_event`` critical-interval anchor
    (already computed by ``compute_critical_events``, reused not reinvented) falls inside
    the backward window. Tier 4 falls back to the step with the largest single-step rise in
    ``D(t)`` inside the window -- i.e. "this pair's divergence has no single discrete
    decision point in the lookback window; the closest thing to one is where the aggregate
    signal itself accelerates fastest."

    Returns:
        Dict with ``step``, ``time_s``, ``tier`` (one of ``"command_jump"``,
        ``"braking_onset"``, ``"largest_risk_rise"`` -- ``"mode"`` is never returned since
        it is structurally unavailable), and tier-specific detail fields.
    """
    dt = float(ep_a.time_s[1] - ep_a.time_s[0]) if len(ep_a.time_s) > 1 else 0.1
    back_steps = max(1, round(search_back_s / dt))
    start = max(1, t_d_step - back_steps)
    tier1_note = (
        "tier 1 (first differing planner mode) is unavailable: this trace schema "
        "(issue-4891-exemplar-trace.v1) carries no per-step categorical planner-mode "
        "field, only continuous commanded linear/angular velocity -- a genuine data "
        "limitation, not a detector gap"
    )

    # Tier 2: first materially different command (a discrete jump, in either agent, in
    # either v or omega) within the backward window.
    for i in range(start, t_d_step + 1):
        jump_v_a = abs(ep_a.cmd_v[i] - ep_a.cmd_v[i - 1])
        jump_v_b = abs(ep_b.cmd_v[i] - ep_b.cmd_v[i - 1])
        jump_o_a = abs(ep_a.cmd_omega[i] - ep_a.cmd_omega[i - 1])
        jump_o_b = abs(ep_b.cmd_omega[i] - ep_b.cmd_omega[i - 1])
        v_hit = max(jump_v_a, jump_v_b) > jump_threshold_v
        o_hit = max(jump_o_a, jump_o_b) > jump_threshold_omega
        if v_hit or o_hit:
            return {
                "step": i,
                "time_s": float(ep_a.time_s[i]),
                "tier": "command_jump",
                "tier1_note": tier1_note,
                "search_window_steps": [start, t_d_step],
                "jump_kind": "v" if v_hit else "omega",
                "jump_in": "A" if max(jump_v_a, jump_o_a) >= max(jump_v_b, jump_o_b) else "B",
                "jump_v_mps": float(max(jump_v_a, jump_v_b)),
                "jump_omega_rad_s": float(max(jump_o_a, jump_o_b)),
            }

    # Tier 3: braking/yield onset (either agent's critical_intervals anchor) inside window.
    for label, events in (("A", events_a), ("B", events_b)):
        brake = events.get("first_braking_event")
        if brake is not None and start <= brake["step"] <= t_d_step:
            return {
                "step": brake["step"],
                "time_s": brake["time_s"],
                "tier": "braking_onset",
                "tier1_note": tier1_note,
                "search_window_steps": [start, t_d_step],
                "braking_in": label,
            }

    # Tier 4 fallback: largest single-step rise in D(t) within the window.
    d = divergence["D"]
    window_start = max(1, start)
    deltas = d[window_start : t_d_step + 1] - d[window_start - 1 : t_d_step]
    if len(deltas) == 0:
        best_step = t_d_step
        best_delta = 0.0
    else:
        best_step = window_start + int(np.argmax(deltas))
        best_delta = float(np.max(deltas))
    return {
        "step": best_step,
        "time_s": float(ep_a.time_s[best_step]),
        "tier": "largest_risk_rise",
        "tier1_note": tier1_note,
        "search_window_steps": [start, t_d_step],
        "tier2_note": "no discrete command jump (v or omega) found in the backward window",
        "tier3_note": "no braking/yield onset (critical_intervals.first_braking_event) "
        "found in the backward window",
        "delta_d": best_delta,
    }


def closest_approach(ep: EpisodeTrace) -> dict[str, Any]:
    """Global closest robot-pedestrian approach within the (possibly truncated) episode,
    to WHICHEVER pedestrian is nearest at that step (may differ from the locked focal
    pedestrian -- kept for the report's cross-check, see ``select_focal_pedestrian``, not
    used to draw the figure's clearance line).

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


def closest_approach_to_focal_ped(ep: EpisodeTrace, focal_ped_id: int) -> dict[str, Any]:
    """Closest robot approach to the LOCKED focal pedestrian specifically (not "nearest of
    any pedestrian" -- see ``closest_approach``). This is what the figure's clearance line
    and label are drawn from, so the labelled focal pedestrian is provably the one the
    clearance line measures to (the task's item-1 verification requirement).

    Returns:
        Dict with ``step``, ``time_s``, ``distance_m``, ``ped_id`` (== ``focal_ped_id``),
        ``robot_xy``, ``ped_xy``.
    """
    dist = clearance_to_ped(ep, focal_ped_id)
    step = int(np.argmin(dist))
    col = ep.ped_ids.index(focal_ped_id)
    return {
        "step": step,
        "time_s": float(ep.time_s[step]),
        "distance_m": float(dist[step]),
        "ped_id": focal_ped_id,
        "robot_xy": ep.robot_xy[step].tolist(),
        "ped_xy": ep.ped_xy[step, col].tolist(),
    }


def compute_delta_gutter(
    ep_a: EpisodeTrace,
    ep_b: EpisodeTrace,
    pivot_step: int,
    focal_ped_id: int,
    events_a: dict[str, dict[str, Any]],
    events_b: dict[str, dict[str, Any]],
    *,
    horizon_s: float = 2.0,
) -> dict[str, Any]:
    """The hinge figure's central delta gutter (design spec: "A narrow central delta gutter
    containing only: Delta t_brake, Delta v_cmd at the pivot, minimum clearance over the
    following horizon, first differing planner mode").

    All four quantities are computed directly from the traces at the pivot found by
    ``find_separator`` -- none are hand-tuned. ``first differing planner mode`` is reported
    as unavailable (same data-schema limitation as ``find_separator`` tier 1) rather than
    fabricated.

    Returns:
        Dict with ``dt_brake_s`` (B's first-braking time minus A's, ``None`` if either is
        unavailable), ``dv_cmd_at_pivot_mps`` / ``domega_cmd_at_pivot_rad_s`` (A minus B at
        ``pivot_step``), ``min_clearance_horizon_m`` per episode (min clearance-to-focal-ped
        over the ``horizon_s`` seconds following the pivot), and ``first_differing_mode``
        (``None`` with a ``reason``).
    """
    dt = float(ep_a.time_s[1] - ep_a.time_s[0]) if len(ep_a.time_s) > 1 else 0.1
    horizon_steps = max(1, round(horizon_s / dt))

    brake_a = events_a.get("first_braking_event")
    brake_b = events_b.get("first_braking_event")
    dt_brake = (
        brake_b["time_s"] - brake_a["time_s"]
        if brake_a is not None and brake_b is not None
        else None
    )

    dv_at_pivot = float(ep_a.cmd_v[pivot_step] - ep_b.cmd_v[pivot_step])
    domega_at_pivot = float(ep_a.cmd_omega[pivot_step] - ep_b.cmd_omega[pivot_step])

    def _min_clearance_over_horizon(ep: EpisodeTrace) -> dict[str, Any]:
        end = min(len(ep.robot_xy), pivot_step + horizon_steps + 1)
        window = clearance_to_ped(ep, focal_ped_id)[pivot_step:end]
        idx = pivot_step + int(np.argmin(window))
        return {"distance_m": float(np.min(window)), "step": idx, "time_s": float(ep.time_s[idx])}

    return {
        "dt_brake_s": dt_brake,
        "dt_brake_detail": {
            "episode_a_first_braking_time_s": brake_a["time_s"] if brake_a else None,
            "episode_b_first_braking_time_s": brake_b["time_s"] if brake_b else None,
        },
        "dv_cmd_at_pivot_mps": dv_at_pivot,
        "domega_cmd_at_pivot_rad_s": domega_at_pivot,
        "min_clearance_horizon_m": {
            "episode_a": _min_clearance_over_horizon(ep_a),
            "episode_b": _min_clearance_over_horizon(ep_b),
        },
        "horizon_s": horizon_s,
        "first_differing_mode": None,
        "first_differing_mode_reason": (
            "unavailable: this trace schema carries no per-step categorical planner-mode "
            "field (see find_separator tier 1 note)"
        ),
    }


def count_near_miss_steps(ep: EpisodeTrace) -> int:
    """Steps whose minimum surface clearance to any pedestrian falls in
    ``[0, NEAR_MISS_DIST)`` -- the exact benchmark near-miss definition
    (``robot_sf.benchmark.metrics._compute_robot_ped_distance_summary``:
    ``near_miss_definition = '0 <= min_clearance_m < near_miss_distance_m'`` with
    surface clearance = center distance - (robot_radius + ped_radius), thresholds from
    ``robot_sf.benchmark.constants.NEAR_MISS_DIST`` /
    ``collision_definition_inventory.DEFAULT_PED_RADIUS`` /
    ``robot_defaults.DEFAULT_ROBOT_RADIUS``). Verified to reproduce the doorway
    re-export episodes.jsonl ``metrics.near_misses`` values for seeds 113 (13) and
    114 (78) when the full trace is loaded.

    Returns:
        Near-miss step count over the loaded (possibly truncated) episode window.
    """
    surface = ep.metrics["clearance_m"] - (DEFAULT_ROBOT_RADIUS + DEFAULT_PED_RADIUS)
    return int(np.count_nonzero((surface >= 0.0) & (surface < NEAR_MISS_DIST)))


def compute_contrast_gutter(
    ep_a: EpisodeTrace,
    ep_b: EpisodeTrace,
    focal_closest_a: dict[str, Any],
    focal_closest_b: dict[str, Any],
    events_a: dict[str, dict[str, Any]],
    events_b: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Central A|B gutter for the matched seed-pair CONTRAST rendering mode (used when
    the two episodes do not share a common start, so pivot-anchored deltas like
    "Delta v_cmd at the pivot" would only measure start-state differences under
    different spawn draws -- semantically weak). Every quantity is per-episode
    well-defined regardless of differing starts, and every number is derived from the
    two loaded traces (or their bundle metadata) -- nothing is invented:

    - minimum clearance to the locked focal pedestrian (center-to-center, the same
      quantity the panels' clearance lines draw),
    - near-miss step counts (benchmark definition, see ``count_near_miss_steps``),
    - steps to termination (``metadata.summary.step_count``, the full episode length),
    - first-braking time per episode (``critical_intervals.first_braking_event``,
      a per-episode deceleration-threshold event whose definition survives different
      starts -- kept as two absolute times rather than a delta).

    Returns:
        Contrast-gutter dict (``mode`` == ``"contrast"``).
    """
    brake_a = events_a.get("first_braking_event")
    brake_b = events_b.get("first_braking_event")
    return {
        "mode": "contrast",
        "min_clearance_focal_m": {
            "episode_a": focal_closest_a["distance_m"],
            "episode_b": focal_closest_b["distance_m"],
        },
        "near_miss_steps": {
            "episode_a": count_near_miss_steps(ep_a),
            "episode_b": count_near_miss_steps(ep_b),
            "definition": (
                "steps with 0 <= min surface clearance < "
                f"{NEAR_MISS_DIST:g} m (robot_radius {DEFAULT_ROBOT_RADIUS:g} m + "
                f"ped_radius {DEFAULT_PED_RADIUS:g} m subtracted from center distance); "
                "matches the benchmark metrics.near_misses definition"
            ),
            "window": "loaded episode window (full trace when no story-window truncation)",
        },
        "steps_to_termination": {
            "episode_a": ep_a.metadata["summary"]["step_count"],
            "episode_b": ep_b.metadata["summary"]["step_count"],
        },
        "first_braking_time_s": {
            "episode_a": brake_a["time_s"] if brake_a else None,
            "episode_b": brake_b["time_s"] if brake_b else None,
        },
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


def _text_overlaps_lines_or_markers(
    ax: plt.Axes, text_artist: Any, renderer: Any, tolerance: float = 2.0
) -> bool:
    """Item 4 / figure_qa gate: check ONE text artist against the exact criteria
    ``figure_qa.lint_figure`` uses (its own private ``_line_bbox_overlaps_text`` /
    ``_point_in_bbox`` / ``_bboxes_overlap`` + ``_is_meaningful_text`` helpers, reused
    verbatim rather than reimplemented), so label placement is judged by the same rules
    the QA gate will apply -- not a separate geometric heuristic that might disagree
    with it. Covers all three of the QA gate's text-adjacent defect types this script
    can trigger: ``text_line_overlap``, ``text_marker_overlap``, and
    ``text_text_overlap`` (this scenario's 6-pedestrian doorway congestion produces
    enough simultaneous labels -- braking/collision/focal/context-ped/clearance -- that
    label-vs-label collisions are as common as label-vs-geometry ones, unlike the
    sparser 4-pedestrian reference pair this closed-loop placer was first written for).

    Returns:
        True if the text artist's rendered bounding box overlaps any visible Line2D,
        scatter-marker offset, or other meaningful text artist already drawn in ``ax``.
    """
    text_bbox = text_artist.get_window_extent(renderer)
    for child in ax.get_children():
        if isinstance(child, Line2D) and child.get_visible():
            if figure_qa._line_bbox_overlaps_text(child, text_bbox, ax, renderer, tolerance):
                return True
        elif isinstance(child, PathCollection) and child.get_visible():
            offsets = child.get_offsets()
            if len(offsets) == 0:
                continue
            transform = ax.transData
            for point in offsets:
                px, py = transform.transform((float(point[0]), float(point[1])))
                if figure_qa._point_in_bbox(px, py, text_bbox, tolerance):
                    return True
        elif child is not text_artist and figure_qa._is_meaningful_text(child):
            other_bbox = child.get_window_extent(renderer)
            if figure_qa._bboxes_overlap(text_bbox, other_bbox, tolerance):
                return True
    return False


def _place_clear_label(  # noqa: PLR0913 - every argument is a distinct annotate() control
    ax: plt.Axes,
    anchor_xy: tuple[float, float],
    text: str,
    *,
    color: str,
    fontsize: float,
    weight: str = "normal",
    radii_points: tuple[float, ...] = (52.0, 68.0, 86.0, 106.0, 128.0),
    n_directions: int = 16,
    draw_leader: bool = True,
) -> None:
    """Place an annotation near ``anchor_xy``, trying ``n_directions`` candidate radial
    offsets at each of ``radii_points`` (nearest radius first) and keeping the first whose
    rendered bbox does not overlap any line/marker already drawn in ``ax`` (checked with
    ``_text_overlaps_lines_or_markers`` -- the exact figure_qa criterion, not a proxy for
    it). Falls back to the very first candidate tried if every direction at every radius
    overlaps something (better than silently placing on top of a line without trying).

    This closed-loop, render-and-check placement replaces several earlier heuristic
    attempts in this module's history (a fixed perpendicular offset; an offset chosen by
    distance-to-trajectory-vertices only; a single-radius direction search) -- the first
    two cleared the specific line they were designed around but not every artist actually
    drawn in the panel. The single-radius search then failed outright: at radius 52 pt
    every one of 16 directions overlapped something, so the loop always fell back to its
    first, overlapping, candidate, and figure_qa.lint_figure kept reporting the *same*
    defect at the *same* pixel location no matter how the direction heuristic upstream of
    it changed -- which is what exposed the real cause: ``Annotation.get_window_extent()``
    (what both this function and figure_qa's own checker call) includes the CONNECTOR ARROW
    when ``arrowprops`` is set, not just the text glyph box. With a leader arrow drawn back
    to an anchor point sitting inside a dense pedestrian cluster, the reported bbox always
    reached back into that clutter regardless of how far the text itself was pushed away --
    so ``draw_leader=False`` is required for anchors in cluttered regions (the caller
    already draws a separate, deliberate dotted clearance line for that visual link; the
    text doesn't need its own second connector).
    """
    fig = ax.figure
    fig.canvas.draw()

    arrowprops = (
        {"arrowstyle": "-", "color": color, "linewidth": 0.5, "shrinkA": 2, "shrinkB": 2}
        if draw_leader
        else None
    )
    fallback_artist = None
    for radius_points in radii_points:
        for k in range(n_directions):
            theta = 2.0 * math.pi * k / n_directions
            dx, dy = math.cos(theta) * radius_points, math.sin(theta) * radius_points
            artist = ax.annotate(
                text,
                anchor_xy,
                fontsize=fontsize,
                color=color,
                weight=weight,
                xytext=(dx, dy),
                textcoords="offset points",
                ha="left" if dx >= 0 else "right",
                va="center",
                zorder=9,
                arrowprops=arrowprops,
                bbox=_LABEL_BBOX,
            )
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            overlap = _text_overlaps_lines_or_markers(ax, artist, renderer)
            if not overlap:
                if fallback_artist is not None:
                    fallback_artist.remove()
                return
            if fallback_artist is None:
                fallback_artist = artist
            else:
                artist.remove()


def _draw_panel(  # noqa: C901, PLR0912, PLR0913, PLR0915 - one-panel figure assembly, mirrors
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
    map_definition: Any | None = None,
    defer_labels: bool = False,
    pivot_label_text: str = "pivot",
    contrast_mode: bool = False,
    style: dict[str, Any] | None = None,
) -> list[tuple[tuple[float, float], str, dict[str, Any]]]:
    """Draw one spatial panel of the hinge figure. Mutates ``ax`` in place.

    ``contrast_mode`` (matched seed-pair contrast framing, for pairs with no shared
    start): the whole robot trajectory is drawn in the episode's own color/linestyle
    (no gray "common prefix" segment -- there is none) and no pivot ring or pivot label
    is drawn (``divergence_step`` / ``common_prefix_end`` / ``pivot_label_text`` are
    ignored). Everything else (pedestrians, braking/clearance/outcome markers,
    start/end glyphs) is unchanged. The default (hinge) mode is untouched for true
    shared-prefix pairs.

    ``style`` selects a per-layout font/declutter table (``_SCREEN_PANEL_STYLE``
    default, reproducing the original prototype values exactly, or
    ``_PRINT_PANEL_STYLE`` for the design-at-final-size print layout).

    When ``defer_labels`` is True, the two dynamically-placed labels (pivot ring,
    clearance) are NOT placed here; instead their ``(anchor_xy, text, style_kwargs)``
    specs are collected and returned, so the caller can place them AFTER
    ``fig.tight_layout()`` runs (tight_layout moves/rescales the axes, which invalidates
    any text-vs-line overlap check performed before it -- see ``render_hinge_figure`` for
    why this two-pass placement is necessary: an earlier single-pass version placed labels
    that measurably cleared every line/marker at draw time, yet figure_qa.lint_figure still
    reported the same overlaps because tight_layout ran afterward and shifted geometry
    under the already-fixed label positions).

    Returns:
        Empty list unless ``defer_labels`` is True, in which case up to two
        ``(anchor_xy, text, style_kwargs)`` tuples for the caller to place later.
    """
    style = style or _SCREEN_PANEL_STYLE
    xmin, xmax, ymin, ymax = bounds
    pending_labels: list[tuple[tuple[float, float], str, dict[str, Any]]] = []

    # -- map obstacles (reused verbatim from the shipped single-episode paper-figure
    # renderer -- item 4: ground the hinge panels in the real corridor geometry instead of
    # a blank background, without reimplementing obstacle drawing). Drawn first / lowest
    # zorder so trajectories and markers stay on top.
    if map_definition is not None:
        tsf._draw_obstacles(ax, map_definition, ((xmin, xmax), (ymin, ymax)))

    # -- context (non-focal) pedestrians, dimmed, stable IDs --------------------------
    # Pedestrians respawn under the same id at their goal (a ~25 m same-step position
    # jump); break the polyline at those jumps instead of drawing a spurious teleport
    # line, reusing the repo's own established fix for this exact artifact
    # (robot_sf.benchmark.trace_scene_figure._contiguous_segments / _TELEPORT_STEP_M).
    # Context/focal pedestrian id labels use the same closed-loop figure_qa-driven
    # placement as the pivot/clearance labels below (_place_clear_label): a fixed
    # offset (the first-cut approach, still applied here for the reference
    # head-on-corridor pair's sparser 4-pedestrian layout) collides routinely in a
    # denser scene -- this pair's 6-pedestrian doorway congestion is exactly such a
    # case (figure_qa.lint_figure caught p4/p5/focal-p2 line overlaps on the first
    # render of this scenario).
    for col, pid in enumerate(ep.ped_ids):
        xy = ep.ped_xy[:, col, :]
        if pid == focal_ped_id:
            continue
        for seg_x, seg_y in _contiguous_segments(list(xy[:, 0]), list(xy[:, 1])):
            ax.plot(seg_x, seg_y, color=COLOR_PED_CONTEXT, linewidth=0.6, alpha=0.55, zorder=1)
        if not style["show_ped_labels"]:
            continue  # print declutter: ids move to the legend's "other pedestrians" entry
        ped_label_spec = (
            (float(xy[0, 0]), float(xy[0, 1])),
            f"p{pid}",
            {
                "color": COLOR_PED_CONTEXT,
                "fontsize": style["ped_fs"],
                "draw_leader": False,
                "radii_points": (18.0, 30.0, 44.0, 60.0),
            },
        )
        if defer_labels:
            pending_labels.append(ped_label_spec)
        else:
            _place_clear_label(ax, *ped_label_spec[:2], **ped_label_spec[2])

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
    focal_label_spec = (
        (float(focal_xy[0, 0]), float(focal_xy[0, 1])),
        f"focal p{focal_ped_id}",
        {
            "color": COLOR_PED_FOCAL_OUTLINE,
            "fontsize": style["annot_fs"],
            "weight": "bold",
            "draw_leader": False,
            "radii_points": style["label_radii"],
        },
    )
    if defer_labels:
        pending_labels.append(focal_label_spec)
    else:
        _place_clear_label(ax, *focal_label_spec[:2], **focal_label_spec[2])

    # -- robot trajectory: common prefix gray, post-pivot in planner style ------------
    xy = ep.robot_xy
    if contrast_mode:
        # Matched seed-pair contrast framing: no shared start, so no gray shared-prefix
        # segment -- the whole trajectory is this episode's own.
        ax.plot(
            xy[:, 0],
            xy[:, 1],
            color=color,
            linewidth=1.8,
            linestyle=linestyle,
            zorder=4,
            solid_capstyle="round",
        )
    else:
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

    # -- pivot ring (design spec: "a conspicuous pivot ring at the first persistent action
    # difference" -- the separator found by find_separator, not the raw geometric
    # divergence point). Label placement uses the same closed-loop figure_qa-driven search
    # as the clearance label (_place_clear_label). Skipped entirely in contrast mode
    # (no shared prefix -> no pivot to ring; the start diamond glyph marks the trace
    # start instead, named "trace start" in the contrast legend).
    if not contrast_mode and divergence_step is not None and divergence_step < len(xy):
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
        pivot_label_spec = (
            (px, py),
            pivot_label_text,
            {
                "color": "#111111",
                "fontsize": style["small_fs"],
                "draw_leader": False,
                "radii_points": style["label_radii"],
            },
        )
        if defer_labels:
            pending_labels.append(pivot_label_spec)
        else:
            _place_clear_label(ax, *pivot_label_spec[:2], **pivot_label_spec[2])

    # -- critical-interval markers (closest_approach / first_braking_event) ------------
    for anchor, info in critical_events.items():
        step = info["step"]
        if step >= len(xy):
            continue
        px, py = xy[step]
        if anchor == "closest_approach":
            continue  # drawn explicitly below with the labelled clearance line
        if anchor == "first_braking_event":
            if not style["show_braking"]:
                continue  # print declutter: braking marker+label dropped (author feedback)
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
            braking_label_spec = (
                (float(px), float(py)),
                "braking",
                {
                    "color": "#111111",
                    "fontsize": style["brake_fs"],
                    "draw_leader": False,
                    "radii_points": style["label_radii"],
                },
            )
            if defer_labels:
                pending_labels.append(braking_label_spec)
            else:
                _place_clear_label(ax, *braking_label_spec[:2], **braking_label_spec[2])

    # -- closest-approach line, labelled with clearance ----------------------------------
    # Label placement uses the closed-loop figure_qa-driven search (_place_clear_label,
    # see its docstring for the two heuristic attempts this replaced): the first cut's
    # fixed diagonal offset put both clearance labels directly on their own dotted line
    # (figure_qa.lint_figure text_line_overlap / text_marker_overlap).
    if closest["ped_xy"] is not None:
        rx, ry = closest["robot_xy"]
        pxg, pyg = closest["ped_xy"]
        ax.plot([rx, pxg], [ry, pyg], color=color, linewidth=0.8, linestyle=":", zorder=5)
        clearance_text = (
            f"{closest['distance_m']:.2f} m"
            if style["clearance_compact"]
            else f"clearance {closest['distance_m']:.2f} m\n@t={closest['time_s']:.1f}s"
        )
        clearance_label_spec = (
            (pxg, pyg),
            clearance_text,
            {
                "color": color,
                "fontsize": style["small_fs"],
                "draw_leader": False,
                "radii_points": style["label_radii"],
            },
        )
        if defer_labels:
            pending_labels.append(clearance_label_spec)
        else:
            _place_clear_label(ax, *clearance_label_spec[:2], **clearance_label_spec[2])

    # -- outcome marker: collision (x) / near-miss (triangle) --------------------------
    if collision_or_near_miss_step is not None and collision_or_near_miss_step < len(xy):
        px, py = xy[collision_or_near_miss_step]
        outcome_label_spec: tuple[Any, ...] | None = None
        if outcome_kind == "collision":
            ax.scatter([px], [py], marker="x", s=90, color=COLOR_COLLISION, linewidth=2.2, zorder=7)
            outcome_label_spec = (
                (float(px), float(py)),
                "collision",
                {
                    "color": COLOR_COLLISION,
                    "fontsize": style["annot_fs"],
                    "weight": "bold",
                    "draw_leader": False,
                    "radii_points": style["label_radii"],
                },
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
            outcome_label_spec = (
                (float(px), float(py)),
                "near miss",
                {
                    "color": COLOR_COLLISION,
                    "fontsize": style["annot_fs"],
                    "weight": "bold",
                    "draw_leader": False,
                    "radii_points": style["label_radii"],
                },
            )
        # Print declutter: the outcome marker's name moves to the legend (the doorway
        # congestion leaves no clear spot near the marker itself -- the radial search
        # would park the text next to the unrelated trace-start diamond).
        if outcome_label_spec is not None and not style["outcome_label_in_legend"]:
            if defer_labels:
                pending_labels.append(outcome_label_spec)
            else:
                _place_clear_label(ax, *outcome_label_spec[:2], **outcome_label_spec[2])

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
        non_completion_label_spec = (
            (float(xy[-1, 0]), float(xy[-1, 1])),
            "episode continues past this crop\n(non-completion / timeout, not shown)",
            {
                "color": color,
                "fontsize": style["small_fs"],
                "draw_leader": False,
                "radii_points": style["label_radii"],
            },
        )
        if defer_labels:
            pending_labels.append(non_completion_label_spec)
        else:
            _place_clear_label(ax, *non_completion_label_spec[:2], **non_completion_label_spec[2])

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(label, fontsize=style["title_fs"], loc="left", color=color, weight="bold")
    ax.tick_params(labelsize=style["tick_fs"])
    ax.set_xlabel("x (m)", fontsize=style["axis_fs"])
    ax.grid(alpha=0.15, linewidth=0.5)
    return pending_labels


def _format_gutter_lines(gutter: dict[str, Any]) -> list[str]:
    """Render the delta-gutter dict (``compute_delta_gutter``) as short text lines.

    Returns:
        Ordered list of one-line strings for the gutter panel.
    """
    lines = ["A | Δ | B", ""]
    if gutter["dt_brake_s"] is not None:
        lines.append(f"Δt_brake\n{gutter['dt_brake_s']:+.2f} s")
    else:
        lines.append("Δt_brake\nn/a (one side\nnever brakes)")
    lines.append("")
    lines.append(f"Δv_cmd @pivot\n{gutter['dv_cmd_at_pivot_mps']:+.2f} m/s")
    lines.append("")
    lines.append(f"Δω_cmd @pivot\n{gutter['domega_cmd_at_pivot_rad_s']:+.2f} rad/s")
    lines.append("")
    min_a = gutter["min_clearance_horizon_m"]["episode_a"]["distance_m"]
    min_b = gutter["min_clearance_horizon_m"]["episode_b"]["distance_m"]
    lines.append(
        f"min clearance\n(+{gutter['horizon_s']:g}s horizon)\nA {min_a:.2f}m / B {min_b:.2f}m"
    )
    lines.append("")
    lines.append("first differing mode\nn/a (no per-step\nmode field)")
    return lines


def _format_contrast_gutter_lines(gutter: dict[str, Any]) -> list[str]:
    """Render the contrast-gutter dict (``compute_contrast_gutter``) as short text lines.

    Contrast vocabulary only -- no pivot/hinge/common-prefix wording (matched seed-pair
    contrast framing).

    Returns:
        Ordered list of one-line strings for the gutter panel.
    """
    clear_a = gutter["min_clearance_focal_m"]["episode_a"]
    clear_b = gutter["min_clearance_focal_m"]["episode_b"]
    near_a = gutter["near_miss_steps"]["episode_a"]
    near_b = gutter["near_miss_steps"]["episode_b"]
    steps_a = gutter["steps_to_termination"]["episode_a"]
    steps_b = gutter["steps_to_termination"]["episode_b"]
    brake_a = gutter["first_braking_time_s"]["episode_a"]
    brake_b = gutter["first_braking_time_s"]["episode_b"]
    # A/B values are stacked on separate lines (not "A .. / B .." on one line) to keep
    # every gutter line narrow: the one-line variant collided with panel B's y-axis
    # tick labels (a cross-axes overlap figure_qa.lint_figure does not check, caught by
    # visual review of the first contrast render).
    lines = ["A | B", ""]
    lines.append(f"min clearance\n(focal ped)\nA {clear_a:.2f} m\nB {clear_b:.2f} m")
    lines.append("")
    lines.append(f"near-miss steps\nA {near_a} / B {near_b}")
    lines.append("")
    lines.append(f"steps to\ntermination\nA {steps_a} / B {steps_b}")
    lines.append("")
    if brake_a is not None and brake_b is not None:
        lines.append(f"first braking\nA @{brake_a:.1f} s\nB @{brake_b:.1f} s")
    elif brake_a is not None or brake_b is not None:
        only = f"A @{brake_a:.1f} s" if brake_a is not None else f"B @{brake_b:.1f} s"
        lines.append(f"first braking\n{only}\n(other side n/a)")
    else:
        lines.append("first braking\nn/a (neither\nside brakes)")
    return lines


def _draw_delta_gutter(ax: plt.Axes, gutter: dict[str, Any]) -> None:
    """Draw the narrow central gutter panel: the hinge-mode delta gutter (design spec:
    "A narrow central delta gutter containing only: Delta t_brake, Delta v_cmd at the
    pivot, minimum clearance over the following horizon, first differing planner mode")
    or, when ``gutter["mode"] == "contrast"``, the matched seed-pair contrast gutter
    (per-episode quantities that stay meaningful across different starts -- see
    ``compute_contrast_gutter``).
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    if gutter.get("mode") == "contrast":
        lines = _format_contrast_gutter_lines(gutter)
    else:
        lines = _format_gutter_lines(gutter)
    n = len(lines)
    for i, text in enumerate(lines):
        y = 1.0 - (i + 0.5) / n
        weight = "bold" if text and "\n" not in text and "|" in text else "normal"
        ax.text(
            0.5,
            y,
            text,
            fontsize=6.3,
            ha="center",
            va="center",
            weight=weight,
            color=COLOR_COMMON_PREFIX if "n/a" in text else "#222222",
        )


def _draw_contrast_strip(ax: plt.Axes, gutter: dict[str, Any], *, fontsize: float) -> None:
    """Draw the print layout's compact single-row contrast strip (replaces the vertical
    central gutter below the panels): four cells, each a small dimmed header over its A/B
    values -- min clearance to the focal pedestrian, near-miss steps, steps to
    termination, first braking. Same data source as the vertical contrast gutter
    (``compute_contrast_gutter``); only the arrangement changes.
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    clear_a = gutter["min_clearance_focal_m"]["episode_a"]
    clear_b = gutter["min_clearance_focal_m"]["episode_b"]
    near_a = gutter["near_miss_steps"]["episode_a"]
    near_b = gutter["near_miss_steps"]["episode_b"]
    steps_a = gutter["steps_to_termination"]["episode_a"]
    steps_b = gutter["steps_to_termination"]["episode_b"]
    brake_a = gutter["first_braking_time_s"]["episode_a"]
    brake_b = gutter["first_braking_time_s"]["episode_b"]
    if brake_a is not None and brake_b is not None:
        brake_val = f"A @{brake_a:.1f} s / B @{brake_b:.1f} s"
    elif brake_a is not None or brake_b is not None:
        only = f"A @{brake_a:.1f} s" if brake_a is not None else f"B @{brake_b:.1f} s"
        brake_val = f"{only} (other n/a)"
    else:
        brake_val = "n/a"
    cells = [
        ("min clearance (focal ped)", f"A {clear_a:.2f} m / B {clear_b:.2f} m"),
        ("near-miss steps", f"A {near_a} / B {near_b}"),
        ("steps to termination", f"A {steps_a} / B {steps_b}"),
        ("first braking", brake_val),
    ]
    # Cell centers: NOT an even quarter-split. At the true (narrower) print width the
    # even split puts the long "min clearance (focal ped)" / "near-miss steps" headers
    # (cells 0/1) close enough to collide; cells 2/3's shorter headers have slack, so
    # centers are nudged left/right to borrow that slack for cells 0/1 (text unchanged).
    cell_x = (0.11, 0.40, 0.635, 0.87)
    for (head, value), x in zip(cells, cell_x, strict=True):
        ax.text(x, 0.74, head, fontsize=fontsize, ha="center", va="center", color="#555555")
        ax.text(x, 0.24, value, fontsize=fontsize, ha="center", va="center", color="#111111")


def _build_legend_elements(
    *,
    contrast_mode: bool,
    print_layout: bool,
    outcome_b: str,
    b_outcome_step: int | None,
) -> list[Line2D]:
    """Legend entries per rendering mode/layout.

    Contrast mode uses contrast vocabulary only -- no pivot/hinge/common-prefix wording
    anywhere in the rendered output (matched seed-pair contrast framing, author ruling
    2026-07-16). The print layout additionally names the pedestrian line styles and the
    outcome marker here, because the print declutter removed their in-panel labels.

    Returns:
        Legend handle list.
    """
    if not contrast_mode:
        return [
            Line2D([0], [0], color=COLOR_COMMON_PREFIX, lw=1.6, label="common prefix (both)"),
            Line2D(
                [0],
                [0],
                color=COLOR_A,
                lw=1.8,
                marker=MARKER_A,
                markersize=5,
                label="A post-pivot",
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
                label="B post-pivot",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markeredgecolor="black",
                markerfacecolor="none",
                markersize=11,
                label="pivot (separator)",
            ),
        ]
    elements = [
        Line2D(
            [0],
            [0],
            color=COLOR_A,
            lw=1.8,
            marker=MARKER_A,
            markersize=5,
            label="episode A trajectory",
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
            label="episode B trajectory",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markeredgecolor="black",
            markerfacecolor="white",
            markersize=7,
            label="trace start",
        ),
    ]
    if print_layout:
        elements += [
            Line2D([0], [0], color=COLOR_PED_FOCAL_OUTLINE, lw=1.1, label="focal pedestrian"),
            Line2D([0], [0], color=COLOR_PED_CONTEXT, lw=0.8, label="other pedestrians"),
        ]
        if b_outcome_step is not None and outcome_b == "collision":
            elements.append(
                Line2D(
                    [0],
                    [0],
                    marker="x",
                    color="none",
                    markeredgecolor=COLOR_COLLISION,
                    markeredgewidth=2.0,
                    markersize=8,
                    label="collision",
                )
            )
        elif b_outcome_step is not None and outcome_b == "near_miss":
            elements.append(
                Line2D(
                    [0],
                    [0],
                    marker="^",
                    color="none",
                    markeredgecolor=COLOR_COLLISION,
                    markerfacecolor="none",
                    markersize=8,
                    label="near miss",
                )
            )
    return elements


def render_hinge_figure(  # noqa: PLR0913 - top-level figure assembly; each argument is a
    # distinct precomputed fact (event dicts, labels, outcome kinds) the two panels need.
    ep_a: EpisodeTrace,
    ep_b: EpisodeTrace,
    *,
    label_a: str,
    label_b: str,
    divergence: dict[str, Any],
    separator: dict[str, Any],
    gutter: dict[str, Any],
    focal_ped_id: int,
    events_a: dict[str, dict[str, Any]],
    events_b: dict[str, dict[str, Any]],
    closest_a: dict[str, Any],
    closest_b: dict[str, Any],
    outcome_a: str,
    outcome_b: str,
    b_outcome_step: int | None,
    headline: str,
    map_definition: Any | None,
    out_pdf: Path,
    out_png: Path,
    pivot_label_text: str = "pivot",
    contrast_mode: bool = False,
    layout: str = "screen",
) -> list[Any]:
    """Render + export the two-panel figure (vector PDF + PNG preview).

    Two rendering modes:
    - hinge mode (default): the original pivot grammar -- gray common prefix, post-pivot
      A/B styling, pivot ring, delta gutter. For true shared-prefix pairs (e.g. the
      same-seed orca/social_force reference pair).
    - ``contrast_mode``: matched seed-pair contrast framing (author ruling 2026-07-16),
      for pairs with no shared start -- full per-episode trajectories, "trace start"
      legend vocabulary instead of pivot/prefix vocabulary, and a contrast gutter
      (``compute_contrast_gutter``) instead of pivot-anchored deltas.

    Two layouts:
    - ``layout="screen"`` (default): the original wide-format prototype layout
      (10.3 x ~3.5 in, vertical central gutter, multi-sentence suptitle) -- unchanged.
    - ``layout="print"`` (contrast mode only): designed at FINAL size -- total width
      exactly ``PRINT_FIG_WIDTH_IN`` (5.906 in, included at \\textwidth with no LaTeX
      rescaling, so pt here == pt on the page). Two equal-aspect panels side by side,
      the contrast gutter folded into a compact single-row strip below the panels
      (``_draw_contrast_strip``), NO suptitle (the LaTeX caption owns the takeaway --
      house style), 8 pt minimum rendered font (``_PRINT_PANEL_STYLE``), decluttered
      panels (no braking marker, no context-ped id labels; those roles move to the
      caption and legend).

    Returns:
        The list of ``figure_qa.lint_figure`` defects found on the rendered figure (before
        the caller closes it), so the CLI can report an exact before/after lint count.
    """
    print_layout = layout == "print"
    panel_style = _PRINT_PANEL_STYLE if print_layout else _SCREEN_PANEL_STYLE
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
    if print_layout:
        # Design at final size: width is EXACTLY PRINT_FIG_WIDTH_IN; height = the two
        # equal-aspect panels' data height + tick/label band + strip row + legend band.
        # No suptitle (the LaTeX caption owns the takeaway).
        fig_w = PRINT_FIG_WIDTH_IN
        panel_data_h = (fig_w / 2.0) * aspect
        fig_h = panel_data_h + 1.45  # ticks/labels/titles + strip + legend
        fig = plt.figure(figsize=(fig_w, fig_h))
        grid = fig.add_gridspec(
            2, 2, height_ratios=(panel_data_h + 0.55, 0.42), hspace=0.08, wspace=0.10
        )
        ax_a = fig.add_subplot(grid[0, 0])
        ax_b = fig.add_subplot(grid[0, 1])
        ax_gutter = fig.add_subplot(grid[1, :])
    else:
        panel_w = 4.4
        fig_h = panel_w * aspect + 1.0
        fig = plt.figure(figsize=(panel_w * 2 + 1.5, fig_h))
        grid = fig.add_gridspec(1, 3, width_ratios=(1.0, 0.34, 1.0), wspace=0.06)
        ax_a = fig.add_subplot(grid[0, 0])
        ax_gutter = fig.add_subplot(grid[0, 1])
        ax_b = fig.add_subplot(grid[0, 2])

    # Pivot ring position = the separator found by find_separator (design spec: "a
    # conspicuous pivot ring at the first persistent action difference"), NOT the raw D(t)
    # persistence-onset step t_d -- the separator is t_d itself or up to ~1s earlier.
    pivot_step = separator["step"]
    common_prefix_end = pivot_step

    # defer_labels=True: draw geometry now, place the two dynamic labels (pivot,
    # clearance) AFTER fig.tight_layout() below -- see _draw_panel's docstring for why a
    # single-pass placement measurably cleared every line/marker at draw time yet still
    # showed up in figure_qa.lint_figure (tight_layout moves the axes afterward).
    pending_a = _draw_panel(
        ax_a,
        ep_a,
        color=COLOR_A,
        marker=MARKER_A,
        linestyle="-",
        label=f"A -- {label_a}",
        divergence_step=pivot_step,
        common_prefix_end=common_prefix_end,
        focal_ped_id=focal_ped_id,
        critical_events=events_a,
        closest=closest_a,
        collision_or_near_miss_step=None,
        outcome_kind=outcome_a,
        bounds=bounds,
        map_definition=map_definition,
        defer_labels=True,
        pivot_label_text=pivot_label_text,
        contrast_mode=contrast_mode,
        style=panel_style,
    )
    pending_b = _draw_panel(
        ax_b,
        ep_b,
        color=COLOR_B,
        marker=MARKER_B,
        linestyle="--",
        label=f"B -- {label_b}",
        divergence_step=pivot_step,
        common_prefix_end=common_prefix_end,
        focal_ped_id=focal_ped_id,
        critical_events=events_b,
        closest=closest_b,
        collision_or_near_miss_step=b_outcome_step,
        outcome_kind=outcome_b,
        bounds=bounds,
        map_definition=map_definition,
        defer_labels=True,
        pivot_label_text=pivot_label_text,
        contrast_mode=contrast_mode,
        style=panel_style,
    )
    ax_b.set_ylabel("")
    ax_a.set_ylabel("y (m)", fontsize=panel_style["axis_fs"])
    if print_layout:
        _draw_contrast_strip(ax_gutter, gutter, fontsize=panel_style["annot_fs"])
    else:
        _draw_delta_gutter(ax_gutter, gutter)

    legend_elements = _build_legend_elements(
        contrast_mode=contrast_mode,
        print_layout=print_layout,
        outcome_b=outcome_b,
        b_outcome_step=b_outcome_step,
    )
    if print_layout:
        # 8 pt legend (>= the 7 pt on-page minimum); 3 columns x 2 rows fits PRINT_FIG_WIDTH_IN.
        fig.legend(
            handles=legend_elements,
            loc="lower center",
            ncol=3,
            fontsize=8,
            frameon=False,
            bbox_to_anchor=(0.5, -0.005),
            handlelength=1.6,
            columnspacing=1.2,
        )
        # NO suptitle: the LaTeX caption owns the takeaway (house style).
        fig.tight_layout(rect=(0, 0.17, 1, 1.0))
    else:
        fig.legend(
            handles=legend_elements,
            loc="lower center",
            ncol=len(legend_elements),
            fontsize=7,
            frameon=False,
            bbox_to_anchor=(0.5, -0.005),
        )
        fig.suptitle(headline, fontsize=9, wrap=True, y=0.995)
        fig.tight_layout(rect=(0, 0.045, 1, 0.975))

    # Place the two dynamic labels per panel now that the final axes layout is fixed
    # (see the defer_labels note above).
    for ax, pending in ((ax_a, pending_a), (ax_b, pending_b)):
        for anchor_xy, text, style_kwargs in pending:
            _place_clear_label(ax, anchor_xy, text, **style_kwargs)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    # Pin CreationDate so re-renders with identical inputs are byte-identical (matplotlib
    # otherwise stamps the wall-clock time into the PDF, the only source of nondeterminism
    # observed in this script -- geometry/text content is already deterministic).
    fig.savefig(out_pdf, metadata={"CreationDate": None})
    # Pin the print PNG to the figure's STANDARD bbox (not a tight crop of the drawn
    # artists). Without this, savefig honors rcParams["savefig.bbox"], which is None
    # locally but can resolve to "tight" under other font/layout environments (e.g.
    # the CI runner), cropping the PNG to the drawn content and shrinking its width
    # below PRINT_FIG_WIDTH_IN. Pinning the standard bbox makes the rendered width
    # deterministically round(PRINT_FIG_WIDTH_IN * dpi) == 1181 at 200 dpi regardless
    # of font/layout env (issue #6088 root cause: the print-design width contract was
    # previously satisfied only by environmental accident, not by construction).
    fig.savefig(out_png, dpi=200, bbox_inches=fig.bbox_inches)

    # QA gate: reuse the repo's own text/marker-collision linter (item 4 -- report the
    # exact before/after defect count, not just "pass/fail").
    defects = figure_qa.lint_figure(fig)
    errors = [d for d in defects if getattr(d, "severity", "") == "error"]
    if errors:
        print(f"figure_qa.lint_figure: {len(errors)} error-severity defect(s):", file=sys.stderr)
        for d in errors:
            print(f"  - {d}", file=sys.stderr)
    else:
        print(f"figure_qa.lint_figure: clean ({len(defects)} non-error defect(s))")

    plt.close(fig)
    return defects


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


#: Lint defect count from the first-cut prototype's output, captured for the record 2026-07-15
#: (``output/butterfly_hinge_proto/butterfly_hinge_figure_proto.pdf``, commit 5124e9faf):
#: 3x ``text_line_overlap`` + 1x ``text_marker_overlap``, all error-severity, all from the
#: divergence/clearance annotation offsets. Kept as a literal here (not re-derived at
#: runtime) so the before/after comparison in the report is against the actual prior
#: artifact, not a re-run of deleted code.
FIRST_CUT_LINT_ERROR_COUNT: int = 4


def _git_commit(repo_dir: Path) -> str | None:
    """Best-effort ``git rev-parse HEAD`` for provenance; ``None`` if unavailable.

    Returns:
        The 40-character commit hash, or ``None`` if git is unavailable or the directory
        is not a git repository.
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def build_provenance_sidecar(  # noqa: PLR0913 - every argument is a distinct provenance fact
    *,
    ep_a: EpisodeTrace,
    ep_b: EpisodeTrace,
    episode_a_bundle: Path,
    episode_b_bundle: Path,
    focal_selection: dict[str, Any],
    divergence: dict[str, Any],
    separator: dict[str, Any],
    gutter: dict[str, Any],
    figure_pdf: Path,
    figure_png: Path,
    qa_defects_before: int,
    qa_defects_after: int,
    framing: dict[str, Any] | None = None,
    figure_layout: str = "screen",
) -> dict[str, Any]:
    """Provenance sidecar (item 4): episode ids, planners, seeds, source commits,
    config/trace hashes -- everything needed to reproduce or audit the shipped still
    independent of the (much larger) ``butterfly_hinge_report.json`` analysis dump.
    ``framing`` records which rendering mode produced the figure (hinge/pivot vs
    matched seed-pair contrast) and, for contrast mode, the author ruling that chose it.

    Returns:
        JSON-serializable provenance dict.
    """
    script_path = Path(__file__).resolve()
    robot_sf_repo = script_path.parents[2]  # scripts/repro/<this file> -> repo root
    trace_a_path = episode_a_bundle / "trace_series.json"
    trace_b_path = episode_b_bundle / "trace_series.json"
    return {
        "schema_version": "butterfly-hinge-provenance.v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "framing": framing or {"mode": "hinge_pivot"},
        "script": {
            "path": str(script_path.relative_to(robot_sf_repo)),
            "repo": str(robot_sf_repo),
            "git_commit": _git_commit(robot_sf_repo),
        },
        "episode_a": {
            "bundle_dir": str(episode_a_bundle),
            "episode_id": ep_a.metadata.get("episode_id"),
            "planner": ep_a.metadata.get("planner"),
            "scenario_id": ep_a.metadata.get("scenario_id"),
            "seed": ep_a.metadata.get("seed"),
            "episode_status": ep_a.metadata.get("episode_status"),
            "source_git_commit": ep_a.metadata.get("git_commit"),
            "trace_series_sha256": _sha256_of_file(trace_a_path),
            "n_steps_used": len(ep_a.robot_xy),
        },
        "episode_b": {
            "bundle_dir": str(episode_b_bundle),
            "episode_id": ep_b.metadata.get("episode_id"),
            "planner": ep_b.metadata.get("planner"),
            "scenario_id": ep_b.metadata.get("scenario_id"),
            "seed": ep_b.metadata.get("seed"),
            "episode_status": ep_b.metadata.get("episode_status"),
            "source_git_commit": ep_b.metadata.get("git_commit"),
            "trace_series_sha256": _sha256_of_file(trace_b_path),
            "n_steps_used": len(ep_b.robot_xy),
        },
        "focal_pedestrian_selection": focal_selection,
        "divergence_detector_config": {
            "scales": divergence["scales"],
            "weights": divergence["weights"],
            "hold_s": DIVERGENCE_HOLD_S,
            "floor_window_s": DIVERGENCE_FLOOR_WINDOW_S,
            "threshold_margin": DIVERGENCE_THRESHOLD_MARGIN,
            "separator_search_back_s": SEPARATOR_SEARCH_BACK_S,
            "command_jump_threshold_omega_rad_s": COMMAND_DIFF_OMEGA_THRESHOLD_RAD_S,
            "command_jump_threshold_v_mps": COMMAND_DIFF_LINEAR_THRESHOLD_MPS,
        },
        "pivot": {"separator": separator, "delta_gutter": gutter},
        "figure": {
            "pdf": str(figure_pdf),
            "pdf_sha256": _sha256_of_file(figure_pdf),
            "png": str(figure_png),
            "layout": figure_layout,
        },
        "figure_qa": {
            "first_cut_error_defects": qa_defects_before,
            "this_render_error_defects": qa_defects_after,
        },
    }


def _compose_contrast_headline(
    ep_a: EpisodeTrace, ep_b: EpisodeTrace, gutter: dict[str, Any], start_sep_m: float
) -> str:
    """Contrast-framing headline: descriptive only, no causal claim, no pivot/hinge/
    common-prefix vocabulary (matched seed-pair contrast framing, author ruling
    2026-07-16). Every number is derived from the loaded traces or their bundle
    metadata.

    Returns:
        Two-sentence headline string.
    """
    return (
        f"Same planner ({ep_a.metadata['planner']}), same scenario "
        f"({ep_a.metadata['scenario_id']}), adjacent seeds "
        f"{ep_a.metadata['seed']} vs {ep_b.metadata['seed']}: the seed changes both "
        f"the pedestrian realization and the robot spawn draw "
        f"(starts {start_sep_m:.1f} m apart). "
        f"The outcome flips from {ep_a.metadata['episode_status']} "
        f"({gutter['steps_to_termination']['episode_a']} steps, "
        f"{gutter['near_miss_steps']['episode_a']} near-miss steps) to "
        f"{ep_b.metadata['episode_status']} "
        f"({gutter['steps_to_termination']['episode_b']} steps, "
        f"{gutter['near_miss_steps']['episode_b']} near-miss steps)."
    )


def _compose_hinge_headline(
    separator: dict[str, Any],
    onset: dict[str, Any],
    focal_closest_a: dict[str, Any],
    focal_closest_b: dict[str, Any],
) -> str:
    """Hinge-mode headline (original pivot grammar, unchanged behavior -- extracted from
    ``main`` verbatim): separator clause per tier + D(t) persistence + clearance contrast.

    Returns:
        Headline string.
    """
    d_clearance = focal_closest_a["distance_m"] - focal_closest_b["distance_m"]
    if separator["tier"] == "command_jump":
        who = separator["jump_in"]
        kind = separator["jump_kind"]
        mag = separator["jump_v_mps"] if kind == "v" else separator["jump_omega_rad_s"]
        unit = "m/s" if kind == "v" else "rad/s"
        separator_clause = (
            f"First separator at t={separator['time_s']:.2f} s: {who} commands a "
            f"{mag:.2f} {unit} {kind}-command jump. "
        )
    elif separator["tier"] == "braking_onset":
        separator_clause = (
            f"First separator at t={separator['time_s']:.2f} s: {separator['braking_in']} "
            f"begins braking (critical_intervals.first_braking_event). "
        )
    else:  # largest_risk_rise
        separator_clause = (
            f"No discrete command jump or braking onset found in the "
            f"{SEPARATOR_SEARCH_BACK_S:g} s window before the divergence became persistent; "
            f"the separator is reported as the step of largest single-step rise in the "
            f"joint-state divergence D(t), at t={separator['time_s']:.2f} s (this pair's "
            f"divergence accumulates smoothly rather than from one sharp decision -- see "
            f"the report's `pivot.separator` for the full tier trail). "
        )
    return (
        f"{separator_clause}"
        f"D(t) becomes persistently divergent (threshold {onset['threshold']:.2f}, "
        f"{DIVERGENCE_THRESHOLD_MARGIN:g}x the {onset['floor']:.2f} pre-divergence floor) "
        f"by t={onset['time_s']:.2f} s. "
        f"B's subsequent minimum clearance to the focal pedestrian "
        f"({focal_closest_b['distance_m']:.2f} m at t={focal_closest_b['time_s']:.1f} s) "
        f"is {d_clearance:.2f} m lower than A's ({focal_closest_a['distance_m']:.2f} m at "
        f"t={focal_closest_a['time_s']:.1f} s)."
    )


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
    parser.add_argument(
        "--layout",
        choices=("screen", "print"),
        default="screen",
        help="Figure layout: 'screen' (original wide prototype layout, default) or 'print' "
        f"(design-at-final-size, {PRINT_FIG_WIDTH_IN:g} in wide for \\textwidth inclusion "
        "with no rescaling, 8 pt minimum fonts, no header text, compact contrast strip; "
        "currently implemented for the contrast rendering mode only).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0915 - linear CLI orchestration
    """CLI entry point.

    Returns:
        Process exit code.
    """
    args = _build_parser().parse_args(argv)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ep_a = load_episode(args.episode_a, args.label_a)
    ep_b = load_episode(args.episode_b, args.label_b, max_steps=args.b_story_steps)

    # Item 1: lock ONE focal pedestrian, shared by both panels, reusing the shipped
    # trace_scene_figure selection rule (see select_focal_pedestrian docstring).
    focal_selection = select_focal_pedestrian(args.episode_a, ep_b)
    focal_ped_id = focal_selection["ped_id"]

    # Item 2: D(t) normalized joint-state divergence detector (replaces the first cut's
    # threshold-on-robot-robot-separation).
    divergence = compute_joint_state_divergence(ep_a, ep_b, focal_ped_id)
    dt = float(ep_a.time_s[1] - ep_a.time_s[0]) if len(ep_a.time_s) > 1 else 0.1
    onset = find_persistence_onset(divergence, dt)
    # A persistence-onset step can genuinely fail to exist for a reason other than "no
    # divergence": if A and B do not share a common start (e.g. two different
    # route_spawn_seed draws of the same scenario, rather than a same-seed/
    # different-planner pair), D(t) is already elevated at t=0 and never dips back to a
    # true pre-interaction floor, so no threshold crossing is possible by construction
    # (the floor itself is computed from that same elevated window). Reported honestly
    # as "no shared prefix" -- same "explicit unavailable, not fabricated" precedent as
    # find_separator's tier-1 (planner mode) note -- rather than erroring out or forcing
    # a fake mid-trace pivot.
    no_shared_prefix = onset["step"] is None
    if no_shared_prefix:
        start_sep_m = float(np.linalg.norm(ep_a.robot_xy[0] - ep_b.robot_xy[0]))
        onset = {
            "step": 0,
            "time_s": float(ep_a.time_s[0]),
            "threshold": onset["threshold"],
            "floor": onset["floor"],
            "hold_steps": onset["hold_steps"],
            "floor_window_steps": onset["floor_window_steps"],
            "no_shared_prefix": True,
            "no_shared_prefix_note": (
                "D(t) never returns to a genuine pre-interaction floor within this "
                "trace, so no persistence-onset threshold crossing exists; A and B "
                f"start {start_sep_m:.2f} m apart (different per-seed spawn draws), "
                "not from a shared prefix"
            ),
        }
    else:
        onset["time_s"] = float(ep_a.time_s[onset["step"]])
    divergence["onset"] = onset

    events_a = compute_critical_events(ep_a)
    events_b = compute_critical_events(ep_b)
    if no_shared_prefix:
        separator = {
            "step": 0,
            "time_s": float(ep_a.time_s[0]),
            "tier": "no_shared_prefix",
            "tier1_note": (
                "tier 1 (first differing planner mode) is unavailable: this trace "
                "schema carries no per-step categorical planner-mode field"
            ),
            "note": (
                "episodes A and B do not share a common start (different "
                "route_spawn_seed draws under the scenario's route_spawn_distribution="
                "'spread', route_spawn_jitter_frac=0.2 spawn randomization), so there is "
                "no shared prefix for a backward separator search to run over; the pivot "
                "is reported as the trace start (step 0) instead of a mid-trace "
                "separator"
            ),
        }
    else:
        separator = find_separator(ep_a, ep_b, divergence, onset["step"], events_a, events_b)

    closest_a = closest_approach(ep_a)
    closest_b = closest_approach(ep_b)
    focal_closest_a = closest_approach_to_focal_ped(ep_a, focal_ped_id)
    focal_closest_b = closest_approach_to_focal_ped(ep_b, focal_ped_id)

    # Contrast mode (matched seed-pair contrast framing, author ruling 2026-07-16):
    # pivot-anchored deltas at "pivot = step 0" would only measure start-state
    # differences under different spawn draws, so the gutter switches to per-episode
    # contrast quantities that stay meaningful across different starts.
    contrast_mode = no_shared_prefix
    if contrast_mode:
        gutter = compute_contrast_gutter(
            ep_a, ep_b, focal_closest_a, focal_closest_b, events_a, events_b
        )
    else:
        gutter = compute_delta_gutter(
            ep_a, ep_b, separator["step"], focal_ped_id, events_a, events_b
        )

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

    pivot_label_text = "pivot"
    if contrast_mode:
        headline = _compose_contrast_headline(ep_a, ep_b, gutter, start_sep_m)
    else:
        headline = _compose_hinge_headline(separator, onset, focal_closest_a, focal_closest_b)

    layout: str = args.layout
    if layout == "print" and not contrast_mode:
        print(
            "warning: --layout print is currently implemented for the contrast rendering "
            "mode only (this pair has a shared prefix -> hinge mode); falling back to the "
            "screen layout",
            file=sys.stderr,
        )
        layout = "screen"

    scenario_id = str(ep_a.metadata["scenario_id"])
    try:
        map_definition = tsf._load_map_definition(scenario_id)
    except (tsf.TraceSchemaError, OSError, KeyError, ValueError, SyntaxError) as exc:
        print(f"warning: could not load map definition for obstacles: {exc}", file=sys.stderr)
        map_definition = None

    out_pdf = out_dir / "butterfly_hinge_figure_proto.pdf"
    out_png = out_dir / "butterfly_hinge_figure_proto.png"
    qa_defects = render_hinge_figure(
        ep_a,
        ep_b,
        label_a=args.label_a,
        label_b=args.label_b,
        divergence=divergence,
        separator=separator,
        gutter=gutter,
        focal_ped_id=focal_ped_id,
        events_a=events_a,
        events_b=events_b,
        closest_a=focal_closest_a,
        closest_b=focal_closest_b,
        outcome_a=outcome_a,
        outcome_b=outcome_b,
        b_outcome_step=b_outcome_step,
        headline=headline,
        map_definition=map_definition,
        out_pdf=out_pdf,
        out_png=out_png,
        pivot_label_text=pivot_label_text,
        contrast_mode=contrast_mode,
        layout=layout,
    )
    qa_errors = [d for d in qa_defects if getattr(d, "severity", "") == "error"]

    framing = (
        {
            "mode": "matched_seed_pair_contrast",
            "author_ruling_2026-07-16": (
                "matched seed-pair contrast framing; hinge/pivot framing not applicable "
                "(no shared prefix); spawn-pinned diagnostic deferred"
            ),
        }
        if contrast_mode
        else {"mode": "hinge_pivot"}
    )

    report: dict[str, Any] = {
        "framing": framing,
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
        "focal_pedestrian_selection": focal_selection,
        "divergence": {k: v for k, v in divergence.items() if k != "D"}
        | {"D": divergence["D"].tolist()},
        "pivot": {"separator": separator, "delta_gutter": gutter},
        "critical_events_a": events_a,
        "critical_events_b": events_b,
        "closest_approach_any_ped_a": closest_a,
        "closest_approach_any_ped_b": closest_b,
        "closest_approach_focal_ped_a": focal_closest_a,
        "closest_approach_focal_ped_b": focal_closest_b,
        "headline": headline,
        "figure_layout": layout,
        "figure_pdf": str(out_pdf),
        "figure_png": str(out_png),
        "figure_pdf_sha256": _sha256_of_file(out_pdf),
        "figure_qa": {
            "first_cut_error_defects": FIRST_CUT_LINT_ERROR_COUNT,
            "this_render_error_defects": len(qa_errors),
            "this_render_all_defects": len(qa_defects),
        },
    }

    if not args.no_video:
        video_out = out_dir / "butterfly_hinge_ab_proto.mp4"
        video_summary = render_ab_video(
            args.episode_a, args.episode_b, out_path=video_out, b_max_steps=args.b_story_steps
        )
        report["video"] = video_summary

    report_path = out_dir / "butterfly_hinge_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    provenance = build_provenance_sidecar(
        ep_a=ep_a,
        ep_b=ep_b,
        episode_a_bundle=args.episode_a,
        episode_b_bundle=args.episode_b,
        focal_selection=focal_selection,
        divergence=divergence,
        separator=separator,
        gutter=gutter,
        figure_pdf=out_pdf,
        figure_png=out_png,
        qa_defects_before=FIRST_CUT_LINT_ERROR_COUNT,
        qa_defects_after=len(qa_errors),
        framing=framing,
        figure_layout=layout,
    )
    provenance_path = out_dir / "butterfly_hinge_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, default=str), encoding="utf-8")

    print(json.dumps(report, indent=2, default=str))
    print(f"provenance sidecar: {provenance_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
