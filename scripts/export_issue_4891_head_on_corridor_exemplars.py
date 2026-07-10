#!/usr/bin/env python3
"""Export exemplar trace-episode bundles for issue #4891 head-on corridor scenarios.

Reads retained campaign data from issue4206_trace_capable_h600_rerun_20260704 (job 13334),
selects exemplar episodes (median + best/worst) from head-on corridor scenarios for 2-3 planners,
and exports trace-episode bundles in the same format as issue_4253/4268 and issue_4848.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from robot_sf.evidence.distance_convention import DistanceConvention
from robot_sf.evidence.writers import (
    extract_marker_date,
    review_marker,
    write_csv,
    write_distance_series_csv,
    write_json,
    write_sha256sums,
)

# Back-compat alias for the shared helper (kept for the module-attribute tests).
_extract_marker_date = extract_marker_date


# Target planners for exemplar selection (classical + social navigation diversity)
TARGET_PLANNERS = ["goal", "orca", "social_force"]

# Scenario class filter: head-on corridor variants
HEAD_ON_CORRIDOR_SCENARIOS = {
    "classic_head_on_corridor_low",
    "classic_head_on_corridor_medium",
}

# Selection metric (path_efficiency: higher is better)
SELECTION_METRIC = "path_efficiency"
SELECTION_MODES = ["median", "best", "worst"]

# Output directory
DEFAULT_OUTPUT_DIR = Path("docs/context/evidence/issue_4891_head_on_corridor_exemplars_2026-07")
DEFAULT_CAMPAIGN_ROOT = Path("output/issue4206-trace-rerun/13334/runs")


class ExemplarExportInputError(ValueError):
    """Raised when an exemplar export input cannot support valid evidence output."""


@dataclass(frozen=True)
class SelectedEpisode:
    """One selected exemplar episode."""

    planner: str
    scenario_id: str
    seed: int
    selection_mode: str
    metric_value: float
    episode_id: str
    status: str


@dataclass(frozen=True)
class TraceRows:
    """Derived figure-ready rows from one recorded episode."""

    trace_rows: list[dict[str, Any]]
    min_distance_rows: list[dict[str, Any]]
    summary: dict[str, Any]


def _repo_root() -> Path:
    """Return the current git worktree root."""
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip())


def _git_commit() -> str:
    """Return the current commit hash, or ``unknown`` outside git."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def _min_distance(
    robot_xy: list[float], pedestrians: list[dict[str, Any]]
) -> tuple[float | None, str | None]:
    """Return the nearest pedestrian distance and id for a trace frame."""
    if not pedestrians:
        return None, None
    nearest_distance = math.inf
    nearest_id: str | None = None
    for pedestrian in pedestrians:
        position = pedestrian.get("position")
        if not isinstance(position, list) or len(position) < 2:
            continue
        distance = math.dist(robot_xy, [float(position[0]), float(position[1])])
        if distance < nearest_distance:
            nearest_distance = distance
            nearest_id = str(pedestrian.get("id", "unknown"))
    if not math.isfinite(nearest_distance):
        return None, None
    return nearest_distance, nearest_id


def derive_trace_rows(record: dict[str, Any]) -> TraceRows:
    """Convert a campaign episode record into trace and min-distance CSV rows."""
    trace = record.get("algorithm_metadata", {}).get("simulation_step_trace")
    if not trace or "steps" not in trace:
        raise ValueError("Episode lacks simulation_step_trace data")

    rows: list[dict[str, Any]] = []
    min_rows: list[dict[str, Any]] = []
    global_min_distance: float | None = None
    global_min_step: int | None = None

    for frame in trace["steps"]:
        robot = frame["robot"]
        robot_xy = [float(robot["position"][0]), float(robot["position"][1])]
        velocity = robot.get("velocity", [0.0, 0.0])
        executed_vx = float(velocity[0])
        executed_vy = float(velocity[1])
        executed_speed = math.hypot(executed_vx, executed_vy)
        action = frame.get("planner", {}).get("selected_action", {})
        commanded_linear = action.get("linear_velocity")
        commanded_angular = action.get("angular_velocity")
        pedestrians = frame.get("pedestrians", [])
        min_distance, nearest_pedestrian_id = _min_distance(robot_xy, pedestrians)
        step = int(frame["step"])
        time_s = float(frame["time_s"])

        if min_distance is not None and (
            global_min_distance is None or min_distance < global_min_distance
        ):
            global_min_distance = min_distance
            global_min_step = step

        rows.append(
            {
                "step": step,
                "time_s": time_s,
                "robot_x_m": robot_xy[0],
                "robot_y_m": robot_xy[1],
                "robot_heading_rad": float(robot.get("heading", 0.0)),
                "executed_vx_m_s": executed_vx,
                "executed_vy_m_s": executed_vy,
                "executed_speed_m_s": executed_speed,
                "commanded_linear_velocity_m_s": commanded_linear,
                "commanded_angular_velocity_rad_s": commanded_angular,
                "nearest_pedestrian_id": nearest_pedestrian_id,
                "min_robot_ped_distance_m": min_distance,
                "pedestrian_count": len(pedestrians),
                "pedestrian_positions_json": json.dumps(
                    [
                        {
                            "id": str(pedestrian.get("id", index)),
                            "x_m": float(pedestrian["position"][0]),
                            "y_m": float(pedestrian["position"][1]),
                        }
                        for index, pedestrian in enumerate(pedestrians)
                        if isinstance(pedestrian.get("position"), list)
                        and len(pedestrian["position"]) >= 2
                    ],
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
        )
        min_rows.append(
            {
                "step": step,
                "time_s": time_s,
                "min_robot_ped_distance_m": min_distance,
                "nearest_pedestrian_id": nearest_pedestrian_id,
            }
        )

    summary = {
        "step_count": len(rows),
        "global_min_robot_ped_distance_m": global_min_distance,
        "global_min_distance_step": global_min_step,
        "episode_status": record.get("status"),
        "termination_reason": record.get("termination_reason"),
        "scenario_id": record.get("scenario_id"),
        "seed": record.get("seed"),
        "planner": record.get("algorithm") or record.get("algorithm_metadata", {}).get("algorithm"),
    }
    return TraceRows(trace_rows=rows, min_distance_rows=min_rows, summary=summary)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a non-empty JSONL file or fail before an empty export can be emitted."""
    if not path.is_file():
        raise ExemplarExportInputError(f"episodes JSONL is not a file: {path}")

    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise ExemplarExportInputError(f"episodes JSONL is empty: {path}")
    return records


def select_exemplars_for_planner(
    episodes: list[dict[str, Any]], planner: str
) -> list[SelectedEpisode]:
    """Select median, best, worst exemplar episodes for one planner.

    Uses quality-aware tie-breaking: when path_efficiency is tied,
    prefers episodes with more steps (richer trace content) and
    filters out single-step collision episodes that lack visualization value.
    """
    # Filter for head-on corridor scenarios only
    corridor_eps = [ep for ep in episodes if ep.get("scenario_id") in HEAD_ON_CORRIDOR_SCENARIOS]

    if not corridor_eps:
        raise ExemplarExportInputError(
            f"planner {planner!r}: no eligible head-on-corridor episodes for exemplar selection"
        )

    # Extract metric values with step count for quality filtering
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for ep in corridor_eps:
        metrics = ep.get("metrics", {})
        val = metrics.get(SELECTION_METRIC)
        if val is not None and isinstance(val, (int, float)):
            if math.isfinite(float(val)):
                # Extract step count for tie-breaking
                step_count = _extract_step_count(ep)
                scored.append((float(val), step_count or 0, ep))

    if not scored:
        raise ExemplarExportInputError(
            f"planner {planner!r}: no finite selection metric {SELECTION_METRIC!r} "
            f"among {len(corridor_eps)} eligible episodes"
        )

    # Filter out single-step episodes (no visualization content)
    MIN_STEP_COUNT = 2
    filtered_scored = [(val, steps, ep) for val, steps, ep in scored if steps >= MIN_STEP_COUNT]

    # Fall back to unfiltered if all episodes are single-step
    if not filtered_scored:
        filtered_scored = scored

    # Sort by metric value (ascending for median index invariance), then by
    # step count ascending. Within a tied path_efficiency group this places the
    # richest trace (most steps) at the highest index, so the "best" exemplar
    # (highest path_efficiency, last index) lands on the episode with the most
    # steps rather than the fewest. A descending secondary key here would
    # invert best/worst and starve the best exemplar of trace content.
    filtered_scored.sort(key=lambda x: (x[0], x[1]))

    selected: list[SelectedEpisode] = []
    for mode in SELECTION_MODES:
        if mode == "best":
            idx = len(filtered_scored) - 1  # higher is better for path_efficiency
        elif mode == "worst":
            idx = 0  # lower is worse for path_efficiency
        else:  # median
            idx = len(filtered_scored) // 2

        val, step_count, ep = filtered_scored[idx]
        selected.append(
            SelectedEpisode(
                planner=planner,
                scenario_id=str(ep.get("scenario_id", "")),
                seed=int(ep.get("seed", 0)),
                selection_mode=mode,
                metric_value=val,
                episode_id=str(ep.get("episode_id", "")),
                status=str(ep.get("status", "")),
            )
        )

    return selected


def _extract_step_count(record: dict[str, Any]) -> int | None:
    """Extract step count from an episode record."""
    # Direct step_count field
    val = record.get("step_count")
    if isinstance(val, (int, float)) and math.isfinite(val) and val > 0:
        return int(val)

    # From summary or metrics
    for path in ("summary.step_count", "metrics.step_count"):
        val = _get_nested(record, path)
        if isinstance(val, (int, float)) and math.isfinite(val) and val > 0:
            return int(val)

    # From trace data
    trace = record.get("algorithm_metadata", {}).get("simulation_step_trace")
    if trace and "steps" in trace:
        return len(trace["steps"])

    return None


def _get_nested(record: dict[str, Any], path: str, default: Any = None) -> Any:
    """Resolve a dotted-path value from a dict."""
    cur: Any = record
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def write_bundle(
    *,
    episode_record: dict[str, Any],
    selection: SelectedEpisode,
    output_dir: Path,
    pin_generated_at: str | None = None,
) -> dict[str, Any]:
    """Write a trace episode bundle for one selected episode."""
    derived = derive_trace_rows(episode_record)

    metadata = {
        "schema_version": "issue-4891-exemplar-trace.v1",
        "issue": "https://github.com/ll7/robot_sf_ll7/issues/4891",
        "claim_boundary": (
            "exemplar trace episode from retained campaign data; "
            "illustrative head-on corridor interaction only; "
            "no statistical, benchmark, or dissertation claim"
        ),
        "generated_at_utc": pin_generated_at or datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "campaign_id": "issue4206_trace_capable_h600_rerun_20260704",
        "campaign_job": "13334",
        "planner": selection.planner,
        "scenario_id": selection.scenario_id,
        "seed": selection.seed,
        "selection_mode": selection.selection_mode,
        "selection_metric": SELECTION_METRIC,
        "selection_metric_value": selection.metric_value,
        "episode_id": selection.episode_id,
        "episode_status": selection.status,
        # Issue #5141: min_robot_ped_distance_m is computed as math.dist(robot_xy,
        # ped_position), i.e. center-to-center; declared explicitly to prevent a
        # surface-clearance misreading of the exported series.
        "distance_convention": DistanceConvention.CENTER_CENTER.value,
        "summary": derived.summary,
    }

    trace_payload = {
        "schema_version": "issue-4891-trace-series.v1",
        "metadata": metadata,
        "frames": episode_record["algorithm_metadata"]["simulation_step_trace"]["steps"],
        "derived_rows": derived.trace_rows,
    }

    write_json(output_dir / "metadata.json", metadata)
    write_json(output_dir / "trace_series.json", trace_payload)
    write_csv(output_dir / "trace_timeseries.csv", derived.trace_rows)
    write_distance_series_csv(
        output_dir / "min_distance_series.csv",
        derived.min_distance_rows,
        convention=DistanceConvention.CENTER_CENTER,
    )
    _write_readme(output_dir, metadata)
    write_sha256sums(output_dir)
    return metadata


def _write_readme(output_dir: Path, metadata: dict[str, Any]) -> None:
    """Write the human-facing evidence bundle README."""
    date = extract_marker_date(metadata)
    readme = f"""{review_marker("robot_sf#4891", marker_date=date)}
# Issue #4891 Exemplar Trace: {metadata["scenario_id"]} ({metadata["planner"]})

Plain-language summary: this directory contains one exemplar trace episode from the
retained `issue4206_trace_capable_h600_rerun_20260704` campaign (job 13334).
It is an illustrative head-on corridor interaction episode and does not establish a
statistical benchmark or dissertation claim.

## Contents

- `trace_timeseries.csv`: per-timestep robot state, commanded action, executed velocity,
  pedestrian positions, and nearest robot-pedestrian distance.
- `min_distance_series.csv`: figure-ready `(step, time_s, min_robot_ped_distance_m)` series.
  Distance convention: `center_center` (robot-to-pedestrian center distance; footprint radii
  are NOT subtracted). Do not read these values as surface clearance.
- `trace_series.json`: raw recorded frames plus derived rows.
- `metadata.json`: provenance, selection criteria, and claim boundary.
- `SHA256SUMS`: checksums for the files above.

## Provenance

- Campaign: `{metadata["campaign_id"]}`
- Job: `{metadata["campaign_job"]}`
- Planner: `{metadata["planner"]}`
- Scenario: `{metadata["scenario_id"]}`
- Seed: `{metadata["seed"]}`
- Selection mode: `{metadata["selection_mode"]}`
- Selection metric: `{metadata["selection_metric"]} = {metadata["selection_metric_value"]}`
- Git commit at generation: `{metadata["git_commit"]}`

## Claim Boundary

This bundle is `illustrative_exemplar` evidence for one head-on corridor episode.
It should be used for visualization and worked example input only. It is not a full
benchmark campaign, not a Slurm or GPU result, and not a statistical comparison.

<!-- /AI-GENERATED -->
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def write_selection_report(
    all_selections: list[SelectedEpisode],
    output_dir: Path,
    marker_date: str | None = None,
    generated_at: str | None = None,
) -> None:
    """Write the selection report listing all selected episodes.

    ``generated_at`` is the provenance timestamp shared with the bundle
    metadata (pinned via ``--pin-generated-at`` for deterministic re-runs).
    When provided, the report's ``Generated:`` line is byte-stable across
    re-runs; it only falls back to wall-clock for interactive, unpinned runs
    (consistent with the metadata behaviour).
    """
    report_lines = [
        review_marker("robot_sf#4891", marker_date=marker_date),
        "# Issue #4891 Head-On Corridor Exemplar Selection Report",
        "",
        f"Generated: {generated_at or datetime.now(UTC).isoformat()}",
        f"Total exemplars selected: {len(all_selections)}",
        "",
        "## Selection Criteria",
        "",
        "- Scenario class: head-on corridor (low/medium density)",
        f"- Planners: {', '.join(TARGET_PLANNERS)}",
        f"- Selection metric: {SELECTION_METRIC} (higher is better)",
        f"- Selection modes: {', '.join(SELECTION_MODES)}",
        "",
        "## Selected Episodes",
        "",
    ]

    for sel in sorted(
        all_selections, key=lambda s: (s.planner, s.scenario_id, s.seed, s.selection_mode)
    ):
        report_lines.extend(
            [
                f"- **{sel.planner}** / {sel.scenario_id} / seed {sel.seed} / {sel.selection_mode}: "
                f"{SELECTION_METRIC}={sel.metric_value:.4f}",
            ]
        )

    report_lines.extend(
        [
            "",
            "## Scenario Class Rationale",
            "",
            "Head-on corridor scenarios were chosen as the third exemplar class to complement "
            "the doorway scenario (issue_4253/4268) and group-crossing scenario (issue #4848). "
            "Head-on corridor introduces opposing pedestrian flows in a constrained space, creating "
            "rich navigation challenges where the robot must negotiate right-of-way with oncoming "
            "pedestrians. The two density levels (low/medium) capture different interaction "
            "regimes: sparse head-on encounters and moderate corridor crowding.",
            "",
            "## Interaction Diversity",
            "",
            "Head-on corridor provides richer interaction diversity than bottleneck scenarios:",
            "- Consistent pedestrian presence (2-4 pedestrians per episode)",
            "- Opposing flow patterns requiring negotiation",
            "- Constrained space amplifying social navigation challenges",
            "- Mix of success and collision outcomes showing planner sensitivity",
            "",
            "## Planner Rationale",
            "",
            "- **goal**: Classical baseline that ignores pedestrians (provides lower bound)",
            "- **orca**: Classical collision-avoidance with time-based navigation (contrast to learned)",
            "- **social_force**: Physics-based social navigation (explicit social force model)",
            "",
            "These three planners span the classical-to-social spectrum and provide diverse "
            "interaction behaviors for visualization and worked examples.",
            "",
            "<!-- /AI-GENERATED -->",
        ]
    )

    (output_dir / "SELECTION_REPORT.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )


def _process_planner(
    planner: str,
    campaign_root: Path,
    output_dir: Path,
    pin_generated_at: str | None = None,
) -> tuple[list[SelectedEpisode], dict[str, Any] | None]:
    """Process one planner: read episodes, select exemplars, write bundles."""
    if not campaign_root.is_dir():
        raise ExemplarExportInputError(f"campaign root is not a directory: {campaign_root}")

    planner_dir = campaign_root / f"{planner}__differential_drive"
    if not planner_dir.is_dir():
        raise ExemplarExportInputError(f"planner directory is not a directory: {planner_dir}")

    episodes_path = planner_dir / "episodes.jsonl"

    print(f"Reading episodes from {episodes_path}")
    episodes = read_jsonl(episodes_path)
    print(f"Found {len(episodes)} episodes for {planner}")

    selections = select_exemplars_for_planner(episodes, planner)
    print(f"Selected {len(selections)} exemplars for {planner}")

    first_metadata: dict[str, Any] | None = None
    for sel in selections:
        episode_record = _find_episode(episodes, sel)
        if episode_record is None:
            print(f"Warning: could not find episode record for {sel}")
            continue

        bundle_dir = (
            output_dir / sel.planner / f"{sel.scenario_id}_seed{sel.seed}_{sel.selection_mode}"
        )
        bundle_dir.mkdir(parents=True, exist_ok=True)

        print(f"Writing bundle to {bundle_dir}")
        metadata = write_bundle(
            episode_record=episode_record,
            selection=sel,
            output_dir=bundle_dir,
            pin_generated_at=pin_generated_at,
        )
        if first_metadata is None:
            first_metadata = metadata

    return selections, first_metadata


def _find_episode(episodes: list[dict[str, Any]], sel: SelectedEpisode) -> dict[str, Any] | None:
    """Find the full episode record for a selection."""
    for ep in episodes:
        if (
            ep.get("scenario_id") == sel.scenario_id
            and ep.get("seed") == sel.seed
            and ep.get("episode_id") == sel.episode_id
        ):
            return ep
    return None


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-root",
        type=Path,
        default=DEFAULT_CAMPAIGN_ROOT,
        help="Path to campaign runs directory (default: output/issue4206-trace-rerun/13334/runs)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for exemplar bundles",
    )
    parser.add_argument(
        "--planners",
        nargs="+",
        default=TARGET_PLANNERS,
        help=f"Planners to select exemplars from (default: {' '.join(TARGET_PLANNERS)})",
    )
    parser.add_argument(
        "--pin-generated-at",
        default=None,
        help=(
            "Override generated_at_utc in all bundle metadata with this ISO 8601 string. "
            "For deterministic re-runs only; do not use wall-clock time."
        ),
    )
    args = parser.parse_args()

    repo_root = _repo_root()
    campaign_root = args.campaign_root
    if not campaign_root.is_absolute():
        campaign_root = repo_root / campaign_root
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir

    try:
        all_selections: list[SelectedEpisode] = []
        bundle_metadata: dict[str, Any] | None = None
        for planner in args.planners:
            selections, metadata = _process_planner(
                planner,
                campaign_root,
                output_dir,
                pin_generated_at=args.pin_generated_at,
            )
            all_selections.extend(selections)
            if bundle_metadata is None and metadata is not None:
                bundle_metadata = metadata
    except ExemplarExportInputError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    marker_date = extract_marker_date(bundle_metadata) if bundle_metadata else None
    generated_at = bundle_metadata.get("generated_at_utc") if bundle_metadata else None
    write_selection_report(
        all_selections, output_dir, marker_date=marker_date, generated_at=generated_at
    )

    print(f"\nExport complete. {len(all_selections)} exemplar bundles written to {output_dir}")
    print(f"Selection report: {output_dir / 'SELECTION_REPORT.md'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
