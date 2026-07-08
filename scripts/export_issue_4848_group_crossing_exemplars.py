#!/usr/bin/env python3
"""Export exemplar trace-episode bundles for issue #4848 group-crossing scenarios.

Reads retained campaign data from issue4206_trace_capable_h600_rerun_20260704 (job 13334),
selects exemplar episodes (median + best/worst) from group_crossing scenarios for 2-3 planners,
and exports trace-episode bundles in the same format as issue_4253/4268.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Target planners for exemplar selection (classical + social navigation diversity)
TARGET_PLANNERS = ["goal", "orca", "social_force"]

# Scenario class filter
GROUP_CROSSING_SCENARIOS = {
    "classic_group_crossing_low",
    "classic_group_crossing_medium",
    "classic_group_crossing_high",
}

# Selection metric (path_efficiency: higher is better)
SELECTION_METRIC = "path_efficiency"
SELECTION_MODES = ["median", "best", "worst"]

# Output directory
DEFAULT_OUTPUT_DIR = Path("docs/context/evidence/issue_4848_group_crossing_exemplars_2026-07")
DEFAULT_CAMPAIGN_ROOT = Path("output/issue4206-trace-rerun/13334/runs")


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


def sha256_file(path: Path) -> str:
    """Compute a SHA-256 hex digest for ``path``."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write CSV rows with stable column ordering."""
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write deterministic JSON."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL file into a list of dicts."""
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def select_exemplars_for_planner(
    episodes: list[dict[str, Any]], planner: str
) -> list[SelectedEpisode]:
    """Select median, best, worst exemplar episodes for one planner."""
    # Filter for group_crossing scenarios only
    group_crossing_eps = [
        ep for ep in episodes
        if ep.get("scenario_id") in GROUP_CROSSING_SCENARIOS
    ]

    if not group_crossing_eps:
        return []

    # Extract metric values
    scored: list[tuple[float, dict[str, Any]]] = []
    for ep in group_crossing_eps:
        metrics = ep.get("metrics", {})
        val = metrics.get(SELECTION_METRIC)
        if val is not None and isinstance(val, (int, float)):
            if math.isfinite(float(val)):
                scored.append((float(val), ep))

    if not scored:
        return []

    # Sort by metric value (ascending for median index invariance)
    scored.sort(key=lambda x: x[0])

    selected: list[SelectedEpisode] = []
    for mode in SELECTION_MODES:
        if mode == "best":
            idx = len(scored) - 1  # higher is better for path_efficiency
        elif mode == "worst":
            idx = 0  # lower is worse for path_efficiency
        else:  # median
            idx = len(scored) // 2

        val, ep = scored[idx]
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


def write_bundle(
    *,
    episode_record: dict[str, Any],
    selection: SelectedEpisode,
    output_dir: Path,
) -> dict[str, Any]:
    """Write a trace episode bundle for one selected episode."""
    derived = derive_trace_rows(episode_record)

    metadata = {
        "schema_version": "issue-4848-exemplar-trace.v1",
        "issue": "https://github.com/ll7/robot_sf_ll7/issues/4848",
        "claim_boundary": (
            "exemplar trace episode from retained campaign data; "
            "illustrative group-crossing interaction only; "
            "no statistical, benchmark, or dissertation claim"
        ),
        "generated_at_utc": datetime.now(UTC).isoformat(),
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
        "summary": derived.summary,
    }

    trace_payload = {
        "schema_version": "issue-4848-trace-series.v1",
        "metadata": metadata,
        "frames": episode_record["algorithm_metadata"]["simulation_step_trace"]["steps"],
        "derived_rows": derived.trace_rows,
    }

    _write_json(output_dir / "metadata.json", metadata)
    _write_json(output_dir / "trace_series.json", trace_payload)
    _write_csv(output_dir / "trace_timeseries.csv", derived.trace_rows)
    _write_csv(output_dir / "min_distance_series.csv", derived.min_distance_rows)
    _write_readme(output_dir, metadata)
    _write_sha256sums(output_dir)
    return metadata


def _write_readme(output_dir: Path, metadata: dict[str, Any]) -> None:
    """Write the human-facing evidence bundle README."""
    readme = f"""# Issue #4848 Exemplar Trace: {metadata["scenario_id"]} ({metadata["planner"]})

Plain-language summary: this directory contains one exemplar trace episode from the
retained `issue4206_trace_capable_h600_rerun_20260704` campaign (job 13334).
It is an illustrative group-crossing interaction episode and does not establish a
statistical benchmark or dissertation claim.

## Contents

- `trace_timeseries.csv`: per-timestep robot state, commanded action, executed velocity,
  pedestrian positions, and nearest robot-pedestrian distance.
- `min_distance_series.csv`: figure-ready `(step, time_s, min_robot_ped_distance_m)` series.
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

This bundle is `illustrative_exemplar` evidence for one group-crossing episode.
It should be used for visualization and worked example input only. It is not a full
benchmark campaign, not a Slurm or GPU result, and not a statistical comparison.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def _write_sha256sums(output_dir: Path) -> None:
    """Write SHA256SUMS for all generated bundle files except itself."""
    files = sorted(
        path for path in output_dir.iterdir() if path.is_file() and path.name != "SHA256SUMS"
    )
    lines = [f"{sha256_file(path)}  {_manifest_label(path)}" for path in files]
    (output_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _manifest_label(path: Path) -> str:
    """Return the SHA256SUMS entry label for a bundle file (repo-relative when possible)."""
    try:
        return path.resolve().relative_to(_repo_root()).as_posix()
    except ValueError:
        return path.name


def write_selection_report(
    all_selections: list[SelectedEpisode],
    output_dir: Path,
) -> None:
    """Write the selection report listing all selected episodes."""
    report_lines = [
        "# Issue #4848 Group-Crossing Exemplar Selection Report",
        "",
        f"Generated: {datetime.now(UTC).isoformat()}",
        f"Total exemplars selected: {len(all_selections)}",
        "",
        "## Selection Criteria",
        "",
        "- Scenario class: group_crossing (low/medium/high density)",
        f"- Planners: {', '.join(TARGET_PLANNERS)}",
        f"- Selection metric: {SELECTION_METRIC} (higher is better)",
        f"- Selection modes: {', '.join(SELECTION_MODES)}",
        "",
        "## Selected Episodes",
        "",
    ]

    for sel in sorted(all_selections, key=lambda s: (s.planner, s.scenario_id, s.seed, s.selection_mode)):
        report_lines.extend([
            f"- **{sel.planner}** / {sel.scenario_id} / seed {sel.seed} / {sel.selection_mode}: "
            f"{SELECTION_METRIC}={sel.metric_value:.4f}",
        ])

    report_lines.extend([
        "",
        "## Scenario Class Rationale",
        "",
        "Group-crossing scenarios were chosen as the second exemplar class to complement "
        "the doorway scenario from issue_4253/4268. Group-crossing introduces bidirectional "
        "pedestrian flow with social group dynamics (50% of pedestrians in groups), providing "
        "richer interaction diversity than single-agent avoidance scenarios. The three density "
        "levels (low/medium/high) capture different interaction regimes: sparse crossing, "
        "moderate crowd navigation, and high-density stress conditions.",
        "",
        "## Planner Rationale",
        "",
        "- **goal**: Classical baseline that ignores pedestrians (provides lower bound)",
        "- **orca**: Classical collision-avoidance with time-based navigation (contrast to learned)",
        "- **social_force**: Physics-based social navigation (explicit social force model)",
        "",
        "These three planners span the classical-to-social spectrum and provide diverse "
        "interaction behaviors for visualization and worked examples.",
    ])

    (output_dir / "SELECTION_REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def _process_planner(
    planner: str,
    campaign_root: Path,
    output_dir: Path,
) -> list[SelectedEpisode]:
    """Process one planner: read episodes, select exemplars, write bundles."""
    planner_dir = campaign_root / f"{planner}__differential_drive"
    if not planner_dir.exists():
        print(f"Warning: planner directory not found: {planner_dir}")
        return []

    episodes_path = planner_dir / "episodes.jsonl"
    if not episodes_path.exists():
        print(f"Warning: episodes.jsonl not found: {episodes_path}")
        return []

    print(f"Reading episodes from {episodes_path}")
    episodes = read_jsonl(episodes_path)
    print(f"Found {len(episodes)} episodes for {planner}")

    selections = select_exemplars_for_planner(episodes, planner)
    print(f"Selected {len(selections)} exemplars for {planner}")

    for sel in selections:
        episode_record = _find_episode(episodes, sel)
        if episode_record is None:
            print(f"Warning: could not find episode record for {sel}")
            continue

        bundle_dir = (
            output_dir
            / sel.planner
            / f"{sel.scenario_id}_seed{sel.seed}_{sel.selection_mode}"
        )
        bundle_dir.mkdir(parents=True, exist_ok=True)

        print(f"Writing bundle to {bundle_dir}")
        write_bundle(
            episode_record=episode_record,
            selection=sel,
            output_dir=bundle_dir,
        )

    return selections


def _find_episode(
    episodes: list[dict[str, Any]], sel: SelectedEpisode
) -> dict[str, Any] | None:
    """Find the full episode record for a selection."""
    for ep in episodes:
        if (ep.get("scenario_id") == sel.scenario_id
                and ep.get("seed") == sel.seed
                and ep.get("episode_id") == sel.episode_id):
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
    args = parser.parse_args()

    repo_root = _repo_root()
    campaign_root = args.campaign_root
    if not campaign_root.is_absolute():
        campaign_root = repo_root / campaign_root
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir

    if not campaign_root.exists():
        print(f"Error: campaign root not found: {campaign_root}", file=sys.stderr)
        return 1

    all_selections: list[SelectedEpisode] = []
    for planner in args.planners:
        all_selections.extend(_process_planner(planner, campaign_root, output_dir))

    write_selection_report(all_selections, output_dir)

    print(f"\nExport complete. {len(all_selections)} exemplar bundles written to {output_dir}")
    print(f"Selection report: {output_dir / 'SELECTION_REPORT.md'}")

    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main())
