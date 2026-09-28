#!/usr/bin/env python3
"""Measure issue #9429 timeout counts and final-waypoint wall distances.

The frozen 0.0.6 publication bundle does not retain simulation step traces.  This
analyzer therefore reports the trace-dependent goal-adjacent classification as
unavailable instead of inferring it from aggregate episode metrics.  Final
waypoints are reconstructed by running only deterministic environment
initialization at the exact source commit recorded by the bundle; no episode is
stepped or rerun.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import subprocess
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

DEFAULT_TAIL_STEPS = 100
DEFAULT_GOAL_ADJACENT_RADIUS_M = 4.0
DEFAULT_WALL_MARGIN_M = 0.5
PREDICATE_VERSION = "goal_adjacent_timeout.v1"
REPORT_SCHEMA_VERSION = "issue_9429_goal_adjacent_timeout_report.v1"


@dataclass(frozen=True, slots=True)
class GoalAdjacentResult:
    """One episode's trace-dependent goal-adjacent classification."""

    value: bool | None
    reason: str
    min_tail_distance_m: float | None = None
    min_episode_distance_m: float | None = None


@dataclass(frozen=True, slots=True)
class FinalWaypointMeasurement:
    """Frozen-source final-waypoint geometry for one scenario and seed."""

    scenario_id: str
    scenario_family: str
    seed: int
    final_x_m: float
    final_y_m: float
    wall_distance_m: float
    completion_radius_m: float


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_bundle_checksums(bundle_root: Path) -> tuple[int, int]:
    """Verify every payload checksum and return ``(files, bytes)``."""

    checksum_path = bundle_root / "checksums.sha256"
    if not checksum_path.is_file():
        raise FileNotFoundError(f"Missing bundle checksum manifest: {checksum_path}")
    files = 0
    total_bytes = 0
    for line_number, raw_line in enumerate(
        checksum_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        try:
            expected, relative = line.split(maxsplit=1)
        except ValueError as exc:
            raise ValueError(f"Malformed checksum line {line_number}: {raw_line!r}") from exc
        relative = relative.lstrip("* ")
        target = bundle_root / relative
        if not target.is_file():
            raise FileNotFoundError(f"Checksum target is missing: {target}")
        observed = _sha256(target)
        if observed != expected:
            raise ValueError(
                f"Checksum mismatch for {relative}: expected {expected}, observed {observed}"
            )
        files += 1
        total_bytes += target.stat().st_size
    return files, total_bytes


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            payload = json.loads(raw_line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(payload)
    return rows


def _scenario_id(row: Mapping[str, Any]) -> str:
    value = row.get("scenario_id")
    if not isinstance(value, str) or not value:
        raise ValueError("Episode row is missing scenario_id")
    return value


def _scenario_family(row: Mapping[str, Any]) -> str:
    params = row.get("scenario_params")
    metadata = params.get("metadata") if isinstance(params, Mapping) else None
    family = metadata.get("archetype") if isinstance(metadata, Mapping) else None
    if not isinstance(family, str) or not family.strip():
        raise ValueError(f"Episode {_scenario_id(row)!r} is missing metadata.archetype")
    return family.strip()


def is_noncollision_timeout(row: Mapping[str, Any]) -> bool:
    """Return the exact issue #9429 timeout base predicate."""

    outcome = row.get("outcome")
    return bool(
        isinstance(outcome, Mapping)
        and outcome.get("timeout_event") is True
        and outcome.get("collision_event") is False
    )


def _trace_robot_positions(row: Mapping[str, Any]) -> tuple[list[tuple[float, float]], str]:
    metadata = row.get("algorithm_metadata")
    trace = metadata.get("simulation_step_trace") if isinstance(metadata, Mapping) else None
    steps = trace.get("steps") if isinstance(trace, Mapping) else None
    if not isinstance(steps, list) or not steps:
        return [], "missing_simulation_step_trace"
    positions: list[tuple[float, float]] = []
    for step_index, step in enumerate(steps):
        robot = step.get("robot") if isinstance(step, Mapping) else None
        position = robot.get("position") if isinstance(robot, Mapping) else None
        if not isinstance(position, list) or len(position) != 2:
            return [], f"malformed_robot_position_at_step_{step_index}"
        try:
            x_m, y_m = float(position[0]), float(position[1])
        except (TypeError, ValueError):
            return [], f"malformed_robot_position_at_step_{step_index}"
        if not math.isfinite(x_m) or not math.isfinite(y_m):
            return [], f"nonfinite_robot_position_at_step_{step_index}"
        positions.append((x_m, y_m))
    return positions, "available"


def classify_goal_adjacent_timeout(
    row: Mapping[str, Any],
    *,
    final_waypoint: tuple[float, float],
    completion_radius_m: float,
    tail_steps: int = DEFAULT_TAIL_STEPS,
    goal_adjacent_radius_m: float = DEFAULT_GOAL_ADJACENT_RADIUS_M,
) -> GoalAdjacentResult:
    """Apply ``goal_adjacent_timeout.v1`` to one episode row.

    The predicate is::

        timeout_event is true
        AND collision_event is false
        AND min(distance(robot_position[t], final_waypoint)
                for t in the final N recorded steps) < goal_adjacent_radius_m
        AND min(distance(robot_position[t], final_waypoint)
                for every recorded episode step) > completion_radius_m

    The strict final comparison mirrors runtime completion, which succeeds at
    distance ``<= completion_radius_m``.  Missing or shorter-than-N traces are
    unavailable, not false.
    """

    if not is_noncollision_timeout(row):
        return GoalAdjacentResult(False, "not_noncollision_timeout")
    if tail_steps <= 0:
        raise ValueError("tail_steps must be positive")
    if not math.isfinite(completion_radius_m) or completion_radius_m <= 0.0:
        raise ValueError("completion_radius_m must be positive and finite")
    if not math.isfinite(goal_adjacent_radius_m) or goal_adjacent_radius_m <= 0.0:
        raise ValueError("goal_adjacent_radius_m must be positive and finite")
    positions, trace_reason = _trace_robot_positions(row)
    if trace_reason != "available":
        return GoalAdjacentResult(None, trace_reason)
    if len(positions) < tail_steps:
        return GoalAdjacentResult(None, f"trace_shorter_than_{tail_steps}_steps")
    goal_x, goal_y = final_waypoint
    distances = [math.hypot(x_m - goal_x, y_m - goal_y) for x_m, y_m in positions]
    min_tail = min(distances[-tail_steps:])
    min_episode = min(distances)
    return GoalAdjacentResult(
        min_tail < goal_adjacent_radius_m and min_episode > completion_radius_m,
        "classified",
        min_tail_distance_m=min_tail,
        min_episode_distance_m=min_episode,
    )


def _git_head(source_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _ensure_frozen_runtime_import(source_root: Path, expected_commit: str) -> None:
    observed_commit = _git_head(source_root)
    if observed_commit != expected_commit:
        raise ValueError(
            f"Frozen source mismatch: bundle records {expected_commit}, checkout is {observed_commit}"
        )
    sys.path.insert(0, str(source_root))
    import robot_sf

    module_path = Path(robot_sf.__file__).resolve()
    if not module_path.is_relative_to(source_root.resolve()):
        raise RuntimeError(
            f"robot_sf imported from {module_path}, outside frozen source root {source_root}"
        )


def _static_wall_distance_m(map_def: Any, point_xy: tuple[float, float]) -> float:
    from shapely.geometry import LineString, Point

    point = Point(point_xy)
    geometries: list[Any] = []
    for obstacle in getattr(map_def, "obstacles", ()):
        geometries.extend(obstacle.iter_polygons())
    for x_start, x_end, y_start, y_end in getattr(map_def, "bounds", ()):
        geometries.append(LineString([(x_start, y_start), (x_end, y_end)]))
    if not geometries:
        raise ValueError("Map has no static obstacle or boundary geometry")
    distance = min(float(point.distance(geometry)) for geometry in geometries)
    if not math.isfinite(distance) or distance < 0.0:
        raise ValueError(f"Invalid final-waypoint wall distance: {distance}")
    return distance


def measure_final_waypoints(
    *,
    source_root: Path,
    scenario_manifest: Path,
    expected_commit: str,
    episode_identities: set[tuple[str, int]],
) -> list[FinalWaypointMeasurement]:
    """Reconstruct final waypoints through frozen-source environment initialization."""

    _ensure_frozen_runtime_import(source_root, expected_commit)
    from loguru import logger

    from robot_sf.benchmark.map_runner.map_runner_env import build_env_config
    from robot_sf.benchmark.map_runner.map_runner_identity import (
        scenario_with_episode_seed_defaults,
    )
    from robot_sf.gym_env.environment_factory import make_robot_env
    from robot_sf.training.scenario_loader import load_scenarios

    logger.remove()
    manifest_path = scenario_manifest
    if not manifest_path.is_absolute():
        manifest_path = source_root / manifest_path
    manifest_path = manifest_path.resolve()
    measurements: list[FinalWaypointMeasurement] = []
    with contextlib.chdir(source_root):
        scenarios = [dict(scenario) for scenario in load_scenarios(manifest_path)]
        by_id = {
            str(scenario.get("name") or scenario.get("scenario_id")): scenario
            for scenario in scenarios
        }
        missing = sorted({scenario_id for scenario_id, _ in episode_identities} - set(by_id))
        if missing:
            raise ValueError(f"Frozen scenario manifest is missing scenario ids: {missing}")
        for scenario_id, seed in sorted(episode_identities):
            scenario = scenario_with_episode_seed_defaults(by_id[scenario_id], seed=seed)
            config = build_env_config(scenario, scenario_path=manifest_path)
            env = make_robot_env(config=config, seed=seed, debug=False)
            try:
                env.reset(seed=seed)
                navigator = env.simulator.robot_navs[0]
                if not navigator.waypoints:
                    raise ValueError(f"{scenario_id} seed {seed} has no robot waypoints")
                final_x_m, final_y_m = map(float, navigator.waypoints[-1])
                completion_radius_m = float(env.simulator.goal_proximity_threshold)
                family = str(scenario.get("metadata", {}).get("archetype") or "").strip()
                if not family:
                    raise ValueError(f"{scenario_id} is missing metadata.archetype")
                measurements.append(
                    FinalWaypointMeasurement(
                        scenario_id=scenario_id,
                        scenario_family=family,
                        seed=seed,
                        final_x_m=final_x_m,
                        final_y_m=final_y_m,
                        wall_distance_m=_static_wall_distance_m(
                            env.simulator.map_def, (final_x_m, final_y_m)
                        ),
                        completion_radius_m=completion_radius_m,
                    )
                )
            finally:
                env.close()
    return measurements


def _quantile(values: Sequence[float], probability: float) -> float:
    """Return a linearly interpolated inclusive quantile."""

    if not values:
        raise ValueError("Cannot compute a quantile from an empty sequence")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize_wall_distances(
    measurements: Iterable[FinalWaypointMeasurement], *, wall_margin_m: float
) -> list[dict[str, Any]]:
    """Summarize final-waypoint wall distances by scenario family and overall."""

    grouped: dict[str, list[FinalWaypointMeasurement]] = defaultdict(list)
    all_rows = list(measurements)
    for row in all_rows:
        grouped[row.scenario_family].append(row)
    grouped["ALL"] = all_rows
    summaries: list[dict[str, Any]] = []
    for family in sorted(grouped, key=lambda value: (value == "ALL", value)):
        rows = grouped[family]
        distances = [row.wall_distance_m for row in rows]
        thresholds = [row.completion_radius_m + wall_margin_m for row in rows]
        within = sum(
            distance <= threshold for distance, threshold in zip(distances, thresholds, strict=True)
        )
        unique_completion = sorted({row.completion_radius_m for row in rows})
        summaries.append(
            {
                "scenario_family": family,
                "final_waypoints": len(rows),
                "completion_radius_m": unique_completion[0]
                if len(unique_completion) == 1
                else None,
                "wall_threshold_m": (
                    unique_completion[0] + wall_margin_m if len(unique_completion) == 1 else None
                ),
                "minimum_m": min(distances),
                "p25_m": _quantile(distances, 0.25),
                "median_m": median(distances),
                "p75_m": _quantile(distances, 0.75),
                "p95_m": _quantile(distances, 0.95),
                "maximum_m": max(distances),
                "within_threshold": within,
                "within_threshold_fraction": within / len(rows),
            }
        )
    return summaries


def _format_number(value: Any, digits: int = 6) -> str:
    if value is None:
        return "NA"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _markdown_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return lines


def _arm_family_rows(
    episodes_by_arm: Mapping[str, list[dict[str, Any]]],
    waypoint_lookup: Mapping[tuple[str, int], FinalWaypointMeasurement],
    *,
    tail_steps: int,
    goal_adjacent_radius_m: float,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    unavailable_reasons: dict[str, int] = defaultdict(int)
    for arm, episodes in sorted(episodes_by_arm.items()):
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for episode in episodes:
            grouped[_scenario_family(episode)].append(episode)
        for family, family_episodes in sorted(grouped.items()):
            timeouts = [episode for episode in family_episodes if is_noncollision_timeout(episode)]
            classifications: list[GoalAdjacentResult] = []
            for episode in timeouts:
                key = (_scenario_id(episode), int(episode["seed"]))
                waypoint = waypoint_lookup[key]
                result = classify_goal_adjacent_timeout(
                    episode,
                    final_waypoint=(waypoint.final_x_m, waypoint.final_y_m),
                    completion_radius_m=waypoint.completion_radius_m,
                    tail_steps=tail_steps,
                    goal_adjacent_radius_m=goal_adjacent_radius_m,
                )
                classifications.append(result)
                if result.value is None:
                    unavailable_reasons[result.reason] += 1
            classified = [result.value for result in classifications if result.value is not None]
            complete_coverage = len(classified) == len(timeouts)
            goal_adjacent = sum(bool(value) for value in classified) if complete_coverage else None
            rows.append(
                {
                    "arm": arm,
                    "scenario_family": family,
                    "episodes": len(family_episodes),
                    "timeouts": len(timeouts),
                    "trace_classified_timeouts": len(classified),
                    "goal_adjacent_timeouts": goal_adjacent,
                    "goal_adjacent_fraction_of_timeouts": (
                        goal_adjacent / len(timeouts)
                        if complete_coverage and timeouts and goal_adjacent is not None
                        else (0.0 if complete_coverage and not timeouts else None)
                    ),
                }
            )
    return rows, dict(sorted(unavailable_reasons.items()))


def _render_report(  # noqa: PLR0913
    *,
    bundle_name: str,
    campaign_id: str,
    source_commit: str,
    scenario_matrix_hash: str,
    config_hash: str,
    checksum_files: int,
    checksum_bytes: int,
    tail_steps: int,
    goal_adjacent_radius_m: float,
    wall_margin_m: float,
    arm_family_rows: list[dict[str, Any]],
    unavailable_reasons: Mapping[str, int],
    wall_summaries: list[dict[str, Any]],
    source_manifest: str,
) -> str:
    total_episodes = sum(row["episodes"] for row in arm_family_rows)
    total_timeouts = sum(row["timeouts"] for row in arm_family_rows)
    total_classified = sum(row["trace_classified_timeouts"] for row in arm_family_rows)
    arm_totals: dict[str, dict[str, int]] = defaultdict(
        lambda: {"episodes": 0, "timeouts": 0, "classified": 0, "goal_adjacent": 0}
    )
    for row in arm_family_rows:
        totals = arm_totals[row["arm"]]
        totals["episodes"] += row["episodes"]
        totals["timeouts"] += row["timeouts"]
        totals["classified"] += row["trace_classified_timeouts"]
        if row["goal_adjacent_timeouts"] is not None:
            totals["goal_adjacent"] += row["goal_adjacent_timeouts"]
    lines = [
        "# Frozen 0.0.6 S30/H600 goal-adjacent timeout counts",
        "",
        "Claim boundary: descriptive analysis of the frozen release bundle only. Timeout counts "
        "come from retained episode outcomes. Final-waypoint wall distances come from deterministic "
        "environment initialization at the bundle's exact source commit. No episode was stepped or "
        "rerun, and no runtime or frozen artifact was changed.",
        "",
        "Evidence status: timeout and waypoint-geometry counts are available. The goal-adjacent "
        f"predicate is unavailable for all {total_classified:,}/{total_timeouts:,} classified/timeout "
        "rows because the bundle did not retain simulation step traces. `NA` is not zero.",
        "",
        "## Input identity",
        "",
        f"- Bundle: `{bundle_name}`",
        f"- Campaign: `{campaign_id}`",
        f"- Source commit: `{source_commit}`",
        f"- Campaign config hash: `{config_hash}`",
        f"- Scenario matrix hash: `{scenario_matrix_hash}`",
        f"- Scenario manifest: `{source_manifest}`",
        f"- Checksum verification: {checksum_files} files, {checksum_bytes} bytes",
        f"- Episode rows read: {total_episodes}",
        "",
        "## Exact predicate",
        "",
        f"Predicate `{PREDICATE_VERSION}` was configured with `N={tail_steps}`, "
        f"`goal_adjacent_radius_m={goal_adjacent_radius_m:.1f}`, and each episode's runtime "
        "completion radius (`robot radius + goal radius`, 2.0 m in this bundle):",
        "",
        "```text",
        "outcome.timeout_event is true",
        "AND outcome.collision_event is false",
        f"AND min(distance(robot_position[t], final_waypoint) for t in final {tail_steps} recorded steps) < {goal_adjacent_radius_m:.1f} m",
        "AND min(distance(robot_position[t], final_waypoint) for t in every recorded episode step) > completion_radius_m",
        "```",
        "",
        "The last comparison is strict because runtime completion is true at distance "
        "`<= completion_radius_m`. A timeout with a missing, malformed, non-finite, or shorter-than-N "
        "simulation step trace is unavailable rather than false.",
        "",
        "Applied result: the frozen episode rows set "
        "`scenario_params.record_simulation_step_trace=false` and contain no "
        "`algorithm_metadata.simulation_step_trace.steps`; therefore the trace-dependent clauses "
        "were not evaluable.",
        "",
        "Unavailable classification reasons:",
        "",
    ]
    lines.extend(f"- `{reason}`: {count}" for reason, count in unavailable_reasons.items())
    lines.extend(
        [
            "",
            "## Counts by arm",
            "",
            *_markdown_table(
                [
                    "arm",
                    "episodes",
                    "timeouts",
                    "trace-classified timeouts",
                    "goal-adjacent timeouts",
                    "goal-adjacent / timeouts",
                ],
                (
                    (
                        arm,
                        totals["episodes"],
                        totals["timeouts"],
                        totals["classified"],
                        (
                            totals["goal_adjacent"]
                            if totals["classified"] == totals["timeouts"]
                            else "NA"
                        ),
                        (
                            _format_number(totals["goal_adjacent"] / totals["timeouts"])
                            if totals["classified"] == totals["timeouts"] and totals["timeouts"]
                            else (
                                "0.000000" if totals["classified"] == totals["timeouts"] else "NA"
                            )
                        ),
                    )
                    for arm, totals in sorted(arm_totals.items())
                ),
            ),
            "",
            "## Counts by arm and scenario family",
            "",
            *_markdown_table(
                [
                    "arm",
                    "scenario family",
                    "episodes",
                    "timeouts",
                    "trace-classified timeouts",
                    "goal-adjacent timeouts",
                    "goal-adjacent / timeouts",
                ],
                (
                    (
                        row["arm"],
                        row["scenario_family"],
                        row["episodes"],
                        row["timeouts"],
                        row["trace_classified_timeouts"],
                        _format_number(row["goal_adjacent_timeouts"]),
                        _format_number(row["goal_adjacent_fraction_of_timeouts"]),
                    )
                    for row in arm_family_rows
                ),
            ),
            "",
            "## Final-waypoint distance to the nearest static wall",
            "",
            "One final waypoint was reconstructed for every scenario/seed pair. All arms share the "
            "same scenario-seed initialization, so the geometry denominator is 48 scenarios × 30 "
            "seeds = 1,440 waypoints, not 20,160 arm-episode rows. The distance is the Shapely point "
            "distance to the nearest parsed static obstacle polygon or map-boundary segment under "
            "the legacy SVG geometry contract at the frozen source commit.",
            "",
            f"`within threshold` means wall distance `<= completion_radius_m + {wall_margin_m:.1f} m`.",
            "",
            *_markdown_table(
                [
                    "scenario family",
                    "waypoints",
                    "completion radius m",
                    "threshold m",
                    "min m",
                    "p25 m",
                    "median m",
                    "p75 m",
                    "p95 m",
                    "max m",
                    "within threshold",
                    "fraction",
                ],
                (
                    (
                        row["scenario_family"],
                        row["final_waypoints"],
                        _format_number(row["completion_radius_m"]),
                        _format_number(row["wall_threshold_m"]),
                        _format_number(row["minimum_m"]),
                        _format_number(row["p25_m"]),
                        _format_number(row["median_m"]),
                        _format_number(row["p75_m"]),
                        _format_number(row["p95_m"]),
                        _format_number(row["maximum_m"]),
                        row["within_threshold"],
                        _format_number(row["within_threshold_fraction"]),
                    )
                    for row in wall_summaries
                ),
            ),
            "",
            "## Reproduction",
            "",
            "Run the committed analyzer from any checkout while pointing `--source-root` at an "
            "unchanged checkout of the source commit above:",
            "",
            "```bash",
            "uv run python scripts/analysis/analyze_goal_adjacent_timeouts_issue_9429.py \\",
            "  --bundle-root /path/to/benchmark_0_0_6_s30_h600_20260911_publication_bundle \\",
            "  --source-root /path/to/robot_sf_ll7-at-31cdfe03 \\",
            "  --output docs/analysis/issue_9429_goal_adjacent_timeouts_0_0_6.md",
            "```",
            "",
            f"Report schema: `{REPORT_SCHEMA_VERSION}`.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_report(args: argparse.Namespace) -> str:
    """Build the complete Markdown report from validated frozen inputs."""
    bundle_root = args.bundle_root.resolve()
    publication_manifest = _read_json(bundle_root / "publication_manifest.json")
    payload_manifest = _read_json(bundle_root / "payload/manifest.json")
    campaign_manifest = _read_json(bundle_root / "payload/campaign_manifest.json")
    checksum_files, checksum_bytes = verify_bundle_checksums(bundle_root)
    declared_totals = publication_manifest.get("totals", {})
    if checksum_files != int(declared_totals.get("file_count", -1)):
        raise ValueError("Checksum file count does not match publication_manifest.json")
    if checksum_bytes != int(declared_totals.get("total_bytes", -1)):
        raise ValueError("Checksum byte count does not match publication_manifest.json")

    runs_root = bundle_root / "payload/runs"
    episode_paths = sorted(runs_root.glob("*/episodes.jsonl"))
    if not episode_paths:
        raise FileNotFoundError(f"No episode files found under {runs_root}")
    episodes_by_arm = {path.parent.name: _load_jsonl(path) for path in episode_paths}
    identity_sets = {
        arm: {(_scenario_id(row), int(row["seed"])) for row in rows}
        for arm, rows in episodes_by_arm.items()
    }
    first_arm = next(iter(identity_sets))
    expected_identities = identity_sets[first_arm]
    mismatched = [
        arm for arm, identities in identity_sets.items() if identities != expected_identities
    ]
    if mismatched:
        raise ValueError(f"Arms do not share the same scenario/seed identities: {mismatched}")

    source_commit = str(payload_manifest["git_hash"])
    waypoints = measure_final_waypoints(
        source_root=args.source_root.resolve(),
        scenario_manifest=args.scenario_manifest,
        expected_commit=source_commit,
        episode_identities=expected_identities,
    )
    waypoint_lookup = {(row.scenario_id, row.seed): row for row in waypoints}
    arm_rows, unavailable_reasons = _arm_family_rows(
        episodes_by_arm,
        waypoint_lookup,
        tail_steps=args.tail_steps,
        goal_adjacent_radius_m=args.goal_adjacent_radius_m,
    )
    wall_summaries = summarize_wall_distances(waypoints, wall_margin_m=args.wall_margin_m)
    return _render_report(
        bundle_name=str(publication_manifest["bundle_name"]),
        campaign_id=str(campaign_manifest["campaign_id"]),
        source_commit=source_commit,
        scenario_matrix_hash=str(payload_manifest["scenario_matrix_hash"]),
        config_hash=str(campaign_manifest["config_hash"]),
        checksum_files=checksum_files,
        checksum_bytes=checksum_bytes,
        tail_steps=args.tail_steps,
        goal_adjacent_radius_m=args.goal_adjacent_radius_m,
        wall_margin_m=args.wall_margin_m,
        arm_family_rows=arm_rows,
        unavailable_reasons=unavailable_reasons,
        wall_summaries=wall_summaries,
        source_manifest=str(args.scenario_manifest),
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--scenario-manifest",
        type=Path,
        default=Path("configs/scenarios/classic_interactions_francis2023.yaml"),
    )
    parser.add_argument("--tail-steps", type=int, default=DEFAULT_TAIL_STEPS)
    parser.add_argument(
        "--goal-adjacent-radius-m", type=float, default=DEFAULT_GOAL_ADJACENT_RADIUS_M
    )
    parser.add_argument("--wall-margin-m", type=float, default=DEFAULT_WALL_MARGIN_M)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Write the requested report and return a process exit code."""
    args = _parse_args(argv)
    report = build_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
