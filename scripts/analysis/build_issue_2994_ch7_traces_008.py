#!/usr/bin/env python3
"""Build Chapter 7 worked-example trace bundles for ll7/diss#2994.

The script reads published 0.0.8 per-episode rows for selection rates, reads a
development-seed Slurm trace sweep, selects shared dev-seed subsets whose planner
outcome marginals match the published cells, and exports trace bundles in the
same file layout used by the older issue #9431 worked examples.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from robot_sf.evidence.distance_convention import DistanceConvention
from robot_sf.evidence.writers import (
    extract_marker_date,
    review_marker,
    sha256_file,
    write_csv,
    write_distance_series_csv,
    write_json,
    write_sha256sums,
    write_text,
)

SOURCE_COMMIT = "373dbfde4f39667cf9e8732dabe7df5118bdeab1"
RELEASE_TAG = f"paper-matrix-v2-h600-s30-0.0.8-{SOURCE_COMMIT}"
DOI = "10.5281/zenodo.23150472"
SCENARIO_CONFIG = "configs/scenarios/classic_interactions_francis2023_release_0_0_8_v1.yaml"
RUN_CONFIG = "issue_2994_ch7_trace_worked_examples_008_dev_sweep.yaml"
ISSUE_URL = "https://github.com/ll7/diss/issues/2994"
DEFAULT_PUBLISHED_BUNDLE = Path(
    "/home/luttkule/lanes/rel008/evidence-round98/main-snapshot/"
    "issue9850_release_snqi_v2_0_0_8_373dbfde4f_r39_20261008_publication_bundle"
)
DEFAULT_DEV_CAMPAIGN_ROOT = Path(
    "/home/luttkule/lanes/traces008/run/output/issue2994_ch7_trace_worked_examples_008_dev_sweep_v2"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/luttkule/lanes/traces008/artifacts/issue_2994_ch7_trace_worked_examples_0_0_8_373dbfde4f"
)

PLANNER_RUNS = {
    "goal": "goal__differential_drive",
    "orca": "orca__differential_drive",
    "social_force": "social_force__differential_drive",
}


@dataclass(frozen=True)
class FigureCell:
    """One Chapter 7 worked-example figure cell to select and export."""

    key: str
    label: str
    scenario_id: str
    planners: tuple[str, str]
    rationale: str
    proposed_pin_key: str


FIGURE_CELLS = (
    FigureCell(
        key="head_on_orca_vs_socialforce",
        label="head-on corridor ORCA vs social force",
        scenario_id="classic_head_on_corridor_medium",
        planners=("orca", "social_force"),
        rationale=(
            "keeps the Figure 7.6 cell but changes the contrast: in 0.0.8 "
            "social force usually completes, so the figure should show timing, "
            "clearance, or the rare non-completion rather than the old stall story"
        ),
        proposed_pin_key="headon_orca_vs_socialforce",
    ),
    FigureCell(
        key="group_crossing_goal_vs_socialforce",
        label="group crossing goal vs social force",
        scenario_id="classic_group_crossing_medium",
        planners=("goal", "social_force"),
        rationale=(
            "medium group crossing preserves a clean baseline-vs-social-force "
            "difference: goal retains collision outcomes while social force completes"
        ),
        proposed_pin_key="group_crossing_goal_vs_socialforce",
    ),
    FigureCell(
        key="speed_clearance_phase_portrait",
        label="speed-clearance phase portrait",
        scenario_id="classic_urban_crossing_medium",
        planners=("orca", "social_force"),
        rationale=(
            "urban crossing is hard for social force under 0.0.8 and exposes a "
            "useful speed-clearance tradeoff against ORCA"
        ),
        proposed_pin_key="speed_clearance_phase_portrait",
    ),
    FigureCell(
        key="ego_occupancy",
        label="ego occupancy",
        scenario_id="classic_doorway_medium",
        planners=("goal", "social_force"),
        rationale=(
            "doorway geometry gives a compact occupancy example: goal collides "
            "often while social force usually reaches the goal zone"
        ),
        proposed_pin_key="ego_occupancy",
    ),
    FigureCell(
        key="dual_panel_interaction",
        label="dual-panel interaction",
        scenario_id="francis2023_robot_crowding",
        planners=("orca", "social_force"),
        rationale=(
            "robot crowding is deliberately hard and seed-sensitive: ORCA has "
            "mixed outcomes while social force mostly fails to finish"
        ),
        proposed_pin_key="dual_panel_interaction",
    ),
)

EXTRA_CANDIDATES = (
    ("classic_head_on_corridor_low", ("orca", "social_force")),
    ("classic_group_crossing_high", ("goal", "social_force")),
    ("francis2023_crowd_navigation", ("orca", "social_force")),
    ("classic_bottleneck_medium", ("orca", "social_force")),
    ("classic_cross_trap_medium", ("orca", "social_force")),
)


@dataclass(frozen=True)
class TraceRows:
    """Derived plotting rows and summary statistics for one recorded episode."""

    trace_rows: list[dict[str, Any]]
    min_distance_rows: list[dict[str, Any]]
    summary: dict[str, Any]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read non-empty JSONL episode rows from ``path``."""

    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"no rows found in {path}")
    return rows


def load_rows(root: Path, planner: str) -> list[dict[str, Any]]:
    """Load published 0.0.8 rows for one planner."""

    return read_jsonl(root / "payload" / "runs" / PLANNER_RUNS[planner] / "episodes.jsonl")


def load_dev_rows(root: Path, planner: str) -> list[dict[str, Any]]:
    """Load development trace-sweep rows for one planner."""

    return read_jsonl(root / "runs" / PLANNER_RUNS[planner] / "episodes.jsonl")


def outcome_label(row: dict[str, Any]) -> str:
    """Classify an episode row into the terminal-outcome bucket used for matching."""

    metrics = row.get("metrics") or {}
    outcome = row.get("outcome") or {}
    if bool(metrics.get("success")):
        return "success"
    if bool(outcome.get("collision_event")) or float(metrics.get("total_collision_count") or 0) > 0:
        return "collision"
    if bool(outcome.get("timeout_event")) or row.get("termination_reason") == "timeout":
        return "timeout"
    if bool(outcome.get("route_complete")):
        return "route_complete_non_success"
    return "other"


def outcome_counts(rows: list[dict[str, Any]]) -> Counter[str]:
    """Count terminal-outcome labels over episode rows."""

    return Counter(outcome_label(row) for row in rows)


def _counts_vector(counter: Counter[str], labels: tuple[str, ...]) -> tuple[int, ...]:
    return tuple(counter.get(label, 0) for label in labels)


def _nearest_bucket_subset(
    buckets: dict[tuple[str, ...], list[int]],
    label_space: tuple[str, ...],
    target_vectors: tuple[tuple[int, ...], ...],
    subset_size: int,
) -> list[int]:
    """Minimize summed absolute marginal-count error using integer bucket counts."""

    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp

    categories = sorted(buckets)
    bucket_count = len(categories)
    targets = np.asarray(target_vectors, dtype=float).ravel()
    marginal_count = len(targets)
    coefficients = np.zeros((marginal_count, bucket_count))
    for bucket_i, labels in enumerate(categories):
        for planner_i, label in enumerate(labels):
            coefficients[planner_i * len(label_space) + label_space.index(label), bucket_i] = 1

    # Each deviation bounds both signs of (selected count - target count).
    constraints = np.zeros((1 + 2 * marginal_count, bucket_count + marginal_count))
    constraints[0, :bucket_count] = 1
    constraints[1 : 1 + marginal_count, :bucket_count] = coefficients
    constraints[1 : 1 + marginal_count, bucket_count:] = -np.eye(marginal_count)
    constraints[1 + marginal_count :, :bucket_count] = coefficients
    constraints[1 + marginal_count :, bucket_count:] = np.eye(marginal_count)
    result = milp(
        c=np.concatenate((np.zeros(bucket_count), np.ones(marginal_count))),
        integrality=np.concatenate((np.ones(bucket_count), np.zeros(marginal_count))),
        bounds=Bounds(
            np.zeros(bucket_count + marginal_count),
            [*(len(buckets[labels]) for labels in categories), *([np.inf] * marginal_count)],
        ),
        constraints=LinearConstraint(
            constraints,
            [subset_size, *([-np.inf] * marginal_count), *targets],
            [subset_size, *targets, *([np.inf] * marginal_count)],
        ),
        options={"mip_rel_gap": 0.0},
    )
    if not result.success or result.x is None:
        raise RuntimeError("nearest development-seed subset optimization failed")
    selected = sorted(
        seed
        for labels, count in zip(categories, result.x[:bucket_count], strict=True)
        for seed in buckets[labels][: round(count)]
    )
    if len(selected) != subset_size:
        raise RuntimeError("nearest development-seed subset has an invalid size")
    return selected


def select_shared_subset(  # noqa: C901 - small dynamic program over paired outcome buckets.
    rows_by_planner: dict[str, list[dict[str, Any]]],
    target_counts_by_planner: dict[str, Counter[str]],
    *,
    subset_size: int = 30,
) -> tuple[list[int], str]:
    """Select shared seeds whose planner outcome marginals match target counts.

    Returns ``(seeds, status)`` where status is ``exact`` or ``nearest``.
    Nearest minimizes the sum of absolute count errors across all planner/label
    marginals, including labels absent from the targets. Within each selected
    outcome bucket, the lowest seeds are used. Rejects candidate seeds outside
    1001..1200 and pools too small to supply the requested shared subset.
    """

    planners = tuple(rows_by_planner)
    rows_by_seed: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for planner, rows in rows_by_planner.items():
        for row in rows:
            rows_by_seed[int(row["seed"])][planner] = row
    complete_seeds = sorted(
        seed
        for seed, planner_rows in rows_by_seed.items()
        if all(p in planner_rows for p in planners)
    )
    # Every selected seed comes from this pool; enforce the policy on both paths.
    if any(not 1001 <= seed <= 1200 for seed in complete_seeds):
        raise ValueError("candidate seeds must be in the development range 1001..1200")
    if len(complete_seeds) < subset_size:
        raise ValueError("not enough shared development seeds for the requested subset")

    buckets: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for seed in complete_seeds:
        labels = tuple(outcome_label(rows_by_seed[seed][planner]) for planner in planners)
        buckets[labels].append(seed)

    label_space = tuple(
        sorted(
            {label for counter in target_counts_by_planner.values() for label in counter}
            | {label for labels in buckets for label in labels}
        )
    )
    target_vectors = tuple(
        _counts_vector(target_counts_by_planner[p], label_space) for p in planners
    )
    categories = sorted(buckets)
    solution: list[int] | None = None

    def rec(
        index: int,
        chosen_count: int,
        counts: tuple[tuple[int, ...], ...],
        picks: list[tuple[tuple[str, ...], int]],
    ) -> bool:
        nonlocal solution
        if chosen_count > subset_size:
            return False
        if any(
            counts[p_i][label_i] > target_vectors[p_i][label_i]
            for p_i in range(len(planners))
            for label_i in range(len(label_space))
        ):
            return False
        if index == len(categories):
            if chosen_count != subset_size or counts != target_vectors:
                return False
            selected: list[int] = []
            for labels, take in picks:
                selected.extend(buckets[labels][:take])
            solution = sorted(selected)
            return True

        labels = categories[index]
        max_take = min(len(buckets[labels]), subset_size - chosen_count)
        for take in range(max_take, -1, -1):
            next_counts = [list(values) for values in counts]
            for planner_i, label in enumerate(labels):
                next_counts[planner_i][label_space.index(label)] += take
            if rec(
                index + 1,
                chosen_count + take,
                tuple(tuple(values) for values in next_counts),
                [*picks, (labels, take)] if take else picks,
            ):
                return True
        return False

    empty = tuple(tuple(0 for _ in label_space) for _ in planners)
    if rec(0, 0, empty, []):
        assert solution is not None
        return solution, "exact"

    return _nearest_bucket_subset(buckets, label_space, target_vectors, subset_size), "nearest"


def _min_distance(
    robot_xy: list[float], pedestrians: list[dict[str, Any]]
) -> tuple[float | None, str | None]:
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
    """Convert a recorded episode row into CSV-ready trace and distance rows."""

    trace = record.get("algorithm_metadata", {}).get("simulation_step_trace")
    if not isinstance(trace, dict) or "steps" not in trace:
        raise ValueError("episode row lacks algorithm_metadata.simulation_step_trace.steps")

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
        pedestrians = frame.get("pedestrians", [])
        min_distance, nearest_pedestrian_id = _min_distance(robot_xy, pedestrians)
        step = int(frame["step"])

        if min_distance is not None and (
            global_min_distance is None or min_distance < global_min_distance
        ):
            global_min_distance = min_distance
            global_min_step = step

        rows.append(
            {
                "step": step,
                "time_s": float(frame["time_s"]),
                "robot_x_m": robot_xy[0],
                "robot_y_m": robot_xy[1],
                "robot_heading_rad": float(robot.get("heading", 0.0)),
                "executed_vx_m_s": executed_vx,
                "executed_vy_m_s": executed_vy,
                "executed_speed_m_s": executed_speed,
                "commanded_linear_velocity_m_s": action.get("linear_velocity"),
                "commanded_angular_velocity_rad_s": action.get("angular_velocity"),
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
                "time_s": float(frame["time_s"]),
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
        "outcome_label": outcome_label(record),
    }
    return TraceRows(trace_rows=rows, min_distance_rows=min_rows, summary=summary)


def write_trace_bundle(
    *,
    row: dict[str, Any],
    cell: FigureCell,
    planner: str,
    output_dir: Path,
    campaign_job: str,
    generated_at: str,
    run_config_sha256: str | None,
) -> dict[str, Any]:
    """Write one trace directory in the Chapter 7 worked-example bundle layout."""

    output_dir.mkdir(parents=True, exist_ok=True)
    derived = derive_trace_rows(row)
    metadata = {
        "schema_version": "issue-2994-ch7-worked-example-trace.v1",
        "issue": ISSUE_URL,
        "claim_boundary": (
            "development-seed trace re-run from the published 0.0.8 source for "
            "Chapter 7 worked-example visualization only; it is not a new "
            "benchmark campaign or release claim"
        ),
        "generated_at_utc": generated_at,
        "source_commit": SOURCE_COMMIT,
        "release_tag": RELEASE_TAG,
        "doi": DOI,
        "campaign_id": "issue2994_ch7_trace_worked_examples_008_dev_sweep_v2",
        "campaign_job": campaign_job,
        "scenario_config": SCENARIO_CONFIG,
        "run_config": RUN_CONFIG,
        "run_config_sha256": run_config_sha256,
        "figure_cell": cell.key,
        "figure_label": cell.label,
        "planner": planner,
        "scenario_id": cell.scenario_id,
        "seed": int(row["seed"]),
        "episode_id": row.get("episode_id"),
        "episode_status": row.get("status"),
        "termination_reason": row.get("termination_reason"),
        "outcome_label": outcome_label(row),
        "distance_convention": DistanceConvention.CENTER_CENTER.value,
        "summary": derived.summary,
    }
    trace_payload = {
        "schema_version": "issue-2994-ch7-worked-example-trace-series.v1",
        "metadata": metadata,
        "frames": row["algorithm_metadata"]["simulation_step_trace"]["steps"],
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
    _write_trace_readme(output_dir, metadata)
    write_sha256sums(output_dir)
    return metadata


def _write_trace_readme(output_dir: Path, metadata: dict[str, Any]) -> None:
    readme = f"""{review_marker("ll7/diss#2994", marker_date=extract_marker_date(metadata))}
# Chapter 7 0.0.8 Worked-Example Trace: {metadata["figure_cell"]}

This directory contains one development-seed trace re-run from the published
robot_sf_ll7 0.0.8 source. It is intended only as figure/caption input for the
Chapter 7 worked examples.

## Contents

- `trace_timeseries.csv`: per-timestep robot state, commanded action, executed velocity,
  pedestrian positions, and nearest robot-pedestrian distance.
- `min_distance_series.csv`: figure-ready center-to-center distance series.
- `trace_series.json`: raw recorded frames plus derived rows.
- `metadata.json`: source, config, Slurm job, scenario, planner, seed, and claim boundary.
- `SHA256SUMS`: checksums for the files above.

## Provenance

- Source commit: `{metadata["source_commit"]}`
- DOI: `{metadata["doi"]}`
- Campaign: `{metadata["campaign_id"]}`
- Slurm job: `{metadata["campaign_job"]}`
- Scenario: `{metadata["scenario_id"]}`
- Planner: `{metadata["planner"]}`
- Outcome label: `{metadata["outcome_label"]}`

## Claim Boundary

Visualization-only trace evidence. Do not treat this bundle as a fresh benchmark,
release admission, or statistical comparison.

<!-- /AI-GENERATED -->
"""
    write_text(output_dir / "README.md", readme)


def _format_counts(counter: Counter[str], total: int) -> str:
    parts = []
    for label in ("success", "collision", "timeout", "route_complete_non_success", "other"):
        if counter.get(label, 0):
            parts.append(f"{label}={counter[label]}/{total}")
    return ", ".join(parts) if parts else f"none/{total}"


def _safe_slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value)


def _trace_sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _top_level_sha256sums(root: Path) -> None:
    rows: list[str] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != "SHA256SUMS"):
        rows.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    (root / "SHA256SUMS").write_text("\n".join(rows) + "\n", encoding="utf-8")


def build_bundle(
    *,
    published_bundle: Path,
    dev_campaign_root: Path,
    output_dir: Path,
    run_config: Path | None,
    campaign_job: str,
) -> dict[str, Any]:
    """Build the full selection report, trace export, manifest, and checksums."""

    generated_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    run_config_sha = sha256_file(run_config) if run_config and run_config.is_file() else None
    output_dir.mkdir(parents=True, exist_ok=True)

    published_by_planner = {
        planner: load_rows(published_bundle, planner) for planner in PLANNER_RUNS
    }
    dev_by_planner = {
        planner: load_dev_rows(dev_campaign_root, planner) for planner in PLANNER_RUNS
    }

    selection_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    manifest_cells: list[dict[str, Any]] = []

    all_cells: list[tuple[str, tuple[str, str], str]] = [
        (cell.scenario_id, cell.planners, cell.key) for cell in FIGURE_CELLS
    ] + [(scenario, planners, "alternate_candidate") for scenario, planners in EXTRA_CANDIDATES]
    for scenario_id, planners, role in all_cells:
        for planner in planners:
            rows = [
                row
                for row in published_by_planner[planner]
                if row.get("scenario_id") == scenario_id
            ]
            counter = outcome_counts(rows)
            candidate_rows.append(
                {
                    "role": role,
                    "scenario_id": scenario_id,
                    "planner": planner,
                    "published_total": len(rows),
                    "published_outcomes": _format_counts(counter, len(rows)),
                }
            )

    for cell in FIGURE_CELLS:
        published_cell_rows = {
            planner: [
                row
                for row in published_by_planner[planner]
                if row.get("scenario_id") == cell.scenario_id
            ]
            for planner in cell.planners
        }
        target_counts = {
            planner: outcome_counts(rows) for planner, rows in published_cell_rows.items()
        }
        dev_cell_rows = {
            planner: [
                row for row in dev_by_planner[planner] if row.get("scenario_id") == cell.scenario_id
            ]
            for planner in cell.planners
        }
        selected_seeds, match_status = select_shared_subset(dev_cell_rows, target_counts)
        rows_by_seed = {
            planner: {int(row["seed"]): row for row in rows}
            for planner, rows in dev_cell_rows.items()
        }
        dev_selected_counts = {
            planner: outcome_counts([rows_by_seed[planner][seed] for seed in selected_seeds])
            for planner in cell.planners
        }

        for planner in cell.planners:
            selection_rows.append(
                {
                    "figure_cell": cell.key,
                    "label": cell.label,
                    "scenario_id": cell.scenario_id,
                    "planner": planner,
                    "published_outcomes": _format_counts(target_counts[planner], 30),
                    "dev_selected_outcomes": _format_counts(
                        dev_selected_counts[planner], len(selected_seeds)
                    ),
                    "distribution_match": match_status,
                    "rationale": cell.rationale,
                }
            )

        manifest_cells.append(
            {
                "figure_cell": cell.key,
                "label": cell.label,
                "scenario_id": cell.scenario_id,
                "planners": list(cell.planners),
                "selected_dev_seed_count": len(selected_seeds),
                "distribution_match": match_status,
                "rationale": cell.rationale,
            }
        )

        for planner in cell.planners:
            for index, seed in enumerate(selected_seeds, 1):
                row = rows_by_seed[planner][seed]
                trace_dir = (
                    output_dir
                    / planner
                    / f"{cell.key}__{cell.scenario_id}__devseed_{seed:04d}__{index:02d}"
                )
                metadata = write_trace_bundle(
                    row=row,
                    cell=cell,
                    planner=planner,
                    output_dir=trace_dir,
                    campaign_job=campaign_job,
                    generated_at=generated_at,
                    run_config_sha256=run_config_sha,
                )
                trace_rows.append(
                    {
                        "figure_cell": cell.key,
                        "planner": planner,
                        "scenario_id": cell.scenario_id,
                        "seed": seed,
                        "outcome_label": metadata["outcome_label"],
                        "trace_uri": trace_dir.relative_to(output_dir).as_posix(),
                        "trace_series_sha256_16": _trace_sha16(trace_dir / "trace_series.json"),
                    }
                )

    _write_csv(output_dir / "selection_table.csv", selection_rows)
    _write_csv(output_dir / "candidate_cells.csv", candidate_rows)
    _write_csv(output_dir / "trace_sha256_16.csv", trace_rows)
    _write_report(output_dir, selection_rows, candidate_rows, trace_rows, generated_at)
    _write_proposed_yaml(output_dir, trace_rows)
    manifest = {
        "schema_version": "issue-2994-ch7-trace-bundle-manifest.v1",
        "generated_at_utc": generated_at,
        "source_commit": SOURCE_COMMIT,
        "release_tag": RELEASE_TAG,
        "doi": DOI,
        "scenario_config": SCENARIO_CONFIG,
        "run_config": RUN_CONFIG,
        "run_config_sha256": run_config_sha,
        "campaign_id": "issue2994_ch7_trace_worked_examples_008_dev_sweep_v2",
        "campaign_job": campaign_job,
        "seed_policy": "development seeds 1001..1200 only; sealed publication seeds are not reused",
        "figure_cells": manifest_cells,
        "trace_count": len(trace_rows),
        "claim_boundary": (
            "selection and visualization trace bundle only; generated traces are "
            "development-seed re-runs and do not replace the published 0.0.8 benchmark rows"
        ),
    }
    write_json(output_dir / "manifest.json", manifest)
    _top_level_sha256sums(output_dir)
    return manifest


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_report(
    output_dir: Path,
    selection_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    trace_rows: list[dict[str, Any]],
    generated_at: str,
) -> None:
    lines = [
        review_marker("ll7/diss#2994", marker_date=generated_at[:10]),
        "# Chapter 7 Worked-Example Trace Selection for robot_sf_ll7 0.0.8",
        "",
        f"Generated: {generated_at}",
        f"Source commit: `{SOURCE_COMMIT}`",
        f"Published rows: Zenodo DOI `{DOI}`",
        "",
        "## Selection Table",
        "",
        "| figure cell | scenario | planner | published 0.0.8 outcomes | dev selected outcomes | match |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in selection_rows:
        lines.append(
            "| {figure_cell} | {scenario_id} | {planner} | {published_outcomes} | "
            "{dev_selected_outcomes} | {distribution_match} |".format(**row)
        )
    lines.extend(
        [
            "",
            "Important: `classic_head_on_corridor_medium` no longer supports the old "
            "social-force-stalls contrast. In the published 0.0.8 rows, social force "
            "completes most of the cell; the replacement figure should change the story.",
            "",
            "## Candidate Cells",
            "",
            "| role | scenario | planner | published outcomes |",
            "| --- | --- | --- | --- |",
        ]
    )
    for row in candidate_rows:
        lines.append("| {role} | {scenario_id} | {planner} | {published_outcomes} |".format(**row))
    lines.extend(
        [
            "",
            "## Trace SHA256-16",
            "",
            "Full per-trace values are in `trace_sha256_16.csv`.",
            "",
            "| figure cell | planner | scenario | trace | sha256_16 |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in trace_rows[:30]:
        lines.append(
            "| {figure_cell} | {planner} | {scenario_id} | `{trace_uri}` | `{trace_series_sha256_16}` |".format(
                **row
            )
        )
    lines.extend(["", "## Proposed Evidence Pins", "", "See `proposed_evidence_pins.yaml`."])
    write_text(output_dir / "SELECTION_REPORT.md", "\n".join(lines) + "\n")


def _write_proposed_yaml(output_dir: Path, trace_rows: list[dict[str, Any]]) -> None:
    by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trace_rows:
        by_cell[row["figure_cell"]].append(row)
    lines = [
        "chapter7_worked_examples_0_0_8:",
        "  bundle: issue_2994_ch7_trace_worked_examples_0_0_8_373dbfde4f",
        f"  source_commit: {SOURCE_COMMIT}",
        f"  doi: {DOI}",
        "  claim_boundary: visualization-only development-seed trace reruns; published 0.0.8 rows remain the statistical source",
        "  figures:",
    ]
    cell_by_key = {cell.key: cell for cell in FIGURE_CELLS}
    for key, rows in by_cell.items():
        cell = cell_by_key[key]
        lines.extend(
            [
                f"    {cell.proposed_pin_key}:",
                f"      scenario_id: {cell.scenario_id}",
                f"      trace_count: {len(rows)}",
                "      traces:",
            ]
        )
        for row in rows:
            lines.extend(
                [
                    f"        - planner: {row['planner']}",
                    f"          uri: {row['trace_uri']}",
                    f"          sha256_16: {row['trace_series_sha256_16']}",
                    f"          outcome_label: {row['outcome_label']}",
                ]
            )
    (output_dir / "proposed_evidence_pins.yaml").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--published-bundle", type=Path, default=DEFAULT_PUBLISHED_BUNDLE)
    parser.add_argument("--dev-campaign-root", type=Path, default=DEFAULT_DEV_CAMPAIGN_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-config", type=Path)
    parser.add_argument("--campaign-job", required=True)
    return parser


def main() -> int:
    """Run the command-line bundle builder."""

    args = _build_parser().parse_args()
    manifest = build_bundle(
        published_bundle=args.published_bundle,
        dev_campaign_root=args.dev_campaign_root,
        output_dir=args.output_dir,
        run_config=args.run_config,
        campaign_job=args.campaign_job,
    )
    print(json.dumps({"output_dir": str(args.output_dir), "trace_count": manifest["trace_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
