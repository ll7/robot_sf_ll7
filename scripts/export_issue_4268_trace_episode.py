#!/usr/bin/env python3
"""Export the issue #4268 single-episode doorway trace evidence bundle.

The script intentionally composes the existing map-runner per-step trace recording path instead
of adding a new recorder. The output is a single illustrative episode for downstream figure work,
not benchmark evidence.
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

from robot_sf.benchmark.classic_interactions_loader import load_classic_matrix, select_scenario
from robot_sf.benchmark.map_runner import _run_map_episode

DEFAULT_OUTPUT_DIR = Path("docs/context/evidence/issue_4253_trace_episode_2026-07")
DEFAULT_SCENARIO_MATRIX = Path("configs/scenarios/classic_interactions.yaml")
DEFAULT_SCENARIO_ID = "classic_doorway_medium"
DEFAULT_SEED = 141
DEFAULT_HORIZON = 100
DEFAULT_DT = 0.1
DEFAULT_PLANNER = "simple_policy"


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
    """Convert a map-runner episode record into trace and min-distance CSV rows."""

    trace = record["algorithm_metadata"]["simulation_step_trace"]
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


def write_bundle(
    *,
    output_dir: Path,
    scenario_matrix: Path,
    scenario_id: str,
    seed: int,
    planner: str,
    horizon: int,
    dt: float,
) -> dict[str, Any]:
    """Run the pinned episode and write the tracked evidence bundle."""

    output_dir.mkdir(parents=True, exist_ok=True)
    scenarios = load_classic_matrix(str(scenario_matrix))
    scenario = select_scenario(scenarios, scenario_id)
    try:
        scenario_matrix_ref = scenario_matrix.relative_to(_repo_root()).as_posix()
    except ValueError:
        scenario_matrix_ref = scenario_matrix.as_posix()
    record = _run_map_episode(
        scenario,
        seed,
        horizon=horizon,
        dt=dt,
        record_forces=False,
        snqi_weights=None,
        snqi_baseline=None,
        algo=planner,
        scenario_path=scenario_matrix,
        record_simulation_step_trace=True,
    )
    derived = derive_trace_rows(record)

    metadata = {
        "schema_version": "issue-4268-trace-episode-metadata.v1",
        "issue": "https://github.com/ll7/robot_sf_ll7/issues/4268",
        "downstream": "ll7/diss#313 item A",
        "claim_boundary": (
            "single illustrative doorway episode only; no statistical, benchmark, "
            "paper, or dissertation claim is established by this bundle"
        ),
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "scenario_matrix": scenario_matrix_ref,
        "scenario_matrix_sha256": sha256_file(scenario_matrix),
        "scenario_id": scenario_id,
        "seed": seed,
        "planner": planner,
        "horizon": horizon,
        "dt": dt,
        "recording_path": "robot_sf.benchmark.map_runner._run_map_episode(record_simulation_step_trace=True)",
        "status": record.get("status"),
        "termination_reason": record.get("termination_reason"),
        "steps": record.get("steps"),
        "summary": derived.summary,
    }

    trace_payload = {
        "schema_version": "issue-4268-trace-series.v1",
        "metadata": metadata,
        "frames": record["algorithm_metadata"]["simulation_step_trace"]["steps"],
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

    readme = f"""# Issue #4268 Trace-Level Doorway Episode

Plain-language summary: this directory contains one reproducible, seed-pinned doorway episode trace
for dissertation Chapter 7 figure work. It is a single illustrative episode and does not establish a
statistical benchmark or dissertation claim.

## Contents

- `trace_timeseries.csv`: per-timestep robot state, commanded action, executed velocity, pedestrian
  positions, and nearest robot-pedestrian distance.
- `min_distance_series.csv`: figure-ready `(step, time_s, min_robot_ped_distance_m)` series.
- `trace_series.json`: raw recorded frames plus derived rows.
- `metadata.json`: scenario, seed, planner, commit, matrix hash, and claim boundary.
- `SHA256SUMS`: checksums for the files above.

## Reproduction

```bash
LOGURU_LEVEL=WARNING uv run python scripts/export_issue_4268_trace_episode.py
```

Pinned run:

- scenario matrix: `{metadata["scenario_matrix"]}`
- scenario id: `{metadata["scenario_id"]}`
- seed: `{metadata["seed"]}`
- planner: `{metadata["planner"]}`
- horizon: `{metadata["horizon"]}`
- dt: `{metadata["dt"]}`
- git commit at generation: `{metadata["git_commit"]}`

## Claim Boundary

This bundle is `analysis_workbench_only` style evidence for one illustrative episode. It should be
used as a trace-level worked example input only. It is not a full benchmark campaign, not a Slurm or
GPU result, and not a statistical comparison.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def _write_sha256sums(output_dir: Path) -> None:
    """Write SHA256SUMS for all generated bundle files except itself."""

    files = sorted(
        path for path in output_dir.iterdir() if path.is_file() and path.name != "SHA256SUMS"
    )
    lines = [f"{sha256_file(path)}  {path.name}" for path in files]
    (output_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scenario-matrix", type=Path, default=DEFAULT_SCENARIO_MATRIX)
    parser.add_argument("--scenario-id", default=DEFAULT_SCENARIO_ID)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--planner", default=DEFAULT_PLANNER)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--dt", type=float, default=DEFAULT_DT)
    return parser


def main() -> int:
    """CLI entry point."""

    args = _build_parser().parse_args()
    repo_root = _repo_root()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    scenario_matrix = args.scenario_matrix
    if not scenario_matrix.is_absolute():
        scenario_matrix = repo_root / scenario_matrix
    metadata = write_bundle(
        output_dir=output_dir,
        scenario_matrix=scenario_matrix,
        scenario_id=args.scenario_id,
        seed=args.seed,
        planner=args.planner,
        horizon=args.horizon,
        dt=args.dt,
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "status": metadata["status"],
                "steps": metadata["steps"],
                "min_distance_m": metadata["summary"]["global_min_robot_ped_distance_m"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
