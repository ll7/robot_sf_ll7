#!/usr/bin/env python3
"""Run a minimal multi-AMV benchmark smoke and emit inter-robot metrics."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from robot_sf.benchmark.aggregate import compute_aggregates
from robot_sf.benchmark.identity.hash_utils import read_jsonl as _read_jsonl
from robot_sf.benchmark.multi_amv import (
    MultiAmvSettings,
    ensure_multi_amv_planner_supported,
    inter_robot_metrics,
    multi_amv_episode_extension,
    multi_amv_settings_from_scenario,
    paired_actuation_feasibility_ranking,
)
from robot_sf.benchmark.schema_validator import load_schema, validate_episode
from robot_sf.benchmark.termination_reason import status_from_termination_reason
from robot_sf.benchmark.thresholds import ensure_metric_parameters
from robot_sf.benchmark.utils import (
    _config_hash,
    _git_hash_fallback,
    validate_episode_success_integrity,
)
from robot_sf.gym_env.environment_factory import make_multi_robot_env
from robot_sf.gym_env.unified_config import MultiRobotConfig
from robot_sf.training.scenario_loader import (
    build_robot_config_from_scenario,
    load_scenarios,
)


def _multi_robot_config_from_scenario(
    scenario: dict[str, Any], scenario_path: Path
) -> MultiRobotConfig:
    """Build a ``MultiRobotConfig`` from an existing benchmark scenario."""
    base = build_robot_config_from_scenario(scenario, scenario_path=scenario_path)
    settings = multi_amv_settings_from_scenario(scenario)
    config_values = {
        field.name: getattr(base, field.name)
        for field in fields(MultiRobotConfig)
        if field.name != "num_robots" and hasattr(base, field.name)
    }
    return MultiRobotConfig(**config_values, num_robots=settings.num_robots)


def _goal_actions(env: Any) -> np.ndarray:
    """Return simple unicycle goal-seeking actions for every robot in the env."""
    actions: list[list[float]] = []
    for sim in env.simulators:
        for robot, goal in zip(sim.robots, sim.goal_pos, strict=False):
            pos = np.asarray(robot.pose[0], dtype=float)
            heading = float(robot.pose[1])
            goal_xy = np.asarray(goal, dtype=float)
            delta = goal_xy - pos
            distance = float(np.linalg.norm(delta))
            target_heading = float(np.arctan2(delta[1], delta[0])) if distance > 1e-9 else heading
            heading_error = ((target_heading - heading + np.pi) % (2.0 * np.pi)) - np.pi
            linear = min(0.6, distance)
            angular = float(np.clip(2.0 * heading_error, -1.0, 1.0))
            actions.append([linear, angular])
    return np.asarray(actions, dtype=np.float32)


def _robot_positions(env: Any) -> np.ndarray:
    """Collect current robot positions from all simulators."""
    positions: list[tuple[float, float]] = []
    for sim in env.simulators:
        positions.extend(tuple(map(float, pos)) for pos in sim.robot_pos)
    return np.asarray(positions, dtype=float)


def build_multi_amv_episode_record(  # noqa: PLR0913
    *,
    scenario_id: str,
    seed: int,
    horizon: int,
    steps_recorded: int,
    settings: MultiAmvSettings,
    inter_robot: dict[str, float | bool],
    planner_family: str,
    planner_status: str,
    planner_note: str | None = None,
    wall_time_sec: float,
    start_timestamp: datetime | None = None,
) -> dict[str, Any]:
    """Build a canonical benchmark-style multi-AMV episode record."""
    extension = multi_amv_episode_extension(
        settings=settings,
        inter_robot=inter_robot,
        planner_family=planner_family,
        planner_status=planner_status,
        planner_note=planner_note,
    )
    collision_events = float(inter_robot.get("inter_robot_collision_events", 0.0) or 0.0)
    collision_detected = collision_events > 0.0
    termination_reason = "collision" if collision_detected else "terminated"
    finished_at = datetime.now(UTC)
    started_at = start_timestamp or finished_at
    scenario_params = {
        "scenario_id": str(scenario_id),
        "algo": str(planner_family),
        "horizon": int(horizon),
        "multi_amv": {"settings": extension["multi_amv"]["settings"]},
    }
    record = {
        "version": "v1",
        "episode_id": f"multi_amv::{scenario_id}::{planner_family}::{seed}::{horizon}",
        "scenario_id": str(scenario_id),
        "seed": int(seed),
        "scenario_params": scenario_params,
        "metrics": {"collisions": collision_events, "inter_robot": dict(inter_robot)},
        "algorithm_metadata": {
            "algorithm": str(planner_family),
            "canonical_algorithm": str(planner_family),
            "status": "ok",
            "baseline_category": "diagnostic",
            "multi_amv_planner_support": ensure_multi_amv_planner_supported(
                planner_family
            ).to_json_dict(),
        },
        "algo": str(planner_family),
        "config_hash": _config_hash(scenario_params),
        "git_hash": _git_hash_fallback(),
        "timestamps": {
            "start": started_at.isoformat(),
            "end": finished_at.isoformat(),
        },
        "status": status_from_termination_reason(termination_reason),
        "steps": int(steps_recorded),
        "horizon": int(horizon),
        "wall_time_sec": float(wall_time_sec),
        "timing": {
            "steps_per_second": (
                float(steps_recorded) / float(wall_time_sec) if wall_time_sec > 0 else 0.0
            )
        },
        "termination_reason": termination_reason,
        "outcome": {
            "route_complete": False,
            "collision_event": collision_detected,
            "timeout_event": False,
        },
        "integrity": {"contradictions": []},
    }
    record.update(extension)
    ensure_metric_parameters(record)
    return record


def run_smoke(*, scenario_path: Path, horizon: int) -> dict[str, Any]:
    """Run the first scenario in ``scenario_path`` and return an episode record."""
    scenario = dict(load_scenarios(scenario_path)[0])
    settings = multi_amv_settings_from_scenario(scenario)
    planner_support = ensure_multi_amv_planner_supported("goal_controller_smoke")
    config = _multi_robot_config_from_scenario(scenario, scenario_path)
    config.sim_config.sim_time_in_secs = max(
        config.sim_config.time_per_step_in_secs,
        horizon * config.sim_config.time_per_step_in_secs,
    )
    env = make_multi_robot_env(num_robots=settings.num_robots, config=config, debug=False)
    positions = []
    started = time.time()
    start_timestamp = datetime.now(UTC)
    try:
        env.reset(seed=0)
        positions.append(_robot_positions(env))
        for _ in range(horizon):
            _obs, _reward, terminated, truncated, _info = env.step(_goal_actions(env))
            positions.append(_robot_positions(env))
            if terminated or truncated:
                break
    finally:
        env.close()
    wall_time_sec = max(1e-9, time.time() - started)
    robot_positions = np.stack(positions, axis=0)
    metrics = inter_robot_metrics(
        robot_positions,
        dt=float(config.sim_config.time_per_step_in_secs),
        settings=settings,
    )
    scenario_id = str(scenario.get("name") or scenario.get("scenario_id") or scenario.get("id"))
    return build_multi_amv_episode_record(
        scenario_id=scenario_id,
        seed=0,
        horizon=horizon,
        steps_recorded=int(robot_positions.shape[0]),
        settings=settings,
        inter_robot=metrics,
        planner_family=planner_support.planner_family,
        planner_status="goal_controller_smoke",
        planner_note=planner_support.rationale,
        wall_time_sec=wall_time_sec,
        start_timestamp=start_timestamp,
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON payload with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from ``path``."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def _planner_status_lookup(campaign_summary: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Build planner status lookup keys from a camera-ready campaign summary."""
    lookup: dict[str, dict[str, str]] = {}
    planner_rows = campaign_summary.get("planner_rows")
    if not isinstance(planner_rows, list):
        return lookup
    for row in planner_rows:
        if not isinstance(row, dict):
            continue
        status = {
            "status": str(row.get("status", "")),
            "readiness_status": str(row.get("readiness_status", "")),
            "availability_status": str(row.get("availability_status", "")),
            "execution_mode": str(row.get("execution_mode", "")),
        }
        for key in ("planner_key", "algo"):
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                lookup[value.strip()] = status
    return lookup


def _apply_campaign_status(
    records: list[dict[str, Any]], campaign_summary: dict[str, Any]
) -> list[dict[str, Any]]:
    """Annotate episode records with campaign-level planner status fields."""
    lookup = _planner_status_lookup(campaign_summary)
    annotated_records: list[dict[str, Any]] = []
    for record in records:
        annotated = dict(record)
        candidates = [
            annotated.get("planner_key"),
            annotated.get("algo"),
        ]
        metadata = annotated.get("algorithm_metadata")
        if isinstance(metadata, dict):
            candidates.extend(
                [
                    metadata.get("planner_key"),
                    metadata.get("canonical_algorithm"),
                    metadata.get("algorithm"),
                ]
            )
        for candidate in candidates:
            if not isinstance(candidate, str):
                continue
            status = lookup.get(candidate.strip())
            if status is None:
                continue
            for key, value in status.items():
                if value and not annotated.get(key):
                    annotated[key] = value
            break
        annotated_records.append(annotated)
    return annotated_records


def _write_episode_jsonl(path: Path, record: dict[str, Any], *, schema_path: Path) -> None:
    """Validate and write one canonical episode record as JSONL."""
    violations = validate_episode_success_integrity(record)
    if violations:
        raise ValueError("Episode integrity contradictions detected: " + "; ".join(violations))
    schema = load_schema(schema_path)
    validate_episode(record, schema)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")


def _write_report(path: Path, *, aggregates: dict[str, Any], record: dict[str, Any]) -> None:
    """Write a compact Markdown report for the smoke output."""
    inter_robot = record["metrics"]["inter_robot"]
    lines = [
        "# Multi-AMV Smoke Report",
        "",
        f"- Scenario: `{record['scenario_id']}`",
        f"- Planner status: `{record['multi_amv']['planner_status']}`",
        f"- Robots: `{record['multi_amv']['settings']['num_robots']}`",
        "",
        "## Inter-Robot Metrics",
        "",
    ]
    for key in sorted(inter_robot):
        lines.append(f"- `{key}`: `{inter_robot[key]}`")
    lines.extend(
        [
            "",
            "## Aggregate Groups",
            "",
        ]
    )
    for group_name in sorted(k for k in aggregates if not str(k).startswith("_")):
        lines.append(f"- `{group_name}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(
    *,
    record: dict[str, Any],
    out: Path,
    episodes_out: Path | None,
    aggregates_out: Path | None,
    report_out: Path | None,
    schema_path: Path,
) -> None:
    """Write requested multi-AMV smoke outputs."""
    _write_json(out, record)
    if episodes_out is not None:
        _write_episode_jsonl(episodes_out, record, schema_path=schema_path)
    aggregates: dict[str, Any] | None = None
    if aggregates_out is not None or report_out is not None:
        aggregates = compute_aggregates([record], group_by="algo")
    if aggregates_out is not None and aggregates is not None:
        _write_json(aggregates_out, aggregates)
    if report_out is not None and aggregates is not None:
        _write_report(report_out, aggregates=aggregates, record=record)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--episodes-out", type=Path)
    parser.add_argument("--aggregates-out", type=Path)
    parser.add_argument("--report-out", type=Path)
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path("robot_sf/benchmark/schemas/episode.schema.v1.json"),
    )
    parser.add_argument("--horizon", type=int, default=40)
    parser.add_argument(
        "--actuation-ranking-episodes",
        type=Path,
        action="append",
        help="Optional episode JSONL to summarize as a paired actuation-feasibility ranking.",
    )
    parser.add_argument(
        "--actuation-ranking-out",
        type=Path,
        help="Output JSON for --actuation-ranking-episodes.",
    )
    parser.add_argument(
        "--actuation-ranking-campaign-summary",
        type=Path,
        help="Optional camera-ready campaign_summary.json used to stamp planner-row statuses.",
    )
    parser.add_argument(
        "--baseline-variant",
        default="hybrid_rule_v3_fast_progress",
        help="Baseline variant key for actuation ranking mode.",
    )
    parser.add_argument(
        "--intervention-variant",
        default="actuation_aware_hybrid_rule_v0",
        help="Intervention variant key for actuation ranking mode.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the multi-AMV smoke CLI."""
    args = parse_args()
    if args.actuation_ranking_episodes is not None:
        if args.actuation_ranking_out is None:
            raise ValueError(
                "--actuation-ranking-out is required with --actuation-ranking-episodes"
            )
        records = [
            record
            for episodes_path in args.actuation_ranking_episodes
            for record in _read_jsonl(episodes_path)
        ]
        if args.actuation_ranking_campaign_summary is not None:
            records = _apply_campaign_status(
                records,
                _read_json(args.actuation_ranking_campaign_summary),
            )
        summary = paired_actuation_feasibility_ranking(
            records,
            baseline_variant=args.baseline_variant,
            intervention_variant=args.intervention_variant,
        )
        _write_json(args.actuation_ranking_out, summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    if args.scenario is None or args.out is None:
        raise ValueError("--scenario and --out are required unless actuation ranking mode is used")
    if args.horizon < 1:
        raise ValueError("--horizon must be >= 1")
    record = run_smoke(scenario_path=args.scenario, horizon=args.horizon)
    write_outputs(
        record=record,
        out=args.out,
        episodes_out=args.episodes_out,
        aggregates_out=args.aggregates_out,
        report_out=args.report_out,
        schema_path=args.schema,
    )
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
