"""CPU smoke comparison for issue #4164 goal-posterior planner consumption.

This script runs a bounded deterministic closed-loop proxy with and without the
opt-in goal-posterior channel. It is smoke evidence that one planner consumes
posterior metadata during action selection; it is not a full benchmark campaign,
planner-performance claim, or calibrated human-intention claim.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from robot_sf.planner.hybrid_rule_local_planner import (
    HybridRuleLocalPlannerAdapter,
    HybridRuleLocalPlannerConfig,
)
from robot_sf.prediction.goal_intention import (
    GoalPosteriorConfig,
    planner_goal_posterior_channel_from_state,
)

CLAIM_BOUNDARY = (
    "smoke evidence: hybrid_rule_local_planner consumes goal-posterior metadata in a "
    "bounded CPU closed-loop proxy; no full benchmark campaign, no calibrated "
    "human-intention claim, no planner-performance claim"
)


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("smoke config must be a YAML mapping")
    return payload


def _posterior_config(payload: dict[str, Any]) -> GoalPosteriorConfig:
    raw = payload.get("posterior_config", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("posterior_config must be a mapping when present")
    return GoalPosteriorConfig(**raw)


def _scenario_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    scenarios = config.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("config.scenarios must be a non-empty list")
    for index, scenario in enumerate(scenarios):
        if not isinstance(scenario, dict):
            raise ValueError(f"scenarios[{index}] must be a mapping")
        for key in ("id", "positions", "velocities", "goals"):
            if key not in scenario:
                raise ValueError(f"scenarios[{index}] missing required key {key!r}")
    return scenarios


def _as_xy_rows(value: Any, *, field_name: str) -> np.ndarray:
    rows = np.asarray(value, dtype=float)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    if rows.ndim != 2 or rows.shape[1] != 2 or not np.all(np.isfinite(rows)):
        raise ValueError(f"{field_name} must be finite xy rows")
    return rows


def _channel_for_scenario(
    *,
    enabled: bool,
    scenario: dict[str, Any],
    posterior_config: GoalPosteriorConfig,
) -> dict[str, Any]:
    positions = _as_xy_rows(scenario["positions"], field_name="positions")
    velocities = _as_xy_rows(scenario["velocities"], field_name="velocities")
    goals = _as_xy_rows(scenario["goals"], field_name="goals")
    pedestrian_ids = scenario.get("pedestrian_ids")
    return planner_goal_posterior_channel_from_state(
        enabled=enabled,
        positions=positions.tolist(),
        velocities=velocities.tolist(),
        goals=goals.tolist(),
        pedestrian_ids=pedestrian_ids,
        config=posterior_config,
    )


def _observation(
    *,
    robot: np.ndarray,
    heading: float,
    goal: np.ndarray,
    speed: float,
    ped_positions: np.ndarray,
    ped_velocities: np.ndarray,
    channel: dict[str, Any] | None,
) -> dict[str, Any]:
    obs: dict[str, Any] = {
        "robot": {
            "position": robot.copy(),
            "heading": np.asarray([heading], dtype=float),
            "speed": np.asarray([speed], dtype=float),
            "radius": np.asarray([0.25], dtype=float),
        },
        "goal": {"current": goal.copy(), "next": goal.copy()},
        "pedestrians": {
            "positions": ped_positions.copy(),
            "velocities": ped_velocities.copy(),
            "count": np.asarray([ped_positions.shape[0]], dtype=float),
            "radius": 0.25,
        },
        "sim": {"timestep": 0.2},
    }
    if channel is not None:
        obs["info"] = {"planner_goal_posterior_channel": channel}
    return obs


def _planner(enabled: bool) -> HybridRuleLocalPlannerAdapter:
    return HybridRuleLocalPlannerAdapter(
        HybridRuleLocalPlannerConfig(
            goal_posterior_avoidance_enabled=enabled,
            goal_posterior_min_confidence=0.5,
            goal_posterior_near_distance=3.0,
            goal_posterior_crossing_lateral_margin=1.0,
            goal_posterior_yield_speed=0.05,
            goal_posterior_turn_rate=0.7,
            goal_posterior_score_bonus=20.0,
        )
    )


def _run_arm(
    *,
    enabled: bool,
    scenario: dict[str, Any],
    posterior_config: GoalPosteriorConfig,
) -> dict[str, Any]:
    start = time.perf_counter()
    planner = _planner(enabled)
    channel = _channel_for_scenario(
        enabled=enabled,
        scenario=scenario,
        posterior_config=posterior_config,
    )
    ped_positions = _as_xy_rows(scenario["positions"], field_name="positions")
    ped_velocities = _as_xy_rows(scenario["velocities"], field_name="velocities")
    robot = np.asarray(scenario.get("robot_start", [0.0, 0.0]), dtype=float)
    goal = np.asarray(scenario.get("robot_goal", [4.0, 0.0]), dtype=float)
    heading = float(scenario.get("robot_heading", 0.0))
    speed = float(scenario.get("robot_speed", 0.0))
    dt = 0.2
    trajectory: list[list[float]] = [[float(robot[0]), float(robot[1])]]
    commands: list[list[float]] = []
    sources: list[str] = []
    selected_terms: list[dict[str, float]] = []
    posterior_diagnostics: list[dict[str, Any] | None] = []
    initial_distance = float(np.linalg.norm(goal - robot))

    for _step in range(int(scenario.get("closed_loop_steps", 4))):
        obs = _observation(
            robot=robot,
            heading=heading,
            goal=goal,
            speed=speed,
            ped_positions=ped_positions,
            ped_velocities=ped_velocities,
            channel=channel,
        )
        linear, angular = planner.plan(obs)
        decision = planner.last_decision() or {}
        commands.append([float(linear), float(angular)])
        sources.append(str(decision.get("selected_source", "unknown")))
        selected_terms.append(dict(decision.get("selected_terms", {})))
        posterior_diagnostics.append(decision.get("goal_posterior_avoidance"))
        heading += float(angular) * dt
        robot = robot + np.asarray([math.cos(heading), math.sin(heading)]) * float(linear) * dt
        speed = float(linear)
        trajectory.append([float(robot[0]), float(robot[1])])

    final_distance = float(np.linalg.norm(goal - robot))
    diagnostics = planner.diagnostics()
    fallback_count = int(diagnostics.get("fallback_count", 0))
    degraded_exclusions = {
        "fallback_count": fallback_count,
        "fallback_or_degraded": bool(fallback_count),
        "posterior_blockers": [
            item.get("blocker")
            for item in posterior_diagnostics
            if isinstance(item, dict) and item.get("blocker")
        ],
    }
    return {
        "channel_present": enabled,
        "planner_consumed_channel": any(
            bool(item.get("consumed")) for item in posterior_diagnostics if isinstance(item, dict)
        ),
        "posterior_active": any(
            bool(item.get("active")) for item in posterior_diagnostics if isinstance(item, dict)
        ),
        "commands": commands,
        "selected_sources": sources,
        "selected_terms": selected_terms,
        "trajectory": trajectory,
        "route_progress": initial_distance - final_distance,
        "final_goal_distance": final_distance,
        "planner_diagnostics": diagnostics,
        "degraded_exclusions": degraded_exclusions,
        "runtime_s": time.perf_counter() - start,
    }


def build_report(config_path: Path) -> dict[str, Any]:
    """Return diagnostic with/without planner-consumption comparison."""
    config = _load_config(config_path)
    posterior_config = _posterior_config(config)
    scenario_reports = []
    for scenario in _scenario_rows(config):
        without_channel = _run_arm(
            enabled=False,
            scenario=scenario,
            posterior_config=posterior_config,
        )
        with_channel = _run_arm(
            enabled=True,
            scenario=scenario,
            posterior_config=posterior_config,
        )
        command_source_changed = (
            without_channel["selected_sources"] != with_channel["selected_sources"]
        )
        trajectory_changed = without_channel["trajectory"] != with_channel["trajectory"]
        scenario_reports.append(
            {
                "scenario_id": scenario["id"],
                "planner_path": "hybrid_rule_local_planner.goal_posterior_avoidance",
                "without_goal_posterior": without_channel,
                "with_goal_posterior": with_channel,
                "command_source_changed": command_source_changed,
                "trajectory_changed": trajectory_changed,
                "route_progress_delta": (
                    with_channel["route_progress"] - without_channel["route_progress"]
                ),
                "runtime_overhead_s": with_channel["runtime_s"] - without_channel["runtime_s"],
                "fallback_or_degraded_exclusions": {
                    "without": without_channel["degraded_exclusions"],
                    "with": with_channel["degraded_exclusions"],
                },
            }
        )
    return {
        "schema_version": "issue_4164_goal_posterior_planner_consumption_smoke.v1",
        "config_path": str(config_path),
        "claim_boundary": CLAIM_BOUNDARY,
        "posterior_config": {
            "heading_kappa": posterior_config.heading_kappa,
            "velocity_min_mps": posterior_config.velocity_min_mps,
            "prior_floor": posterior_config.prior_floor,
            "config_hash": posterior_config.config_hash,
        },
        "scenarios": scenario_reports,
    }


def main(argv: list[str] | None = None) -> int:
    """Run issue #4164 CPU smoke report CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/benchmarks/issue_4164_goal_intention_smoke.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/benchmarks/issue_4164_goal_intention_smoke.json"),
    )
    args = parser.parse_args(argv)
    report = build_report(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
