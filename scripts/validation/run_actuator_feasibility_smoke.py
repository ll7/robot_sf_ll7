#!/usr/bin/env python3
"""Actuator-feasibility diagnostic smoke (issue #6056).

Runs the experimental actuator-feasibility evaluator over a small, fully deterministic
scenario table and writes a JSON + Markdown diagnostic report that shows the
geometry-only versus actuator-feasible distinction and which actuator limit was
violated in each row.

This is **diagnostic only** — it does not run benchmark episodes, does not measure
planner quality, and does not establish benchmark evidence. All actuator limits used
here are provisional defaults unless a caller supplies a measured config.

Usage::

    uv run python scripts/validation/run_actuator_feasibility_smoke.py
    uv run python scripts/validation/run_actuator_feasibility_smoke.py \\
        --output-root output/diagnostics/actuator_feasibility_smoke_issue_6056
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from robot_sf.benchmark.actuator_feasibility import (
    ACTUATOR_FEASIBILITY_CLAIM_BOUNDARY,
    ACTUATOR_FEASIBILITY_SCHEMA,
    ActuatorLimitsConfig,
    evaluate_actuator_feasibility,
    load_actuator_limits,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / "output/diagnostics/actuator_feasibility_smoke_issue_6056"


def _straight_trajectory(
    *, n_steps: int, speed: float, dt_s: float = 0.1
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic (positions, velocities) for a constant-speed +x path."""
    t = np.arange(n_steps, dtype=float) * dt_s
    positions = np.stack([t * speed, np.zeros(n_steps)], axis=1)
    velocities = np.tile([speed, 0.0], (n_steps, 1))
    return positions, velocities


def _hard_brake_trajectory(*, n_steps: int, dt_s: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic trajectory that commands a hard deceleration."""
    speeds = np.linspace(1.5, 0.1, n_steps)
    velocities = np.stack([speeds, np.zeros(n_steps)], axis=1)
    positions = np.cumsum(velocities * dt_s, axis=0)
    return positions, velocities


def build_scenarios() -> list[dict[str, Any]]:
    """Build the deterministic scenario table for the smoke.

    Each row is a single encounter geometry + trajectory + clearance chosen so the
    table exercises the full verdict range (actuator-feasible, geometry-only-clear,
    infeasible) and each actuator-limit predicate.
    """
    positions, velocities = _straight_trajectory(n_steps=4, speed=1.0)
    feasible_trajectory, feasible_vel = _straight_trajectory(n_steps=4, speed=0.5)

    accel_positions = np.array([[0.0, 0.0], [0.05, 0.0], [0.20, 0.0], [0.45, 0.0]])
    accel_vel = np.array([[0.2, 0.0], [1.5, 0.0], [1.5, 0.0], [1.5, 0.0]])

    hard_brake_pos, hard_brake_vel = _hard_brake_trajectory(n_steps=5)

    return [
        {
            "key": "actuator_feasible_slow_ampy_clearance",
            "description": "Slow straight path with ample clearance: physically executable.",
            "robot_positions": feasible_trajectory,
            "robot_velocities": feasible_vel,
            "dt_s": 0.1,
            "hazard_clearance_m": 5.0,
            "config": ActuatorLimitsConfig(),
        },
        {
            "key": "geometry_only_clear_brake_deadline_missed",
            "description": (
                "Geometric room exists but command+brake latency makes stopping infeasible."
            ),
            "robot_positions": positions,
            "robot_velocities": velocities,
            "dt_s": 0.1,
            "hazard_clearance_m": 0.5,
            "config": ActuatorLimitsConfig(),
        },
        {
            "key": "infeasible_already_in_contact",
            "description": "Negative clearance (already in contact): infeasible regardless.",
            "robot_positions": positions,
            "robot_velocities": velocities,
            "dt_s": 0.1,
            "hazard_clearance_m": -0.1,
            "config": ActuatorLimitsConfig(),
        },
        {
            "key": "accel_limit_exceeded",
            "description": "Commanded acceleration exceeds the provisional max_accel.",
            "robot_positions": accel_positions,
            "robot_velocities": accel_vel,
            "dt_s": 0.1,
            "hazard_clearance_m": 10.0,
            "config": ActuatorLimitsConfig(max_accel_mps2=0.5),
        },
        {
            "key": "decel_limit_exceeded",
            "description": "Commanded deceleration exceeds the provisional max_decel.",
            "robot_positions": hard_brake_pos,
            "robot_velocities": hard_brake_vel,
            "dt_s": 0.1,
            "hazard_clearance_m": 10.0,
            "config": ActuatorLimitsConfig(max_decel_mps2=0.5),
        },
    ]


def run_scenario(scenario: dict[str, Any]) -> dict[str, Any]:
    """Run one scenario and attach the report fields to a JSON-serializable row."""
    report = evaluate_actuator_feasibility(
        robot_positions=scenario["robot_positions"],
        robot_velocities=scenario["robot_velocities"],
        dt_s=scenario["dt_s"],
        hazard_clearance_m=scenario["hazard_clearance_m"],
        config=scenario["config"],
    )
    return {
        "key": scenario["key"],
        "description": scenario["description"],
        "hazard_clearance_m": scenario["hazard_clearance_m"],
        "verdict": report.verdict,
        "geometrically_clear": report.geometrically_clear,
        "physically_feasible": report.physically_feasible,
        "violated_limits": list(report.violated_limits),
        "max_speed_mps": report.max_speed_mps,
        "stopping_distance_m": report.stopping_distance_m,
        "observed_max_accel_mps2": report.observed_max_accel_mps2,
        "observed_max_decel_mps2": report.observed_max_decel_mps2,
        "observed_max_yaw_rate_radps": report.observed_max_yaw_rate_radps,
        "observed_max_steering_rate_radps": report.observed_max_steering_rate_radps,
    }


def build_summary(rows: list[dict[str, Any]], *, output_root: Path) -> dict[str, Any]:
    """Build the machine-readable diagnostic summary."""
    verdicts: dict[str, int] = {}
    for row in rows:
        verdicts[row["verdict"]] = verdicts.get(row["verdict"], 0) + 1
    return {
        "schema_version": ACTUATOR_FEASIBILITY_SCHEMA,
        "issue": 6056,
        "benchmark_evidence": False,
        "evidence_status": "diagnostic_only_not_benchmark_evidence",
        "claim_boundary": ACTUATOR_FEASIBILITY_CLAIM_BOUNDARY,
        "output_root": _repo_relative(output_root),
        "row_count": len(rows),
        "verdict_counts": dict(sorted(verdicts.items())),
        "rows": rows,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    """Render a concise Markdown diagnostic report."""
    rows = summary["rows"]
    lines = [
        "# Actuator-feasibility diagnostic smoke (issue #6056)",
        "",
        f"- Benchmark evidence: `{str(summary['benchmark_evidence']).lower()}`",
        f"- Verdict counts: `{summary['verdict_counts']}`",
        "",
        "This smoke shows the geometry-only versus actuator-feasible distinction and which "
        "actuator limit was violated per row. It is diagnostic only and does not run "
        "benchmark episodes.",
        "",
        "| Scenario | Verdict | Geom. clear | Phys. feasible | Clearance (m) | "
        "Stop dist. (m) | Violated limits |",
        "| --- | --- | :---: | :---: | ---: | ---: | --- |",
    ]
    for row in rows:
        violated = ", ".join(row["violated_limits"]) if row["violated_limits"] else "—"
        stop_dist = (
            "—" if row["stopping_distance_m"] is None else f"{row['stopping_distance_m']:.3f}"
        )
        lines.append(
            f"| {row['key']} | `{row['verdict']}` | "
            f"{'yes' if row['geometrically_clear'] else 'no'} | "
            f"{'yes' if row['physically_feasible'] else 'no'} | "
            f"{row['hazard_clearance_m']:.2f} | {stop_dist} | {violated} |"
        )
    lines.extend(
        [
            "",
            f"> {ACTUATOR_FEASIBILITY_CLAIM_BOUNDARY}",
            "",
        ]
    )
    return "\n".join(lines)


def _repo_relative(path: Path) -> str:
    """Return a repository-relative path when possible."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the actuator-feasibility diagnostic smoke and write artifacts."""
    args = _build_parser().parse_args(argv)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    scenarios = build_scenarios()
    rows = [run_scenario(scenario) for scenario in scenarios]
    summary = build_summary(rows, output_root=output_root)

    summary_path = output_root / "summary.json"
    markdown_path = output_root / "summary.md"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(render_markdown(summary), encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    # Sanity-check the default config loader path so config-first usage is exercised.
    _ = load_actuator_limits({"actuator_limits": {"schema_version": ACTUATOR_FEASIBILITY_SCHEMA}})
    raise SystemExit(main())
