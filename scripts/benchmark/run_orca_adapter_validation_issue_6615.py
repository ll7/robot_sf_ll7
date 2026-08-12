#!/usr/bin/env python3
"""Run the bounded, diagnostic-only ORCA adapter validation smoke.

This harness exercises the native ORCA adapter on a fixed set of synthetic
observations and writes the optional ``orca_adapter_trace.v1`` records plus a
small divergence summary.  It is intentionally not a benchmark campaign: no
planner ranking, success claim, or dissertation-hedge update is inferred.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from robot_sf.planner import socnav as socnav_module
from robot_sf.planner.socnav import ORCAPlannerAdapter, SocNavPlannerConfig

REPORT_SCHEMA_VERSION = "issue_6615_orca_adapter_validation.v1"
TRACE_SCHEMA_VERSION = "orca_adapter_trace.v1"


def _observation(
    *,
    goal: tuple[float, float],
    heading: float,
    pedestrians: tuple[tuple[float, float, float, float], ...] = (),
) -> dict[str, Any]:
    """Build one deterministic SocNav observation for the smoke cases."""
    count = max(1, len(pedestrians))
    positions = np.zeros((count, 2), dtype=np.float32)
    velocities = np.zeros((count, 2), dtype=np.float32)
    for index, (x, y, vx, vy) in enumerate(pedestrians):
        positions[index] = (x, y)
        velocities[index] = (vx, vy)
    return {
        "robot": {
            "position": np.array([0.0, 0.0], dtype=np.float32),
            "heading": np.array([heading], dtype=np.float32),
            "speed": np.array([0.0, 0.0], dtype=np.float32),
            "radius": np.array([0.3], dtype=np.float32),
        },
        "goal": {
            "current": np.asarray(goal, dtype=np.float32),
            "next": np.asarray(goal, dtype=np.float32),
        },
        "pedestrians": {
            "positions": positions,
            "velocities": velocities,
            "radius": np.full((count,), 0.3, dtype=np.float32),
            "count": np.array([float(len(pedestrians))], dtype=np.float32),
        },
        "map": {"size": np.array([10.0, 10.0], dtype=np.float32)},
        "sim": {"timestep": np.array([0.1], dtype=np.float32)},
    }


def _git_head() -> str:
    """Return the current checkout SHA for report provenance."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _markdown_report(payload: dict[str, Any]) -> str:
    """Render the compact human-readable smoke report."""
    lines = [
        "# Issue #6615 ORCA adapter-validation smoke",
        "",
        "Status: **diagnostic-only**; this is not benchmark evidence and cannot retire the dissertation ORCA hedge.",
        "",
        f"- report schema: `{payload['schema_version']}`",
        f"- adapter trace schema: `{payload['trace_schema_version']}`",
        f"- execution mode: `{payload['execution_mode']}`",
        f"- source commit: `{payload['source_commit']}`",
        "",
        "## Fixed smoke cases",
        "",
        "| case | linear command (m/s) | angular command (rad/s) |",
        "| --- | ---: | ---: |",
    ]
    for row in payload["cases"]:
        lines.append(
            f"| `{row['case']}` | {row['command_v_mps']:.6f} | {row['command_w_rad_s']:.6f} |"
        )
    summary = payload["adapter_trace_summary"]
    lines.extend(
        [
            "",
            "## Divergence summary",
            "",
            f"- samples: `{summary['sample_count']}`",
            f"- angle error mean / p50 / p95 (rad): `{summary.get('angle_error_rad_mean')}` / "
            f"`{summary.get('angle_error_rad_p50')}` / `{summary.get('angle_error_rad_p95')}`",
            f"- speed delta mean / p50 / p95 (m/s): `{summary.get('speed_delta_mps_mean')}` / "
            f"`{summary.get('speed_delta_mps_p50')}` / `{summary.get('speed_delta_mps_p95')}`",
            "",
            "Interpretation is limited to measurability of the holonomic-to-unicycle projection.",
        ]
    )
    return "\n".join(lines) + "\n"


def run(output_dir: Path) -> tuple[Path, Path]:
    """Run native fixed-case smoke and write JSON/Markdown reports."""
    if socnav_module.rvo2 is None:
        raise RuntimeError(
            "native rvo2 is unavailable; refusing to run the ORCA adapter smoke in fallback mode"
        )

    config = SocNavPlannerConfig(
        max_linear_speed=1.0,
        max_angular_speed=1.0,
        orca_adapter_trace_enabled=True,
    )
    adapter = ORCAPlannerAdapter(config=config, allow_fallback=False)
    cases = (
        ("aligned_clear", (5.0, 0.0), 0.0, ()),
        ("lateral_goal", (3.0, 2.0), 0.0, ()),
        ("crossing_pedestrian", (5.0, 0.0), 0.0, ((2.0, 0.8, -0.2, 0.0),)),
        ("goal_behind_robot", (4.0, 0.0), math.pi, ()),
    )
    case_rows: list[dict[str, Any]] = []
    for case_name, goal, heading, pedestrians in cases:
        observation = _observation(
            goal=goal,
            heading=heading,
            pedestrians=pedestrians,
        )
        linear, angular = adapter.plan(observation)
        case_rows.append(
            {
                "case": case_name,
                "command_v_mps": float(linear),
                "command_w_rad_s": float(angular),
            }
        )

    diagnostics = adapter.diagnostics()
    summary = diagnostics["adapter_trace_summary"]
    if diagnostics["adapter_trace_schema_version"] != TRACE_SCHEMA_VERSION:
        raise RuntimeError("unexpected adapter trace schema version")
    if summary["sample_count"] != len(cases):
        raise RuntimeError("adapter trace did not capture one record per smoke case")

    payload: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "status": "diagnostic_complete",
        "evidence_tier": "smoke_diagnostic",
        "execution_mode": "native",
        "fallback_used": False,
        "source_commit": _git_head(),
        "cases": case_rows,
        "adapter_trace_summary": summary,
        "adapter_trace": diagnostics["adapter_trace"],
        "claim_boundary": {
            "benchmark_evidence": False,
            "planner_quality_claim": False,
            "native_orca_equivalence_claim": False,
            "dissertation_hedge_update": False,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "orca_adapter_validation.json"
    markdown_path = output_dir / "orca_adapter_validation.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown_report(payload), encoding="utf-8")
    return json_path, markdown_path


def main() -> int:
    """Parse CLI arguments and run the bounded smoke."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    json_path, markdown_path = run(args.output_dir)
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
