#!/usr/bin/env python3
"""Run a synthetic SocNavBench personal-space velocity sensitivity smoke."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOCNAV_ROOT = ROOT / "third_party" / "socnavbench"
DEFAULT_OUTPUT_DIR = ROOT / "output/validation/issue_1323_socnav_personal_space_velocity/latest"


@dataclass(frozen=True)
class SmokeRow:
    """One synthetic personal-space cost evaluation."""

    velocity_aware: bool
    agent_speed: float
    velocity_scale: float
    value: float
    delta_from_default: float


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agent-speed",
        type=float,
        default=2.0,
        help="Synthetic non-ego agent speed used for enabled-mode rows.",
    )
    parser.add_argument(
        "--velocity-scale",
        dest="velocity_scales",
        action="append",
        type=float,
        default=None,
        help="Velocity scale to evaluate. Repeat to override the default sweep.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for summary.json and summary.md outputs.",
    )
    return parser.parse_args(argv)


def _repo_relative(path: Path) -> str:
    """Return a repository-relative path when possible."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _with_socnav_imports() -> None:
    """Prepare upstream SocNavBench imports that depend on cwd-relative params."""
    sys.path.insert(0, str(SOCNAV_ROOT))
    os.chdir(SOCNAV_ROOT)


def _trajectory_at(x: float, y: float):
    """Create a one-step ego trajectory at ``(x, y)``."""
    from trajectory.trajectory import Trajectory

    return Trajectory.from_pos3_array(np.array([[[x, y, 0.0]]], dtype=np.float32))


def _sim_state_for_agent(*, heading: float, speed: float):
    """Create a one-agent SocNavBench sim state for synthetic objective evaluation."""
    from simulators.sim_state import AgentState, SimState
    from trajectory.trajectory import SystemConfig

    agent = AgentState(
        name="ped_0",
        goal_config=None,
        start_config=None,
        current_config=SystemConfig.from_pos3([0.0, 0.0, heading], v=speed),
    )
    return SimState(environment=None, pedestrians={"ped_0": agent}, robots={})


def _personal_space_value(
    *,
    velocity_aware: bool,
    agent_speed: float,
    velocity_scale: float,
) -> float:
    """Evaluate the objective at a fixed point beside the synthetic pedestrian."""
    from dotmap import DotMap
    from objectives.personal_space_cost import PersonalSpaceCost

    objective = PersonalSpaceCost(
        DotMap(
            psc_scale=1.0,
            use_agent_velocity=velocity_aware,
            agent_velocity_scale=velocity_scale,
            min_agent_speed=1e-3,
        )
    )
    values = objective.evaluate_objective(
        _trajectory_at(0.0, 1.0),
        {0: _sim_state_for_agent(heading=0.0, speed=agent_speed)},
    )
    return float(values[0, 0])


def _build_rows(agent_speed: float, velocity_scales: tuple[float, ...]) -> list[SmokeRow]:
    """Build the default-off row, zero-speed fallback row, and enabled sweep rows."""
    default_value = _personal_space_value(
        velocity_aware=False,
        agent_speed=agent_speed,
        velocity_scale=1.0,
    )
    rows = [
        SmokeRow(
            velocity_aware=False,
            agent_speed=agent_speed,
            velocity_scale=1.0,
            value=default_value,
            delta_from_default=0.0,
        )
    ]

    zero_value = _personal_space_value(
        velocity_aware=True,
        agent_speed=0.0,
        velocity_scale=1.0,
    )
    rows.append(
        SmokeRow(
            velocity_aware=True,
            agent_speed=0.0,
            velocity_scale=1.0,
            value=zero_value,
            delta_from_default=zero_value - default_value,
        )
    )

    for velocity_scale in velocity_scales:
        value = _personal_space_value(
            velocity_aware=True,
            agent_speed=agent_speed,
            velocity_scale=velocity_scale,
        )
        rows.append(
            SmokeRow(
                velocity_aware=True,
                agent_speed=agent_speed,
                velocity_scale=velocity_scale,
                value=value,
                delta_from_default=value - default_value,
            )
        )
    return rows


def _failures(rows: list[SmokeRow]) -> list[str]:
    """Return smoke-gate failures for the synthetic sweep."""
    default = rows[0].value
    zero = rows[1].value
    enabled_rows = sorted(rows[2:], key=lambda row: row.velocity_scale)
    failures: list[str] = []
    if not np.isclose(zero, default):
        failures.append("zero-speed velocity-aware row did not fall back to default")
    if not any(not np.isclose(row.value, default) for row in enabled_rows):
        failures.append("enabled velocity-aware sweep did not differ from default")
    for row in enabled_rows:
        if not np.isfinite(row.value):
            failures.append(
                f"enabled velocity-aware row was not finite for scale={row.velocity_scale}"
            )
    return failures


def _write_markdown(path: Path, rows: list[SmokeRow], failures: list[str]) -> None:
    """Write a compact human-readable smoke summary."""
    lines = [
        "# Issue 1323 SocNavBench Personal-Space Velocity Smoke",
        "",
        "| velocity_aware | agent_speed | velocity_scale | value | delta_from_default |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        aware = str(row.velocity_aware).lower()
        lines.append(
            f"| {aware} | {row.agent_speed:.3f} | {row.velocity_scale:.3f} | "
            f"{row.value:.6f} | {row.delta_from_default:.6f} |"
        )
    lines.extend(["", "## Result", ""])
    if failures:
        lines.extend(f"- FAIL: {failure}" for failure in failures)
    else:
        lines.append("- PASS: enabled positive-speed sweep contains finite, non-default rows.")
        lines.append("- PASS: zero-speed enabled mode falls back to default behavior.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Run the synthetic sweep and write summary artifacts."""
    os.environ.setdefault("LOGURU_LEVEL", "WARNING")
    args = _parse_args(argv)
    velocity_scales = tuple(args.velocity_scales or (0.5, 1.0, 2.0))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    original_cwd = Path.cwd()
    try:
        _with_socnav_imports()
        rows = _build_rows(args.agent_speed, velocity_scales)
    finally:
        os.chdir(original_cwd)

    failures = _failures(rows)
    summary: dict[str, Any] = {
        "passed": not failures,
        "failures": failures,
        "rows": [asdict(row) for row in rows],
    }
    summary_json = output_dir / "summary.json"
    summary_md = output_dir / "summary.md"
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(summary_md, rows, failures)

    print(json.dumps({**summary, "summary_json": _repo_relative(summary_json)}, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
