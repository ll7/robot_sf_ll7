#!/usr/bin/env python3
"""Probe source parity for the Social-Navigation-PyEnvs ORCA integration."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class ScenarioParitySummary:
    """Parity summary for one upstream scenario."""

    name: str
    scenario: str
    seed: int
    trace_steps: int
    wrapper_mean_xy_error: float
    wrapper_max_xy_error: float
    oracle_mean_xy_error: float
    oracle_max_xy_error: float
    heading_velocity_mismatch_steps: int
    sample_rows: list[dict[str, Any]]


@dataclass
class ParityReport:
    """Structured ORCA parity report."""

    issue: int
    repo_root: str
    repo_remote_url: str
    verdict: str
    root_cause: str
    projection_role: str
    scenarios: list[ScenarioParitySummary]


def _uv_command() -> str:
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv executable not found on PATH")
    return uv


def _extract_remote_url(repo_root: Path) -> str:
    proc = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == 0 and proc.stdout.strip():
        return proc.stdout.strip()
    return "unknown"


def _scenario_specs() -> list[dict[str, Any]]:
    return [
        {
            "name": "circular_crossing_hsfm_new_guo",
            "scenario": "circular_crossing",
            "seed": 1002,
            "decision_limit": 5,
            "config": {
                "insert_robot": True,
                "human_policy": "hsfm_new_guo",
                "headless": True,
                "runge_kutta": False,
                "robot_visible": True,
                "robot_radius": 0.3,
                "circle_radius": 7,
                "n_actors": 3,
                "randomize_human_positions": True,
                "randomize_human_attributes": False,
            },
        },
        {
            "name": "parallel_traffic_orca",
            "scenario": "parallel_traffic",
            "seed": 1000,
            "decision_limit": 8,
            "config": {
                "insert_robot": True,
                "human_policy": "orca",
                "headless": True,
                "runge_kutta": False,
                "robot_visible": True,
                "robot_radius": 0.3,
                "traffic_length": 14,
                "traffic_height": 3,
                "n_actors": 10,
                "randomize_human_attributes": False,
            },
        },
    ]


def _child_script() -> str:
    return textwrap.dedent(
        r"""
        import json
        import math
        import os
        import sys
        from pathlib import Path

        import numpy as np

        np.NaN = np.nan

        spec = json.loads(os.environ["SOCNAV_PARITY_SPEC"])
        repo_root = Path(os.environ["SOCNAV_UPSTREAM_ROOT"]).resolve()
        sys.path.insert(0, str(repo_root))

        from social_gym.social_nav_sim import SocialNavSim
        from social_gym.src.utils import is_multiple
        from robot_sf.planner.social_navigation_pyenvs_orca import (
            SocialNavigationPyEnvsORCAConfig,
            SocialNavigationPyEnvsORCAAdapter,
        )

        np.random.seed(int(spec["seed"]))
        sim = SocialNavSim(
            config_data=spec["config"],
            scenario=spec["scenario"],
            parallelize_robot=False,
            parallelize_humans=False,
        )
        sim.set_time_step(1 / 20)
        sim.set_robot_time_step(1 / 4)
        sim.set_robot_policy(policy_name="orca", crowdnav_policy=True)
        adapter = SocialNavigationPyEnvsORCAAdapter(
            SocialNavigationPyEnvsORCAConfig(repo_root=repo_root)
        )

        rows = []
        max_steps = 600
        steps = 0
        while len(rows) < int(spec["decision_limit"]) and steps < max_steps:
            if is_multiple(sim.sim_t, sim.robot.time_step):
                ob = [human.get_observable_state() for human in sim.humans]
                action = sim.robot.act(ob)
                speed = float(np.linalg.norm(sim.robot.linear_velocity))
                actual_heading = float(sim.robot.yaw)
                inferred_heading = (
                    float(math.atan2(sim.robot.linear_velocity[1], sim.robot.linear_velocity[0]))
                    if speed > 1e-8
                    else actual_heading
                )
                ped_positions = np.array([h.position for h in sim.humans], dtype=float)
                ped_velocities = np.array([h.linear_velocity for h in sim.humans], dtype=float)
                ped_radius = float(sim.humans[0].radius) if sim.humans else 0.3

                obs = {
                    "robot": {
                        "position": sim.robot.position.copy(),
                        "heading": [actual_heading],
                        "speed": [speed],
                        "velocity_xy": np.array(sim.robot.linear_velocity, dtype=float),
                        "radius": [float(sim.robot.radius)],
                    },
                    "goal": {"current": np.array(sim.robot.goals[0], dtype=float)},
                    "pedestrians": {
                        "positions": ped_positions,
                        "velocities": ped_velocities,
                        "count": [len(sim.humans)],
                        "radius": [ped_radius],
                    },
                }
                _, _, wrapper_meta = adapter.act(obs, time_step=sim.robot.time_step)

                oracle_obs = {
                    "robot": {
                        "position": sim.robot.position.copy(),
                        "heading": [inferred_heading],
                        "speed": [speed],
                        "velocity_xy": np.array(sim.robot.linear_velocity, dtype=float),
                        "radius": [float(sim.robot.radius)],
                    },
                    "goal": {"current": np.array(sim.robot.goals[0], dtype=float)},
                    "pedestrians": obs["pedestrians"],
                }
                _, _, oracle_meta = adapter.act(oracle_obs, time_step=sim.robot.time_step)

                rows.append(
                    {
                        "t": float(sim.sim_t),
                        "robot_yaw": actual_heading,
                        "robot_speed": speed,
                        "robot_velocity_xy": [
                            float(sim.robot.linear_velocity[0]),
                            float(sim.robot.linear_velocity[1]),
                        ],
                        "velocity_heading_rad": inferred_heading,
                        "upstream_action_xy": [float(action.vx), float(action.vy)],
                        "wrapper_action_xy": [float(x) for x in wrapper_meta["upstream_action_xy"]],
                        "wrapper_projected_vw": [
                            float(x) for x in wrapper_meta["projected_command_vw"]
                        ],
                        "oracle_heading_action_xy": [
                            float(x) for x in oracle_meta["upstream_action_xy"]
                        ],
                    }
                )
            sim.update()
            steps += 1

        print(json.dumps({"rows": rows}, indent=2))
        """
    )


def _run_child_scenario(
    repo_root: Path, spec: dict[str, Any], timeout_seconds: int
) -> dict[str, Any]:
    env = os.environ.copy()
    env["SOCNAV_PARITY_SPEC"] = json.dumps(spec)
    env["SOCNAV_UPSTREAM_ROOT"] = str(repo_root.resolve())
    proc = subprocess.run(
        [
            _uv_command(),
            "run",
            "--with",
            "socialforce",
            "python",
            "-c",
            _child_script(),
        ],
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Parity child run failed for {spec['name']}: {(proc.stderr or proc.stdout).strip()[-4000:]}"
        )
    return json.loads(proc.stdout)


def _xy_error(a: list[float], b: list[float]) -> float:
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    return float((dx * dx + dy * dy) ** 0.5)


def _summarize_scenario(spec: dict[str, Any], payload: dict[str, Any]) -> ScenarioParitySummary:
    rows = list(payload.get("rows", []))
    wrapper_errors = [
        _xy_error(row["upstream_action_xy"], row["wrapper_action_xy"]) for row in rows
    ]
    oracle_errors = [
        _xy_error(row["upstream_action_xy"], row["oracle_heading_action_xy"]) for row in rows
    ]
    mismatch_steps = 0
    for row in rows:
        speed = float(row["robot_speed"])
        if speed <= 1e-8:
            continue
        if abs(float(row["robot_yaw"]) - float(row["velocity_heading_rad"])) > 1e-3:
            mismatch_steps += 1
    return ScenarioParitySummary(
        name=str(spec["name"]),
        scenario=str(spec["scenario"]),
        seed=int(spec["seed"]),
        trace_steps=len(rows),
        wrapper_mean_xy_error=(sum(wrapper_errors) / len(wrapper_errors))
        if wrapper_errors
        else 0.0,
        wrapper_max_xy_error=max(wrapper_errors) if wrapper_errors else 0.0,
        oracle_mean_xy_error=(sum(oracle_errors) / len(oracle_errors)) if oracle_errors else 0.0,
        oracle_max_xy_error=max(oracle_errors) if oracle_errors else 0.0,
        heading_velocity_mismatch_steps=mismatch_steps,
        sample_rows=rows[:5],
    )


def _determine_verdict(scenarios: list[ScenarioParitySummary]) -> tuple[str, str, str]:
    if any(
        item.wrapper_mean_xy_error > 0.05 and item.oracle_mean_xy_error < 1e-4 for item in scenarios
    ):
        return (
            "adapter has material contract mismatch",
            (
                "Robot SF SocNav observations expose heading plus scalar speed, while the upstream "
                "ORCA path consumes full planar self velocity. On upstream scenario snapshots, the "
                "real adapter diverges when robot yaw and velocity heading differ, but an "
                "oracle-heading control restores parity."
            ),
            (
                "The ActionXY->unicycle_vw projection is still a second mismatch layer, but the "
                "raw ActionXY parity already fails before projection on nontrivial upstream traces."
            ),
        )
    if all(item.wrapper_max_xy_error < 1e-4 for item in scenarios):
        return (
            "adapter appears source-faithful but benchmark-misaligned",
            "Raw upstream ActionXY traces match across the tested upstream scenarios.",
            (
                "Remaining performance differences would then be explained primarily by scenario "
                "mismatch and downstream unicycle execution."
            ),
        )
    return (
        "inconclusive because scenario parity could not be achieved",
        "The tested scenarios did not provide a stable enough signal to separate contract mismatch from scenario mismatch.",
        "Projection impact remains unresolved.",
    )


def run_probe(repo_root: Path, timeout_seconds: int) -> ParityReport:
    """Run the ORCA parity probe against fixed upstream scenarios."""
    scenarios = [
        _summarize_scenario(spec, _run_child_scenario(repo_root, spec, timeout_seconds))
        for spec in _scenario_specs()
    ]
    verdict, root_cause, projection_role = _determine_verdict(scenarios)
    return ParityReport(
        issue=649,
        repo_root=str(repo_root),
        repo_remote_url=_extract_remote_url(repo_root),
        verdict=verdict,
        root_cause=root_cause,
        projection_role=projection_role,
        scenarios=scenarios,
    )


def _render_markdown(report: ParityReport) -> str:
    scenario_sections: list[str] = []
    for item in report.scenarios:
        sample_table = "\n".join(
            (
                f"| {row['t']:.2f} | {row['upstream_action_xy']} | {row['wrapper_action_xy']} | "
                f"{row['oracle_heading_action_xy']} | {row['robot_yaw']:.3f} | "
                f"{row['velocity_heading_rad']:.3f} |"
            )
            for row in item.sample_rows
        )
        scenario_sections.append(
            "\n".join(
                [
                    f"### {item.name}",
                    "",
                    f"- scenario: `{item.scenario}`",
                    f"- seed: `{item.seed}`",
                    f"- traced decision steps: `{item.trace_steps}`",
                    f"- wrapper mean ActionXY error: `{item.wrapper_mean_xy_error:.4f}`",
                    f"- wrapper max ActionXY error: `{item.wrapper_max_xy_error:.4f}`",
                    f"- oracle-heading mean ActionXY error: `{item.oracle_mean_xy_error:.4f}`",
                    f"- heading/velocity mismatch steps: `{item.heading_velocity_mismatch_steps}`",
                    "",
                    "| t | upstream ActionXY | adapter ActionXY | oracle-heading ActionXY | yaw | velocity heading |",
                    "|---:|---|---|---|---:|---:|",
                    sample_table or "| n/a | n/a | n/a | n/a | n/a | n/a |",
                ]
            )
        )

    if report.verdict == "adapter has material contract mismatch":
        interpretation = [
            "- `circular_crossing_hsfm_new_guo` stays parity-clean because the robot moves in a straight line, so yaw and velocity heading remain aligned.",
            "- `parallel_traffic_orca` exposes the real mismatch: once upstream ORCA produces lateral velocity components, the Robot SF wrapper cannot reconstruct the same self velocity from heading plus scalar speed.",
            "- The oracle-heading control shows the upstream state mapping is otherwise close; the failure is concentrated in the self-velocity contract, not in pedestrian packing.",
        ]
    elif report.verdict == "adapter appears source-faithful but benchmark-misaligned":
        interpretation = [
            "- `circular_crossing_hsfm_new_guo` remains parity-clean on the straight-line trace.",
            "- `parallel_traffic_orca` now also matches upstream raw `ActionXY`, even though yaw and velocity heading diverge on several steps.",
            "- The remaining benchmark gap therefore sits after raw upstream policy inference: scenario mismatch, downstream unicycle execution, or both.",
        ]
    else:
        interpretation = [
            "- The tested scenarios were not sufficient to separate contract mismatch from downstream execution effects.",
            "- Use the JSON report and traced samples to decide which scenario or contract dimension needs a tighter probe next.",
        ]

    return "\n".join(
        [
            "# Issue 649 Social-Navigation-PyEnvs ORCA Parity Probe",
            "",
            f"Verdict: `{report.verdict}`",
            "",
            "## Root Cause",
            "",
            report.root_cause,
            "",
            "## Projection Role",
            "",
            report.projection_role,
            "",
            "## Scenarios",
            "",
            *scenario_sections,
            "",
            "## Interpretation",
            "",
            *interpretation,
        ]
    )


def main() -> None:
    """Run the parity probe and write JSON/Markdown artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    args = parser.parse_args()

    report = run_probe(args.repo_root, timeout_seconds=args.timeout_seconds)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(_render_markdown(report), encoding="utf-8")


if __name__ == "__main__":
    main()
