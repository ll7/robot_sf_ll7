"""Build the issue #2168 AMMV-aware Social Force paired diagnostic summary.

This tool deliberately keeps the evidence compact. It consumes disposable benchmark JSONL
outputs, runs one direct mechanism probe that exposes planner metadata, and writes a tracked
diagnostic pack. The output is not benchmark-strength or paper-facing evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import yaml

from robot_sf.baselines.social_force import Observation, SFPlannerConfig, SocialForcePlanner


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _metric_mean(records: list[dict[str, Any]], key: str) -> float | None:
    values = [record.get("metrics", {}).get(key) for record in records]
    numeric = [float(value) for value in values if isinstance(value, int | float | bool)]
    if not numeric:
        return None
    return sum(numeric) / len(numeric)


def _metric_min(records: list[dict[str, Any]], key: str) -> float | None:
    values = [record.get("metrics", {}).get(key) for record in records]
    numeric = [float(value) for value in values if isinstance(value, int | float | bool)]
    if not numeric:
        return None
    return min(numeric)


def _summarize_records(label: str, path: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    statuses: dict[str, int] = {}
    for record in records:
        status = str(record.get("status", "unknown"))
        statuses[status] = statuses.get(status, 0) + 1
    return {
        "source_label": label,
        "source_note": "worktree-local disposable benchmark JSONL; tracked evidence keeps checksum only",
        "sha256": _sha256(path),
        "episode_count": len(records),
        "scenario_ids": sorted({str(record.get("scenario_id")) for record in records}),
        "seeds": [record.get("seed") for record in records],
        "status_counts": statuses,
        "success_rate": _metric_mean(records, "success"),
        "collision_count_total": sum(
            int(record.get("metrics", {}).get("collisions") or 0) for record in records
        ),
        "min_clearance_min_m": _metric_min(records, "min_clearance"),
        "mean_clearance_mean_m": _metric_mean(records, "mean_clearance"),
        "avg_speed_mean_mps": _metric_mean(records, "avg_speed"),
        "ped_force_mean_mean": _metric_mean(records, "ped_force_mean"),
        "ped_force_q95_mean": _metric_mean(records, "ped_force_q95"),
        "execution_modes": sorted(
            {
                str(
                    record.get("algorithm_metadata", {})
                    .get("planner_kinematics", {})
                    .get("execution_mode", "unknown")
                )
                for record in records
            }
        ),
        "ammv_metadata_present": any(
            "ammv_force_magnitude" in record.get("algorithm_metadata", {}) for record in records
        ),
    }


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a mapping")
    return payload


def _speed(action: dict[str, float]) -> float:
    if "vx" in action and "vy" in action:
        return math.hypot(float(action["vx"]), float(action["vy"]))
    return float(action.get("v", 0.0))


def _mechanism_probe_spec(probe_name: str) -> dict[str, Any]:
    """Return a deterministic direct-mechanism probe specification."""
    probes: dict[str, dict[str, Any]] = {
        "issue_2168_close_front_agent_probe": {
            "description": "Close static pedestrian in front of the robot.",
            "seed": 42,
            "steps": 20,
            "dt": 0.1,
            "close_clearance_threshold_m": 0.5,
            "robot": {
                "position": [0.0, 0.0],
                "velocity": [1.0, 0.0],
                "goal": [4.0, 0.0],
                "radius": 0.3,
            },
            "agents": [
                {
                    "position": [0.3, 0.1],
                    "velocity": [0.0, 0.0],
                    "radius": 0.3,
                }
            ],
        },
        "issue_3202_anticipatory_crossing_probe": {
            "description": (
                "Near-field crossing pedestrian at the robot path conflict point; "
                "direct SocialForcePlanner diagnostic only."
            ),
            "seed": 3202,
            "steps": 24,
            "dt": 0.1,
            "close_clearance_threshold_m": 0.5,
            "robot": {
                "position": [0.0, 0.0],
                "velocity": [1.0, 0.0],
                "goal": [4.0, 0.0],
                "radius": 0.3,
            },
            "agents": [
                {
                    "position": [0.34, 0.18],
                    "velocity": [0.0, -0.45],
                    "radius": 0.3,
                }
            ],
        },
    }
    if probe_name not in probes:
        choices = ", ".join(sorted(probes))
        raise ValueError(f"Unknown mechanism probe '{probe_name}'. Expected one of: {choices}")
    return probes[probe_name]


def _min_robot_agent_clearance(
    robot_pos: list[float],
    agents: list[dict[str, Any]],
    *,
    robot_radius: float,
) -> float:
    """Return the minimum clearance between the robot and configured agents."""
    clearances = []
    for agent in agents:
        agent_radius = float(agent.get("radius", 0.3))
        clearances.append(
            math.dist(robot_pos, list(agent["position"])) - robot_radius - agent_radius
        )
    return min(clearances) if clearances else float("inf")


def _run_mechanism_probe(
    ammv_config_path: Path,
    *,
    probe_name: str = "issue_2168_close_front_agent_probe",
) -> dict[str, Any]:
    """Run a deterministic direct probe that exposes AMMV metadata."""
    spec = _mechanism_probe_spec(probe_name)
    dt = float(spec["dt"])
    steps = int(spec["steps"])
    close_clearance_threshold_m = float(spec["close_clearance_threshold_m"])
    robot_spec = dict(spec["robot"])
    robot_radius = float(robot_spec.get("radius", 0.3))
    agents = [dict(agent) for agent in spec["agents"]]
    obs = Observation(
        dt=dt,
        robot=robot_spec,
        agents=agents,
        obstacles=[],
    )
    planners = {
        "default_social_force": SocialForcePlanner(SFPlannerConfig(), seed=42),
        "ammv_social_force": SocialForcePlanner(_load_config(ammv_config_path), seed=42),
    }
    traces: dict[str, dict[str, Any]] = {}
    for name, planner in planners.items():
        robot_pos = list(robot_spec["position"])
        robot_vel = list(robot_spec["velocity"])
        speeds: list[float] = []
        lateral_velocities: list[float] = []
        clearances: list[float] = []
        ammv_force_magnitudes: list[float] = []
        intrusion_counts: list[int] = []
        for _ in range(steps):
            step_obs = Observation(
                dt=dt,
                robot={
                    "position": robot_pos,
                    "velocity": robot_vel,
                    "goal": robot_spec["goal"],
                    "radius": robot_spec.get("radius", 0.3),
                },
                agents=obs.agents,
                obstacles=[],
            )
            action = planner.step(step_obs)
            vx = float(action.get("vx", action.get("v", 0.0)))
            vy = float(action.get("vy", 0.0))
            robot_pos = [robot_pos[0] + vx * dt, robot_pos[1] + vy * dt]
            robot_vel = [vx, vy]
            clearance = _min_robot_agent_clearance(
                robot_pos,
                agents,
                robot_radius=robot_radius,
            )
            metadata = planner.get_metadata()
            speeds.append(_speed(action))
            lateral_velocities.append(vy)
            clearances.append(clearance)
            ammv_force_magnitudes.append(float(metadata.get("ammv_force_magnitude", 0.0) or 0.0))
            diagnostics = metadata.get("ammv_diagnostics") or {}
            intrusion_counts.append(int(diagnostics.get("intrusion_count", 0) or 0))
        traces[name] = {
            "mean_robot_speed_mps": sum(speeds) / len(speeds),
            "max_abs_lateral_velocity_mps": max(abs(value) for value in lateral_velocities),
            "final_robot_lateral_offset_m": robot_pos[1],
            "min_robot_ped_clearance_m": min(clearances),
            "interaction_duration_steps_clearance_lt_0p5m": sum(
                1 for value in clearances if value < close_clearance_threshold_m
            ),
            "max_ammv_force_magnitude": max(ammv_force_magnitudes),
            "max_intrusion_count": max(intrusion_counts),
        }
    default = traces["default_social_force"]
    ammv = traces["ammv_social_force"]
    paired_delta = {
        "mean_robot_speed_mps": ammv["mean_robot_speed_mps"] - default["mean_robot_speed_mps"],
        "max_abs_lateral_velocity_mps": (
            ammv["max_abs_lateral_velocity_mps"] - default["max_abs_lateral_velocity_mps"]
        ),
        "final_robot_lateral_offset_m": (
            ammv["final_robot_lateral_offset_m"] - default["final_robot_lateral_offset_m"]
        ),
        "min_robot_ped_clearance_m": (
            ammv["min_robot_ped_clearance_m"] - default["min_robot_ped_clearance_m"]
        ),
    }
    behavioral_delta_found = any(abs(value) > 1e-9 for value in paired_delta.values())
    ammv_force_activated = ammv["max_ammv_force_magnitude"] > 0.0
    verdict = (
        "behavioral_delta_found"
        if behavioral_delta_found and ammv_force_activated
        else "no_behavioral_delta_found"
    )
    return {
        "name": probe_name,
        "description": spec["description"],
        "seed": int(spec["seed"]),
        "steps": steps,
        "dt": dt,
        "status": "diagnostic",
        "traces": traces,
        "paired_delta": paired_delta,
        "verdict": verdict,
        "unsupported_requested_fields": {
            "pedestrian_lateral_deviation": (
                "unsupported: SocialForcePlanner is a robot planner and does not update pedestrian "
                "trajectories in this direct mechanism probe"
            ),
            "pedestrian_speed_adaptation": (
                "unsupported: pedestrian dynamics are simulator-owned, not planner-owned"
            ),
        },
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _redacted_benchmark_command(
    *,
    scenario_config: Path,
    output_label: str,
    ammv_config: Path | None = None,
    base_seed: int = 111,
    horizon: int = 100,
) -> str:
    command = (
        "uv run robot_sf_bench run --matrix "
        f"{scenario_config.as_posix()} --out <{output_label}> "
        f"--base-seed {base_seed} --repeats 1 --horizon {horizon} --dt 0.1 --record-forces "
        "--no-video --video-renderer none --algo social_force "
    )
    if ammv_config is not None:
        command += f"--algo-config {ammv_config.as_posix()} "
    return command + "--workers 1 --no-resume --structured-output json"


def _write_outputs(summary: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    readme_path = output_dir / "README.md"
    checksums_path = output_dir / "SHA256SUMS"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    default = summary["benchmark_probe"]["default_social_force"]
    ammv = summary["benchmark_probe"]["ammv_social_force"]
    mechanism = summary["mechanism_probe"]
    readme_path.write_text(
        "\n".join(
            [
                f"# {summary['title']}",
                "",
                "This evidence pack records a local diagnostic, not benchmark-strength or",
                "paper-facing evidence. The benchmark probe ran the same single scenario",
                "with default Social Force and AMMV-aware Social Force; the mechanism probe",
                "checks whether the AMMV term activates in a controlled close-agent setup.",
                "",
                "## Benchmark Probe",
                "",
                f"- Scenario: `{summary['scenario_id']}`.",
                f"- Seeds: `{default['seeds']}`.",
                f"- Default status counts: `{default['status_counts']}`.",
                f"- AMMV status counts: `{ammv['status_counts']}`.",
                f"- Default min clearance: `{default['min_clearance_min_m']:.6f}` m.",
                f"- AMMV min clearance: `{ammv['min_clearance_min_m']:.6f}` m.",
                f"- Default mean speed: `{default['avg_speed_mean_mps']:.6f}` m/s.",
                f"- AMMV mean speed: `{ammv['avg_speed_mean_mps']:.6f}` m/s.",
                f"- AMMV metadata surfaced in episode JSONL: `{ammv['ammv_metadata_present']}`.",
                "",
                "## Mechanism Probe",
                "",
                f"- Probe: `{mechanism['name']}`, seed `{mechanism['seed']}`.",
                "- AMMV max force magnitude: "
                f"`{mechanism['traces']['ammv_social_force']['max_ammv_force_magnitude']:.6f}`.",
                "- AMMV max intrusion count: "
                f"`{mechanism['traces']['ammv_social_force']['max_intrusion_count']}`.",
                f"- Verdict: `{mechanism['verdict']}`.",
                "- Mean speed delta: "
                f"`{mechanism['paired_delta']['mean_robot_speed_mps']:.6f}` m/s.",
                "- Max lateral-velocity delta: "
                f"`{mechanism['paired_delta']['max_abs_lateral_velocity_mps']:.6f}` m/s.",
                "- Min robot-pedestrian clearance delta: "
                f"`{mechanism['paired_delta']['min_robot_ped_clearance_m']:.6f}` m.",
                "- Final robot lateral-offset delta: "
                f"`{mechanism['paired_delta']['final_robot_lateral_offset_m']:.6f}` m.",
                "",
                "## Interpretation",
                "",
                "The named benchmark slice is contextual diagnostic output. The direct",
                "mechanism probe is the AMMV-specific same-seed comparison surface because",
                "it runs `SocialForcePlanner` directly and exposes AMMV force diagnostics.",
                "Pedestrian lateral deviation and speed adaptation remain unsupported by",
                "this robot-planner-only probe. Classify this result as diagnostic only.",
                "",
                "## Limitations",
                "",
                *[f"- {limitation}" for limitation in summary["limitations"]],
                "",
            ]
        ),
        encoding="utf-8",
    )
    checksum_lines = [
        f"{_sha256(summary_path)}  ./summary.json",
        f"{_sha256(readme_path)}  ./README.md",
    ]
    checksums_path.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")


def _infer_scenario_id(
    *,
    default_records: list[dict[str, Any]],
    ammv_records: list[dict[str, Any]],
    fallback: Path,
) -> str:
    """Infer a compact scenario identifier from paired episode records."""
    scenario_ids = {
        str(record.get("scenario_id"))
        for record in default_records + ammv_records
        if record.get("scenario_id") is not None
    }
    if len(scenario_ids) == 1:
        return next(iter(scenario_ids))
    if scenario_ids:
        return ",".join(sorted(scenario_ids))
    return fallback.stem


def main() -> int:
    """Run the paired diagnostic and write the tracked summary pack."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--default-jsonl", required=True, type=Path)
    parser.add_argument("--ammv-jsonl", required=True, type=Path)
    parser.add_argument("--ammv-config", required=True, type=Path)
    parser.add_argument("--scenario-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--mechanism-probe",
        default="issue_2168_close_front_agent_probe",
        choices=sorted(
            [
                "issue_2168_close_front_agent_probe",
                "issue_3202_anticipatory_crossing_probe",
            ]
        ),
    )
    parser.add_argument("--base-seed", type=int, default=111)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument(
        "--schema-version", default="issue_2168_ammv_social_force_pair_diagnostic.v1"
    )
    parser.add_argument(
        "--title",
        default="Issue #2168 AMMV-Aware Social Force Pair Diagnostic 2026-06-03",
    )
    parser.add_argument("--scenario-id")
    args = parser.parse_args()

    default_records = _read_jsonl(args.default_jsonl)
    ammv_records = _read_jsonl(args.ammv_jsonl)
    scenario_id = args.scenario_id or _infer_scenario_id(
        default_records=default_records,
        ammv_records=ammv_records,
        fallback=args.scenario_config,
    )
    limitations = [
        "One selected diagnostic slice only; no benchmark-strength or paper-facing claim.",
        "Benchmark runner episode records did not surface AMMV force/intrusion metadata.",
        "Both planner rows ran in adapter mode under differential-drive benchmark execution.",
        "Pedestrian lateral deviation and speed adaptation are unsupported by the direct robot-planner mechanism probe.",
    ]
    if args.mechanism_probe == "issue_3202_anticipatory_crossing_probe":
        limitations.append(
            "`scripts/validation/run_multi_amv_smoke.py` is not used because it is a "
            "multi-robot goal-controller smoke, not a default-vs-AMMV Social Force harness."
        )
        limitations.append(
            "The direct probe resolves #3202 as diagnostic mechanism evidence; it does not "
            "promote AMMV to benchmark-valid comparison evidence."
        )
    summary = {
        "schema_version": args.schema_version,
        "title": args.title,
        "classification": "diagnostic",
        "benchmark_evidence": False,
        "paper_facing": False,
        "git_head": _git_head(),
        "scenario_id": scenario_id,
        "scenario_config": args.scenario_config.as_posix(),
        "ammv_config": args.ammv_config.as_posix(),
        "commands": {
            "validate": (
                f"uv run robot_sf_bench validate-config --matrix {args.scenario_config.as_posix()}"
            ),
            "preview": (
                "uv run robot_sf_bench preview-scenarios --matrix "
                f"{args.scenario_config.as_posix()}"
            ),
            "default_run": (
                _redacted_benchmark_command(
                    scenario_config=args.scenario_config,
                    output_label="worktree-local-default-jsonl",
                    base_seed=args.base_seed,
                    horizon=args.horizon,
                )
            ),
            "ammv_run": (
                _redacted_benchmark_command(
                    scenario_config=args.scenario_config,
                    output_label="worktree-local-ammv-jsonl",
                    ammv_config=args.ammv_config,
                    base_seed=args.base_seed,
                    horizon=args.horizon,
                )
            ),
        },
        "benchmark_probe": {
            "default_social_force": _summarize_records(
                "worktree-local-default-jsonl",
                args.default_jsonl,
                default_records,
            ),
            "ammv_social_force": _summarize_records(
                "worktree-local-ammv-jsonl",
                args.ammv_jsonl,
                ammv_records,
            ),
        },
        "mechanism_probe": _run_mechanism_probe(
            args.ammv_config,
            probe_name=args.mechanism_probe,
        ),
        "limitations": limitations,
    }
    _write_outputs(summary, args.output_dir)
    print(json.dumps({"output_dir": args.output_dir.as_posix(), "classification": "diagnostic"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
