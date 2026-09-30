"""Dev-only B1-B4 diagnostics through the native map episode executor.

Run separately on the base and changed checkout with --output outside the checkout.
No fallback or held-out seeds; the three release scenarios and ten seeds are fixed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path

import numpy as np
import yaml
from loguru import logger

from robot_sf.benchmark.map_runner.map_runner import _build_policy, _run_map_episode
from robot_sf.planner import socnav_sampling_v2 as sampling
from robot_sf.planner.socnav_base import SamplingPlannerAdapter
from robot_sf.planner.socnav_sacadrl import SACADRLPlannerAdapter, _sacadrl_actions
from robot_sf.robot.differential_drive import DifferentialDriveRobot, DifferentialDriveSettings
from robot_sf.training.scenario_loader import load_scenarios

SCENARIOS = (
    "classic_group_crossing_high",
    "classic_doorway_medium",
    "classic_head_on_corridor_medium",
)
MATRIX = Path("configs/scenarios/classic_interactions_francis2023_release_0_0_8_v1.yaml")
CONFIGS = {
    "sacadrl": Path("configs/algos/socnav_release_v0_0_8.yaml"),
    "socnav_sampling": Path("configs/algos/socnav_sampling_release_v0_0_8.yaml"),
}


def native_candidate(start, heading, speed0, target, speed, seconds, dt, limits, settings, omega):  # noqa: PLR0913
    """Independent oracle: command the actual robot, including wheel odometry."""
    gain, rate, _, _ = limits
    drive = DifferentialDriveRobot(settings)
    drive.state.pose = (tuple(start), heading)
    drive.state.velocity = (speed0, omega)
    drive.state.wheel_speeds = drive.movement._resulting_wheel_speeds(drive.current_speed)
    points = []
    for _ in range(max(1, int(np.ceil(seconds / dt)))):
        err = np.arctan2(np.sin(target - drive.pose[1]), np.cos(target - drive.pose[1]))
        command = np.array([speed, np.clip(gain * err, -rate, rate)])
        drive.apply_action(tuple((command - drive.current_speed) / dt), dt)
        points.append(drive.pos)
    return np.asarray(points)


def main():  # noqa: C901, PLR0915
    """Write append-only episode records with candidate forecast and input diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", type=int, default=10, choices=range(1, 11))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "manifest.json").exists() or (args.output / "episodes.jsonl").exists():
        raise FileExistsError("Use a fresh output directory; preserve existing or partial evidence")
    logger.remove()
    logger.add(args.output / "runtime.log", level="WARNING")
    scenarios = {s["name"]: dict(s) for s in load_scenarios(MATRIX)}
    missing = set(SCENARIOS) - scenarios.keys()
    if missing:
        raise ValueError(f"Missing scenarios: {missing}; available: {list(scenarios)}")
    manifest = {
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "dirty": subprocess.check_output(["git", "status", "--porcelain"], text=True),
        "seeds": list(range(1001, 1001 + args.seeds)),
        "scenarios": SCENARIOS,
        "claim_boundary": "development diagnostics; no held-out performance claim",
        "dependency_versions": {
            package: importlib.metadata.version(package)
            for package in ("numpy", "tensorflow", "gymnasium", "scipy", "numba")
        },
        "files_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                MATRIX,
                *CONFIGS.values(),
                Path(__file__),
                Path(sampling.__file__),
                Path("robot_sf/planner/socnav_sacadrl.py"),
                Path("robot_sf/planner/socnav_base.py"),
                Path("robot_sf/robot/differential_drive.py"),
            ]
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    current = {}
    context = {"omega": 0.0, "settings": DifferentialDriveSettings()}
    original_bind = SamplingPlannerAdapter.bind_env
    original_plan = SamplingPlannerAdapter.plan
    original_sac = SACADRLPlannerAdapter.plan
    original_rollout = sampling._rollout
    original_blocked = sampling._first_blocked

    def bind(adapter, env):
        original_bind(adapter, env)
        context["settings"] = env.env_config.robot_config

    def plan(adapter, observation):
        robot, _, _ = adapter._socnav_fields(observation)
        context["omega"] = float(np.asarray(robot.get("angular_velocity", [0.0])).reshape(-1)[0])
        return original_plan(adapter, observation)

    def sac(adapter, observation):
        command = original_sac(adapter, observation)
        vec, pref, _ = adapter._build_network_input(observation)
        _, _, peds = adapter._socnav_fields(observation)
        count = int(np.asarray(peds.get("count", [0])).reshape(-1)[0])
        current["steps"] += 1
        current["max_command_v"] = max(current["max_command_v"], command[0])
        current["pref_speed"] = pref
        current["max_observed"] = max(current["max_observed"], int(vec[0, 0]))
        current["max_pedestrians"] = max(current["max_pedestrians"], count)
        current["omitted_agent_steps"] += count > int(vec[0, 0])
        current["commands"].add(tuple(np.round(command, 8)))
        # Same checkpoint and host state; isolate three vs nineteen agents.
        saved = adapter.config.sacadrl_max_other_agents
        adapter.config.sacadrl_max_other_agents = 19 if saved == 3 else 3
        shadow, _, _ = adapter._build_network_input(observation)
        adapter.config.sacadrl_max_other_agents = saved
        model = adapter._ensure_model()
        current["agent_limit_action_changes"] += int(np.argmax(model.predict(vec)[0])) != int(
            np.argmax(model.predict(shadow)[0])
        )
        return command

    def rollout(*values, **kwargs):
        result = original_rollout(*values, **kwargs)
        # Escape samples are explicitly future aligned headings; their initial rate is zero.
        omega = context["omega"] if values[2] != 0 or values[1] != values[3] else 0.0
        reference = native_candidate(*values, context["settings"], omega)
        context["reference"] = reference
        error = float(np.max(np.linalg.norm(result[0] - reference, axis=1)))
        current["candidates"] += 1
        current["error_sum"] += error
        current["max_forecast_error_m"] = max(current["max_forecast_error_m"], error)
        return result

    def blocked(start, points, clearance, peds, radius, threshold, velocity, dt):
        actual = original_blocked(start, points, clearance, peds, radius, threshold, velocity, dt)
        native = original_blocked(
            start, context["reference"], clearance, peds, radius, threshold, velocity, dt
        )
        current["blocked_index_changes"] += actual != native
        current["false_clear_candidates"] += actual == len(points) and native < len(points)
        return actual

    SamplingPlannerAdapter.bind_env = bind
    SamplingPlannerAdapter.plan = plan
    SACADRLPlannerAdapter.plan = sac
    sampling._rollout = rollout
    sampling._first_blocked = blocked
    policies = {}

    def builder(algo, config, **kwargs):
        if algo not in policies:
            policies[algo] = _build_policy(algo, config, **kwargs)
        return policies[algo]

    for algo, config_path in CONFIGS.items():
        config = yaml.safe_load(config_path.read_text())
        config["allow_fallback"] = False
        for name in SCENARIOS:
            for seed in manifest["seeds"]:
                current = {
                    "steps": 0,
                    "max_command_v": 0.0,
                    "pref_speed": None,
                    "max_observed": 0,
                    "max_pedestrians": 0,
                    "omitted_agent_steps": 0,
                    "agent_limit_action_changes": 0,
                    "commands": set(),
                    "candidates": 0,
                    "error_sum": 0.0,
                    "max_forecast_error_m": 0.0,
                    "blocked_index_changes": 0,
                    "false_clear_candidates": 0,
                }
                row = _run_map_episode(
                    scenarios[name],
                    seed,
                    horizon=None,
                    dt=0.1,
                    record_forces=False,
                    snqi_weights=None,
                    snqi_baseline=None,
                    algo=algo,
                    scenario_path=MATRIX,
                    algo_config=config,
                    algo_config_path=str(config_path),
                    policy_builder=builder,
                    close_policy=False,
                )
                current["commands"] = sorted(current["commands"])
                result = {
                    "algo": algo,
                    "scenario": name,
                    "seed": seed,
                    "diagnostics": current,
                    "episode": row,
                }
                with (args.output / "episodes.jsonl").open("a") as stream:
                    stream.write(
                        json.dumps(
                            result,
                            default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x),
                        )
                        + "\n"
                    )
                print(
                    json.dumps(
                        {
                            "algo": algo,
                            "scenario": name,
                            "seed": seed,
                            "outcome": row.get("outcome"),
                            "steps": row.get("steps"),
                            "diagnostics": current,
                        }
                    ),
                    flush=True,
                )
    actions = _sacadrl_actions()
    discrete = np.column_stack((actions[:, 0], np.clip(actions[:, 1] / 0.1, -1, 1)))
    (args.output / "actions.json").write_text(
        json.dumps(
            {
                "raw": actions.tolist(),
                "effective": discrete.tolist(),
                "distinct": len(np.unique(np.round(discrete, 8), axis=0)),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
