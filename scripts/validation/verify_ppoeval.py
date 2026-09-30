"""Local executable evidence for PPOEVAL; deterministic hand or historical references."""

# ruff: noqa: E402
import sys

sys.modules["triton"] = None
import argparse
import importlib.util
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from robot_sf.baselines.ppo import PPOPlanner
from robot_sf.benchmark.map_runner.map_runner_env import build_env_config
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.robot.differential_drive import DifferentialDriveRobot, DifferentialDriveSettings
from robot_sf.training.scenario_loader import load_scenarios
from scripts.analysis.compare_ppoeval import summarize

parser = argparse.ArgumentParser()
parser.add_argument("--variant", required=True)
parser.add_argument("--populations", action="store_true")
parser.add_argument("--output-dir", type=Path, required=True)
args = parser.parse_args()


def fixture(commands, success, clearance):
    """Build hand-calculated episode traces for independent metric assertions."""
    steps = [
        {
            "planner": {
                "ppoeval_proposal": {"requested_command": c},
                "selected_action": {
                    "linear_velocity": max(0, min(2, c[0])),
                    "angular_velocity": c[1],
                },
            },
            "pedestrians": [] if clearance is None else [{"surface_clearance_m": clearance}],
            "robot": {"position": [(i + 1) * 2, 0]},
            "time_s": i + 1,
        }
        for i, c in enumerate(commands)
    ]
    return {
        "steps": len(steps),
        "outcome": {
            "route_complete": bool(success),
            "collision_event": False,
            "timeout_event": not success,
        },
        "metrics": {"success": success},
        "algorithm_metadata": {
            "status": "ok",
            "simulation_step_trace": {"reset": {"robot": {"position": [0, 0]}}, "steps": steps},
        },
    }


r = summarize([fixture([[-1, 0], [3, 1]], 1, -0.1), fixture([[0, 0], [1, 0]], 0, None)])
assert r["fraction_commands_clipped_release"] == 0.5
assert r["fraction_v_negative"] == 0.25 and r["fraction_v_above_2"] == 0.25
assert r["mean_abs_delta_requested_v_m_s"] == 2.5
assert r["mean_abs_delta_requested_omega_rad_s"] == 0.5
assert r["mean_time_to_goal_s_success_only"] == 2
assert r["minimum_pedestrian_clearance_m"] == -0.1
assert r["success"] == 1 and r["timeout"] == 1
assert summarize([fixture([[0, 0]], 1, None)])["minimum_pedestrian_clearance_m"] is None
print("comparison_hand_fixture PASS", json.dumps(r, allow_nan=False))
settings = DifferentialDriveSettings(
    max_linear_speed=3 if args.variant == "v3" else 2, allow_backwards=args.variant == "v3"
)
if args.variant == "v3":
    settings.diagnostic_training_plant = True
robot = DifferentialDriveRobot(settings)
planner_cfg = {
    "model_id": "ppo_expert_br06_v3_15m_all_maps_randomized_20260304T075200",
    "action_space": "unicycle",
    "obs_mode": "dict",
    "device": "cpu",
    "v_max": 3 if args.variant == "v3" else 2,
    "fallback_to_goal": False,
}
if args.variant == "v3":
    planner_cfg["diagnostic_training_plant"] = True
planner = PPOPlanner(planner_cfg, defer_model_loading=True)
current = np.array([0.6, 0.2])
raw = np.array([-1.2, 0.5])
cmd = (
    planner._action_vec_to_dict_from_array(raw, current)
    if args.variant != "v1"
    else planner._action_vec_to_dict_from_array(raw)
)
expected = {"v1": (0, 0.5), "v2": (0, 0.7), "v3": (-0.6, 0.7)}[args.variant]
np.testing.assert_allclose([cmd["v"], cmd["omega"]], expected, rtol=0, atol=1e-12)
action = (np.array(expected) - current) / 0.1
result = robot.movement._robot_velocity(tuple(current), tuple(action), 0.1)
np.testing.assert_allclose(
    result,
    {"v1": (0.5, 0.3), "v2": (0.5, 0.3), "v3": (-0.6, 0.7)}[args.variant],
    rtol=0,
    atol=1e-12,
)
print("semantic_and_plant_literals PASS", cmd, result)
if args.variant == "v3":
    source = subprocess.check_output(
        [
            "git",
            "show",
            "9fb131b6d7ff062887caab33ca8713ae05167ebe:robot_sf/robot/differential_drive.py",
        ]
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "historical.py"
        path.write_bytes(source)
        spec = importlib.util.spec_from_file_location("ppoeval_historical", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        old = module.DifferentialDriveRobot(
            module.DifferentialDriveSettings(max_linear_speed=3, allow_backwards=True)
        )
        for dt in [0.05, 0.1, 0.2]:
            current = (0.6, 0.2)
            for delta in [(-1.2, 0.5), (5.0, -3.0), (-8.0, 2.0), (0.2, -0.1)]:
                reference = old.movement._robot_velocity(current, delta)
                accel = tuple(float(d) / dt for d in delta)
                got = robot.movement._robot_velocity(current, accel, dt)
                np.testing.assert_allclose(got, reference, rtol=0, atol=1e-12)
                current = got
    print("historical_training_transition_reference PASS 12 transitions")
if args.populations:
    rows = []
    for scenario in load_scenarios(Path("configs/scenarios/ppoeval_subset.yaml")):
        cfg = build_env_config(
            scenario, scenario_path=Path("configs/scenarios/ppoeval_subset.yaml")
        )
        env = make_robot_env(config=cfg, seed=1001, debug=False)
        try:
            obs, _ = env.reset(seed=1001)
            count = int(np.asarray(obs["pedestrians_count"]).reshape(-1)[0])
            empty = scenario["metadata"]["ppoeval_family"] == "empty-world"
            assert (count == 0) == empty, (scenario["name"], count)
            rows.append(
                {
                    "scenario": scenario["name"],
                    "family": scenario["metadata"]["ppoeval_family"],
                    "pedestrians": count,
                }
            )
        finally:
            env.close()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "ppoeval_population_check.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )
    print("population_check PASS", len(rows))
