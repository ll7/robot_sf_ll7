"""KINPROBE diagnostic harness: real map runner, explicit dev seeds, serial worker."""

import os

os.environ.update(
    OMP_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    TF_NUM_INTRAOP_THREADS="1",
    TF_NUM_INTEROP_THREADS="1",
    CUDA_VISIBLE_DEVICES="",
    SDL_VIDEODRIVER="dummy",
)
import argparse
import copy
import gzip
import hashlib
import json
import math
import socket
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
import pysocialforce
import yaml
from loguru import logger

import robot_sf
from robot_sf.benchmark.algorithm_metadata import planner_contract_for_algorithm
from robot_sf.benchmark.map_runner import map_runner_episode as episode
from robot_sf.benchmark.map_runner.map_runner import _build_policy
from robot_sf.benchmark.planner_command_contract import planner_kinematics_compatibility
from robot_sf.planner.kinematics_model import BicycleDriveKinematicsModel
from robot_sf.robot.action_adapters import holonomic_to_diff_drive_action
from robot_sf.sim.spawn_validation import reset_spawn_clearance
from robot_sf.training.scenario_loader import load_scenarios

logger.remove()
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ["BIKEFIX_OUTPUT"]).resolve()
OUT.mkdir(parents=True, exist_ok=True)
logger.add(OUT / f"runtime-{os.getpid()}.log", level="WARNING")
SCENPATH = ROOT / "configs/scenarios/classic_interactions_francis2023_release_0_0_8_v1.yaml"
TEMPLATE = (
    ROOT / "configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_template.yaml"
)
NAMES = [
    "classic_cross_trap_high",
    "classic_station_platform_medium",
    "classic_t_intersection_medium",
    "francis2023_frontal_approach",
    "francis2023_narrow_hallway",
    "francis2023_robot_crowding",
]
ROSTER = yaml.safe_load(TEMPLATE.read_text())["planners"]
SCENS = {s["name"]: dict(s) for s in load_scenarios(SCENPATH) if s["name"] in NAMES}
assert len(ROSTER) == 14 and set(SCENS) == set(NAMES)
assert str(ROOT) in robot_sf.__file__ and str(ROOT) in pysocialforce.__file__


def prepare():
    """Write explicit dev axes and exact input identities before any episode."""
    maps = OUT / "maps"
    maps.mkdir(parents=True, exist_ok=True)
    for bearing in [0, 45, 90, 135, 180]:
        x, y = 200 + 6 * math.cos(math.radians(bearing)), 200 - 6 * math.sin(math.radians(bearing))
        svg = f'<svg width="400" height="400" viewBox="0 0 400 400" xmlns="http://www.w3.org/2000/svg" xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"><g inkscape:label="robot"><rect x="199.99" y="199.99" width=".02" height=".02" inkscape:label="robot_spawn_zone"/><rect x="{x - 0.01}" y="{y - 0.01}" width=".02" height=".02" inkscape:label="robot_goal_zone"/><path d="M 200 200 L {x} {y}" inkscape:label="robot_route_0_0" fill="none" stroke="blue"/></g></svg>'
        (maps / f"empty-{bearing}.svg").write_text(svg)
    (OUT / "resolved_scenarios.json").write_text(json.dumps(SCENS, indent=2, default=str))
    compat = {}
    for p in ROSTER:
        cfg = yaml.safe_load((ROOT / p["algo_config"]).read_text()) if p.get("algo_config") else {}
        compat[p["key"]] = {
            "helper": planner_kinematics_compatibility(
                algo=p["algo"], robot_kinematics="bicycle_drive", algo_config=cfg
            ),
            "registry_action": planner_contract_for_algorithm(
                p["algo"], robot_kinematics="bicycle_drive"
            ).to_metadata()["action_contract"],
        }
    (OUT / "compatibility.json").write_text(json.dumps(compat, indent=2))
    manifest = {
        "source_sha": os.environ.get("BIKEFIX_SOURCE_SHA")
        or subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "kind": "diagnostic-only",
        "roster": ROSTER,
        "probe1_seeds": list(range(1001, 1006)),
        "probe2_seeds": list(range(1001, 1011)),
        "bearings": [0, 45, 90, 135, 180],
        "horizon": 600,
        "dt": 0.1,
        "python": os.sys.executable,
        "source_files_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                ROOT / "robot_sf/planner/kinematics_model.py",
                ROOT / "robot_sf/planner/classic_planner_adapter.py",
                ROOT / "robot_sf/benchmark/map_runner_policies/map_runner_actions.py",
                ROOT / "robot_sf/benchmark/map_runner/map_runner_episode.py",
            ]
        },
        "robot_sf": robot_sf.__file__,
        "pysocialforce": pysocialforce.__file__,
        "template_sha256": hashlib.sha256(TEMPLATE.read_bytes()).hexdigest(),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        json.dumps(
            {
                "resolved_seeds": manifest["probe2_seeds"],
                "planners": list(compat),
                "scenarios": list(SCENS),
            }
        ),
        flush=True,
    )


ACTIVE = {}
ORIG_MAKE = episode.make_robot_env
ORIG_ACTION = episode._policy_command_to_env_action
ORIG_PROJECT = BicycleDriveKinematicsModel.project
ORIG_SAFETY = episode._step_safety_filters


def safety_capture(state, slc, **kwargs):
    """Retain actual interventions without changing the safety transform."""
    result = ORIG_SAFETY(state, slc, **kwargs)
    labels = []
    if state.safety_wrapper_trace:
        record = state.safety_wrapper_trace[-1]
        if record.get("intervened"):
            labels.append(record["intervention"])
        if record.get("deadlock_recovery", {}).get("recovery_active"):
            labels.append("deadlock_recovery")
    if state.cbf_filter_trace and state.cbf_filter_trace[-1].get("intervened"):
        labels.append("cbf")
    ACTIVE["safety_interventions"] = labels
    return result


episode._step_safety_filters = safety_capture


def project_capture(self, command):
    """Capture requested commands before the real projection."""
    result = ORIG_PROJECT(self, command)
    ACTIVE.setdefault("projections", []).append(
        [
            list(map(float, command)),
            list(map(float, result)),
            bool(self.is_feasible(command)),
            bool(
                abs(command[0]) < 1e-3
                and result[0] > max(command[0], 0.0)
                and abs(command[1]) > 1e-6
            ),
        ]
    )
    return result


BicycleDriveKinematicsModel.project = project_capture


def action_capture(*, env, config, command, **kwargs):
    """Observe the production conversion without replacing its behavior."""
    robot = env.simulator.robots[0]
    if isinstance(command, dict):
        expected = holonomic_to_diff_drive_action(
            np.array([command.get("vx", 0), command.get("vy", 0)]),
            robot.pose,
            max_linear_speed=2.0,
            max_angular_speed=1.0,
        )
        ACTIVE["pending"] = list(map(float, expected))
        ACTIVE["world"] = dict(command)
    else:
        ACTIVE["pending"] = list(map(float, command))
        ACTIVE["world"] = None
    return ORIG_ACTION(env=env, config=config, command=command, **kwargs)


episode._policy_command_to_env_action = action_capture


def make_capture(*args, **kwargs):
    """Assert physical settings and record commands and achieved motion."""
    assert 1001 <= int(kwargs["seed"]) <= 1030, "STOP: non-dev seed"
    env = ORIG_MAKE(*args, **kwargs)
    cfg = env.simulator.robots[0].config
    expected = ACTIVE["expected_config"]
    for k, v in expected.items():
        if k != "type":
            assert getattr(cfg, k) == v, (k, getattr(cfg, k), v)
    ACTIVE["robot_config"] = vars(cfg).copy()
    orig_step = env.step

    def step(action):
        robot = env.simulator.robots[0]
        before = float(robot.pose[1])
        pre_pose = list(map(float, robot.pos))
        pre_speed = float(robot.current_speed[0])
        projections = ACTIVE.pop("projections", [])
        final_command = ACTIVE.pop("pending", [0.0, 0.0])
        command = final_command
        if projections:
            command = projections[0][0]
        result = orig_step(action)
        theta = float(robot.pose[1])
        yaw = math.atan2(math.sin(theta - before), math.cos(theta - before)) / 0.1
        v = float(robot.current_speed[0])
        x, y = map(float, robot.pos)
        clearances = reset_spawn_clearance(env.simulator)
        finite_clearance = [
            value
            for key, value in clearances.items()
            if key.endswith("_surface_clearance_m") and value is not None and math.isfinite(value)
        ]
        row = {
            "step": len(ACTIVE["trace"]),
            "cmd": command,
            "final_command": final_command,
            "pre_pose": pre_pose,
            "yaw": yaw,
            "v": v,
            "pre_v": pre_speed,
            "pose": [x, y, theta],
            "action": list(map(float, action)),
            "proj": projections,
            "world": ACTIVE.pop("world", None),
            "creeping": bool(any(p[3] for p in projections) and projections[-1][1][0] > 0.0),
            "safety_interventions": ACTIVE.pop("safety_interventions", []),
            "min_clearance_m": min(finite_clearance) if finite_clearance else None,
            "ped_clearance_m": clearances["robot_pedestrian_min_surface_clearance_m"],
            "obstacle_clearance_m": clearances["robot_obstacle_min_surface_clearance_m"],
        }
        ACTIVE["trace"].append(row)
        return result

    env.step = step
    return env


episode.make_robot_env = make_capture


def reset_hook(bearing):
    """Make paired empty-world start states and goal bearings exact."""

    def hook(env, obs):
        robot = env.simulator.robots[0]
        if bearing is not None:
            goal = (
                200 + 6 * math.cos(math.radians(bearing)),
                200 + 6 * math.sin(math.radians(bearing)),
            )
            robot.reset_state(((200.0, 200.0), 0.0))
            nav = env.simulator.robot_navs[0]
            nav.waypoints = [goal]
            nav.waypoint_id = 0
            nav.pos = (200.0, 200.0)
            nav.reached_waypoint = False
            env.state.distance_to_goal = env.state.prev_distance_to_goal = 6.0
            env.state.sensors.reset_cache()
            refreshed = env.state.sensors.next_obs()
            obs.update(refreshed)
        ped = np.asarray(env.simulator.ped_pos).tolist()
        nav = env.simulator.robot_navs[0]
        payload = {
            "pose": robot.pose,
            "peds": ped,
            "route": nav.waypoints,
            "goal": nav.current_waypoint,
            "heading": float(robot.pose[1]),
        }
        ACTIVE["reset"] = payload
        ACTIVE["projections"] = []
        return {
            "diagnostic_reset_sha256": hashlib.sha256(
                json.dumps(payload, sort_keys=True).encode()
            ).hexdigest()
        }

    return hook


def t60_plant(arm):
    """Load the explicit diagnostic plant and optional creep arm."""
    variant = "30deg" if arm.startswith("T60-30") else "45deg"
    plant = yaml.safe_load((ROOT / f"configs/robots/t60_bicycle_{variant}_v1.yaml").read_text())[
        "robot_config"
    ]
    plant["creep_speed"] = 0.1 if arm.endswith("-on") else 0.0
    if os.environ.get("BIKEFIX_REVIEW_BASE") == "1":
        plant.pop("creep_speed")  # At44c386d6 creep was implicit, not a setting.
    return plant


def plant_and_policy(p, arm):
    """Resolve explicit diagnostic plant and policy copies without release edits."""
    if arm in {"BI-base", "BI-fixed", "BI-off", "BI-on", "DD-legacy"}:
        plant = {
            "type": "bicycle_drive",
            "radius": 1.0,
            "wheelbase": 0.85,
            "max_steer": 0.6,
            "max_velocity": 2.0,
            "max_accel": 1.0,
            "max_decel": 1.0,
            "allow_backwards": False,
        }
        if arm == "DD-legacy":
            plant = {
                "type": "differential_drive",
                "radius": 1.0,
                "max_linear_speed": 2.0,
                "allow_backwards": False,
            }
        elif arm in {"BI-off", "BI-on"}:
            plant["creep_speed"] = 0.1 if arm == "BI-on" else 0.0
        policy_config = None
    else:
        if arm == "DD":
            plant = {
                "type": "differential_drive",
                "radius": 0.64,
                "max_linear_speed": 1.34,
                "max_linear_accel": 1.0,
                "max_linear_decel": 1.0,
                "allow_backwards": False,
            }
        else:
            plant = t60_plant(arm)
        # Diagnostic copies: never rewrite frozen release policy files.
        policy_config = (
            yaml.safe_load((ROOT / p["algo_config"]).read_text()) if p.get("algo_config") else {}
        )

        def bind(mapping):
            for k, v in list(mapping.items()):
                if isinstance(v, dict):
                    bind(v)
                elif k in {
                    "robot_radius",
                    "robot_radius_m",
                    "robot_radius_default",
                    "predictive_robot_radius",
                    "guard_robot_radius_m",
                }:
                    mapping[k] = 0.64
                elif k in {"v_max", "max_linear_speed", "max_speed"}:
                    mapping[k] = min(float(v), 1.34)

        bind(policy_config)
    return plant, policy_config


def run_cell(p, arm, probe, name, seed):
    """Run one diagnostic cell with explicit dev seed and preserved motion trace."""
    assert 1001 <= seed <= 1030, "STOP: non-dev seed"
    bearing = int(name) if probe == 1 else None
    if probe == 1:
        scenario = {
            "name": f"kinprobe_empty_{bearing}",
            "map_file": str(OUT / "maps" / f"empty-{bearing}.svg"),
            "seeds": [seed],
            "simulation_config": {
                "max_episode_steps": 600,
                "time_per_step_in_secs": 0.1,
                "difficulty": 0,
                "social_force_kernel_version": "wrapped_v2",
            },
            "_diagnostic_remove_pedestrian_actors": True,
        }
        scenario_path = OUT / "empty.yaml"
    else:
        scenario = copy.deepcopy(SCENS[name])
        scenario["seeds"] = [seed]
        scenario.pop("seed_set", None)
        scenario_path = SCENPATH
    plant, policy_config = plant_and_policy(p, arm)
    scenario["robot_config"] = plant
    ACTIVE.clear()
    ACTIVE.update(trace=[], projections=[], expected_config=plant)
    key = f"p{probe}-{p['key']}-{arm}-{name}-{seed}"
    start = time.monotonic()
    result = {
        "cell": key,
        "planner": p["key"],
        "arm": arm,
        "probe": probe,
        "scenario": str(name),
        "seed": seed,
        "execution_host": socket.gethostname(),
        "python": os.sys.executable,
        "source_sha": os.environ.get("BIKEFIX_SOURCE_SHA")
        or subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "policy_overrides": policy_config,
    }
    try:
        records = []
        r = episode.run_map_episode(
            scenario,
            seed,
            horizon=600,
            dt=0.1,
            record_forces=False,
            snqi_weights=None,
            snqi_baseline=None,
            algo=p["algo"],
            scenario_path=scenario_path,
            algo_config_path=p.get("algo_config"),
            algo_config=policy_config,
            adapter_impact_eval=p.get("adapter_impact_eval", False),
            record_simulation_step_trace=False,
            policy_builder=_build_policy,
            pair_reset_hook=reset_hook(bearing),
            runtime_input_records=records,
        )
        result.update(
            status=r.get("status"),
            success=bool(r["metrics"].get("success", False)),
            steps=r.get("steps"),
            termination=r.get("termination_reason"),
            outcome=r.get("outcome"),
            metadata=r.get("algorithm_metadata"),
            metrics=r.get("metrics"),
            inputs=records,
        )
        with gzip.open(OUT / "records" / f"{key}.json.gz", "wt") as f:
            json.dump(r, f, default=str)
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        result.update(
            status="error", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()
        )
    trace = ACTIVE["trace"]
    result.update(
        steps=len(trace),
        time_s=len(trace) * 0.1,
        wall_s=time.monotonic() - start,
        reset=ACTIVE.get("reset"),
        robot_config=ACTIVE.get("robot_config"),
        stuck=displacement_stuck_steps(trace, ACTIVE.get("reset", {}).get("pose")),
        zero_turn=sum(t["creeping"] for t in trace),
        was_creeping_at_termination=bool(trace and trace[-1]["creeping"]),
        min_clearance_last_2s_m=min(
            (t["min_clearance_m"] for t in trace[-20:] if t["min_clearance_m"] is not None),
            default=None,
        ),
        true_infeasible=sum(
            abs(t["cmd"][1])
            > abs(t["cmd"][0])
            * math.tan(plant.get("max_steer", 0.6))
            / plant.get("wheelbase", 0.85)
            + 1e-6
            for t in trace
        ),
        underreported=sum(
            bool(t["proj"])
            and t["proj"][0][2]
            and abs(t["cmd"][1])
            > abs(t["cmd"][0])
            * math.tan(plant.get("max_steer", 0.6))
            / plant.get("wheelbase", 0.85)
            + 1e-6
            for t in trace
        ),
        world_steps=sum(t["world"] is not None for t in trace),
    )
    with gzip.open(OUT / "traces" / f"{key}.json.gz", "wt") as f:
        json.dump(trace, f)
    with (OUT / "results" / f"{key}.json").open("w") as f:
        json.dump(result, f, default=str)
    print(
        json.dumps(
            {
                k: result.get(k)
                for k in ["cell", "status", "success", "time_s", "stuck", "wall_s", "error"]
            }
        ),
        flush=True,
    )
    return result


def displacement_stuck_steps(trace, reset_pose):
    """Count full overlapping 2s windows with <.05m net motion and a requested command.

    At least one raw (pre-projection) linear or yaw component must exceed 1e-6
    during the window. Count its ending step; windows overlap at the .1s stride.
    Neither creep nor the commanded/achieved yaw law appears in the motion test.
    """
    count = 0
    for end in range(19, len(trace)):
        start = end - 19
        initial = trace[start - 1]["pose"][:2] if start else reset_pose[0]
        moved = math.dist(initial, trace[end]["pose"][:2])
        issued = any(max(map(abs, t["cmd"])) > 1e-6 for t in trace[start : end + 1])
        count += moved < 0.05 and issued
    return count


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--arms", default="DD,T60-30,T60-45")
    ap.add_argument("--planners", default="")
    ap.add_argument("--probe", type=int, default=0)
    ap.add_argument("--cells-json", type=Path, help="Explicit dev cells for an instrumented replay")
    a = ap.parse_args()
    assert set(a.arms.split(",")) <= {
        "DD",
        "T60-30",
        "T60-45",
        "T60-30-on",
        "T60-45-on",
        "DD-legacy",
        "BI-base",
        "BI-fixed",
        "BI-off",
        "BI-on",
    }
    for directory in ["results", "traces", "records"]:
        (OUT / directory).mkdir(parents=True, exist_ok=True)
    if a.prepare:
        prepare()
    elif a.smoke:
        for p in ROSTER:
            if a.planners and p["key"] not in a.planners.split(","):
                continue
            for arm in a.arms.split(","):
                run_cell(p, arm, 1, 90, 1001)
    else:
        cells = []
        if a.cells_json:
            for cell in json.loads(a.cells_json.read_text()):
                p = next(p for p in ROSTER if p["key"] == cell["planner"])
                assert 1001 <= int(cell["seed"]) <= 1010, "STOP: non-dev replay seed"
                assert cell["scenario"] in NAMES or int(cell["probe"]) == 1
                cells.append(
                    (p, cell["arm"], int(cell["probe"]), cell["scenario"], int(cell["seed"]))
                )
        for p in ROSTER:
            if a.cells_json:
                break
            if a.planners and p["key"] not in a.planners.split(","):
                continue
            for probe, names, seeds in [
                (1, [0, 45, 90, 135, 180], range(1001, 1006)),
                (2, NAMES, range(1001, 1011)),
            ]:
                if a.probe and probe != a.probe:
                    continue
                for arm in a.arms.split(","):
                    for name in names:
                        for seed in seeds:
                            cells.append((p, arm, probe, name, seed))
        for i, cell in enumerate(cells):
            if i % a.workers != a.worker:
                continue
            p, arm, probe, name, seed = cell
            path = OUT / "results" / f"p{probe}-{p['key']}-{arm}-{name}-{seed}.json"
            if path.exists():
                continue
            run_cell(*cell)
