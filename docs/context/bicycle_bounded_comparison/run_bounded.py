"""Bounded bicycle comparison harness: release planners, matched DD/bicycle plants.

Diagnostic-only, development seeds only. Production code is not modified; this
script observes the real map-runner path (same pattern as
scripts/validation/run_bicycle_probe.py) and writes one result JSON, one
compressed step trace and one compressed runner record per episode.

Usage (from the repository root):
    BOUNDED_OUTPUT=<dir> uv run --frozen python \
        docs/context/bicycle_bounded_comparison/run_bounded.py \
        --contract docs/context/bicycle_bounded_comparison/contract.yaml \
        --stage {smoke,pilot,main} [--worker I --workers N]
"""

# ruff: noqa: C901, D103, RUF007

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
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml
from loguru import logger

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/validation"))
if TYPE_CHECKING:
    from scripts.validation.bicycle_probe_metrics import displacement_stuck_steps
else:
    from bicycle_probe_metrics import displacement_stuck_steps

import robot_sf  # noqa: E402
from robot_sf.benchmark.map_runner import map_runner_episode as _episode  # noqa: E402
from robot_sf.benchmark.map_runner.map_runner import _build_policy  # noqa: E402
from robot_sf.planner.kinematics_model import BicycleDriveKinematicsModel  # noqa: E402
from robot_sf.robot.action_adapters import holonomic_to_diff_drive_action  # noqa: E402
from robot_sf.sim.spawn_validation import reset_spawn_clearance  # noqa: E402
from robot_sf.training.scenario_loader import load_scenarios  # noqa: E402

episode: Any = _episode

logger.remove()
OUT = Path(os.environ["BOUNDED_OUTPUT"]).resolve()
OUT.mkdir(parents=True, exist_ok=True)
logger.add(OUT / f"runtime-{os.getpid()}.log", level="WARNING")

FORBIDDEN = set(range(111, 141))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_contract(path):
    c = yaml.safe_load(Path(path).read_text())
    template = yaml.safe_load((ROOT / c["planner_source"]).read_text())["planners"]
    roster = []
    for key in c["planners"]:
        entry = next(p for p in template if p["key"] == key)
        roster.append(dict(entry))
    cells = {}
    for cell in c["cells"]:
        scens = {s["name"]: s for s in load_scenarios(ROOT / cell["manifest"])}
        cells[cell["id"]] = (dict(scens[cell["scenario"]]), ROOT / cell["manifest"])
    return c, roster, cells


def seeds_for(c, stage, probe):
    block = c["seeds"][stage]["empty" if probe == 1 else "scenario"]
    seeds = list(range(block[0], block[1] + 1))
    allowed = set()
    for lo, hi in c["seeds"]["allowlist"]:
        allowed |= set(range(lo, hi + 1))
    assert set(seeds) <= allowed and not set(seeds) & FORBIDDEN, "STOP: seed outside allowlist"
    return seeds


ACTIVE = {}
ORIG_MAKE = episode.make_robot_env
ORIG_ACTION = episode._policy_command_to_env_action
ORIG_PROJECT_WITH_INFO = BicycleDriveKinematicsModel.project_with_creep_info
ORIG_SAFETY = episode._step_safety_filters
ORIG_CONVERT = episode._step_convert_and_execute


def conversion_capture(state, slc, *, policy_command, step_is_native, env):
    ACTIVE["native_action"] = bool(step_is_native)
    if step_is_native:
        ACTIVE["pending"] = list(map(float, policy_command[:2]))
    return ORIG_CONVERT(
        state, slc, policy_command=policy_command, step_is_native=step_is_native, env=env
    )


def safety_capture(state, slc, **kwargs):
    result = ORIG_SAFETY(state, slc, **kwargs)
    labels = []
    if state.safety_wrapper_trace and state.safety_wrapper_trace[-1].get("intervened"):
        labels.append(state.safety_wrapper_trace[-1]["intervention"])
    if state.cbf_filter_trace and state.cbf_filter_trace[-1].get("intervened"):
        labels.append("cbf")
    ACTIVE["safety_interventions"] = labels
    return result


def project_capture(self, command):
    result, creep_applied = ORIG_PROJECT_WITH_INFO(self, command)
    ACTIVE.setdefault("projections", []).append(
        [
            list(map(float, command)),
            list(map(float, result)),
            bool(self.is_feasible(command)),
            creep_applied,
        ]
    )
    return result, creep_applied


def action_capture(*, env, config, command, **kwargs):
    robot = env.simulator.robots[0]
    if isinstance(command, dict):
        ACTIVE["world"] = {
            k: (float(v) if isinstance(v, (int, float)) else v) for k, v in command.items()
        }
        ACTIVE["pending"] = None
    else:
        ACTIVE["pending"] = list(map(float, command))
        ACTIVE["world"] = None
    del robot
    return ORIG_ACTION(env=env, config=config, command=command, **kwargs)


def make_capture(*args, **kwargs):
    assert int(kwargs["seed"]) in ACTIVE["allowed_seeds"], "STOP: seed outside allowlist"
    env: Any = ORIG_MAKE(*args, **kwargs)
    cfg = env.simulator.robots[0].config
    for k, v in ACTIVE["expected_config"].items():
        if k in {"type", "creep_speed", "limited_reverse"}:
            continue
        assert getattr(cfg, k) == v, (k, getattr(cfg, k), v)
    want = ACTIVE["expected_config"]["type"]
    got = type(cfg).__name__.lower()
    assert want.split("_")[0] in got, ("plant type", got, want)
    ACTIVE["robot_config"] = {k: v for k, v in vars(cfg).items() if not k.startswith("_")}
    orig_step = env.step

    def step(action):
        robot = env.simulator.robots[0]
        before = float(robot.pose[1])
        pre_pose = list(map(float, robot.pos))
        projections = ACTIVE.pop("projections", [])
        final_command = ACTIVE.pop("pending", None)
        command = projections[0][0] if projections else final_command
        result = orig_step(action)
        theta = float(robot.pose[1])
        yaw = math.atan2(math.sin(theta - before), math.cos(theta - before)) / 0.1
        x, y = map(float, robot.pos)
        cl = reset_spawn_clearance(env.simulator)
        finite = [
            v
            for k, v in cl.items()
            if k.endswith("_surface_clearance_m") and v is not None and math.isfinite(v)
        ]
        ACTIVE["trace"].append(
            {
                "step": len(ACTIVE["trace"]),
                "cmd": command if command is not None else [0.0, 0.0],
                "final_command": final_command,
                "native_action": ACTIVE.pop("native_action", False),
                "pre_pose": pre_pose,
                "yaw": yaw,
                "v": float(robot.current_speed[0]),
                "pose": [x, y, theta],
                "action": list(map(float, action)),
                "proj": projections,
                "world": ACTIVE.pop("world", None),
                "creeping": bool(any(p[3] for p in projections) and projections[-1][1][0] > 0.0),
                "safety_interventions": ACTIVE.pop("safety_interventions", []),
                "min_clearance_m": min(finite) if finite else None,
                "ped_clearance_m": cl["robot_pedestrian_min_surface_clearance_m"],
                "obstacle_clearance_m": cl["robot_obstacle_min_surface_clearance_m"],
            }
        )
        return result

    env.step = step
    return env


episode._step_convert_and_execute = conversion_capture
episode._step_safety_filters = safety_capture
BicycleDriveKinematicsModel.project_with_creep_info = project_capture
episode._policy_command_to_env_action = action_capture
episode.make_robot_env = make_capture
assert holonomic_to_diff_drive_action  # production conversion stays the executed one


def reset_hook(bearing, radius):
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
            obs.update(env.state.sensors.next_obs())
        nav = env.simulator.robot_navs[0]
        payload = {
            "pose": robot.pose,
            "peds": np.asarray(env.simulator.ped_pos).tolist(),
            "route": nav.waypoints,
            "goal": nav.current_waypoint,
            "heading": float(robot.pose[1]),
        }
        ACTIVE["reset"] = payload
        ACTIVE["projections"] = []
        return {
            "diagnostic_reset_sha256": hashlib.sha256(
                json.dumps(payload, sort_keys=True, default=float).encode()
            ).hexdigest()
        }

    del radius
    return hook


def prepare_maps():
    maps = OUT / "maps"
    maps.mkdir(parents=True, exist_ok=True)
    for bearing in [0, 45, 90, 135, 180]:
        x = 200 + 6 * math.cos(math.radians(bearing))
        y = 200 - 6 * math.sin(math.radians(bearing))
        svg = (
            '<svg width="400" height="400" viewBox="0 0 400 400" '
            'xmlns="http://www.w3.org/2000/svg" '
            'xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"><g inkscape:label="robot">'
            '<rect x="199.99" y="199.99" width=".02" height=".02" inkscape:label="robot_spawn_zone"/>'
            f'<rect x="{x - 0.01}" y="{y - 0.01}" width=".02" height=".02" '
            'inkscape:label="robot_goal_zone"/>'
            f'<path d="M 200 200 L {x} {y}" inkscape:label="robot_route_0_0" fill="none" '
            'stroke="blue"/></g></svg>'
        )
        (maps / f"empty-{bearing}.svg").write_text(svg)


def find_fallbacks(obj, prefix=""):
    """Return every truthy metadata key that names a fallback or degraded path."""
    hits = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            name = f"{prefix}.{k}" if prefix else str(k)
            lk = str(k).lower()
            if ("fallback" in lk or "degraded" in lk) and not isinstance(v, (dict, list)):
                if v not in (None, False, 0, "", "none", "not_applicable", "disabled", []):
                    hits[name] = v
            hits.update(find_fallbacks(v, name))
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:5]):
            hits.update(find_fallbacks(v, f"{prefix}[{i}]"))
    return hits


def run_cell(c, p, plant_key, probe, cell_id, seed, cells, stage):
    plant = dict(c["plants"][plant_key])
    bearing = int(cell_id) if probe == 1 else None
    if probe == 1:
        scenario = {
            "name": f"bounded_empty_{bearing}",
            "map_file": str(OUT / "maps" / f"empty-{bearing}.svg"),
            "seeds": [seed],
            "simulation_config": {
                "max_episode_steps": c["empty_probe"]["horizon_steps"],
                "time_per_step_in_secs": c["dt"],
                "difficulty": 0,
                "social_force_kernel_version": "wrapped_v2",
            },
            "_diagnostic_remove_pedestrian_actors": True,
        }
        scenario_path = OUT / "empty.yaml"
    else:
        base, scenario_path = cells[cell_id]
        scenario = copy.deepcopy(base)
        scenario["seeds"] = [seed]
        scenario.pop("seed_set", None)
    scenario["robot_config"] = plant
    ACTIVE.clear()
    ACTIVE.update(
        trace=[],
        projections=[],
        expected_config=plant,
        allowed_seeds=set(seeds_for(c, stage, probe)),
    )
    key = f"{stage}-p{probe}-{p['key']}-{plant_key}-{cell_id}-{seed}"
    start = time.monotonic()
    result = {
        "cell": key,
        "stage": stage,
        "planner": p["key"],
        "plant": plant_key,
        "probe": probe,
        "scenario": str(cell_id),
        "scenario_name": scenario.get("name"),
        "seed": seed,
        "source_sha": os.environ.get("BOUNDED_SOURCE_SHA")
        or subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
    }
    try:
        r = episode.run_map_episode(
            scenario,
            seed,
            horizon=c["horizon"],
            dt=c["dt"],
            record_forces=False,
            snqi_weights=None,
            snqi_baseline=None,
            algo=p["algo"],
            scenario_path=scenario_path,
            algo_config_path=p.get("algo_config"),
            adapter_impact_eval=True,
            record_simulation_step_trace=False,
            safety_wrapper=None,
            cbf_safety_filter=None,
            policy_builder=_build_policy,
            pair_reset_hook=reset_hook(bearing, plant["radius"]),
            runtime_input_records=[],
        )
        meta = r.get("algorithm_metadata") or {}
        fb = find_fallbacks(meta)
        status = str(r.get("status"))
        metrics = r.get("metrics") or {}
        view = (r.get("integrity") or {}).get("effective_view") or {}
        spawn = r.get("spawn_validity") or {}
        if status == "error" or meta.get("status") not in (None, "ok"):
            mode = "error"
        elif fb:
            mode = "fallback"
        elif view.get("degraded") or spawn.get("invalid_run"):
            mode = "degraded"
        else:
            mode = "native"  # requested planner/plant path executed; no fallback or degradation
        outcome = r.get("outcome") or {}
        result.update(
            status=status,
            success=bool(metrics.get("success", False)),
            collision=bool(outcome.get("collision_event")),
            timeout=bool(outcome.get("timeout_event")),
            ped_collisions=metrics.get("ped_collision_count"),
            obstacle_collisions=metrics.get("obstacle_collision_count"),
            wall_collisions=metrics.get("wall_collisions"),
            termination=r.get("termination_reason"),
            outcome=outcome,
            metrics=metrics,
            fallback_hits=fb,
            meta_status=meta.get("status"),
            command_path=(meta.get("planner_kinematics") or {}).get("execution_mode"),
            degraded_reason=view.get("degraded_reason"),
            invalid_run=spawn.get("invalid_run"),
            mode=mode,
        )
        with gzip.open(OUT / "records" / f"{key}.json.gz", "wt") as f:
            json.dump(r, f, default=str)
    except (ValueError, RuntimeError, OSError, ImportError, KeyError, AssertionError) as exc:
        result.update(
            status="error",
            mode="error",
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
    trace = ACTIVE["trace"]
    reset_pose = (ACTIVE.get("reset") or {}).get("pose")
    result.update(
        steps=len(trace),
        time_s=len(trace) * c["dt"],
        wall_s=time.monotonic() - start,
        reset=ACTIVE.get("reset"),
        robot_config=ACTIVE.get("robot_config"),
        stuck=displacement_stuck_steps(trace, reset_pose) if reset_pose else None,
        creep_steps=sum(t["creeping"] for t in trace),
        path_length_m=float(
            sum(
                math.dist(a["pose"][:2], b["pose"][:2])
                for a, b in zip(trace, trace[1:], strict=False)
            )
        ),
        max_abs_yaw_rate=max((abs(t["yaw"]) for t in trace), default=0.0),
        min_clearance_m=min(
            (t["min_clearance_m"] for t in trace if t["min_clearance_m"] is not None), default=None
        ),
        final_speed=trace[-1]["v"] if trace else None,
    )
    with gzip.open(OUT / "traces" / f"{key}.json.gz", "wt") as f:
        json.dump(trace, f, default=float)
    with (OUT / "results" / f"{key}.json").open("w") as f:
        json.dump(result, f, default=str)
    print(
        json.dumps(
            {
                k: result.get(k)
                for k in [
                    "cell",
                    "status",
                    "mode",
                    "command_path",
                    "success",
                    "collision",
                    "timeout",
                    "time_s",
                    "wall_s",
                    "error",
                ]
            }
        ),
        flush=True,
    )
    return result


def write_manifest(c, contract_path, roster):
    files = [
        contract_path,
        Path(__file__).resolve(),
        ROOT / "uv.lock",
        ROOT / "scripts/validation/bicycle_probe_metrics.py",
        ROOT / c["planner_source"],
    ]
    files += [ROOT / p["algo_config"] for p in roster if p.get("algo_config")]
    files += [ROOT / cell["manifest"] for cell in c["cells"]]
    files += [ROOT / f for f in c["hashed_source_files"]]
    manifest = {
        "source_sha": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "git_dirty": subprocess.check_output(
            ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"], text=True
        ).strip(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "robot_sf_file_under_root": str(ROOT) in robot_sf.__file__,
        "platform": platform.platform(),
        "files_sha256": {
            str(Path(f).resolve().relative_to(ROOT))
            if str(Path(f).resolve()).startswith(str(ROOT))
            else str(f): sha(f)
            for f in files
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--stage", choices=["smoke", "pilot", "main"], required=True)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    c, roster, cells = load_contract(a.contract)
    for d in ["results", "traces", "records"]:
        (OUT / d).mkdir(parents=True, exist_ok=True)
    prepare_maps()
    if a.worker == 0:
        write_manifest(c, a.contract.resolve(), roster)
    jobs = []
    for p in roster:
        for plant_key in c["plants"]:
            for probe, names in [(1, c["empty_probe"]["bearings"]), (2, list(cells))]:
                if a.stage == "smoke" and probe == 1:
                    continue
                stage_names = names[:1] if a.stage == "smoke" else names
                for name in stage_names:
                    for seed in seeds_for(c, a.stage, probe):
                        jobs.append((p, plant_key, probe, name, seed))
    if a.list:
        print(len(jobs))
        return
    for i, job in enumerate(jobs):
        if i % a.workers != a.worker:
            continue
        p, plant_key, probe, name, seed = job
        key = f"{a.stage}-p{probe}-{p['key']}-{plant_key}-{name}-{seed}"
        if (OUT / "results" / f"{key}.json").exists():
            continue
        run_cell(c, p, plant_key, probe, name, seed, cells, a.stage)


if __name__ == "__main__":
    main()
