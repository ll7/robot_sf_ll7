"""Instrumented acquisition harness for the ORCA adapter-attribution diagnostic.

Runs the unchanged map-runner episode loop of the pinned robot_sf commit and adds
one ``xorca`` record to every step of the runner's own ``simulation_step_trace``:

* ``pref_world``: the planner's preferred (goal-seeking) world velocity, ORCA only;
* ``solver_world``: raw ``plan_velocity_world`` output (ORCA solver / social-force
  world velocity) before any differential-drive projection;
* ``safe_heading``: occupancy heading selection and penalty used by the ORCA
  differential-drive projection (absent on the holonomic path);
* ``adapter_cmd``: the adapter's ``plan()`` result ``(v, w)`` (differential drive only);
* ``pre``/``post``: robot pose and speed read from the simulator before policy
  inference and after ``env.step``.  ``post`` minus ``pre`` over dt is measured
  motion, not an implied projection.

The runner already records the feasibility-projected command
(``planner.selected_action``) and the environment action
(``planner.applied_environment_action``) per step.  Instrumentation wraps functions
at runtime only; no file of the robot_sf package is modified.

Usage (one job = one configuration x one cell x a list of seeds)::

    python run_xorca.py --contract contract.yaml --config orca_dd --cell doorway_matched_2p40 \
        --seeds 1161 1162 --out-dir <run_dir>
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml

if TYPE_CHECKING:
    from robot_sf.benchmark.map_runner.map_runner_episode import (
        _StepLoopConfig,
        _StepLoopState,
        _StepSimResult,
    )

_CAP: dict[str, Any] = {"calls": [], "pending": None, "fallback_steps": 0}


def _vec(value: Any) -> list[float] | None:
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.size < 2:
        return None
    return [float(arr[0]), float(arr[1])]


def _robot_snapshot(env: Any) -> dict[str, Any] | None:
    sim = getattr(env, "simulator", None)
    robots = getattr(sim, "robots", None)
    if not robots:
        return None
    robot = robots[0]
    try:
        (x, y), heading = robot.pose
    except (TypeError, ValueError):
        return None
    speed = getattr(robot, "current_speed", None)
    out: dict[str, Any] = {"x": float(x), "y": float(y), "heading": float(heading)}
    if speed is not None:
        out["v"] = float(speed[0])
        out["w"] = float(speed[1])
    return out


# Doorway geometry of the matched-route cells (wall x=15..16 m, opening centred on y=5 m).
DOOR_WALL_X = (15.0, 16.0)
DOOR_CENTER_Y = 5.0


def _doorway_gap(centers: Any, radii: Any) -> dict[str, Any]:
    """Free vertical gap left between the solver's inflated obstacle squares at the wall.

    Squares are ``center +/- radius`` (as passed to RVO2).  Only squares whose x-extent
    overlaps the wall are counted; ``None`` when the wall is not in the solver's view.
    """
    c = np.asarray(centers, dtype=float).reshape(-1, 2)
    r = np.asarray(radii, dtype=float).reshape(-1)
    out: dict[str, Any] = {
        "n_obstacles": int(c.shape[0]),
        "door_gap_m": None,
        "max_radius_m": float(r.max()) if r.size else None,
    }
    if not c.size:
        return out
    at_wall = (c[:, 0] + r >= DOOR_WALL_X[0]) & (c[:, 0] - r <= DOOR_WALL_X[1])
    below = at_wall & (c[:, 1] < DOOR_CENTER_Y)
    above = at_wall & (c[:, 1] > DOOR_CENTER_Y)
    if below.any() and above.any():
        lower = float((c[below, 1] + r[below]).max())
        upper = float((c[above, 1] - r[above]).min())
        out["door_gap_m"] = upper - lower
    return out


def _robot_settings(env: Any) -> dict[str, Any] | None:
    """Realized plant settings (radius, speed, turn and acceleration caps)."""
    import dataclasses

    robots = getattr(getattr(env, "simulator", None), "robots", None)
    if not robots:
        return None
    cfg = getattr(robots[0], "config", None)
    if cfg is None or not dataclasses.is_dataclass(cfg):
        return None
    out = {
        k: v for k, v in dataclasses.asdict(cfg).items() if isinstance(v, (int, float, str, bool))
    }
    out["class"] = type(robots[0]).__name__
    return out


def install_instrumentation() -> None:  # noqa: C901, PLR0915
    """Wrap planner and step-loop functions; idempotent per process."""
    if _CAP.get("installed"):
        return
    from robot_sf.benchmark.map_runner import map_runner_episode as ep
    from robot_sf.planner import socnav_occupancy, socnav_orca, socnav_social_force

    orca_cls = socnav_orca.ORCAPlannerAdapter
    sf_cls = socnav_social_force.SocialForcePlannerAdapter

    def _wrap_pvw(cls: Any, tag: str) -> None:
        original = cls.plan_velocity_world

        def wrapped(self: Any, observation: dict) -> Any:
            value = original(self, observation)
            _CAP["calls"].append({"kind": "solver", "planner": tag, "v": _vec(value)})
            return value

        cls.plan_velocity_world = wrapped

    def _wrap_plan(cls: Any, tag: str) -> None:
        original = cls.plan

        def wrapped(self: Any, observation: dict) -> Any:
            value = original(self, observation)
            _CAP["calls"].append({"kind": "adapter_cmd", "planner": tag, "v": _vec(value)})
            return value

        cls.plan = wrapped

    _wrap_pvw(orca_cls, "orca")
    _wrap_pvw(sf_cls, "social_force")
    _wrap_plan(orca_cls, "orca")
    _wrap_plan(sf_cls, "social_force")

    original_heuristic = orca_cls._heuristic_velocity_world

    def heuristic(self: Any, observation: dict) -> Any:
        _CAP["calls"].append({"kind": "heuristic_fallback"})
        return original_heuristic(self, observation)

    orca_cls._heuristic_velocity_world = heuristic

    # ``_preferred_velocity`` is resolved through the instance so that static,
    # class and instance method definitions are all called with their own binding.
    original_pref = orca_cls._preferred_velocity

    def pref(goal: Any, robot_pos: Any, robot_heading: float, max_speed: float) -> Any:
        value = original_pref(goal, robot_pos, robot_heading, max_speed)
        ego = _vec(value)
        world = None
        if ego is not None:
            c, s = math.cos(float(robot_heading)), math.sin(float(robot_heading))
            world = [c * ego[0] - s * ego[1], s * ego[0] + c * ego[1]]
        _CAP["calls"].append({"kind": "pref", "v": world})
        return value

    # These targets are dynamic because instrumentation replaces their attributes at runtime.
    instrumented_orca: Any = orca_cls
    instrumented_orca._preferred_velocity = staticmethod(pref)

    original_bias = orca_cls._apply_commit_bias

    def bias(self: Any, **kwargs: Any) -> Any:
        value = original_bias(self, **kwargs)
        _CAP["calls"].append({"kind": "pref_final", "v": _vec(value)})
        return value

    orca_cls._apply_commit_bias = bias

    original_obst = orca_cls._extract_obstacles_from_grid

    def obstacles(self: Any, observation: dict, robot_pos: Any, robot_heading: float) -> Any:
        centers, radii = original_obst(self, observation, robot_pos, robot_heading)
        _CAP["calls"].append({"kind": "obstacles", **_doorway_gap(centers, radii)})
        return centers, radii

    orca_cls._extract_obstacles_from_grid = obstacles

    mixin = socnav_occupancy.OccupancyAwarePlannerMixin
    original_safe = mixin._get_safe_heading

    def safe(self: Any, robot_pos: Any, base_direction: Any, observation: dict) -> Any:
        direction, penalty = original_safe(self, robot_pos, base_direction, observation)
        _CAP["calls"].append(
            {
                "kind": "safe_heading",
                "base": _vec(base_direction),
                "dir": _vec(direction),
                "penalty": float(penalty),
            }
        )
        return direction, penalty

    mixin._get_safe_heading = safe

    original_spi = ep._step_policy_inference

    def spi(
        state: _StepLoopState, slc: _StepLoopConfig, *, env: Any
    ) -> tuple[Any, np.ndarray | None]:
        _CAP["calls"] = []
        pre = _robot_snapshot(env)
        out = original_spi(state, slc, env=env)
        _CAP["pending"] = {"calls": list(_CAP["calls"]), "pre": pre, "env": env}
        return out

    instrumented_episode: Any = ep
    instrumented_episode._step_policy_inference = spi

    original_trace = ep._step_build_simulation_trace

    def trace(
        state: _StepLoopState, slc: _StepLoopConfig, *, step_idx: int, sim: _StepSimResult
    ) -> None:
        original_trace(state, slc, step_idx=step_idx, sim=sim)
        pending = _CAP.get("pending")
        if not pending or not state.simulation_step_trace:
            return
        calls = pending["calls"]

        def first(kind: str) -> dict[str, Any] | None:
            return next((c for c in calls if c["kind"] == kind), None)

        solver = first("solver")
        adapter = first("adapter_cmd")
        prefc = first("pref")
        pref_final = first("pref_final")
        obst = first("obstacles")
        safe_c = first("safe_heading")
        fallback = any(c["kind"] == "heuristic_fallback" for c in calls)
        record = {
            "schema": "xorca-step.v2",
            "pref_world": prefc["v"] if prefc else None,
            "pref_final_world": pref_final["v"] if pref_final else None,
            "obstacles": (
                {k: obst[k] for k in ("n_obstacles", "door_gap_m", "max_radius_m")}
                if obst
                else None
            ),
            "solver_world": solver["v"] if solver else None,
            "solver_planner": solver["planner"] if solver else None,
            "adapter_cmd": adapter["v"] if adapter else None,
            "safe_heading": (
                {k: safe_c[k] for k in ("base", "dir", "penalty")} if safe_c else None
            ),
            "heuristic_fallback": bool(fallback),
            "solver_call_count": sum(1 for c in calls if c["kind"] == "solver"),
            "pre": pending["pre"],
            "post": _robot_snapshot(pending["env"]),
        }
        if step_idx == 0:
            record["robot_settings"] = _robot_settings(pending["env"])
        state.simulation_step_trace[-1]["xorca"] = record
        _CAP["pending"] = None

    instrumented_episode._step_build_simulation_trace = trace
    _CAP["installed"] = True


def _scenario_payload(contract: dict[str, Any], contract_dir: Path, cell: str) -> dict[str, Any]:
    matrix_path = (contract_dir / contract["scenario_matrix"]).resolve()
    scenarios = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))["scenarios"]
    match = [s for s in scenarios if s["name"] == cell]
    if len(match) != 1:
        raise SystemExit(f"cell {cell!r} not found exactly once in {matrix_path}")
    scenario = dict(match[0])
    scenario["map_file"] = str((matrix_path.parent / scenario["map_file"]).resolve())
    return scenario


def main() -> int:
    """Run one acquisition job and write its receipt.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--config", required=True)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--seeds", required=True, nargs="+", type=int)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--tag", default="main")
    parser.add_argument(
        "--no-instrument",
        action="store_true",
        help="neutrality check: run the same job without any wrapper",
    )
    args = parser.parse_args()

    args.contract = args.contract.resolve()
    args.out_dir = args.out_dir.resolve()
    contract_dir = args.contract.parent
    contract = yaml.safe_load(args.contract.read_text(encoding="utf-8"))
    forbidden = set(range(111, 141))
    for seed in args.seeds:
        if seed in forbidden or seed >= 10000:
            raise SystemExit("refusing a seed outside the development allowlist")
    configs = {c["key"]: c for c in contract["configurations"]}
    cfg = configs[args.config]

    if not args.no_instrument:
        install_instrumentation()
    from robot_sf.benchmark.map_runner import run_map_batch
    from robot_sf.common.artifact_paths import get_repository_root

    root = get_repository_root()
    # Frozen planner configs include siblings by repository-relative path.
    os.chdir(root)
    scenario = _scenario_payload(contract, contract_dir, args.cell)
    scenario["seeds"] = list(args.seeds)
    robot_cfg = dict(scenario.get("robot_config") or {})
    robot_cfg["type"] = cfg["kinematics"]
    if cfg["kinematics"] == "holonomic":
        robot_cfg["command_mode"] = "vx_vy"
    scenario["robot_config"] = robot_cfg

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.tag}__{args.config}__{args.cell}__{args.seeds[0]}-{args.seeds[-1]}"
    out_path = args.out_dir / f"{stem}.jsonl"
    if out_path.exists():
        out_path.unlink()
    algo_config = cfg.get("algo_config")
    t0 = time.time()
    summary = run_map_batch(
        [scenario],
        out_path,
        root / "robot_sf/benchmark/schemas/episode.schema.v1.json",
        scenario_path=contract_dir / contract["scenario_matrix"],
        horizon=int(contract["horizon_steps"]),
        dt=float(contract["dt"]),
        record_forces=True,
        algo=cfg["algo"],
        algo_config_path=str(root / algo_config) if algo_config else None,
        benchmark_profile=cfg.get("benchmark_profile", "baseline-safe"),
        socnav_missing_prereq_policy="fail-fast",
        record_simulation_step_trace=True,
        workers=1,
        resume=False,
    )
    elapsed = time.time() - t0
    receipt = {
        "config": args.config,
        "cell": args.cell,
        "seeds": list(args.seeds),
        "tag": args.tag,
        "elapsed_s": elapsed,
        "out": out_path.name,
        "summary": {k: summary.get(k) for k in ("total_jobs", "written", "failed_jobs", "failures")}
        if isinstance(summary, dict)
        else None,
    }
    (args.out_dir / f"{stem}.receipt.json").write_text(
        json.dumps(receipt, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps({k: receipt[k] for k in ("config", "cell", "elapsed_s")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
