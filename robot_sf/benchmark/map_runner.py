"""Map-based benchmark runner using Gym environments and scenario YAMLs."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml
from loguru import logger

from robot_sf.benchmark.metrics import EpisodeData, compute_all_metrics, snqi
from robot_sf.benchmark.obstacle_sampling import sample_obstacle_points
from robot_sf.benchmark.path_utils import compute_shortest_path_length
from robot_sf.benchmark.scenario_schema import validate_scenario_list
from robot_sf.benchmark.schema_validator import load_schema, validate_episode
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.gym_env.observation_mode import ObservationMode
from robot_sf.nav.occupancy_grid import GridConfig
from robot_sf.planner.socnav import (
    ORCAPlannerAdapter,
    SACADRLPlannerAdapter,
    SamplingPlannerAdapter,
    SocialForcePlannerAdapter,
    SocNavBenchSamplingAdapter,
    SocNavPlannerConfig,
)
from robot_sf.training.scenario_loader import build_robot_config_from_scenario, load_scenarios

if TYPE_CHECKING:
    from collections.abc import Callable

    from robot_sf.gym_env.unified_config import RobotSimulationConfig


def _config_hash(obj: dict[str, Any]) -> str:
    data = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()[:16]


def _git_hash_fallback() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def compute_episode_id(scenario_params: dict[str, Any], seed: int) -> str:
    scenario_id = (
        scenario_params.get("id")
        or scenario_params.get("name")
        or scenario_params.get("scenario_id")
        or "unknown"
    )
    return f"{scenario_id}--{seed}"


def index_existing(out_path: Path) -> set[str]:
    ids: set[str] = set()
    try:
        with out_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                eid = rec.get("episode_id") if isinstance(rec, dict) else None
                if isinstance(eid, str):
                    ids.add(eid)
    except FileNotFoundError:
        return set()
    return ids


def _parse_algo_config(algo_config_path: str | None) -> dict[str, Any]:
    if not algo_config_path:
        return {}
    path = Path(algo_config_path)
    if not path.exists():
        raise FileNotFoundError(f"Algorithm config file not found: {algo_config_path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise TypeError("Algorithm config must be a mapping (YAML dict).")
    return data


def _build_socnav_config(cfg: dict[str, Any]) -> SocNavPlannerConfig:
    try:
        return SocNavPlannerConfig(**cfg)
    except TypeError:
        return SocNavPlannerConfig()


def _goal_policy(obs: dict[str, Any], *, max_speed: float = 1.0) -> tuple[float, float]:
    robot = obs.get("robot", {})
    goal = obs.get("goal", {})
    robot_pos = np.asarray(robot.get("position", [0.0, 0.0]), dtype=float)
    heading = float(np.asarray(robot.get("heading", [0.0]), dtype=float)[0])
    goal_pos = np.asarray(goal.get("current", [0.0, 0.0]), dtype=float)
    vec = goal_pos - robot_pos
    dist = float(np.linalg.norm(vec))
    if dist < 1e-6:
        return 0.0, 0.0
    desired_heading = float(np.arctan2(vec[1], vec[0]))
    heading_error = ((desired_heading - heading + np.pi) % (2 * np.pi)) - np.pi
    angular = float(np.clip(heading_error, -1.0, 1.0))
    linear = float(np.clip(dist, 0.0, max_speed * max(0.0, 1.0 - abs(heading_error) / np.pi)))
    return linear, angular


def _post_process_metrics(
    metrics_raw: dict[str, Any],
    *,
    snqi_weights: dict[str, float] | None,
    snqi_baseline: dict[str, dict[str, float]] | None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = dict(metrics_raw.items())
    metrics["success"] = bool(metrics.get("success", 0.0) == 1.0)
    fq = {k: v for k, v in metrics.items() if str(k).startswith("force_q")}
    if fq:
        metrics["force_quantiles"] = {
            "q50": float(fq.get("force_q50", float("nan"))),
            "q90": float(fq.get("force_q90", float("nan"))),
            "q95": float(fq.get("force_q95", float("nan"))),
        }
        for k in list(fq.keys()):
            metrics.pop(k, None)
    if snqi_weights is not None:
        metrics["snqi"] = snqi(metrics, snqi_weights, baseline_stats=snqi_baseline)
    return metrics


def _build_policy(  # noqa: C901
    algo: str,
    algo_config: dict[str, Any],
) -> tuple[Callable[[dict[str, Any]], tuple[float, float]], dict[str, Any]]:
    algo_key = algo.lower().strip()
    meta: dict[str, Any] = {"algorithm": algo_key}

    if algo_key in {"goal", "simple", "goal_policy"}:
        meta.update(
            {"status": "ok", "config": algo_config, "config_hash": _config_hash(algo_config)}
        )

        def _policy(obs: dict[str, Any]) -> tuple[float, float]:
            return _goal_policy(obs, max_speed=float(algo_config.get("max_speed", 1.0)))

        return _policy, meta

    socnav_cfg = _build_socnav_config(algo_config)

    if algo_key in {"socnav_sampling", "sampling"}:
        adapter = SamplingPlannerAdapter(config=socnav_cfg)
    elif algo_key in {"social_force", "sf"}:
        adapter = SocialForcePlannerAdapter(config=socnav_cfg)
    elif algo_key in {"orca"}:
        adapter = ORCAPlannerAdapter(config=socnav_cfg)
    elif algo_key in {"sacadrl", "sa_cadrl"}:
        adapter = SACADRLPlannerAdapter(config=socnav_cfg)
    elif algo_key in {"socnav_bench"}:
        adapter = SocNavBenchSamplingAdapter(config=socnav_cfg)
    elif algo_key in {"rvo", "dwa", "teb"}:
        adapter = SamplingPlannerAdapter(config=socnav_cfg)
        meta.update({"status": "placeholder", "fallback_reason": "unimplemented"})
    else:
        raise ValueError(f"Unknown map-based algorithm '{algo}'.")

    if "status" not in meta:
        meta["status"] = "ok"
    meta["config"] = algo_config
    meta["config_hash"] = _config_hash(algo_config)

    def _policy(obs: dict[str, Any]) -> tuple[float, float]:
        return adapter.plan(obs)

    return _policy, meta


def _resolve_seed_list(path: Path) -> dict[str, list[int]]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        return {}
    return {str(k): [int(s) for s in v] for k, v in data.items() if isinstance(v, list)}


def _suite_key(scenario_path: Path) -> str:
    stem = scenario_path.stem.lower()
    if "classic" in stem:
        return "classic_interactions"
    if "francis" in stem:
        return "francis2023"
    return "default"


def _select_seeds(
    scenario: dict[str, Any],
    *,
    suite_seeds: dict[str, list[int]],
    suite_key: str,
) -> list[int]:
    seeds = scenario.get("seeds")
    if isinstance(seeds, list) and seeds:
        return [int(s) for s in seeds]
    if suite_seeds.get(suite_key):
        return list(suite_seeds[suite_key])
    if suite_seeds.get("default"):
        return list(suite_seeds["default"])
    return [0]


def _validate_behavior_sanity(scenario: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    meta = scenario.get("metadata") if isinstance(scenario.get("metadata"), dict) else {}
    behavior = str(meta.get("behavior") or "").strip().lower()
    single_peds = (
        scenario.get("single_pedestrians")
        if isinstance(scenario.get("single_pedestrians"), list)
        else []
    )

    if behavior in {"wait", "join", "leave", "follow", "lead", "accompany"}:
        if not single_peds:
            errors.append("behavior requires single_pedestrians entries")
        else:
            for ped in single_peds:
                if not isinstance(ped, dict):
                    continue
                role = str(ped.get("role") or "").strip().lower()
                if behavior == "wait" and not ped.get("wait_at"):
                    errors.append("wait behavior requires wait_at rules")
                    break
                if behavior in {"join", "leave", "follow", "lead", "accompany"} and not role:
                    errors.append("role behavior requires role field")
                    break

    return errors


def _build_env_config(
    scenario: dict[str, Any],
    *,
    scenario_path: Path,
) -> RobotSimulationConfig:
    config = build_robot_config_from_scenario(scenario, scenario_path=scenario_path)
    config.observation_mode = ObservationMode.SOCNAV_STRUCT
    config.use_occupancy_grid = True
    config.include_grid_in_observation = True
    config.grid_config = GridConfig(
        resolution=0.5,
        width=32.0,
        height=32.0,
        use_ego_frame=True,
    )
    return config


def _vel_and_acc(positions: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    if positions.shape[0] < 2:
        return np.zeros_like(positions), np.zeros_like(positions)
    vel = np.gradient(positions, dt, axis=0)
    acc = np.gradient(vel, dt, axis=0)
    return vel, acc


def _stack_ped_positions(traj: list[np.ndarray], *, fill_value: float = np.nan) -> np.ndarray:
    if not traj:
        return np.zeros((0, 0, 2), dtype=float)
    max_k = max(p.shape[0] for p in traj)
    stacked = np.full((len(traj), max_k, 2), fill_value, dtype=float)
    for i, arr in enumerate(traj):
        if arr.size == 0:
            continue
        stacked[i, : arr.shape[0]] = arr
    return stacked


def _run_map_episode(  # noqa: C901
    scenario: dict[str, Any],
    seed: int,
    *,
    horizon: int | None,
    dt: float | None,
    record_forces: bool,
    snqi_weights: dict[str, float] | None,
    snqi_baseline: dict[str, dict[str, float]] | None,
    algo: str,
    algo_config_path: str | None,
    scenario_path: Path,
) -> dict[str, Any]:
    ts_start = datetime.now(UTC).isoformat()
    start_time = time.time()
    config = _build_env_config(scenario, scenario_path=scenario_path)
    max_steps = int(scenario.get("simulation_config", {}).get("max_episode_steps", 0) or 0)
    horizon_val = int(horizon) if horizon and horizon > 0 else max_steps
    if horizon_val <= 0:
        horizon_val = 200
    if dt is not None and dt > 0:
        config.sim_config.time_per_step_in_secs = float(dt)

    policy_cfg = _parse_algo_config(algo_config_path)
    policy_fn, algo_meta = _build_policy(algo, policy_cfg)

    env = make_robot_env(config=config, seed=int(seed), debug=False)
    obs, _ = env.reset(seed=int(seed))

    robot_positions: list[np.ndarray] = []
    ped_positions: list[np.ndarray] = []
    ped_forces: list[np.ndarray] = []
    reached_goal_step: int | None = None

    map_def = None
    goal_vec = np.zeros(2, dtype=float)
    try:
        for step_idx in range(horizon_val):
            action_v, action_w = policy_fn(obs)
            action = np.array([action_v, action_w], dtype=float)
            obs, _reward, terminated, truncated, info = env.step(action)

            robot_pos = np.asarray(env.simulator.robot_pos[0], dtype=float)
            peds = np.asarray(env.simulator.ped_pos, dtype=float)
            if record_forces:
                forces = getattr(env.simulator, "last_ped_forces", None)
                if forces is None:
                    forces_arr = np.zeros_like(peds, dtype=float)
                else:
                    forces_arr = np.asarray(forces, dtype=float)
                    if forces_arr.shape != peds.shape:
                        forces_arr = np.zeros_like(peds, dtype=float)
            else:
                forces_arr = np.zeros_like(peds, dtype=float)

            robot_positions.append(robot_pos)
            ped_positions.append(peds)
            ped_forces.append(forces_arr)

            if reached_goal_step is None and bool(info.get("success")):
                reached_goal_step = step_idx
            if terminated or truncated:
                break
        if getattr(env, "simulator", None) is not None:
            map_def = env.simulator.map_def
            goal_vec = np.asarray(env.simulator.goal_pos[0], dtype=float)
    finally:
        env.close()

    robot_pos_arr = np.asarray(robot_positions, dtype=float)
    robot_vel_arr, robot_acc_arr = _vel_and_acc(
        robot_pos_arr, config.sim_config.time_per_step_in_secs
    )
    ped_pos_arr = _stack_ped_positions(ped_positions)
    ped_forces_arr = _stack_ped_positions(ped_forces, fill_value=np.nan)

    obstacles = (
        sample_obstacle_points(map_def.obstacles, map_def.bounds) if map_def is not None else None
    )
    if robot_pos_arr.size:
        shortest_path = compute_shortest_path_length(map_def, robot_pos_arr[0], goal_vec)
    else:
        shortest_path = float("nan")
    env.close()

    if robot_pos_arr.size == 0:
        metrics_raw = {"success": 0.0, "time_to_goal_norm": float("nan"), "collisions": 0.0}
    else:
        ep = EpisodeData(
            robot_pos=robot_pos_arr,
            robot_vel=robot_vel_arr,
            robot_acc=robot_acc_arr,
            peds_pos=ped_pos_arr,
            ped_forces=ped_forces_arr,
            obstacles=obstacles,
            goal=goal_vec,
            dt=float(config.sim_config.time_per_step_in_secs),
            reached_goal_step=reached_goal_step,
        )
        metrics_raw = compute_all_metrics(ep, horizon=horizon_val, shortest_path_len=shortest_path)
    metrics = _post_process_metrics(
        metrics_raw,
        snqi_weights=snqi_weights,
        snqi_baseline=snqi_baseline,
    )

    ts_end = datetime.now(UTC).isoformat()
    scenario_params = dict(scenario)
    scenario_id = (
        scenario.get("name") or scenario.get("scenario_id") or scenario.get("id") or "unknown"
    )
    scenario_params.setdefault("id", scenario_id)
    scenario_params.setdefault("algo", algo)
    steps_taken = int(robot_pos_arr.shape[0])
    wall_time = float(max(1e-9, time.time() - start_time))
    timing = {"steps_per_second": float(steps_taken) / wall_time if wall_time > 0 else 0.0}
    status = "success" if metrics.get("success") else "failure"
    if metrics.get("collisions"):
        status = "collision"
    record = {
        "version": "v1",
        "episode_id": compute_episode_id(scenario_params, seed),
        "scenario_id": scenario_id,
        "seed": seed,
        "scenario_params": scenario_params,
        "metrics": metrics,
        "algorithm_metadata": algo_meta,
        "algo": algo,
        "config_hash": _config_hash(scenario_params),
        "git_hash": _git_hash_fallback(),
        "timestamps": {"start": ts_start, "end": ts_end},
        "status": status,
        "steps": steps_taken,
        "horizon": horizon_val,
        "wall_time_sec": wall_time,
        "timing": timing,
    }
    return record


def _write_validated(out_path: Path, schema: dict[str, Any], record: dict[str, Any]) -> None:
    validate_episode(record, schema)
    with out_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def _run_map_job_worker(job: tuple[dict[str, Any], int, dict[str, Any]]) -> dict[str, Any]:
    scenario, seed, params = job
    return _run_map_episode(
        scenario,
        seed,
        horizon=params.get("horizon"),
        dt=params.get("dt"),
        record_forces=bool(params.get("record_forces", True)),
        snqi_weights=params.get("snqi_weights"),
        snqi_baseline=params.get("snqi_baseline"),
        algo=str(params.get("algo", "goal")),
        algo_config_path=params.get("algo_config_path"),
        scenario_path=Path(params.get("scenario_path")),
    )


def run_map_batch(  # noqa: C901
    scenarios_or_path: list[dict[str, Any]] | str | Path,
    out_path: str | Path,
    schema_path: str | Path,
    *,
    horizon: int | None = None,
    dt: float | None = None,
    record_forces: bool = True,
    snqi_weights: dict[str, float] | None = None,
    snqi_baseline: dict[str, dict[str, float]] | None = None,
    algo: str = "goal",
    algo_config_path: str | None = None,
    workers: int = 1,
    resume: bool = True,
) -> dict[str, Any]:
    """Run map-based scenarios and append episode records.

    Returns:
        Summary payload with counts and failure details.
    """
    scenarios_is_path = isinstance(scenarios_or_path, (str, Path))
    if scenarios_is_path:
        scenario_path = Path(scenarios_or_path)
        scenarios = load_scenarios(scenario_path)
    else:
        scenario_path = Path(".")
        scenarios = list(scenarios_or_path)

    errors = validate_scenario_list([dict(s) for s in scenarios])
    if errors:
        raise ValueError(f"Scenario validation failed: {errors[:3]} (total {len(errors)})")

    suite_seeds = _resolve_seed_list(Path("configs/benchmarks/seed_list_v1.yaml"))
    suite_key = _suite_key(scenario_path)

    filtered: list[dict[str, Any]] = []
    for scenario in scenarios:
        if (
            scenario.get("supported") is False
            or scenario.get("metadata", {}).get("supported") is False
        ):
            continue
        errors = _validate_behavior_sanity(scenario)
        if errors:
            logger.warning("Skipping scenario '{}': {}", scenario.get("name"), errors)
            continue
        filtered.append(scenario)

    jobs: list[tuple[dict[str, Any], int]] = []
    for scenario in filtered:
        seeds = _select_seeds(scenario, suite_seeds=suite_seeds, suite_key=suite_key)
        for seed in seeds:
            jobs.append((scenario, int(seed)))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    schema = load_schema(schema_path)

    if resume and out_path.exists():
        existing = index_existing(out_path)
        jobs = [
            (sc, seed) for sc, seed in jobs if compute_episode_id(dict(sc), seed) not in existing
        ]

    fixed_params = {
        "horizon": horizon,
        "dt": dt,
        "record_forces": record_forces,
        "snqi_weights": snqi_weights,
        "snqi_baseline": snqi_baseline,
        "algo": algo,
        "algo_config_path": algo_config_path,
        "scenario_path": str(scenario_path),
    }

    wrote = 0
    failures: list[dict[str, Any]] = []
    if workers <= 1:
        for scenario, seed in jobs:
            try:
                rec = _run_map_job_worker((scenario, seed, fixed_params))
                _write_validated(out_path, schema, rec)
                wrote += 1
            except Exception as exc:  # pragma: no cover - error path
                failures.append(
                    {
                        "scenario_id": scenario.get("name", "unknown"),
                        "seed": seed,
                        "error": repr(exc),
                    }
                )
    else:
        with ProcessPoolExecutor(max_workers=int(workers)) as ex:
            future_to_job: dict[Any, tuple[dict[str, Any], int]] = {}
            for scenario, seed in jobs:
                fut = ex.submit(_run_map_job_worker, (scenario, seed, fixed_params))
                future_to_job[fut] = (scenario, seed)
            for fut in as_completed(future_to_job):
                scenario, seed = future_to_job[fut]
                try:
                    rec = fut.result()
                    _write_validated(out_path, schema, rec)
                    wrote += 1
                except Exception as exc:  # pragma: no cover
                    failures.append(
                        {
                            "scenario_id": scenario.get("name", "unknown"),
                            "seed": seed,
                            "error": repr(exc),
                        }
                    )

    return {
        "total_jobs": len(jobs),
        "written": wrote,
        "failures": failures,
        "out_path": str(out_path),
    }


__all__ = ["run_map_batch"]
