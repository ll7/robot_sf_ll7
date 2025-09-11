"""Episode runner scaffold for the Social Navigation Benchmark.

Responsibilities:
 - Load scenario parameter matrix (YAML list) and expand repeats
 - Deterministically generate scenarios (using `generate_scenario`)
 - Step a simple baseline policy toward each agent's goal (very naive)
 - Collect per-step robot + pedestrian states and (optionally) forces
 - Build EpisodeData and compute metrics + SNQI (weights optional)
 - Validate record against JSON schema and write JSONL

This is an initial lightweight runner; performance optimizations and
multi-processing are future work.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import yaml

from robot_sf.benchmark.metrics import EpisodeData, compute_all_metrics, snqi
from robot_sf.benchmark.scenario_generator import generate_scenario
from robot_sf.benchmark.schema_validator import load_schema, validate_episode
from robot_sf.sim.fast_pysf_wrapper import FastPysfWrapper


def _load_baseline_planner(algo: str, algo_config_path: Optional[str], seed: int):
    """Load and construct a baseline planner from the registry.

    Returns (planner, ObservationCls, config_dict).
    """
    try:
        from robot_sf.baselines import get_baseline
        from robot_sf.baselines.social_force import Observation
    except ImportError as e:
        raise RuntimeError(f"Failed to import baseline algorithms: {e}")

    try:
        planner_class = get_baseline(algo)
    except KeyError:
        from robot_sf.baselines import list_baselines

        available = list_baselines()
        raise ValueError(f"Unknown algorithm '{algo}'. Available: {available}")

    # Load configuration if provided
    config = {}
    if algo_config_path:
        config_path = Path(algo_config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Algorithm config file not found: {algo_config_path}")

        with config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

    planner = planner_class(config, seed=seed)
    return planner, Observation, config


def _build_observation(ObservationCls, robot_pos, robot_vel, robot_goal, ped_positions, dt):
    agents = [
        {"position": pos.tolist(), "velocity": [0.0, 0.0], "radius": 0.35} for pos in ped_positions
    ]
    return ObservationCls(
        dt=dt,
        robot={
            "position": robot_pos.tolist(),
            "velocity": robot_vel.tolist(),
            "goal": robot_goal.tolist(),
            "radius": 0.3,
        },
        agents=agents,
        obstacles=[],
    )


def _git_hash_fallback() -> str:
    # Best effort; avoid importing subprocess if not needed later
    try:
        import subprocess

        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:  # pragma: no cover
        return "unknown"


def _config_hash(obj: Any) -> str:
    data = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()[:16]


def load_scenario_matrix(path: str | Path) -> List[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        docs = list(yaml.safe_load_all(f))
    # Allow either YAML stream of docs or a single list
    if len(docs) == 1 and isinstance(docs[0], list):
        scenarios = docs[0]
    else:
        scenarios = docs
    return [dict(s) for s in scenarios]


def _simple_robot_policy(robot_pos: np.ndarray, goal: np.ndarray, speed: float = 1.0) -> np.ndarray:
    """Return a velocity vector pointing toward goal with capped speed."""
    vec = goal - robot_pos
    dist = float(np.linalg.norm(vec))
    if dist < 1e-9:
        return np.zeros(2)
    dir_unit = vec / dist
    v = dir_unit * min(speed, dist)
    return v


def _prepare_robot_points(
    robot_start: Sequence[float] | None, robot_goal: Sequence[float] | None
) -> tuple[np.ndarray, np.ndarray]:
    if robot_start is None:
        rs = np.array([0.3, 3.0], dtype=float)
    else:
        rs = np.asarray(robot_start, dtype=float)
    if robot_goal is None:
        rg = np.array([9.7, 3.0], dtype=float)
    else:
        rg = np.asarray(robot_goal, dtype=float)
    return rs, rg


def _build_episode_data(
    robot_pos_traj: list[np.ndarray],
    robot_vel_traj: list[np.ndarray],
    robot_acc_traj: list[np.ndarray],
    peds_pos_traj: list[np.ndarray],
    ped_forces_traj: list[np.ndarray],
    goal: np.ndarray,
    dt: float,
    reached_goal_step: int | None,
) -> EpisodeData:
    return EpisodeData(
        robot_pos=np.vstack(robot_pos_traj),
        robot_vel=np.vstack(robot_vel_traj),
        robot_acc=np.vstack(robot_acc_traj),
        peds_pos=(np.stack(peds_pos_traj) if peds_pos_traj else np.zeros((0, 0, 2))),
        ped_forces=(np.stack(ped_forces_traj) if ped_forces_traj else np.zeros((0, 0, 2))),
        goal=goal,
        dt=dt,
        reached_goal_step=reached_goal_step,
    )


def _create_robot_policy(algo: str, algo_config_path: Optional[str], seed: int):
    """Create a robot policy function based on the specified algorithm."""

    if algo == "simple_policy":
        # Original simple policy for backward compatibility
        def policy(
            robot_pos: np.ndarray,
            _robot_vel: np.ndarray,
            robot_goal: np.ndarray,
            _ped_positions: np.ndarray,
            _dt: float,
        ) -> np.ndarray:
            return _simple_robot_policy(robot_pos, robot_goal, speed=1.0)

        return policy, {}

    # Load baseline planner
    planner, Observation, _config = _load_baseline_planner(algo, algo_config_path, seed)

    def policy_fn(
        robot_pos: np.ndarray,
        robot_vel: np.ndarray,
        robot_goal: np.ndarray,
        ped_positions: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Policy function that uses the baseline planner."""
        # Create observation
        obs = _build_observation(Observation, robot_pos, robot_vel, robot_goal, ped_positions, dt)

        # Get action from planner
        action = planner.step(obs)

        # Convert action to velocity (handle both action spaces)
        if "vx" in action and "vy" in action:
            return np.array([action["vx"], action["vy"]], dtype=float)
        elif "v" in action and "omega" in action:
            # Simple conversion from unicycle to velocity
            # This is a simplification - real conversion would need robot heading
            v = action["v"]
            _omega = action["omega"]  # Not used in this simplified conversion
            # Assume current velocity direction with magnitude adjustment
            current_speed = np.linalg.norm(robot_vel)
            if current_speed > 1e-6:
                direction = robot_vel / current_speed
                return direction * v
            else:
                # If stationary, move toward goal
                goal_dir = robot_goal - robot_pos
                if np.linalg.norm(goal_dir) > 1e-6:
                    return goal_dir / np.linalg.norm(goal_dir) * v
                else:
                    return np.zeros(2)
        else:
            raise ValueError(f"Invalid action format from {algo}: {action}")

    # Get metadata for episode record
    metadata = planner.get_metadata() if hasattr(planner, "get_metadata") else {"algorithm": algo}

    return policy_fn, metadata


def run_episode(
    scenario_params: Dict[str, Any],
    seed: int,
    *,
    horizon: int = 100,
    dt: float = 0.1,
    robot_start: Sequence[float] | None = None,
    robot_goal: Sequence[float] | None = None,
    record_forces: bool = True,
    snqi_weights: Dict[str, float] | None = None,
    snqi_baseline: Dict[str, Dict[str, float]] | None = None,
    algo: str = "simple_policy",
    algo_config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a single episode and return a metrics record dict.

    The robot can use different algorithms based on the 'algo' parameter.
    """
    # Create robot policy based on algorithm
    robot_policy, algo_metadata = _create_robot_policy(algo, algo_config_path, seed)

    gen = generate_scenario(scenario_params, seed=seed)
    sim = gen.simulator
    if sim is None:
        raise RuntimeError("pysocialforce not available; cannot run episode")
    wrapper = FastPysfWrapper(sim)

    # Determine robot start/goal: default start left boundary mid-height, goal right boundary mid-height
    robot_start_arr, robot_goal_arr = _prepare_robot_points(robot_start, robot_goal)

    robot_pos = robot_start_arr.copy()
    robot_vel = np.zeros(2)

    # Buffers
    robot_pos_traj: List[np.ndarray] = []
    robot_vel_traj: List[np.ndarray] = []
    robot_acc_traj: List[np.ndarray] = []
    peds_pos_traj: List[np.ndarray] = []
    ped_forces_traj: List[np.ndarray] = []

    reached_goal_step: int | None = None
    goal_radius = 0.3

    last_vel = robot_vel.copy()

    def _step_robot(
        curr_pos: np.ndarray, curr_vel: np.ndarray, ped_positions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        new_vel = robot_policy(curr_pos, curr_vel, robot_goal_arr, ped_positions, dt)
        new_pos = curr_pos + new_vel * dt
        acc_vec = (new_vel - curr_vel) / dt
        return new_pos, new_vel, acc_vec

    for t in range(horizon):
        # Capture pedestrian positions directly via pysocialforce API
        ped_pos = sim.peds.pos().copy()
        peds_pos_traj.append(ped_pos)

        # Policy (now algorithm-based)
        robot_pos, robot_vel, acc = _step_robot(robot_pos, last_vel, ped_pos)
        last_vel = robot_vel.copy()

        # Force sampling per pedestrian if requested
        if record_forces:
            forces = np.zeros_like(ped_pos)
            for i, p in enumerate(ped_pos):
                forces[i] = wrapper.get_forces_at(p)
            ped_forces_traj.append(forces)
        else:
            ped_forces_traj.append(np.zeros_like(ped_pos))

        robot_pos_traj.append(robot_pos.copy())
        robot_vel_traj.append(robot_vel.copy())
        robot_acc_traj.append(acc.copy())

        # Goal check
        if reached_goal_step is None and np.linalg.norm(robot_goal_arr - robot_pos) < goal_radius:
            reached_goal_step = t
            break

        # Advance pedestrian simulation one step
        sim.step()

    ep: EpisodeData = _build_episode_data(
        robot_pos_traj,
        robot_vel_traj,
        robot_acc_traj,
        peds_pos_traj,
        ped_forces_traj,
        np.asarray(robot_goal_arr, dtype=float),
        dt,
        reached_goal_step,
    )

    metrics_raw = compute_all_metrics(ep, horizon=horizon)
    metrics = _post_process_metrics(
        metrics_raw, snqi_weights=snqi_weights, snqi_baseline=snqi_baseline
    )

    # Build record per schema
    record = {
        "episode_id": f"{scenario_params.get('id', 'unknown')}--{seed}",
        "scenario_id": scenario_params.get("id", "unknown"),
        "seed": seed,
        "scenario_params": scenario_params,
        "metrics": metrics,
        # Include algorithm metadata for verification / reproducibility
        "algorithm_metadata": algo_metadata,
        "config_hash": _config_hash(scenario_params),
        "git_hash": _git_hash_fallback(),
        "timestamps": {
            "start": datetime.now(timezone.utc).isoformat(),
            "end": datetime.now(timezone.utc).isoformat(),
        },
    }
    return record


def validate_and_write(
    record: Dict[str, Any], schema_path: str | Path, out_path: str | Path
) -> None:
    schema = load_schema(schema_path)
    validate_episode(record, schema)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


__all__ = [
    "load_scenario_matrix",
    "run_episode",
    "run_batch",
    "validate_and_write",
]


def _post_process_metrics(
    metrics_raw: Dict[str, Any],
    *,
    snqi_weights: Dict[str, float] | None,
    snqi_baseline: Dict[str, Dict[str, float]] | None,
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {k: v for k, v in metrics_raw.items()}
    metrics["success"] = bool(metrics.get("success", 0.0) == 1.0)
    fq = {k: v for k, v in metrics.items() if k.startswith("force_q")}
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
    for count_key in ("collisions", "near_misses", "force_exceed_events"):
        if count_key in metrics and metrics[count_key] is not None:
            try:
                metrics[count_key] = int(metrics[count_key])
            except Exception:  # pragma: no cover
                pass
    return metrics


def _expand_jobs(
    scenarios: List[Dict[str, Any]], base_seed: int = 0, repeats_override: int | None = None
) -> List[tuple[Dict[str, Any], int]]:
    jobs: List[tuple[Dict[str, Any], int]] = []
    for sc in scenarios:
        reps = int(sc.get("repeats", 1)) if repeats_override is None else int(repeats_override)
        for r in range(reps):
            jobs.append((sc, base_seed + r))
    return jobs


def run_batch(
    scenarios_or_path: List[Dict[str, Any]] | str | Path,
    out_path: str | Path,
    schema_path: str | Path,
    *,
    base_seed: int = 0,
    repeats_override: int | None = None,
    horizon: int = 100,
    dt: float = 0.1,
    record_forces: bool = True,
    snqi_weights: Dict[str, float] | None = None,
    snqi_baseline: Dict[str, Dict[str, float]] | None = None,
    append: bool = True,
    fail_fast: bool = False,
    progress_cb: Optional[
        Callable[[int, int, Dict[str, Any], int, bool, Optional[str]], None]
    ] = None,
    algo: str = "simple_policy",
    algo_config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a batch of episodes and write JSONL records.

    scenarios_or_path: either a list of scenario dicts or a YAML file path.
    Returns a summary dict with counts and failures.
    """
    # Load scenarios
    if isinstance(scenarios_or_path, (str, Path)):
        scenarios = load_scenario_matrix(scenarios_or_path)
    else:
        scenarios = scenarios_or_path

    jobs = _expand_jobs(scenarios, base_seed=base_seed, repeats_override=repeats_override)

    # Prepare output
    out_path = Path(out_path)
    if not append and out_path.exists():
        out_path.unlink()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    schema = load_schema(schema_path)

    wrote = 0
    failures: List[Dict[str, Any]] = []
    total = len(jobs)
    for idx, (sc, seed) in enumerate(jobs, start=1):
        try:
            rec = run_episode(
                sc,
                seed,
                horizon=horizon,
                dt=dt,
                record_forces=record_forces,
                snqi_weights=snqi_weights,
                snqi_baseline=snqi_baseline,
                algo=algo,
                algo_config_path=algo_config_path,
            )
            # Validate and append
            validate_episode(rec, schema)
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            wrote += 1
            if progress_cb is not None:
                try:
                    progress_cb(idx, total, sc, seed, True, None)
                except Exception:  # pragma: no cover - progress best-effort
                    pass
        except Exception as e:  # pragma: no cover - error path
            failures.append(
                {
                    "scenario_id": sc.get("id", "unknown"),
                    "seed": seed,
                    "error": repr(e),
                }
            )
            if progress_cb is not None:
                try:
                    progress_cb(idx, total, sc, seed, False, repr(e))
                except Exception:  # pragma: no cover
                    pass
            if fail_fast:
                raise

    return {
        "total_jobs": len(jobs),
        "written": wrote,
        "failures": failures,
        "out_path": str(out_path),
    }
