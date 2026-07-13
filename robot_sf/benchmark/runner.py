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

import json
import multiprocessing as mp
import os
import platform
import shlex
import sys
import time
import uuid
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeoutError
from datetime import (
    UTC,  # type: ignore[attr-defined]
    datetime,
)
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import jsonschema
import numpy as np
import yaml
from loguru import logger

try:
    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
except ImportError:  # pragma: no cover - optional dependency
    ImageSequenceClip = None  # type: ignore[assignment]

try:
    from robot_sf.baselines import SIMPLE_POLICY_ALIASES, get_baseline, list_baselines
    from robot_sf.baselines.social_force import Observation
except ImportError as exc:  # pragma: no cover - optional baseline dependency
    get_baseline = None  # type: ignore[assignment]
    list_baselines = None  # type: ignore[assignment]
    Observation = None  # type: ignore[assignment]
    SIMPLE_POLICY_ALIASES = frozenset({"simple_policy", "goal", "simple", "goal_policy"})
    _BASELINE_IMPORT_ERROR = exc
else:
    _BASELINE_IMPORT_ERROR = None

from robot_sf.benchmark.algorithm_metadata import enrich_algorithm_metadata
from robot_sf.benchmark.circuit_breaker import (  # noqa: F401 - compatibility export.
    _CIRCUIT_BREAKER_MSG_PREFIX_LEN,
    DEFAULT_CIRCUIT_BREAKER_THRESHOLD,
    build_abort_metadata,
    error_signature,
    normalize_circuit_breaker_threshold,
)
from robot_sf.benchmark.constants import EPISODE_SCHEMA_VERSION
from robot_sf.benchmark.event_ledger import validate_record_event_ledger
from robot_sf.benchmark.local_model_artifacts import validate_no_local_model_artifacts
from robot_sf.benchmark.manifest import load_manifest, save_manifest
from robot_sf.benchmark.map_runner import run_map_batch
from robot_sf.benchmark.metrics import EpisodeData, compute_all_metrics, post_process_metrics
from robot_sf.benchmark.obstacle_sampling import sample_obstacle_points
from robot_sf.benchmark.scenario_generator import generate_scenario
from robot_sf.benchmark.schema_validator import load_schema, validate_episode
from robot_sf.benchmark.termination_reason import (
    build_outcome_payload,
    metric_scalar,
    outcome_contradictions,
    resolve_termination_reason,
    status_from_termination_reason,
)
from robot_sf.benchmark.thresholds import ensure_metric_parameters
from robot_sf.benchmark.utils import (
    _config_hash,
    _git_hash_fallback,
    attach_track_metadata,
    compute_episode_id,
    episode_identity_hash,
    index_existing,
    normalize_track_field,
    validate_episode_success_integrity,
)
from robot_sf.sim.fast_pysf_wrapper import FastPysfWrapper
from robot_sf.training.scenario_loader import load_scenarios
from robot_sf.training.task_bundles import is_task_bundle_reference

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


DEFAULT_BENCHMARK_ROBOT_RADIUS_M = 0.3
DEFAULT_BENCHMARK_PED_RADIUS_M = 0.35


def _apply_track_metadata_to_scenarios(
    scenarios: list[dict[str, Any]],
    *,
    observation_mode: str | None,
    observation_level: str | None,
    benchmark_track: str | None,
    track_schema_version: str | None,
) -> list[dict[str, Any]]:
    """Attach track metadata to scenario payloads used for rows and resume identity.

    Returns:
        Scenario payloads with additive track fields, or the original list when no track is set.
    """
    if (
        observation_mode is None
        and observation_level is None
        and benchmark_track is None
        and track_schema_version is None
    ):
        return scenarios
    tracked: list[dict[str, Any]] = []
    for scenario in scenarios:
        payload = dict(scenario)
        if observation_mode is not None:
            payload["observation_mode"] = observation_mode
        if observation_level is not None:
            payload["observation_level"] = observation_level
        if benchmark_track is not None:
            payload["benchmark_track"] = benchmark_track
        if track_schema_version is not None:
            payload["track_schema_version"] = track_schema_version
        tracked.append(payload)
    return tracked


def _load_baseline_planner(algo: str, algo_config_path: str | None, seed: int):
    """Load and construct a baseline planner from the registry.

    Returns (planner, ObservationCls, config_dict).

    Returns:
        Tuple of (planner instance, Observation class, config dict).
    """
    if get_baseline is None or Observation is None:
        raise RuntimeError(f"Failed to import baseline algorithms: {_BASELINE_IMPORT_ERROR}")

    try:
        planner_class = get_baseline(algo)
    except KeyError:
        available = list_baselines() if list_baselines is not None else []
        raise ValueError(f"Unknown algorithm '{algo}'. Available: {available}")

    # Load configuration if provided
    config = {}
    if algo_config_path:
        config_path = Path(algo_config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Algorithm config file not found: {algo_config_path}")

        with config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
            if not isinstance(cfg, dict):
                raise TypeError("Algorithm config must be a mapping (YAML dict).")
            validate_no_local_model_artifacts(cfg, config_path=config_path)
            config = cfg
    planner = planner_class(config, seed=seed)
    return planner, Observation, config


def _positive_float_or_default(value: Any, default: float) -> float:
    """Return a positive finite float or a fallback default."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(parsed) or parsed <= 0.0:
        return default
    return parsed


def _nested_value(mapping: dict[str, Any], parent_key: str, child_key: str) -> Any:
    """Return a nested mapping value when present."""
    if not isinstance(mapping, dict):
        return None
    parent = mapping.get(parent_key)
    if isinstance(parent, dict):
        return parent.get(child_key)
    return None


def _scenario_robot_radius_m(scenario_params: dict[str, Any]) -> float:
    """Resolve robot radius from scenario metadata or benchmark defaults.

    Returns:
        Positive robot radius in meters.
    """
    if not isinstance(scenario_params, dict):
        return DEFAULT_BENCHMARK_ROBOT_RADIUS_M
    for value in (
        scenario_params.get("robot_radius"),
        _nested_value(scenario_params, "robot_config", "radius"),
        _nested_value(scenario_params, "robot_config", "robot_radius"),
        _nested_value(scenario_params, "robot", "radius"),
    ):
        radius = _positive_float_or_default(value, float("nan"))
        if np.isfinite(radius):
            return radius
    return DEFAULT_BENCHMARK_ROBOT_RADIUS_M


def _scenario_ped_radius_m(scenario_params: dict[str, Any]) -> float:
    """Resolve pedestrian radius from scenario metadata or benchmark defaults.

    Returns:
        Positive pedestrian radius in meters.
    """
    if not isinstance(scenario_params, dict):
        return DEFAULT_BENCHMARK_PED_RADIUS_M
    for value in (
        scenario_params.get("ped_radius"),
        scenario_params.get("pedestrian_radius"),
        _nested_value(scenario_params, "simulation_config", "ped_radius"),
        _nested_value(scenario_params, "sim_config", "ped_radius"),
        _nested_value(scenario_params, "pedestrians", "radius"),
    ):
        radius = _positive_float_or_default(value, float("nan"))
        if np.isfinite(radius):
            return radius
    return DEFAULT_BENCHMARK_PED_RADIUS_M


def _build_observation(
    ObservationCls,
    robot_pos,
    robot_vel,
    robot_goal,
    ped_positions,
    dt,
    *,
    robot_radius: float = DEFAULT_BENCHMARK_ROBOT_RADIUS_M,
    ped_radius: float = DEFAULT_BENCHMARK_PED_RADIUS_M,
):
    """Build an Observation instance from robot and pedestrian state.

    Args:
        ObservationCls: The Observation class to instantiate.
        robot_pos: Current robot position array.
        robot_vel: Current robot velocity array.
        robot_goal: Robot goal position array.
        ped_positions: Array of pedestrian positions.
        dt: Timestep duration.
        robot_radius: Robot radius in meters.
        ped_radius: Shared pedestrian radius in meters.

    Returns:
        Observation instance with current state data.
    """
    agents = [
        {"position": pos.tolist(), "velocity": [0.0, 0.0], "radius": float(ped_radius)}
        for pos in ped_positions
    ]
    return ObservationCls(
        dt=dt,
        robot={
            "position": robot_pos.tolist(),
            "velocity": robot_vel.tolist(),
            "goal": robot_goal.tolist(),
            "radius": float(robot_radius),
        },
        agents=agents,
        obstacles=[],
    )


# Safety/robustness defaults for any baseline policy
POLICY_STEP_TIMEOUT_SECS: float = 0.2  # step(obs) time budget; fallback to zero action on timeout
FINAL_SPEED_CLAMP: float = 2.0  # m/s cap to prevent unrealistic velocities


_episode_identity_hash = episode_identity_hash


def _planner_step_worker(conn: Any, planner: Any) -> None:
    """Run planner steps in an isolated child process."""
    try:
        while True:
            try:
                command, payload = conn.recv()
            except EOFError:
                break
            if command == "close":
                break
            if command != "step":
                conn.send(("error", ("RuntimeError", f"unknown command: {command!r}")))
                continue
            try:
                conn.send(("ok", planner.step(payload)))
            except Exception as exc:  # pragma: no cover - defensive child-process path
                conn.send(("error", (type(exc).__name__, str(exc))))
    finally:
        conn.close()


class _PlannerStepProcess:
    """Persistent process boundary for planner.step with hard timeout cleanup."""

    def __init__(self, planner: Any, *, timeout_s: float) -> None:
        if "fork" not in mp.get_all_start_methods():
            raise RuntimeError(
                "planner step timeout isolation requires multiprocessing fork support"
            )
        self._planner = planner
        self._timeout_s = timeout_s
        self._ctx = mp.get_context("fork")
        self._process: mp.Process | None = None
        self._conn: Any | None = None

    def step(self, obs: Any) -> Any:
        """Run one planner step or raise when timeout/isolation fails.

        Returns:
            Planner action payload returned by the worker process.
        """
        self._ensure_worker()
        assert self._conn is not None
        assert self._process is not None
        try:
            self._conn.send(("step", obs))
        except (BrokenPipeError, EOFError, OSError) as exc:
            self.close()
            raise RuntimeError("planner step worker was unavailable") from exc

        deadline = time.monotonic() + self._timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._terminate_worker()
                raise FuturesTimeoutError()
            if self._conn.poll(min(remaining, 0.01)):
                try:
                    status, payload = self._conn.recv()
                except EOFError as exc:
                    self.close()
                    raise RuntimeError(
                        "planner step worker exited before returning an action"
                    ) from exc
                if status == "ok":
                    return payload
                error_type, message = payload
                raise RuntimeError(f"Planner step failed in worker ({error_type}: {message})")
            if not self._process.is_alive():
                self._process.join(timeout=0)
                self.close()
                raise RuntimeError("planner step worker exited without returning an action")

    def close(self) -> None:
        """Close the worker process and IPC handle."""
        conn = self._conn
        process = self._process
        self._conn = None
        self._process = None

        if conn is not None:
            try:
                if process is not None and process.is_alive():
                    conn.send(("close", None))
            except (BrokenPipeError, EOFError, OSError):
                pass
            conn.close()
        if process is not None:
            process.join(timeout=0.1)
            if process.is_alive():
                self._terminate_process(process)

    def _ensure_worker(self) -> None:
        """Start the persistent worker process if needed."""
        if self._process is not None and self._process.is_alive() and self._conn is not None:
            return
        self.close()
        parent_conn, child_conn = self._ctx.Pipe(duplex=True)
        process = self._ctx.Process(target=_planner_step_worker, args=(child_conn, self._planner))
        process.start()
        child_conn.close()
        self._process = process
        self._conn = parent_conn

    def _terminate_worker(self) -> None:
        """Terminate the current worker after a timeout."""
        process = self._process
        conn = self._conn
        self._process = None
        self._conn = None
        if conn is not None:
            conn.close()
        if process is not None:
            self._terminate_process(process)

    @staticmethod
    def _terminate_process(process: mp.Process) -> None:
        """Terminate, then kill if necessary, and reap a worker process."""
        if process.is_alive():
            process.terminate()
            process.join(timeout=0.1)
        if process.is_alive():
            process.kill()
            process.join(timeout=0.1)


def load_scenario_matrix(path: str | Path) -> list[dict[str, Any]]:
    """Load a scenario matrix from YAML (stream, list, or mapping manifest).

    Routing (see issues #5429, #5433):
      - A **multi-document** YAML stream is an abstract scenario matrix; each
        document is returned directly without include expansion.
      - A **single document that is a list** is likewise an abstract scenario
        matrix (the natural ``yaml.safe_dump([s1, s2])`` form) and is returned
        directly. It is *not* sent through the map-oriented manifest loader,
        which would emit misleading ``missing name``/``no map_file`` warnings
        for abstract benchmark scenarios that legitimately carry neither.
      - A **single document that is a mapping** containing a ``scenarios:`` list
        of purely abstract entries (no ``name``/``scenario_id``/``map_file``/``map_id``)
        and no manifest-only keys (``includes``, ``select_scenarios``,
        ``scenario_overrides``, ``scenario_overrides_by_name``, ``map_search_paths``)
        is likewise an abstract scenario matrix and is returned directly.
      - All other **single-document mappings** are treated as manifests and
        deferred to the include-aware :func:`load_scenarios` (supports
        ``includes``, ``scenarios:``, ``select_scenarios``, overrides, and
        per-scenario ``map_file``/``map_id`` references).

    Args:
        path: Scenario matrix YAML path or a ``bundle:`` task-bundle reference.

    Returns:
        List of scenario dictionaries.

    Raises:
        ValueError: If the file is empty or a single-document list/mapping
            yields no scenarios (fail closed instead of a silent ``written=0``
            run).
    """
    if is_task_bundle_reference(path):
        return [dict(s) for s in load_scenarios(path)]

    scenario_path = Path(path)
    with scenario_path.open("r", encoding="utf-8") as f:
        docs = list(yaml.safe_load_all(f))
    if not docs:
        raise ValueError(f"Scenario matrix '{scenario_path}' is empty.")
    # Preserve legacy YAML stream behavior (multiple docs) without include expansion.
    if len(docs) > 1:
        return [dict(s) for s in docs]
    # A single-document top-level list is an abstract scenario matrix, symmetric
    # with the multi-document stream above; return it directly rather than routing
    # abstract scenarios through the map-oriented manifest validator.
    single_doc = docs[0]
    if isinstance(single_doc, list):
        if not single_doc:
            raise ValueError(
                f"Scenario matrix '{scenario_path}' contains an empty scenario list; "
                "scenario config has no runnable scenarios."
            )
        if any(not isinstance(scenario, Mapping) for scenario in single_doc):
            raise ValueError(f"Scenario matrix '{scenario_path}' list entries must be mappings.")
        # Preserve the legacy manifest path for named or map-referenced entries.  Those entries
        # rely on validation, map-id resolution, and relative-path rebasing even when the YAML
        # document itself is a top-level list.
        if any(
            any(key in scenario for key in ("name", "scenario_id", "map_file", "map_id"))
            for scenario in single_doc
        ):
            scenarios = load_scenarios(scenario_path, base_dir=scenario_path)
            return [dict(s) for s in scenarios]
        return [dict(s) for s in single_doc]
    # A single-document mapping with a ``scenarios:`` list whose entries are all
    # abstract (no name/scenario_id/map_file/map_id) and whose top-level keys
    # carry no manifest-only features (includes, select, overrides, search paths)
    # is an abstract scenario matrix in mapping form.  Return it directly so the
    # map-oriented manifest validator does not emit misleading warnings (#5433).
    if isinstance(single_doc, Mapping) and "scenarios" in single_doc:
        _MANIFEST_ONLY_KEYS = frozenset(
            {
                "includes",
                "include",
                "scenario_files",
                "select_scenarios",
                "scenario_overrides",
                "scenario_overrides_by_name",
                "map_search_paths",
            }
        )
        scenario_entries = single_doc["scenarios"]
        if (
            not _MANIFEST_ONLY_KEYS & single_doc.keys()
            and isinstance(scenario_entries, list)
            and scenario_entries
            and all(isinstance(s, Mapping) for s in scenario_entries)
            and not any(
                any(k in s for k in ("name", "scenario_id", "map_file", "map_id"))
                for s in scenario_entries
            )
        ):
            return [dict(s) for s in scenario_entries]
    # Single-document mapping: defer to include-aware loader for manifests.
    scenarios = load_scenarios(scenario_path, base_dir=scenario_path)
    return [dict(s) for s in scenarios]


def _simple_robot_policy(robot_pos: np.ndarray, goal: np.ndarray, speed: float = 1.0) -> np.ndarray:
    """Return a velocity vector pointing toward goal with capped speed.

    Returns:
        Velocity vector as 2D numpy array.
    """
    vec = goal - robot_pos
    dist = float(np.linalg.norm(vec))
    if dist < 1e-9:
        return np.zeros(2)
    dir_unit = vec / dist
    v = dir_unit * min(speed, dist)
    return v


def _prepare_robot_points(
    robot_start: Sequence[float] | None,
    robot_goal: Sequence[float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert robot start/goal to numpy arrays with defaults.

    Args:
        robot_start: Robot starting position or None for default.
        robot_goal: Robot goal position or None for default.

    Returns:
        Tuple of (start position array, goal position array).
    """
    if robot_start is None:
        rs = np.array([0.3, 3.0], dtype=float)
    else:
        rs = np.asarray(robot_start, dtype=float)
    if robot_goal is None:
        rg = np.array([9.7, 3.0], dtype=float)
    else:
        rg = np.asarray(robot_goal, dtype=float)
    return rs, rg


def _stack_or_zero(
    traj: list[np.ndarray],
    *,
    stack_fn: Callable[[Sequence[np.ndarray]], np.ndarray],
    empty_shape: tuple[int, ...],
) -> np.ndarray:
    """Stack recorded trajectory data or return a zero-length array of known shape.

    Note: To avoid unnecessary memory allocation, `empty_shape` should have zero in the first dimension.

    Returns:
        Stacked trajectory array or empty array with specified shape.
    """
    if traj:
        return stack_fn(traj)
    else:
        # Ensure empty_shape[0] == 0 for lazy evaluation
        assert empty_shape[0] == 0, (
            "empty_shape should have zero in the first dimension for lazy evaluation"
        )
        # Return a zero-length array with the correct shape and dtype
        return np.empty(empty_shape)


def _build_episode_data(  # noqa: PLR0913
    robot_pos_traj: list[np.ndarray],
    robot_vel_traj: list[np.ndarray],
    robot_acc_traj: list[np.ndarray],
    peds_pos_traj: list[np.ndarray],
    ped_forces_traj: list[np.ndarray],
    obstacles: np.ndarray | None,
    goal: np.ndarray,
    dt: float,
    reached_goal_step: int | None,
    robot_radius: float = DEFAULT_BENCHMARK_ROBOT_RADIUS_M,
    ped_radius: float = DEFAULT_BENCHMARK_PED_RADIUS_M,
) -> EpisodeData:
    """Assemble EpisodeData from trajectory buffers and metadata.

    Returns:
        EpisodeData instance.
    """
    robot_pos = _stack_or_zero(robot_pos_traj, stack_fn=np.vstack, empty_shape=(0, 2))
    robot_vel = _stack_or_zero(robot_vel_traj, stack_fn=np.vstack, empty_shape=(0, 2))
    robot_acc = _stack_or_zero(robot_acc_traj, stack_fn=np.vstack, empty_shape=(0, 2))
    peds_pos = _stack_or_zero(peds_pos_traj, stack_fn=np.stack, empty_shape=(0, 0, 2))
    ped_forces = _stack_or_zero(ped_forces_traj, stack_fn=np.stack, empty_shape=(0, 0, 2))

    return EpisodeData(
        robot_pos=robot_pos,
        robot_vel=robot_vel,
        robot_acc=robot_acc,
        peds_pos=peds_pos,
        ped_forces=ped_forces,
        obstacles=obstacles,
        goal=goal,
        dt=dt,
        reached_goal_step=reached_goal_step,
        robot_radius=float(robot_radius),
        ped_radius=float(ped_radius),
    )


def _create_robot_policy(  # noqa: C901, PLR0915
    algo: str,
    algo_config_path: str | None,
    seed: int,
    *,
    robot_radius: float = DEFAULT_BENCHMARK_ROBOT_RADIUS_M,
    ped_radius: float = DEFAULT_BENCHMARK_PED_RADIUS_M,
):
    """Create a robot policy function based on the specified algorithm.

    Returns:
        Tuple of (policy function, metadata dict).
    """

    def _simple_policy_adapter():
        """Create simple policy that navigates directly toward goal.

        Returns:
            Tuple of (policy function, metadata dict).
        """

        def policy(
            robot_pos: np.ndarray,
            _robot_vel: np.ndarray,
            robot_goal: np.ndarray,
            _ped_positions: np.ndarray,
            _dt: float,
        ) -> np.ndarray:
            """Compute velocity command toward goal.

            Args:
                robot_pos: Current robot position.
                _robot_vel: Current robot velocity (unused).
                robot_goal: Target goal position.
                _ped_positions: Pedestrian positions (unused).
                _dt: Timestep duration (unused).

            Returns:
                Velocity command as 2D array.
            """
            return _simple_robot_policy(robot_pos, robot_goal, speed=1.0)

        return policy, {
            "algorithm": "simple_policy",
            "config": {},
            "config_hash": "na",
            "status": "ok",
        }

    if algo in SIMPLE_POLICY_ALIASES:
        policy_fn, metadata = _simple_policy_adapter()
        return (
            policy_fn,
            enrich_algorithm_metadata(
                algo=algo,
                metadata=metadata,
                execution_mode="native",
            ),
        )

    planner, Observation, algo_config = _load_baseline_planner(algo, algo_config_path, seed)

    def _clamp_speed(vel: np.ndarray) -> np.ndarray:
        """Clamp velocity magnitude to maximum allowed speed.

        Args:
            vel: Velocity vector to clamp.

        Returns:
            Clamped velocity vector.
        """
        speed = float(np.linalg.norm(vel))
        if speed > FINAL_SPEED_CLAMP and speed > 1e-9:
            return vel / speed * FINAL_SPEED_CLAMP
        return vel

    def _action_to_velocity(
        action: dict[str, float],
        robot_pos: np.ndarray,
        robot_vel: np.ndarray,
        robot_goal: np.ndarray,
    ) -> np.ndarray:
        """Convert planner action dict to velocity vector.

        Args:
            action: Action dict with either (vx, vy) or (v, omega) keys.
            robot_pos: Current robot position.
            robot_vel: Current robot velocity.
            robot_goal: Target goal position.

        Returns:
            Velocity vector as 2D array.
        """
        if "vx" in action and "vy" in action:
            return _clamp_speed(np.array([action["vx"], action["vy"]], dtype=float))
        if "v" in action and "omega" in action:
            v = action["v"]
            current_speed = np.linalg.norm(robot_vel)
            if current_speed > 1e-6:
                vel = robot_vel / current_speed * v
            else:
                goal_dir = robot_goal - robot_pos
                if np.linalg.norm(goal_dir) > 1e-6:
                    vel = goal_dir / np.linalg.norm(goal_dir) * v
                else:
                    vel = np.zeros(2)
            return _clamp_speed(vel)
        raise ValueError(f"Invalid action format from {algo}: {action}")

    metadata = planner.get_metadata() if hasattr(planner, "get_metadata") else {"algorithm": algo}
    timeout_metadata: dict[str, Any] = {
        "isolation": "process",
        "step_timeout_s": POLICY_STEP_TIMEOUT_SECS,
        "step_timeouts": 0,
        "worker_errors": 0,
        "fallback_actions": 0,
    }
    step_runner: _PlannerStepProcess | None
    try:
        step_runner = _PlannerStepProcess(planner, timeout_s=POLICY_STEP_TIMEOUT_SECS)
    except RuntimeError as exc:
        step_runner = None
        timeout_metadata["isolation"] = "unavailable"
        timeout_metadata["error"] = str(exc)
        metadata["status"] = "policy_step_isolation_unavailable"
        metadata["fallback_reason"] = "policy_step_isolation_unavailable"

    def policy_fn(
        robot_pos: np.ndarray,
        robot_vel: np.ndarray,
        robot_goal: np.ndarray,
        ped_positions: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Policy function that uses the baseline planner.

        Returns:
            Velocity command as 2D array.
        """
        obs = _build_observation(
            Observation,
            robot_pos,
            robot_vel,
            robot_goal,
            ped_positions,
            dt,
            robot_radius=robot_radius,
            ped_radius=ped_radius,
        )

        try:
            if step_runner is None:
                raise RuntimeError("policy step isolation unavailable")
            action = step_runner.step(obs)
        except FuturesTimeoutError:
            timeout_metadata["step_timeouts"] += 1
            timeout_metadata["fallback_actions"] += 1
            metadata["status"] = "policy_step_timeout_fallback"
            metadata["fallback_reason"] = "policy_step_timeout"
            action = {"vx": 0.0, "vy": 0.0}
        except (RuntimeError, TypeError, ValueError) as exc:
            timeout_metadata["worker_errors"] += 1
            timeout_metadata["fallback_actions"] += 1
            metadata["status"] = "policy_step_error_fallback"
            metadata["fallback_reason"] = "policy_step_error"
            timeout_metadata["last_error"] = str(exc)
            logger.warning("Planner step failed unexpectedly: %s", exc)
            action = {"vx": 0.0, "vy": 0.0}

        # Convert action to velocity (handle both action spaces)
        return _action_to_velocity(action, robot_pos, robot_vel, robot_goal)

    if step_runner is not None:
        policy_fn.close = step_runner.close  # type: ignore[attr-defined]
    # Ensure consistent metadata schema
    metadata.setdefault("algorithm", algo)
    metadata["config"] = algo_config
    metadata["config_hash"] = _config_hash(algo_config)
    metadata.setdefault("status", "ok")
    metadata["policy_step_timeout"] = timeout_metadata
    metadata = enrich_algorithm_metadata(
        algo=algo,
        metadata=metadata,
        execution_mode="native",
    )

    return policy_fn, metadata


def _close_robot_policy(policy: Any) -> None:
    """Close optional policy resources after an episode."""
    close_fn = getattr(policy, "close", None)
    if callable(close_fn):
        close_fn()


def _append_video_skip_note(record: dict[str, Any], note: str) -> None:
    """Append a human-readable note to the record's notes field."""
    existing = record.get("notes")
    if existing:
        record["notes"] = f"{existing}; {note}"
    else:
        record["notes"] = note


def _emit_video_skip(
    *,
    record: dict[str, Any],
    episode_id: str,
    scenario_id: str,
    seed: int | None,
    renderer: str,
    reason: str,
    steps: int | None,
    error: str | None = None,
) -> None:
    """Record a video skip reason in logs and episode notes."""
    context = {
        "episode_id": episode_id,
        "scenario_id": scenario_id,
        "seed": seed,
        "renderer": renderer,
        "reason": reason,
        "steps": steps if steps is not None else -1,
    }
    if error:
        context["error"] = error
    try:
        logger.warning(
            (
                "Video skipped: reason={reason} episode_id={episode_id} "
                "scenario_id={scenario_id} seed={seed} renderer={renderer} steps={steps}"
            ),
            **context,
        )
    except (AttributeError, TypeError):
        # Logging failure -> ignore
        pass

    note_parts = [f"video skipped ({renderer}): {reason}"]
    if steps is not None:
        note_parts.append(f"steps={steps}")
    if error:
        note_parts.append(f"error={error}")
    _append_video_skip_note(record, " ".join(note_parts))


def _try_encode_synthetic_video(
    robot_pos_traj: list[np.ndarray],
    *,
    episode_id: str,
    scenario_id: str,
    out_dir: Path,
    fps: int = 10,
    seed: int | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Encode a very lightweight synthetic MP4 from robot positions.

    - Draws a simple red dot for the robot position per frame on a black canvas.
    - Uses moviepy ImageSequenceClip if available; returns None when unavailable.

    Returns:
        Tuple of (video metadata dict or None, error info dict or None).
    """
    if ImageSequenceClip is None:
        _skip_info = {
            "reason": "moviepy-missing",
            "renderer": "synthetic",
            "steps": len(robot_pos_traj),
        }
        return None, _skip_info
    # Successfully imported moviepy; proceed with encoding attempt

    N = len(robot_pos_traj)
    if N == 0:
        _skip_info = {
            "reason": "no-frames",
            "renderer": "synthetic",
            "steps": 0,
        }
        return None, _skip_info
    H, W = 128, 128
    # Determine bounds for simple normalization
    xs = np.array([p[0] for p in robot_pos_traj], dtype=float)
    ys = np.array([p[1] for p in robot_pos_traj], dtype=float)
    min_x, max_x = float(xs.min()), float(xs.max())
    min_y, max_y = float(ys.min()), float(ys.max())
    # Pad 10%
    pad_x = max(1e-6, 0.1 * (max_x - min_x))
    pad_y = max(1e-6, 0.1 * (max_y - min_y))
    min_x -= pad_x
    max_x += pad_x
    min_y -= pad_y
    max_y += pad_y

    def to_px(x: float, y: float) -> tuple[int, int]:
        # Normalize to [0,1] then scale to pixels; y inverted for image coords
        """Map world coordinates to image pixel coordinates.

        Returns:
            Pixel coordinates (x, y).
        """
        nx = 0.0 if max_x == min_x else (x - min_x) / (max_x - min_x)
        ny = 0.0 if max_y == min_y else (y - min_y) / (max_y - min_y)
        px = int(nx * (W - 1))
        py = int((1.0 - ny) * (H - 1))
        return px, py

    frames: list[np.ndarray] = []
    for i in range(N):
        img = np.zeros((H, W, 3), dtype=np.uint8)
        x, y = robot_pos_traj[i]
        px, py = to_px(float(x), float(y))
        # Draw small 3x3 red square centered at (px,py)
        x0, x1 = max(0, px - 1), min(W, px + 2)
        y0, y1 = max(0, py - 1), min(H, py + 2)
        img[y0:y1, x0:x1, :] = np.array([220, 30, 30], dtype=np.uint8)
        frames.append(img)
    mp4_path = out_dir / f"video_{episode_id}.mp4"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        _skip_info = {
            "reason": "unwritable-path",
            "renderer": "synthetic",
            "steps": N,
            "error": str(exc),
        }
        return None, _skip_info
    clip = ImageSequenceClip(frames, fps=fps)  # type: ignore
    try:
        # Keep args minimal for environment compatibility
        clip.write_videofile(str(mp4_path), codec="libx264", fps=fps)
    except OSError as exc:
        _skip_info = {
            "reason": "write-failed",
            "renderer": "synthetic",
            "steps": N,
            "error": str(exc),
        }
        return None, _skip_info
    except (RuntimeError, ValueError) as exc:
        _skip_info = {
            "reason": "encode-failed",
            "renderer": "synthetic",
            "steps": N,
            "error": str(exc),
        }
        return None, _skip_info
    finally:
        try:
            clip.close()  # type: ignore[attr-defined]
        except (AttributeError, OSError):  # pragma: no cover - close best effort
            pass
    try:
        size = mp4_path.stat().st_size
    except (OSError, FileNotFoundError):
        size = 0
    return {
        "status": "success",
        "path": str(mp4_path),
        "format": "mp4",
        "filesize_bytes": int(size),
        "frames": int(N),
        "renderer": "synthetic",
    }, None


def _annotate_and_check_video_perf(
    record: dict[str, Any],
    vid: dict[str, Any],
    perf_start: float,
    enc_start: float,
    enc_end: float,
) -> None:
    """Annotate manifest with encode timing and enforce optional budgets.

    Adds keys: encode_seconds, overhead_ratio, overhead_budget_status,
    overhead_budget_enforced, overhead_soft_threshold, and overhead_hard_threshold.
    Budget env vars:
      - ROBOT_SF_VIDEO_OVERHEAD_SOFT (default 0.10)
      - ROBOT_SF_VIDEO_OVERHEAD_HARD (default 0.50)
      - ROBOT_SF_PERF_ENFORCE (any non-empty to enforce)
      - ROBOT_SF_TEST_OVERRIDE_OVERHEAD_RATIO (for testing only — forces a ratio)
    """
    encode_seconds = float(max(0.0, enc_end - enc_start))
    total_elapsed = float(max(1e-9, enc_end - perf_start))
    override_ratio_raw = os.getenv("ROBOT_SF_TEST_OVERRIDE_OVERHEAD_RATIO")
    if override_ratio_raw:
        overhead_ratio = float(override_ratio_raw)
    else:
        overhead_ratio = float(encode_seconds / total_elapsed)
    vid["encode_seconds"] = encode_seconds
    vid["overhead_ratio"] = overhead_ratio
    record["video"] = vid

    soft = float(os.getenv("ROBOT_SF_VIDEO_OVERHEAD_SOFT", "0.10"))
    hard = float(os.getenv("ROBOT_SF_VIDEO_OVERHEAD_HARD", "0.50"))
    enforce = bool(os.getenv("ROBOT_SF_PERF_ENFORCE"))
    if overhead_ratio > hard:
        budget_status = "hard_breach"
    elif overhead_ratio > soft:
        budget_status = "soft_breach"
    else:
        budget_status = "ok"
    vid["overhead_budget_status"] = budget_status
    vid["overhead_budget_enforced"] = enforce
    vid["overhead_soft_threshold"] = soft
    vid["overhead_hard_threshold"] = hard

    if overhead_ratio > hard:
        if enforce:
            raise RuntimeError(
                f"video overhead hard breach: ratio={overhead_ratio:.3f} > {hard:.3f}",
            )
        else:
            try:
                logger.warning(
                    (
                        "Video overhead hard breach but continue: "
                        "ratio={ratio:.3f} > {hard:.3f} episode_id={episode_id} "
                        "scenario_id={scenario_id} seed={seed} renderer={renderer}"
                    ),
                    ratio=overhead_ratio,
                    hard=hard,
                    episode_id=record.get("episode_id"),
                    scenario_id=record.get("scenario_id"),
                    seed=record.get("seed"),
                    renderer=vid.get("renderer"),
                )
            except (
                ValueError,
                TypeError,
                KeyError,
                AttributeError,
            ):  # pragma: no cover - logging optional
                pass
    elif overhead_ratio > soft:
        if enforce:
            raise RuntimeError(
                f"video overhead soft breach: ratio={overhead_ratio:.3f} > {soft:.3f}",
            )
        else:
            try:
                logger.warning(
                    (
                        "Video overhead soft breach: "
                        "ratio={ratio:.3f} > {soft:.3f} episode_id={episode_id} "
                        "scenario_id={scenario_id} seed={seed} renderer={renderer}"
                    ),
                    ratio=overhead_ratio,
                    soft=soft,
                    episode_id=record.get("episode_id"),
                    scenario_id=record.get("scenario_id"),
                    seed=record.get("seed"),
                    renderer=vid.get("renderer"),
                )
            except (
                ValueError,
                TypeError,
                KeyError,
                AttributeError,
            ):  # pragma: no cover - logging optional
                pass


def _maybe_encode_video(
    *,
    record: dict[str, Any],
    robot_pos_traj: list[np.ndarray],
    videos_dir: str | None,
    video_enabled: bool,
    video_renderer: str,
    perf_start: float,
) -> None:
    """Best-effort video encoding wrapper with perf annotation and budget checks.

    Swallows all exceptions to keep batch robust.
    """
    if not (video_enabled and str(video_renderer) == "synthetic" and videos_dir is not None):
        return
    episode_id = record["episode_id"]
    scenario_id = record["scenario_id"]
    try:
        enc_t0 = time.perf_counter()
        vid, skip_info = _try_encode_synthetic_video(
            robot_pos_traj,
            episode_id=episode_id,
            scenario_id=scenario_id,
            out_dir=Path(videos_dir),
            fps=10,
            seed=record.get("seed"),
        )
        enc_t1 = time.perf_counter()
        if vid is not None and int(vid.get("filesize_bytes", 0)) > 0:
            _annotate_and_check_video_perf(record, vid, perf_start, enc_t0, enc_t1)
        else:
            reason_payload = skip_info or {
                "reason": "encoder-empty",
                "renderer": str(video_renderer),
                "steps": len(robot_pos_traj),
            }
            _emit_video_skip(
                record=record,
                episode_id=episode_id,
                scenario_id=scenario_id,
                seed=record.get("seed"),
                renderer=str(reason_payload.get("renderer", video_renderer)),
                reason=str(reason_payload.get("reason", "unknown")),
                steps=reason_payload.get("steps"),
                error=reason_payload.get("error"),
            )
    except RuntimeError:
        # Budget enforcement: bubble up to runner to record a failure
        raise
    except (TypeError, ValueError, OSError) as exc:  # pragma: no cover - defensive path
        logger.opt(exception=exc).warning(
            "Synthetic video encoding failure for episode_id={} scenario_id={} renderer={}; "
            "continuing benchmark run.",
            episode_id,
            scenario_id,
            video_renderer,
        )


def _simulate_episode_with_policy(
    scenario_params: dict[str, Any],
    seed: int,
    robot_policy: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float], np.ndarray],
    horizon: int,
    dt: float,
    robot_start: Sequence[float] | None,
    robot_goal: Sequence[float] | None,
    record_forces: bool,
) -> tuple[
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[tuple[float, float, float, float]],
    np.ndarray,
    int | None,
]:
    """Run a synthetic episode and return trajectories and metadata.

    Returns:
        Tuple of trajectories, obstacles, goal, and reached-goal step.
    """
    gen = generate_scenario(scenario_params, seed=seed)
    sim = gen.simulator
    if sim is None:
        raise RuntimeError("pysocialforce not available; cannot run episode")
    wrapper = FastPysfWrapper(sim)

    # Determine robot start/goal: defaults if absent
    robot_start_arr, robot_goal_arr = _prepare_robot_points(robot_start, robot_goal)
    robot_pos = robot_start_arr.copy()
    robot_vel = np.zeros(2)

    robot_pos_traj: list[np.ndarray] = []
    robot_vel_traj: list[np.ndarray] = []
    robot_acc_traj: list[np.ndarray] = []
    peds_pos_traj: list[np.ndarray] = []
    ped_forces_traj: list[np.ndarray] = []

    reached_goal_step: int | None = None
    goal_radius = 0.3
    last_vel = robot_vel.copy()

    # Record initial state at t=0 to capture immediate collisions
    initial_ped_pos = sim.peds.pos().copy()
    peds_pos_traj.append(initial_ped_pos)
    if record_forces:
        ped_forces_traj.append(wrapper.get_forces_at_points(initial_ped_pos))
    else:
        ped_forces_traj.append(np.zeros_like(initial_ped_pos, dtype=float))
    robot_pos_traj.append(robot_pos.copy())
    robot_vel_traj.append(robot_vel.copy())
    robot_acc_traj.append(np.zeros_like(robot_vel, dtype=float))

    def _step_robot(
        curr_pos: np.ndarray,
        curr_vel: np.ndarray,
        ped_positions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Advance robot state using the policy output.

        Returns:
            Tuple of (new_pos, new_vel, acceleration).
        """
        new_vel = robot_policy(curr_pos, curr_vel, robot_goal_arr, ped_positions, dt)
        new_pos = curr_pos + new_vel * dt
        acc_vec = (new_vel - curr_vel) / dt
        return new_pos, new_vel, acc_vec

    for t in range(horizon):
        ped_pos = peds_pos_traj[-1]

        robot_pos, robot_vel, acc = _step_robot(robot_pos, last_vel, ped_pos)
        last_vel = robot_vel.copy()

        sim.step()

        ped_pos_next = sim.peds.pos().copy()
        peds_pos_traj.append(ped_pos_next)

        if record_forces:
            ped_forces_traj.append(wrapper.get_forces_at_points(ped_pos_next))
        else:
            ped_forces_traj.append(np.zeros_like(ped_pos_next, dtype=float))

        robot_pos_traj.append(robot_pos.copy())
        robot_vel_traj.append(robot_vel.copy())
        robot_acc_traj.append(acc.copy())

        if reached_goal_step is None and np.linalg.norm(robot_goal_arr - robot_pos) < goal_radius:
            reached_goal_step = t + 1
            break

    return (
        robot_pos_traj,
        robot_vel_traj,
        robot_acc_traj,
        peds_pos_traj,
        ped_forces_traj,
        gen.obstacles,
        np.asarray(robot_goal_arr, dtype=float),
        reached_goal_step,
    )


def _compute_metrics(
    ep: EpisodeData,
    horizon: int,
    snqi_weights: dict[str, float] | None,
    snqi_baseline: dict[str, dict[str, float]] | None,
    experimental_ped_impact: bool = False,
    ped_impact_radius_m: float = 2.0,
    ped_impact_window_steps: int = 5,
) -> dict[str, Any]:
    """Compute metrics and optional SNQI post-processing.

    Returns:
        Metrics dictionary for the episode.
    """
    metrics_raw = compute_all_metrics(
        ep,
        horizon=horizon,
        experimental_ped_impact=experimental_ped_impact,
        ped_impact_radius_m=ped_impact_radius_m,
        ped_impact_window_steps=ped_impact_window_steps,
    )
    return post_process_metrics(
        metrics_raw,
        snqi_weights=snqi_weights,
        snqi_baseline=snqi_baseline,
    )


def _build_episode_record(
    scenario_params: dict[str, Any],
    seed: int,
    metrics: dict[str, Any],
    algo_metadata: dict[str, Any],
    ts_start: str,
    termination_reason: str,
    outcome: dict[str, bool],
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a JSON-serializable episode record.

    Returns:
        Episode record dictionary.
    """
    episode_id = compute_episode_id(scenario_params, seed)
    algo_name = str(scenario_params.get("algo") or algo_metadata.get("algorithm") or "unknown")
    enriched_algo_metadata = enrich_algorithm_metadata(algo=algo_name, metadata=algo_metadata)
    attach_track_metadata(
        enriched_algo_metadata,
        benchmark_track=normalize_track_field(
            scenario_params.get("benchmark_track"), field_name="benchmark_track"
        ),
        track_schema_version=normalize_track_field(
            scenario_params.get("track_schema_version"), field_name="track_schema_version"
        ),
        observation_level=scenario_params.get("observation_level"),
        observation_mode=scenario_params.get("observation_mode"),
    )
    contradictions = outcome_contradictions(
        termination_reason=termination_reason,
        outcome=outcome,
        metrics=metrics,
    )
    if contradictions:
        raise ValueError(
            f"Episode integrity contradictions for scenario '{scenario_params.get('id', 'unknown')}', "
            f"seed={seed}: {'; '.join(contradictions)}"
        )
    record = {
        "version": "v1",
        "episode_id": episode_id,
        "scenario_id": scenario_params.get("id", "unknown"),
        "seed": seed,
        "scenario_params": scenario_params,
        "metrics": metrics,
        "algorithm_metadata": enriched_algo_metadata,
        "config_hash": _config_hash(scenario_params),
        "git_hash": _git_hash_fallback(),
        "timestamps": {"start": ts_start, "end": ts_start},
        "termination_reason": termination_reason,
        "status": status_from_termination_reason(termination_reason),
        "outcome": outcome,
        "integrity": {"contradictions": contradictions},
    }
    if provenance is not None:
        record["provenance"] = provenance
    algo_value = scenario_params.get("algo")
    if algo_value is not None:
        record["algo"] = algo_value
    for key in ("observation_mode", "observation_level", "benchmark_track", "track_schema_version"):
        value = scenario_params.get(key)
        if value is not None:
            record[key] = value
    ensure_metric_parameters(record)
    return record


def run_episode(  # noqa: PLR0913
    scenario_params: dict[str, Any],
    seed: int,
    *,
    horizon: int = 100,
    dt: float = 0.1,
    robot_start: Sequence[float] | None = None,
    robot_goal: Sequence[float] | None = None,
    record_forces: bool = True,
    snqi_weights: dict[str, float] | None = None,
    snqi_baseline: dict[str, dict[str, float]] | None = None,
    algo: str = "simple_policy",
    algo_config_path: str | None = None,
    # Video options (optional)
    video_enabled: bool = False,
    video_renderer: str = "none",
    videos_dir: str | None = None,
    experimental_ped_impact: bool = False,
    ped_impact_radius_m: float = 2.0,
    ped_impact_window_steps: int = 5,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a single episode and return a metrics record dict.

    The robot can use different algorithms based on the 'algo' parameter.

    Returns:
        Episode record dictionary with metrics, trajectories, and metadata.
    """
    # Wall-clock start time for timestamps and perf accounting
    perf_start = time.perf_counter()
    ts_start = datetime.now(UTC).isoformat()
    robot_radius = _scenario_robot_radius_m(scenario_params)
    ped_radius = _scenario_ped_radius_m(scenario_params)
    # Create robot policy based on algorithm
    robot_policy, algo_metadata = _create_robot_policy(
        algo,
        algo_config_path,
        seed,
        robot_radius=robot_radius,
        ped_radius=ped_radius,
    )

    # Simulate episode
    try:
        trajectories = _simulate_episode_with_policy(
            scenario_params,
            seed,
            robot_policy,
            horizon,
            dt,
            robot_start,
            robot_goal,
            record_forces,
        )
    finally:
        _close_robot_policy(robot_policy)
    (
        robot_pos_traj,
        robot_vel_traj,
        robot_acc_traj,
        peds_pos_traj,
        ped_forces_traj,
        obstacle_segments,
        robot_goal_arr,
        reached_goal_step,
    ) = trajectories

    obstacles = sample_obstacle_points(obstacle_segments)

    # Build episode data
    ep = _build_episode_data(
        robot_pos_traj,
        robot_vel_traj,
        robot_acc_traj,
        peds_pos_traj,
        ped_forces_traj,
        obstacles,
        robot_goal_arr,
        dt,
        reached_goal_step,
        robot_radius=robot_radius,
        ped_radius=ped_radius,
    )

    # Compute metrics
    metrics = _compute_metrics(
        ep,
        horizon,
        snqi_weights,
        snqi_baseline,
        experimental_ped_impact=experimental_ped_impact,
        ped_impact_radius_m=ped_impact_radius_m,
        ped_impact_window_steps=ped_impact_window_steps,
    )
    collision = bool(metric_scalar(metrics, "collisions", "collision_rate") > 0.0)
    route_complete = bool(metric_scalar(metrics, "success", "success_rate") > 0.0)
    ended = route_complete or collision
    termination_reason = resolve_termination_reason(
        terminated=ended,
        truncated=False,
        success=route_complete,
        collision=collision,
        reached_max_steps=not ended,
    )
    outcome = build_outcome_payload(
        route_complete=route_complete,
        collision=collision,
        timeout=not ended,
    )

    # Build record
    scenario_params_record = dict(scenario_params)
    scenario_params_record.setdefault("algo", algo)
    record = _build_episode_record(
        scenario_params_record,
        seed,
        metrics,
        algo_metadata,
        ts_start,
        termination_reason,
        outcome,
        provenance=provenance,
    )

    steps_taken = max(0, len(robot_pos_traj) - 1)

    # Handle video
    video_positions = robot_pos_traj[1:] if len(robot_pos_traj) > 1 else []
    _maybe_encode_video(
        record=record,
        robot_pos_traj=video_positions,
        videos_dir=videos_dir,
        video_enabled=video_enabled,
        video_renderer=video_renderer,
        perf_start=perf_start,
    )

    # Update end time
    record["timestamps"]["end"] = datetime.now(UTC).isoformat()
    record["steps"] = steps_taken
    record["horizon"] = horizon
    wall_time = float(max(1e-9, time.perf_counter() - perf_start))
    record["wall_time_sec"] = wall_time
    record["timing"] = {
        "steps_per_second": float(steps_taken) / wall_time if wall_time > 0 else 0.0,
    }
    return record


def validate_and_write(
    record: dict[str, Any],
    schema_path: str | Path,
    out_path: str | Path,
) -> None:
    """Validate an episode record against schema and append to JSONL."""
    schema = load_schema(schema_path)
    violations = validate_episode_success_integrity(record)
    violations.extend(validate_record_event_ledger(record))
    if violations:
        raise ValueError("Episode integrity contradictions detected: " + "; ".join(violations))
    validate_episode(record, schema)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


__all__ = [
    "DEFAULT_CIRCUIT_BREAKER_THRESHOLD",
    "load_scenario_matrix",
    "run_batch",
    "run_episode",
    "validate_and_write",
]


def _expand_jobs(
    scenarios: list[dict[str, Any]],
    base_seed: int = 0,
    repeats_override: int | None = None,
) -> list[tuple[dict[str, Any], int]]:
    """Expand scenarios into (scenario, seed) jobs.

    Returns:
        List of (scenario, seed) tuples.
    """
    jobs: list[tuple[dict[str, Any], int]] = []
    for sc in scenarios:
        reps = int(sc.get("repeats", 1)) if repeats_override is None else int(repeats_override)
        for r in range(reps):
            jobs.append((sc, base_seed + r))
    return jobs


def _run_job_worker(job: tuple[dict[str, Any], int, dict[str, Any]]) -> dict[str, Any]:
    """Top-level worker function to run a single episode.

    Accepts a tuple of (scenario_dict, seed, fixed_params_dict) and returns a record dict.
    This must remain at module top-level for multiprocessing 'spawn' pickling.

    Returns:
        Episode record dictionary.
    """
    sc, seed, params = job
    return run_episode(
        sc,
        seed,
        horizon=int(params["horizon"]),
        dt=float(params["dt"]),
        record_forces=bool(params["record_forces"]),
        snqi_weights=params.get("snqi_weights"),
        snqi_baseline=params.get("snqi_baseline"),
        algo=str(params["algo"]),
        algo_config_path=params.get("algo_config_path"),
        video_enabled=bool(params.get("video_enabled", False)),
        video_renderer=str(params.get("video_renderer", "none")),
        videos_dir=params.get("videos_dir"),
        experimental_ped_impact=bool(params.get("experimental_ped_impact", False)),
        ped_impact_radius_m=float(params.get("ped_impact_radius_m", 2.0)),
        ped_impact_window_steps=int(params.get("ped_impact_window_steps", 5)),
        provenance=params.get("provenance"),
    )


def _write_validated_record(out_path: Path, schema: dict[str, Any], rec: dict[str, Any]) -> None:
    """Validate a record against schema and append to JSONL."""
    violations = validate_episode_success_integrity(rec)
    violations.extend(validate_record_event_ledger(rec))
    if violations:
        raise ValueError("Episode integrity contradictions detected: " + "; ".join(violations))
    validate_episode(rec, schema)
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")


_error_signature = error_signature


def _run_batch_sequential(  # noqa: C901, D417
    jobs: list[tuple[dict[str, Any], int]],
    *,
    out_path: Path,
    schema: dict[str, Any],
    fixed_params: dict[str, Any],
    progress_cb: Callable[[int, int, dict[str, Any], int, bool, str | None], None] | None,
    fail_fast: bool,
    circuit_breaker_threshold: int | None = None,
) -> tuple[int, list[dict[str, Any]], dict[str, Any] | None]:
    """Run batch jobs sequentially and return write count + failures + abort metadata.

    Circuit breaker: when ``circuit_breaker_threshold`` consecutive episode failures
    share the same error signature (exception type + normalised message prefix), the
    loop aborts early with status ``aborted_systematic_failure``.  A success or a
    failure with a different signature resets the counter.

    Args:
        circuit_breaker_threshold: Consecutive identical failures before abort.

    Returns:
        Tuple of (written_count, failure_records, abort_metadata_or_None).
    """
    circuit_breaker_threshold = normalize_circuit_breaker_threshold(circuit_breaker_threshold)
    wrote = 0
    failures: list[dict[str, Any]] = []
    total = len(jobs)
    abort_metadata: dict[str, Any] | None = None

    cb_tracker: dict[str, Any] = {
        "signature": None,
        "consecutive": 0,
        "first_fail_idx": None,
    }

    for idx, (sc, seed) in enumerate(jobs, start=1):
        try:
            rec = _run_job_worker((sc, seed, fixed_params))
            _write_validated_record(out_path, schema, rec)
            wrote += 1
            # Reset circuit breaker on success
            cb_tracker["signature"] = None
            cb_tracker["consecutive"] = 0
            cb_tracker["first_fail_idx"] = None
            if progress_cb is not None:
                try:
                    progress_cb(idx, total, sc, seed, True, None)
                except Exception:  # pragma: no cover - progress best-effort
                    pass
        except Exception as e:  # pragma: no cover - error path
            logger.exception(
                "Benchmark batch job failed in serial execution: scenario_id={} seed={}",
                sc.get("id", "unknown"),
                seed,
            )
            failures.append(
                {
                    "scenario_id": sc.get("id", "unknown"),
                    "seed": seed,
                    "error": repr(e),
                },
            )
            if progress_cb is not None:
                try:
                    progress_cb(idx, total, sc, seed, False, repr(e))
                except Exception:  # pragma: no cover
                    pass

            # Circuit breaker logic
            if (
                abort_metadata is None
                and circuit_breaker_threshold is not None
                and circuit_breaker_threshold > 0
            ):
                sig = _error_signature(e)
                if cb_tracker["signature"] is None:
                    cb_tracker["signature"] = sig
                    cb_tracker["consecutive"] = 1
                    cb_tracker["first_fail_idx"] = idx
                elif sig == cb_tracker["signature"]:
                    cb_tracker["consecutive"] += 1
                else:
                    # Different signature resets the streak
                    cb_tracker["signature"] = sig
                    cb_tracker["consecutive"] = 1
                    cb_tracker["first_fail_idx"] = idx

                if cb_tracker["consecutive"] >= circuit_breaker_threshold:
                    projected_remaining = total - idx
                    abort_metadata = build_abort_metadata(
                        signature=sig,
                        consecutive_failures=cb_tracker["consecutive"],
                        first_fail_index=cb_tracker["first_fail_idx"],
                        episodes_completed_before_onset=wrote,
                        total_jobs=total,
                    )
                    logger.warning(
                        "Circuit breaker tripped: %d consecutive identical failures "
                        "(%s). Aborting arm after %d/%d jobs. Projected episodes saved: %d.",
                        cb_tracker["consecutive"],
                        sig[0],
                        idx,
                        total,
                        projected_remaining,
                    )
                    # Stop processing further jobs
                    break

            if fail_fast:
                raise

    return wrote, failures, abort_metadata


def _run_batch_parallel(  # noqa: C901
    jobs: list[tuple[dict[str, Any], int]],
    *,
    out_path: Path,
    schema: dict[str, Any],
    fixed_params: dict[str, Any],
    workers: int,
    progress_cb: Callable[[int, int, dict[str, Any], int, bool, str | None], None] | None,
    fail_fast: bool,
) -> tuple[int, list[dict[str, Any]], dict[str, Any] | None]:
    """Run batch jobs in parallel and return write count + failures + abort metadata.

    Circuit breaker is not available for parallel execution because jobs complete
    concurrently.  ``abort_metadata`` is always None.

    Returns:
        Tuple of (written_count, failure_records, abort_metadata_or_None).
    """
    wrote = 0
    failures: list[dict[str, Any]] = []
    total = len(jobs)
    results_by_idx: dict[int, dict[str, Any]] = {}
    # Submit all jobs
    with ProcessPoolExecutor(max_workers=int(workers)) as ex:
        future_to_job: dict[Any, tuple[int, dict[str, Any], int]] = {}
        for idx, (sc, seed) in enumerate(jobs, start=1):
            fut = ex.submit(_run_job_worker, (sc, seed, fixed_params))
            future_to_job[fut] = (idx, sc, seed)
        for fut in as_completed(future_to_job):
            idx, sc, seed = future_to_job[fut]
            try:
                results_by_idx[idx] = fut.result()
                if progress_cb is not None:
                    try:
                        progress_cb(idx, total, sc, seed, True, None)
                    except Exception:  # pragma: no cover
                        pass
            except Exception as e:  # pragma: no cover
                logger.exception(
                    "Benchmark batch job failed in parallel execution: scenario_id={} seed={}",
                    sc.get("id", "unknown"),
                    seed,
                )
                failures.append(
                    {
                        "scenario_id": sc.get("id", "unknown"),
                        "seed": seed,
                        "error": repr(e),
                    },
                )
                if progress_cb is not None:
                    try:
                        progress_cb(idx, total, sc, seed, False, repr(e))
                    except Exception:  # pragma: no cover
                        pass
                if fail_fast:
                    for f in future_to_job:
                        f.cancel()
                    raise
    for idx in sorted(results_by_idx):
        try:
            _write_validated_record(out_path, schema, results_by_idx[idx])
            wrote += 1
        except (
            OSError,
            ValueError,
            TypeError,
            KeyError,
            jsonschema.ValidationError,
        ) as e:  # pragma: no cover - write/validate path
            logger.exception(
                "Benchmark batch write/validation failed for scenario_id={} seed={}",
                results_by_idx[idx].get("scenario_id", "unknown"),
                results_by_idx[idx].get("seed", -1),
            )
            failures.append(
                {
                    "scenario_id": results_by_idx[idx].get("scenario", {}).get("id", "unknown"),
                    "seed": results_by_idx[idx].get("seed", -1),
                    "error": repr(e),
                },
            )
            if fail_fast:
                raise
    return wrote, failures, None


def _prepare_batch_setup(
    scenarios_or_path: list[dict[str, Any]] | str | Path,
    out_path: str | Path,
    schema_path: str | Path,
    append: bool,
) -> tuple[list[dict[str, Any]], Path, dict[str, Any]]:
    """Prepare scenarios, output path, and schema for batch processing.

    Returns:
        Tuple of (scenarios list, output path, schema dict).
    """
    # Load scenarios
    scenarios_is_path = isinstance(scenarios_or_path, str | Path)
    if scenarios_is_path:
        scenarios = load_scenario_matrix(cast("str | Path", scenarios_or_path))
    else:
        # scenarios_or_path is already list[dict[str, Any]]
        scenarios = cast("list[dict[str, Any]]", scenarios_or_path)

    # Prepare output
    out_path = Path(out_path)
    if not append and out_path.exists():
        out_path.unlink()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    schema = load_schema(schema_path)
    return scenarios, out_path, schema


def _setup_fixed_params(  # noqa: PLR0913
    out_path: Path,
    horizon: int,
    dt: float,
    record_forces: bool,
    snqi_weights: dict[str, float] | None,
    snqi_baseline: dict[str, dict[str, float]] | None,
    video_enabled: bool,
    video_renderer: str,
    algo: str,
    algo_config_path: str | None,
    experimental_ped_impact: bool = False,
    ped_impact_radius_m: float = 2.0,
    ped_impact_window_steps: int = 5,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Set up the fixed parameters dict for job execution.

    Returns:
        Dictionary with fixed parameters for all episodes.
    """
    videos_dir = (out_path.parent / "videos").as_posix()
    return {
        "horizon": horizon,
        "dt": dt,
        "record_forces": record_forces,
        "snqi_weights": snqi_weights,
        "snqi_baseline": snqi_baseline,
        "algo": algo,
        "algo_config_path": algo_config_path,
        "video_enabled": bool(video_enabled) and str(video_renderer) != "none",
        "video_renderer": str(video_renderer),
        "videos_dir": videos_dir,
        "experimental_ped_impact": bool(experimental_ped_impact),
        "ped_impact_radius_m": float(ped_impact_radius_m),
        "ped_impact_window_steps": int(ped_impact_window_steps),
        "provenance": provenance,
    }


def _filter_resume_jobs(
    jobs: list[tuple[dict[str, Any], int]],
    out_path: Path,
    resume: bool,
) -> list[tuple[dict[str, Any], int]]:
    """Filter jobs based on resume logic, skipping existing episodes.

    Returns:
        Filtered list of (scenario dict, seed) tuples excluding completed episodes.
    """
    if not resume or not out_path.exists():
        return jobs

    # Try fast-path via manifest; fall back to scanning JSONL if stale/missing
    existing_ids = load_manifest(
        out_path,
        expected_identity_hash=_episode_identity_hash(),
        expected_schema_version=EPISODE_SCHEMA_VERSION,
    ) or index_existing(out_path)

    if not existing_ids:
        return jobs

    filtered: list[tuple[dict[str, Any], int]] = []
    for sc, seed in jobs:
        eid = compute_episode_id(sc, seed)
        if eid not in existing_ids:
            filtered.append((sc, seed))
    return filtered


def _run_jobs(
    jobs: list[tuple[dict[str, Any], int]],
    out_path: Path,
    schema: dict[str, Any],
    fixed_params: dict[str, Any],
    workers: int,
    progress_cb: Callable[[int, int, dict[str, Any], int, bool, str | None], None] | None,
    fail_fast: bool,
    circuit_breaker_threshold: int | None,
) -> tuple[int, list[dict[str, Any]], dict[str, Any] | None]:
    """Execute jobs using sequential or parallel processing.

    Returns:
        Tuple of (written_count, failure_records, abort_metadata_or_None).
    """
    if workers <= 1:
        return _run_batch_sequential(
            jobs,
            out_path=out_path,
            schema=schema,
            fixed_params=fixed_params,
            progress_cb=progress_cb,
            fail_fast=fail_fast,
            circuit_breaker_threshold=circuit_breaker_threshold,
        )
    else:
        return _run_batch_parallel(
            jobs,
            out_path=out_path,
            schema=schema,
            fixed_params=fixed_params,
            workers=workers,
            progress_cb=progress_cb,
            fail_fast=fail_fast,
        )


def _finalize_batch(
    out_path: Path,
    wrote: int,
    resume: bool,
    provenance_input_paths: list[Path] | None = None,
) -> dict[str, Any]:
    """Finalize batch processing: save manifest and optional performance snapshot.

    Returns:
        Summary dictionary with episode counts and outcome statistics.
    """
    # Save/update manifest to speed up future resume if we wrote anything
    if resume and wrote > 0 and out_path.exists():
        # Re-index by scanning (cheap) to ensure we capture exactly what's on disk
        save_manifest(
            out_path,
            index_existing(out_path),
            identity_hash=_episode_identity_hash(),
            schema_version=EPISODE_SCHEMA_VERSION,
            input_paths=provenance_input_paths,
        )

    # Optional: write a small performance snapshot for video encoding if requested
    try:
        if os.getenv("ROBOT_SF_VIDEO_PERF_SNAPSHOT"):
            vids: list[dict] = []
            try:
                with out_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        rec = json.loads(line)
                        v = rec.get("video") if isinstance(rec, dict) else None
                        if isinstance(v, dict):
                            vids.append(v)
            except FileNotFoundError:
                vids = []
            total_frames = sum(int(v.get("frames", 0)) for v in vids)
            total_encode = sum(float(v.get("encode_seconds", 0.0)) for v in vids)
            overheads = [float(v.get("overhead_ratio", 0.0)) for v in vids if "overhead_ratio" in v]
            snap = {
                "episodes": len(vids),
                "total_frames": int(total_frames),
                "total_encode_seconds": float(total_encode),
                "encode_ms_per_frame": (1000.0 * total_encode / total_frames)
                if total_frames > 0
                else None,
                "mean_overhead_ratio": (
                    float(sum(overheads) / len(overheads)) if overheads else None
                ),
                "environment": {
                    "os": platform.platform(),
                    "python": platform.python_version(),
                    "processor": platform.processor(),
                },
            }
            perf_path = out_path.parent / "videos" / "perf_snapshot.json"
            perf_path.parent.mkdir(parents=True, exist_ok=True)
            with perf_path.open("w", encoding="utf-8") as f:
                json.dump(snap, f, indent=2)
    except (OSError, ValueError, TypeError, KeyError):
        # Best-effort; ignore snapshot errors
        pass

    return {
        "total_jobs": 0,  # Will be set by caller
        "written": wrote,
        "failures": [],  # Will be set by caller
        "out_path": str(out_path),
    }


def run_batch(  # noqa: PLR0913
    scenarios_or_path: list[dict[str, Any]] | str | Path,
    out_path: str | Path,
    schema_path: str | Path,
    *,
    base_seed: int = 0,
    repeats_override: int | None = None,
    horizon: int = 100,
    dt: float = 0.1,
    record_forces: bool = True,
    snqi_weights: dict[str, float] | None = None,
    snqi_baseline: dict[str, dict[str, float]] | None = None,
    video_enabled: bool = False,
    video_renderer: str = "none",
    append: bool = True,
    fail_fast: bool = False,
    progress_cb: Callable[[int, int, dict[str, Any], int, bool, str | None], None] | None = None,
    algo: str = "simple_policy",
    algo_config_path: str | None = None,
    benchmark_profile: str = "baseline-safe",
    socnav_missing_prereq_policy: str = "fail-fast",
    adapter_impact_eval: bool = False,
    experimental_ped_impact: bool = False,
    ped_impact_radius_m: float = 2.0,
    ped_impact_window_steps: int = 5,
    observation_mode: str | None = None,
    observation_level: str | None = None,
    benchmark_track: str | None = None,
    track_schema_version: str | None = None,
    observation_noise: dict[str, Any] | None = None,
    synthetic_actuation_profile: dict[str, Any] | None = None,
    latency_stress_profile: dict[str, Any] | None = None,
    record_planner_decision_trace: bool = False,
    record_simulation_step_trace: bool = False,
    workers: int = 1,
    resume: bool = True,
    circuit_breaker_threshold: int | None = None,
) -> dict[str, Any]:
    """Run a batch of episodes and write JSONL records.

    scenarios_or_path: either a list of scenario dicts or a YAML file path.
    Returns a summary dict with counts and failures.

    Returns:
        Summary dictionary with episode counts, failures, and execution metadata.
    """
    circuit_breaker_threshold = normalize_circuit_breaker_threshold(circuit_breaker_threshold)

    # Prepare batch setup
    scenarios, out_path, schema = _prepare_batch_setup(
        scenarios_or_path,
        out_path,
        schema_path,
        append,
    )
    benchmark_track = normalize_track_field(benchmark_track, field_name="benchmark_track")
    track_schema_version = normalize_track_field(
        track_schema_version,
        field_name="track_schema_version",
    )
    scenarios = _apply_track_metadata_to_scenarios(
        scenarios,
        observation_mode=observation_mode,
        observation_level=observation_level,
        benchmark_track=benchmark_track,
        track_schema_version=track_schema_version,
    )

    # Map-based scenario detection: delegate to map runner
    if scenarios and any("map_file" in sc or "simulation_config" in sc for sc in scenarios):
        return run_map_batch(
            scenarios_or_path,
            out_path,
            schema_path,
            horizon=horizon if horizon > 0 else None,
            dt=dt if dt > 0 else None,
            record_forces=record_forces,
            snqi_weights=snqi_weights,
            snqi_baseline=snqi_baseline,
            algo=algo,
            algo_config_path=algo_config_path,
            benchmark_profile=benchmark_profile,
            socnav_missing_prereq_policy=socnav_missing_prereq_policy,
            adapter_impact_eval=adapter_impact_eval,
            experimental_ped_impact=experimental_ped_impact,
            ped_impact_radius_m=ped_impact_radius_m,
            ped_impact_window_steps=ped_impact_window_steps,
            observation_mode=observation_mode,
            observation_level=observation_level,
            benchmark_track=benchmark_track,
            track_schema_version=track_schema_version,
            observation_noise=observation_noise,
            synthetic_actuation_profile=synthetic_actuation_profile,
            latency_stress_profile=latency_stress_profile,
            circuit_breaker_threshold=circuit_breaker_threshold,
            record_planner_decision_trace=record_planner_decision_trace,
            record_simulation_step_trace=record_simulation_step_trace,
            workers=workers,
            resume=resume,
        )

    # Expand jobs
    jobs = _expand_jobs(scenarios, base_seed=base_seed, repeats_override=repeats_override)

    # Set up fixed parameters
    from robot_sf.benchmark.release_protocol import BENCHMARK_PROTOCOL_VERSION  # noqa: PLC0415

    provenance = {
        "protocol_version": BENCHMARK_PROTOCOL_VERSION,
        "commit_hash": _git_hash_fallback(),
        "base_seed": base_seed,
        "run_id": uuid.uuid4().hex,
        "python_version": platform.python_version(),
        "config_identity": {
            "schema_path": str(schema_path),
            "algo": str(algo),
            "algo_config_path": str(algo_config_path) if algo_config_path is not None else None,
            "scenario_count": len(scenarios),
            "scenario_matrix_hash": _config_hash(scenarios),
        },
    }
    if hasattr(sys, "argv") and sys.argv:
        provenance["invocation"] = shlex.join(sys.argv)

    fixed_params = _setup_fixed_params(
        out_path,
        horizon,
        dt,
        record_forces,
        snqi_weights,
        snqi_baseline,
        video_enabled,
        video_renderer,
        algo,
        algo_config_path,
        experimental_ped_impact,
        ped_impact_radius_m,
        ped_impact_window_steps,
        provenance=provenance,
    )

    # Filter jobs for resume
    jobs = _filter_resume_jobs(jobs, out_path, resume)

    # Run jobs
    wrote, failures, abort_metadata = _run_jobs(
        jobs,
        out_path,
        schema,
        fixed_params,
        workers,
        progress_cb,
        fail_fast,
        circuit_breaker_threshold,
    )

    # Finalize and return summary
    provenance_input_paths = [Path(schema_path)]
    if isinstance(scenarios_or_path, str | Path):
        provenance_input_paths.append(Path(scenarios_or_path))
    summary = _finalize_batch(
        out_path,
        wrote,
        resume,
        provenance_input_paths=provenance_input_paths,
    )
    summary["total_jobs"] = len(jobs)
    summary["failures"] = failures
    if abort_metadata is not None:
        summary["abort"] = abort_metadata
    if benchmark_track is not None:
        summary["benchmark_track"] = benchmark_track
    if track_schema_version is not None:
        summary["track_schema_version"] = track_schema_version
    return summary
