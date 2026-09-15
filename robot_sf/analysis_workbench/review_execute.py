"""Bounded scenario-experiment executor for review bundles (SREV-22, issue #9293).

This module owns the ``srev22-review-execute`` component surface: it consumes
the versioned SREV-01 contracts (``component-request.v1`` / ``component-result.v1`` /
``experiment-recipe.v1``), validates the execution config (never arbitrary code),
and runs isolated CPU control/treatment episodes for finite candidate
interventions from a recipe.

Execution reuses the canonical owners for computation and never replaces them:

* scenario construction and stepping use the real simulator path
  (:mod:`robot_sf.sim.simulator` with ``SinglePedestrianDefinition`` speed /
  start-delay semantics) driven by the stateless goal-directed holonomic policy
  (the ``simple_policy`` planner family from :mod:`robot_sf.benchmark.runner`);
* pair verdicts (survived / falsified / inconclusive) use the canonical
  :func:`robot_sf.benchmark.counterfactual_pair.evaluate_counterfactual_pair`;
* the pair-manifest tool (``scripts/tools/create_counterfactual_scenario_pair.py``)
  remains manifest-only and is never treated as execution.

Evidence boundary: fixture/diagnostic smoke proof only. Outputs are isolated
control/treatment telemetry, measured activation traces, an attempt ledger, and
preserved receipts. Nothing here is campaign or evidence-admission authority,
and no benchmark, planner, or simulator semantics are changed.

Known fixture-path limitation (observed, not worked around): a
``single_pedestrian_start_delay_offset`` intervention holds the pedestrian but
the canonical release path leaves ``max_speeds`` at zero, so the delayed
pedestrian never moves. Such candidates resolve to ``unavailable`` with reason
``intervention_not_executable`` instead of synthesizing motion. Simulator
behavior itself is out of scope for this leaf.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from robot_sf.analysis_workbench.review_contracts import (
    COMPONENT_REQUEST_SCHEMA_VERSION,
    COMPONENT_RESULT_SCHEMA_VERSION,
    ComponentRequest,
    ComponentResult,
    ReviewContractsValidationError,
    component_request_from_dict,
    component_result_from_dict,
    experiment_recipe_from_dict,
)
from robot_sf.benchmark.counterfactual_pair import (
    PairHypothesis,
    evaluate_counterfactual_pair,
)
from robot_sf.errors import RobotSfError

COMPONENT_ID = "srev22-review-execute"
COMPONENT_VERSION = "1.0.0"
COMPONENT_DESCRIPTOR_SCHEMA_VERSION = "component-descriptor.v1"

EXECUTE_REPORT_SCHEMA_VERSION = "execute-report.v1"
ATTEMPT_LEDGER_SCHEMA_VERSION = "attempt-ledger.v1"
ACTIVATION_TRACE_SCHEMA_VERSION = "activation-trace.v1"
PRESERVATION_MANIFEST_SCHEMA_VERSION = "preservation-manifest.v1"

SUPPORTED_INPUT_VERSIONS = (COMPONENT_REQUEST_SCHEMA_VERSION,)
REQUIRED_CAPABILITIES = ("bounded-execution",)
OUTPUT_TYPES = (
    EXECUTE_REPORT_SCHEMA_VERSION,
    ATTEMPT_LEDGER_SCHEMA_VERSION,
    ACTIVATION_TRACE_SCHEMA_VERSION,
    PRESERVATION_MANIFEST_SCHEMA_VERSION,
)

SUPPORTED_PLANNERS = ("simple_policy",)
SUPPORTED_FACTORS = (
    "single_pedestrian_speed_offset",
    "single_pedestrian_start_delay_offset",
)
EXECUTABLE_FACTORS = ("single_pedestrian_speed_offset",)
SUPPORTED_MEASUREMENTS = (
    "min_robot_ped_distance_m",
    "ped_mean_speed_m_s",
    "robot_goal_reached",
    "ped_motion_onset_step",
)

_DT_S = 0.1
_ROBOT_GOAL = (16.8, 16.8)
_ROBOT_SPAWN = (1.2, 1.2)
_PED_START = (10.0, 1.0)
_PED_GOAL = (10.0, 19.0)
_GOAL_REACHED_RADIUS_M = 0.5


class ReviewExecuteError(RobotSfError, ValueError):
    """Raised when a review-execute request, config, or recipe is unusable."""

    def __init__(self, errors: list[str], *, source: str | Path | None = None):
        """Build an actionable execution-contract error."""
        self.errors = tuple(errors)
        self.source = str(source) if source is not None else None
        prefix = f"{self.source}: " if self.source else ""
        super().__init__(prefix + "; ".join(errors))


@dataclass(frozen=True, slots=True)
class ExecuteConfig:
    """Validated execution config (closed allowlist, never arbitrary code)."""

    planner: str = "simple_policy"
    seed: int = 7
    horizon_steps: int = 60
    robot_speed_m_s: float = 1.0
    max_candidates: int = 3
    max_executions: int = 6
    wall_timeout_s: float = 600.0
    per_execution_timeout_s: float = 120.0
    activation_speed_tolerance_m_s: float = 0.05
    motion_epsilon_m: float = 0.05
    required_component_version: str | None = None
    intervention_parameters: dict[str, Any] = field(default_factory=dict)
    recipe: dict[str, Any] = field(default_factory=dict)


def descriptor() -> dict[str, Any]:
    """Describe the review-execute component and its capability contract.

    Returns:
        Component descriptor document honoring ``component-descriptor.v1``.
    """
    return {
        "schema_version": COMPONENT_DESCRIPTOR_SCHEMA_VERSION,
        "component_id": COMPONENT_ID,
        "component_version": COMPONENT_VERSION,
        "supported_input_versions": list(SUPPORTED_INPUT_VERSIONS),
        "required_capabilities": list(REQUIRED_CAPABILITIES),
        "optional_capabilities": [],
        "output_types": list(OUTPUT_TYPES),
    }


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def validate_execute_config(raw: Any, *, source: Any = None) -> ExecuteConfig:
    """Validate raw execution config against the closed allowlist.

    Args:
        raw: Raw config mapping from the component request.
        source: Optional source label for error messages.

    Returns:
        Validated execution config with defaults applied.

    Raises:
        ReviewExecuteError: For unknown keys, bad types, or out-of-range values.
    """
    if not isinstance(raw, dict):
        raise ReviewExecuteError(["config must be a mapping"], source=source)
    allowed = {
        "planner",
        "seed",
        "horizon_steps",
        "robot_speed_m_s",
        "max_candidates",
        "max_executions",
        "wall_timeout_s",
        "per_execution_timeout_s",
        "activation_speed_tolerance_m_s",
        "motion_epsilon_m",
        "required_component_version",
        "intervention_parameters",
        "recipe",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ReviewExecuteError(
            [f"unknown config keys are rejected: {', '.join(unknown)}"], source=source
        )
    if "recipe" not in raw or not isinstance(raw["recipe"], dict):
        raise ReviewExecuteError(
            ["config.recipe must be an experiment-recipe mapping"], source=source
        )
    errors: list[str] = []
    planner = raw.get("planner", "simple_policy")
    if planner not in SUPPORTED_PLANNERS:
        errors.append(f"unsupported planner: {planner!r}; supported: {list(SUPPORTED_PLANNERS)}")
    seed = _check_int_field(raw, "seed", 7, minimum=0, maximum=None, errors=errors)
    horizon = _check_int_field(raw, "horizon_steps", 60, minimum=1, maximum=600, errors=errors)
    robot_speed = _check_float_field(
        raw, "robot_speed_m_s", 1.0, minimum=0.1, maximum=2.0, errors=errors
    )
    max_candidates = _check_int_field(
        raw, "max_candidates", 3, minimum=1, maximum=None, errors=errors
    )
    max_executions = _check_int_field(
        raw, "max_executions", 6, minimum=1, maximum=None, errors=errors
    )
    wall_timeout = _check_float_field(
        raw, "wall_timeout_s", 600.0, minimum=0.0, maximum=None, errors=errors
    )
    if wall_timeout <= 0.0:
        errors.append("wall_timeout_s must be positive")
    per_execution_timeout = _check_float_field(
        raw, "per_execution_timeout_s", 120.0, minimum=0.0, maximum=None, errors=errors
    )
    if per_execution_timeout <= 0.0:
        errors.append("per_execution_timeout_s must be positive")
    speed_tol = _check_float_field(
        raw, "activation_speed_tolerance_m_s", 0.05, minimum=0.0, maximum=None, errors=errors
    )
    motion_eps = _check_float_field(
        raw, "motion_epsilon_m", 0.05, minimum=0.0, maximum=None, errors=errors
    )
    required_version = raw.get("required_component_version")
    if required_version is not None and not isinstance(required_version, str):
        errors.append("required_component_version must be a string")
    intervention_parameters = raw.get("intervention_parameters", {})
    if not isinstance(intervention_parameters, dict):
        errors.append("intervention_parameters must be a mapping")
    if errors:
        raise ReviewExecuteError(errors, source=source)
    return ExecuteConfig(
        planner=str(planner),
        seed=int(seed),
        horizon_steps=int(horizon),
        robot_speed_m_s=float(robot_speed),
        max_candidates=int(max_candidates),
        max_executions=int(max_executions),
        wall_timeout_s=float(wall_timeout),
        per_execution_timeout_s=float(per_execution_timeout),
        activation_speed_tolerance_m_s=float(speed_tol),
        motion_epsilon_m=float(motion_eps),
        required_component_version=required_version,
        intervention_parameters=dict(intervention_parameters),
        recipe=dict(raw["recipe"]),
    )


def _check_int_field(
    raw: dict[str, Any],
    key: str,
    default: int,
    *,
    minimum: int,
    maximum: int | None,
    errors: list[str],
) -> int:
    """Validate one integer config field, recording errors instead of raising.

    Returns:
        Validated integer value, or the default when invalid.
    """
    value = raw.get(key, default)
    if not _is_int(value):
        errors.append(f"{key} must be an integer")
        return default
    if value < minimum or (maximum is not None and value > maximum):
        bound = f"{minimum}.." if maximum is None else f"{minimum}..{maximum}"
        errors.append(f"{key} must be within {bound}")
        return default
    return int(value)


def _check_float_field(
    raw: dict[str, Any],
    key: str,
    default: float,
    *,
    minimum: float,
    maximum: float | None,
    errors: list[str],
) -> float:
    """Validate one numeric config field, recording errors instead of raising.

    Returns:
        Validated float value, or the default when invalid.
    """
    value = raw.get(key, default)
    if not _is_finite_number(value):
        errors.append(f"{key} must be a finite number")
        return default
    numeric = float(value)
    if numeric < minimum or (maximum is not None and numeric > maximum):
        bound = f">= {minimum}" if maximum is None else f"within {minimum}..{maximum}"
        errors.append(f"{key} must be {bound}")
        return default
    return numeric


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _repo_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    digest = completed.stdout.strip()
    return digest if digest else "unknown"


def _unsupported_capabilities(request: ComponentRequest) -> list[str]:
    supported = set(REQUIRED_CAPABILITIES)
    return [name for name in request.required_capabilities if name not in supported]


def _execute_episode_job(job: dict[str, Any]) -> dict[str, Any]:
    """Run one control/treatment episode inside an owned child process.

    Args:
        job: Plain-data episode spec (seed, horizon, speeds, delays).

    Returns:
        Plain-data telemetry payload or an error payload; never raises.
    """
    try:
        import numpy as np  # noqa: PLC0415 - lazy: keep module import light

        from robot_sf.common.seed import set_global_seed  # noqa: PLC0415 - lazy: child-process sim stack
        from robot_sf.gym_env.unified_config import RobotSimulationConfig  # noqa: PLC0415 - lazy: child-process sim stack
        from robot_sf.nav.global_route import GlobalRoute  # noqa: PLC0415 - lazy: child-process sim stack
        from robot_sf.nav.map_config import (  # noqa: PLC0415 - lazy: child-process sim stack
            MapDefinition,
            MapDefinitionPool,
            SinglePedestrianDefinition,
        )
        from robot_sf.robot.holonomic_drive import HolonomicDriveSettings  # noqa: PLC0415 - lazy: child-process sim stack
        from robot_sf.sim.sim_config import SimulationSettings  # noqa: PLC0415 - lazy: child-process sim stack
        from robot_sf.sim.simulator import init_simulators  # noqa: PLC0415 - lazy: child-process sim stack

        set_global_seed(int(job["seed"]))
        horizon = int(job["horizon_steps"])
        robot_speed = float(job["robot_speed_m_s"])
        width, height = 20.0, 20.0
        spawn_zone = ((1.0, 1.0), (2.0, 1.0), (1.0, 2.0))
        goal_zone = ((16.0, 16.0), (17.0, 16.0), (16.0, 17.0))
        bounds = [
            ((0.0, 0.0), (width, 0.0)),
            ((width, 0.0), (width, height)),
            ((width, height), (0.0, height)),
            ((0.0, height), (0.0, 0.0)),
        ]
        route = GlobalRoute(
            spawn_id=0,
            goal_id=0,
            waypoints=[_ROBOT_SPAWN, _ROBOT_GOAL],
            spawn_zone=spawn_zone,
            goal_zone=goal_zone,
        )
        pedestrian = SinglePedestrianDefinition(
            id="ped-0",
            start=_PED_START,
            goal=_PED_GOAL,
            speed_m_s=float(job["ped_speed_m_s"]),
            start_delay_s=float(job.get("ped_start_delay_s", 0.0)),
        )
        map_def = MapDefinition(
            width=width,
            height=height,
            obstacles=[],
            robot_spawn_zones=[spawn_zone],
            ped_spawn_zones=[spawn_zone],
            robot_goal_zones=[goal_zone],
            bounds=bounds,
            robot_routes=[route],
            ped_goal_zones=[goal_zone],
            ped_crowded_zones=[],
            ped_routes=[route],
            single_pedestrians=[pedestrian],
        )
        env_config = RobotSimulationConfig(
            map_pool=MapDefinitionPool(map_defs={"srev22-tiny-crossing": map_def}),
            sim_config=SimulationSettings(
                difficulty=0,
                ped_density_by_difficulty=[0.0],
                population_size=1,
            ),
            robot_config=HolonomicDriveSettings(),
        )
        simulator = init_simulators(
            env_config,
            map_def,
            num_robots=1,
            random_start_pos=False,
            peds_have_obstacle_forces=True,
        )[0]
        goal = np.array(_ROBOT_GOAL, dtype=float)
        ped_traj = [simulator.pysf_sim.peds.pos().copy()]
        robot_traj = [np.asarray(simulator.robots[0].pos, dtype=float).copy()]
        for _ in range(horizon):
            robot_pos = np.asarray(simulator.robots[0].pos, dtype=float)
            offset = goal - robot_pos
            distance = float(np.linalg.norm(offset))
            if distance > 0.3:
                command = offset / distance * robot_speed
            else:
                command = np.zeros(2)
            simulator.step_once([(float(command[0]), float(command[1]))])
            ped_traj.append(simulator.pysf_sim.peds.pos().copy())
            robot_traj.append(np.asarray(simulator.robots[0].pos, dtype=float).copy())
        return {
            "status": "ok",
            "steps_completed": horizon,
            "ped_traj": np.stack(ped_traj)[:, 0, :].tolist(),
            "robot_traj": np.stack(robot_traj).tolist(),
        }
    except Exception as error:  # noqa: BLE001 - child must report, never raise
        return {"status": "error", "error": f"{type(error).__name__}: {error}"}


def _sleep_job(job: dict[str, Any]) -> dict[str, Any]:
    """Deterministic slow job used to prove child timeout/termination.

    Returns:
        Status payload after the sleep elapses.
    """
    time.sleep(float(job.get("sleep_s", 5.0)))
    return {"status": "ok"}


def _child_main(entry_name: str, payload: dict[str, Any], conn: Any) -> None:
    """Module-level child entry so spawn and fork contexts can both start it."""
    entry = _execute_episode_job if entry_name == "episode" else _sleep_job
    try:
        conn.send(entry(payload))
    except Exception as error:  # noqa: BLE001 - transport must survive
        try:
            conn.send({"status": "error", "error": f"{type(error).__name__}: {error}"})
        except Exception:  # noqa: BLE001 - nothing left to report through
            pass
    finally:
        conn.close()


def _run_owned_child(
    job: dict[str, Any], timeout_s: float, *, target: str = "episode"
) -> dict[str, Any]:
    """Run one job in an owned child process with timeout and termination.

    Args:
        job: Plain-data job payload for the child target.
        timeout_s: Wall-clock budget for the child execution.
        target: ``"episode"`` for episode execution or ``"sleep"`` for tests.

    Returns:
        Child payload, or a timeout/interrupt marker. A timed-out or
        interrupted child is always terminated before returning.
    """
    context = multiprocessing.get_context()
    parent_conn, child_conn = context.Pipe(duplex=False)
    process = context.Process(target=_child_main, args=(target, job, child_conn))
    try:
        process.start()
    except OSError as error:
        parent_conn.close()
        child_conn.close()
        return {"outcome": "error", "error": f"child spawn failed: {error}"}
    # The parent never uses the child end; close it only after start so the
    # forked child inherits a live descriptor.
    child_conn.close()
    try:
        if parent_conn.poll(timeout_s):
            try:
                payload = parent_conn.recv()
            except EOFError as error:
                payload = {"status": "error", "error": f"child closed pipe: {error}"}
            process.join(10)
            if process.is_alive():
                process.terminate()
                process.join(10)
            if isinstance(payload, dict):
                return {"outcome": "ok", "payload": payload}
            return {"outcome": "error", "error": "child returned a non-mapping payload"}
        process.terminate()
        process.join(10)
        return {"outcome": "timeout", "error": f"child exceeded {timeout_s:g}s and was terminated"}
    except KeyboardInterrupt:
        process.terminate()
        process.join(10)
        return {"outcome": "interrupted", "error": "cancelled by user; owned child terminated"}
    finally:
        parent_conn.close()
        if process.is_alive():
            process.terminate()
            process.join(10)
        process.close()


def _telemetry_metrics(
    payload: dict[str, Any], *, horizon: int, motion_epsilon_m: float
) -> dict[str, Any] | None:
    """Derive measured metrics from child telemetry.

    Returns:
        Metric mapping, or None when the telemetry is unusable.
    """
    try:
        import numpy as np  # noqa: PLC0415 - lazy: keep module import light

        if payload.get("status") != "ok":
            return None
        if int(payload.get("steps_completed", -1)) != horizon:
            return None
        ped = np.asarray(payload["ped_traj"], dtype=float)
        robot = np.asarray(payload["robot_traj"], dtype=float)
        if ped.shape != (horizon + 1, 2) or robot.shape != (horizon + 1, 2):
            return None
        if not bool(np.all(np.isfinite(ped))) or not bool(np.all(np.isfinite(robot))):
            return None
        step_speeds = np.linalg.norm(np.diff(ped, axis=0), axis=1) / _DT_S
        displacement = float(np.linalg.norm(ped[-1] - ped[0]))
        robot_displacement = float(np.linalg.norm(robot[-1] - robot[0]))
        distances = np.linalg.norm(ped - robot, axis=1)
        onset_candidates = np.flatnonzero(np.linalg.norm(ped - ped[0], axis=1) > motion_epsilon_m)
        onset = int(onset_candidates[0]) if onset_candidates.size else horizon
        goal_distance = float(np.linalg.norm(robot[-1] - np.array(_ROBOT_GOAL)))
        return {
            "ped_mean_speed_m_s": float(np.mean(step_speeds)),
            "min_robot_ped_distance_m": float(np.min(distances)),
            "robot_goal_reached": 1 if goal_distance < _GOAL_REACHED_RADIUS_M else 0,
            "ped_motion_onset_step": onset,
            "ped_displacement_m": displacement,
            "robot_displacement_m": robot_displacement,
        }
    except (KeyError, TypeError, ValueError):
        return None


def _select_candidates(recipe: dict[str, Any], max_candidates: int) -> list[dict[str, Any]]:
    interventions = recipe["interventions"]
    ordered = sorted(
        interventions, key=lambda item: (int(item.get("priority", 0)), str(item["intervention_id"]))
    )
    return [dict(item) for item in ordered[:max_candidates]]


def _intervention_update(
    factor: str, params: Any, *, control_speed: float, control_delay: float
) -> tuple[dict[str, Any] | None, str | None]:
    """Resolve an intervention to a treatment spec delta.

    Returns:
        Tuple of (treatment delta, unavailability reason); exactly one is set.
    """
    if factor not in SUPPORTED_FACTORS:
        return None, f"unsupported intervention factor: {factor}"
    if not isinstance(params, dict):
        return None, "missing intervention_parameters for candidate"
    if factor == "single_pedestrian_speed_offset":
        delta = params.get("speed_delta_m_s")
        if not _is_finite_number(delta) or float(delta) == 0.0:
            return None, "speed intervention requires a finite non-zero speed_delta_m_s"
        if abs(float(delta)) > 1.0:
            return None, "speed_delta_m_s exceeds the fixture bound of 1.0 m/s"
        updated = control_speed + float(delta)
        if not 0.0 < updated <= 3.0:
            return None, "updated pedestrian speed leaves the (0, 3.0] m/s validity range"
        return {"ped_speed_m_s": updated, "ped_start_delay_s": control_delay}, None
    delay_delta = params.get("dt_s")
    if not _is_finite_number(delay_delta) or float(delay_delta) == 0.0:
        return None, "start-delay intervention requires a finite non-zero dt_s"
    if abs(float(delay_delta)) > 5.0:
        return None, "dt_s exceeds the fixture bound of 5.0 s"
    if control_delay + float(delay_delta) < 0.0:
        return None, "updated start delay would be negative"
    # Observed canonical-simulator limitation: the start-delay release path leaves
    # max_speeds at zero, so the delayed pedestrian never moves. Never synthesize
    # motion; report the candidate as unavailable instead.
    return None, (
        "intervention_not_executable: the canonical single-pedestrian start-delay "
        "release path holds max_speeds at zero, so the delayed pedestrian never "
        "moves on the fixture path; refusing to synthesize motion"
    )


def _specs_match_except(
    control_spec: dict[str, Any], treatment_spec: dict[str, Any], allowed_key: str
) -> bool:
    if set(control_spec) != set(treatment_spec):
        return False
    differences = [key for key in control_spec if control_spec[key] != treatment_spec[key]]
    return differences == [allowed_key]


@dataclass
class _Executor:
    request: ComponentRequest
    config: ExecuteConfig
    recipe: dict[str, Any]
    output_dir: Path
    resume: bool = False
    _attempts: list[dict[str, Any]] = field(default_factory=list)
    _executions_consumed: int = 0
    _wall_elapsed_s: float = 0.0
    _started_at: float = 0.0
    _candidate_reports: list[dict[str, Any]] = field(default_factory=list)
    _traces: list[dict[str, Any]] = field(default_factory=list)

    def _elapsed(self) -> float:
        return self._wall_elapsed_s + (time.monotonic() - self._started_at)

    def _budget_remaining(self) -> bool:
        return self._executions_consumed < self.config.max_executions

    def _record_attempt(self, attempt: dict[str, Any]) -> None:
        self._attempts.append(attempt)

    def _load_resume_ledger(self) -> None:
        ledger_path = self.output_dir / "attempt-ledger.json"
        try:
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ReviewExecuteError(
                [f"cannot resume: unreadable attempt ledger: {error}"]
            ) from error
        if (
            not isinstance(ledger, dict)
            or ledger.get("schema_version") != ATTEMPT_LEDGER_SCHEMA_VERSION
            or ledger.get("request_id") != self.request.request_id
        ):
            raise ReviewExecuteError(["cannot resume: ledger request identity mismatch"])
        attempts = ledger.get("attempts", [])
        if not isinstance(attempts, list):
            raise ReviewExecuteError(["cannot resume: ledger attempts are malformed"])
        self._attempts = [dict(entry) for entry in attempts]
        consumed = ledger.get("executions_consumed", 0)
        self._executions_consumed = int(consumed) if _is_int(consumed) else 0
        elapsed = ledger.get("wall_elapsed_s", 0.0)
        self._wall_elapsed_s = float(elapsed) if _is_finite_number(elapsed) else 0.0

    def _attempt_done(self, candidate_id: str, kind: str) -> bool:
        return any(
            entry.get("candidate_id") == candidate_id and entry.get("kind") == kind
            for entry in self._attempts
        )

    def _write_ledger(self) -> str:
        """Write the attempt ledger, returning the file-bytes SHA-256.

        Returns:
            Hex digest of the ledger file bytes.
        """
        payload = {
            "schema_version": ATTEMPT_LEDGER_SCHEMA_VERSION,
            "request_id": self.request.request_id,
            "component_id": COMPONENT_ID,
            "recipe_id": str(self.recipe.get("recipe_id", "")),
            "attempts": list(self._attempts),
            "executions_consumed": self._executions_consumed,
            "wall_elapsed_s": round(self._elapsed(), 3),
        }
        return _write_json(self.output_dir / "attempt-ledger.json", payload)

    def _run_episode(self, candidate_id: str, kind: str, spec: dict[str, Any]) -> dict[str, Any]:
        job = {
            "seed": self.config.seed,
            "horizon_steps": self.config.horizon_steps,
            "robot_speed_m_s": self.config.robot_speed_m_s,
            "ped_speed_m_s": spec["ped_speed_m_s"],
            "ped_start_delay_s": spec["ped_start_delay_s"],
        }
        started = time.monotonic()
        outcome = _run_owned_child(job, self.config.per_execution_timeout_s)
        elapsed = time.monotonic() - started
        self._executions_consumed += 1
        if outcome["outcome"] == "ok":
            child_payload = outcome["payload"]
            metrics = _telemetry_metrics(
                child_payload,
                horizon=self.config.horizon_steps,
                motion_epsilon_m=self.config.motion_epsilon_m,
            )
            if metrics is None:
                attempt = {
                    "candidate_id": candidate_id,
                    "kind": kind,
                    "status": "failed",
                    "reason": "execution_error: child telemetry unusable or incomplete",
                    "elapsed_s": round(elapsed, 3),
                }
            else:
                attempt = {
                    "candidate_id": candidate_id,
                    "kind": kind,
                    "status": "ok",
                    "elapsed_s": round(elapsed, 3),
                }
                attempt["metrics"] = metrics
            self._record_attempt(attempt)
            return attempt
        if outcome["outcome"] == "timeout":
            attempt = {
                "candidate_id": candidate_id,
                "kind": kind,
                "status": "timed_out",
                "reason": f"per_execution_timeout: {outcome.get('error', '')}",
                "elapsed_s": round(elapsed, 3),
            }
            self._record_attempt(attempt)
            return attempt
        if outcome["outcome"] == "interrupted":
            attempt = {
                "candidate_id": candidate_id,
                "kind": kind,
                "status": "cancelled",
                "reason": f"cancelled_by_user: {outcome.get('error', '')}",
                "elapsed_s": round(elapsed, 3),
            }
            self._record_attempt(attempt)
            raise _ExecutionCancelled(attempt["reason"])
        attempt = {
            "candidate_id": candidate_id,
            "kind": kind,
            "status": "failed",
            "reason": f"execution_error: {outcome.get('error', '')}",
            "elapsed_s": round(elapsed, 3),
        }
        self._record_attempt(attempt)
        return attempt

    def _control_fidelity_ok(self, metrics: dict[str, Any]) -> tuple[bool, str]:
        if metrics["ped_displacement_m"] <= self.config.motion_epsilon_m:
            return False, "control pedestrian shows no measured motion"
        if metrics["robot_displacement_m"] <= self.config.motion_epsilon_m:
            return False, "control robot shows no measured motion"
        return True, ""

    def _execute_candidate(
        self, candidate: dict[str, Any], measurement: dict[str, Any]
    ) -> dict[str, Any]:
        candidate_id = str(candidate["intervention_id"])
        factor = str(candidate["factor"])
        params = self.config.intervention_parameters.get(candidate_id)
        update, reason = _intervention_update(
            factor,
            params,
            control_speed=float(self.recipe["control_conditions"].get("ped_speed_m_s", 1.0)),
            control_delay=float(self.recipe["control_conditions"].get("ped_start_delay_s", 0.0)),
        )
        if update is None:
            report = {
                "intervention_id": candidate_id,
                "factor": factor,
                "status": "unavailable",
                "reason": str(reason),
            }
            self._candidate_reports.append(report)
            return report
        control_spec = {
            "scenario_id": str(self.recipe["source_identity"].get("scenario_id", "")),
            "seed": self.config.seed,
            "horizon_steps": self.config.horizon_steps,
            "robot_speed_m_s": self.config.robot_speed_m_s,
            "ped_speed_m_s": float(self.recipe["control_conditions"].get("ped_speed_m_s", 1.0)),
            "ped_start_delay_s": float(
                self.recipe["control_conditions"].get("ped_start_delay_s", 0.0)
            ),
        }
        treatment_spec = dict(control_spec)
        changed_key = (
            "ped_speed_m_s" if factor == "single_pedestrian_speed_offset" else "ped_start_delay_s"
        )
        treatment_spec[changed_key] = update[changed_key]
        if not _specs_match_except(control_spec, treatment_spec, changed_key):
            raise ReviewExecuteError(["internal nonintervened config comparison failed"])
        control_attempt: dict[str, Any] | None = None
        if self._attempt_done(candidate_id, "control"):
            control_attempt = next(
                entry
                for entry in self._attempts
                if entry.get("candidate_id") == candidate_id and entry.get("kind") == "control"
            )
        else:
            control_attempt = self._run_episode(candidate_id, "control", control_spec)
        if control_attempt.get("status") == "timed_out":
            raise _ExecutionTimeout(str(control_attempt.get("reason", "")))
        if control_attempt.get("status") != "ok":
            report = {
                "intervention_id": candidate_id,
                "factor": factor,
                "status": "failed",
                "reason": f"control execution failed: {control_attempt.get('reason', '')}",
                "nonintervened_config_match": True,
            }
            self._candidate_reports.append(report)
            return report
        control_metrics = control_attempt["metrics"]
        fidelity_ok, fidelity_reason = self._control_fidelity_ok(control_metrics)
        if not fidelity_ok:
            # Failed control fidelity blocks treatment interpretation.
            report = {
                "intervention_id": candidate_id,
                "factor": factor,
                "status": "failed",
                "reason": f"control_fidelity_failure: {fidelity_reason}",
                "control_metrics": control_metrics,
                "nonintervened_config_match": True,
            }
            self._candidate_reports.append(report)
            return report
        treatment_attempt: dict[str, Any] | None = None
        if self._attempt_done(candidate_id, "treatment"):
            treatment_attempt = next(
                entry
                for entry in self._attempts
                if entry.get("candidate_id") == candidate_id and entry.get("kind") == "treatment"
            )
        else:
            treatment_attempt = self._run_episode(candidate_id, "treatment", treatment_spec)
        if treatment_attempt.get("status") == "timed_out":
            raise _ExecutionTimeout(str(treatment_attempt.get("reason", "")))
        if treatment_attempt.get("status") != "ok":
            report = {
                "intervention_id": candidate_id,
                "factor": factor,
                "status": "failed",
                "reason": f"treatment execution failed: {treatment_attempt.get('reason', '')}",
                "control_metrics": control_metrics,
                "nonintervened_config_match": True,
            }
            self._candidate_reports.append(report)
            return report
        treatment_metrics = treatment_attempt["metrics"]
        metric_name = str(measurement["name"])
        expected_direction = str(measurement["expected_direction"])
        control_activated = control_metrics["ped_displacement_m"] > self.config.motion_epsilon_m
        treatment_activated = (
            abs(treatment_metrics["ped_mean_speed_m_s"] - control_metrics["ped_mean_speed_m_s"])
            > self.config.activation_speed_tolerance_m_s
        )
        pair_result = evaluate_counterfactual_pair(
            {
                "mechanism_activated": control_activated,
                "metrics": {metric_name: control_metrics[metric_name]},
            },
            {
                "mechanism_activated": treatment_activated,
                "metrics": {metric_name: treatment_metrics[metric_name]},
            },
            PairHypothesis(
                expected_mechanism=factor,
                outcome_metric=metric_name,
                expected_direction=expected_direction,
            ),
        )
        self._traces.append(
            {
                "schema_version": ACTIVATION_TRACE_SCHEMA_VERSION,
                "intervention_id": candidate_id,
                "factor": factor,
                "control_activated": control_activated,
                "treatment_activated": treatment_activated,
                "control_metrics": control_metrics,
                "treatment_metrics": treatment_metrics,
            }
        )
        report = {
            "intervention_id": candidate_id,
            "factor": factor,
            "status": "complete",
            "verdict": pair_result.verdict,
            "verdict_reason": pair_result.reason,
            "control_metrics": control_metrics,
            "treatment_metrics": treatment_metrics,
            "control_activated": control_activated,
            "treatment_activated": treatment_activated,
            "nonintervened_config_match": True,
        }
        self._candidate_reports.append(report)
        return report


class _ExecutionTimeout(Exception):
    """Internal signal: an owned child exceeded its execution timeout."""


class _ExecutionCancelled(Exception):
    """Internal signal: execution was cancelled; the owned child was terminated."""


def _measurement_for_recipe(recipe: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    measurements = recipe.get("measurements", [])
    if not measurements:
        return None, "recipe carries no measurements"
    first = measurements[0]
    name = str(first.get("name", ""))
    direction = str(first.get("expected_direction", ""))
    if name not in SUPPORTED_MEASUREMENTS:
        return None, f"unsupported measurement: {name}"
    if direction not in ("increase", "decrease"):
        return None, f"measurement {name} needs expected_direction increase|decrease"
    return dict(first), None


def _commit_provenance() -> dict[str, Any]:
    return {
        "component_id": COMPONENT_ID,
        "component_version": COMPONENT_VERSION,
        "commit": _repo_commit(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def _write_complete_outputs(
    executor: _Executor, provenance: dict[str, Any]
) -> list[dict[str, Any]]:
    output_dir = executor.output_dir
    request = executor.request
    report = {
        "schema_version": EXECUTE_REPORT_SCHEMA_VERSION,
        "request_id": request.request_id,
        "component_id": COMPONENT_ID,
        "recipe_id": str(executor.recipe.get("recipe_id", "")),
        "source_identity": dict(executor.recipe.get("source_identity", {})),
        "candidates": list(executor._candidate_reports),
        "budget": {
            "max_candidates": executor.config.max_candidates,
            "max_executions": executor.config.max_executions,
            "executions_consumed": executor._executions_consumed,
            "wall_timeout_s": executor.config.wall_timeout_s,
            "wall_elapsed_s": round(executor._elapsed(), 3),
        },
        "provenance": provenance,
    }
    traces = {
        "schema_version": ACTIVATION_TRACE_SCHEMA_VERSION,
        "request_id": request.request_id,
        "traces": list(executor._traces),
    }
    executor._write_ledger()
    ledger_path = output_dir / "attempt-ledger.json"
    ledger_digest = hashlib.sha256(ledger_path.read_bytes()).hexdigest()
    report_digest = _write_json(output_dir / "execute-report.json", report)
    traces_digest = _write_json(output_dir / "activation-traces.json", traces)
    manifest = {
        "schema_version": PRESERVATION_MANIFEST_SCHEMA_VERSION,
        "request_id": request.request_id,
        "recipe_id": str(executor.recipe.get("recipe_id", "")),
        "recipe_digest": _canonical_digest(executor.recipe),
        "config_digest": _canonical_digest(executor.config.recipe),
        "source_identity": dict(executor.recipe.get("source_identity", {})),
        "retrieval_destination": str(executor.recipe.get("preservation_destination", "")),
        "artifacts": {
            "execute-report.json": report_digest,
            "activation-traces.json": traces_digest,
            "attempt-ledger.json": ledger_digest,
        },
        "provenance": provenance,
    }
    manifest_digest = _write_json(output_dir / "preservation-manifest.json", manifest)
    prefix = Path(request.output_directory)
    return [
        {
            "artifact_id": "execute-report.json",
            "uri": str(prefix / "execute-report.json"),
            "sha256": report_digest,
        },
        {
            "artifact_id": "activation-traces.json",
            "uri": str(prefix / "activation-traces.json"),
            "sha256": traces_digest,
        },
        {
            "artifact_id": "attempt-ledger.json",
            "uri": str(prefix / "attempt-ledger.json"),
            "sha256": ledger_digest,
        },
        {
            "artifact_id": "preservation-manifest.json",
            "uri": str(prefix / "preservation-manifest.json"),
            "sha256": manifest_digest,
        },
    ]


def _diagnostics(executor: _Executor) -> list[dict[str, Any]]:
    return [
        {
            "intervention_id": report.get("intervention_id"),
            "status": report.get("status"),
            "verdict": report.get("verdict", ""),
            "reason": report.get("reason", report.get("verdict_reason", "")),
        }
        for report in executor._candidate_reports
    ]


def _final_result(
    request: ComponentRequest,
    *,
    status: str,
    reason: str,
    artifacts: tuple[dict[str, Any], ...] = (),
    diagnostics: tuple[dict[str, Any], ...] = (),
    provenance: dict[str, Any] | None = None,
) -> ComponentResult:
    payload = {
        "schema_version": COMPONENT_RESULT_SCHEMA_VERSION,
        "request_id": request.request_id,
        "component_id": request.component_id,
        "status": status,
        "reason": reason,
    }
    if artifacts:
        payload["artifacts"] = [dict(entry) for entry in artifacts]
    if diagnostics:
        payload["diagnostics"] = [dict(entry) for entry in diagnostics]
    if provenance:
        payload["provenance"] = dict(provenance)
    try:
        return component_result_from_dict(payload)
    except ReviewContractsValidationError as error:
        return ComponentResult(
            request_id=request.request_id,
            component_id=request.component_id,
            status="failed",
            reason=f"internal_result_invalid: {'; '.join(error.errors)}",
        )


def _admit_request(
    request: ComponentRequest,
) -> tuple[
    ExecuteConfig | None, dict[str, Any] | None, dict[str, Any] | None, ComponentResult | None
]:
    """Run admission gates for component, capability, config, and recipe identity.

    Returns:
        Tuple of (config, recipe document, measurement, early result); exactly
        one of the early result or the full triple is set.
    """
    if request.component_id != COMPONENT_ID:
        return (
            None,
            None,
            None,
            _final_result(
                request,
                status="unavailable",
                reason=f"unsupported component: {request.component_id}",
            ),
        )
    missing = _unsupported_capabilities(request)
    if missing:
        return (
            None,
            None,
            None,
            _final_result(
                request,
                status="unavailable",
                reason=f"missing capabilities: {', '.join(sorted(missing))}",
            ),
        )
    try:
        config = validate_execute_config(request.config, source="config")
    except ReviewExecuteError as error:
        return (
            None,
            None,
            None,
            _final_result(
                request, status="failed", reason=f"invalid_config: {'; '.join(error.errors)}"
            ),
        )
    if (
        config.required_component_version is not None
        and config.required_component_version != COMPONENT_VERSION
    ):
        return (
            None,
            None,
            None,
            _final_result(
                request,
                status="unavailable",
                reason=(
                    "incompatible_version: required "
                    f"{config.required_component_version} != {COMPONENT_VERSION}"
                ),
            ),
        )
    try:
        validated_recipe = experiment_recipe_from_dict(config.recipe, source="config.recipe")
    except ReviewContractsValidationError as error:
        return (
            None,
            None,
            None,
            _final_result(
                request, status="failed", reason=f"corrupt_recipe: {'; '.join(error.errors)}"
            ),
        )
    recipe_doc = validated_recipe.document
    source_identity = recipe_doc.get("source_identity", {})
    if (
        not isinstance(source_identity, dict)
        or not isinstance(source_identity.get("scenario_id"), str)
        or not source_identity["scenario_id"].strip()
    ):
        return (
            None,
            None,
            None,
            _final_result(
                request,
                status="failed",
                reason="invalid_source_identity: source_identity.scenario_id must be a non-empty string",
            ),
        )
    control_conditions = recipe_doc.get("control_conditions", {})
    if not isinstance(control_conditions, dict):
        return (
            None,
            None,
            None,
            _final_result(
                request, status="failed", reason="invalid_control_conditions: mapping required"
            ),
        )
    control_speed = control_conditions.get("ped_speed_m_s", 1.0)
    if not _is_finite_number(control_speed) or not 0.0 < float(control_speed) <= 3.0:
        return (
            None,
            None,
            None,
            _final_result(
                request,
                status="failed",
                reason="invalid_control_conditions: ped_speed_m_s must be within (0, 3.0]",
            ),
        )
    measurement, measurement_reason = _measurement_for_recipe(recipe_doc)
    if measurement is None:
        return (
            None,
            None,
            None,
            _final_result(
                request,
                status="unavailable",
                reason=f"unsupported_measurement: {measurement_reason}",
            ),
        )
    return config, recipe_doc, measurement, None


def _prepare_output_dir(
    request: ComponentRequest, root: Path, *, resume: bool
) -> tuple[Path | None, ComponentResult | None]:
    """Resolve the output directory honoring collision and resume semantics.

    Returns:
        Tuple of (output directory, early result); exactly one is set.
    """
    output_dir = root / request.output_directory
    if output_dir.exists():
        if not resume:
            return None, _final_result(
                request,
                status="failed",
                reason=f"output_collision: output already exists: {request.output_directory}",
            )
        return output_dir, None
    if resume:
        return None, _final_result(
            request, status="failed", reason="cannot resume: output directory does not exist"
        )
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        return None, _final_result(
            request, status="failed", reason=f"unwritable output directory: {error}"
        )
    return output_dir, None


def _drive_candidates(
    executor: _Executor, config: ExecuteConfig, measurement: dict[str, Any]
) -> tuple[str, str]:
    """Execute selected candidates within budget.

    Returns:
        Tuple of terminal (status, reason) for the drive.
    """
    candidates = _select_candidates(executor.recipe, config.max_candidates)
    status = "complete"
    reason = "all selected candidates reached a terminal state"
    try:
        for candidate in candidates:
            if not executor._budget_remaining():
                return "partial", "execution_budget_exhausted: stopping before the next candidate"
            if executor._elapsed() >= config.wall_timeout_s:
                return "partial", "wall_timeout: stopping before the next candidate"
            executor._execute_candidate(candidate, measurement)
    except _ExecutionTimeout as error:
        return "cancelled", f"per_execution_timeout: {error}; owned child terminated"
    except _ExecutionCancelled as error:
        return "cancelled", f"cancelled_by_user: {error}"
    return status, reason


def _settle(
    executor: _Executor, provenance: dict[str, Any], status: str, reason: str
) -> ComponentResult:
    """Settle the final result envelope for a driven executor.

    Returns:
        Validated component result for the recorded candidate reports.
    """
    request = executor.request
    terminal = [
        report
        for report in executor._candidate_reports
        if report.get("status") in ("complete", "unavailable")
    ]
    succeeded = [
        report for report in executor._candidate_reports if report.get("status") == "complete"
    ]
    if status == "complete" and not succeeded:
        status = "failed"
        reason = "no candidate completed: " + "; ".join(
            str(report.get("reason", report.get("status", "")))
            for report in executor._candidate_reports
        )
    if status == "complete":
        artifacts = _write_complete_outputs(executor, provenance)
        return _final_result(
            request,
            status="complete",
            reason=reason,
            artifacts=tuple(artifacts),
            diagnostics=tuple(_diagnostics(executor)),
            provenance=provenance,
        )
    try:
        executor._write_ledger()
    except OSError:
        pass
    if status == "partial" and not terminal:
        status = "failed"
        reason = f"{reason}; no candidate reached a terminal state"
    return _final_result(
        request,
        status=status,
        reason=reason,
        diagnostics=tuple(_diagnostics(executor)),
        provenance=provenance,
    )


def run(
    request: ComponentRequest, *, base: Path | None = None, resume: bool = False
) -> ComponentResult:
    """Execute one bounded review-execute request.

    Args:
        request: Validated component request.
        base: Base directory the request output directory resolves under.
        resume: Reuse an existing output directory ledger instead of failing
            on output collision.

    Returns:
        Component result with artifacts (complete only), diagnostics, and
        provenance.
    """
    root = base if base is not None else Path.cwd()
    config, recipe_doc, measurement, early = _admit_request(request)
    if early is not None or config is None or recipe_doc is None or measurement is None:
        assert early is not None
        return early
    output_dir, dir_early = _prepare_output_dir(request, root, resume=resume)
    if dir_early is not None or output_dir is None:
        assert dir_early is not None
        return dir_early
    executor = _Executor(
        request=request, config=config, recipe=recipe_doc, output_dir=output_dir, resume=resume
    )
    if resume:
        try:
            executor._load_resume_ledger()
        except ReviewExecuteError as error:
            return _final_result(request, status="failed", reason="; ".join(error.errors))
    provenance = _commit_provenance()
    provenance["recipe_id"] = str(recipe_doc.get("recipe_id", ""))
    provenance["source_identity"] = dict(recipe_doc.get("source_identity", {}))
    executor._started_at = time.monotonic()
    try:
        status, reason = _drive_candidates(executor, config, measurement)
    except ReviewExecuteError as error:
        return _final_result(request, status="failed", reason="; ".join(error.errors))
    return _settle(executor, provenance, status, reason)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Execute SREV-22 bounded review experiments.")
    parser.add_argument("--input", required=True, help="Component request JSON file.")
    parser.add_argument("--config", required=False, default=None, help="Optional config JSON file.")
    parser.add_argument("--output", required=True, help="Output directory (must not exist).")
    parser.add_argument("--base", required=False, default=None, help="Base directory for output.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the attempt ledger in an existing output directory.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the review-execute component.

    Args:
        argv: Command-line arguments (defaults to process arguments).

    Returns:
        Process exit code (0 when the result status is complete).
    """
    args = _build_parser().parse_args(argv)
    try:
        payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReviewExecuteError([f"cannot read request: {error}"]) from error
    if args.config is not None:
        try:
            config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ReviewExecuteError([f"cannot read config: {error}"]) from error
        if isinstance(config, dict):
            payload = {**payload, "config": {**payload.get("config", {}), **config}}
    payload = {**payload, "output_directory": args.output}
    request = component_request_from_dict(payload, source=args.input)
    result = run(
        request, base=Path(args.base) if args.base is not None else None, resume=args.resume
    )
    print(json.dumps(asdict(result), sort_keys=True, indent=2))  # noqa: T201 - CLI output
    return 0 if result.status == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
