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
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

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
REQUIRED_TELEMETRY_METRICS = frozenset(
    {
        "ped_mean_speed_m_s",
        "min_robot_ped_distance_m",
        "robot_goal_reached",
        "ped_motion_onset_step",
        "ped_displacement_m",
        "robot_displacement_m",
    }
)

# These are intentionally small hard ceilings.  The executor is a diagnostic
# component, so a caller cannot turn a fixture request into an unbounded
# campaign by changing the JSON config or recipe budget.
MAX_CANDIDATES = 3
MAX_EXECUTIONS = 6
MAX_WALL_TIMEOUT_S = 600.0
MAX_PER_EXECUTION_TIMEOUT_S = 120.0
MAX_SEED = 2**32 - 1
SUPPORTED_STOP_RULES = frozenset(
    {
        "exhausted_candidates",
        "execution_budget_exhausted",
        "wall_timeout",
        "control_fidelity_failure_blocks_treatment",
    }
)
ALLOWED_INTERVENTION_PARAMETER_KEYS = frozenset({"speed_delta_m_s", "dt_s"})
DIAGNOSTIC_EVIDENCE_BOUNDARY = "diagnostic_only"
DEPENDENT_FAMILY_STATUS = "standalone_fixture_only"
_SAFE_PRESERVATION_DESTINATION_PREFIXES = ("external:", "artifact:", "fixture:")
_CHILD_TARGETS = frozenset({"episode", "sleep"})
_SUPPORTED_FIXTURE_SCENARIO_ID = "srev22-tiny-crossing"
_SUPPORTED_FIXTURE_SOURCE_REFERENCE = (
    ("artifact_id", "recipe-srev22-smoke"),
    ("uri", "recipe.json"),
    ("format", "experiment-recipe.v1"),
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
    unknown = sorted((key for key in raw if key not in allowed), key=str)
    if unknown:
        raise ReviewExecuteError(
            ["unknown config keys are rejected: " + ", ".join(str(key) for key in unknown)],
            source=source,
        )
    if "recipe" not in raw or not isinstance(raw["recipe"], dict):
        raise ReviewExecuteError(
            ["config.recipe must be an experiment-recipe mapping"], source=source
        )
    errors: list[str] = []
    planner = raw.get("planner", "simple_policy")
    if not isinstance(planner, str) or planner not in SUPPORTED_PLANNERS:
        errors.append(f"unsupported planner: {planner!r}; supported: {list(SUPPORTED_PLANNERS)}")
    seed = _check_int_field(raw, "seed", 7, minimum=0, maximum=MAX_SEED, errors=errors)
    horizon = _check_int_field(raw, "horizon_steps", 60, minimum=1, maximum=600, errors=errors)
    robot_speed = _check_float_field(
        raw, "robot_speed_m_s", 1.0, minimum=0.1, maximum=2.0, errors=errors
    )
    max_candidates = _check_int_field(
        raw, "max_candidates", MAX_CANDIDATES, minimum=1, maximum=MAX_CANDIDATES, errors=errors
    )
    max_executions = _check_int_field(
        raw, "max_executions", MAX_EXECUTIONS, minimum=1, maximum=MAX_EXECUTIONS, errors=errors
    )
    wall_timeout = _check_float_field(
        raw,
        "wall_timeout_s",
        MAX_WALL_TIMEOUT_S,
        minimum=0.0,
        maximum=MAX_WALL_TIMEOUT_S,
        errors=errors,
    )
    if wall_timeout <= 0.0:
        errors.append("wall_timeout_s must be positive")
    per_execution_timeout = _check_float_field(
        raw,
        "per_execution_timeout_s",
        MAX_PER_EXECUTION_TIMEOUT_S,
        minimum=0.0,
        maximum=MAX_PER_EXECUTION_TIMEOUT_S,
        errors=errors,
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
    if required_version is not None and (
        not isinstance(required_version, str) or not required_version.strip()
    ):
        errors.append("required_component_version must be a non-empty string")
    intervention_parameters = raw.get("intervention_parameters", {})
    if not isinstance(intervention_parameters, dict):
        errors.append("intervention_parameters must be a mapping")
    else:
        _validate_intervention_parameters(intervention_parameters, errors)
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


def _validate_intervention_parameters(parameters: dict[Any, Any], errors: list[str]) -> None:
    """Keep intervention parameters declarative and limited to scalar deltas."""
    for candidate_id, value in parameters.items():
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            errors.append("intervention_parameters keys must be non-empty strings")
            continue
        if not isinstance(value, dict):
            errors.append(f"intervention_parameters[{candidate_id!r}] must be a mapping")
            continue
        unknown = sorted(
            (key for key in value if key not in ALLOWED_INTERVENTION_PARAMETER_KEYS),
            key=str,
        )
        if unknown:
            errors.append(
                f"intervention_parameters[{candidate_id!r}] has unknown keys: "
                + ", ".join(str(key) for key in unknown)
            )
        for key, parameter_value in value.items():
            if key in ALLOWED_INTERVENTION_PARAMETER_KEYS and not _is_finite_number(
                parameter_value
            ):
                errors.append(
                    f"intervention_parameters[{candidate_id!r}].{key} must be a finite number"
                )


def _mapping_unknown_keys(value: Any, allowed: set[str]) -> list[str]:
    """Return stable stringified unknown keys for a declarative mapping."""
    if not isinstance(value, dict):
        return []
    return sorted((str(key) for key in value if key not in allowed), key=str)


def _validate_recipe_execution_contract(  # noqa: C901, PLR0912, PLR0915
    recipe: dict[str, Any], config: ExecuteConfig
) -> list[str]:
    """Validate the executor-owned recipe subset and its finite safety envelope.

    Returns:
        Stable validation errors; an empty list means the contract is usable.
    """
    errors: list[str] = []
    try:
        _canonical_digest(recipe)
    except (TypeError, ValueError):
        errors.append("invalid_recipe: recipe must contain strict-JSON values")
    source_identity = recipe.get("source_identity")
    if not isinstance(source_identity, dict):
        return ["invalid_source_identity: mapping required"]
    if source_identity.get("scenario_id") != _SUPPORTED_FIXTURE_SCENARIO_ID:
        errors.append(
            "invalid_source_identity: scenario_id must bind the supported fixture "
            f"{_SUPPORTED_FIXTURE_SCENARIO_ID!r}"
        )
    if source_identity.get("source_ref") != _supported_fixture_source_reference():
        errors.append(
            "invalid_source_identity: source_ref must bind the immutable supported "
            "fixture source reference"
        )
    kind = source_identity.get("kind")
    if kind not in {"fixture", "diagnostic"}:
        errors.append(
            "invalid_evidence_boundary: source_identity.kind must be fixture or diagnostic"
        )
    if source_identity.get("evidence_boundary") != DIAGNOSTIC_EVIDENCE_BOUNDARY:
        errors.append("invalid_evidence_boundary: evidence_boundary must be diagnostic_only")
    if source_identity.get("scientific_claim_allowed") is not False:
        errors.append("invalid_evidence_boundary: scientific claims are not allowed")
    if source_identity.get("dependent_family_status") != DEPENDENT_FAMILY_STATUS:
        errors.append(
            "invalid_evidence_boundary: dependent_family_status must be standalone_fixture_only"
        )

    control_conditions = recipe.get("control_conditions")
    allowed_control_keys = {"ped_speed_m_s", "ped_start_delay_s"}
    unknown_control = _mapping_unknown_keys(control_conditions, allowed_control_keys)
    if unknown_control:
        errors.append("unknown control_conditions keys are rejected: " + ", ".join(unknown_control))
    if not isinstance(control_conditions, dict):
        errors.append("invalid_control_conditions: mapping required")
    else:
        delay = control_conditions.get("ped_start_delay_s", 0.0)
        if not _is_finite_number(delay) or float(delay) < 0.0 or float(delay) > MAX_WALL_TIMEOUT_S:
            errors.append(
                "invalid_control_conditions: ped_start_delay_s must be finite within 0..600 s"
            )

    interventions = recipe.get("interventions")
    if not isinstance(interventions, list):
        errors.append("invalid_interventions: interventions must be a list")
        interventions = []
    candidate_factors = {
        str(item["intervention_id"]): item.get("factor")
        for item in interventions
        if isinstance(item, dict) and isinstance(item.get("intervention_id"), str)
    }
    unknown_parameter_candidates = sorted(
        (str(key) for key in set(config.intervention_parameters) - set(candidate_factors)),
        key=str,
    )
    if unknown_parameter_candidates:
        errors.append(
            "intervention_parameters reference unknown candidates: "
            + ", ".join(unknown_parameter_candidates)
        )
    for candidate_id, factor in candidate_factors.items():
        parameters = config.intervention_parameters.get(candidate_id)
        if not isinstance(parameters, dict):
            continue
        if factor == "single_pedestrian_speed_offset" and "dt_s" in parameters:
            errors.append(
                f"intervention_parameters[{candidate_id!r}] has a delay parameter for a speed factor"
            )
        if factor == "single_pedestrian_start_delay_offset" and "speed_delta_m_s" in parameters:
            errors.append(
                f"intervention_parameters[{candidate_id!r}] has a speed parameter for a delay factor"
            )

    budget = recipe.get("budget")
    allowed_budget_keys = {"max_candidates", "max_executions", "wall_timeout_s"}
    unknown_budget = _mapping_unknown_keys(budget, allowed_budget_keys)
    if unknown_budget:
        errors.append("unknown recipe budget keys are rejected: " + ", ".join(unknown_budget))
    if not isinstance(budget, dict):
        errors.append("invalid_budget: recipe budget must be a mapping")
    else:
        missing_budget = sorted((str(key) for key in allowed_budget_keys - set(budget)), key=str)
        if missing_budget:
            errors.append("invalid_budget: required fields missing: " + ", ".join(missing_budget))
        recipe_max_candidates = _check_int_field(
            budget,
            "max_candidates",
            MAX_CANDIDATES,
            minimum=1,
            maximum=MAX_CANDIDATES,
            errors=errors,
        )
        recipe_max_executions = _check_int_field(
            budget,
            "max_executions",
            MAX_EXECUTIONS,
            minimum=1,
            maximum=MAX_EXECUTIONS,
            errors=errors,
        )
        recipe_wall_timeout = _check_float_field(
            budget,
            "wall_timeout_s",
            MAX_WALL_TIMEOUT_S,
            minimum=0.0,
            maximum=MAX_WALL_TIMEOUT_S,
            errors=errors,
        )
        if recipe_wall_timeout <= 0.0:
            errors.append("budget.wall_timeout_s must be positive")
        if config.max_candidates > recipe_max_candidates:
            errors.append("invalid_budget: config.max_candidates exceeds recipe budget")
        if config.max_executions > recipe_max_executions:
            errors.append("invalid_budget: config.max_executions exceeds recipe budget")
        if config.wall_timeout_s > recipe_wall_timeout:
            errors.append("invalid_budget: config.wall_timeout_s exceeds recipe budget")

    stop_rules = recipe.get("stop_rules")
    if not isinstance(stop_rules, list) or not all(isinstance(rule, str) for rule in stop_rules):
        errors.append("invalid_stop_rules: stop_rules must be a list of strings")
    else:
        if len(stop_rules) != len(set(stop_rules)):
            errors.append("invalid_stop_rules: duplicate stop rules are rejected")
        unknown_stop_rules = sorted(set(stop_rules) - SUPPORTED_STOP_RULES)
        missing_stop_rules = sorted(SUPPORTED_STOP_RULES - set(stop_rules))
        if unknown_stop_rules:
            errors.append("invalid_stop_rules: unsupported rules: " + ", ".join(unknown_stop_rules))
        if missing_stop_rules:
            errors.append(
                "invalid_stop_rules: required rules missing: " + ", ".join(missing_stop_rules)
            )

    measurements = recipe.get("measurements")
    if isinstance(measurements, list) and len(measurements) != 1:
        errors.append("unsupported_measurement: exactly one driving measurement is supported")

    preservation_destination = recipe.get("preservation_destination")
    if not isinstance(preservation_destination, str) or not preservation_destination.strip():
        errors.append("invalid_preservation_destination: non-empty destination required")
    elif (
        not preservation_destination.startswith(_SAFE_PRESERVATION_DESTINATION_PREFIXES)
        or Path(preservation_destination).is_absolute()
        or ".." in Path(preservation_destination).parts
    ):
        errors.append(
            "invalid_preservation_destination: use a relative external:, artifact:, or fixture: URI"
        )
    return errors


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: Any) -> str:
    """Atomically write a component-owned strict-JSON artifact.

    Returns:
        SHA-256 digest of the written file bytes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or path.is_symlink():
        raise OSError(f"refusing to write through symlink: {path}")
    text = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    if tmp_path.is_symlink():
        raise OSError(f"refusing to write through symlink: {tmp_path}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    file_descriptor = os.open(tmp_path, flags, 0o600)
    with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
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


def _supported_fixture_source_reference() -> dict[str, str]:
    """Return the immutable logical source reference for the supported fixture."""
    return dict(_SUPPORTED_FIXTURE_SOURCE_REFERENCE)


def _request_source_reference(request: ComponentRequest) -> dict[str, str] | None:
    """Return the one source reference accepted by the fixture execution path."""
    if len(request.sources) != 1:
        return None
    source = request.sources[0]
    return {
        "artifact_id": source.artifact_id,
        "uri": source.uri,
        "format": source.format,
    }


def _episode_job_identity_error(job: dict[str, Any]) -> str | None:
    """Reject episode jobs that are not bound to the hard-coded fixture.

    Returns:
        An invalid-identity reason, or ``None`` for the supported fixture.
    """
    if job.get("scenario_id") != _SUPPORTED_FIXTURE_SCENARIO_ID:
        return (
            "invalid_source_identity: episode scenario_id is not bound to the "
            f"supported fixture {_SUPPORTED_FIXTURE_SCENARIO_ID!r}"
        )
    if job.get("source_ref") != _supported_fixture_source_reference():
        return (
            "invalid_source_identity: episode source_ref is not bound to the "
            "immutable supported fixture source reference"
        )
    return None


def _execute_episode_job(job: dict[str, Any]) -> dict[str, Any]:
    """Run one control/treatment episode inside an owned child process.

    Args:
        job: Plain-data episode spec (seed, horizon, speeds, delays).

    Returns:
        Plain-data telemetry payload or an error payload; never raises.
    """
    try:
        identity_error = _episode_job_identity_error(job)
        if identity_error is not None:
            return {"status": "error", "error": identity_error}
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
    try:
        if entry_name == "episode":
            entry = _execute_episode_job
        elif entry_name == "sleep":
            entry = _sleep_job
        else:
            raise ValueError(f"unsupported child target: {entry_name!r}")
        conn.send(entry(payload))
    except Exception as error:  # noqa: BLE001 - transport must survive
        try:
            conn.send({"status": "error", "error": f"{type(error).__name__}: {error}"})
        except Exception:  # noqa: BLE001 - nothing left to report through
            pass
    finally:
        conn.close()


def _terminate_owned_process(process: Any) -> None:
    """Stop an owned child within a small fixed cleanup allowance."""
    if not process.is_alive():
        return
    process.terminate()
    process.join(1.0)
    if process.is_alive() and hasattr(process, "kill"):
        process.kill()
        process.join(1.0)


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
    if target not in _CHILD_TARGETS:
        return {"outcome": "error", "error": f"unsupported child target: {target!r}"}
    if not _is_finite_number(timeout_s) or float(timeout_s) <= 0.0:
        return {"outcome": "error", "error": "child timeout must be a positive finite number"}
    context = multiprocessing.get_context("spawn")
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
            process.join(1.0)
            _terminate_owned_process(process)
            if isinstance(payload, dict):
                return {"outcome": "ok", "payload": payload}
            return {"outcome": "error", "error": "child returned a non-mapping payload"}
        _terminate_owned_process(process)
        return {"outcome": "timeout", "error": f"child exceeded {timeout_s:g}s and was terminated"}
    except KeyboardInterrupt:
        _terminate_owned_process(process)
        return {"outcome": "interrupted", "error": "cancelled by user; owned child terminated"}
    finally:
        parent_conn.close()
        _terminate_owned_process(process)
        if not process.is_alive():
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
    except (AttributeError, KeyError, OverflowError, TypeError, ValueError):
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


def _request_identity_document(request: ComponentRequest) -> dict[str, Any]:
    """Return the location-independent request identity used by resume checks."""
    return {
        "request_id": request.request_id,
        "component_id": request.component_id,
        "sources": [
            {
                "artifact_id": source.artifact_id,
                "uri": source.uri,
                "format": source.format,
            }
            for source in request.sources
        ],
        "required_capabilities": list(request.required_capabilities),
    }


def _config_identity_document(config: ExecuteConfig) -> dict[str, Any]:
    """Return config fields that cannot change during a resume."""
    return {
        "planner": config.planner,
        "seed": config.seed,
        "horizon_steps": config.horizon_steps,
        "robot_speed_m_s": config.robot_speed_m_s,
        "per_execution_timeout_s": config.per_execution_timeout_s,
        "activation_speed_tolerance_m_s": config.activation_speed_tolerance_m_s,
        "motion_epsilon_m": config.motion_epsilon_m,
        "required_component_version": config.required_component_version,
        "intervention_parameters": dict(config.intervention_parameters),
        "recipe_digest": _canonical_digest(config.recipe),
    }


def _config_budget_document(config: ExecuteConfig) -> dict[str, Any]:
    """Return the bounded controls that may be extended on resume."""
    return {
        "max_candidates": config.max_candidates,
        "max_executions": config.max_executions,
        "wall_timeout_s": config.wall_timeout_s,
    }


def _config_document(config: ExecuteConfig) -> dict[str, Any]:
    """Return the complete effective config for provenance and logical hashing."""
    return {
        **_config_identity_document(config),
        **_config_budget_document(config),
        "recipe": dict(config.recipe),
    }


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

    def _budget_remaining(self, required_executions: int = 1) -> bool:
        return (
            required_executions >= 0
            and self._executions_consumed + required_executions <= self.config.max_executions
        )

    def _wall_remaining(self) -> float:
        return max(0.0, self.config.wall_timeout_s - self._elapsed())

    def _reserve_wall_budget(self, required_executions: int) -> None:
        """Require each reserved execution's full timeout before starting a pair."""
        if required_executions <= 0:
            return
        required_wall_s = required_executions * self.config.per_execution_timeout_s
        wall_remaining = self._wall_remaining()
        if wall_remaining < required_wall_s:
            raise _ExecutionWallTimeout(
                "wall_timeout: reserving "
                f"{required_executions} execution(s) requires {required_wall_s:g}s, "
                f"but only {wall_remaining:g}s remains"
            )

    def _record_attempt(self, attempt: dict[str, Any]) -> None:
        self._attempts.append(attempt)

    def _record_progress(self) -> None:
        """Persist attempts and candidate state after every execution boundary."""
        try:
            self._write_ledger()
        except (OSError, TypeError, ValueError) as error:
            raise ReviewExecuteError([f"output_write_failed: {error}"]) from error

    def _candidate_report(self, candidate_id: str) -> dict[str, Any] | None:
        return next(
            (
                report
                for report in self._candidate_reports
                if report.get("intervention_id") == candidate_id
            ),
            None,
        )

    def _attempt(self, candidate_id: str, kind: str) -> dict[str, Any] | None:
        return next(
            (
                entry
                for entry in self._attempts
                if entry.get("candidate_id") == candidate_id and entry.get("kind") == kind
            ),
            None,
        )

    def _record_candidate_report(self, report: dict[str, Any]) -> None:
        """Persist one terminal candidate outcome without duplicating it on resume."""
        candidate_id = report.get("intervention_id")
        if self._candidate_report(str(candidate_id)) is None:
            self._candidate_reports.append(report)
        self._record_progress()

    def _load_resume_ledger(self) -> None:  # noqa: C901, PLR0912, PLR0915
        ledger_path = self.output_dir / "attempt-ledger.json"
        try:
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ReviewExecuteError(
                [f"cannot resume: unreadable attempt ledger: {error}"]
            ) from error
        if not isinstance(ledger, dict):
            raise ReviewExecuteError(["cannot resume: ledger is not a mapping"])
        if ledger.get("request_id") != self.request.request_id:
            raise ReviewExecuteError(["cannot resume: ledger request identity mismatch"])
        if ledger.get("schema_version") != ATTEMPT_LEDGER_SCHEMA_VERSION:
            raise ReviewExecuteError(["cannot resume: ledger schema version mismatch"])
        if ledger.get("component_id") != COMPONENT_ID:
            raise ReviewExecuteError(["cannot resume: ledger component identity mismatch"])
        expected_request_digest = _canonical_digest(_request_identity_document(self.request))
        if ledger.get("request_digest") != expected_request_digest:
            raise ReviewExecuteError(["cannot resume: ledger request identity mismatch"])
        expected_recipe_digest = _canonical_digest(self.recipe)
        if ledger.get("recipe_digest") != expected_recipe_digest:
            raise ReviewExecuteError(["cannot resume: ledger recipe identity mismatch"])
        if ledger.get("recipe_id") != str(self.recipe.get("recipe_id", "")):
            raise ReviewExecuteError(["cannot resume: ledger recipe identity mismatch"])
        if ledger.get("evidence_boundary") != DIAGNOSTIC_EVIDENCE_BOUNDARY:
            raise ReviewExecuteError(["cannot resume: ledger evidence boundary is invalid"])
        if ledger.get("scientific_claim_allowed") is not False:
            raise ReviewExecuteError(["cannot resume: ledger scientific-claim boundary is invalid"])
        if ledger.get("dependent_family_status") != DEPENDENT_FAMILY_STATUS:
            raise ReviewExecuteError(["cannot resume: ledger dependent-family boundary is invalid"])
        if ledger.get("config_identity_digest") != _canonical_digest(
            _config_identity_document(self.config)
        ):
            raise ReviewExecuteError(["cannot resume: ledger immutable config mismatch"])

        prior_budget = ledger.get("budget")
        if not isinstance(prior_budget, dict):
            raise ReviewExecuteError(["cannot resume: ledger budget identity is missing"])
        expected_budget_keys = set(_config_budget_document(self.config))
        if set(prior_budget) != expected_budget_keys:
            raise ReviewExecuteError(["cannot resume: ledger budget identity is malformed"])
        prior_candidates = prior_budget.get("max_candidates")
        prior_executions = prior_budget.get("max_executions")
        prior_wall = prior_budget.get("wall_timeout_s")
        if not _is_int(prior_candidates) or not _is_int(prior_executions):
            raise ReviewExecuteError(["cannot resume: ledger budget identity is malformed"])
        if (
            not _is_finite_number(prior_wall)
            or prior_candidates < 1
            or prior_candidates > MAX_CANDIDATES
            or prior_executions < 1
            or prior_executions > MAX_EXECUTIONS
            or float(prior_wall) <= 0.0
            or float(prior_wall) > MAX_WALL_TIMEOUT_S
        ):
            raise ReviewExecuteError(["cannot resume: ledger budget identity is malformed"])
        if (
            self.config.max_candidates < prior_candidates
            or self.config.max_executions < prior_executions
            or self.config.wall_timeout_s < float(prior_wall)
        ):
            raise ReviewExecuteError(["cannot resume: budget ceilings cannot be reduced"])

        attempts = ledger.get("attempts", [])
        if not isinstance(attempts, list):
            raise ReviewExecuteError(["cannot resume: ledger attempts are malformed"])
        candidate_ids = {
            str(item.get("intervention_id"))
            for item in self.recipe.get("interventions", [])
            if isinstance(item, dict) and isinstance(item.get("intervention_id"), str)
        }
        candidate_factors = {
            item["intervention_id"]: item.get("factor")
            for item in self.recipe.get("interventions", [])
            if isinstance(item, dict) and isinstance(item.get("intervention_id"), str)
        }
        allowed_attempt_statuses = {"ok", "failed", "timed_out", "cancelled"}
        seen_attempts: set[tuple[str, str]] = set()
        validated_attempts: list[dict[str, Any]] = []
        for index, raw_entry in enumerate(attempts):
            if not isinstance(raw_entry, dict):
                raise ReviewExecuteError([f"cannot resume: ledger attempt {index} is malformed"])
            entry = cast("dict[str, Any]", raw_entry)
            candidate_id = entry.get("candidate_id")
            kind = entry.get("kind")
            status = entry.get("status")
            key = (str(candidate_id), str(kind))
            if (
                not isinstance(candidate_id, str)
                or candidate_id not in candidate_ids
                or kind not in {"control", "treatment"}
                or status not in allowed_attempt_statuses
                or key in seen_attempts
            ):
                raise ReviewExecuteError([f"cannot resume: ledger attempt {index} is invalid"])
            elapsed = entry.get("elapsed_s", 0.0)
            if not _is_finite_number(elapsed) or float(elapsed) < 0.0:
                raise ReviewExecuteError(
                    [f"cannot resume: ledger attempt {index} timing is invalid"]
                )
            if status == "ok":
                metrics = entry.get("metrics")
                if not isinstance(metrics, dict):
                    raise ReviewExecuteError(
                        [f"cannot resume: ledger attempt {index} metrics are missing"]
                    )
                metrics = cast("dict[str, Any]", metrics)
                if set(metrics) != REQUIRED_TELEMETRY_METRICS or any(
                    not _is_finite_number(metrics[key]) for key in REQUIRED_TELEMETRY_METRICS
                ):
                    raise ReviewExecuteError(
                        [f"cannot resume: ledger attempt {index} metrics are invalid"]
                    )
            seen_attempts.add(key)
            validated_attempts.append(dict(entry))

        candidate_reports = ledger.get("candidate_reports")
        traces = ledger.get("traces")
        if not isinstance(candidate_reports, list) or not isinstance(traces, list):
            raise ReviewExecuteError(["cannot resume: ledger candidate state is missing"])
        seen_reports: set[str] = set()
        validated_reports: list[dict[str, Any]] = []
        for index, raw_report in enumerate(candidate_reports):
            if not isinstance(raw_report, dict):
                raise ReviewExecuteError([f"cannot resume: candidate report {index} is malformed"])
            report = cast("dict[str, Any]", raw_report)
            candidate_id = report.get("intervention_id")
            if (
                not isinstance(candidate_id, str)
                or candidate_id not in candidate_ids
                or candidate_id in seen_reports
                or report.get("status") not in {"complete", "unavailable", "failed"}
                or report.get("factor") != candidate_factors.get(candidate_id)
            ):
                raise ReviewExecuteError([f"cannot resume: candidate report {index} is invalid"])
            seen_reports.add(candidate_id)
            validated_reports.append(dict(report))
        seen_traces: set[str] = set()
        validated_traces: list[dict[str, Any]] = []
        for index, raw_trace in enumerate(traces):
            if not isinstance(raw_trace, dict):
                raise ReviewExecuteError([f"cannot resume: activation trace {index} is malformed"])
            trace = cast("dict[str, Any]", raw_trace)
            candidate_id = trace.get("intervention_id")
            if (
                trace.get("schema_version") != ACTIVATION_TRACE_SCHEMA_VERSION
                or not isinstance(candidate_id, str)
                or candidate_id not in seen_reports
                or candidate_id in seen_traces
                or trace.get("factor") != candidate_factors.get(candidate_id)
            ):
                raise ReviewExecuteError([f"cannot resume: activation trace {index} is invalid"])
            seen_traces.add(candidate_id)
            validated_traces.append(dict(trace))

        for report in validated_reports:
            candidate_id = report["intervention_id"]
            matching_attempts = [
                entry for entry in validated_attempts if entry["candidate_id"] == candidate_id
            ]
            if report["status"] == "complete" and (
                {entry["kind"] for entry in matching_attempts} != {"control", "treatment"}
                or any(entry["status"] != "ok" for entry in matching_attempts)
                or candidate_id not in seen_traces
                or report.get("nonintervened_config_match") is not True
                or not isinstance(report.get("control_metrics"), dict)
                or not isinstance(report.get("treatment_metrics"), dict)
            ):
                raise ReviewExecuteError(
                    [f"cannot resume: complete candidate {candidate_id} is not reproducible"]
                )
            if any(
                set(report[key]) != REQUIRED_TELEMETRY_METRICS
                or any(
                    not _is_finite_number(report[key][metric])
                    for metric in REQUIRED_TELEMETRY_METRICS
                )
                for key in ("control_metrics", "treatment_metrics")
            ):
                raise ReviewExecuteError(
                    [f"cannot resume: complete candidate {candidate_id} metrics are invalid"]
                )
        if seen_traces != {
            report["intervention_id"]
            for report in validated_reports
            if report["status"] == "complete"
        }:
            raise ReviewExecuteError(["cannot resume: activation traces do not match reports"])
        consumed = ledger.get("executions_consumed", 0)
        if not _is_int(consumed) or consumed != len(validated_attempts) or consumed < 0:
            raise ReviewExecuteError(["cannot resume: ledger execution count is inconsistent"])
        if consumed > self.config.max_executions or consumed > prior_executions:
            raise ReviewExecuteError(["cannot resume: ledger exceeds execution budget"])
        self._attempts = validated_attempts
        self._candidate_reports = validated_reports
        self._traces = validated_traces
        self._executions_consumed = int(consumed)
        elapsed = ledger.get("wall_elapsed_s", 0.0)
        if not _is_finite_number(elapsed) or float(elapsed) < 0.0:
            raise ReviewExecuteError(["cannot resume: ledger wall timing is invalid"])
        self._wall_elapsed_s = float(elapsed)

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
            "request_digest": _canonical_digest(_request_identity_document(self.request)),
            "recipe_digest": _canonical_digest(self.recipe),
            "config_identity_digest": _canonical_digest(_config_identity_document(self.config)),
            "budget": _config_budget_document(self.config),
            "attempts": list(self._attempts),
            "executions_consumed": self._executions_consumed,
            "candidate_reports": list(self._candidate_reports),
            "traces": list(self._traces),
            "evidence_boundary": DIAGNOSTIC_EVIDENCE_BOUNDARY,
            "scientific_claim_allowed": False,
            "dependent_family_status": DEPENDENT_FAMILY_STATUS,
            "wall_elapsed_s": round(self._elapsed(), 3),
        }
        return _write_json(self.output_dir / "attempt-ledger.json", payload)

    def _run_episode(self, candidate_id: str, kind: str, spec: dict[str, Any]) -> dict[str, Any]:
        if not self._budget_remaining():
            raise _ExecutionBudgetExhausted("execution_budget_exhausted: no execution slot remains")
        wall_remaining = self._wall_remaining()
        if wall_remaining <= 0.0:
            raise _ExecutionWallTimeout("wall_timeout: wall budget exhausted")
        job = {
            "seed": self.config.seed,
            "horizon_steps": self.config.horizon_steps,
            "robot_speed_m_s": self.config.robot_speed_m_s,
            "scenario_id": spec["scenario_id"],
            "source_ref": dict(spec["source_ref"]),
            "ped_speed_m_s": spec["ped_speed_m_s"],
            "ped_start_delay_s": spec["ped_start_delay_s"],
        }
        started = time.monotonic()
        child_timeout = min(self.config.per_execution_timeout_s, wall_remaining)
        try:
            outcome = _run_owned_child(job, child_timeout)
        except Exception as error:  # noqa: BLE001 - convert child boundary failures to a result
            outcome = {"outcome": "error", "error": f"{type(error).__name__}: {error}"}
        elapsed = time.monotonic() - started
        self._executions_consumed += 1
        if not isinstance(outcome, dict):
            outcome = {"outcome": "error", "error": "child returned a non-mapping outcome"}
        outcome_kind = outcome.get("outcome")
        if outcome_kind == "ok":
            child_payload = outcome["payload"]
            metrics = _telemetry_metrics(
                child_payload if isinstance(child_payload, dict) else {},
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
            self._record_progress()
            return attempt
        if outcome_kind == "timeout":
            timeout_reason = (
                "wall_timeout"
                if child_timeout < self.config.per_execution_timeout_s
                else "per_execution_timeout"
            )
            attempt = {
                "candidate_id": candidate_id,
                "kind": kind,
                "status": "timed_out",
                "reason": f"{timeout_reason}: {outcome.get('error', '')}",
                "elapsed_s": round(elapsed, 3),
            }
            self._record_attempt(attempt)
            self._record_progress()
            if timeout_reason == "wall_timeout":
                raise _ExecutionWallTimeout(attempt["reason"])
            raise _ExecutionTimeout(attempt["reason"])
        if outcome_kind == "interrupted":
            attempt = {
                "candidate_id": candidate_id,
                "kind": kind,
                "status": "cancelled",
                "reason": f"cancelled_by_user: {outcome.get('error', '')}",
                "elapsed_s": round(elapsed, 3),
            }
            self._record_attempt(attempt)
            self._record_progress()
            raise _ExecutionCancelled(attempt["reason"])
        attempt = {
            "candidate_id": candidate_id,
            "kind": kind,
            "status": "failed",
            "reason": f"execution_error: {outcome.get('error', '')}",
            "elapsed_s": round(elapsed, 3),
        }
        self._record_attempt(attempt)
        self._record_progress()
        return attempt

    def _control_fidelity_ok(self, metrics: dict[str, Any]) -> tuple[bool, str]:
        if metrics["ped_displacement_m"] <= self.config.motion_epsilon_m:
            return False, "control pedestrian shows no measured motion"
        if metrics["robot_displacement_m"] <= self.config.motion_epsilon_m:
            return False, "control robot shows no measured motion"
        return True, ""

    def _execute_candidate(  # noqa: C901, PLR0912, PLR0915
        self, candidate: dict[str, Any], measurement: dict[str, Any]
    ) -> dict[str, Any]:
        candidate_id = str(candidate["intervention_id"])
        factor = str(candidate["factor"])
        existing_report = self._candidate_report(candidate_id)
        if existing_report is not None:
            return existing_report
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
            self._record_candidate_report(report)
            return report
        source_identity = self.recipe["source_identity"]
        control_spec = {
            "scenario_id": str(source_identity["scenario_id"]),
            "source_ref": dict(source_identity["source_ref"]),
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
        control_attempt = self._attempt(candidate_id, "control")
        treatment_attempt = self._attempt(candidate_id, "treatment")
        if treatment_attempt is not None and control_attempt is None:
            raise ReviewExecuteError(["resume ledger has treatment without control"])
        required_executions = 0
        if control_attempt is None:
            required_executions = 2
        elif control_attempt.get("status") == "ok" and treatment_attempt is None:
            required_executions = 1
        if not self._budget_remaining(required_executions):
            raise _ExecutionBudgetExhausted(
                "execution_budget_exhausted: reserving a complete control/treatment pair"
            )
        self._reserve_wall_budget(required_executions)
        if control_attempt is None:
            control_attempt = self._run_episode(candidate_id, "control", control_spec)
        if control_attempt.get("status") in {"timed_out", "cancelled"}:
            if not self.resume:
                raise ReviewExecuteError([str(control_attempt.get("reason", "control failed"))])
            report = {
                "intervention_id": candidate_id,
                "factor": factor,
                "status": "failed",
                "reason": f"control execution did not complete: {control_attempt.get('reason', '')}",
                "nonintervened_config_match": True,
            }
            self._record_candidate_report(report)
            return report
        if control_attempt.get("status") != "ok":
            report = {
                "intervention_id": candidate_id,
                "factor": factor,
                "status": "failed",
                "reason": f"control execution failed: {control_attempt.get('reason', '')}",
                "nonintervened_config_match": True,
            }
            self._record_candidate_report(report)
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
            self._record_candidate_report(report)
            return report
        treatment_attempt = self._attempt(candidate_id, "treatment")
        if treatment_attempt is None:
            self._reserve_wall_budget(1)
            treatment_attempt = self._run_episode(candidate_id, "treatment", treatment_spec)
        if treatment_attempt.get("status") in {"timed_out", "cancelled"}:
            if not self.resume:
                raise ReviewExecuteError([str(treatment_attempt.get("reason", "treatment failed"))])
            report = {
                "intervention_id": candidate_id,
                "factor": factor,
                "status": "failed",
                "reason": f"treatment execution did not complete: {treatment_attempt.get('reason', '')}",
                "control_metrics": control_metrics,
                "nonintervened_config_match": True,
            }
            self._record_candidate_report(report)
            return report
        if treatment_attempt.get("status") != "ok":
            report = {
                "intervention_id": candidate_id,
                "factor": factor,
                "status": "failed",
                "reason": f"treatment execution failed: {treatment_attempt.get('reason', '')}",
                "control_metrics": control_metrics,
                "nonintervened_config_match": True,
            }
            self._record_candidate_report(report)
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
        self._record_progress()
        return report


class _ExecutionBudgetExhausted(Exception):
    """Internal signal: no complete control/treatment pair fits the budget."""


class _ExecutionWallTimeout(Exception):
    """Internal signal: the bounded wall-clock budget was exhausted."""


class _ExecutionTimeout(Exception):
    """Internal signal: an owned child exceeded its execution timeout."""


class _ExecutionCancelled(Exception):
    """Internal signal: execution was cancelled; the owned child was terminated."""


def _measurement_for_recipe(recipe: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    measurements = recipe.get("measurements", [])
    if not isinstance(measurements, list) or not measurements:
        return None, "recipe carries no measurements"
    if len(measurements) != 1:
        return None, "exactly one driving measurement is supported"
    first = measurements[0]
    if not isinstance(first, dict):
        return None, "driving measurement must be a mapping"
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
        "evidence_boundary": DIAGNOSTIC_EVIDENCE_BOUNDARY,
        "benchmark_success": False,
        "scientific_claim_allowed": False,
        "dependent_family_status": DEPENDENT_FAMILY_STATUS,
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
        "evidence_boundary": DIAGNOSTIC_EVIDENCE_BOUNDARY,
        "benchmark_success": False,
        "scientific_claim_allowed": False,
        "dependent_family_status": DEPENDENT_FAMILY_STATUS,
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
        "evidence_boundary": DIAGNOSTIC_EVIDENCE_BOUNDARY,
        "scientific_claim_allowed": False,
        "dependent_family_status": DEPENDENT_FAMILY_STATUS,
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
        "config_digest": _canonical_digest(_config_document(executor.config)),
        "source_identity": dict(executor.recipe.get("source_identity", {})),
        "retrieval_destination": str(executor.recipe.get("preservation_destination", "")),
        "evidence_boundary": DIAGNOSTIC_EVIDENCE_BOUNDARY,
        "benchmark_success": False,
        "scientific_claim_allowed": False,
        "dependent_family_status": DEPENDENT_FAMILY_STATUS,
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


def _admit_request(  # noqa: C901
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
    recipe_errors = _validate_recipe_execution_contract(recipe_doc, config)
    if recipe_errors:
        return (
            None,
            None,
            None,
            _final_result(request, status="failed", reason=recipe_errors[0]),
        )
    request_source_ref = _request_source_reference(request)
    if request_source_ref is None:
        return (
            None,
            None,
            None,
            _final_result(
                request,
                status="failed",
                reason="invalid_source_identity: exactly one source reference is required",
            ),
        )
    if request_source_ref != _supported_fixture_source_reference():
        return (
            None,
            None,
            None,
            _final_result(
                request,
                status="failed",
                reason=(
                    "invalid_source_identity: request source must match the immutable "
                    "supported fixture source reference"
                ),
            ),
        )
    if source_identity.get("source_ref") != request_source_ref:
        return (
            None,
            None,
            None,
            _final_result(
                request,
                status="failed",
                reason="invalid_source_identity: recipe source_ref does not match request source",
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


def _prepare_output_dir(  # noqa: C901
    request: ComponentRequest, root: Path, *, resume: bool
) -> tuple[Path | None, ComponentResult | None]:
    """Resolve the output directory honoring collision and resume semantics.

    Returns:
        Tuple of (output directory, early result); exactly one is set.
    """
    if not isinstance(request.output_directory, str):
        return None, _final_result(
            request, status="failed", reason="invalid_output_path: string required"
        )
    relative = Path(request.output_directory)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or not relative.parts
        or relative == Path(".")
    ):
        return None, _final_result(
            request,
            status="failed",
            reason="invalid_output_path: relative component directory required",
        )
    try:
        root = root.resolve(strict=True)
        output_dir = root / relative
        if any(
            root.joinpath(*relative.parts[:index]).is_symlink()
            for index in range(1, len(relative.parts) + 1)
        ):
            return None, _final_result(
                request,
                status="failed",
                reason="invalid_output_path: output path contains a symlinked component",
            )
        resolved_output = output_dir.resolve(strict=False)
        resolved_output.relative_to(root)
    except (OSError, ValueError) as error:
        return None, _final_result(
            request,
            status="failed",
            reason=f"invalid_output_path: output escapes its base: {error}",
        )
    if output_dir.is_symlink():
        return None, _final_result(
            request, status="failed", reason="invalid_output_path: output directory is a symlink"
        )
    if output_dir.exists():
        if not output_dir.is_dir():
            return None, _final_result(
                request,
                status="failed",
                reason=f"output_collision: output path is not a directory: {request.output_directory}",
            )
        if not resume:
            return None, _final_result(
                request,
                status="failed",
                reason=f"output_collision: output already exists: {request.output_directory}",
            )
        try:
            unsafe_children = [entry.name for entry in output_dir.iterdir() if entry.is_symlink()]
        except OSError as error:
            return None, _final_result(
                request, status="failed", reason=f"unreadable output directory: {error}"
            )
        if unsafe_children:
            return None, _final_result(
                request,
                status="failed",
                reason="invalid_output_path: output contains symlinked artifacts",
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
            if executor._candidate_report(str(candidate["intervention_id"])) is not None:
                continue
            executor._execute_candidate(candidate, measurement)
    except _ExecutionBudgetExhausted as error:
        return "partial", str(error)
    except _ExecutionWallTimeout as error:
        return "partial", str(error)
    except _ExecutionTimeout as error:
        return "partial", f"per_execution_timeout: {error}; owned child terminated"
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
    selected_ids = {
        str(candidate["intervention_id"])
        for candidate in _select_candidates(executor.recipe, executor.config.max_candidates)
    }
    reports_by_id = {
        str(report.get("intervention_id")): report for report in executor._candidate_reports
    }
    unresolved = sorted(selected_ids - set(reports_by_id))
    succeeded = [
        report for report in executor._candidate_reports if report.get("status") == "complete"
    ]
    failed = [report for report in executor._candidate_reports if report.get("status") == "failed"]
    if status == "complete" and not succeeded:
        if failed:
            status = "failed"
            reason = "candidate_execution_failed: " + "; ".join(
                str(report.get("reason", "failed")) for report in failed
            )
        else:
            status = "unavailable"
            reason = "no candidate completed: " + "; ".join(
                str(report.get("reason", report.get("status", "")))
                for report in executor._candidate_reports
            )
    elif status == "complete" and unresolved:
        status = "partial"
        reason = f"incomplete_candidates: {', '.join(unresolved)}"
    elif status == "complete" and failed:
        status = "partial"
        reason = "candidate_execution_failed: " + "; ".join(
            str(report.get("reason", "failed")) for report in failed
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
    except (OSError, TypeError, ValueError) as error:
        reason = f"{reason}; output_write_failed: {error}"
        status = "failed"
    if status == "partial" and not executor._candidate_reports and not executor._attempts:
        status = "failed"
        reason = f"{reason}; no candidate or execution reached a terminal state"
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
    provenance["request_digest"] = _canonical_digest(_request_identity_document(request))
    provenance["recipe_digest"] = _canonical_digest(recipe_doc)
    provenance["config_digest"] = _canonical_digest(_config_document(config))
    provenance["sources"] = [
        {
            "artifact_id": source.artifact_id,
            "uri": source.uri,
            "format": source.format,
        }
        for source in request.sources
    ]
    provenance["output_directory"] = request.output_directory
    executor._started_at = time.monotonic()
    try:
        status, reason = _drive_candidates(executor, config, measurement)
        return _settle(executor, provenance, status, reason)
    except Exception as error:  # noqa: BLE001 - fail closed while preserving the ledger
        reason = f"execution_failed: {type(error).__name__}: {error}"
        try:
            executor._write_ledger()
        except (OSError, TypeError, ValueError) as ledger_error:
            reason += f"; output_write_failed: {ledger_error}"
        return _final_result(
            request,
            status="failed",
            reason=reason,
            diagnostics=tuple(_diagnostics(executor)),
            provenance=provenance,
        )


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
    if not isinstance(payload, dict):
        raise ReviewExecuteError(["request JSON must be a mapping"])
    if args.config is not None:
        try:
            config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ReviewExecuteError([f"cannot read config: {error}"]) from error
        if not isinstance(config, dict):
            raise ReviewExecuteError(["config JSON must be a mapping"])
        request_config = payload.get("config", {})
        if not isinstance(request_config, dict):
            raise ReviewExecuteError(["request config must be a mapping"])
        payload = {**payload, "config": {**request_config, **config}}
    payload = {**payload, "output_directory": args.output}
    request = component_request_from_dict(payload, source=args.input)
    result = run(
        request, base=Path(args.base) if args.base is not None else None, resume=args.resume
    )
    print(json.dumps(asdict(result), sort_keys=True, indent=2))  # noqa: T201 - CLI output
    return 0 if result.status == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
