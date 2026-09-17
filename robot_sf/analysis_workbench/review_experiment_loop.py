"""Finite, journaled follow-up experiment loop (SREV-24, issue #9296).

The SREV-22 executor owns simulator execution and source admission.  This
module owns the smaller product-level concern around it: an explicitly
authorised finite session, deterministic candidate ordering, pair reservation,
crash-safe progress, and truthful retention of every outcome.  The loop never
turns a recipe into an authority source.  In particular, a source root,
receipt, recipe, identity, or budget may only come from the caller and the
admitted SREV contracts.

``run`` is the standalone component/CLI surface.  ``ExperimentLoop`` is also
usable with a small injected executor in tests or by an offline caller.  An
injected executor receives stable operation IDs and must make those IDs
idempotent.  The default native adapter delegates to
:mod:`review_execute`, preserving its no-follow source-admission boundary and
its real supported fixture execution path.

This is diagnostic tooling only.  A successful loop is not benchmark,
scientific, causal, or paper evidence.
"""

# The loop deliberately keeps its state-machine transitions together: splitting
# each transition into a helper would make the crash/recovery contract harder to
# audit than the small amount of local branching.
# ruff: noqa: C901, PLR0912, PLR0913, PLR0915, T201

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import platform
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast

from robot_sf.analysis_workbench import review_execute
from robot_sf.analysis_workbench.review_contracts import (
    COMPONENT_REQUEST_SCHEMA_VERSION,
    COMPONENT_RESULT_SCHEMA_VERSION,
    ComponentRequest,
    ComponentResult,
    ReviewContractsValidationError,
    component_request_from_dict,
    component_result_from_dict,
    experiment_recipe_canonical_digest,
    experiment_recipe_from_dict,
)
from robot_sf.analysis_workbench.review_experiment_report import OUTCOMES as RECORDED_OUTCOMES
from robot_sf.analysis_workbench.review_hypotheses import (
    DEFAULT_MAX_CANDIDATES as HYPOTHESIS_MAX_CANDIDATES,
)
from robot_sf.analysis_workbench.review_hypotheses import (
    DEFAULT_MAX_CONCURRENT_LOCAL_CPU_PROCESSES as HYPOTHESIS_MAX_PROCESSES,
)
from robot_sf.analysis_workbench.review_hypotheses import (
    DEFAULT_MAX_ELAPSED_SECONDS as HYPOTHESIS_MAX_ELAPSED_SECONDS,
)
from robot_sf.analysis_workbench.review_hypotheses import (
    DEFAULT_MAX_SIMULATOR_EXECUTIONS as HYPOTHESIS_MAX_EXECUTIONS,
)
from robot_sf.analysis_workbench.review_hypotheses import (
    RESERVED_CONTROL_TREATMENT_PAIR,
)
from robot_sf.benchmark.counterfactual_pair import PairHypothesis, evaluate_counterfactual_pair
from robot_sf.benchmark.research_answerability import (
    AnswerabilityContractError,
    AnswerabilityResult,
    evaluate_answerability,
)

COMPONENT_ID = "srev24-review-experiment-loop"
COMPONENT_VERSION = "1.0.0"
COMPONENT_DESCRIPTOR_SCHEMA_VERSION = "component-descriptor.v1"
SESSION_JOURNAL_SCHEMA_VERSION = "experiment-loop-session.v1"
LOOP_REPORT_SCHEMA_VERSION = "experiment-loop-report.v1"
SESSION_JOURNAL_FILENAME = "experiment-loop-journal.json"
LEGACY_SESSION_JOURNAL_FILENAME = "session-journal.json"
LOOP_REPORT_FILENAME = "experiment-loop-report.json"
# Short aliases are kept for callers that consume the versioned component
# without depending on artifact filename spelling.
JOURNAL_SCHEMA_VERSION = SESSION_JOURNAL_SCHEMA_VERSION
REPORT_SCHEMA_VERSION = LOOP_REPORT_SCHEMA_VERSION

DEFAULT_MAX_CANDIDATES = HYPOTHESIS_MAX_CANDIDATES
DEFAULT_MAX_EXECUTIONS = HYPOTHESIS_MAX_EXECUTIONS
DEFAULT_MAX_ELAPSED_SECONDS = float(HYPOTHESIS_MAX_ELAPSED_SECONDS)
DEFAULT_MAX_CONCURRENT_LOCAL_CPU_PROCESSES = HYPOTHESIS_MAX_PROCESSES
# Public names mirror the SREV-21 recipe vocabulary while keeping the shorter
# names convenient for the loop API.
DEFAULT_MAX_SIMULATOR_EXECUTIONS = DEFAULT_MAX_EXECUTIONS
MAX_CANDIDATES = DEFAULT_MAX_CANDIDATES
MAX_EXECUTIONS = DEFAULT_MAX_EXECUTIONS
MAX_WALL_TIMEOUT_S = DEFAULT_MAX_ELAPSED_SECONDS
DEFAULT_MAX_RETRIES = 0
MAX_MAX_RETRIES = 3
DIAGNOSTIC_EVIDENCE_BOUNDARY = "diagnostic_only"
DEPENDENT_FAMILY_STATUS = "standalone_fixture_only"

OUTCOMES = tuple(outcome for outcome in RECORDED_OUTCOMES if outcome != "contradictory")
NEGATIVE_OUTCOME_STATUSES = ("failed", "unavailable", "cancelled")
TERMINAL_CANDIDATE_STATES = frozenset({"complete", "failed", "unavailable", "cancelled"})
SUPPORTED_STOP_RULES = frozenset(
    {
        "exhausted_candidates",
        "execution_budget_exhausted",
        "wall_timeout",
        "control_fidelity_failure_blocks_treatment",
        "cancellation_requested",
    }
)
_RESUMABLE_STOP_REASONS = frozenset({"execution_budget_exhausted", "wall_timeout"})

# These controls may be increased on an explicit resume, but never reduced.
# The consumed execution count and elapsed time stay in the journal, so an
# extension is a continuation rather than a fresh budget or child session.
_RESUMABLE_BUDGET_KEYS = frozenset(
    {"max_candidates", "max_executions", "wall_timeout_s", "max_retries"}
)

_ALLOWED_CONFIG_KEYS = frozenset(
    {
        "recipe",
        "executor_config",
        "autonomous",
        "read_only",
        "mode",
        "max_candidates",
        "max_executions",
        "wall_timeout_s",
        "max_retries",
        "session_id",
        "answerability",
        "cancel_requested",
        "source_admission",
    }
)
_SUPPORTED_FACTORS = frozenset(review_execute.SUPPORTED_FACTORS)
_SUPPORTED_MEASUREMENTS = frozenset(review_execute.SUPPORTED_MEASUREMENTS)


class ExperimentLoopError(ValueError):
    """Raised for an invalid loop contract or an unsafe resume envelope."""


class OperationExecutor(Protocol):
    """Minimal injected executor protocol used by :class:`ExperimentLoop`."""

    def execute(self, **kwargs: Any) -> Mapping[str, Any] | ComponentResult:
        """Execute one idempotent control or treatment operation."""


@dataclass(frozen=True, slots=True)
class LoopBudget:
    """Hard bounded session budget."""

    max_candidates: int = DEFAULT_MAX_CANDIDATES
    max_executions: int = DEFAULT_MAX_EXECUTIONS
    wall_timeout_s: float = DEFAULT_MAX_ELAPSED_SECONDS
    max_concurrent_local_cpu_processes: int = DEFAULT_MAX_CONCURRENT_LOCAL_CPU_PROCESSES
    max_retries: int = DEFAULT_MAX_RETRIES

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe budget document."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class LoopPolicy:
    """Caller-controlled start and cancellation policy."""

    autonomous: bool = False
    read_only: bool = False
    cancel_requested: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return the policy identity retained in the journal."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class _ValidatedInput:
    recipe: dict[str, Any]
    budget: LoopBudget
    policy: LoopPolicy
    session_id: str
    executor_config: dict[str, Any]
    answerability: dict[str, Any] | None


def descriptor() -> dict[str, Any]:
    """Return the standalone SREV-24 component descriptor."""

    return {
        "schema_version": COMPONENT_DESCRIPTOR_SCHEMA_VERSION,
        "component_id": COMPONENT_ID,
        "component_version": COMPONENT_VERSION,
        "supported_input_versions": [COMPONENT_REQUEST_SCHEMA_VERSION],
        "required_capabilities": [],
        "optional_capabilities": ["bounded-execution", "recorded-experiment-results"],
        "output_types": [LOOP_REPORT_SCHEMA_VERSION, SESSION_JOURNAL_SCHEMA_VERSION],
    }


def _canonical_digest(value: Any) -> str:
    """Hash strict JSON values using one location-independent encoding.

    Returns:
        A hexadecimal SHA-256 digest.
    """

    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise ExperimentLoopError(f"strict JSON required: {error}") from error
    return hashlib.sha256(encoded).hexdigest()


def _config_identity_document(
    request_config: Mapping[str, Any], policy: Mapping[str, Any]
) -> dict[str, Any]:
    """Return the immutable part of one request for resume validation.

    Session ceilings are intentionally excluded: an explicit resume may
    extend them within the same recipe's hard limits.  Cancellation is also a
    per-invocation control, allowing a caller to cancel a persisted session
    without changing its execution identity.  Nested executor configuration
    remains immutable, including any child-specific limits.
    """

    config = dict(request_config)
    for key in _RESUMABLE_BUDGET_KEYS | {"cancel_requested"}:
        config.pop(key, None)
    policy_identity = {key: policy.get(key, False) for key in ("autonomous", "read_only")}
    return {"request_config": config, "policy": policy_identity}


def _json_bytes(payload: Any) -> bytes:
    """Encode one artifact with deterministic strict JSON bytes.

    Returns:
        UTF-8 JSON bytes terminated by one newline.
    """

    try:
        return (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(
            "utf-8"
        )
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise ExperimentLoopError(f"strict JSON required: {error}") from error


def _atomic_write_json(path: Path, payload: Any) -> str:
    """Atomically persist JSON and return its file-byte digest.

    The temporary file is fsynced before replacement and the parent directory
    is fsynced where the platform permits it.  A journal write is therefore a
    committed state transition, not merely a best-effort status message.

    Returns:
        A hexadecimal digest of the committed file bytes.
    """

    content = _json_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or path.is_symlink():
        raise OSError(f"refusing to write through symlink: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.is_symlink():
        raise OSError(f"refusing to write through symlink: {temporary}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    file_descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        file_descriptor = -1
        os.replace(temporary, path)
        try:
            directory_descriptor = os.open(path.parent, os.O_RDONLY)
        except OSError:
            directory_descriptor = -1
        if directory_descriptor >= 0:
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
    finally:
        if file_descriptor >= 0:
            os.close(file_descriptor)
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(content).hexdigest()


def _safe_text(value: Any, *, field_name: str, maximum: int = 256) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExperimentLoopError(f"{field_name} must be a non-empty string")
    if len(value) > maximum or "\x00" in value:
        raise ExperimentLoopError(f"{field_name} is too long or contains a NUL byte")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ExperimentLoopError(f"{field_name} contains invalid Unicode") from error
    return value


def _finite_float(value: Any, *, field_name: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExperimentLoopError(f"{field_name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ExperimentLoopError(f"{field_name} must be finite and >= {minimum:g}")
    return result


def _bounded_int(value: Any, *, field_name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ExperimentLoopError(f"{field_name} must be an integer")
    if value < minimum or value > maximum:
        raise ExperimentLoopError(f"{field_name} must be within {minimum}..{maximum}")
    return int(value)


def _request_identity(request: ComponentRequest) -> dict[str, Any]:
    """Return the location-independent request identity retained in a journal."""

    return {
        "request_id": request.request_id,
        "component_id": request.component_id,
        "sources": [
            {
                "artifact_id": source.artifact_id,
                "uri": source.uri,
                "format": source.format,
                "schema": source.schema,
                "sha256": source.sha256,
                "source_commit": source.source_commit,
                "config_identity": source.config_identity,
                "units": source.units,
                "coordinate_frame": source.coordinate_frame,
            }
            for source in request.sources
        ],
        "required_capabilities": list(request.required_capabilities),
    }


def _candidate_order(recipe: Mapping[str, Any]) -> list[dict[str, Any]]:
    interventions = recipe.get("interventions")
    if not isinstance(interventions, list) or not interventions:
        raise ExperimentLoopError("invalid_recipe: interventions must be a non-empty list")
    seen: set[str] = set()
    candidates: list[dict[str, Any]] = []
    for index, raw in enumerate(interventions):
        if not isinstance(raw, Mapping):
            raise ExperimentLoopError(f"invalid_recipe: intervention {index} must be a mapping")
        candidate = dict(raw)
        candidate_id = _safe_text(candidate.get("intervention_id"), field_name="intervention_id")
        if candidate_id in seen:
            raise ExperimentLoopError(f"invalid_recipe: duplicate intervention_id {candidate_id}")
        seen.add(candidate_id)
        priority = candidate.get("priority", 0)
        if isinstance(priority, bool) or not isinstance(priority, int) or priority < 0:
            raise ExperimentLoopError(
                f"invalid_recipe: candidate {candidate_id} priority must be non-negative integer"
            )
        candidate["priority"] = int(priority)
        candidate["intervention_id"] = candidate_id
        candidates.append(candidate)
    return sorted(candidates, key=lambda item: (int(item["priority"]), item["intervention_id"]))


def _budget_from_config(config: Mapping[str, Any], recipe: Mapping[str, Any]) -> LoopBudget:
    recipe_budget = recipe.get("budget")
    if not isinstance(recipe_budget, Mapping):
        raise ExperimentLoopError("invalid_budget: recipe budget must be a mapping")
    recipe_candidates = _bounded_int(
        recipe_budget.get(
            "max_candidates",
            recipe_budget.get("max_candidate_interventions", DEFAULT_MAX_CANDIDATES),
        ),
        field_name="recipe.budget.max_candidates",
        minimum=1,
        maximum=DEFAULT_MAX_CANDIDATES,
    )
    recipe_executions = _bounded_int(
        recipe_budget.get(
            "max_executions",
            recipe_budget.get("max_simulator_executions", DEFAULT_MAX_EXECUTIONS),
        ),
        field_name="recipe.budget.max_executions",
        minimum=1,
        maximum=DEFAULT_MAX_EXECUTIONS,
    )
    recipe_wall = _finite_float(
        recipe_budget.get(
            "wall_timeout_s",
            recipe_budget.get("max_elapsed_seconds", DEFAULT_MAX_ELAPSED_SECONDS),
        ),
        field_name="recipe.budget.wall_timeout_s",
        minimum=0.001,
    )
    if recipe_wall > DEFAULT_MAX_ELAPSED_SECONDS:
        raise ExperimentLoopError("invalid_budget: recipe wall timeout exceeds 600 seconds")
    max_candidates = _bounded_int(
        config.get("max_candidates", recipe_candidates),
        field_name="max_candidates",
        minimum=1,
        maximum=DEFAULT_MAX_CANDIDATES,
    )
    max_executions = _bounded_int(
        config.get("max_executions", recipe_executions),
        field_name="max_executions",
        minimum=1,
        maximum=DEFAULT_MAX_EXECUTIONS,
    )
    wall_timeout = _finite_float(
        config.get("wall_timeout_s", recipe_wall), field_name="wall_timeout_s", minimum=0.001
    )
    max_retries = _bounded_int(
        config.get("max_retries", DEFAULT_MAX_RETRIES),
        field_name="max_retries",
        minimum=0,
        maximum=MAX_MAX_RETRIES,
    )
    if max_candidates > recipe_candidates:
        raise ExperimentLoopError("invalid_budget: max_candidates exceeds recipe budget")
    if max_executions > recipe_executions:
        raise ExperimentLoopError("invalid_budget: max_executions exceeds recipe budget")
    if wall_timeout > recipe_wall:
        raise ExperimentLoopError("invalid_budget: wall_timeout_s exceeds recipe budget")
    return LoopBudget(
        max_candidates=max_candidates,
        max_executions=max_executions,
        wall_timeout_s=wall_timeout,
        max_concurrent_local_cpu_processes=DEFAULT_MAX_CONCURRENT_LOCAL_CPU_PROCESSES,
        max_retries=max_retries,
    )


def _validate_loop_budget(recipe: Mapping[str, Any], budget: LoopBudget) -> None:
    """Enforce hard and recipe ceilings for direct ``ExperimentLoop`` callers."""

    recipe_budget = recipe.get("budget")
    if not isinstance(recipe_budget, Mapping):
        raise ExperimentLoopError("invalid_budget: recipe budget must be a mapping")
    recipe_candidates = recipe_budget.get(
        "max_candidates",
        recipe_budget.get("max_candidate_interventions", DEFAULT_MAX_CANDIDATES),
    )
    recipe_executions = recipe_budget.get(
        "max_executions",
        recipe_budget.get("max_simulator_executions", DEFAULT_MAX_EXECUTIONS),
    )
    recipe_wall = recipe_budget.get(
        "wall_timeout_s",
        recipe_budget.get("max_elapsed_seconds", DEFAULT_MAX_ELAPSED_SECONDS),
    )
    _bounded_int(
        budget.max_candidates,
        field_name="budget.max_candidates",
        minimum=1,
        maximum=DEFAULT_MAX_CANDIDATES,
    )
    _bounded_int(
        budget.max_executions,
        field_name="budget.max_executions",
        minimum=1,
        maximum=DEFAULT_MAX_EXECUTIONS,
    )
    _finite_float(
        budget.wall_timeout_s,
        field_name="budget.wall_timeout_s",
        minimum=0.001,
    )
    if float(budget.wall_timeout_s) > DEFAULT_MAX_ELAPSED_SECONDS:
        raise ExperimentLoopError("invalid_budget: wall timeout exceeds 600 seconds")
    _bounded_int(
        budget.max_retries,
        field_name="budget.max_retries",
        minimum=0,
        maximum=MAX_MAX_RETRIES,
    )
    if budget.max_concurrent_local_cpu_processes != DEFAULT_MAX_CONCURRENT_LOCAL_CPU_PROCESSES:
        raise ExperimentLoopError("invalid_budget: only one local CPU process is supported")
    if (
        not isinstance(recipe_candidates, int)
        or isinstance(recipe_candidates, bool)
        or budget.max_candidates > recipe_candidates
        or not isinstance(recipe_executions, int)
        or isinstance(recipe_executions, bool)
        or budget.max_executions > recipe_executions
        or not isinstance(recipe_wall, (int, float))
        or isinstance(recipe_wall, bool)
        or not math.isfinite(float(recipe_wall))
        or budget.wall_timeout_s > float(recipe_wall)
    ):
        raise ExperimentLoopError("invalid_budget: direct loop budget exceeds recipe budget")


def _validate_input(
    request: ComponentRequest,
    *,
    autonomous: bool,
    read_only: bool,
    resume: bool,
) -> _ValidatedInput:
    config = request.config
    if not isinstance(config, Mapping):
        raise ExperimentLoopError("invalid_config: request config must be a mapping")
    unknown = sorted(str(key) for key in config if key not in _ALLOWED_CONFIG_KEYS)
    if unknown:
        raise ExperimentLoopError(
            "invalid_config: unknown keys are rejected: " + ", ".join(unknown)
        )
    recipe = config.get("recipe")
    if not isinstance(recipe, Mapping):
        raise ExperimentLoopError(
            "corrupt_recipe: config.recipe must be an experiment-recipe mapping"
        )
    try:
        validated = experiment_recipe_from_dict(dict(recipe), source="config.recipe")
    except ReviewContractsValidationError as error:
        raise ExperimentLoopError("corrupt_recipe: " + "; ".join(error.errors)) from error
    recipe_document = dict(validated.document)
    # The hard ceiling is a candidate-session limit, not a requirement to
    # mutate the recipe; selected candidates are the deterministic prefix below
    # the declared ceiling.
    _candidate_order(recipe_document)
    budget = _budget_from_config(config, recipe_document)
    mode = config.get("mode", "autonomous")
    if mode not in {"autonomous", "read_only"}:
        raise ExperimentLoopError("invalid_config: mode must be autonomous or read_only")
    effective_autonomous = bool(autonomous or config.get("autonomous", False))
    effective_read_only = bool(read_only or config.get("read_only", False) or mode == "read_only")
    if effective_read_only:
        effective_autonomous = False
    if not isinstance(config.get("autonomous", False), bool):
        raise ExperimentLoopError("invalid_config: autonomous must be boolean")
    if not isinstance(config.get("read_only", False), bool):
        raise ExperimentLoopError("invalid_config: read_only must be boolean")
    if not isinstance(config.get("cancel_requested", False), bool):
        raise ExperimentLoopError("invalid_config: cancel_requested must be boolean")
    if "session_id" in config:
        session_id = _safe_text(config["session_id"], field_name="session_id")
    else:
        session_id = f"{request.request_id}:{_canonical_digest(recipe_document)[:16]}"
    executor_config = config.get("executor_config", {})
    if not isinstance(executor_config, Mapping):
        raise ExperimentLoopError("invalid_config: executor_config must be a mapping")
    answerability = config.get("answerability")
    if answerability is not None:
        if not isinstance(answerability, Mapping):
            raise ExperimentLoopError("invalid_config: answerability must be a mapping")
        answerability = dict(answerability)
        try:
            evaluate_answerability(answerability)
        except (AnswerabilityContractError, KeyError, TypeError, ValueError) as error:
            raise ExperimentLoopError(f"invalid_answerability: {error}") from error
    del resume  # resume is used by the journal owner; validation is stateless.
    return _ValidatedInput(
        recipe=recipe_document,
        budget=budget,
        policy=LoopPolicy(
            autonomous=effective_autonomous,
            read_only=effective_read_only,
            cancel_requested=bool(config.get("cancel_requested", False)),
        ),
        session_id=session_id,
        executor_config=dict(executor_config),
        answerability=answerability,
    )


def _answerability_document(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    result: AnswerabilityResult = evaluate_answerability(value)
    return result.as_dict()


def _status_from_result(value: Any) -> tuple[str, dict[str, Any]]:
    """Normalize fake/native operation output without losing retained fields.

    Returns:
        A normalized status and retained result mapping.
    """

    if isinstance(value, ComponentResult):
        payload = asdict(value)
    elif isinstance(value, Mapping):
        payload = dict(value)
    else:
        return "failed", {"status": "failed", "reason": "executor returned a non-mapping result"}
    nested = payload.get("result")
    if isinstance(nested, Mapping) and len(payload) <= 4:
        payload = {
            **dict(nested),
            **{key: item for key, item in payload.items() if key != "result"},
        }
    status = payload.get("status")
    if status == "terminal":
        payload["terminal"] = True
        status = "ok"
    if status in {"ok", "complete", "success", "succeeded"}:
        return "ok", payload
    if status in {"cancelled", "canceled"}:
        return "cancelled", payload
    if status in {"unavailable", "not_available"}:
        return "unavailable", payload
    if status in {"failed", "error", "timeout", "timed_out"}:
        return "failed", payload
    if any(key in payload for key in ("metrics", "measurement", "telemetry", "activated")):
        return "ok", payload
    return "failed", {**payload, "status": "failed", "reason": "executor result has no status"}


def _operation_id(session_id: str, candidate_id: str, kind: str, attempt: int = 1) -> str:
    base = f"{session_id}:candidate:{candidate_id}:{kind}"
    return base if attempt == 1 else f"{base}:retry:{attempt - 1}"


def _extract_metrics(result: Mapping[str, Any]) -> dict[str, Any]:
    for key in ("metrics", "measurement", "telemetry"):
        candidate = result.get(key)
        if isinstance(candidate, Mapping):
            return dict(candidate)
    return {
        key: value
        for key, value in result.items()
        if key.endswith(("_m", "_m_s", "_step")) or key in {"robot_goal_reached"}
    }


def _explicit_activation(result: Mapping[str, Any]) -> bool | None:
    for key in ("mechanism_activated", "activated", "activation"):
        value = result.get(key)
        if isinstance(value, bool):
            return value
        if isinstance(value, Mapping):
            nested = value.get("activated")
            if isinstance(nested, bool):
                return nested
    return None


def _activation(
    result: Mapping[str, Any], *, control: Mapping[str, Any] | None, motion_epsilon: float
) -> bool:
    explicit = _explicit_activation(result)
    if explicit is not None:
        return explicit
    metrics = _extract_metrics(result)
    displacement = metrics.get("ped_displacement_m")
    if isinstance(displacement, (int, float)) and not isinstance(displacement, bool):
        if control is None:
            return float(displacement) > motion_epsilon
    if control is not None:
        control_metrics = _extract_metrics(control)
        treatment_speed = metrics.get("ped_mean_speed_m_s")
        control_speed = control_metrics.get("ped_mean_speed_m_s")
        if isinstance(treatment_speed, (int, float)) and isinstance(control_speed, (int, float)):
            return abs(float(treatment_speed) - float(control_speed)) > motion_epsilon
    return False


def _control_fidelity(result: Mapping[str, Any], *, motion_epsilon: float) -> tuple[bool, str]:
    for key in ("control_fidelity", "fidelity"):
        value = result.get(key)
        if value is False:
            return False, f"{key} reported false"
        if isinstance(value, Mapping):
            status = value.get("status")
            if status in {"fail", "failed", "invalid"}:
                return False, str(value.get("reason", f"{key} failed"))
        if value in {"failed", "fail", "invalid"}:
            return False, f"{key} reported {value}"
    metrics = _extract_metrics(result)
    for key, label in (
        ("ped_displacement_m", "control pedestrian"),
        ("robot_displacement_m", "control robot"),
    ):
        value = metrics.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if float(value) <= motion_epsilon:
                return False, f"{label} shows no measured motion"
    return True, ""


def _reason(result: Mapping[str, Any], fallback: str) -> str:
    value = result.get("reason", result.get("error", fallback))
    return str(value) if value is not None else fallback


class ExperimentLoop:
    """Run one finite candidate set against a durable session journal.

    The constructor intentionally receives an already admitted source
    provenance document.  The public :func:`run` performs the canonical
    SREV-22 admission preflight before constructing this class; direct callers
    should pass the corresponding proof from their own trusted boundary.
    """

    def __init__(
        self,
        request: ComponentRequest,
        *,
        recipe: Mapping[str, Any],
        budget: LoopBudget,
        policy: LoopPolicy,
        journal_path: Path,
        executor: Any,
        source_admission: Mapping[str, Any],
        provenance: Mapping[str, Any] | None = None,
        session_id: str | None = None,
        resume: bool = False,
        cancel: Callable[[], bool] | Any | None = None,
        supported_factors: set[str] | frozenset[str] | None = None,
        supported_measurements: set[str] | frozenset[str] | None = None,
    ) -> None:
        """Create a journal owner around one already-admitted source."""
        self.request = request
        self.recipe = dict(recipe)
        self.budget = budget
        self.policy = policy
        self.journal_path = journal_path
        self.executor = executor
        self.source_admission = dict(source_admission)
        self.provenance = dict(provenance or {})
        _validate_loop_budget(self.recipe, budget)
        self.session_id = (
            session_id or f"{request.request_id}:{_canonical_digest(self.recipe)[:16]}"
        )
        self.cancel = cancel
        self.supported_factors = set(supported_factors or _SUPPORTED_FACTORS)
        self.supported_measurements = set(supported_measurements or _SUPPORTED_MEASUREMENTS)
        self.candidates = _candidate_order(self.recipe)
        self.measurement = self._measurement()
        self._elapsed_base = 0.0
        self._started_at = time.monotonic()
        self._journal = self._new_journal()
        if resume:
            self._load_journal()
        else:
            if journal_path.exists():
                raise ExperimentLoopError(
                    f"output_collision: journal already exists: {journal_path}"
                )
            self._persist()

    def _measurement(self) -> dict[str, Any]:
        measurements = self.recipe.get("measurements")
        if not isinstance(measurements, list) or len(measurements) != 1:
            raise ExperimentLoopError(
                "unsupported_measurement: exactly one driving measurement is supported"
            )
        measurement = measurements[0]
        if not isinstance(measurement, Mapping):
            raise ExperimentLoopError("unsupported_measurement: measurement must be a mapping")
        name = _safe_text(measurement.get("name"), field_name="measurement.name")
        direction = measurement.get("expected_direction")
        if direction not in {"increase", "decrease"}:
            raise ExperimentLoopError(
                "unsupported_measurement: expected_direction must be increase|decrease"
            )
        if name not in self.supported_measurements:
            raise ExperimentLoopError(f"unsupported_measurement: {name}")
        return {
            "name": name,
            "units": _safe_text(measurement.get("units"), field_name="measurement.units"),
            "expected_direction": direction,
        }

    def _candidate_documents(
        self, candidates: list[Mapping[str, Any]] | None = None
    ) -> list[dict[str, Any]]:
        if candidates is None:
            candidates = self.candidates[: self.budget.max_candidates]
        return [
            {
                "intervention_id": item["intervention_id"],
                "priority": item["priority"],
                "factor": item.get("factor", ""),
            }
            for item in candidates
        ]

    def _new_journal(self) -> dict[str, Any]:
        selected = self.candidates[: self.budget.max_candidates]
        return {
            "schema_version": SESSION_JOURNAL_SCHEMA_VERSION,
            "component_id": COMPONENT_ID,
            "component_version": COMPONENT_VERSION,
            "session_id": self.session_id,
            "request_id": self.request.request_id,
            "request_digest": _canonical_digest(_request_identity(self.request)),
            "recipe_id": str(self.recipe.get("recipe_id", "")),
            "recipe_digest": experiment_recipe_canonical_digest(self.recipe),
            "config_digest": _canonical_digest(
                {
                    "request_config": dict(self.request.config),
                    "budget": self.budget.to_dict(),
                    "policy": self.policy.to_dict(),
                }
            ),
            "config_identity_digest": _canonical_digest(
                _config_identity_document(self.request.config, self.policy.to_dict())
            ),
            "policy": self.policy.to_dict(),
            "budget": self.budget.to_dict(),
            "source_admission": dict(self.source_admission),
            "evidence_boundary": DIAGNOSTIC_EVIDENCE_BOUNDARY,
            "scientific_claim_allowed": False,
            "dependent_family_status": DEPENDENT_FAMILY_STATUS,
            "candidate_order": self._candidate_documents(),
            "candidate_catalog": self._candidate_documents(self.candidates),
            "candidates": {
                str(item["intervention_id"]): {
                    "intervention_id": str(item["intervention_id"]),
                    "priority": int(item["priority"]),
                    "factor": str(item.get("factor", "")),
                    "state": "pending",
                    "attempts": 0,
                    "operation_ids": [],
                }
                for item in selected
            },
            "operations": [],
            "outcomes": [],
            "reservations": [],
            "accounting": {
                "controls": 0,
                "treatments": 0,
                "failures": 0,
                "retries": 0,
                "fidelity_attempts": 0,
            },
            "executions_consumed": 0,
            "reserved_executions": 0,
            "elapsed_s": 0.0,
            "status": "running",
            "stop_reason": "",
            "provenance": dict(self.provenance),
            "answerability": None,
            "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "updated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        }

    def _elapsed(self) -> float:
        return self._elapsed_base + max(0.0, time.monotonic() - self._started_at)

    def _persist(self) -> None:
        self._journal["elapsed_s"] = round(self._elapsed(), 6)
        self._journal["updated_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
        _atomic_write_json(self.journal_path, self._journal)
        # A stable alias keeps older consumers from needing a migration while
        # the canonical artifact remains explicitly versioned above.
        alias = self.journal_path.with_name(LEGACY_SESSION_JOURNAL_FILENAME)
        if alias != self.journal_path:
            _atomic_write_json(alias, self._journal)

    def _load_journal(self) -> None:
        try:
            payload = json.loads(self.journal_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, RecursionError) as error:
            raise ExperimentLoopError(
                f"cannot resume: unreadable session journal: {error}"
            ) from error
        if not isinstance(payload, dict):
            raise ExperimentLoopError("cannot resume: session journal is not an object")
        expected = self._new_journal()
        for key in (
            "schema_version",
            "component_id",
            "component_version",
            "session_id",
            "request_id",
            "request_digest",
            "recipe_id",
            "recipe_digest",
            "source_admission",
        ):
            if payload.get(key) != expected.get(key):
                raise ExperimentLoopError(f"cannot resume: journal {key} identity mismatch")
        for key in ("evidence_boundary", "scientific_claim_allowed", "dependent_family_status"):
            if payload.get(key) != expected.get(key):
                raise ExperimentLoopError(f"cannot resume: journal {key} boundary mismatch")
        if payload.get("config_identity_digest") != expected.get("config_identity_digest"):
            raise ExperimentLoopError("cannot resume: journal immutable config identity mismatch")
        prior_policy = payload.get("policy")
        expected_policy = expected["policy"]
        if not isinstance(prior_policy, Mapping):
            raise ExperimentLoopError("cannot resume: journal policy is malformed")
        for key in ("autonomous", "read_only"):
            if prior_policy.get(key) != expected_policy.get(key):
                raise ExperimentLoopError(f"cannot resume: journal policy {key} mismatch")
        if payload.get("scientific_claim_allowed") is True or (
            isinstance(payload.get("provenance"), Mapping)
            and payload["provenance"].get("scientific_claim_allowed") is True
        ):
            raise ExperimentLoopError("cannot resume: scientific claim boundary was widened")
        prior_budget = payload.get("budget")
        current_budget = expected["budget"]
        if not isinstance(prior_budget, Mapping):
            raise ExperimentLoopError("cannot resume: journal budget is malformed")
        if set(prior_budget) != set(current_budget):
            raise ExperimentLoopError("cannot resume: journal budget identity is malformed")
        for key in _RESUMABLE_BUDGET_KEYS:
            prior_value = prior_budget.get(key)
            current_value = current_budget.get(key)
            if isinstance(current_value, float):
                valid_prior = (
                    isinstance(prior_value, (int, float))
                    and not isinstance(prior_value, bool)
                    and math.isfinite(float(prior_value))
                )
                not_reduced = valid_prior and float(current_value) >= float(prior_value)
            else:
                valid_prior = isinstance(prior_value, int) and not isinstance(prior_value, bool)
                not_reduced = valid_prior and int(current_value) >= int(prior_value)
            if not valid_prior:
                raise ExperimentLoopError("cannot resume: journal budget identity is malformed")
            if not not_reduced:
                raise ExperimentLoopError("cannot resume: budget ceilings cannot be reduced")
        if prior_budget.get("max_concurrent_local_cpu_processes") != current_budget.get(
            "max_concurrent_local_cpu_processes"
        ):
            raise ExperimentLoopError("cannot resume: concurrent process budget mismatch")
        prior_catalog = payload.get("candidate_catalog")
        if prior_catalog != expected.get("candidate_catalog"):
            raise ExperimentLoopError("cannot resume: journal candidate catalog mismatch")
        prior_order = payload.get("candidate_order")
        current_order = expected["candidate_order"]
        if not isinstance(prior_order, list) or prior_order != current_order[: len(prior_order)]:
            raise ExperimentLoopError("cannot resume: journal candidate order mismatch")
        if len(prior_order) > len(current_order):
            raise ExperimentLoopError("cannot resume: candidate budget was reduced")
        if not isinstance(payload.get("operations"), list) or not isinstance(
            payload.get("candidates"), dict
        ):
            raise ExperimentLoopError("cannot resume: journal state is malformed")
        prior_candidate_ids = {
            str(item.get("intervention_id"))
            for item in prior_order
            if isinstance(item, Mapping) and isinstance(item.get("intervention_id"), str)
        }
        if set(payload["candidates"]) != prior_candidate_ids:
            raise ExperimentLoopError("cannot resume: journal candidate state mismatch")
        for candidate_id, candidate_state in payload["candidates"].items():
            if not isinstance(candidate_state, Mapping):
                raise ExperimentLoopError("cannot resume: malformed candidate state")
            if candidate_state.get("intervention_id") != candidate_id:
                raise ExperimentLoopError("cannot resume: candidate identity mismatch")
            if candidate_state.get("state") not in {
                "pending",
                "reserved",
                "dispatching",
                "complete",
                "failed",
                "unavailable",
                "cancelled",
            }:
                raise ExperimentLoopError("cannot resume: invalid candidate state")
            operation_ids_for_candidate = candidate_state.get("operation_ids", [])
            if not isinstance(operation_ids_for_candidate, list) or not all(
                isinstance(operation_id, str) for operation_id in operation_ids_for_candidate
            ):
                raise ExperimentLoopError("cannot resume: malformed candidate operation IDs")
        consumed = payload.get("executions_consumed", 0)
        if (
            not isinstance(consumed, int)
            or isinstance(consumed, bool)
            or consumed < 0
            or consumed > self.budget.max_executions
        ):
            raise ExperimentLoopError("cannot resume: execution accounting is invalid")
        reserved = payload.get("reserved_executions", 0)
        if (
            not isinstance(reserved, int)
            or isinstance(reserved, bool)
            or reserved < 0
            or reserved > self.budget.max_executions
        ):
            raise ExperimentLoopError("cannot resume: reservation accounting is invalid")
        accounting = payload.get("accounting")
        if not isinstance(accounting, Mapping) or any(
            not isinstance(accounting.get(key), int)
            or isinstance(accounting.get(key), bool)
            or accounting[key] < 0
            for key in ("controls", "treatments", "failures", "retries", "fidelity_attempts")
        ):
            raise ExperimentLoopError("cannot resume: operation accounting is invalid")
        operation_ids: set[str] = set()
        dispatch_total = 0
        for operation in payload["operations"]:
            if not isinstance(operation, dict) or not isinstance(
                operation.get("operation_id"), str
            ):
                raise ExperimentLoopError("cannot resume: malformed operation record")
            operation_id = str(operation["operation_id"])
            if operation_id in operation_ids:
                raise ExperimentLoopError("cannot resume: duplicate operation ID")
            operation_ids.add(operation_id)
            candidate_id = operation.get("candidate_id")
            kind = operation.get("kind")
            attempt = operation.get("attempt")
            if (
                not isinstance(candidate_id, str)
                or candidate_id not in prior_candidate_ids
                or not isinstance(kind, str)
                or kind not in {"control", "treatment"}
                or not isinstance(attempt, int)
                or isinstance(attempt, bool)
                or attempt < 1
                or attempt > self.budget.max_retries + 1
                or operation_id != _operation_id(self.session_id, candidate_id, kind, attempt)
            ):
                raise ExperimentLoopError("cannot resume: malformed operation identity")
            if operation.get("state") not in {
                "reserved",
                "dispatching",
                "completed",
                "failed",
                "cancelled",
                "unavailable",
            }:
                raise ExperimentLoopError("cannot resume: invalid operation state")
            dispatch_count = operation.get("dispatch_count", 0)
            if (
                not isinstance(dispatch_count, int)
                or isinstance(dispatch_count, bool)
                or dispatch_count < 0
                or dispatch_count > 1
            ):
                raise ExperimentLoopError("cannot resume: invalid operation dispatch count")
            dispatch_total += dispatch_count
        if dispatch_total != consumed:
            raise ExperimentLoopError(
                "cannot resume: operation accounting does not match dispatches"
            )
        outcomes = payload.get("outcomes")
        if not isinstance(outcomes, list):
            raise ExperimentLoopError("cannot resume: journal outcomes are malformed")
        outcome_ids: set[str] = set()
        for outcome in outcomes:
            if not isinstance(outcome, Mapping) or not isinstance(
                outcome.get("intervention_id"), str
            ):
                raise ExperimentLoopError("cannot resume: malformed outcome record")
            outcome_id = str(outcome["intervention_id"])
            if outcome_id not in prior_candidate_ids or outcome_id in outcome_ids:
                raise ExperimentLoopError("cannot resume: duplicate or unknown outcome record")
            outcome_ids.add(outcome_id)
        existing_candidate_ids = set(payload["candidates"])
        for candidate in current_order:
            candidate_id = str(candidate["intervention_id"])
            if candidate_id in existing_candidate_ids:
                continue
            payload["candidates"][candidate_id] = {
                "intervention_id": candidate_id,
                "priority": int(candidate["priority"]),
                "factor": str(candidate.get("factor", "")),
                "state": "pending",
                "attempts": 0,
                "operation_ids": [],
            }
        payload["candidate_order"] = current_order
        payload["budget"] = self.budget.to_dict()
        payload["policy"] = self.policy.to_dict()
        payload["config_digest"] = expected["config_digest"]
        payload["config_identity_digest"] = expected["config_identity_digest"]
        self._journal = payload
        elapsed = payload.get("elapsed_s", 0.0)
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(float(elapsed))
            or float(elapsed) < 0.0
        ):
            raise ExperimentLoopError("cannot resume: elapsed accounting is invalid")
        self._elapsed_base = float(elapsed)
        self._started_at = time.monotonic()
        self._persist()

    def _cancelled(self) -> bool:
        if self.policy.cancel_requested:
            return True
        if self.cancel is None:
            return False
        if callable(self.cancel):
            try:
                return bool(self.cancel())
            except (TypeError, RuntimeError):
                return False
        is_set = getattr(self.cancel, "is_set", None)
        return bool(is_set()) if callable(is_set) else bool(self.cancel)

    def _operation(self, operation_id: str) -> dict[str, Any] | None:
        return next(
            (
                operation
                for operation in self._journal["operations"]
                if operation.get("operation_id") == operation_id
            ),
            None,
        )

    def _candidate_state(self, candidate_id: str) -> dict[str, Any]:
        return cast("dict[str, Any]", self._journal["candidates"][candidate_id])

    def _reserve_pair(self, candidate: Mapping[str, Any]) -> bool:
        candidate_id = str(candidate["intervention_id"])
        state = self._candidate_state(candidate_id)
        # A crash may leave a fully reserved pair with one dispatch already
        # consumed.  Resume must settle that exact reservation even when the
        # remaining *new* budget is smaller than two.
        if state.get("state") in {"reserved", "dispatching"}:
            return True
        if state.get("state") in TERMINAL_CANDIDATE_STATES:
            return True
        remaining = self.budget.max_executions - int(self._journal["executions_consumed"])
        reserved = int(self._journal["reserved_executions"])
        if remaining - reserved < RESERVED_CONTROL_TREATMENT_PAIR:
            return False
        if self._elapsed() >= self.budget.wall_timeout_s:
            return False
        state["state"] = "reserved"
        state["reservation"] = RESERVED_CONTROL_TREATMENT_PAIR
        self._journal["reserved_executions"] += RESERVED_CONTROL_TREATMENT_PAIR
        self._journal["reservations"].append(
            {
                "candidate_id": candidate_id,
                "required_executions": RESERVED_CONTROL_TREATMENT_PAIR,
                "state": "reserved",
            }
        )
        self._persist()
        return True

    def _release_reservation(self, candidate_id: str) -> None:
        state = self._candidate_state(candidate_id)
        amount = int(state.pop("reservation", 0))
        self._journal["reserved_executions"] = max(
            0, int(self._journal["reserved_executions"]) - amount
        )
        for reservation in self._journal["reservations"]:
            if (
                reservation.get("candidate_id") == candidate_id
                and reservation.get("state") == "reserved"
            ):
                reservation["state"] = "released"
                break

    def _call_recovery(self, operation_id: str) -> tuple[str, dict[str, Any]] | None:
        for method_name in ("result_for", "recover", "lookup", "get_result"):
            method = getattr(self.executor, method_name, None)
            if not callable(method):
                continue
            try:
                value = method(operation_id)
            except (KeyError, LookupError):
                continue
            if value is None:
                continue
            return _status_from_result(value)
        return None

    def _invoke_executor(
        self,
        operation_id: str,
        candidate: Mapping[str, Any],
        kind: str,
        spec: Mapping[str, Any],
        attempt: int,
    ) -> Any:
        method = getattr(self.executor, "execute", None)
        if method is None:
            method = getattr(self.executor, "run", self.executor)
        if not callable(method):
            raise ExperimentLoopError("executor is not callable")
        payload = {
            "operation_id": operation_id,
            "candidate": dict(candidate),
            "kind": kind,
            "spec": dict(spec),
            "attempt": attempt,
        }
        try:
            signature = inspect.signature(method)
        except (TypeError, ValueError):
            return method(**payload)
        parameters = signature.parameters
        if any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
        ):
            return method(**payload)
        accepted = {key: value for key, value in payload.items() if key in parameters}
        if accepted:
            return method(**accepted)
        if len(parameters) == 1:
            return method(payload)
        ordered = [payload[key] for key in ("operation_id", "kind", "candidate", "spec", "attempt")]
        return method(*ordered[: len(parameters)])

    def _run_operation(
        self,
        candidate: Mapping[str, Any],
        kind: str,
        spec: Mapping[str, Any],
    ) -> tuple[str, dict[str, Any], str]:
        candidate_id = str(candidate["intervention_id"])
        attempts = 0
        while attempts <= self.budget.max_retries:
            attempts += 1
            operation_id = _operation_id(self.session_id, candidate_id, kind, attempts)
            operation = self._operation(operation_id)
            if operation is not None and operation.get("state") == "completed":
                state, result = _status_from_result(operation.get("result", {}))
                return state, result, operation_id
            if operation is not None and operation.get("state") in {
                "failed",
                "cancelled",
                "unavailable",
            }:
                state, result = _status_from_result(operation.get("result", {}))
                if (
                    state == "failed"
                    and bool(result.get("retryable", result.get("retry", False)))
                    and attempts <= self.budget.max_retries
                ):
                    # The failed attempt is already durable.  Continue with a
                    # distinct retry operation ID; never dispatch a terminal
                    # record under its original ID a second time.
                    continue
                return state, result, operation_id
            if operation is not None and operation.get("state") == "dispatching":
                recovered = self._call_recovery(operation_id)
                if recovered is None:
                    operation["state"] = "failed"
                    operation["result"] = {
                        "status": "failed",
                        "reason": "crash_recovery_unknown: dispatch had started and no idempotent result was available",
                    }
                    operation["failure_accounted"] = True
                    self._journal["accounting"]["failures"] += 1
                    self._persist()
                    return "failed", cast("dict[str, Any]", operation["result"]), operation_id
                state, result = recovered
                operation["state"] = "completed" if state == "ok" else state
                operation["result"] = result
                if state == "failed" and not operation.get("failure_accounted", False):
                    self._journal["accounting"]["failures"] += 1
                    operation["failure_accounted"] = True
                self._persist()
                if (
                    state == "failed"
                    and bool(result.get("retryable", result.get("retry", False)))
                    and attempts <= self.budget.max_retries
                ):
                    continue
                return state, result, operation_id
            if operation is None:
                if int(self._journal["executions_consumed"]) >= self.budget.max_executions:
                    return (
                        "failed",
                        {
                            "status": "failed",
                            "reason": "execution_budget_exhausted: retry cannot be dispatched",
                        },
                        operation_id,
                    )
                operation = {
                    "operation_id": operation_id,
                    "candidate_id": candidate_id,
                    "kind": kind,
                    "attempt": attempts,
                    "state": "reserved",
                    "dispatch_count": 0,
                    "failure_accounted": False,
                }
                self._journal["operations"].append(operation)
                self._candidate_state(candidate_id)["operation_ids"].append(operation_id)
            operation["state"] = "dispatching"
            operation["dispatch_count"] = int(operation.get("dispatch_count", 0)) + 1
            self._journal["executions_consumed"] += 1
            accounting = self._journal["accounting"]
            accounting["controls" if kind == "control" else "treatments"] += 1
            if attempts > 1:
                accounting["retries"] += 1
            self._candidate_state(candidate_id)["attempts"] = (
                int(self._candidate_state(candidate_id).get("attempts", 0)) + 1
            )
            self._persist()
            try:
                raw_result = self._invoke_executor(operation_id, candidate, kind, spec, attempts)
            except BaseException as error:
                operation["result"] = {
                    "status": "failed",
                    "reason": f"dispatch_interrupted: {type(error).__name__}: {error}",
                }
                # Persisting ``dispatching`` is deliberate only for process
                # interruption.  A regular executor exception is a settled
                # failed attempt and must not be replayed on resume.
                if not isinstance(error, (KeyboardInterrupt, SystemExit)):
                    operation["state"] = "failed"
                    operation["failure_accounted"] = True
                    self._journal["accounting"]["failures"] += 1
                self._persist()
                if isinstance(error, (KeyboardInterrupt, SystemExit)):
                    raise
                return "failed", cast("dict[str, Any]", operation["result"]), operation_id
            state, result = _status_from_result(raw_result)
            operation["state"] = {
                "ok": "completed",
                "cancelled": "cancelled",
                "unavailable": "unavailable",
            }.get(state, "failed")
            operation["result"] = result
            if state == "failed":
                self._journal["accounting"]["failures"] += 1
                operation["failure_accounted"] = True
            self._persist()
            if (
                state == "failed"
                and bool(result.get("retryable", result.get("retry", False)))
                and attempts <= self.budget.max_retries
            ):
                continue
            return state, result, operation_id
        return "failed", {"status": "failed", "reason": "retry_budget_exhausted"}, operation_id

    def _pair_spec(self, candidate: Mapping[str, Any], kind: str) -> dict[str, Any]:
        control_conditions = self.recipe.get("control_conditions", {})
        if not isinstance(control_conditions, Mapping):
            raise ExperimentLoopError("invalid_control_conditions: mapping required")
        spec = dict(control_conditions)
        spec.update(
            {
                "scenario_id": self.recipe.get("source_identity", {}).get("scenario_id", ""),
                "source_identity": dict(self.recipe.get("source_identity", {})),
                "candidate_id": str(candidate["intervention_id"]),
                "kind": kind,
                "intervention": dict(candidate),
            }
        )
        if kind == "treatment":
            treatment = candidate.get("treatment", candidate.get("parameters", {}))
            if isinstance(treatment, Mapping):
                spec.update(dict(treatment))
        return spec

    def _unsupported_candidate_reason(self, candidate: Mapping[str, Any]) -> str | None:
        factor = candidate.get("factor")
        if factor not in self.supported_factors:
            return f"unsupported_recipe: intervention factor {factor!r} is unsupported"
        return None

    def _record_outcome(self, candidate_id: str, outcome: Mapping[str, Any]) -> None:
        state = self._candidate_state(candidate_id)
        state.update(
            {key: value for key, value in outcome.items() if key not in {"intervention_id"}}
        )
        self._journal["outcomes"].append(dict(outcome))
        state["state"] = str(outcome.get("status", "failed"))
        self._release_reservation(candidate_id)
        self._persist()

    def _execute_candidate(self, candidate: Mapping[str, Any]) -> dict[str, Any]:
        candidate_id = str(candidate["intervention_id"])
        factor = str(candidate.get("factor", ""))
        state = self._candidate_state(candidate_id)
        if state.get("state") in TERMINAL_CANDIDATE_STATES:
            existing = [
                item
                for item in self._journal["outcomes"]
                if item.get("intervention_id") == candidate_id
            ]
            return dict(existing[-1]) if existing else dict(state)
        unsupported = self._unsupported_candidate_reason(candidate)
        if unsupported is not None:
            outcome = {
                "intervention_id": candidate_id,
                "priority": int(candidate["priority"]),
                "factor": factor,
                "status": "unavailable",
                "outcome": "inconclusive",
                "reason": unsupported,
                "negative": True,
                "operation_ids": [],
            }
            self._record_outcome(candidate_id, outcome)
            return outcome
        control_state, control, control_operation = self._run_operation(
            candidate, "control", self._pair_spec(candidate, "control")
        )
        operation_ids = [control_operation]
        if control_state in {"cancelled", "unavailable", "failed"}:
            outcome = {
                "intervention_id": candidate_id,
                "priority": int(candidate["priority"]),
                "factor": factor,
                "status": (
                    "cancelled"
                    if control_state == "cancelled"
                    else ("unavailable" if control_state == "unavailable" else "failed")
                ),
                "outcome": "inconclusive",
                "reason": f"control_{control_state}: {_reason(control, 'control did not complete')}",
                "control": control,
                "negative": True,
                "operation_ids": operation_ids,
            }
            self._record_outcome(candidate_id, outcome)
            return outcome
        fidelity_ok, fidelity_reason = _control_fidelity(
            control, motion_epsilon=float(self.recipe.get("motion_epsilon_m", 0.05))
        )
        self._journal["accounting"]["fidelity_attempts"] += 1
        self._persist()
        if not fidelity_ok:
            outcome = {
                "intervention_id": candidate_id,
                "priority": int(candidate["priority"]),
                "factor": factor,
                "status": "failed",
                "outcome": "inconclusive",
                "reason": f"control_fidelity_failure: {fidelity_reason}",
                "control": control,
                "control_fidelity": {"status": "failed", "reason": fidelity_reason},
                "negative": True,
                "operation_ids": operation_ids,
            }
            self._record_outcome(candidate_id, outcome)
            return outcome
        if self._cancelled():
            outcome = {
                "intervention_id": candidate_id,
                "priority": int(candidate["priority"]),
                "factor": factor,
                "status": "cancelled",
                "outcome": "inconclusive",
                "reason": "cancellation_requested",
                "control": control,
                "negative": True,
                "operation_ids": operation_ids,
            }
            self._record_outcome(candidate_id, outcome)
            return outcome
        treatment_state, treatment, treatment_operation = self._run_operation(
            candidate, "treatment", self._pair_spec(candidate, "treatment")
        )
        operation_ids.append(treatment_operation)
        if treatment_state in {"cancelled", "unavailable", "failed"}:
            outcome = {
                "intervention_id": candidate_id,
                "priority": int(candidate["priority"]),
                "factor": factor,
                "status": (
                    "cancelled"
                    if treatment_state == "cancelled"
                    else ("unavailable" if treatment_state == "unavailable" else "failed")
                ),
                "outcome": "inconclusive",
                "reason": f"treatment_{treatment_state}: {_reason(treatment, 'treatment did not complete')}",
                "control": control,
                "treatment": treatment,
                "negative": True,
                "operation_ids": operation_ids,
            }
            self._record_outcome(candidate_id, outcome)
            return outcome
        metric_name = str(self.measurement["name"])
        control_metrics = _extract_metrics(control)
        treatment_metrics = _extract_metrics(treatment)
        if metric_name not in control_metrics or metric_name not in treatment_metrics:
            outcome = {
                "intervention_id": candidate_id,
                "priority": int(candidate["priority"]),
                "factor": factor,
                "status": "complete",
                "outcome": "inconclusive",
                "reason": f"measurement_missing: {metric_name}",
                "control": control,
                "treatment": treatment,
                "negative": True,
                "operation_ids": operation_ids,
            }
            self._record_outcome(candidate_id, outcome)
            return outcome
        motion_epsilon = float(self.recipe.get("motion_epsilon_m", 0.05))
        control_activated = _activation(control, control=None, motion_epsilon=motion_epsilon)
        treatment_activated = _activation(treatment, control=control, motion_epsilon=motion_epsilon)
        try:
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
                    expected_direction=str(self.measurement["expected_direction"]),
                ),
            )
            verdict = pair_result.verdict
            verdict_reason = pair_result.reason
        except (KeyError, TypeError, ValueError) as error:
            verdict = "inconclusive"
            verdict_reason = f"pair_evaluation_unavailable: {error}"
        if verdict not in OUTCOMES:
            verdict = "inconclusive"
        outcome = {
            "intervention_id": candidate_id,
            "priority": int(candidate["priority"]),
            "factor": factor,
            "status": "complete",
            "outcome": verdict,
            "verdict": verdict,
            "reason": verdict_reason,
            "control": control,
            "treatment": treatment,
            "control_activated": control_activated,
            "treatment_activated": treatment_activated,
            "activation": {
                "control": control_activated,
                "treatment": treatment_activated,
                "measured": True,
            },
            "negative": verdict != "survived",
            "operation_ids": operation_ids,
            "terminal": bool(
                control.get("terminal", False)
                or treatment.get("terminal", False)
                or treatment.get("terminal_condition_reached", False)
            ),
        }
        self._record_outcome(candidate_id, outcome)
        return outcome

    def _terminal_reason(self, outcomes: list[Mapping[str, Any]]) -> tuple[str, str]:
        if self._cancelled():
            return "cancelled", "cancellation_requested"
        if self._elapsed() >= self.budget.wall_timeout_s:
            return "partial", "wall_timeout"
        failed = [item for item in outcomes if item.get("status") == "failed"]
        cancelled = [item for item in outcomes if item.get("status") == "cancelled"]
        unavailable = [item for item in outcomes if item.get("status") == "unavailable"]
        complete = [item for item in outcomes if item.get("status") == "complete"]
        selected_count = min(len(self.candidates), self.budget.max_candidates)
        if len(outcomes) >= selected_count:
            if unavailable:
                return (
                    "partial" if complete else "unavailable",
                    "candidate_unavailable"
                    if complete
                    else "all selected recipes are unsupported or unavailable",
                )
            if cancelled:
                return "cancelled", str(cancelled[-1].get("reason", "cancellation_requested"))
            if failed:
                return "partial" if complete else "failed", "candidate_execution_failed"
            return "complete", "exhausted_candidates"
        if int(self._journal["executions_consumed"]) >= self.budget.max_executions:
            return "partial", "execution_budget_exhausted"
        if cancelled:
            return "cancelled", str(cancelled[-1].get("reason", "cancellation_requested"))
        if failed and any(
            "control_fidelity_failure" in str(item.get("reason", "")) for item in failed
        ):
            return (
                "failed" if not complete else "partial",
                "control_fidelity_failure_blocks_treatment",
            )
        if not complete and unavailable:
            return "unavailable", "all selected recipes are unsupported or unavailable"
        if failed:
            return "partial" if complete else "failed", "candidate_execution_failed"
        return "complete", "exhausted_candidates"

    def run(self) -> ComponentResult:
        """Drive the finite loop and settle a truthful report/journal pair.

        Returns:
            A status-bearing component result. Complete results reference the
            report and journal artifacts; non-complete results retain the
            journal without advertising complete artifacts.
        """

        if self.policy.read_only:
            self._journal["status"] = "unavailable"
            self._journal["stop_reason"] = "read_only_never_executes"
            self._persist()
            return self._result("unavailable", "read_only_never_executes")
        if not self.policy.autonomous:
            self._journal["status"] = "unavailable"
            self._journal["stop_reason"] = "autonomous_start_authorization_required"
            self._persist()
            return self._result("unavailable", "autonomous_start_authorization_required")
        outcomes: list[dict[str, Any]] = [dict(item) for item in self._journal.get("outcomes", [])]
        persisted_status = self._journal.get("status")
        persisted_reason = str(self._journal.get("stop_reason", ""))
        if persisted_status in {"complete", "failed", "unavailable", "cancelled"} or (
            persisted_status == "partial" and persisted_reason not in _RESUMABLE_STOP_REASONS
        ):
            return self._result(
                str(persisted_status),
                persisted_reason or "resumed terminal session",
            )
        selected = self.candidates[: self.budget.max_candidates]
        try:
            for candidate in selected:
                candidate_id = str(candidate["intervention_id"])
                if self._candidate_state(candidate_id).get("state") in TERMINAL_CANDIDATE_STATES:
                    continue
                if self._cancelled():
                    self._journal["status"] = "cancelled"
                    self._journal["stop_reason"] = "cancellation_requested"
                    break
                if self._elapsed() >= self.budget.wall_timeout_s:
                    self._journal["status"] = "partial"
                    self._journal["stop_reason"] = "wall_timeout"
                    break
                if not self._reserve_pair(candidate):
                    self._journal["status"] = "partial"
                    self._journal["stop_reason"] = "execution_budget_exhausted"
                    break
                outcome = self._execute_candidate(candidate)
                outcomes = [dict(item) for item in self._journal["outcomes"]]
                if outcome.get("status") == "cancelled":
                    self._journal["status"] = "cancelled"
                    self._journal["stop_reason"] = str(
                        outcome.get("reason", "cancellation_requested")
                    )
                    break
                if outcome.get("status") == "failed" and "control_fidelity_failure" in str(
                    outcome.get("reason", "")
                ):
                    self._journal["status"] = (
                        "failed"
                        if not any(item.get("status") == "complete" for item in outcomes)
                        else "partial"
                    )
                    self._journal["stop_reason"] = "control_fidelity_failure_blocks_treatment"
                    break
                if outcome.get("terminal"):
                    self._journal["status"] = (
                        "complete" if outcome.get("status") == "complete" else "partial"
                    )
                    self._journal["stop_reason"] = "recipe_terminal_condition"
                    break
            else:
                status, reason = self._terminal_reason(outcomes)
                self._journal["status"] = status
                self._journal["stop_reason"] = reason
        except ExperimentLoopError:
            raise
        except BaseException:
            # The dispatching operation and its journal are intentionally left
            # recoverable.  Callers may resume with the same operation IDs.
            self._persist()
            raise
        self._persist()
        return self._result(str(self._journal["status"]), str(self._journal["stop_reason"]))

    def _report_payload(self) -> dict[str, Any]:
        outcomes = [dict(item) for item in self._journal.get("outcomes", [])]
        return {
            "schema_version": LOOP_REPORT_SCHEMA_VERSION,
            "component_id": COMPONENT_ID,
            "component_version": COMPONENT_VERSION,
            "session_id": self.session_id,
            "request_id": self.request.request_id,
            "recipe_id": str(self.recipe.get("recipe_id", "")),
            "recipe_digest": experiment_recipe_canonical_digest(self.recipe),
            "source_identity": dict(self.recipe.get("source_identity", {})),
            "source_admission": dict(self._journal.get("source_admission", {})),
            "evidence_boundary": DIAGNOSTIC_EVIDENCE_BOUNDARY,
            "scientific_claim_allowed": False,
            "dependent_family_status": DEPENDENT_FAMILY_STATUS,
            "status": self._journal.get("status", "running"),
            "stop_reason": self._journal.get("stop_reason", ""),
            "candidate_order": self._journal.get("candidate_order", []),
            "outcomes": outcomes,
            "negative_outcomes": [item for item in outcomes if item.get("negative")],
            "budget": {
                **self.budget.to_dict(),
                "executions_consumed": self._journal.get("executions_consumed", 0),
                "reserved_executions": self._journal.get("reserved_executions", 0),
                "elapsed_s": self._journal.get("elapsed_s", 0.0),
            },
            "accounting": dict(self._journal.get("accounting", {})),
            "operations": list(self._journal.get("operations", [])),
            "answerability": self._journal.get("answerability"),
            "provenance": {
                **dict(self.provenance),
                "component_id": COMPONENT_ID,
                "component_version": COMPONENT_VERSION,
                "request_digest": _canonical_digest(_request_identity(self.request)),
                "recipe_digest": experiment_recipe_canonical_digest(self.recipe),
                "config_digest": _canonical_digest(dict(self.request.config)),
                "config": dict(self.request.config),
                "source_admission": dict(self._journal.get("source_admission", {})),
                "tool": {
                    "component_id": COMPONENT_ID,
                    "component_version": COMPONENT_VERSION,
                    "executor_component_id": review_execute.COMPONENT_ID,
                    "executor_component_version": review_execute.COMPONENT_VERSION,
                },
                "evidence_boundary": DIAGNOSTIC_EVIDENCE_BOUNDARY,
                "benchmark_success": False,
                "scientific_claim_allowed": False,
                "dependent_family_status": DEPENDENT_FAMILY_STATUS,
                "python": sys.version.split()[0],
                "platform": platform.platform(),
            },
        }

    def _result(self, status: str, reason: str) -> ComponentResult:
        report = self._report_payload()
        report_path = self.journal_path.with_name(LOOP_REPORT_FILENAME)
        report_digest = _atomic_write_json(report_path, report)
        journal_digest = hashlib.sha256(self.journal_path.read_bytes()).hexdigest()
        artifacts = ()
        if status == "complete":
            prefix = Path(self.request.output_directory)
            artifacts = (
                {
                    "artifact_id": LOOP_REPORT_FILENAME,
                    "uri": str(prefix / LOOP_REPORT_FILENAME),
                    "sha256": report_digest,
                },
                {
                    "artifact_id": SESSION_JOURNAL_FILENAME,
                    "uri": str(prefix / SESSION_JOURNAL_FILENAME),
                    "sha256": journal_digest,
                },
            )
        try:
            return component_result_from_dict(
                {
                    "schema_version": COMPONENT_RESULT_SCHEMA_VERSION,
                    "request_id": self.request.request_id,
                    "component_id": COMPONENT_ID,
                    "status": status,
                    "reason": reason,
                    "artifacts": list(artifacts),
                    "diagnostics": [
                        {
                            "intervention_id": item.get("intervention_id"),
                            "status": item.get("status"),
                            "outcome": item.get("outcome"),
                            "reason": item.get("reason", ""),
                        }
                        for item in self._journal.get("outcomes", [])
                    ],
                    "provenance": report["provenance"],
                }
            )
        except ReviewContractsValidationError as error:
            return ComponentResult(
                request_id=self.request.request_id,
                component_id=COMPONENT_ID,
                status="failed",
                reason=f"internal_result_invalid: {'; '.join(error.errors)}",
            )


class _NativeExecutorAdapter:
    """Adapt one SREV-22 session to operation-level loop reads.

    SREV-22 already performs the real paired execution and persists its own
    attempt ledger.  The adapter invokes it once; subsequent operation reads
    come from its report, so the outer journal can expose stable control and
    treatment operation IDs without introducing a second simulator runner.
    """

    def __init__(
        self,
        request: ComponentRequest,
        *,
        base: Path,
        executor_config: Mapping[str, Any],
        recipe: Mapping[str, Any],
        admission_config: Mapping[str, Any] | review_execute.ExecutorAdmissionConfig,
        resume: bool,
    ) -> None:
        self.request = request
        self.base = base
        self.executor_config = dict(executor_config)
        self.recipe = dict(recipe)
        self.admission_config = admission_config
        self.resume = resume
        self.child_result: ComponentResult | None = None
        self.reports: dict[str, dict[str, Any]] = {}

    def _invoke(self) -> None:
        config = dict(self.executor_config)
        config["recipe"] = self.recipe
        config["max_candidates"] = min(
            int(config.get("max_candidates", DEFAULT_MAX_CANDIDATES)), DEFAULT_MAX_CANDIDATES
        )
        config["max_executions"] = min(
            int(config.get("max_executions", DEFAULT_MAX_EXECUTIONS)), DEFAULT_MAX_EXECUTIONS
        )
        child_request = ComponentRequest(
            request_id=self.request.request_id,
            component_id=review_execute.COMPONENT_ID,
            sources=self.request.sources,
            output_directory="executor",
            config=config,
            required_capabilities=self.request.required_capabilities,
        )
        self.child_result = review_execute.run(
            child_request,
            base=self.base,
            resume=self.resume,
            admission_config=self.admission_config,
        )
        report_path = self.base / "executor" / "execute-report.json"
        ledger_path = self.base / "executor" / "attempt-ledger.json"
        payload: Any = None
        for path in (report_path, ledger_path):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError, RecursionError):
                continue
            if isinstance(payload, dict) and "candidates" in payload:
                break
        if isinstance(payload, Mapping):
            reports = payload.get("candidates", payload.get("candidate_reports", []))
            if isinstance(reports, list):
                self.reports = {
                    str(item["intervention_id"]): dict(item)
                    for item in reports
                    if isinstance(item, Mapping) and isinstance(item.get("intervention_id"), str)
                }

    def _candidate_report(self, candidate_id: str) -> dict[str, Any] | None:
        if self.child_result is None:
            self._invoke()
        return self.reports.get(candidate_id)

    def _operation_result(self, candidate_id: str, kind: str) -> Mapping[str, Any]:
        report = self._candidate_report(candidate_id)
        if report is None:
            if self.child_result is not None and self.child_result.status == "failed":
                return {
                    "status": "failed",
                    "reason": self.child_result.reason
                    or "native executor failed admission or execution",
                }
            return {
                "status": "unavailable",
                "reason": "native executor did not produce candidate report",
            }
        if report.get("status") == "unavailable":
            return {
                "status": "unavailable",
                "reason": report.get("reason", "candidate unavailable"),
            }
        if report.get("status") == "failed":
            if kind == "control" and isinstance(report.get("control_metrics"), Mapping):
                return {
                    "status": "ok",
                    "metrics": dict(report["control_metrics"]),
                    "fidelity": False,
                    "reason": report.get("reason", "control failed"),
                }
            return {"status": "failed", "reason": report.get("reason", "candidate failed")}
        metrics_key = "control_metrics" if kind == "control" else "treatment_metrics"
        metrics = report.get(metrics_key)
        if not isinstance(metrics, Mapping):
            return {"status": "failed", "reason": f"native report lacks {metrics_key}"}
        activated_key = "control_activated" if kind == "control" else "treatment_activated"
        return {
            "status": "ok",
            "metrics": dict(metrics),
            "mechanism_activated": bool(report.get(activated_key, False)),
            "fidelity": True,
        }

    def execute(self, **kwargs: Any) -> Mapping[str, Any]:
        candidate = kwargs.get("candidate", {})
        candidate_id = str(candidate.get("intervention_id", ""))
        kind = str(kwargs.get("kind", ""))
        return self._operation_result(candidate_id, kind)

    def result_for(self, operation_id: str) -> Mapping[str, Any] | None:
        # The child attempt ledger is the idempotent recovery authority.  A
        # dispatching outer operation is recovered by resuming that ledger,
        # never by blindly issuing a second simulator operation.
        marker = ":candidate:"
        if marker not in operation_id:
            return None
        candidate_and_kind = operation_id.split(marker, maxsplit=1)[1]
        if ":retry:" in candidate_and_kind:
            candidate_and_kind = candidate_and_kind.split(":retry:", maxsplit=1)[0]
        if ":" not in candidate_and_kind:
            return None
        candidate_id, kind = candidate_and_kind.rsplit(":", maxsplit=1)
        if not candidate_id or kind not in {"control", "treatment"}:
            return None
        return self._operation_result(candidate_id, kind)


def _native_executor_config(
    validated: _ValidatedInput,
) -> dict[str, Any]:
    config = dict(validated.executor_config)
    # Child limits are owned by the loop.  A nested executor config may tune
    # simulator details, but it cannot silently widen the outer session's
    # candidate, execution, or elapsed budget.
    config["max_candidates"] = validated.budget.max_candidates
    config["max_executions"] = validated.budget.max_executions
    config["wall_timeout_s"] = validated.budget.wall_timeout_s
    config.setdefault("planner", "simple_policy")
    config.setdefault("seed", 7)
    config.setdefault("horizon_steps", 60)
    config.setdefault("robot_speed_m_s", 1.0)
    config.setdefault("per_execution_timeout_s", 120.0)
    config.setdefault("intervention_parameters", {})
    return config


def _preflight_native_admission(
    request: ComponentRequest,
    *,
    recipe: Mapping[str, Any],
    executor_config: Mapping[str, Any],
    admission_config: Mapping[str, Any] | review_execute.ExecutorAdmissionConfig | None,
) -> tuple[dict[str, Any] | None, review_execute.ExecutorAdmissionConfig | None, str | None]:
    if admission_config is None:
        return None, None, "source_admission: explicit launcher admission configuration is required"
    try:
        if isinstance(admission_config, review_execute.ExecutorAdmissionConfig):
            admission = review_execute.validate_executor_admission_config(
                admission_config.to_dict()
            )
        else:
            admission = review_execute.validate_executor_admission_config(admission_config)
        child_config_payload = dict(executor_config)
        child_config_payload["recipe"] = dict(recipe)
        child_config_payload["admission"] = admission.to_dict()
        child_config = review_execute.validate_execute_config(child_config_payload)
        child_request = ComponentRequest(
            request_id=request.request_id,
            component_id=review_execute.COMPONENT_ID,
            sources=request.sources,
            output_directory="executor",
            config=child_config_payload,
            required_capabilities=request.required_capabilities,
        )
        proof, failure = review_execute._resolve_executor_admission(
            child_request,
            child_config,
            dict(recipe),
            admission=admission,
        )
        if failure is not None or proof is None:
            return (
                None,
                admission,
                "; ".join(str(item) for item in (failure or ("failed", "no proof"))),
            )
        document = proof.to_dict()
        proof_root_fd = proof.root_fd
        try:
            os.close(proof_root_fd)
        except OSError:
            pass
        return document, admission, None
    except (
        review_execute.ReviewExecuteError,
        OSError,
        TypeError,
        ValueError,
        ReviewContractsValidationError,
    ) as error:
        return None, None, f"source_admission: {error}"


def _prepare_output_directory(
    request: ComponentRequest, base: Path, *, resume: bool
) -> tuple[Path | None, str | None]:
    if not isinstance(request.output_directory, str):
        return None, "invalid_output_path: output_directory must be a string"
    relative = Path(request.output_directory)
    if (
        relative.is_absolute()
        or not relative.parts
        or relative == Path(".")
        or ".." in relative.parts
    ):
        return None, "invalid_output_path: relative component directory required"
    try:
        root = base.resolve(strict=True)
        output = root.joinpath(*relative.parts)
        if any(
            root.joinpath(*relative.parts[:index]).is_symlink()
            for index in range(1, len(relative.parts) + 1)
        ):
            return None, "invalid_output_path: output path contains a symlinked component"
        if output.is_symlink():
            return None, "invalid_output_path: output directory is a symlink"
        if output.exists():
            if not output.is_dir():
                return None, "output_collision: output path is not a directory"
            if not resume:
                return None, f"output_collision: output already exists: {request.output_directory}"
            return output, None
        if resume:
            return None, "cannot resume: output directory does not exist"
        output.mkdir(parents=True, exist_ok=False)
        return output, None
    except (OSError, RuntimeError, ValueError) as error:
        return None, f"invalid_output_path: {error}"


def _provenance(request: ComponentRequest, recipe: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "component_id": COMPONENT_ID,
        "component_version": COMPONENT_VERSION,
        "commit": review_execute._repo_commit(),
        "request_digest": _canonical_digest(_request_identity(request)),
        "recipe_digest": experiment_recipe_canonical_digest(recipe),
        "source_refs": [
            {"artifact_id": source.artifact_id, "uri": source.uri, "format": source.format}
            for source in request.sources
        ],
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "evidence_boundary": DIAGNOSTIC_EVIDENCE_BOUNDARY,
        "benchmark_success": False,
        "scientific_claim_allowed": False,
        "dependent_family_status": DEPENDENT_FAMILY_STATUS,
    }


def _admission_status(reason: str) -> str:
    """Map canonical admission diagnostics to truthful result status.

    Returns:
        ``unavailable`` for a missing/stale capability and ``failed`` for an
        integrity or malformed-input failure.
    """

    lowered = reason.lower()
    if any(
        marker in lowered
        for marker in (
            "explicit launcher admission",
            "source_missing",
            "source_escaped_root",
            "allowed_root",
            "receipt_stale",
            "source_mutated",
            "preservation_receipt_stale",
        )
    ):
        return "unavailable" if "mutated" not in lowered else "failed"
    return "failed"


def run(
    request: ComponentRequest | Mapping[str, Any],
    *,
    base: Path | None = None,
    resume: bool = False,
    autonomous: bool = False,
    read_only: bool = False,
    admission_config: review_execute.ExecutorAdmissionConfig | Mapping[str, Any] | None = None,
    executor: Any | None = None,
    source_admission: Mapping[str, Any] | None = None,
    cancel: Callable[[], bool] | Any | None = None,
) -> ComponentResult:
    """Run one authorised bounded experiment session.

    ``autonomous=True`` (or the equivalent request policy) is the explicit
    caller start boundary.  A read-only request always returns without
    launching an executor.  For the native path, ``admission_config`` is
    mandatory and is checked by the merged SREV-22/#9417 resolver before the
    first dispatch.  An injected executor may be used with an already
    verified ``source_admission`` proof, which keeps unit tests independent of
    simulator availability without weakening the production path.

    Returns:
        A validated component result with complete artifacts only when the
        finite session settled successfully.
    """

    if not isinstance(request, ComponentRequest):
        try:
            request = component_request_from_dict(request)
        except (ReviewContractsValidationError, TypeError, ValueError) as error:
            return ComponentResult(
                request_id="unknown",
                component_id=COMPONENT_ID,
                status="failed",
                reason=f"invalid_input: {error}",
            )
    if request.component_id != COMPONENT_ID:
        return ComponentResult(
            request_id=request.request_id,
            component_id=COMPONENT_ID,
            status="unavailable",
            reason=f"unsupported component: {request.component_id}",
        )
    try:
        validated = _validate_input(
            request, autonomous=autonomous, read_only=read_only, resume=resume
        )
    except ExperimentLoopError as error:
        return ComponentResult(
            request_id=request.request_id,
            component_id=COMPONENT_ID,
            status="failed"
            if not str(error).startswith("unsupported_measurement")
            else "unavailable",
            reason=str(error),
        )
    if validated.policy.read_only:
        return ComponentResult(
            request_id=request.request_id,
            component_id=COMPONENT_ID,
            status="unavailable",
            reason="read_only_never_executes",
        )
    if not validated.policy.autonomous:
        return ComponentResult(
            request_id=request.request_id,
            component_id=COMPONENT_ID,
            status="unavailable",
            reason="autonomous_start_authorization_required",
        )
    root = base if base is not None else Path.cwd()
    provenance = _provenance(request, validated.recipe)
    native_config: dict[str, Any] | None = None
    normalized_admission: review_execute.ExecutorAdmissionConfig | None = None
    if executor is None:
        native_config = _native_executor_config(validated)
        source_document, normalized_admission, admission_error = _preflight_native_admission(
            request,
            recipe=validated.recipe,
            executor_config=native_config,
            admission_config=admission_config,
        )
        if admission_error is not None or source_document is None or normalized_admission is None:
            return ComponentResult(
                request_id=request.request_id,
                component_id=COMPONENT_ID,
                status=_admission_status(admission_error or "source_admission: no admitted proof"),
                reason=admission_error or "source_admission: no admitted proof",
            )
    else:
        source_document = dict(source_admission or {})
        if (
            not source_document
            or source_document.get("status") not in {"admitted", "provided"}
            or source_document.get("scientific_claim_allowed") is True
        ):
            return ComponentResult(
                request_id=request.request_id,
                component_id=COMPONENT_ID,
                status="unavailable",
                reason="source_admission: injected executor requires an admitted source proof",
            )
    output_dir, output_error = _prepare_output_directory(request, root, resume=resume)
    if output_error is not None or output_dir is None:
        return ComponentResult(
            request_id=request.request_id,
            component_id=COMPONENT_ID,
            status="failed",
            reason=output_error or "invalid_output_path",
        )
    if executor is None:
        assert native_config is not None and normalized_admission is not None
        executor = _NativeExecutorAdapter(
            request,
            base=output_dir,
            executor_config=native_config,
            recipe=validated.recipe,
            admission_config=normalized_admission,
            resume=resume,
        )
    try:
        loop = ExperimentLoop(
            request,
            recipe=validated.recipe,
            budget=validated.budget,
            policy=validated.policy,
            journal_path=output_dir / SESSION_JOURNAL_FILENAME,
            executor=executor,
            source_admission=source_document,
            provenance=provenance,
            session_id=validated.session_id,
            resume=resume,
            cancel=cancel,
        )
        if validated.answerability is not None:
            loop._journal["answerability"] = _answerability_document(validated.answerability)
            loop._persist()
        return loop.run()
    except ExperimentLoopError as error:
        return ComponentResult(
            request_id=request.request_id,
            component_id=COMPONENT_ID,
            status="failed",
            reason=str(error),
        )
    except (OSError, ValueError, TypeError) as error:
        return ComponentResult(
            request_id=request.request_id,
            component_id=COMPONENT_ID,
            status="failed",
            reason=f"execution_failed: {type(error).__name__}: {error}",
        )


def _result_document(result: ComponentResult) -> dict[str, Any]:
    return {"schema_version": COMPONENT_RESULT_SCHEMA_VERSION, **asdict(result)}


# Compatibility aliases for callers that name the product-level operation
# rather than the SREV component identifier.
ExperimentSession = ExperimentLoop
run_session = run


def _read_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SREV-24 bounded follow-up experiments.")
    parser.add_argument("--input", required=True, help="component-request.v1 JSON file")
    parser.add_argument(
        "--config", default=None, help="optional config JSON merged over request config"
    )
    parser.add_argument(
        "--admission-config", default=None, help="launcher-owned executor-admission.v1 JSON"
    )
    parser.add_argument("--output", required=True, help="relative output directory")
    parser.add_argument("--base", default=None, help="base directory for output")
    parser.add_argument("--resume", action="store_true", help="resume an existing session journal")
    parser.add_argument(
        "--autonomous", action="store_true", help="authorize execution for this invocation"
    )
    parser.add_argument("--read-only", action="store_true", help="inspect policy; never execute")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; return zero only for a complete loop.

    Returns:
        Process exit code.
    """

    args = _build_parser().parse_args(argv)
    try:
        payload = _read_json(Path(args.input))
    except (OSError, ValueError, RecursionError):
        print(
            json.dumps(
                _result_document(
                    ComponentResult(
                        "unknown",
                        COMPONENT_ID,
                        "failed",
                        reason="invalid_input: request JSON cannot be parsed safely",
                    )
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 1
    if not isinstance(payload, dict):
        print(
            json.dumps(
                _result_document(
                    ComponentResult(
                        "unknown",
                        COMPONENT_ID,
                        "failed",
                        reason="invalid_input: request must be an object",
                    )
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 1
    if args.config is not None:
        try:
            override = _read_json(Path(args.config))
        except (OSError, ValueError, RecursionError):
            print(
                json.dumps(
                    _result_document(
                        ComponentResult(
                            "unknown",
                            COMPONENT_ID,
                            "failed",
                            reason="invalid_input: config JSON cannot be parsed safely",
                        )
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 1
        if not isinstance(override, dict) or not isinstance(payload.get("config", {}), dict):
            print(
                json.dumps(
                    _result_document(
                        ComponentResult(
                            "unknown",
                            COMPONENT_ID,
                            "failed",
                            reason="invalid_input: config must be an object",
                        )
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 1
        payload = {**payload, "config": {**payload.get("config", {}), **override}}
    payload = {
        **payload,
        "output_directory": args.output,
        "component_id": COMPONENT_ID,
    }
    try:
        request = component_request_from_dict(payload, source=args.input)
    except (ReviewContractsValidationError, RecursionError, ValueError, TypeError):
        print(
            json.dumps(
                _result_document(
                    ComponentResult(
                        "unknown",
                        COMPONENT_ID,
                        "failed",
                        reason="invalid_input: request does not satisfy component-request.v1",
                    )
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 1
    admission: Any = None
    if args.admission_config is not None:
        try:
            admission = _read_json(Path(args.admission_config))
        except (OSError, ValueError, RecursionError):
            result = ComponentResult(
                request.request_id,
                COMPONENT_ID,
                "failed",
                reason="invalid_input: admission config cannot be parsed safely",
            )
            print(json.dumps(_result_document(result), indent=2, sort_keys=True))
            return 1
    result = run(
        request,
        base=Path(args.base) if args.base is not None else None,
        resume=args.resume,
        autonomous=args.autonomous,
        read_only=args.read_only,
        admission_config=admission,
    )
    print(json.dumps(_result_document(result), indent=2, sort_keys=True))
    return 0 if result.status == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
