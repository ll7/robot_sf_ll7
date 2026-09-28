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
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from itertools import pairwise
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
SESSION_LOCK_FILENAME = ".experiment-loop.lock"
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
INCOMPLETE_CANDIDATE_STATE = "incomplete"
RESERVABLE_PAIR_EXECUTIONS = frozenset({1, RESERVED_CONTROL_TREATMENT_PAIR})
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

_NATIVE_LEDGER_KEYS = frozenset(
    {
        "schema_version",
        "request_id",
        "component_id",
        "recipe_id",
        "request_digest",
        "recipe_digest",
        "config_identity_digest",
        "budget",
        "attempts",
        "executions_consumed",
        "candidate_reports",
        "traces",
        "evidence_boundary",
        "scientific_claim_allowed",
        "dependent_family_status",
        "source_admission",
        "wall_elapsed_s",
    }
)
_NATIVE_EXECUTE_REPORT_KEYS = frozenset(
    {
        "schema_version",
        "request_id",
        "component_id",
        "recipe_id",
        "source_identity",
        "source_admission",
        "evidence_boundary",
        "benchmark_success",
        "scientific_claim_allowed",
        "dependent_family_status",
        "candidates",
        "budget",
        "provenance",
    }
)
_NATIVE_PROVENANCE_KEYS = frozenset(
    {
        "component_id",
        "component_version",
        "commit",
        "python",
        "platform",
        "created_utc",
        "evidence_boundary",
        "benchmark_success",
        "scientific_claim_allowed",
        "dependent_family_status",
        "recipe_id",
        "source_identity",
        "request_digest",
        "recipe_digest",
        "config_digest",
        "source_admission",
        "sources",
        "output_directory",
    }
)
_NATIVE_COMPLETE_REPORT_KEYS = frozenset(
    {
        "intervention_id",
        "factor",
        "status",
        "verdict",
        "verdict_reason",
        "control_metrics",
        "treatment_metrics",
        "control_activated",
        "treatment_activated",
        "nonintervened_config_match",
    }
)
_NATIVE_FAILED_REPORT_KEYS = frozenset(
    {"intervention_id", "factor", "status", "reason", "nonintervened_config_match"}
)
_NATIVE_FAILED_REPORT_WITH_CONTROL_KEYS = _NATIVE_FAILED_REPORT_KEYS | {"control_metrics"}
_NATIVE_UNAVAILABLE_REPORT_KEYS = frozenset({"intervention_id", "factor", "status", "reason"})


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


def _validate_injected_source_proof(
    request: ComponentRequest,
    recipe: Mapping[str, Any],
    proof: Mapping[str, Any] | None,
    *,
    base: Path,
) -> dict[str, Any] | None:
    """Validate the narrow offline-injection admission boundary.

    The injected executor is a test/offline seam, not an admission authority.
    A caller-provided ``{"status": "admitted"}`` marker is therefore never
    sufficient.  The proof must bind the exact request and recipe, identify a
    regular source file below the invocation base, and carry a digest measured
    from that file.  Keeping the root below ``base`` prevents an offline caller
    from turning this seam into an arbitrary host-file reader (for example
    ``/etc/passwd``).

    Returns:
        A copied proof mapping when it satisfies the boundary, otherwise
        ``None``.
    """

    if not isinstance(proof, Mapping):
        return None
    if proof.get("status") != "admitted":
        return None
    if proof.get("evidence_boundary") != DIAGNOSTIC_EVIDENCE_BOUNDARY:
        return None
    if proof.get("scientific_claim_allowed") is not False:
        return None
    if proof.get("dependent_family_status") != DEPENDENT_FAMILY_STATUS:
        return None
    if proof.get("request_digest") != _canonical_digest(_request_identity(request)):
        return None
    if proof.get("recipe_digest") != experiment_recipe_canonical_digest(recipe):
        return None
    source = proof.get("source")
    if not isinstance(source, Mapping):
        return None
    source_root_value = proof.get("source_root")
    if not isinstance(source_root_value, str) or not source_root_value:
        return None
    source_uri = source.get("uri")
    if not isinstance(source_uri, str) or not source_uri:
        return None
    source_relative = Path(source_uri)
    if (
        source_relative.is_absolute()
        or not source_relative.parts
        or source_relative == Path(".")
        or ".." in source_relative.parts
        or "\\" in source_uri
    ):
        return None
    if not isinstance(source.get("sha256"), str) or len(source["sha256"]) != 64:
        return None
    if any(not isinstance(source.get(key), str) for key in ("artifact_id", "format")):
        return None
    request_source = next(
        (
            item
            for item in request.sources
            if item.artifact_id == source.get("artifact_id")
            and item.uri == source_uri
            and item.format == source.get("format")
        ),
        None,
    )
    if request_source is None:
        return None
    if request_source.sha256 and request_source.sha256 != source.get("sha256"):
        return None
    source_identity = recipe.get("source_identity")
    if isinstance(source_identity, Mapping):
        recipe_source = source_identity.get("source_ref")
        if isinstance(recipe_source, Mapping) and (
            recipe_source.get("artifact_id") != source.get("artifact_id")
            or recipe_source.get("uri") != source_uri
            or recipe_source.get("format") != source.get("format")
        ):
            return None
    try:
        base_root = base.resolve(strict=True)
        source_root = Path(source_root_value).resolve(strict=True)
        source_root.relative_to(base_root)
        source_path = source_root.joinpath(*source_relative.parts)
        resolved_source = source_path.resolve(strict=True)
        resolved_source.relative_to(source_root)
        if source_path.is_symlink() or not resolved_source.is_file():
            return None
        actual_digest = hashlib.sha256(resolved_source.read_bytes()).hexdigest()
    except (OSError, RuntimeError, ValueError):
        return None
    if actual_digest != source.get("sha256"):
        return None
    return dict(proof)


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


def _native_fidelity_attempts(result: Mapping[str, Any]) -> int | None:
    """Return child fidelity checks carried by one normalized control result."""

    value = result.get("native_fidelity_attempts", 0)
    if isinstance(value, bool) or not isinstance(value, int) or value not in {0, 1}:
        return None
    return int(value)


def _has_valid_result_metrics(value: Any) -> bool:
    """Require a completed operation result to carry finite telemetry.

    Returns:
        ``True`` when the result has a non-empty finite metric mapping.
    """

    status, payload = _status_from_result(value)
    if status != "ok":
        return False
    metrics = _extract_metrics(payload)
    return bool(metrics) and all(
        isinstance(metric, (int, float))
        and not isinstance(metric, bool)
        and math.isfinite(float(metric))
        for metric in metrics.values()
    )


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
) -> bool | None:
    """Return only an executor-declared activation value.

    The compatibility helper intentionally does not infer activation from
    motion metrics.  ``None`` is the truthful result when the executor omits
    its activation contract.
    """

    del control, motion_epsilon
    return _explicit_activation(result)


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


def _pair_telemetry(
    control: Mapping[str, Any],
    treatment: Mapping[str, Any],
    *,
    factor: str,
    measurement: Mapping[str, Any],
    motion_epsilon: float,
) -> dict[str, Any]:
    """Recompute the diagnostic pair verdict and measured activation flags.

    Returns:
        Canonical outcome, reason, activation flags, and metric availability.
    """

    metric_name = str(measurement["name"])
    control_metrics = _extract_metrics(control)
    treatment_metrics = _extract_metrics(treatment)
    # Finite motion/metric values do not prove that the intended intervention
    # mechanism was active.  Activation is a separate executor contract and
    # must be explicit for both sides of the pair before a measured verdict
    # can be emitted.  Keep ``None`` visible for missing telemetry instead of
    # manufacturing a false activation bit from sparse metrics.
    control_activated = _explicit_activation(control)
    treatment_activated = _explicit_activation(treatment)
    if control_activated is None or treatment_activated is None:
        missing_sides = [
            side
            for side, value in (
                ("control", control_activated),
                ("treatment", treatment_activated),
            )
            if value is None
        ]
        return {
            "outcome": "inconclusive",
            "reason": "activation_missing: explicit activation telemetry is required for "
            + ", ".join(missing_sides),
            "control_activated": control_activated,
            "treatment_activated": treatment_activated,
            "activation_available": False,
            "measurement_available": False,
        }
    if metric_name not in control_metrics or metric_name not in treatment_metrics:
        return {
            "outcome": "inconclusive",
            "reason": f"measurement_missing: {metric_name}",
            "control_activated": control_activated,
            "treatment_activated": treatment_activated,
            "activation_available": True,
            "measurement_available": False,
        }
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
                expected_direction=str(measurement["expected_direction"]),
            ),
        )
        verdict = pair_result.verdict
        reason = pair_result.reason
    except (KeyError, TypeError, ValueError) as error:
        verdict = "inconclusive"
        reason = f"pair_evaluation_unavailable: {error}"
    if verdict not in OUTCOMES:
        verdict = "inconclusive"
    return {
        "outcome": verdict,
        "reason": reason,
        "control_activated": control_activated,
        "treatment_activated": treatment_activated,
        "activation_available": True,
        "measurement_available": True,
    }


def _reason(result: Mapping[str, Any], fallback: str) -> str:
    value = result.get("reason", result.get("error", fallback))
    return str(value) if value is not None else fallback


def _is_negative_outcome(outcome: Mapping[str, Any]) -> bool:
    """Derive the negative export classification from the outcome itself.

    The journal flag is a convenience for readers, not an authority.  Every
    failed, unavailable, or cancelled operation is negative by definition;
    only a complete pair that actually survived is non-negative.

    Returns:
        ``True`` when the canonical outcome classification is negative.
    """

    status = outcome.get("status")
    if status in NEGATIVE_OUTCOME_STATUSES:
        return True
    return status == "complete" and outcome.get("outcome") != "survived"


def _is_retryable_failed_operation(operation: Mapping[str, Any]) -> bool:
    """Return whether one durable operation can be retried on an extension."""

    if operation.get("state") != "failed":
        return False
    status, result = _status_from_result(operation.get("result"))
    return status == "failed" and (result.get("retryable") is True or result.get("retry") is True)


def _remaining_pair_executions_for_operations(
    operations: list[Mapping[str, Any]], *, max_retries: int
) -> int | None:
    """Count the exact next dispatches needed to settle one pair.

    Returns:
        The number of authorized operations still needed, or ``None`` when
        the retained operation history cannot continue under the retry policy.
    """

    operations_by_kind: dict[str, list[Mapping[str, Any]]] = {
        "control": [],
        "treatment": [],
    }
    for operation in operations:
        kind = operation.get("kind")
        if kind in operations_by_kind:
            operations_by_kind[str(kind)].append(operation)
    required = 0
    for kind in ("control", "treatment"):
        kind_operations = operations_by_kind[kind]
        if not kind_operations:
            required += 1
            continue
        final_operation = max(kind_operations, key=lambda item: int(item["attempt"]))
        state, _result = _status_from_result(final_operation.get("result"))
        if final_operation.get("state") in {"reserved", "dispatching"}:
            required += 1
            continue
        if final_operation.get("state") == "completed" and state == "ok":
            continue
        if (
            _is_retryable_failed_operation(final_operation)
            and int(final_operation.get("attempt", 0)) <= max_retries
        ):
            required += 1
            continue
        return None
    return required


class ExperimentLoop:
    """Run one finite candidate set against a durable session journal.

    The constructor intentionally receives an already admitted source
    provenance document.  The public :func:`run` performs the canonical
    SREV-22 admission preflight before constructing this class; direct injected
    callers must pass the same measured request/recipe-bound proof used by the
    offline seam.
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
        proof_base: Path | None = None,
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
        if not isinstance(executor, _NativeExecutorAdapter):
            validated_source_admission = _validate_injected_source_proof(
                request,
                self.recipe,
                source_admission,
                base=proof_base or journal_path.parent,
            )
            if validated_source_admission is None:
                raise ExperimentLoopError(
                    "source_admission: injected executor requires a validated "
                    "request/recipe-bound source proof"
                )
            self.source_admission = validated_source_admission
        else:
            self.source_admission = dict(source_admission)
        self.provenance = dict(provenance or {})
        # Answerability is derived from the immutable request/config input,
        # never from a caller-asserted or mutable journal field.  The public
        # entrypoint validates this optional contract before constructing the
        # loop; direct callers use the same request-bound source.
        configured_answerability = (
            self.request.config.get("answerability")
            if isinstance(self.request.config, Mapping)
            else None
        )
        if configured_answerability is not None and not isinstance(
            configured_answerability, Mapping
        ):
            raise ExperimentLoopError("invalid_config: answerability must be a mapping")
        self._answerability_document = _answerability_document(configured_answerability)
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
        self._session_lock_path = journal_path.with_name(SESSION_LOCK_FILENAME)
        self._session_lock_fd = -1
        # A fresh invocation reports an ordinary output collision before it
        # claims a lock.  Resume claims the lock first, so two controllers can
        # never read/repair/dispatch the same journal concurrently.
        if not resume and journal_path.exists():
            raise ExperimentLoopError(f"output_collision: journal already exists: {journal_path}")
        try:
            self._acquire_session_lock()
            self._journal = self._new_journal()
            if resume:
                self._load_journal()
            else:
                self._persist()
            self._bind_executor_operation_map()
        except BaseException:
            self._release_session_lock()
            raise

    def _acquire_session_lock(self) -> None:
        """Claim exclusive controller ownership for this session directory."""

        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(self._session_lock_path, flags, 0o600)
        except FileExistsError as error:
            raise ExperimentLoopError(
                f"session_lock_owned: another controller owns {self._session_lock_path}"
            ) from error
        except OSError as error:
            raise ExperimentLoopError(f"session_lock_unavailable: {error}") from error
        owner = {
            "pid": os.getpid(),
            "host": platform.node(),
            "session_id": self.session_id,
            "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        try:
            content = _json_bytes(owner)
            offset = 0
            while offset < len(content):
                offset += os.write(descriptor, content[offset:])
            os.fsync(descriptor)
            self._session_lock_fd = descriptor
        except BaseException:
            os.close(descriptor)
            self._session_lock_path.unlink(missing_ok=True)
            raise

    def _release_session_lock(self) -> None:
        """Release only the lock inode owned by this controller."""

        descriptor = self._session_lock_fd
        if descriptor < 0:
            return
        self._session_lock_fd = -1
        try:
            own_inode = os.fstat(descriptor).st_ino
            try:
                same_inode = os.stat(self._session_lock_path).st_ino == own_inode
            except OSError:
                same_inode = False
            if same_inode:
                self._session_lock_path.unlink(missing_ok=True)
        finally:
            os.close(descriptor)

    def _bind_executor_operation_map(self) -> None:
        method = getattr(self.executor, "bind_operation_map", None)
        if callable(method):
            method(self._journal.get("operations", []))

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
            "elapsed_floor_s": 0.0,
            "status": "running",
            "stop_reason": "",
            "provenance": dict(self.provenance),
            "answerability": self._answerability_document,
            "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "updated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        }

    def _elapsed(self) -> float:
        return self._elapsed_base + max(0.0, time.monotonic() - self._started_at)

    def _persist(self) -> None:
        elapsed = self._elapsed()
        prior_floor = self._journal.get("elapsed_floor_s", 0.0)
        if (
            isinstance(prior_floor, bool)
            or not isinstance(prior_floor, (int, float))
            or not math.isfinite(float(prior_floor))
            or float(prior_floor) < 0.0
        ):
            raise ExperimentLoopError("elapsed accounting floor is invalid")
        persisted_elapsed = max(float(elapsed), float(prior_floor))
        self._journal["elapsed_s"] = round(persisted_elapsed, 6)
        self._journal["elapsed_floor_s"] = round(persisted_elapsed, 6)
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
        if "answerability" not in payload or payload["answerability"] != expected.get(
            "answerability"
        ):
            raise ExperimentLoopError("cannot resume: journal answerability mismatch")
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
        journal_status = payload.get("status")
        if journal_status not in {
            "running",
            "complete",
            "partial",
            "failed",
            "unavailable",
            "cancelled",
        }:
            raise ExperimentLoopError("cannot resume: invalid journal status")
        if not isinstance(payload.get("stop_reason"), str):
            raise ExperimentLoopError("cannot resume: journal stop reason is malformed")
        prior_candidate_ids = {
            str(item.get("intervention_id"))
            for item in prior_order
            if isinstance(item, Mapping) and isinstance(item.get("intervention_id"), str)
        }
        candidate_by_id = {
            str(item["intervention_id"]): item
            for item in prior_order
            if isinstance(item, Mapping) and isinstance(item.get("intervention_id"), str)
        }
        candidate_order_index = {
            candidate_id: index for index, candidate_id in enumerate(candidate_by_id)
        }
        if set(payload["candidates"]) != prior_candidate_ids:
            raise ExperimentLoopError("cannot resume: journal candidate state mismatch")
        for candidate_id, candidate_state in payload["candidates"].items():
            if not isinstance(candidate_state, Mapping):
                raise ExperimentLoopError("cannot resume: malformed candidate state")
            if candidate_state.get("intervention_id") != candidate_id:
                raise ExperimentLoopError("cannot resume: candidate identity mismatch")
            candidate_document = candidate_by_id[candidate_id]
            if candidate_state.get("priority") != candidate_document.get(
                "priority"
            ) or candidate_state.get("factor") != candidate_document.get("factor", ""):
                raise ExperimentLoopError("cannot resume: candidate metadata mismatch")
            if candidate_state.get("state") not in {
                "pending",
                "reserved",
                "dispatching",
                "complete",
                "failed",
                "unavailable",
                "cancelled",
                INCOMPLETE_CANDIDATE_STATE,
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
        operation_by_id: dict[str, Mapping[str, Any]] = {}
        operation_ids_by_candidate: dict[str, list[str]] = {
            candidate_id: [] for candidate_id in prior_candidate_ids
        }
        operations_by_candidate_kind: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
        dispatched_by_kind = {"control": 0, "treatment": 0}
        failed_operations = 0
        retry_operations = 0
        active_reservations: set[str] = set()
        first_operation_index_by_candidate: dict[str, int] = {}
        last_attempt_by_candidate_kind: dict[tuple[str, str], int] = {}
        control_seen_by_candidate: set[str] = set()
        treatment_seen_by_candidate: set[str] = set()
        previous_sequence: int | None = None
        for operation in payload["operations"]:
            operation_index = len(operation_by_id)
            if not isinstance(operation, dict) or not isinstance(
                operation.get("operation_id"), str
            ):
                raise ExperimentLoopError("cannot resume: malformed operation record")
            operation_id = str(operation["operation_id"])
            if operation_id in operation_ids:
                raise ExperimentLoopError("cannot resume: duplicate operation ID")
            operation_ids.add(operation_id)
            operation_by_id[operation_id] = operation
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
            sequence = operation.get("sequence", operation_index)
            if (
                not isinstance(sequence, int)
                or isinstance(sequence, bool)
                or sequence < 0
                or (previous_sequence is not None and sequence <= previous_sequence)
            ):
                raise ExperimentLoopError("cannot resume: operation chronology is inconsistent")
            operation["sequence"] = sequence
            previous_sequence = sequence
            first_operation_index_by_candidate.setdefault(candidate_id, operation_index)
            prior_attempt = last_attempt_by_candidate_kind.get((candidate_id, kind))
            if prior_attempt is not None and attempt <= prior_attempt:
                raise ExperimentLoopError("cannot resume: operation retry sequence is inconsistent")
            last_attempt_by_candidate_kind[(candidate_id, kind)] = attempt
            if kind == "control":
                if candidate_id in treatment_seen_by_candidate:
                    raise ExperimentLoopError(
                        "cannot resume: control operation follows treatment operation"
                    )
                control_seen_by_candidate.add(candidate_id)
            else:
                if candidate_id not in control_seen_by_candidate:
                    raise ExperimentLoopError(
                        "cannot resume: treatment operation lacks prior control sequence"
                    )
                treatment_seen_by_candidate.add(candidate_id)
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
            operation_ids_by_candidate[candidate_id].append(operation_id)
            operations_by_candidate_kind.setdefault((candidate_id, kind), []).append(operation)
            if dispatch_count:
                dispatched_by_kind[kind] += dispatch_count
            state = operation.get("state")
            result = operation.get("result")
            result_status, result_document = _status_from_result(result)
            native_fidelity_attempts = _native_fidelity_attempts(result_document)
            if native_fidelity_attempts is None or (native_fidelity_attempts and kind != "control"):
                raise ExperimentLoopError("cannot resume: invalid native fidelity accounting")
            if dispatch_count == 0:
                if state != "reserved" or result is not None:
                    raise ExperimentLoopError("cannot resume: reserved operation is inconsistent")
            elif state == "reserved":
                raise ExperimentLoopError("cannot resume: dispatched operation is still reserved")
            elif state == "dispatching":
                if result is not None:
                    if result_status != "failed" or not str(
                        result_document.get("reason", "")
                    ).startswith("dispatch_interrupted:"):
                        raise ExperimentLoopError(
                            "cannot resume: dispatching operation result is inconsistent"
                        )
            else:
                expected_state = {
                    "ok": "completed",
                    "failed": "failed",
                    "cancelled": "cancelled",
                    "unavailable": "unavailable",
                }.get(result_status)
                if expected_state != state:
                    raise ExperimentLoopError("cannot resume: operation result/state mismatch")
            if state == "completed" and not _has_valid_result_metrics(result):
                raise ExperimentLoopError("cannot resume: completed operation result is invalid")
            if state == "failed":
                if operation.get("failure_accounted") is not True:
                    raise ExperimentLoopError("cannot resume: failed operation is unaccounted")
                failed_operations += 1
            elif operation.get("failure_accounted") is True:
                raise ExperimentLoopError(
                    "cannot resume: non-failed operation is failure-accounted"
                )
            if dispatch_count and attempt > 1:
                retry_operations += 1
        observed_candidate_sequence = sorted(
            first_operation_index_by_candidate,
            key=first_operation_index_by_candidate.__getitem__,
        )
        expected_candidate_sequence = sorted(
            observed_candidate_sequence,
            key=candidate_order_index.__getitem__,
        )
        if observed_candidate_sequence != expected_candidate_sequence:
            raise ExperimentLoopError(
                "cannot resume: candidate operation chronology is inconsistent"
            )
        for (candidate_id, kind), candidate_kind_operations in operations_by_candidate_kind.items():
            ordered_operations = sorted(
                candidate_kind_operations, key=lambda operation: int(operation["attempt"])
            )
            if [int(operation["attempt"]) for operation in ordered_operations] != list(
                range(1, len(ordered_operations) + 1)
            ):
                raise ExperimentLoopError(
                    f"cannot resume: {candidate_id} {kind} retry attempts are not contiguous"
                )
            for predecessor, _retry in pairwise(ordered_operations):
                predecessor_status, predecessor_result = _status_from_result(
                    predecessor.get("result")
                )
                if (
                    predecessor.get("state") != "failed"
                    or predecessor_status != "failed"
                    or (
                        predecessor_result.get("retryable") is not True
                        and predecessor_result.get("retry") is not True
                    )
                ):
                    raise ExperimentLoopError(
                        f"cannot resume: {candidate_id} {kind} retry predecessor is invalid"
                    )
        if dispatch_total != consumed:
            raise ExperimentLoopError(
                "cannot resume: operation accounting does not match dispatches"
            )
        outcomes = payload.get("outcomes")
        if not isinstance(outcomes, list):
            raise ExperimentLoopError("cannot resume: journal outcomes are malformed")
        outcome_ids: set[str] = set()
        outcome_by_id: dict[str, Mapping[str, Any]] = {}
        for outcome in outcomes:
            if not isinstance(outcome, Mapping) or not isinstance(
                outcome.get("intervention_id"), str
            ):
                raise ExperimentLoopError("cannot resume: malformed outcome record")
            outcome_id = str(outcome["intervention_id"])
            if outcome_id not in prior_candidate_ids or outcome_id in outcome_ids:
                raise ExperimentLoopError("cannot resume: duplicate or unknown outcome record")
            if outcome.get("status") not in TERMINAL_CANDIDATE_STATES:
                raise ExperimentLoopError("cannot resume: invalid outcome status")
            if (
                outcome.get("status") in NEGATIVE_OUTCOME_STATUSES
                and outcome.get("negative") is not True
            ):
                raise ExperimentLoopError(
                    "cannot resume: non-complete outcome negative flag is invalid"
                )
            if outcome.get("status") in NEGATIVE_OUTCOME_STATUSES and any(
                key in outcome for key in ("activation", "control_activated", "treatment_activated")
            ):
                raise ExperimentLoopError(
                    "cannot resume: non-complete outcome activation is invalid"
                )
            outcome_operation_ids = outcome.get("operation_ids", [])
            if not isinstance(outcome_operation_ids, list) or not all(
                isinstance(item, str) for item in outcome_operation_ids
            ):
                raise ExperimentLoopError("cannot resume: malformed outcome operation IDs")
            outcome_ids.add(outcome_id)
            outcome_by_id[outcome_id] = outcome

        reservation_records = payload.get("reservations")
        if not isinstance(reservation_records, list):
            raise ExperimentLoopError("cannot resume: journal reservations are malformed")
        reservation_by_candidate: dict[str, Mapping[str, Any]] = {}
        for reservation in reservation_records:
            if not isinstance(reservation, Mapping):
                raise ExperimentLoopError("cannot resume: malformed reservation record")
            reservation_id = reservation.get("candidate_id")
            if (
                not isinstance(reservation_id, str)
                or reservation_id not in prior_candidate_ids
                or reservation_id in reservation_by_candidate
                or reservation.get("required_executions") not in RESERVABLE_PAIR_EXECUTIONS
                or reservation.get("state") not in {"reserved", "released"}
            ):
                raise ExperimentLoopError("cannot resume: invalid reservation record")
            reservation_by_candidate[reservation_id] = reservation

        # Validate the persisted aggregate against the reservation records as
        # they were written before any recovery normalization.  A process can
        # crash after settling control but before dispatching treatment: the
        # candidate reservation is still the original full-pair amount while
        # the durable operation history authorizes only the remaining side.
        # Keep that crash window admissible, then replace the aggregate with
        # the canonical post-normalization amount below before any dispatch.
        persisted_reserved_executions = int(payload.get("reserved_executions", 0))
        persisted_reservation_total = 0

        for candidate_id, candidate_state in payload["candidates"].items():
            candidate_operations = operation_ids_by_candidate[candidate_id]
            candidate_operation_ids = candidate_state.get("operation_ids", [])
            if candidate_operation_ids != candidate_operations or len(
                set(candidate_operation_ids)
            ) != len(candidate_operation_ids):
                raise ExperimentLoopError("cannot resume: candidate operation index mismatch")
            attempts = candidate_state.get("attempts")
            if (
                not isinstance(attempts, int)
                or isinstance(attempts, bool)
                or attempts
                != sum(
                    int(operation_by_id[item].get("dispatch_count", 0))
                    for item in candidate_operations
                )
            ):
                raise ExperimentLoopError("cannot resume: candidate attempt accounting mismatch")
            candidate_outcome = outcome_by_id.get(candidate_id)
            candidate_state_name = candidate_state.get("state")
            if candidate_outcome is None:
                if candidate_state_name in TERMINAL_CANDIDATE_STATES:
                    raise ExperimentLoopError("cannot resume: terminal candidate lacks outcome")
                if candidate_state_name == "pending" and candidate_operations:
                    # A widened retry ceiling deliberately reopens a failed
                    # candidate while retaining every prior operation for
                    # accounting and idempotent recovery.  Such a pending
                    # candidate is valid only when all retained operations
                    # are settled and at least one is a retryable failure.
                    if not (
                        candidate_state.get("retry_reopened") is True
                        and all(
                            operation_by_id[item].get("state")
                            in {"completed", "failed", "cancelled", "unavailable"}
                            for item in candidate_operations
                        )
                        and any(
                            _is_retryable_failed_operation(operation_by_id[item])
                            for item in candidate_operations[-1:]
                        )
                    ):
                        raise ExperimentLoopError("cannot resume: pending candidate has operations")
                elif candidate_state_name == INCOMPLETE_CANDIDATE_STATE:
                    if (
                        journal_status not in {"running", "partial"}
                        or candidate_state.get("incomplete_reason")
                        not in {
                            "execution_budget_exhausted",
                            "wall_timeout",
                        }
                        or not candidate_operations
                        or any(
                            operation_by_id[item].get("state")
                            not in {"completed", "failed", "cancelled", "unavailable"}
                            for item in candidate_operations
                        )
                    ):
                        raise ExperimentLoopError(
                            "cannot resume: incomplete candidate state is inconsistent"
                        )
            elif candidate_state_name != candidate_outcome.get("status"):
                raise ExperimentLoopError("cannot resume: candidate state/outcome mismatch")
            if (
                candidate_outcome is not None
                and candidate_outcome.get("operation_ids") != candidate_operations
            ):
                final_operation_ids: list[str] = []
                for kind in ("control", "treatment"):
                    kind_operations = [
                        operation_by_id[item]
                        for item in candidate_operations
                        if operation_by_id[item].get("kind") == kind
                        and operation_by_id[item].get("state") != "reserved"
                    ]
                    if kind_operations:
                        final_operation = max(
                            kind_operations, key=lambda operation: int(operation["attempt"])
                        )
                        final_operation_ids.append(final_operation["operation_id"])
                if candidate_outcome.get("operation_ids") != final_operation_ids:
                    raise ExperimentLoopError("cannot resume: outcome operation index mismatch")
                # Journals written before retry history became part of the
                # outcome index remain recoverable when their final pair is
                # otherwise valid; normalize them on the first successful
                # load so subsequent writes retain every attempt.
                candidate_outcome["operation_ids"] = list(candidate_operations)
            if candidate_outcome is not None and candidate_outcome.get("status") == "complete":
                operation_kinds = {
                    str(operation_by_id[item].get("kind")) for item in candidate_operations
                }
                if operation_kinds != {"control", "treatment"}:
                    raise ExperimentLoopError(
                        "cannot resume: complete outcome lacks control/treatment pair"
                    )
                final_operations: dict[str, Mapping[str, Any]] = {}
                for kind in ("control", "treatment"):
                    kind_operations = [
                        operation_by_id[item]
                        for item in candidate_operations
                        if operation_by_id[item].get("kind") == kind
                    ]
                    final_operation = max(
                        kind_operations, key=lambda operation: int(operation["attempt"])
                    )
                    if (
                        final_operation.get("state") != "completed"
                        or _status_from_result(final_operation.get("result"))[0] != "ok"
                        or not _has_valid_result_metrics(final_operation.get("result"))
                    ):
                        raise ExperimentLoopError(
                            f"cannot resume: complete {kind} operation is not valid"
                        )
                    final_operations[kind] = final_operation
                outcome = candidate_outcome
                expected_candidate = candidate_by_id[candidate_id]
                if (
                    outcome.get("priority") != expected_candidate.get("priority")
                    or outcome.get("factor") != expected_candidate.get("factor", "")
                    or outcome.get("outcome") not in OUTCOMES
                ):
                    raise ExperimentLoopError("cannot resume: complete outcome shape is invalid")
                for result_key in ("control", "treatment"):
                    if not _has_valid_result_metrics(outcome.get(result_key)):
                        raise ExperimentLoopError(
                            f"cannot resume: complete outcome {result_key} result is invalid"
                        )
                    if _canonical_digest(outcome[result_key]) != _canonical_digest(
                        final_operations[result_key].get("result")
                    ):
                        raise ExperimentLoopError(
                            f"cannot resume: complete outcome {result_key} does not match operation"
                        )
                pair_observation = _pair_telemetry(
                    cast("Mapping[str, Any]", final_operations["control"].get("result")),
                    cast("Mapping[str, Any]", final_operations["treatment"].get("result")),
                    factor=str(expected_candidate.get("factor", "")),
                    measurement=self.measurement,
                    motion_epsilon=float(self.recipe.get("motion_epsilon_m", 0.05)),
                )
                if not pair_observation["activation_available"]:
                    raise ExperimentLoopError(
                        "cannot resume: complete outcome lacks explicit activation telemetry"
                    )
                if outcome.get("outcome") != pair_observation["outcome"]:
                    raise ExperimentLoopError(
                        "cannot resume: complete outcome contradicts pair telemetry"
                    )
                if (
                    not isinstance(outcome.get("reason"), str)
                    or outcome.get("reason") != (pair_observation["reason"])
                ):
                    raise ExperimentLoopError(
                        "cannot resume: complete outcome reason contradicts pair telemetry"
                    )
                if pair_observation["measurement_available"] and (
                    outcome.get("verdict") != pair_observation["outcome"]
                ):
                    raise ExperimentLoopError(
                        "cannot resume: complete verdict contradicts pair telemetry"
                    )
                if (
                    not pair_observation["measurement_available"]
                    and "verdict" in outcome
                    and outcome.get("verdict") != pair_observation["outcome"]
                ):
                    raise ExperimentLoopError(
                        "cannot resume: complete verdict contradicts pair telemetry"
                    )
                if not isinstance(outcome.get("negative"), bool):
                    raise ExperimentLoopError(
                        "cannot resume: complete outcome negative flag invalid"
                    )
                if outcome["negative"] != (pair_observation["outcome"] != "survived"):
                    raise ExperimentLoopError(
                        "cannot resume: complete negative flag contradicts pair telemetry"
                    )
                for key in ("control_activated", "treatment_activated"):
                    if key in outcome and not isinstance(outcome.get(key), bool):
                        raise ExperimentLoopError(
                            "cannot resume: complete outcome activation is invalid"
                        )
                if pair_observation["measurement_available"] and any(
                    outcome.get(key) != pair_observation[key]
                    for key in ("control_activated", "treatment_activated")
                ):
                    raise ExperimentLoopError(
                        "cannot resume: complete activation contradicts pair telemetry"
                    )
                activation = outcome.get("activation")
                if activation is not None and (
                    not isinstance(activation, Mapping)
                    or activation.get("measured") is not True
                    or any(
                        not isinstance(activation.get(key), bool)
                        for key in ("control", "treatment")
                    )
                ):
                    raise ExperimentLoopError(
                        "cannot resume: complete outcome activation shape is invalid"
                    )
                if (
                    isinstance(activation, Mapping)
                    and all(key in outcome for key in ("control_activated", "treatment_activated"))
                    and (
                        activation["control"] != outcome["control_activated"]
                        or activation["treatment"] != outcome["treatment_activated"]
                    )
                ):
                    raise ExperimentLoopError("cannot resume: complete outcome activation mismatch")
                if pair_observation["measurement_available"]:
                    if not isinstance(activation, Mapping) or (
                        activation.get("control") != pair_observation["control_activated"]
                        or activation.get("treatment") != pair_observation["treatment_activated"]
                    ):
                        raise ExperimentLoopError(
                            "cannot resume: complete activation contradicts pair telemetry"
                        )
                elif isinstance(activation, Mapping) and (
                    activation.get("control") != pair_observation["control_activated"]
                    or activation.get("treatment") != pair_observation["treatment_activated"]
                ):
                    raise ExperimentLoopError(
                        "cannot resume: complete activation contradicts pair telemetry"
                    )
            reservation = reservation_by_candidate.get(candidate_id)
            is_reserved = candidate_state_name in {"reserved", "dispatching"}
            if is_reserved:
                active_reservations.add(candidate_id)
                if (
                    reservation is None
                    or candidate_state.get("reservation") != reservation.get("required_executions")
                    or candidate_state.get("reservation") not in RESERVABLE_PAIR_EXECUTIONS
                ):
                    raise ExperimentLoopError(
                        "cannot resume: candidate reservation amount is invalid"
                    )
                persisted_reservation_total += int(reservation["required_executions"])
                expected_reservation = _remaining_pair_executions_for_operations(
                    [operation_by_id[item] for item in candidate_operations],
                    max_retries=int(current_budget["max_retries"]),
                )
                all_candidate_operations_settled = all(
                    operation_by_id[item].get("state")
                    in {"completed", "failed", "cancelled", "unavailable"}
                    for item in candidate_operations
                )
                if expected_reservation is None and not (
                    candidate_operations and all_candidate_operations_settled
                ):
                    raise ExperimentLoopError(
                        "cannot resume: reserved pair has no authorized remaining operation"
                    )
                if (
                    expected_reservation in RESERVABLE_PAIR_EXECUTIONS
                    and reservation.get("required_executions") != expected_reservation
                ):
                    if not candidate_operations:
                        raise ExperimentLoopError(
                            "cannot resume: new candidate reservation must reserve a full pair"
                        )
                    # A crash can persist a completed control operation while
                    # the original full-pair reservation is still present.
                    # Normalize that durable reservation to the exact
                    # remaining pair work before accounting or dispatch.
                    if not isinstance(reservation, dict) or not isinstance(candidate_state, dict):
                        raise ExperimentLoopError("cannot resume: reservation record is immutable")
                    reservation["required_executions"] = expected_reservation
                    candidate_state["reservation"] = expected_reservation
            elif "reservation" in candidate_state:
                raise ExperimentLoopError("cannot resume: inactive candidate retains reservation")
            if is_reserved != (reservation is not None and reservation.get("state") == "reserved"):
                raise ExperimentLoopError("cannot resume: reservation/candidate state mismatch")
            if candidate_state_name in TERMINAL_CANDIDATE_STATES and reservation is not None:
                if reservation.get("state") != "released":
                    raise ExperimentLoopError(
                        "cannot resume: terminal candidate retains reservation"
                    )
            if (
                candidate_outcome is None
                and not is_reserved
                and reservation is not None
                and candidate_state_name != INCOMPLETE_CANDIDATE_STATE
            ):
                raise ExperimentLoopError(
                    "cannot resume: pending candidate has released reservation"
                )
            if candidate_state_name == "dispatching" and not any(
                operation_by_id[item].get("state") == "dispatching" for item in candidate_operations
            ):
                raise ExperimentLoopError("cannot resume: dispatching candidate lacks dispatch")

        expected_reserved_executions = sum(
            int(reservation_by_candidate[candidate_id]["required_executions"])
            for candidate_id in active_reservations
        )
        if persisted_reserved_executions != persisted_reservation_total:
            raise ExperimentLoopError("cannot resume: reservation accounting is inconsistent")
        # ``reservation_by_candidate`` was normalized from the authoritative
        # operation history above.  Persist the reconciled aggregate so the
        # treatment-only continuation and its eventual release cannot retain
        # the stale full-pair count from the crash window.
        payload["reserved_executions"] = expected_reserved_executions
        if int(accounting["controls"]) != dispatched_by_kind["control"]:
            raise ExperimentLoopError("cannot resume: control accounting is inconsistent")
        if int(accounting["treatments"]) != dispatched_by_kind["treatment"]:
            raise ExperimentLoopError("cannot resume: treatment accounting is inconsistent")
        if int(accounting["failures"]) != failed_operations:
            raise ExperimentLoopError("cannot resume: failure accounting is inconsistent")
        if int(accounting["retries"]) != retry_operations:
            raise ExperimentLoopError("cannot resume: retry accounting is inconsistent")
        if int(accounting["fidelity_attempts"]) > 2 * int(accounting["controls"]):
            raise ExperimentLoopError("cannot resume: fidelity accounting is inconsistent")
        observed_fidelity_attempts = sum(
            1 + int(_native_fidelity_attempts(_status_from_result(operation.get("result"))[1]) or 0)
            for operation in operation_by_id.values()
            if operation.get("kind") == "control"
            and operation.get("state") == "completed"
            and _status_from_result(operation.get("result"))[0] == "ok"
        )

        terminal_candidate_ids = {
            candidate_id
            for candidate_id, candidate_state in payload["candidates"].items()
            if candidate_state.get("state") in TERMINAL_CANDIDATE_STATES
        }
        if journal_status != "running" and active_reservations:
            raise ExperimentLoopError("cannot resume: terminal journal retains reservation")
        if journal_status != "running" and any(
            operation.get("state") in {"reserved", "dispatching"}
            for operation in operation_by_id.values()
        ):
            raise ExperimentLoopError("cannot resume: terminal journal retains active operation")
        if (
            journal_status != "running"
            and int(accounting["fidelity_attempts"]) != observed_fidelity_attempts
        ):
            raise ExperimentLoopError("cannot resume: fidelity accounting does not match controls")
        for candidate_id, outcome in outcome_by_id.items():
            candidate_operation_states = {
                operation_by_id[item].get("state")
                for item in operation_ids_by_candidate[candidate_id]
            }
            if outcome.get("status") == "complete" and (
                not candidate_operation_states
                or not candidate_operation_states.issubset({"completed", "failed"})
            ):
                raise ExperimentLoopError("cannot resume: complete outcome has active operation")
        if journal_status == "complete":
            stop_reason = payload["stop_reason"]
            complete_outcomes = all(
                outcome.get("status") == "complete" for outcome in outcome_by_id.values()
            )
            if not complete_outcomes or not outcome_by_id:
                raise ExperimentLoopError(
                    "cannot resume: complete journal has non-complete outcomes"
                )
            if stop_reason == "exhausted_candidates":
                if terminal_candidate_ids != prior_candidate_ids:
                    raise ExperimentLoopError(
                        "cannot resume: exhausted journal has pending candidates"
                    )
            elif stop_reason == "recipe_terminal_condition":
                if not any(outcome.get("terminal") is True for outcome in outcome_by_id.values()):
                    raise ExperimentLoopError(
                        "cannot resume: terminal journal lacks terminal outcome"
                    )
            else:
                raise ExperimentLoopError(
                    "cannot resume: complete journal stop reason is inconsistent"
                )
        elif journal_status == "unavailable":
            if (
                not outcome_by_id
                or any(outcome.get("status") != "unavailable" for outcome in outcome_by_id.values())
                or terminal_candidate_ids != prior_candidate_ids
            ):
                raise ExperimentLoopError(
                    "cannot resume: unavailable journal outcomes are inconsistent"
                )
        elif journal_status == "failed":
            if not any(outcome.get("status") == "failed" for outcome in outcome_by_id.values()):
                raise ExperimentLoopError("cannot resume: failed journal lacks failed outcome")
            if any(outcome.get("status") == "complete" for outcome in outcome_by_id.values()):
                raise ExperimentLoopError("cannot resume: failed journal retains complete outcome")
            if payload["stop_reason"] not in {
                "candidate_execution_failed",
                "control_fidelity_failure_blocks_treatment",
            }:
                raise ExperimentLoopError(
                    "cannot resume: failed journal stop reason is inconsistent"
                )
        elif journal_status == "partial":
            stop_reason = payload["stop_reason"]
            if stop_reason not in {
                "execution_budget_exhausted",
                "wall_timeout",
                "candidate_execution_failed",
                "candidate_unavailable",
                "control_fidelity_failure_blocks_treatment",
                "recipe_terminal_condition",
            }:
                raise ExperimentLoopError(
                    "cannot resume: partial journal stop reason is inconsistent"
                )
            if (
                stop_reason == "execution_budget_exhausted"
                and terminal_candidate_ids == prior_candidate_ids
            ):
                raise ExperimentLoopError(
                    "cannot resume: exhausted pair budget has no pending candidate"
                )
            if stop_reason == "candidate_execution_failed" and not any(
                outcome.get("status") == "failed" for outcome in outcome_by_id.values()
            ):
                raise ExperimentLoopError("cannot resume: partial journal lacks failed outcome")
            if stop_reason == "candidate_unavailable" and not any(
                outcome.get("status") == "unavailable" for outcome in outcome_by_id.values()
            ):
                raise ExperimentLoopError(
                    "cannot resume: partial journal lacks unavailable outcome"
                )
            if stop_reason == "control_fidelity_failure_blocks_treatment" and not any(
                outcome.get("status") == "failed"
                and "control_fidelity_failure" in str(outcome.get("reason", ""))
                for outcome in outcome_by_id.values()
            ):
                raise ExperimentLoopError("cannot resume: partial journal lacks fidelity failure")
            if stop_reason == "recipe_terminal_condition" and not any(
                outcome.get("terminal") is True for outcome in outcome_by_id.values()
            ):
                raise ExperimentLoopError("cannot resume: partial journal lacks terminal outcome")
        elif journal_status == "cancelled":
            if any(
                outcome.get("status") not in TERMINAL_CANDIDATE_STATES
                for outcome in outcome_by_id.values()
            ):
                raise ExperimentLoopError("cannot resume: cancelled journal outcome is invalid")
            if "cancel" not in payload["stop_reason"].lower():
                raise ExperimentLoopError(
                    "cannot resume: cancelled journal stop reason is inconsistent"
                )
        # A terminal exhausted-candidates session may be continued when a
        # caller explicitly widens the candidate ceiling and the deterministic
        # catalog contributes new pending candidates.  Other terminal reasons
        # (cancellation, recipe terminal conditions, unsupported/admission
        # failures) remain immutable.  Likewise, a candidate-execution
        # failure is reopened only when the caller widens the retry ceiling and
        # the retained final operation is explicitly retryable.
        candidate_ceiling_widened = len(current_order) > len(prior_order)
        reopen_exhausted = (
            journal_status == "complete"
            and payload["stop_reason"] == "exhausted_candidates"
            and candidate_ceiling_widened
        )
        retryable_failed_candidates: set[str] = set()
        if (
            journal_status in {"failed", "partial"}
            and payload["stop_reason"] == "candidate_execution_failed"
            and int(current_budget["max_retries"]) > int(prior_budget["max_retries"])
        ):
            for candidate_id, candidate_state in payload["candidates"].items():
                if candidate_state.get("state") != "failed":
                    continue
                candidate_operations = operation_ids_by_candidate.get(candidate_id, [])
                if not candidate_operations:
                    continue
                # Only the final durable attempt can explain the retained
                # failed candidate outcome.  A historical retryable failure
                # followed by a permanent failure must remain terminal.
                final_operation = operation_by_id[candidate_operations[-1]]
                if (
                    _is_retryable_failed_operation(final_operation)
                    and int(final_operation.get("attempt", 0))
                    <= int(current_budget["max_retries"]) + 1
                ):
                    retryable_failed_candidates.add(candidate_id)

        # Only resumable sessions may admit a widened candidate prefix.  Keep
        # immutable terminal sessions closed without leaving a complete
        # journal containing unexecuted candidates.
        admit_widened_candidates = (
            journal_status == "running"
            or (journal_status == "partial" and payload["stop_reason"] in _RESUMABLE_STOP_REASONS)
            or reopen_exhausted
            or bool(retryable_failed_candidates)
        )
        effective_order = current_order if admit_widened_candidates else prior_order
        existing_candidate_ids = set(payload["candidates"])
        for candidate in effective_order:
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
        payload["candidate_order"] = effective_order
        payload["budget"] = self.budget.to_dict()
        payload["policy"] = self.policy.to_dict()
        payload["config_digest"] = expected["config_digest"]
        payload["config_identity_digest"] = expected["config_identity_digest"]
        if reopen_exhausted or retryable_failed_candidates:
            for candidate_id in retryable_failed_candidates:
                payload["candidates"][candidate_id]["state"] = "pending"
                payload["candidates"][candidate_id]["retry_reopened"] = True
                payload["candidates"][candidate_id].pop("reservation", None)
            if retryable_failed_candidates:
                payload["outcomes"] = [
                    outcome
                    for outcome in payload["outcomes"]
                    if outcome.get("intervention_id") not in retryable_failed_candidates
                ]
            payload["status"] = "running"
            payload["stop_reason"] = ""
        self._journal = payload
        elapsed = payload.get("elapsed_s", 0.0)
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(float(elapsed))
            or float(elapsed) < 0.0
        ):
            raise ExperimentLoopError("cannot resume: elapsed accounting is invalid")
        # Current-schema journals must carry the persisted monotonic floor.
        # Falling back to mutable ``elapsed_s`` would let an attacker lower
        # both values and reset the wall deadline on resume.
        if "elapsed_floor_s" not in payload:
            raise ExperimentLoopError("cannot resume: elapsed accounting floor is missing")
        elapsed_floor = payload["elapsed_floor_s"]
        if (
            isinstance(elapsed_floor, bool)
            or not isinstance(elapsed_floor, (int, float))
            or not math.isfinite(float(elapsed_floor))
            or float(elapsed_floor) < 0.0
            or float(elapsed) < float(elapsed_floor)
        ):
            raise ExperimentLoopError("cannot resume: elapsed accounting is not monotonic")
        self._journal["elapsed_floor_s"] = float(elapsed_floor)
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

    def _remaining_pair_executions(self, candidate_id: str) -> int | None:
        """Count the next authorized operations for an incomplete pair.

        Returns:
            The number of immediate control/treatment dispatches needed to
            make progress, or ``None`` when the retained pair is inconsistent
            with retry policy.
        """

        return _remaining_pair_executions_for_operations(
            [
                operation
                for operation in self._journal["operations"]
                if operation.get("candidate_id") == candidate_id
            ],
            max_retries=self.budget.max_retries,
        )

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
        required_executions = RESERVED_CONTROL_TREATMENT_PAIR
        if state.get("state") == INCOMPLETE_CANDIDATE_STATE or state.get("retry_reopened") is True:
            required_executions = self._remaining_pair_executions(candidate_id) or 0
            if required_executions not in RESERVABLE_PAIR_EXECUTIONS:
                return False
        remaining = self.budget.max_executions - int(self._journal["executions_consumed"])
        reserved = int(self._journal["reserved_executions"])
        if remaining - reserved < required_executions:
            return False
        if self._elapsed() >= self.budget.wall_timeout_s:
            return False
        state["state"] = "reserved"
        state["reservation"] = required_executions
        self._journal["reserved_executions"] += required_executions
        self._journal["reservations"].append(
            {
                "candidate_id": candidate_id,
                "required_executions": required_executions,
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

    def _pair_settled(self, candidate_id: str) -> bool:
        """Return whether a native adapter already settled this whole pair."""

        method = getattr(self.executor, "pair_settled", None)
        if not callable(method):
            return False
        try:
            return bool(method(candidate_id))
        except (KeyError, LookupError, TypeError, ValueError):
            return False

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
                    if self._cancelled():
                        return (
                            "cancelled",
                            {
                                "status": "cancelled",
                                "reason": "cancellation_requested",
                            },
                            "",
                        )
                    if self._elapsed() >= self.budget.wall_timeout_s:
                        return (
                            "failed",
                            {
                                "status": "failed",
                                "reason": "wall_timeout: retry dispatch prevented",
                            },
                            "",
                        )
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
                if state == "ok" and not _has_valid_result_metrics(result):
                    state = "failed"
                    result = {
                        **result,
                        "status": "failed",
                        "reason": "invalid_success_result: successful operation lacks finite metrics",
                    }
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
                    if self._cancelled():
                        return (
                            "cancelled",
                            {
                                "status": "cancelled",
                                "reason": "cancellation_requested",
                            },
                            "",
                        )
                    if self._elapsed() >= self.budget.wall_timeout_s:
                        return (
                            "failed",
                            {
                                "status": "failed",
                                "reason": "wall_timeout: retry dispatch prevented",
                            },
                            "",
                        )
                    continue
                return state, result, operation_id
            if operation is None:
                pair_settled = kind == "treatment" and self._pair_settled(candidate_id)
                if self._cancelled() and not pair_settled:
                    return (
                        "cancelled",
                        {
                            "status": "cancelled",
                            "reason": "cancellation_requested",
                        },
                        "",
                    )
                if self._elapsed() >= self.budget.wall_timeout_s and not pair_settled:
                    return (
                        "failed",
                        {
                            "status": "failed",
                            "reason": "wall_timeout: dispatch prevented",
                        },
                        "",
                    )
                if int(self._journal["executions_consumed"]) >= self.budget.max_executions:
                    return (
                        "budget_exhausted",
                        {
                            "status": "failed",
                            "reason": "execution_budget_exhausted: retry cannot be dispatched",
                        },
                        "",
                    )
                operation = {
                    "operation_id": operation_id,
                    "sequence": len(self._journal["operations"]),
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
            if state == "ok" and not _has_valid_result_metrics(result):
                state = "failed"
                result = {
                    **result,
                    "status": "failed",
                    "reason": "invalid_success_result: successful operation lacks finite metrics",
                }
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
        normalized_outcome = dict(outcome)
        status = normalized_outcome.get("status")
        # Keep the persisted convenience flag canonical.  The report exporter
        # derives this independently so a mutable flag cannot hide a negative
        # terminal result.
        if status in NEGATIVE_OUTCOME_STATUSES:
            normalized_outcome["negative"] = True
            for key in ("activation", "control_activated", "treatment_activated"):
                normalized_outcome.pop(key, None)
        elif status == "complete" and "outcome" in normalized_outcome:
            normalized_outcome["negative"] = normalized_outcome.get("outcome") != "survived"
        # Candidate operation IDs include every durable retry attempt.  The
        # outcome is the resume index as well as the user-facing summary, so
        # retain that complete history instead of only the final attempt IDs.
        normalized_outcome["operation_ids"] = list(state.get("operation_ids", []))
        state.update(
            {
                key: value
                for key, value in normalized_outcome.items()
                if key not in {"intervention_id"}
            }
        )
        self._journal["outcomes"].append(normalized_outcome)
        state["state"] = str(normalized_outcome.get("status", "failed"))
        self._release_reservation(candidate_id)
        self._persist()

    @staticmethod
    def _canonical_report_outcome(outcome: Mapping[str, Any]) -> dict[str, Any]:
        """Export an outcome without trusting mutable convenience fields.

        Returns:
            A report-safe copy with canonical negative classification and no
            activation claim on a non-complete outcome.
        """

        normalized = dict(outcome)
        status = normalized.get("status")
        if status in NEGATIVE_OUTCOME_STATUSES:
            normalized["negative"] = True
            for key in ("activation", "control_activated", "treatment_activated"):
                normalized.pop(key, None)
        elif status == "complete" and "outcome" in normalized:
            normalized["negative"] = normalized.get("outcome") != "survived"
        return normalized

    def _record_incomplete(
        self,
        candidate_id: str,
        candidate: Mapping[str, Any],
        *,
        reason: str,
        incomplete_reason: str,
        operation_ids: list[str],
        control: Mapping[str, Any] | None = None,
        treatment: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Persist a bounded incomplete candidate without inventing a failure.

        Returns:
            A transient inconclusive marker used to settle the session as
            execution-budget-limited without adding a terminal outcome.
        """

        if incomplete_reason not in {"execution_budget_exhausted", "wall_timeout"}:
            raise ExperimentLoopError("invalid incomplete candidate reason")
        state = self._candidate_state(candidate_id)
        state["state"] = INCOMPLETE_CANDIDATE_STATE
        state["incomplete_reason"] = incomplete_reason
        self._release_reservation(candidate_id)
        self._persist()
        outcome = {
            "intervention_id": candidate_id,
            "priority": int(candidate["priority"]),
            "factor": str(candidate.get("factor", "")),
            "status": INCOMPLETE_CANDIDATE_STATE,
            "outcome": "inconclusive",
            "reason": reason,
            "incomplete_reason": incomplete_reason,
            "operation_ids": list(state.get("operation_ids", operation_ids)),
        }
        if control is not None:
            outcome["control"] = dict(control)
        if treatment is not None:
            outcome["treatment"] = dict(treatment)
        return outcome

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
        operation_ids = [control_operation] if control_operation else []
        if control_state == "budget_exhausted":
            return self._record_incomplete(
                candidate_id,
                candidate,
                reason=_reason(control, "execution budget exhausted before control retry"),
                incomplete_reason="execution_budget_exhausted",
                operation_ids=operation_ids,
                control=control,
            )
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
        # Recovery may re-enter this candidate after the control operation has
        # already completed.  Derive the count from durable successful control
        # operations so a resumed pair cannot double-count the fidelity check.
        observed_fidelity_attempts = sum(
            1 + int(_native_fidelity_attempts(_status_from_result(operation.get("result"))[1]) or 0)
            for operation in self._journal["operations"]
            if operation.get("kind") == "control"
            and operation.get("state") == "completed"
            and _status_from_result(operation.get("result"))[0] == "ok"
        )
        self._journal["accounting"]["fidelity_attempts"] = observed_fidelity_attempts
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
        if self._elapsed() >= self.budget.wall_timeout_s and not self._pair_settled(candidate_id):
            return self._record_incomplete(
                candidate_id,
                candidate,
                reason="wall_timeout_before_treatment",
                incomplete_reason="wall_timeout",
                operation_ids=operation_ids,
                control=control,
            )
        treatment_already_started = any(
            operation.get("candidate_id") == candidate_id
            and operation.get("kind") == "treatment"
            and operation.get("state") != "reserved"
            for operation in self._journal["operations"]
        )
        if (
            self._cancelled()
            and not bool(control.get("native_pair_complete", False))
            and not treatment_already_started
        ):
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
        if treatment_operation:
            operation_ids.append(treatment_operation)
        if treatment_state == "budget_exhausted":
            return self._record_incomplete(
                candidate_id,
                candidate,
                reason=_reason(treatment, "execution budget exhausted before treatment"),
                incomplete_reason="execution_budget_exhausted",
                operation_ids=operation_ids,
                control=control,
                treatment=treatment,
            )
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
        pair_observation = _pair_telemetry(
            control,
            treatment,
            factor=factor,
            measurement=self.measurement,
            motion_epsilon=float(self.recipe.get("motion_epsilon_m", 0.05)),
        )
        if not pair_observation["activation_available"]:
            outcome = {
                "intervention_id": candidate_id,
                "priority": int(candidate["priority"]),
                "factor": factor,
                "status": "unavailable",
                "outcome": "inconclusive",
                "reason": pair_observation["reason"],
                "control": control,
                "treatment": treatment,
                "negative": True,
                "operation_ids": operation_ids,
            }
            self._record_outcome(candidate_id, outcome)
            return outcome
        if not pair_observation["measurement_available"]:
            outcome = {
                "intervention_id": candidate_id,
                "priority": int(candidate["priority"]),
                "factor": factor,
                "status": "complete",
                "outcome": "inconclusive",
                "reason": pair_observation["reason"],
                "control": control,
                "treatment": treatment,
                "negative": True,
                "operation_ids": operation_ids,
            }
            self._record_outcome(candidate_id, outcome)
            return outcome
        verdict = str(pair_observation["outcome"])
        verdict_reason = str(pair_observation["reason"])
        control_activated = bool(pair_observation["control_activated"])
        treatment_activated = bool(pair_observation["treatment_activated"])
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
        """Drive the loop while holding exclusive session ownership.

        Returns:
            The settled component result.
        """

        if self._session_lock_fd < 0:
            self._acquire_session_lock()
        try:
            return self._run_locked()
        finally:
            self._release_session_lock()

    def _run_locked(self) -> ComponentResult:
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
                candidate_state = self._candidate_state(candidate_id)
                if candidate_state.get("state") in TERMINAL_CANDIDATE_STATES:
                    continue
                if self._cancelled() and candidate_state.get("state") not in {
                    "reserved",
                    "dispatching",
                }:
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
                # A child can finish with a recipe-terminal marker at the
                # same instant cancellation or the wall budget becomes true.
                # Recheck the controller boundary after the result and make it
                # authoritative over that marker.
                if self._cancelled():
                    self._journal["status"] = "cancelled"
                    self._journal["stop_reason"] = "cancellation_requested"
                    break
                if self._elapsed() >= self.budget.wall_timeout_s:
                    self._journal["status"] = "partial"
                    self._journal["stop_reason"] = "wall_timeout"
                    break
                if outcome.get("status") == INCOMPLETE_CANDIDATE_STATE:
                    self._journal["status"] = "partial"
                    self._journal["stop_reason"] = str(
                        outcome.get("incomplete_reason", "execution_budget_exhausted")
                    )
                    break
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
        outcomes = [
            self._canonical_report_outcome(item) for item in self._journal.get("outcomes", [])
        ]
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
            "negative_outcomes": [item for item in outcomes if _is_negative_outcome(item)],
            "budget": {
                **self.budget.to_dict(),
                "executions_consumed": self._journal.get("executions_consumed", 0),
                "reserved_executions": self._journal.get("reserved_executions", 0),
                "elapsed_s": self._journal.get("elapsed_s", 0.0),
            },
            "accounting": dict(self._journal.get("accounting", {})),
            "operations": list(self._journal.get("operations", [])),
            "answerability": self._answerability_document,
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
    attempt ledger.  The adapter resumes that ledger once per outer candidate
    prefix.  This keeps a child invocation to one newly eligible pair and
    prevents a child from dispatching later candidates after cancellation or a
    wall/budget stop in the outer controller.
    """

    def __init__(
        self,
        request: ComponentRequest,
        *,
        base: Path,
        executor_config: Mapping[str, Any],
        recipe: Mapping[str, Any],
        admission_config: Mapping[str, Any] | review_execute.ExecutorAdmissionConfig,
        source_admission: Mapping[str, Any] | None = None,
        resume: bool,
        cancel: Callable[[], bool] | Any | None = None,
    ) -> None:
        self.request = request
        self.base = base
        self.executor_config = dict(executor_config)
        self.recipe = dict(recipe)
        self.admission_config = admission_config
        self.source_admission = (
            dict(source_admission) if isinstance(source_admission, Mapping) else None
        )
        self.resume = resume
        self.cancel = cancel
        self.child_result: ComponentResult | None = None
        self.reports: dict[str, dict[str, Any]] = {}
        self._nested_attempts: dict[tuple[str, str], dict[str, Any]] = {}
        self._nested_conflict: str | None = None
        self._child_started = False
        self._operation_context: dict[str, tuple[str, str]] = {}
        self._active_dispatch: tuple[str, str] | None = None

    def bind_operation_map(self, operations: Any) -> None:
        """Bind exact journal operation identities for crash recovery.

        Operation IDs are opaque.  In particular, candidate IDs may contain
        ``:retry:`` and must never be recovered by delimiter stripping.
        """

        if not isinstance(operations, list):
            return
        for operation in operations:
            if not isinstance(operation, Mapping):
                continue
            operation_id = operation.get("operation_id")
            candidate_id = operation.get("candidate_id")
            kind = operation.get("kind")
            if (
                isinstance(operation_id, str)
                and isinstance(candidate_id, str)
                and kind in {"control", "treatment"}
            ):
                self._operation_context[operation_id] = (candidate_id, str(kind))
                if operation.get("state") == "dispatching":
                    self._active_dispatch = (candidate_id, str(kind))

    def _cancelled(self) -> bool:
        if self.cancel is None:
            return False
        if callable(self.cancel):
            try:
                return bool(self.cancel())
            except (TypeError, RuntimeError):
                return False
        is_set = getattr(self.cancel, "is_set", None)
        return bool(is_set()) if callable(is_set) else bool(self.cancel)

    def _child_blocks_followup(self) -> bool:
        """Return whether the child boundary forbids a later candidate call."""

        if self.child_result is None:
            return False
        if self.child_result.status in {"failed", "cancelled"}:
            return True
        if self.child_result.status != "unavailable":
            return False
        reason = self.child_result.reason.lower()
        admission_markers = (
            "source_admission",
            "receipt",
            "preservation",
            "allowed_root",
            "source_missing",
            "source_escaped_root",
        )
        return not self.reports or any(marker in reason for marker in admission_markers)

    def _child_request(self, candidate_id: str) -> ComponentRequest:
        """Build the request envelope used for a native dispatch.

        Returns:
            The bounded child request for ``candidate_id``.
        """

        ordered_candidates = _candidate_order(self.recipe)
        try:
            candidate_index = next(
                index
                for index, item in enumerate(ordered_candidates, start=1)
                if str(item["intervention_id"]) == candidate_id
            )
        except StopIteration as error:
            raise ExperimentLoopError(
                f"native executor candidate is unknown: {candidate_id}"
            ) from error
        config = dict(self.executor_config)
        config["recipe"] = self.recipe
        config["max_candidates"] = min(candidate_index, DEFAULT_MAX_CANDIDATES)
        config["max_executions"] = min(
            int(config.get("max_executions", DEFAULT_MAX_EXECUTIONS)), DEFAULT_MAX_EXECUTIONS
        )
        return ComponentRequest(
            request_id=self.request.request_id,
            component_id=review_execute.COMPONENT_ID,
            sources=self.request.sources,
            output_directory="executor",
            config=config,
            required_capabilities=self.request.required_capabilities,
        )

    def _child_request_and_config(
        self, candidate_id: str
    ) -> tuple[ComponentRequest, review_execute.ExecuteConfig, dict[str, Any]]:
        """Build and validate the exact child envelope used for recovery.

        Returns:
            Child request, validated effective config, and its admission proof.

        Raises:
            ExperimentLoopError: If the native child envelope cannot be
                validated before recovery.
        """

        child_request = self._child_request(candidate_id)
        config = dict(child_request.config)
        if isinstance(self.admission_config, review_execute.ExecutorAdmissionConfig):
            admission = review_execute.validate_executor_admission_config(
                self.admission_config.to_dict()
            )
        else:
            admission = review_execute.validate_executor_admission_config(self.admission_config)
        effective_payload = {**config, "admission": admission.to_dict()}
        try:
            validated = review_execute.validate_execute_config(effective_payload)
        except review_execute.ReviewExecuteError as error:
            raise ExperimentLoopError(
                f"native child config is invalid: {'; '.join(error.errors)}"
            ) from error
        return child_request, validated, admission.to_dict()

    @staticmethod
    def _native_metrics_valid(value: Any) -> bool:
        return (
            isinstance(value, Mapping)
            and set(value) == review_execute.REQUIRED_TELEMETRY_METRICS
            and all(
                isinstance(item, (int, float))
                and not isinstance(item, bool)
                and math.isfinite(float(item))
                for item in value.values()
            )
        )

    def _validate_nested_recovery(
        self,
        documents: Mapping[str, Mapping[str, Any]],
        report_documents: Mapping[str, Mapping[str, Mapping[str, Any]]],
        nested_attempts: Mapping[tuple[str, str], Mapping[str, Any]],
    ) -> str | None:
        """Validate child artifacts before using them for native recovery.

        The child executor remains the final ledger authority.  This local
        envelope check prevents a self-consistent but stale or unbound file
        from being used before that child call, and gives the recovery path a
        current candidate/operation binding to enforce.

        Returns:
            A fail-closed reason, or ``None`` when the envelope is supported.
        """

        if not self.resume or self.child_result is not None or not documents:
            return None
        if self.source_admission is None:
            return "nested child recovery requires an admitted source proof"
        ledger = documents.get("attempt-ledger")
        if ledger is None:
            return "nested child attempt ledger is required for recovery"
        if set(ledger) != _NATIVE_LEDGER_KEYS:
            return "nested child attempt ledger schema is unsupported"
        try:
            child_request, child_config, _admission = self._child_request_and_config(
                self._active_dispatch[0] if self._active_dispatch is not None else ""
            )
        except (
            ExperimentLoopError,
            review_execute.ReviewExecuteError,
            TypeError,
            ValueError,
        ) as error:
            return f"nested child recovery binding is invalid: {error}"
        expected_request_digest = review_execute._canonical_digest(
            review_execute._request_identity_document(child_request)
        )
        expected_recipe_digest = review_execute._canonical_digest(self.recipe)
        expected_config_identity_digest = review_execute._canonical_digest(
            review_execute._config_identity_document(child_config)
        )
        if (
            ledger.get("schema_version") != review_execute.ATTEMPT_LEDGER_SCHEMA_VERSION
            or ledger.get("request_id") != child_request.request_id
            or ledger.get("component_id") != review_execute.COMPONENT_ID
            or ledger.get("recipe_id") != str(self.recipe.get("recipe_id", ""))
            or ledger.get("request_digest") != expected_request_digest
            or ledger.get("recipe_digest") != expected_recipe_digest
            or ledger.get("config_identity_digest") != expected_config_identity_digest
            or ledger.get("source_admission") != self.source_admission
            or ledger.get("evidence_boundary") != DIAGNOSTIC_EVIDENCE_BOUNDARY
            or ledger.get("scientific_claim_allowed") is not False
            or ledger.get("dependent_family_status") != DEPENDENT_FAMILY_STATUS
        ):
            return "nested child recovery identity binding mismatch"
        expected_budget = review_execute._config_budget_document(child_config)
        prior_budget = ledger.get("budget")
        if not isinstance(prior_budget, Mapping) or set(prior_budget) != set(expected_budget):
            return "nested child recovery budget binding is malformed"
        if (
            not isinstance(prior_budget.get("max_candidates"), int)
            or isinstance(prior_budget.get("max_candidates"), bool)
            or not isinstance(prior_budget.get("max_executions"), int)
            or isinstance(prior_budget.get("max_executions"), bool)
            or not isinstance(prior_budget.get("wall_timeout_s"), (int, float))
            or isinstance(prior_budget.get("wall_timeout_s"), bool)
            or not math.isfinite(float(prior_budget.get("wall_timeout_s")))
            or int(prior_budget["max_candidates"]) < 1
            or int(prior_budget["max_executions"]) < 1
            or float(prior_budget["wall_timeout_s"]) <= 0.0
            or int(prior_budget["max_candidates"]) > int(expected_budget["max_candidates"])
            or int(prior_budget["max_executions"]) > int(expected_budget["max_executions"])
            or float(prior_budget["wall_timeout_s"]) > float(expected_budget["wall_timeout_s"])
        ):
            return "nested child recovery budget binding is invalid"
        prior_config = replace(
            child_config,
            max_candidates=int(prior_budget["max_candidates"]),
            max_executions=int(prior_budget["max_executions"]),
            wall_timeout_s=float(prior_budget["wall_timeout_s"]),
        )
        config_digests = {
            review_execute._canonical_digest(review_execute._config_document(child_config)),
            review_execute._canonical_digest(review_execute._config_document(prior_config)),
        }
        attempts = ledger.get("attempts")
        if not isinstance(attempts, list):
            return "nested child attempt ledger attempts are malformed"
        validated_attempt_keys: set[tuple[str, str]] = set()
        attempt_candidate_ids: set[str] = set()
        recipe_candidate_ids = {
            str(item["intervention_id"]) for item in _candidate_order(self.recipe)
        }
        for index, attempt in enumerate(attempts):
            if not isinstance(attempt, Mapping):
                return f"nested child attempt {index} is malformed"
            candidate = attempt.get("candidate_id")
            kind = attempt.get("kind")
            status = attempt.get("status")
            key = (candidate, kind)
            if (
                not isinstance(candidate, str)
                or candidate not in recipe_candidate_ids
                or kind not in {"control", "treatment"}
                or status not in {"ok", "failed", "timed_out", "cancelled"}
                or key in validated_attempt_keys
                or set(attempt)
                != (
                    {"candidate_id", "kind", "status", "elapsed_s", "metrics"}
                    if status == "ok"
                    else {"candidate_id", "kind", "status", "reason", "elapsed_s"}
                )
            ):
                return f"nested child attempt {index} is unsupported"
            elapsed = attempt.get("elapsed_s")
            if (
                not isinstance(elapsed, (int, float))
                or isinstance(elapsed, bool)
                or not math.isfinite(float(elapsed))
                or float(elapsed) < 0.0
            ):
                return f"nested child attempt {index} timing is invalid"
            if status == "ok" and not self._native_metrics_valid(attempt.get("metrics")):
                return f"nested child attempt {index} metrics are invalid"
            validated_attempt_keys.add(key)
            attempt_candidate_ids.add(candidate)
        if any(
            kind == "treatment" and (candidate, "control") not in validated_attempt_keys
            for candidate, kind in validated_attempt_keys
        ):
            return "nested child treatment attempt lacks its control"
        consumed = ledger.get("executions_consumed")
        if not isinstance(consumed, int) or isinstance(consumed, bool) or consumed != len(attempts):
            return "nested child execution accounting is inconsistent"
        wall_elapsed = ledger.get("wall_elapsed_s")
        if (
            not isinstance(wall_elapsed, (int, float))
            or isinstance(wall_elapsed, bool)
            or not math.isfinite(float(wall_elapsed))
            or float(wall_elapsed) < 0.0
        ):
            return "nested child ledger wall timing is invalid"
        ledger_reports = report_documents.get("attempt-ledger", {})
        candidate_by_id = {
            str(item["intervention_id"]): item for item in _candidate_order(self.recipe)
        }
        for candidate_id, report in ledger_reports.items():
            if candidate_id not in candidate_by_id or not isinstance(report, Mapping):
                return "nested child candidate report is not bound to the recipe"
            factor = candidate_by_id[candidate_id].get("factor", "")
            status = report.get("status")
            allowed_keys = {
                "complete": _NATIVE_COMPLETE_REPORT_KEYS,
                "failed": _NATIVE_FAILED_REPORT_KEYS,
                "unavailable": _NATIVE_UNAVAILABLE_REPORT_KEYS,
            }.get(status)
            if allowed_keys is None:
                return f"nested child candidate report {candidate_id} is unsupported"
            if status == "failed":
                if set(report) not in {
                    _NATIVE_FAILED_REPORT_KEYS,
                    _NATIVE_FAILED_REPORT_WITH_CONTROL_KEYS,
                }:
                    return f"nested child candidate report {candidate_id} is unsupported"
            elif set(report) != allowed_keys:
                return f"nested child candidate report {candidate_id} is unsupported"
            if report.get("factor") != factor:
                return f"nested child candidate report {candidate_id} factor is not bound"
            if not isinstance(report.get("reason"), str) and status in {"failed", "unavailable"}:
                return f"nested child candidate report {candidate_id} reason is malformed"
            matching = {
                kind: attempt
                for (attempt_candidate, kind), attempt in nested_attempts.items()
                if attempt_candidate == candidate_id
            }
            if status == "complete":
                if set(matching) != {"control", "treatment"} or any(
                    item.get("status") != "ok" for item in matching.values()
                ):
                    return f"nested child complete report {candidate_id} is not paired"
                if report.get("nonintervened_config_match") is not True or any(
                    not self._native_metrics_valid(report.get(key))
                    for key in ("control_metrics", "treatment_metrics")
                ):
                    return f"nested child complete report {candidate_id} is malformed"
                if (
                    not isinstance(report.get("verdict"), str)
                    or not isinstance(report.get("verdict_reason"), str)
                    or not isinstance(report.get("control_activated"), bool)
                    or not isinstance(report.get("treatment_activated"), bool)
                ):
                    return f"nested child complete report {candidate_id} is malformed"
                if any(
                    review_execute._canonical_digest(report[f"{kind}_metrics"])
                    != review_execute._canonical_digest(matching[kind].get("metrics"))
                    for kind in ("control", "treatment")
                ):
                    return f"nested child report {candidate_id} metrics are not ledger-bound"
            elif status == "failed":
                if report.get("nonintervened_config_match") is not True:
                    return f"nested child failed report {candidate_id} is malformed"
                for kind in ("control", "treatment"):
                    metrics = report.get(f"{kind}_metrics")
                    if metrics is not None and not self._native_metrics_valid(metrics):
                        return f"nested child failed report {candidate_id} metrics are invalid"
                if not matching:
                    return f"nested child failed report {candidate_id} lacks attempts"
                if any(
                    attempt.get("status") == "ok"
                    and report.get(f"{kind}_metrics") is not None
                    and review_execute._canonical_digest(report[f"{kind}_metrics"])
                    != review_execute._canonical_digest(attempt.get("metrics"))
                    for kind, attempt in matching.items()
                ):
                    return f"nested child failed report {candidate_id} metrics are not ledger-bound"
            elif matching:
                return f"nested child unavailable report {candidate_id} disagrees with attempts"
        traces = ledger.get("traces")
        if not isinstance(traces, list) or not all(isinstance(item, Mapping) for item in traces):
            return "nested child activation traces are malformed"
        trace_keys = {
            "schema_version",
            "intervention_id",
            "factor",
            "control_activated",
            "treatment_activated",
            "control_metrics",
            "treatment_metrics",
        }
        if any(
            set(trace) != trace_keys
            or trace.get("schema_version") != review_execute.ACTIVATION_TRACE_SCHEMA_VERSION
            for trace in traces
        ):
            return "nested child activation trace schema is unsupported"
        complete_ids = {
            candidate_id
            for candidate_id, report in ledger_reports.items()
            if report.get("status") == "complete"
        }
        trace_ids = {str(trace.get("intervention_id")) for trace in traces}
        if trace_ids != complete_ids or len(trace_ids) != len(traces):
            return "nested child activation traces do not match reports"
        if any(
            trace.get("factor")
            != candidate_by_id.get(str(trace.get("intervention_id")), {}).get("factor", "")
            for trace in traces
        ):
            return "nested child activation trace is not recipe-bound"
        try:
            review_execute._verify_resume_envelope(
                attempts=[dict(item) for item in attempts],
                reports=[dict(item) for item in ledger_reports.values()],
                traces=[dict(item) for item in traces],
                config=child_config,
                recipe=self.recipe,
            )
        except (review_execute.ReviewExecuteError, TypeError, ValueError, KeyError) as error:
            return f"nested child recovery result validation failed: {error}"
        if set(ledger_reports) != set(report_documents.get("execute-report", ledger_reports)):
            return "nested child report documents disagree"
        active = self._active_dispatch
        if active is not None:
            active_candidate, active_kind = active
            ordered_ids = [str(item["intervention_id"]) for item in _candidate_order(self.recipe)]
            try:
                active_index = ordered_ids.index(active_candidate)
            except ValueError:
                return "nested child recovery candidate is not recipe-bound"
            reported_ids = set(ledger_reports) | attempt_candidate_ids
            expected_prefix = set(ordered_ids[:active_index])
            if not reported_ids.issubset(expected_prefix | {active_candidate}):
                return "nested child recovery candidate is outside current prefix"
            if active_candidate not in reported_ids and reported_ids != expected_prefix:
                return "nested child recovery current dispatch is not bound"
            if (
                active_candidate in reported_ids
                and (active_candidate, active_kind) not in validated_attempt_keys
            ):
                return "nested child recovery operation is not bound to current dispatch"
        execute_payload = documents.get("execute-report")
        if execute_payload is not None:
            if set(execute_payload) != _NATIVE_EXECUTE_REPORT_KEYS:
                return "nested child execute report schema is unsupported"
            if (
                execute_payload.get("schema_version")
                != review_execute.EXECUTE_REPORT_SCHEMA_VERSION
                or execute_payload.get("request_id") != child_request.request_id
                or execute_payload.get("component_id") != review_execute.COMPONENT_ID
                or execute_payload.get("recipe_id") != str(self.recipe.get("recipe_id", ""))
                or execute_payload.get("source_identity") != self.recipe.get("source_identity")
                or execute_payload.get("source_admission") != self.source_admission
                or execute_payload.get("evidence_boundary") != DIAGNOSTIC_EVIDENCE_BOUNDARY
                or execute_payload.get("benchmark_success") is not False
                or execute_payload.get("scientific_claim_allowed") is not False
                or execute_payload.get("dependent_family_status") != DEPENDENT_FAMILY_STATUS
            ):
                return "nested child execute report identity binding mismatch"
            budget = execute_payload.get("budget")
            if not isinstance(budget, Mapping) or set(budget) != {
                "max_candidates",
                "max_executions",
                "executions_consumed",
                "wall_timeout_s",
                "wall_elapsed_s",
            }:
                return "nested child execute report budget is malformed"
            if (
                budget.get("executions_consumed") != consumed
                or not isinstance(budget.get("max_candidates"), int)
                or isinstance(budget.get("max_candidates"), bool)
                or not isinstance(budget.get("max_executions"), int)
                or isinstance(budget.get("max_executions"), bool)
                or not isinstance(budget.get("wall_timeout_s"), (int, float))
                or isinstance(budget.get("wall_timeout_s"), bool)
                or not isinstance(budget.get("wall_elapsed_s"), (int, float))
                or isinstance(budget.get("wall_elapsed_s"), bool)
                or not math.isfinite(float(budget.get("wall_timeout_s")))
                or not math.isfinite(float(budget.get("wall_elapsed_s")))
                or float(budget.get("wall_timeout_s")) <= 0.0
                or float(budget.get("wall_elapsed_s")) < 0.0
                or any(
                    budget.get(key) != prior_budget.get(key)
                    for key in (
                        "max_candidates",
                        "max_executions",
                        "wall_timeout_s",
                    )
                )
            ):
                return "nested child execute report accounting disagrees with ledger"
            provenance = execute_payload.get("provenance")
            if not isinstance(provenance, Mapping) or set(provenance) != _NATIVE_PROVENANCE_KEYS:
                return "nested child execute report provenance is missing"
            expected_sources = [
                {
                    "artifact_id": source.artifact_id,
                    "uri": source.uri,
                    "format": source.format,
                }
                for source in child_request.sources
            ]
            if (
                provenance.get("recipe_id") != str(self.recipe.get("recipe_id", ""))
                or provenance.get("request_digest") != expected_request_digest
                or provenance.get("recipe_digest") != expected_recipe_digest
                or provenance.get("source_identity") != self.recipe.get("source_identity")
                or provenance.get("source_admission") != self.source_admission
                or provenance.get("component_id") != review_execute.COMPONENT_ID
                or provenance.get("evidence_boundary") != DIAGNOSTIC_EVIDENCE_BOUNDARY
                or provenance.get("benchmark_success") is not False
                or provenance.get("scientific_claim_allowed") is not False
                or provenance.get("dependent_family_status") != DEPENDENT_FAMILY_STATUS
                or provenance.get("sources") != expected_sources
                or provenance.get("output_directory") != child_request.output_directory
            ):
                return "nested child execute report provenance binding mismatch"
            if provenance.get("config_digest") not in config_digests:
                return "nested child execute report config binding mismatch"
        return None

    def _read_nested_state(self) -> None:
        """Refresh child reports and attempts without dispatching any work."""

        if self._nested_conflict is not None:
            return
        report_path = self.base / "executor" / "execute-report.json"
        ledger_path = self.base / "executor" / "attempt-ledger.json"
        documents: dict[str, Mapping[str, Any]] = {}
        for label, path in (("execute-report", report_path), ("attempt-ledger", ledger_path)):
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError, RecursionError) as error:
                self._nested_conflict = f"nested child {label} is unreadable: {error}"
                self.reports = {}
                self._nested_attempts = {}
                return
            if not isinstance(payload, Mapping):
                self._nested_conflict = f"nested child {label} is not a mapping"
                self.reports = {}
                self._nested_attempts = {}
                return
            documents[label] = payload

        def report_map(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]] | None:
            raw_reports = payload.get("candidates", payload.get("candidate_reports", []))
            if not isinstance(raw_reports, list):
                return None
            parsed: dict[str, dict[str, Any]] = {}
            for item in raw_reports:
                if not isinstance(item, Mapping) or not isinstance(
                    item.get("intervention_id"), str
                ):
                    return None
                candidate_id = str(item["intervention_id"])
                if candidate_id in parsed:
                    return None
                parsed[candidate_id] = dict(item)
            return parsed

        report_documents: dict[str, dict[str, dict[str, Any]]] = {}
        for label, payload in documents.items():
            parsed = report_map(payload)
            if parsed is None:
                self._nested_conflict = f"nested child {label} reports are malformed"
                self.reports = {}
                self._nested_attempts = {}
                return
            report_documents[label] = parsed

        nested_attempts: dict[tuple[str, str], dict[str, Any]] = {}
        ledger_payload = documents.get("attempt-ledger")
        if ledger_payload is not None:
            raw_attempts = ledger_payload.get("attempts", [])
            if not isinstance(raw_attempts, list):
                self._nested_conflict = "nested child attempt ledger is malformed"
                self.reports = {}
                self._nested_attempts = {}
                return
            if isinstance(raw_attempts, list):
                for item in raw_attempts:
                    if (
                        isinstance(item, Mapping)
                        and isinstance(item.get("candidate_id"), str)
                        and item.get("kind") in {"control", "treatment"}
                    ):
                        key = (str(item["candidate_id"]), str(item["kind"]))
                        if key in nested_attempts:
                            self._nested_conflict = "nested child attempt ledger has duplicate pair"
                            self.reports = {}
                            self._nested_attempts = {}
                            return
                        nested_attempts[key] = dict(item)
                    else:
                        self._nested_conflict = "nested child attempt ledger has malformed attempt"
                        self.reports = {}
                        self._nested_attempts = {}
                        return

        report_documents_by_kind = list(report_documents.items())
        if len(report_documents_by_kind) == 2:
            first_label, first_reports = report_documents_by_kind[0]
            second_label, second_reports = report_documents_by_kind[1]
            try:
                reports_disagree = set(first_reports) != set(second_reports) or any(
                    _canonical_digest(first_reports[candidate_id])
                    != _canonical_digest(second_reports[candidate_id])
                    for candidate_id in set(first_reports) & set(second_reports)
                )
            except (ExperimentLoopError, RecursionError, TypeError, ValueError):
                reports_disagree = True
            if reports_disagree:
                self._nested_conflict = f"nested child {first_label} disagrees with {second_label}"
                self.reports = {}
                self._nested_attempts = {}
                return
        if ledger_payload is not None and nested_attempts:
            ledger_reports = report_documents.get("attempt-ledger", {})
            attempt_candidate_ids = {candidate_id for candidate_id, _kind in nested_attempts}
            if "execute-report" in report_documents and not attempt_candidate_ids.issubset(
                set(ledger_reports)
            ):
                self._nested_conflict = "nested child attempt ledger disagrees with reports"
                self.reports = {}
                self._nested_attempts = {}
                return
            for candidate_id, report in ledger_reports.items():
                matching = [
                    attempt
                    for (attempt_candidate_id, _kind), attempt in nested_attempts.items()
                    if attempt_candidate_id == candidate_id
                ]
                if not matching:
                    continue
                statuses = {str(attempt.get("status")) for attempt in matching}
                report_status = report.get("status")
                if report_status == "complete" and (
                    {(str(attempt.get("kind"))) for attempt in matching} != {"control", "treatment"}
                    or statuses != {"ok"}
                ):
                    self._nested_conflict = "nested child complete report disagrees with attempts"
                    self.reports = {}
                    self._nested_attempts = {}
                    return
                if report_status == "unavailable" and matching:
                    self._nested_conflict = (
                        "nested child unavailable report disagrees with attempts"
                    )
                    self.reports = {}
                    self._nested_attempts = {}
                    return
                if report_status in {"complete", "failed"}:
                    for attempt in matching:
                        if attempt.get("status") != "ok":
                            continue
                        report_metrics = report.get(f"{attempt.get('kind')}_metrics")
                        attempt_metrics = attempt.get("metrics")
                        try:
                            metrics_disagree = (
                                not isinstance(report_metrics, Mapping)
                                or not isinstance(attempt_metrics, Mapping)
                                or (
                                    _canonical_digest(report_metrics)
                                    != _canonical_digest(attempt_metrics)
                                )
                            )
                        except (ExperimentLoopError, RecursionError, TypeError, ValueError):
                            metrics_disagree = True
                        if metrics_disagree:
                            self._nested_conflict = (
                                "nested child report metrics disagree with attempts"
                            )
                            self.reports = {}
                            self._nested_attempts = {}
                            return
                if report_status == "failed" and statuses == {"ok"}:
                    reason = str(report.get("reason", "")).lower()
                    if "control_fidelity_failure" in reason:
                        expected_kinds = {"control"}
                    elif "treatment" in reason:
                        expected_kinds = {"control", "treatment"}
                    else:
                        expected_kinds = set()
                    if expected_kinds != {
                        str(attempt.get("kind")) for attempt in matching
                    } or expected_kinds == {"control", "treatment"}:
                        self._nested_conflict = "nested child failed report disagrees with attempts"
                        self.reports = {}
                        self._nested_attempts = {}
                        return

        recovery_error = self._validate_nested_recovery(
            documents,
            report_documents,
            nested_attempts,
        )
        if recovery_error is not None:
            self._nested_conflict = recovery_error
            self.reports = {}
            self._nested_attempts = {}
            return

        prior_reports = dict(self.reports)
        if "attempt-ledger" in report_documents:
            self.reports = dict(report_documents["attempt-ledger"])
        elif "execute-report" in report_documents:
            self.reports = dict(report_documents["execute-report"])
        else:
            self.reports = prior_reports
        self._nested_attempts = nested_attempts

        # A crash can land after the child has persisted both attempts but
        # before its final candidate report.  Reconstruct only that already
        # settled pair locally; no child invocation is needed (or permitted)
        # while cancellation is active.
        for candidate in _candidate_order(self.recipe):
            candidate_id = str(candidate["intervention_id"])
            if candidate_id in self.reports:
                continue
            control = nested_attempts.get((candidate_id, "control"))
            treatment = nested_attempts.get((candidate_id, "treatment"))
            if control is None:
                continue
            factor = str(candidate.get("factor", ""))
            control_status = control.get("status")
            if control_status != "ok":
                self.reports[candidate_id] = {
                    "intervention_id": candidate_id,
                    "factor": factor,
                    "status": "failed",
                    "reason": f"control execution failed: {control.get('reason', '')}",
                }
                continue
            control_metrics = control.get("metrics")
            if not isinstance(control_metrics, Mapping):
                continue
            if treatment is None:
                continue
            treatment_status = treatment.get("status")
            if treatment_status == "ok" and isinstance(treatment.get("metrics"), Mapping):
                # The child attempt ledger carries metrics but not the
                # explicit activation contract.  Do not infer a measured
                # activation bit from those metrics; a missing child report
                # must be re-opened through the child executor instead.
                continue
            elif treatment_status in {"failed", "timed_out", "cancelled"}:
                self.reports[candidate_id] = {
                    "intervention_id": candidate_id,
                    "factor": factor,
                    "status": "failed",
                    "reason": f"treatment execution failed: {treatment.get('reason', '')}",
                    "control_metrics": dict(control_metrics),
                }

    def _nested_attempt_result(self, candidate_id: str, kind: str) -> Mapping[str, Any] | None:
        """Map one already persisted child attempt to an outer operation result.

        Returns:
            A normalized operation result, or ``None`` when no child attempt
            was persisted for the requested pair side.
        """

        attempt = self._nested_attempts.get((candidate_id, kind))
        if attempt is None:
            return None
        status = attempt.get("status")
        if status == "ok" and isinstance(attempt.get("metrics"), Mapping):
            result: dict[str, Any] = {
                "status": "ok",
                "metrics": dict(attempt["metrics"]),
            }
            if self._nested_pair_attempts_complete(candidate_id):
                result["native_pair_complete"] = True
            if kind == "control":
                # The child records one fidelity check for every successful
                # control attempt, even when a crash occurred before its
                # candidate report was written.
                result["native_fidelity_attempts"] = 1
            return result
        if status == "cancelled":
            return {"status": "cancelled", "reason": attempt.get("reason", "cancelled")}
        return {"status": "failed", "reason": attempt.get("reason", "child attempt failed")}

    def _nested_pair_attempts_complete(self, candidate_id: str) -> bool:
        """Return whether the child ledger durably settled both pair sides."""

        return all(
            attempt is not None
            and attempt.get("status") == "ok"
            and isinstance(attempt.get("metrics"), Mapping)
            for kind in ("control", "treatment")
            for attempt in (self._nested_attempts.get((candidate_id, kind)),)
        )

    @staticmethod
    def _is_control_fidelity_failure(report: Mapping[str, Any]) -> bool:
        return (
            report.get("status") == "failed"
            and "control_fidelity_failure" in str(report.get("reason", ""))
            and isinstance(report.get("control_metrics"), Mapping)
        )

    @staticmethod
    def _is_treatment_failure(report: Mapping[str, Any]) -> bool:
        reason = str(report.get("reason", "")).lower()
        return (
            report.get("status") == "failed"
            and isinstance(report.get("control_metrics"), Mapping)
            and "treatment" in reason
            and "control_fidelity" not in reason
        )

    def _child_failure_allows_pair_report(self, report: Mapping[str, Any]) -> bool:
        """Allow a candidate-side child report to explain a failed child call.

        A prior report may still be present when the child fails its admission
        boundary.  Only the child executor's explicit candidate-execution
        failure envelope may therefore rehydrate a control/treatment pair;
        admission failures remain authoritative over every retained report.

        Returns:
            ``True`` only when the failed child result is a candidate-side
            execution envelope and the retained report is also failed.
        """

        if self.child_result is None or self.child_result.status != "failed":
            return False
        child_reason = self.child_result.reason.lower()
        admission_markers = (
            "source_admission",
            "receipt",
            "preservation",
            "allowed_root",
            "source_missing",
            "source_escaped_root",
            "invalid_output_path",
        )
        if any(marker in child_reason for marker in admission_markers):
            return False
        return "candidate_execution_failed" in child_reason and report.get("status") == "failed"

    def _native_control_result(
        self, report: Mapping[str, Any], *, fidelity: bool
    ) -> Mapping[str, Any]:
        metrics = report.get("control_metrics")
        if not isinstance(metrics, Mapping):
            return {"status": "failed", "reason": "native report lacks control_metrics"}
        result: dict[str, Any] = {
            "status": "ok",
            "metrics": dict(metrics),
            "fidelity": fidelity,
            "native_fidelity_attempts": 1,
            "native_pair_complete": self._pair_complete(report),
        }
        activation = report.get("control_activated")
        if isinstance(activation, bool):
            result["mechanism_activated"] = activation
        return result

    def _invoke(self, candidate_id: str) -> None:
        child_request = self._child_request(candidate_id)
        self.child_result = review_execute.run(
            child_request,
            base=self.base,
            resume=self.resume or self._child_started,
            admission_config=self.admission_config,
        )
        self._child_started = True
        self._read_nested_state()

    def _candidate_report(self, candidate_id: str) -> dict[str, Any] | None:
        self._read_nested_state()
        if self._nested_conflict is not None:
            return None
        # A report discovered while recovering is only a hint until the
        # child executor has re-opened and validated its own ledger.  This
        # authorization call is also what prevents a stale nested report from
        # completing an outer operation without a current child boundary.
        if self.resume and self.child_result is None and self.reports.get(candidate_id) is not None:
            self._invoke(candidate_id)
        elif self.reports.get(candidate_id) is None and (not self._child_blocks_followup()):
            self._invoke(candidate_id)
        return self.reports.get(candidate_id)

    def pair_settled(self, candidate_id: str) -> bool:
        """Expose atomic child-pair completion to the outer deadline guard.

        Returns:
            ``True`` when a nested report settled the pair.
        """

        self._read_nested_state()
        if self._nested_conflict is not None:
            return False
        report = self.reports.get(candidate_id)
        return (
            report is not None and self._pair_complete(report)
        ) or self._nested_pair_attempts_complete(candidate_id)

    def _pair_complete(self, report: Mapping[str, Any]) -> bool:
        """Report whether the nested child already settled both operations.

        Returns:
            ``True`` when the outer controller must settle both operation
            records even if cancellation arrived during the child call.
        """

        if report.get("status") == "complete":
            return True
        if report.get("status") != "failed":
            return False
        reason = str(report.get("reason", "")).lower()
        return "treatment" in reason and isinstance(report.get("control_metrics"), Mapping)

    def _operation_result(
        self, candidate_id: str, kind: str, *, recovering: bool = False
    ) -> Mapping[str, Any]:
        # A native child invocation is itself one atomic control/treatment
        # pair.  Once its report exists, cancellation must settle both outer
        # operation records from that pair so nested and outer accounting do
        # not diverge; the next candidate is the cancellation boundary.
        if recovering:
            # Recovery is allowed to inspect and settle an already dispatched
            # child operation before honoring a newly requested cancellation.
            # It must not start a new child pair merely because a stale outer
            # dispatch record exists.
            self._read_nested_state()
            if self._nested_conflict is not None:
                return {"status": "failed", "reason": self._nested_conflict}
            report = self.reports.get(candidate_id)
            if report is None and self._cancelled():
                recovered_attempt = self._nested_attempt_result(candidate_id, kind)
                return recovered_attempt or {
                    "status": "cancelled",
                    "reason": "cancellation_requested",
                }
            if report is None or (self.resume and self.child_result is None):
                report = self._candidate_report(candidate_id)
        else:
            self._read_nested_state()
            if self._nested_conflict is not None:
                return {"status": "failed", "reason": self._nested_conflict}
            if self._cancelled():
                # A validated child report carries the explicit activation
                # contract; prefer it over the lower-level attempt metrics.
                # Only a crash-before-report pair may be reconciled directly
                # from its source-bound ledger while cancellation is active.
                if candidate_id not in self.reports:
                    recovered_attempt = self._nested_attempt_result(candidate_id, kind)
                    if recovered_attempt is not None:
                        return recovered_attempt
                    if kind != "treatment":
                        return {"status": "cancelled", "reason": "cancellation_requested"}
            report = self._candidate_report(candidate_id)
        if self._nested_conflict is not None:
            return {"status": "failed", "reason": self._nested_conflict}
        # A child admission/execution failure is authoritative even if an old
        # report remains in the nested ledger.  Retained data must not turn a
        # failed child boundary into an apparent successful operation.
        if self.child_result is not None and self.child_result.status in {
            "failed",
            "unavailable",
            "cancelled",
        }:
            if (
                isinstance(report, Mapping)
                and self._child_failure_allows_pair_report(report)
                and self._is_control_fidelity_failure(report)
                and kind == "control"
            ):
                return self._native_control_result(report, fidelity=False)
            if (
                isinstance(report, Mapping)
                and self._child_failure_allows_pair_report(report)
                and self._is_treatment_failure(report)
                and kind == "control"
            ):
                return self._native_control_result(report, fidelity=True)
            return {
                "status": self.child_result.status,
                "reason": self.child_result.reason or f"native executor {self.child_result.status}",
            }
        if report is None:
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
            if kind == "control" and self._is_control_fidelity_failure(report):
                return self._native_control_result(report, fidelity=False)
            if kind == "control" and self._is_treatment_failure(report):
                return self._native_control_result(report, fidelity=True)
            return {"status": "failed", "reason": report.get("reason", "candidate failed")}
        metrics_key = "control_metrics" if kind == "control" else "treatment_metrics"
        metrics = report.get(metrics_key)
        if not isinstance(metrics, Mapping):
            return {"status": "failed", "reason": f"native report lacks {metrics_key}"}
        activated_key = "control_activated" if kind == "control" else "treatment_activated"
        activation = report.get(activated_key)
        if not isinstance(activation, bool):
            return {
                "status": "unavailable",
                "reason": f"native report lacks {activated_key} activation telemetry",
            }
        result = {
            "status": "ok",
            "metrics": dict(metrics),
            "mechanism_activated": activation,
            "fidelity": True,
            "native_pair_complete": self._pair_complete(report),
        }
        if kind == "control":
            result["native_fidelity_attempts"] = 1
        return result

    def execute(self, **kwargs: Any) -> Mapping[str, Any]:
        candidate = kwargs.get("candidate", {})
        candidate_id = str(candidate.get("intervention_id", ""))
        kind = str(kwargs.get("kind", ""))
        operation_id = kwargs.get("operation_id")
        if isinstance(operation_id, str) and candidate_id and kind in {"control", "treatment"}:
            self._operation_context[operation_id] = (candidate_id, kind)
            self._active_dispatch = (candidate_id, kind)
        return self._operation_result(candidate_id, kind)

    def result_for(self, operation_id: str) -> Mapping[str, Any] | None:
        # The child attempt ledger is the idempotent recovery authority.  A
        # dispatching outer operation is recovered by resuming that ledger,
        # never by blindly issuing a second simulator operation.
        context = self._operation_context.get(operation_id)
        if context is None:
            return None
        candidate_id, kind = context
        self._active_dispatch = (candidate_id, kind)
        return self._operation_result(candidate_id, kind, recovering=True)


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
        source_document = _validate_injected_source_proof(
            request,
            validated.recipe,
            source_admission,
            base=root,
        )
        if source_document is None:
            return ComponentResult(
                request_id=request.request_id,
                component_id=COMPONENT_ID,
                status="unavailable",
                reason=(
                    "source_admission: injected executor requires a validated request/recipe-"
                    "bound source proof below the invocation base"
                ),
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
            source_admission=source_document,
            resume=resume,
            cancel=cancel,
        )
    loop: ExperimentLoop | None = None
    try:
        loop = ExperimentLoop(
            request,
            recipe=validated.recipe,
            budget=validated.budget,
            policy=validated.policy,
            journal_path=output_dir / SESSION_JOURNAL_FILENAME,
            executor=executor,
            source_admission=source_document,
            proof_base=root,
            provenance=provenance,
            session_id=validated.session_id,
            resume=resume,
            cancel=cancel,
        )
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
    finally:
        if loop is not None:
            loop._release_session_lock()


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
