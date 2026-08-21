"""Shared execution and evidence envelope for bounded DWA diagnostics.

The issue-specific DWA diagnostics answer different questions.  This module owns
only the repeatable operational envelope around them: deterministic scenario
selection, one-episode map-runner execution or replay, planner-trace retrieval,
common row normalization, provenance lookup, and safe artifact publication.

A fourth DWA diagnostic can reuse the envelope by constructing a
``DwaDiagnosticRequest``, calling ``collect_episode``, flattening its returned
``trace_steps`` with ``flatten_trace_step`` plus only its own fields, and using
``summarize_episode``/the atomic writers for shared output plumbing.  Its
thresholds, calculations, comparisons, and report conclusion should remain in
the new wrapper.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robot_sf.benchmark.map_runner.map_runner import run_map_batch
from robot_sf.benchmark.result_provenance import validate_result_provenance_manifest
from robot_sf.common.atomic_io import atomic_write_text
from robot_sf.evidence.distance_convention import DistanceConvention
from robot_sf.evidence.writers import register_evidence
from robot_sf.training.scenario_loader import load_scenarios

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HORIZON = 100
DEFAULT_DT = 0.1
_VALID_EXECUTION_MODES = frozenset({"native", "adapter", "mixed"})
_NON_SUCCESS_EXECUTION_STATUSES = frozenset(
    {"fallback", "degraded", "failed", "partial-failure", "not_available"}
)


@dataclass(frozen=True)
class DwaDiagnosticRequest:
    """Inputs for one deterministic DWA diagnostic episode."""

    config_path: Path
    scenario: str
    seed: int
    algorithm: str
    output_dir: Path
    existing_result: Path | None = None
    episode_id: str | None = None
    matrix_path: Path | None = None
    schema_path: Path | None = None
    horizon: int = DEFAULT_HORIZON
    dt: float = DEFAULT_DT


@dataclass(frozen=True)
class DwaDiagnosticEpisode:
    """Typed source envelope passed from the harness to an issue analyzer."""

    request: DwaDiagnosticRequest
    episode_row: Mapping[str, Any]
    trace_steps: tuple[Mapping[str, Any], ...]
    source_artifacts: Mapping[str, Path]
    provenance: Mapping[str, Any]

    @property
    def steps(self) -> tuple[Mapping[str, Any], ...]:
        """Return the raw planner-decision trace steps."""
        return self.trace_steps


def load_scenario(
    name: str,
    seed: int,
    matrix_path: Path,
    *,
    load_scenarios_fn: Callable[..., Sequence[Mapping[str, Any]]] = load_scenarios,
) -> dict[str, Any]:
    """Load exactly one named scenario and pin it to one seed.

    Duplicate names are rejected instead of silently selecting the last row in
    a mapping, because an ambiguous scenario identity invalidates a diagnostic.

    Returns:
        A copied scenario mapping with one pinned integer seed.
    """
    scenarios = load_scenarios_fn(matrix_path, base_dir=matrix_path.parent)
    matches = [row for row in scenarios if str(row.get("name")) == name]
    if not matches:
        raise KeyError(f"scenario {name!r} is absent from matrix {matrix_path}")
    if len(matches) != 1:
        raise ValueError(f"scenario {name!r} is ambiguous in matrix {matrix_path}")
    scenario = dict(matches[0])
    scenario["seeds"] = [_require_integer(seed, "scenario seed")]
    return scenario


def read_single_episode_record(jsonl_path: Path) -> dict[str, Any]:
    """Read exactly one JSON object from a one-episode JSONL artifact.

    Returns:
        The parsed episode record.
    """
    lines = [line for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError(f"expected exactly one episode record in {jsonl_path}, got {len(lines)}")
    record = json.loads(lines[0])
    if not isinstance(record, dict):
        raise ValueError(f"episode record in {jsonl_path} must be a JSON object")
    return record


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"provenance artifact {path} must be a JSON object")
    try:
        validate_result_provenance_manifest(payload)
    except ValueError as exc:
        raise ValueError(f"invalid provenance artifact {path}: {exc}") from exc
    return payload


def _resolve_path(value: Path | None, fallback: Path | None, label: str) -> Path:
    path = value if value is not None else fallback
    if path is None:
        raise ValueError(f"{label} is required when an existing result is not supplied")
    return path


def _require_integer(value: Any, label: str) -> int:
    """Return a JSON integer without accepting boolean or lossy coercions."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    return value


def _require_finite_number(value: Any, label: str) -> float:
    """Return a finite JSON number without accepting strings or booleans."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{label} must be a finite number")
    return normalized


def _require_non_empty_string(value: Any, label: str) -> str:
    """Return a non-empty string without coercing malformed values."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _validate_identity(
    record: Mapping[str, Any],
    request: DwaDiagnosticRequest,
    provenance: Mapping[str, Any],
) -> None:
    scenario_id = record.get("scenario_id")
    _require_non_empty_string(request.scenario, "request scenario")
    if not isinstance(scenario_id, str) or scenario_id != request.scenario:
        raise ValueError(
            f"episode scenario mismatch: expected {request.scenario!r}, got {scenario_id!r}"
        )
    record_seed = _require_integer(record.get("seed"), "episode record seed")
    request_seed = _require_integer(request.seed, "request seed")
    if record_seed != request_seed:
        raise ValueError(f"episode seed mismatch: expected {request.seed}, got {record_seed}")

    rows = provenance.get("rows")
    if rows is None:
        return
    if not isinstance(rows, list):
        raise ValueError("provenance rows must be a list")
    matching_rows: list[Mapping[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("provenance rows must contain objects")
        row_seed = _require_integer(row.get("seed"), "provenance row seed")
        row_scenario = row.get("scenario_id")
        if not isinstance(row_scenario, str):
            raise ValueError("provenance row scenario_id must be a string")
        if row_scenario == request.scenario and row_seed == request_seed:
            matching_rows.append(row)
    if len(matching_rows) != 1:
        raise ValueError(
            "expected exactly one provenance row for "
            f"{request.scenario!r} seed {request.seed}, got {len(matching_rows)}"
        )


def _extract_trace_steps(record: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    metadata = record.get("algorithm_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("episode record is missing algorithm_metadata")
    trace = metadata.get("planner_decision_trace")
    if not isinstance(trace, Mapping):
        raise ValueError("episode record is missing planner_decision_trace")
    if trace.get("schema_version") != "planner-decision-trace.v1":
        raise ValueError("planner_decision_trace.schema_version must be planner-decision-trace.v1")
    _require_finite_number(trace.get("dt"), "planner_decision_trace.dt")
    steps = trace.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("planner_decision_trace.steps must be a non-empty list")
    for expected_step, step in enumerate(steps):
        if not isinstance(step, Mapping):
            raise ValueError("planner_decision_trace.steps must contain objects")
        _validate_trace_step(step, expected_step=expected_step)
    return tuple(steps)


def _validate_trace_step_index(step: Mapping[str, Any], expected_step: int | None) -> None:
    if "step" not in step:
        raise ValueError("planner trace step is missing step")
    step_index = _require_integer(step.get("step"), "trace step")
    if step_index < 0 or (expected_step is not None and step_index != expected_step):
        raise ValueError(f"planner trace step index must be contiguous from zero, got {step_index}")


def _validate_trace_command(step: Mapping[str, Any]) -> None:
    command = step.get("selected_command")
    if not isinstance(command, Sequence) or isinstance(command, (str, bytes)) or len(command) != 2:
        raise ValueError("selected_command must contain exactly two numeric values")
    _require_finite_number(command[0], "selected_command[0]")
    _require_finite_number(command[1], "selected_command[1]")


def _validate_trace_scalars(step: Mapping[str, Any]) -> None:
    for key in (
        "selected_score",
        "distance_to_goal_m",
        "route_progress_from_start_m",
        "robot_x_m",
        "robot_y_m",
    ):
        if key in step and step[key] is not None:
            _require_finite_number(step[key], f"trace {key}")


def _validate_candidate_counts(step: Mapping[str, Any]) -> None:
    counts: dict[str, int] = {}
    for key in ("candidate_total", "candidate_feasible", "candidate_infeasible"):
        if key in step and step[key] is not None:
            count = _require_integer(step[key], f"trace {key}")
            if count < 0:
                raise ValueError(f"trace {key} must be non-negative")
            counts[key] = count
    if len(counts) == 3 and counts["candidate_total"] != (
        counts["candidate_feasible"] + counts["candidate_infeasible"]
    ):
        raise ValueError("trace candidate counts do not conserve candidate_total")


def _validate_trace_nested_fields(step: Mapping[str, Any]) -> None:
    window = step.get("dynamic_window")
    if window is not None:
        if not isinstance(window, Mapping):
            raise ValueError("dynamic_window must be an object when present")
        for key in ("v_min", "v_max", "w_min", "w_max"):
            if key in window and window[key] is not None:
                _require_finite_number(window[key], f"dynamic_window.{key}")

    target = step.get("target_goal")
    if target is not None:
        if not isinstance(target, Mapping):
            raise ValueError("target_goal must be an object when present")
        if "kind" in target and target["kind"] is not None:
            _require_non_empty_string(target["kind"], "target_goal.kind")
        for key in ("x", "y"):
            if key in target and target[key] is not None:
                _require_finite_number(target[key], f"target_goal.{key}")


def _validate_trace_step(step: Mapping[str, Any], *, expected_step: int | None = None) -> None:
    """Validate the common DWA step fields before issue-specific analysis."""
    _validate_trace_step_index(step, expected_step)
    _validate_trace_command(step)
    _validate_trace_scalars(step)
    _validate_candidate_counts(step)
    _validate_trace_nested_fields(step)


def _normalised_algorithm(value: Any, label: str) -> str:
    return _require_non_empty_string(value, label).strip().lower()


def _validate_record_algorithm(record: Mapping[str, Any], expected_algorithm: str) -> None:
    record_algorithm = record.get("algo")
    if record_algorithm is not None and not isinstance(record_algorithm, str):
        raise ValueError("episode algo must be a string")
    if isinstance(record_algorithm, str) and record_algorithm.strip().lower() != expected_algorithm:
        raise ValueError(
            f"episode algorithm mismatch: expected {expected_algorithm!r}, got {record_algorithm!r}"
        )


def _validate_metadata_algorithm(metadata: Mapping[str, Any], expected_algorithm: str) -> None:
    canonical_algorithm = metadata.get("canonical_algorithm")
    if canonical_algorithm is None:
        canonical_algorithm = metadata.get("algorithm")
    actual_algorithm = _normalised_algorithm(canonical_algorithm, "algorithm metadata identity")
    if actual_algorithm != expected_algorithm:
        raise ValueError(
            "algorithm metadata mismatch: "
            f"expected {expected_algorithm!r}, got {actual_algorithm!r}"
        )


def _validate_metadata_status(metadata: Mapping[str, Any]) -> None:
    status = _normalised_algorithm(metadata.get("status"), "algorithm metadata status")
    if status in _NON_SUCCESS_EXECUTION_STATUSES:
        raise ValueError(f"DWA diagnostic source has non-success execution status {status!r}")
    fallback_or_degraded = metadata.get("fallback_or_degraded")
    if fallback_or_degraded is not None and not isinstance(fallback_or_degraded, bool):
        raise ValueError("algorithm metadata fallback_or_degraded must be boolean")
    if fallback_or_degraded is True or metadata.get("evidence_eligible") is False:
        raise ValueError("DWA diagnostic source is marked fallback or degraded")


def _validate_kinematics(metadata: Mapping[str, Any]) -> None:
    kinematics = metadata.get("planner_kinematics")
    if not isinstance(kinematics, Mapping):
        raise ValueError("algorithm metadata is missing planner_kinematics")
    execution_mode = _normalised_algorithm(
        kinematics.get("execution_mode"), "planner_kinematics.execution_mode"
    )
    if execution_mode not in _VALID_EXECUTION_MODES:
        raise ValueError(f"unsupported DWA diagnostic execution mode {execution_mode!r}")


def _validate_nested_execution_statuses(metadata: Mapping[str, Any]) -> None:
    for block_name in ("foresight_prediction", "adapter_impact"):
        block = metadata.get(block_name)
        if not isinstance(block, Mapping):
            continue
        block_status = block.get("status")
        if (
            isinstance(block_status, str)
            and block_status.strip().lower() in _NON_SUCCESS_EXECUTION_STATUSES
        ):
            raise ValueError(f"DWA diagnostic source {block_name} is {block_status!r}")


def _validate_algorithm_and_execution(
    record: Mapping[str, Any], request: DwaDiagnosticRequest
) -> None:
    """Bind the source row to the requested planner and reject degraded execution."""
    expected_algorithm = _normalised_algorithm(request.algorithm, "request algorithm")
    _validate_record_algorithm(record, expected_algorithm)
    metadata = record.get("algorithm_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("episode record is missing algorithm_metadata")
    _validate_metadata_algorithm(metadata, expected_algorithm)
    _validate_metadata_status(metadata)
    _validate_kinematics(metadata)
    _validate_nested_execution_statuses(metadata)


def _resolve_manifest_path(raw_path: Any, *, base_dir: Path, label: str) -> Path:
    """Resolve a manifest path relative to its sidecar when it is not absolute.

    Returns:
        The normalized absolute artifact path.
    """
    path = Path(_require_non_empty_string(raw_path, label))
    return (base_dir / path if not path.is_absolute() else path).resolve()


def _validate_input_binding(
    inputs: Mapping[str, Any], *, role: str, expected_path: Path | None
) -> None:
    """Bind one request input to the SHA-256 recorded in the result manifest."""
    if expected_path is None:
        return
    if not expected_path.is_file():
        raise ValueError(f"requested provenance input is missing: {expected_path}")
    entry = inputs.get(role)
    if not isinstance(entry, Mapping) or entry.get("artifact_status") != "available":
        raise ValueError(f"provenance input {role} is not available")
    digest = entry.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"provenance input {role} has no SHA-256 binding")
    observed = sha256_file(expected_path)
    if observed.lower() != digest.lower():
        raise ValueError(f"provenance input {role} does not match requested file bytes")


def _validate_provenance_campaign(
    provenance: Mapping[str, Any], request: DwaDiagnosticRequest
) -> None:
    campaign = provenance.get("campaign_identity")
    if not isinstance(campaign, Mapping):
        raise ValueError("provenance campaign_identity must be an object")
    algorithm = _normalised_algorithm(campaign.get("algorithm"), "provenance algorithm")
    if algorithm != _normalised_algorithm(request.algorithm, "request algorithm"):
        raise ValueError("provenance algorithm does not match the request")
    if _require_integer(campaign.get("total_jobs"), "provenance total_jobs") != 1:
        raise ValueError("DWA diagnostic requires exactly one scheduled job")
    if _require_integer(campaign.get("written"), "provenance written") != 1:
        raise ValueError("DWA diagnostic provenance must contain exactly one written row")
    completeness = provenance.get("completeness")
    if not isinstance(completeness, Mapping) or completeness.get("status") != "complete":
        raise ValueError("DWA diagnostic provenance is incomplete")


def _provenance_row(provenance: Mapping[str, Any]) -> Mapping[str, Any]:
    rows = provenance.get("rows")
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], Mapping):
        raise ValueError("DWA diagnostic provenance must contain exactly one row")
    return rows[0]


def _validate_provenance_row_identity(row: Mapping[str, Any], record: Mapping[str, Any]) -> None:
    if row.get("episode_id") != record.get("episode_id"):
        raise ValueError("provenance episode_id does not match the episode record")
    if row.get("config_hash") != record.get("config_hash"):
        raise ValueError("provenance config_hash does not match the episode record")
    if row.get("repo_commit") != record.get("git_hash"):
        raise ValueError("provenance repo_commit does not match the episode record")
    if _require_integer(row.get("jsonl_line"), "provenance jsonl_line") != 0:
        raise ValueError("DWA diagnostic provenance row must point to JSONL line zero")


def _validate_provenance_artifact_binding(
    provenance: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    result_path: Path,
    provenance_path: Path,
) -> None:
    raw_artifacts = provenance.get("raw_artifacts")
    if not isinstance(raw_artifacts, list):
        raise ValueError("provenance raw_artifacts must be a list")
    raw_entry = next(
        (
            entry
            for entry in raw_artifacts
            if isinstance(entry, Mapping) and entry.get("kind") == "episodes_jsonl"
        ),
        None,
    )
    if not isinstance(raw_entry, Mapping) or raw_entry.get("artifact_status") != "available":
        raise ValueError("provenance episodes_jsonl artifact is not available")
    raw_path = _resolve_manifest_path(
        raw_entry.get("path"), base_dir=provenance_path.parent, label="raw artifact path"
    )
    if raw_path != result_path.resolve():
        raise ValueError("provenance raw artifact does not match the episode result")
    raw_digest = raw_entry.get("sha256")
    if not isinstance(raw_digest, str) or sha256_file(result_path).lower() != raw_digest.lower():
        raise ValueError("provenance raw artifact digest does not match the episode result")
    row_raw_path = _resolve_manifest_path(
        row.get("raw_artifact"),
        base_dir=provenance_path.parent,
        label="provenance row raw_artifact",
    )
    if row_raw_path != result_path.resolve():
        raise ValueError("provenance row raw artifact does not match the episode result")


def _validate_provenance_simulator_settings(
    row: Mapping[str, Any], request: DwaDiagnosticRequest
) -> None:
    settings = row.get("simulator_settings")
    if not isinstance(settings, Mapping):
        raise ValueError("provenance simulator_settings must be an object")
    if _require_integer(settings.get("horizon"), "provenance horizon") != _require_integer(
        request.horizon, "request horizon"
    ):
        raise ValueError("provenance horizon does not match the request")
    provenance_dt = _require_finite_number(settings.get("dt"), "provenance dt")
    request_dt = _require_finite_number(request.dt, "request dt")
    if not math.isclose(provenance_dt, request_dt, abs_tol=1e-12):
        raise ValueError("provenance dt does not match the request")


def _validate_provenance_inputs(
    provenance: Mapping[str, Any],
    request: DwaDiagnosticRequest,
    *,
    matrix_path: Path | None,
    schema_path: Path | None,
) -> None:
    inputs = provenance.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("provenance inputs must be an object")
    _validate_input_binding(inputs, role="algo_config", expected_path=Path(request.config_path))
    _validate_input_binding(inputs, role="scenario_matrix", expected_path=matrix_path)
    _validate_input_binding(inputs, role="schema_path", expected_path=schema_path)


def _validate_provenance_bindings(
    provenance: Mapping[str, Any],
    *,
    record: Mapping[str, Any],
    request: DwaDiagnosticRequest,
    result_path: Path,
    provenance_path: Path,
    matrix_path: Path | None,
    schema_path: Path | None,
) -> None:
    """Bind a complete result manifest to the requested row and input artifacts."""
    _validate_provenance_campaign(provenance, request)
    row = _provenance_row(provenance)
    _validate_provenance_row_identity(row, record)
    _validate_provenance_artifact_binding(
        provenance, row, result_path=result_path, provenance_path=provenance_path
    )
    _validate_provenance_simulator_settings(row, request)
    _validate_provenance_inputs(
        provenance, request, matrix_path=matrix_path, schema_path=schema_path
    )


def collect_episode(
    request: DwaDiagnosticRequest,
    *,
    matrix_path: Path | None = None,
    schema_path: Path | None = None,
    run_map_batch_fn: Callable[..., Any] = run_map_batch,
    load_scenario_fn: Callable[..., dict[str, Any]] = load_scenario,
) -> DwaDiagnosticEpisode:
    """Execute or load one DWA episode and return its normalized source envelope.

    Returns:
        A typed episode envelope containing the record, raw planner trace, and
        source provenance paths.
    """
    resolved_matrix = request.matrix_path or matrix_path
    resolved_schema = request.schema_path or schema_path
    _require_integer(request.seed, "request seed")
    _require_integer(request.horizon, "request horizon")
    _require_finite_number(request.dt, "request dt")
    if request.existing_result is not None:
        result_path = request.existing_result
    else:
        resolved_matrix = _resolve_path(resolved_matrix, None, "matrix_path")
        resolved_schema = _resolve_path(resolved_schema, None, "schema_path")
        scenario = load_scenario_fn(request.scenario, request.seed, resolved_matrix)
        request.output_dir.mkdir(parents=True, exist_ok=True)
        episode_name = request.episode_id or request.scenario
        result_path = request.output_dir / f"episodes_{episode_name}.jsonl"
        if result_path.exists():
            result_path.unlink()
        stale_provenance_path = Path(f"{result_path}.provenance.json")
        if stale_provenance_path.exists():
            stale_provenance_path.unlink()
        runner_summary = run_map_batch_fn(
            [scenario],
            result_path,
            schema_path=resolved_schema,
            scenario_path=resolved_matrix,
            horizon=request.horizon,
            dt=request.dt,
            record_forces=False,
            algo=request.algorithm,
            algo_config_path=str(request.config_path),
            benchmark_profile="experimental",
            workers=1,
            resume=False,
            record_planner_decision_trace=True,
        )
        if isinstance(runner_summary, Mapping):
            availability = runner_summary.get("benchmark_availability")
            if isinstance(availability, Mapping):
                readiness = availability.get("readiness_status")
                availability_status = availability.get("availability_status")
                if readiness in {"fallback", "degraded"} or availability_status in {
                    "failed",
                    "partial-failure",
                    "not_available",
                }:
                    raise ValueError(
                        "DWA diagnostic runner did not produce a usable execution: "
                        f"readiness={readiness!r}, availability={availability_status!r}"
                    )

    record = read_single_episode_record(result_path)
    provenance_path = Path(f"{result_path}.provenance.json")
    provenance = _read_json_object(provenance_path) if provenance_path.exists() else {}
    _validate_identity(record, request, provenance)
    _validate_algorithm_and_execution(record, request)
    if provenance:
        _validate_provenance_bindings(
            provenance,
            record=record,
            request=request,
            result_path=result_path,
            provenance_path=provenance_path,
            matrix_path=resolved_matrix,
            schema_path=resolved_schema,
        )
    trace_steps = _extract_trace_steps(record)
    source_artifacts: dict[str, Path] = {"episodes_jsonl": result_path}
    if provenance_path.exists():
        source_artifacts["provenance"] = provenance_path
    return DwaDiagnosticEpisode(
        request=request,
        episode_row=record,
        trace_steps=trace_steps,
        source_artifacts=source_artifacts,
        provenance=provenance,
    )


def flatten_trace_step(
    step: Mapping[str, Any],
    *,
    episode_id: str,
    scenario_id: str,
    seed: int,
    extra_fields: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize common planner-trace fields and append issue-specific fields.

    Returns:
        One flattened trace row suitable for JSON and CSV diagnostics.
    """
    _validate_trace_step(step)
    command = step["selected_command"]
    window_value = step.get("dynamic_window")
    target_value = step.get("target_goal")
    window = window_value if isinstance(window_value, Mapping) else {}
    target = target_value if isinstance(target_value, Mapping) else {}
    normalized_seed = _require_integer(seed, "trace seed")
    normalized_step = _require_integer(step.get("step", -1), "trace step")
    row = {
        "episode_id": episode_id,
        "scenario_id": scenario_id,
        "seed": normalized_seed,
        "step": normalized_step,
        "selected_source": str(step.get("selected_source", "unknown")),
        "selected_v_mps": float(command[0]),
        "selected_w_radps": float(command[1]),
        "selected_score": step.get("selected_score"),
        "constraint_reason": str(step.get("constraint_reason", "unknown")),
        "candidate_total": step.get("candidate_total"),
        "candidate_feasible": step.get("candidate_feasible"),
        "candidate_infeasible": step.get("candidate_infeasible"),
        "feasible_score_min": step.get("feasible_score_min"),
        "feasible_score_max": step.get("feasible_score_max"),
        "dynamic_window_v_min": window.get("v_min"),
        "dynamic_window_v_max": window.get("v_max"),
        "dynamic_window_w_min": window.get("w_min"),
        "dynamic_window_w_max": window.get("w_max"),
        "target_goal_kind": target.get("kind"),
        "target_goal_x": target.get("x"),
        "target_goal_y": target.get("y"),
        "distance_to_goal_m": step.get("distance_to_goal_m"),
        "route_progress_from_start_m": step.get("route_progress_from_start_m"),
        "robot_x_m": step.get("robot_x_m"),
        "robot_y_m": step.get("robot_y_m"),
    }
    if extra_fields:
        row.update(extra_fields)
    return row


def first_unrecoverable_step(rows: Sequence[Mapping[str, Any]]) -> int | None:
    """Return the first step where every rollout candidate is infeasible."""
    for row in rows:
        feasible = row.get("candidate_feasible")
        if feasible is not None and int(feasible) == 0:
            return int(row["step"])
    return None


def first_infeasible_candidate_step(rows: Sequence[Mapping[str, Any]]) -> int | None:
    """Return the first step with at least one infeasible rollout candidate."""
    for row in rows:
        infeasible = row.get("candidate_infeasible")
        if infeasible is not None and int(infeasible) > 0:
            return int(row["step"])
    return None


def route_progress_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return finite route-progress statistics shared by all three diagnostics."""
    if not rows:
        return {"status": "no_steps"}
    distances: list[float] = []
    progresses: list[float] = []
    skipped_non_finite_rows = 0
    skipped_non_finite_cells = 0
    for row in rows:
        row_has_non_finite_value = False
        for key, values in (
            ("distance_to_goal_m", distances),
            ("route_progress_from_start_m", progresses),
        ):
            raw_value = row.get(key)
            if raw_value in (None, ""):
                continue
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                value = float("nan")
            if math.isfinite(value):
                values.append(value)
            else:
                row_has_non_finite_value = True
                skipped_non_finite_cells += 1
        if row_has_non_finite_value:
            skipped_non_finite_rows += 1
    initial = distances[0] if distances else None
    final = distances[-1] if distances else None
    return {
        "initial_distance_to_goal_m": initial,
        "final_distance_to_goal_m": final,
        "min_distance_to_goal_m": min(distances) if distances else None,
        "max_route_progress_from_start_m": max(progresses) if progresses else None,
        "final_route_progress_from_start_m": progresses[-1] if progresses else None,
        "net_progress_m": (float(initial) - float(final))
        if initial is not None and final is not None
        else None,
        "progress_ratio_of_initial": (
            (float(initial) - float(final)) / float(initial)
            if initial not in (None, 0.0) and final is not None
            else None
        ),
        "skipped_non_finite_rows": skipped_non_finite_rows,
        "skipped_non_finite_cells": skipped_non_finite_cells,
    }


def constraint_reason_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Return sorted per-constraint-reason step counts."""
    counts: dict[str, int] = {}
    for row in rows:
        reason = str(row.get("constraint_reason", "unknown"))
        counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


def summarize_episode(
    *,
    episode_id: str,
    record: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    extra_fields: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the common episode summary and merge issue-specific measurements.

    Returns:
        The shared summary fields plus any issue-specific measurements.
    """
    outcome = record.get("outcome", {})
    outcome = outcome if isinstance(outcome, Mapping) else {}
    summary = {
        "episode_id": episode_id,
        "scenario_id": record.get("scenario_id"),
        "seed": record.get("seed"),
        "termination_reason": record.get("termination_reason"),
        "steps": record.get("steps"),
        "route_complete": bool(outcome.get("route_complete")),
        "collision_event": bool(outcome.get("collision_event")),
        "timeout_event": bool(outcome.get("timeout_event")),
        "trace_step_count": len(rows),
        "constraint_reason_counts": constraint_reason_counts(rows),
        "route_progress": route_progress_summary(rows),
        "first_infeasible_candidate_step": first_infeasible_candidate_step(rows),
        "first_all_infeasible_step": first_unrecoverable_step(rows),
        "last_selected_command": {
            "v_mps": rows[-1].get("selected_v_mps") if rows else None,
            "w_radps": rows[-1].get("selected_w_radps") if rows else None,
        },
        "last_selected_score": rows[-1].get("selected_score") if rows else None,
    }
    if extra_fields:
        summary.update(extra_fields)
    return summary


def write_steps_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    """Write normalized trace rows atomically with the repository distance convention."""
    if not rows:
        raise ValueError(f"cannot write empty steps CSV: {path}")
    resolved = DistanceConvention.CENTER_CENTER.value
    stream = io.StringIO(newline="")
    stream.write("# AI-GENERATED NEEDS-REVIEW\n")
    stream.write(f"# distance_convention: {resolved}\n")
    writer = csv.DictWriter(stream, fieldnames=list(fields), lineterminator="\n")
    writer.writeheader()
    writer.writerows({field: row.get(field) for field in fields} for row in rows)
    atomic_write_text(path, stream.getvalue())
    _register_if_evidence(path)


def _register_if_evidence(path: Path) -> None:
    try:
        relative = path.resolve().relative_to(REPO_ROOT)
    except ValueError:
        return
    if Path("docs/context/evidence") not in relative.parents:
        return
    try:
        register_evidence(path, area="benchmark_evidence", repo_root=REPO_ROOT)
    except (FileNotFoundError, ValueError):
        # Publication must remain usable for disposable/external evidence paths.
        return


def write_json_atomic(path: Path, payload: Mapping[str, Any], *, review_marker: bool) -> None:
    """Write deterministic JSON through the durable atomic text writer."""
    marked_payload = dict(payload)
    if review_marker:
        marked_payload["review_marker"] = "AI-GENERATED NEEDS-REVIEW"
    atomic_write_text(path, json.dumps(marked_payload, indent=2, sort_keys=True) + "\n")
    _register_if_evidence(path)


def write_markdown_atomic(path: Path, content: str) -> None:
    """Publish generated Markdown atomically and preserve evidence registration."""
    atomic_write_text(path, content)
    _register_if_evidence(path)


def repo_relative_path(path: Path) -> str:
    """Return a stable repository-relative path when possible."""
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one input artifact."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def trace_commit() -> str:
    """Return the current commit hash for provenance, or ``unknown``."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
