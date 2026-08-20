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

import hashlib
import json
import math
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robot_sf.benchmark.map_runner.map_runner import run_map_batch
from robot_sf.common.atomic_io import atomic_write_text
from robot_sf.evidence.distance_convention import DistanceConvention
from robot_sf.evidence.writers import register_evidence, write_distance_series_csv
from robot_sf.training.scenario_loader import load_scenarios

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HORIZON = 100
DEFAULT_DT = 0.1


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


def _validate_identity(
    record: Mapping[str, Any],
    request: DwaDiagnosticRequest,
    provenance: Mapping[str, Any],
) -> None:
    scenario_id = record.get("scenario_id")
    if str(scenario_id) != request.scenario:
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
        if str(row.get("scenario_id")) == request.scenario and row_seed == request_seed:
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
    steps = trace.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("planner_decision_trace.steps must be a non-empty list")
    if any(not isinstance(step, Mapping) for step in steps):
        raise ValueError("planner_decision_trace.steps must contain objects")
    return tuple(steps)


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
        run_map_batch_fn(
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

    record = read_single_episode_record(result_path)
    provenance_path = Path(f"{result_path}.provenance.json")
    provenance = _read_json_object(provenance_path) if provenance_path.exists() else {}
    _validate_identity(record, request, provenance)
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
    command = step.get("selected_command")
    if command is None:
        command = []
    if not isinstance(command, Sequence) or isinstance(command, (str, bytes)):
        raise ValueError("selected_command must be a sequence")
    window_value = step.get("dynamic_window")
    target_value = step.get("target_goal")
    if window_value is not None and not isinstance(window_value, Mapping):
        raise ValueError("dynamic_window must be an object when present")
    if target_value is not None and not isinstance(target_value, Mapping):
        raise ValueError("target_goal must be an object when present")
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
        "selected_v_mps": float(command[0]) if len(command) > 0 else None,
        "selected_w_radps": float(command[1]) if len(command) > 1 else None,
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
    """Write normalized trace rows with the repository distance convention."""
    if not rows:
        raise ValueError(f"cannot write empty steps CSV: {path}")
    write_distance_series_csv(
        path,
        [{field: row.get(field) for field in fields} for row in rows],
        convention=DistanceConvention.CENTER_CENTER,
    )


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
    marked_payload = (
        {"review_marker": "AI-GENERATED NEEDS-REVIEW", **payload} if review_marker else payload
    )
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
