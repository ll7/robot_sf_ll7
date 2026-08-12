"""Parquet analytics export for benchmark episode JSONL records."""

# ruff: noqa: DOC201, C901, PLR0912, PLR0915

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import combinations, pairwise
from pathlib import Path
from typing import Any

from robot_sf.benchmark.analysis_trace import trace_artifact_sha256, trace_coverage
from robot_sf.benchmark.errors import EpisodeRecordInputError
from robot_sf.benchmark.termination_reason import canonical_outcome_flags

try:  # Optional analytics dependency; validated when export is invoked.
    import pyarrow as pa
    import pyarrow.parquet as pq
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    pa = None
    pq = None

EXPORT_SCHEMA_VERSION = "benchmark_parquet_export.v1"
CAMPAIGN_RESULT_STORE_SCHEMA_VERSION = "campaign-result-store.v2"

_TABLE_FILENAMES = {
    "episodes": "episodes.parquet",
    "metrics": "metrics.parquet",
    "scenario_params": "scenario_params.parquet",
    "algorithm_metadata": "algorithm_metadata.parquet",
}


@dataclass(frozen=True)
class ParquetExportResult:
    """Summary of files written by a benchmark Parquet export."""

    output_dir: Path
    record_count: int
    table_paths: dict[str, Path]
    metadata_path: Path
    duckdb_examples_path: Path


@dataclass(frozen=True)
class CampaignResultStoreV2Result:
    """Summary of a provenance-first campaign result store."""

    output_dir: Path
    record_count: int
    table_paths: dict[str, Path]
    manifest_path: Path
    checksum_path: Path


class ParquetDependencyError(RuntimeError):
    """Raised when the optional analytics dependencies are not installed."""


def export_episodes_jsonl_to_parquet(
    input_paths: Sequence[str | Path] | str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> ParquetExportResult:
    """Convert benchmark episode JSONL records into Parquet analytics tables.

    JSONL remains the canonical source artifact. The exported tables are derived
    views intended for SQL analytics, campaign comparison, and failure mining.

    Args:
        input_paths: One or more benchmark episode JSONL files.
        output_dir: Directory that receives the Parquet tables and metadata.
        overwrite: Replace existing export files when True.

    Returns:
        Summary of the generated files and row counts.

    Raises:
        FileExistsError: If export files already exist and overwrite is False.
        RuntimeError: If the optional PyArrow dependency is unavailable.
    """
    pa, pq = _load_pyarrow()
    paths = _normalize_paths(input_paths)
    out_dir = Path(output_dir)
    table_paths = {name: out_dir / filename for name, filename in _TABLE_FILENAMES.items()}
    metadata_path = out_dir / "metadata.json"
    duckdb_examples_path = out_dir / "duckdb_examples.sql"
    _ensure_can_write([*table_paths.values(), metadata_path, duckdb_examples_path], overwrite)

    records = _read_jsonl_files(paths)
    rows = _build_rows(records)

    out_dir.mkdir(parents=True, exist_ok=True)
    schemas = _schemas(pa)
    for table_name, table_rows in rows.items():
        table = pa.Table.from_pylist(table_rows, schema=schemas[table_name])
        pq.write_table(table, table_paths[table_name])

    metadata = _build_metadata(
        paths=paths,
        record_count=len(records),
        rows=rows,
        table_paths=table_paths,
    )
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    duckdb_examples_path.write_text(_duckdb_examples_sql(), encoding="utf-8")

    return ParquetExportResult(
        output_dir=out_dir,
        record_count=len(records),
        table_paths=table_paths,
        metadata_path=metadata_path,
        duckdb_examples_path=duckdb_examples_path,
    )


_CAMPAIGN_V2_TABLE_FILENAMES = {
    "episodes": "episodes.parquet",
    "steps": "steps.parquet",
    "actors": "actors.parquet",
    "events": "events.parquet",
    "features": "features.parquet",
    "cells": "cells.parquet",
    "comparisons": "comparisons.parquet",
}


def export_campaign_result_store_v2(
    input_paths: Sequence[str | Path] | str | Path,
    output_dir: str | Path,
    *,
    study_id: str = "campaign",
    command: str = "",
    overwrite: bool = False,
) -> CampaignResultStoreV2Result:
    """Build the canonical v2 campaign store from episode JSONL records.

    This is an additive companion to the existing Parquet export.  The JSONL
    remains the source of truth; v2 adds normalized state, event, feature, cell,
    and comparison tables while preserving unavailable historical fields.
    """

    pa_mod, pq_mod = _load_pyarrow()
    paths = _normalize_paths(input_paths)
    out_dir = Path(output_dir)
    table_paths = {
        name: out_dir / filename for name, filename in _CAMPAIGN_V2_TABLE_FILENAMES.items()
    }
    manifest_path = out_dir / "manifest.json"
    checksum_path = out_dir / "SHA256SUMS"
    _ensure_can_write([*table_paths.values(), manifest_path, checksum_path], overwrite)
    records = _read_jsonl_files(paths, annotate_source_path=True)
    tables = _build_campaign_v2_rows(records, paths)
    out_dir.mkdir(parents=True, exist_ok=True)
    schemas = _campaign_v2_schemas(pa_mod)
    for table_name, rows in tables.items():
        pq_mod.write_table(
            pa_mod.Table.from_pylist(rows, schema=schemas[table_name]),
            table_paths[table_name],
        )
    manifest = {
        "schema_version": CAMPAIGN_RESULT_STORE_SCHEMA_VERSION,
        "study_id": str(study_id),
        "command": str(command),
        "source_files": [{"path": path.name, "sha256": _path_sha256(path)} for path in paths],
        "tables": {
            name: {"file": path.name, "rows": len(tables[name])}
            for name, path in table_paths.items()
        },
        "unavailable_policy": "missing historical fields are typed unavailable; never inferred",
        "duckdb": "query Parquet tables directly with read_parquet()",
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checksums = [
        f"{_path_sha256(path)}  {path.name}" for path in [*table_paths.values(), manifest_path]
    ]
    checksum_path.write_text("\n".join(checksums) + "\n", encoding="utf-8")
    return CampaignResultStoreV2Result(
        output_dir=out_dir,
        record_count=len(records),
        table_paths=table_paths,
        manifest_path=manifest_path,
        checksum_path=checksum_path,
    )


def _campaign_v2_schemas(pa_mod: Any) -> dict[str, Any]:
    """Return stable schemas for the v2 normalized tables."""

    string = pa_mod.string()
    number = pa_mod.float64()
    integer = pa_mod.int64()
    boolean = pa_mod.bool_()
    return {
        "episodes": pa_mod.schema(
            [
                ("run_id", string),
                ("episode_id", string),
                ("planner", string),
                ("scenario_id", string),
                ("scenario_family", string),
                ("seed", integer),
                ("row_status", string),
                ("artifact_uri", string),
                ("artifact_sha256", string),
                ("trace_coverage_json", string),
                ("analysis_trace_json", string),
                ("outcome_json", string),
                ("provenance_json", string),
                ("execution_status", string),
            ]
        ),
        "steps": pa_mod.schema(
            [
                ("episode_id", string),
                ("step", integer),
                ("time_s", number),
                ("robot_x", number),
                ("robot_y", number),
                ("heading_rad", number),
                ("robot_vx", number),
                ("robot_vy", number),
                ("requested_linear_m_s", number),
                ("requested_turn_rate_rad_s", number),
                ("applied_linear_m_s", number),
                ("applied_turn_rate_rad_s", number),
                ("coordinate_frame", string),
                ("units_json", string),
            ]
        ),
        "actors": pa_mod.schema(
            [
                ("episode_id", string),
                ("step", integer),
                ("actor_id", string),
                ("actor_kind", string),
                ("x", number),
                ("y", number),
                ("vx", number),
                ("vy", number),
                ("heading_rad", number),
                ("radius_m", number),
            ]
        ),
        "events": pa_mod.schema(
            [
                ("episode_id", string),
                ("event_id", string),
                ("event_type", string),
                ("time_s", number),
                ("status", string),
                ("reason", string),
                ("details_json", string),
            ]
        ),
        "features": pa_mod.schema(
            [
                ("episode_id", string),
                ("feature_name", string),
                ("value_number", number),
                ("units", string),
                ("profile_version", string),
                ("status", string),
                ("unavailable_reason", string),
            ]
        ),
        "cells": pa_mod.schema(
            [
                ("cell_id", string),
                ("planner", string),
                ("scenario_id", string),
                ("config_hash", string),
                ("config_digest", string),
                ("scenario_digest", string),
                ("map_digest", string),
                ("outcome_counts_json", string),
                ("entropy", number),
                ("seed_count", integer),
                ("uncertainty_json", string),
                ("boundary_context_json", string),
                ("representative_episode_id", string),
                ("representative_status", string),
                ("boundary_status", string),
                ("outlier_status", string),
            ]
        ),
        "comparisons": pa_mod.schema(
            [
                ("comparison_id", string),
                ("left_episode_id", string),
                ("right_episode_id", string),
                ("compatibility_status", string),
                ("reason", string),
                ("compatibility_receipt_json", string),
                ("outcome_delta", number),
                ("clearance_delta_m", number),
                ("event_time_shift_s", number),
                ("trajectory_separation_m", number),
                ("control_sequence_difference", number),
                ("linear_control_sequence_difference_m_s", number),
                ("turn_control_sequence_difference_rad_s", number),
                ("progress_delta_m", number),
                ("shared_prefix", boolean),
            ]
        ),
    }


def _build_campaign_v2_rows(
    records: Sequence[dict[str, Any]], paths: Sequence[Path]
) -> dict[str, list[dict[str, Any]]]:
    """Normalize records into the seven v2 tables."""

    source_sha = _path_sha256(paths[0]) if paths and paths[0].is_file() else None
    episodes: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    actors: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    features: list[dict[str, Any]] = []
    by_cell: dict[tuple[str, str, str, str, str, str], list[dict[str, Any]]] = {}
    for record in records:
        episode_id = str(record.get("episode_id") or "")
        planner = _resolve_algo(record) or "unknown"
        scenario_id = _string_or_none(record.get("scenario_id")) or "unknown"
        scenario_family = _resolve_scenario_family(record) or scenario_id
        coverage = trace_coverage(record)
        provenance = (
            record.get("provenance") if isinstance(record.get("provenance"), Mapping) else {}
        )
        provenance = dict(provenance)
        source_path = _string_or_none(record.get("_source_path"))
        record_source_sha = _string_or_none(record.get("_source_sha256")) or source_sha
        if source_path:
            # This is source-file lineage, not an inferred trace artifact URI.
            provenance.setdefault("source_file", Path(source_path).name)
            provenance.setdefault("source_file_sha256", record_source_sha)
        result_provenance = record.get("result_provenance")
        if isinstance(result_provenance, Mapping):
            for key, value in result_provenance.items():
                provenance.setdefault(str(key), value)
        trace = (
            record.get("algorithm_metadata", {}).get("analysis_trace")
            if isinstance(record.get("algorithm_metadata"), Mapping)
            else None
        )
        if isinstance(trace, Mapping):
            for key in (
                "map_digest",
                "scenario_digest",
                "config_hash",
                "config_digest",
                "git_hash",
                "planner_commit",
                "dt",
                "horizon",
                "map_file",
                "units",
                "coordinate_frame",
                "actor_geometry",
                "actor_id_source",
            ):
                if provenance.get(key) in (None, "") and trace.get(key) not in (None, ""):
                    provenance[key] = trace.get(key)
        explicit_artifact_sha = _string_or_none(provenance.get("artifact_sha256"))
        if explicit_artifact_sha is None and isinstance(trace, Mapping):
            explicit_artifact_sha = _string_or_none(trace.get("artifact_sha256"))
        row_status = str(record.get("row_status") or record.get("status") or "native")
        episodes.append(
            {
                "run_id": str(
                    record.get("run_id")
                    or provenance.get("run_id")
                    or record_source_sha
                    or "unknown"
                ),
                "episode_id": episode_id,
                "planner": planner,
                "scenario_id": scenario_id,
                "scenario_family": scenario_family,
                "seed": _int_or_none(record.get("seed")),
                "row_status": row_status,
                "artifact_uri": _string_or_none(provenance.get("artifact_uri")),
                "artifact_sha256": explicit_artifact_sha,
                "trace_coverage_json": _json_or_none(coverage),
                "analysis_trace_json": _json_or_none(trace),
                "outcome_json": _json_or_none(record.get("outcome")),
                "provenance_json": _json_or_none(provenance),
                "execution_status": str(record.get("status") or "unknown"),
            }
        )
        trace_steps = trace.get("steps") if isinstance(trace, Mapping) else None
        if isinstance(trace_steps, list) and coverage.get("status") == "complete":
            for raw_step in trace_steps:
                if not isinstance(raw_step, Mapping):
                    continue
                step_row, actor_rows = _campaign_step_rows(episode_id, raw_step)
                steps.append(step_row)
                actors.extend(actor_rows)
            for index, raw_event in (
                enumerate(trace.get("events", [])) if isinstance(trace.get("events"), list) else []
            ):
                if isinstance(raw_event, Mapping):
                    events.append(_campaign_event_row(episode_id, index, raw_event))
        else:
            events.append(
                {
                    "episode_id": episode_id,
                    "event_id": "trace-unavailable",
                    "event_type": "telemetry",
                    "time_s": None,
                    "status": "unavailable",
                    "reason": str(coverage.get("reason") or "trace_unavailable"),
                    "details_json": None,
                }
            )
        episode_features = _campaign_feature_rows(record, episode_id, trace_steps, coverage)
        features.extend(episode_features)
        trace_mapping = trace if isinstance(trace, Mapping) else {}
        by_cell.setdefault(
            (
                planner,
                scenario_id,
                str(trace_mapping.get("config_hash") or provenance.get("config_hash") or ""),
                str(trace_mapping.get("config_digest") or provenance.get("config_digest") or ""),
                str(
                    trace_mapping.get("scenario_digest") or provenance.get("scenario_digest") or ""
                ),
                str(trace_mapping.get("map_digest") or provenance.get("map_digest") or ""),
            ),
            [],
        ).append(record)

    cells = _campaign_cell_rows(by_cell)
    comparisons = _campaign_comparison_rows(records)
    return {
        "episodes": episodes,
        "steps": steps,
        "actors": actors,
        "events": events,
        "features": features,
        "cells": cells,
        "comparisons": comparisons,
    }


def _campaign_step_rows(
    episode_id: str, step: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Flatten one trace step into state and actor rows."""

    robot = step.get("robot") if isinstance(step.get("robot"), Mapping) else {}
    controls = step.get("controls") if isinstance(step.get("controls"), Mapping) else {}
    requested = controls.get("requested") if isinstance(controls.get("requested"), Mapping) else {}
    applied = controls.get("applied") if isinstance(controls.get("applied"), Mapping) else {}
    position = robot.get("position") if isinstance(robot.get("position"), list) else [None, None]
    velocity = robot.get("velocity") if isinstance(robot.get("velocity"), list) else [None, None]
    step_row = {
        "episode_id": episode_id,
        "step": _int_or_none(step.get("step")),
        "time_s": _float_or_none(step.get("time_s")),
        "robot_x": _float_or_none(position[0] if len(position) > 0 else None),
        "robot_y": _float_or_none(position[1] if len(position) > 1 else None),
        "heading_rad": _float_or_none(robot.get("heading")),
        "robot_vx": _float_or_none(velocity[0] if len(velocity) > 0 else None),
        "robot_vy": _float_or_none(velocity[1] if len(velocity) > 1 else None),
        "requested_linear_m_s": _float_or_none(requested.get("linear_m_s")),
        "requested_turn_rate_rad_s": _float_or_none(requested.get("turn_rate_rad_s")),
        "applied_linear_m_s": _float_or_none(applied.get("linear_m_s")),
        "applied_turn_rate_rad_s": _float_or_none(applied.get("turn_rate_rad_s")),
        "coordinate_frame": "world",
        "units_json": _json_or_none(
            {"position": "m", "velocity": "m/s", "heading": "rad", "time": "s"}
        ),
    }
    actor_rows: list[dict[str, Any]] = []
    actor_rows.append(_campaign_actor_row(episode_id, step, "robot", "robot", robot))
    pedestrians = step.get("pedestrians") if isinstance(step.get("pedestrians"), list) else []
    for index, actor in enumerate(pedestrians):
        if isinstance(actor, Mapping):
            actor_rows.append(
                _campaign_actor_row(
                    episode_id,
                    step,
                    str(actor.get("actor_id") or f"pedestrian-{index}"),
                    "pedestrian",
                    actor,
                )
            )
    return step_row, actor_rows


def _campaign_actor_row(
    episode_id: str, step: Mapping[str, Any], actor_id: str, kind: str, actor: Mapping[str, Any]
) -> dict[str, Any]:
    """Build one actor row."""

    position = actor.get("position") if isinstance(actor.get("position"), list) else [None, None]
    velocity = actor.get("velocity") if isinstance(actor.get("velocity"), list) else [None, None]
    return {
        "episode_id": episode_id,
        "step": _int_or_none(step.get("step")),
        "actor_id": actor_id,
        "actor_kind": kind,
        "x": _float_or_none(position[0] if len(position) > 0 else None),
        "y": _float_or_none(position[1] if len(position) > 1 else None),
        "vx": _float_or_none(velocity[0] if len(velocity) > 0 else None),
        "vy": _float_or_none(velocity[1] if len(velocity) > 1 else None),
        "heading_rad": _float_or_none(actor.get("heading")),
        "radius_m": _float_or_none(actor.get("radius_m")),
    }


def _campaign_event_row(episode_id: str, index: int, event: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a typed trace event."""

    event_time = event.get("time_s")
    if event_time is None:
        event_time = event.get("collision_time")
    event_type = event.get("event_type") or event.get("type")
    if event_type is None and (
        event.get("collision_time") is not None
        or event.get("collision_partner_id") is not None
        or event.get("collision") is True
    ):
        event_type = "collision"
    return {
        "episode_id": episode_id,
        "event_id": str(event.get("event_id") or event.get("id") or f"event-{index:04d}"),
        "event_type": str(event_type or "unknown"),
        "time_s": _float_or_none(event_time),
        "status": str(event.get("status") or "observed"),
        "reason": _string_or_none(event.get("reason")),
        "details_json": _json_or_none(event),
    }


def _campaign_feature_rows(
    record: Mapping[str, Any], episode_id: str, trace_steps: Any, coverage: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Derive interpretable feature rows, with typed unavailable reasons."""

    profile = "case-workbench-metrics.v1"
    rows: list[dict[str, Any]] = []
    if not isinstance(trace_steps, list) or not trace_steps or not _trace_metrics_usable(coverage):
        for name, units in (
            ("surface_clearance_min", "m"),
            ("progress", "m"),
            ("control_effort", "composite"),
            ("applied_linear_control_effort", "m"),
            ("applied_turn_control_effort", "rad"),
            ("applied_linear_speed_mean", "m/s"),
            ("applied_turn_rate_abs_integral", "rad"),
            ("event_time", "s"),
            ("ttc_min", "s"),
            ("cpa_min", "m"),
            ("closing_speed_max", "m/s"),
            ("braking_response_time", "s"),
            ("turning_response_time", "s"),
            ("critical_duration_integral", "s"),
            ("stall_duration", "s"),
            ("reversal_count", "count"),
            ("detour_ratio", "ratio"),
            ("clipping_steps", "count"),
            ("fallback_steps", "count"),
            ("outcome_score", "indicator"),
        ):
            rows.append(
                {
                    "episode_id": episode_id,
                    "feature_name": name,
                    "value_number": None,
                    "units": units,
                    "profile_version": profile,
                    "status": "unavailable",
                    "unavailable_reason": str(coverage.get("reason") or "trace_unavailable"),
                }
            )
        return rows
    clearance_values: list[float] = []
    speeds: list[float] = []
    applied_speeds: list[float] = []
    turns: list[float] = []
    for step in trace_steps:
        if not isinstance(step, Mapping):
            continue
        robot = step.get("robot") if isinstance(step.get("robot"), Mapping) else {}
        pos = robot.get("position") if isinstance(robot.get("position"), list) else []
        velocity = robot.get("velocity") if isinstance(robot.get("velocity"), list) else []
        if len(velocity) >= 2 and all(isinstance(v, (int, float)) for v in velocity[:2]):
            speeds.append(float((float(velocity[0]) ** 2 + float(velocity[1]) ** 2) ** 0.5))
        controls = step.get("controls") if isinstance(step.get("controls"), Mapping) else {}
        applied = controls.get("applied") if isinstance(controls.get("applied"), Mapping) else {}
        applied_linear = applied.get("linear_m_s")
        if isinstance(applied_linear, (int, float)):
            applied_speeds.append(abs(float(applied_linear)))
        turn = applied.get("turn_rate_rad_s")
        if isinstance(turn, (int, float)):
            turns.append(abs(float(turn)))
        for actor in (
            step.get("pedestrians", []) if isinstance(step.get("pedestrians"), list) else []
        ):
            if not isinstance(actor, Mapping):
                continue
            actor_pos = actor.get("position") if isinstance(actor.get("position"), list) else []
            if (
                len(pos) >= 2
                and len(actor_pos) >= 2
                and all(isinstance(v, (int, float)) for v in [*pos[:2], *actor_pos[:2]])
            ):
                robot_radius = float(robot.get("radius_m") or 0.0)
                actor_radius = float(actor.get("radius_m") or 0.0)
                distance = (
                    (float(pos[0]) - float(actor_pos[0])) ** 2
                    + (float(pos[1]) - float(actor_pos[1])) ** 2
                ) ** 0.5
                clearance_values.append(distance - robot_radius - actor_radius)
    metadata = record.get("algorithm_metadata")
    trace = metadata.get("analysis_trace") if isinstance(metadata, Mapping) else None
    trace_declared_dt = _float_or_none(trace.get("dt")) if isinstance(trace, Mapping) else None
    dt = trace_declared_dt if trace_declared_dt is not None else _float_or_none(record.get("dt"))
    if dt is None or not math.isfinite(dt) or dt <= 0.0:
        # A complete analysis trace always carries its actual simulator timestep.
        # Do not silently turn an absent value into the historical 0.1 s default.
        return _unavailable_feature_rows(episode_id, "analysis_trace_timestep_unavailable")
    metric_mapping = record.get("metrics") if isinstance(record.get("metrics"), Mapping) else {}
    recorded_clearance = _mapping_number(
        metric_mapping, "surface_clearance_min", "min_surface_clearance", "min_separation"
    )
    recorded_progress = _mapping_number(
        metric_mapping, "progress", "route_progress", "distance_travelled"
    )
    event_time = _first_trace_event_time(record, trace_steps)
    outcome = record.get("outcome") if isinstance(record.get("outcome"), Mapping) else {}
    route_complete, collision = canonical_outcome_flags(outcome)
    outcome_score = 1.0 if route_complete else (-1.0 if collision else 0.0)
    traveled = _trace_path_length(trace_steps)
    route_length = _mapping_number(
        metric_mapping, "shortest_path_length", "route_length", "optimal_path_length"
    )
    detour_ratio = None
    if route_length is not None and route_length > 0.0 and traveled is not None:
        detour_ratio = traveled / route_length
    values = {
        "surface_clearance_min": (
            recorded_clearance
            if recorded_clearance is not None
            else (min(clearance_values) if clearance_values else None),
            "m",
        ),
        "applied_linear_speed_mean": (
            sum(applied_speeds) / len(applied_speeds) if applied_speeds else None,
            "m/s",
        ),
        "applied_turn_rate_abs_integral": (sum(turns) * dt if turns else None, "rad"),
        "progress": (
            recorded_progress if recorded_progress is not None else traveled,
            "m",
        ),
        "control_effort": (
            (sum(applied_speeds) * dt + sum(turns) * dt) if applied_speeds or turns else None,
            "composite",
        ),
        "applied_linear_control_effort": (
            sum(applied_speeds) * dt if applied_speeds else None,
            "m",
        ),
        "applied_turn_control_effort": (
            sum(turns) * dt if turns else None,
            "rad",
        ),
        "event_time": (event_time, "s"),
        "outcome_score": (outcome_score, "indicator"),
        "detour_ratio": (detour_ratio, "ratio"),
    }
    advanced = _advanced_trace_features(trace_steps, dt=dt)
    if detour_ratio is not None:
        advanced["detour_ratio"] = (detour_ratio, "ratio")
    values.update(advanced)
    for name, (value, units) in values.items():
        rows.append(
            {
                "episode_id": episode_id,
                "feature_name": name,
                "value_number": value,
                "units": units,
                "profile_version": profile,
                "status": "available" if value is not None else "unavailable",
                "unavailable_reason": None
                if value is not None
                else "source_trace_missing_metric_fields",
            }
        )
    return rows


def _trace_metrics_usable(coverage: Mapping[str, Any]) -> bool:
    """Return whether local trace fields support formula derivation.

    Provenance failures keep an episode ineligible for evidence admission, but
    they do not make otherwise valid local kinematics mathematically
    unavailable.  Structural/timing/artifact failures remain fail-closed.
    """

    if coverage.get("status") == "complete":
        return True
    required = (
        "has_initial_state",
        "stable_actor_ids",
        "radii",
        "controls",
        "finite_states",
        "monotonic_time",
        "coordinate_frame",
        "units",
        "artifact_hash",
    )
    return all(coverage.get(name) is True for name in required) and coverage.get("timing") is True


def _unavailable_feature_rows(episode_id: str, reason: str) -> list[dict[str, Any]]:
    """Return the complete v1 feature vocabulary with typed unavailable values."""

    profile = "case-workbench-metrics.v1"
    names = (
        ("surface_clearance_min", "m"),
        ("progress", "m"),
        ("control_effort", "composite"),
        ("applied_linear_control_effort", "m"),
        ("applied_turn_control_effort", "rad"),
        ("applied_linear_speed_mean", "m/s"),
        ("applied_turn_rate_abs_integral", "rad"),
        ("event_time", "s"),
        ("ttc_min", "s"),
        ("cpa_min", "m"),
        ("closing_speed_max", "m/s"),
        ("braking_response_time", "s"),
        ("turning_response_time", "s"),
        ("critical_duration_integral", "s"),
        ("stall_duration", "s"),
        ("reversal_count", "count"),
        ("detour_ratio", "ratio"),
        ("clipping_steps", "count"),
        ("fallback_steps", "count"),
        ("outcome_score", "indicator"),
    )
    return [
        {
            "episode_id": episode_id,
            "feature_name": name,
            "value_number": None,
            "units": units,
            "profile_version": profile,
            "status": "unavailable",
            "unavailable_reason": reason,
        }
        for name, units in names
    ]


def derive_episode_metrics(record: Mapping[str, Any]) -> dict[str, float]:
    """Return available v2 feature values for one episode without writing files."""

    metadata = record.get("algorithm_metadata")
    trace = metadata.get("analysis_trace") if isinstance(metadata, Mapping) else None
    trace_steps = trace.get("steps") if isinstance(trace, Mapping) else None
    coverage = record.get("trace_coverage")
    if not isinstance(coverage, Mapping):
        coverage = trace_coverage(dict(record))
    rows = _campaign_feature_rows(
        record,
        str(record.get("episode_id") or ""),
        trace_steps,
        coverage,
    )
    return {
        str(row["feature_name"]): float(row["value_number"])
        for row in rows
        if isinstance(row.get("value_number"), (int, float))
        and not isinstance(row.get("value_number"), bool)
    }


def _advanced_trace_features(
    trace_steps: list[Any], *, dt: float
) -> dict[str, tuple[float | None, str]]:
    """Derive timing/control features from fields present in an analysis trace."""

    speeds: list[float] = []
    times: list[float] = []
    signed_speeds: list[float] = []
    turn_times: list[float] = []
    ttc_values: list[float] = []
    cpa_values: list[float] = []
    frame_clearances: list[float | None] = []
    closing_speeds: list[float] = []
    clipping_steps = 0
    fallback_steps = 0
    for step in trace_steps:
        if not isinstance(step, Mapping):
            continue
        robot = step.get("robot") if isinstance(step.get("robot"), Mapping) else {}
        step_time = _float_or_none(step.get("time_s"))
        if step_time is not None:
            times.append(step_time)
        velocity = robot.get("velocity") if isinstance(robot.get("velocity"), list) else []
        if len(velocity) >= 2 and all(isinstance(v, (int, float)) for v in velocity[:2]):
            vx, vy = float(velocity[0]), float(velocity[1])
            speeds.append(float(math.hypot(vx, vy)))
            heading = robot.get("heading")
            if isinstance(heading, (int, float)):
                signed_speeds.append(vx * math.cos(float(heading)) + vy * math.sin(float(heading)))
            else:
                signed_speeds.append(float("nan"))
        controls = step.get("controls") if isinstance(step.get("controls"), Mapping) else {}
        applied = controls.get("applied") if isinstance(controls.get("applied"), Mapping) else {}
        if not signed_speeds or not math.isfinite(signed_speeds[-1]):
            applied_linear = applied.get("linear_m_s")
            if isinstance(applied_linear, (int, float)):
                if signed_speeds:
                    signed_speeds[-1] = float(applied_linear)
                else:
                    signed_speeds.append(float(applied_linear))
        if (
            isinstance(applied.get("turn_rate_rad_s"), (int, float))
            and abs(float(applied["turn_rate_rad_s"])) > 1e-9
        ):
            turn_time = _float_or_none(step.get("time_s"))
            if turn_time is not None:
                turn_times.append(turn_time)
        planner = step.get("planner") if isinstance(step.get("planner"), Mapping) else {}
        amv = planner.get("amv") if isinstance(planner.get("amv"), Mapping) else {}
        clipping_steps += int(bool(amv.get("command_clipped") or amv.get("yaw_rate_saturated")))
        fallback_steps += int(
            str(planner.get("execution_mode") or "").lower() in {"fallback", "degraded"}
        )
        rp = robot.get("position") if isinstance(robot.get("position"), list) else []
        rv = robot.get("velocity") if isinstance(robot.get("velocity"), list) else []
        step_clearances: list[float] = []
        for actor in (
            step.get("pedestrians", []) if isinstance(step.get("pedestrians"), list) else []
        ):
            if not isinstance(actor, Mapping):
                continue
            ap = actor.get("position") if isinstance(actor.get("position"), list) else []
            av = actor.get("velocity") if isinstance(actor.get("velocity"), list) else []
            if len(rp) < 2 or len(ap) < 2 or len(rv) < 2 or len(av) < 2:
                continue
            rel_p = (float(ap[0]) - float(rp[0]), float(ap[1]) - float(rp[1]))
            rel_v = (float(av[0]) - float(rv[0]), float(av[1]) - float(rv[1]))
            distance = math.hypot(*rel_p)
            robot_radius = float(robot.get("radius_m") or 0.0)
            actor_radius = float(actor.get("radius_m") or 0.0)
            clearance = distance - robot_radius - actor_radius
            step_clearances.append(clearance)
            vv = rel_v[0] ** 2 + rel_v[1] ** 2
            if vv <= 1e-12:
                cpa_values.append(distance)
                continue
            tau = max(0.0, -(rel_p[0] * rel_v[0] + rel_p[1] * rel_v[1]) / vv)
            cpa_values.append(math.hypot(rel_p[0] + tau * rel_v[0], rel_p[1] + tau * rel_v[1]))
            if distance > 1e-12:
                closing_speed = max(0.0, -(rel_p[0] * rel_v[0] + rel_p[1] * rel_v[1]) / distance)
                closing_speeds.append(closing_speed)
            contact_time = _time_to_contact(rel_p, rel_v, robot_radius + actor_radius)
            if contact_time is not None:
                ttc_values.append(contact_time)
        frame_clearances.append(min(step_clearances) if step_clearances else None)
    braking_time = None
    for index in range(1, len(speeds)):
        if speeds[index] < speeds[index - 1] - 1e-9:
            braking_time = times[index] if index < len(times) else float(index) * dt
            break
    stall_duration = _interval_duration(speeds, times, lambda value: value < 1.0e-3, fallback_dt=dt)
    critical_duration = _interval_duration(
        [value if value is not None else float("inf") for value in frame_clearances],
        times,
        lambda value: value <= 0.5,
        fallback_dt=dt,
    )
    finite_signed = [value for value in signed_speeds if math.isfinite(value)]
    reversal_count = sum(
        1
        for before, after in pairwise(finite_signed)
        if abs(before) > 1e-3 and abs(after) > 1e-3 and before * after < 0.0
    )
    return {
        "ttc_min": (min(ttc_values) if ttc_values else None, "s"),
        "cpa_min": (min(cpa_values) if cpa_values else None, "m"),
        "closing_speed_max": (max(closing_speeds) if closing_speeds else None, "m/s"),
        "braking_response_time": (braking_time, "s"),
        "turning_response_time": (min(turn_times) if turn_times else None, "s"),
        "critical_duration_integral": (critical_duration, "s")
        if any(value is not None for value in frame_clearances)
        else (None, "s"),
        "stall_duration": (stall_duration if speeds else None, "s"),
        "reversal_count": (float(reversal_count) if finite_signed else None, "count"),
        "detour_ratio": (None, "ratio"),
        "clipping_steps": (float(clipping_steps), "count"),
        "fallback_steps": (float(fallback_steps), "count"),
    }


def _interval_duration(
    values: list[float],
    times: list[float],
    predicate: Any,
    *,
    fallback_dt: float,
) -> float:
    """Integrate a per-frame predicate over following recorded time intervals."""

    if len(values) < 2:
        return 0.0
    total = 0.0
    for index in range(1, len(values)):
        delta = (
            times[index] - times[index - 1]
            if index < len(times) and index - 1 < len(times)
            else fallback_dt
        )
        if math.isfinite(delta) and delta > 0.0 and predicate(values[index - 1]):
            total += delta
    return total


def _time_to_contact(
    relative_position: tuple[float, float],
    relative_velocity: tuple[float, float],
    radius: float,
) -> float | None:
    """Return the earliest non-negative time at which two discs touch."""

    px, py = relative_position
    vx, vy = relative_velocity
    radius = max(0.0, float(radius))
    c = px * px + py * py - radius * radius
    if c <= 0.0:
        return 0.0
    a = vx * vx + vy * vy
    if a <= 1.0e-12:
        return None
    b = px * vx + py * vy
    if b >= 0.0:
        return None
    discriminant = b * b - a * c
    if discriminant < 0.0:
        return None
    root = math.sqrt(max(0.0, discriminant))
    first = (-b - root) / a
    return first if first >= 0.0 else None


def _record_metric_view(record: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return recorded metrics augmented only by deterministic trace features."""

    recorded = record.get("metrics") if isinstance(record.get("metrics"), Mapping) else {}
    derived = derive_episode_metrics(record)
    merged = dict(derived)
    merged.update(recorded)
    return merged


def _outcome_label(record: Mapping[str, Any]) -> str:
    """Return a canonical outcome label for cell context."""

    outcome = record.get("outcome") if isinstance(record.get("outcome"), Mapping) else {}
    route_complete, collision = canonical_outcome_flags(outcome)
    if collision:
        return "collision"
    if route_complete:
        return "success"
    return str(
        outcome.get("termination_reason")
        or record.get("termination_reason")
        or record.get("status")
        or "unknown"
    )


def _canonical_uncertainty(record: Mapping[str, Any]) -> Mapping[str, Any]:
    """Preserve canonical aggregate uncertainty when the source supplied it."""

    for container_name in ("cell", "aggregate", "canonical_aggregate"):
        container = record.get(container_name)
        if isinstance(container, Mapping):
            uncertainty = container.get("uncertainty")
            if isinstance(uncertainty, Mapping):
                return dict(uncertainty)
    return {"source": "canonical_aggregate", "status": "unavailable"}


def _sha256_text(value: str) -> str:
    """Return a stable SHA-256 digest for a short identity string."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _campaign_cell_rows(
    by_cell: Mapping[tuple[str, str, str, str, str, str], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Aggregate outcome mixtures and entropy per planner/scenario cell."""

    rows: list[dict[str, Any]] = []
    for (
        planner,
        scenario_id,
        config_hash,
        config_digest,
        scenario_digest,
        map_digest,
    ), records in sorted(by_cell.items()):
        eligible_records = [record for record in records if _cell_record_eligible(record)]
        if not eligible_records:
            eligible_records = []
        counts: dict[str, int] = {}
        for record in eligible_records:
            outcome = record.get("outcome") if isinstance(record.get("outcome"), Mapping) else {}
            route_complete, collision = canonical_outcome_flags(outcome)
            label = (
                "collision"
                if collision
                else (
                    "success"
                    if route_complete
                    else str(
                        outcome.get("termination_reason")
                        or record.get("termination_reason")
                        or record.get("status")
                        or "unknown"
                    )
                )
            )
            counts[label] = counts.get(label, 0) + 1
        total = sum(counts.values())
        entropy = (
            -sum(
                (count / total) * math.log2(count / total)
                for count in counts.values()
                if count and total
            )
            if total
            else None
        )
        representative, representative_status = _cell_medoid(eligible_records)
        boundary_status = _cell_boundary_status(eligible_records)
        first_record = eligible_records[0] if eligible_records else {}
        aggregate = _canonical_uncertainty(first_record)
        identity_suffix = _sha256_text(
            "|".join((config_hash, config_digest, scenario_digest, map_digest))
        )[:12]
        rows.append(
            {
                "cell_id": f"{scenario_id}::{planner}::{identity_suffix}",
                "planner": planner,
                "scenario_id": scenario_id,
                "config_hash": config_hash,
                "config_digest": config_digest,
                "scenario_digest": scenario_digest,
                "map_digest": map_digest,
                "outcome_counts_json": _json_or_none(dict(sorted(counts.items()))),
                "entropy": entropy,
                "seed_count": len(eligible_records),
                "uncertainty_json": _json_or_none(aggregate),
                "boundary_context_json": _json_or_none(
                    {
                        "status": boundary_status,
                        "source": "outcome_and_clearance",
                        "geometry_adapter": "unavailable",
                    }
                ),
                "representative_episode_id": representative,
                "representative_status": representative_status,
                "boundary_status": boundary_status,
                "outlier_status": (
                    "candidate"
                    if len(eligible_records) >= 3
                    else "unavailable:insufficient_replicates"
                ),
            }
        )
    return rows


def _cell_record_eligible(record: Mapping[str, Any]) -> bool:
    """Keep fallback/degraded rows out of canonical cell aggregates."""

    blocked_statuses = {
        "fallback",
        "degraded",
        "failed",
        "failure",
        "error",
        "truncated",
        "terminated",
        "unavailable",
        "partial",
        "partial_failure",
        "diagnostic_only",
        "diagnostic_stub",
        "adapter",
    }
    if any(
        str(record.get(key) or "").lower() in blocked_statuses for key in ("row_status", "status")
    ):
        return False
    metadata = record.get("algorithm_metadata")
    if isinstance(metadata, Mapping):
        if metadata.get("evidence_eligible") is False:
            return False
        if str(metadata.get("execution_mode") or "").lower() in {"fallback", "degraded"}:
            return False
        foresight = metadata.get("foresight_prediction")
        if isinstance(foresight, Mapping) and (
            foresight.get("evidence_eligible") is False
            or str(foresight.get("status") or "").lower() in {"fallback", "degraded"}
        ):
            return False
    integrity = record.get("integrity")
    if isinstance(integrity, Mapping) and integrity.get("contradictions"):
        return False
    coverage = record.get("trace_coverage")
    if isinstance(coverage, Mapping) and coverage.get("status") not in {None, "complete"}:
        return False
    metadata = record.get("algorithm_metadata")
    trace = metadata.get("analysis_trace") if isinstance(metadata, Mapping) else None
    provenance = record.get("provenance")
    artifact_sha = provenance.get("artifact_sha256") if isinstance(provenance, Mapping) else None
    if (
        not isinstance(trace, Mapping)
        or not isinstance(artifact_sha, str)
        or artifact_sha != trace.get("artifact_sha256")
        or artifact_sha != trace_artifact_sha256(trace)
    ):
        return False
    return True


def _cell_medoid(records: Sequence[Mapping[str, Any]]) -> tuple[str | None, str]:
    """Select a deterministic metric medoid for one scenario/planner cell."""

    if not records:
        return None, "unavailable:no_records"
    vectors = []
    for record in records:
        metrics = _record_metric_view(record)
        vectors.append(
            (
                _mapping_number(metrics, "surface_clearance_min", "min_surface_clearance"),
                _mapping_number(metrics, "progress", "route_progress"),
                _mapping_number(metrics, "control_effort", "action_effort"),
                str(record.get("episode_id") or ""),
            )
        )
    centers = []
    for index in range(3):
        values = sorted(value[index] for value in vectors if value[index] is not None)
        centers.append(values[len(values) // 2] if values else None)
    ranked = sorted(
        vectors,
        key=lambda value: (
            sum(
                abs(float(value[index]) - float(centers[index]))
                for index in range(3)
                if value[index] is not None and centers[index] is not None
            ),
            value[3],
        ),
    )
    return ranked[0][3], "medoid"


def _cell_boundary_status(records: Sequence[Mapping[str, Any]]) -> str:
    """Classify observed outcome/clearance boundary context without geometry claims."""

    labels = {_outcome_label(record) for record in records}
    clearances = []
    for record in records:
        metrics = _record_metric_view(record)
        value = _mapping_number(metrics, "surface_clearance_min", "min_surface_clearance")
        if value is not None:
            clearances.append(value)
    if len(labels) > 1:
        return "mixed_outcomes"
    if clearances and min(clearances) <= 0.0 <= max(clearances):
        return "clearance_boundary"
    return "not_observed"


def _campaign_comparison_rows(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create conservative pair receipts; incompatible starts are never compared."""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        key = str(record.get("scenario_id") or "unknown")
        grouped.setdefault(key, []).append(record)
    rows: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        ordered = sorted(
            group, key=lambda row: (str(row.get("algo") or ""), str(row.get("episode_id") or ""))
        )
        for left, right in combinations(ordered, 2):
            same_planner = str(left.get("algo") or "") == str(right.get("algo") or "")
            same_seed = _int_or_none(left.get("seed")) == _int_or_none(right.get("seed"))
            if same_planner and same_seed:
                continue
            compatible, reason = _comparison_compatibility(left, right)
            deltas = _comparison_deltas(left, right) if compatible else {}
            receipt = _comparison_receipt(left, right, compatible=compatible, reason=reason)
            rows.append(
                {
                    "comparison_id": f"{left.get('episode_id')}__{right.get('episode_id')}",
                    "left_episode_id": str(left.get("episode_id") or ""),
                    "right_episode_id": str(right.get("episode_id") or ""),
                    "compatibility_status": "compatible" if compatible else "incompatible",
                    "reason": None if compatible else reason,
                    "compatibility_receipt_json": _json_or_none(receipt),
                    "outcome_delta": deltas.get("outcome_delta"),
                    "clearance_delta_m": deltas.get("clearance_delta_m"),
                    "event_time_shift_s": deltas.get("event_time_shift_s"),
                    "trajectory_separation_m": deltas.get("trajectory_separation_m"),
                    "control_sequence_difference": deltas.get("control_sequence_difference"),
                    "linear_control_sequence_difference_m_s": deltas.get(
                        "linear_control_sequence_difference_m_s"
                    ),
                    "turn_control_sequence_difference_rad_s": deltas.get(
                        "turn_control_sequence_difference_rad_s"
                    ),
                    "progress_delta_m": deltas.get("progress_delta_m"),
                    "shared_prefix": False,
                }
            )
    return rows


def _comparison_compatibility(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> tuple[bool, str]:
    """Check the receipts required before deriving pair deltas."""

    for record in (left, right):
        if not _comparison_record_eligible(record):
            return False, "execution_or_provenance_ineligible"
    left_trace = _analysis_trace(left)
    right_trace = _analysis_trace(right)
    if left_trace is None or right_trace is None:
        return False, "analysis_trace_unavailable"
    left_config = _string_or_none(left_trace.get("config_digest"))
    right_config = _string_or_none(right_trace.get("config_digest"))
    if left_config is None or right_config is None or left_config != right_config:
        return False, "config_digest_mismatch"
    for field, reason in (
        ("scenario_digest", "scenario_digest_mismatch"),
        ("map_digest", "map_digest_mismatch"),
        ("coordinate_frame", "coordinate_frame_mismatch"),
    ):
        left_value = _string_or_none(left_trace.get(field))
        right_value = _string_or_none(right_trace.get(field))
        if left_value is None or right_value is None or left_value != right_value:
            return False, reason
    if left_trace.get("units") != right_trace.get("units"):
        return False, "units_mismatch"
    if _float_or_none(left_trace.get("dt")) != _float_or_none(right_trace.get("dt")):
        return False, "dt_mismatch"
    if left_trace.get("horizon") != right_trace.get("horizon"):
        return False, "horizon_mismatch"
    if left_trace.get("actor_geometry") != right_trace.get("actor_geometry"):
        return False, "actor_geometry_mismatch"
    for record, record_trace in ((left, left_trace), (right, right_trace)):
        for status_value in (record.get("row_status"), record.get("status")):
            status = str(status_value or "").lower()
            if status in {
                "fallback",
                "degraded",
                "failed",
                "failure",
                "error",
                "truncated",
                "terminated",
                "unavailable",
                "partial",
                "diagnostic_only",
                "diagnostic_stub",
                "adapter",
            }:
                return False, f"execution_status:{status}"
        provenance = record.get("provenance")
        if not isinstance(provenance, Mapping) or not provenance.get("artifact_sha256"):
            return False, "artifact_sha256_missing"
        embedded_sha = _string_or_none(record_trace.get("artifact_sha256"))
        if embedded_sha is None or embedded_sha != trace_artifact_sha256(record_trace):
            return False, "artifact_sha256_invalid"
        if provenance.get("artifact_sha256") != embedded_sha:
            return False, "artifact_sha256_mismatch"
    left_start = _trace_start_signature(left_trace)
    right_start = _trace_start_signature(right_trace)
    if left_start is None or right_start is None or not _close_nested(left_start, right_start):
        return False, "initial_state_mismatch"
    return True, ""


def _comparison_record_eligible(record: Mapping[str, Any]) -> bool:
    """Apply the same fail-closed execution caveats used by case selection."""

    blocked = {
        "fallback",
        "degraded",
        "failed",
        "failure",
        "error",
        "truncated",
        "terminated",
        "unavailable",
        "partial",
        "partial_failure",
        "diagnostic_only",
        "diagnostic_stub",
        "adapter",
    }
    if any(str(record.get(key) or "").lower() in blocked for key in ("row_status", "status")):
        return False
    metadata = record.get("algorithm_metadata")
    if isinstance(metadata, Mapping):
        if metadata.get("evidence_eligible") is False:
            return False
        if any(
            str(metadata.get(key) or "").lower() in blocked
            for key in ("status", "readiness_status", "preflight_status", "execution_mode")
        ):
            return False
        foresight = metadata.get("foresight_prediction")
        if isinstance(foresight, Mapping) and (
            foresight.get("evidence_eligible") is False
            or str(foresight.get("status") or "").lower() in blocked
        ):
            return False
    integrity = record.get("integrity")
    return not (isinstance(integrity, Mapping) and integrity.get("contradictions"))


def _comparison_receipt(
    left: Mapping[str, Any], right: Mapping[str, Any], *, compatible: bool, reason: str
) -> dict[str, Any]:
    """Serialize the exact compatibility inputs so a delta is auditable."""

    def trace_fields(record: Mapping[str, Any]) -> dict[str, Any]:
        trace = _analysis_trace(record) or {}
        return {
            "episode_id": record.get("episode_id"),
            "planner": record.get("algo") or record.get("planner"),
            "seed": record.get("seed"),
            "config_hash": trace.get("config_hash"),
            "config_digest": trace.get("config_digest"),
            "scenario_digest": trace.get("scenario_digest"),
            "map_digest": trace.get("map_digest"),
            "coordinate_frame": trace.get("coordinate_frame"),
            "units": trace.get("units"),
            "dt": trace.get("dt"),
            "horizon": trace.get("horizon"),
            "actor_geometry": trace.get("actor_geometry"),
            "artifact_sha256": trace.get("artifact_sha256"),
            "start_signature": _trace_start_signature(trace),
        }

    return {
        "schema_version": "comparison-receipt.v1",
        "status": "compatible" if compatible else "incompatible",
        "reason": reason or None,
        "left": trace_fields(left),
        "right": trace_fields(right),
        "shared_prefix": False,
        "alignment": "absolute_time_and_recorded_step_index",
        "dtw": False,
    }


def is_comparison_compatible(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Return whether two records have a complete physical compatibility receipt."""

    compatible, _reason = _comparison_compatibility(left, right)
    return compatible


def _comparison_deltas(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> dict[str, float | None]:
    """Derive absolute-time, index-aligned deltas without time warping."""

    left_trace = _analysis_trace(left)
    right_trace = _analysis_trace(right)
    if left_trace is None or right_trace is None:
        return {}
    left_metrics = _trace_comparison_metrics(left, left_trace)
    right_metrics = _trace_comparison_metrics(right, right_trace)
    aligned = _aligned_trace_pairs(left_trace, right_trace)
    trajectory = None
    control = None
    linear_control = None
    turn_control = None
    if aligned:
        trajectory = sum(
            math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])) for a, b in aligned
        ) / len(aligned)
        control_values = [
            abs(float(a[2]) - float(b[2])) + abs(float(a[3]) - float(b[3]))
            for a, b in aligned
            if a[2] is not None and a[3] is not None and b[2] is not None and b[3] is not None
        ]
        control = sum(control_values) / len(control_values) if control_values else None
        linear_values = [
            abs(float(a[2]) - float(b[2]))
            for a, b in aligned
            if a[2] is not None and b[2] is not None
        ]
        turn_values = [
            abs(float(a[3]) - float(b[3]))
            for a, b in aligned
            if a[3] is not None and b[3] is not None
        ]
        linear_control = sum(linear_values) / len(linear_values) if linear_values else None
        turn_control = sum(turn_values) / len(turn_values) if turn_values else None
    return {
        "outcome_delta": _outcome_number(right) - _outcome_number(left),
        "clearance_delta_m": _difference(
            right_metrics.get("clearance"), left_metrics.get("clearance")
        ),
        "event_time_shift_s": _difference(
            right_metrics.get("event_time"), left_metrics.get("event_time")
        ),
        "trajectory_separation_m": trajectory,
        "control_sequence_difference": control,
        "linear_control_sequence_difference_m_s": linear_control,
        "turn_control_sequence_difference_rad_s": turn_control,
        "progress_delta_m": _difference(
            right_metrics.get("progress"), left_metrics.get("progress")
        ),
    }


def _analysis_trace(record: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Return the complete analysis trace when present."""

    metadata = record.get("algorithm_metadata")
    trace = metadata.get("analysis_trace") if isinstance(metadata, Mapping) else None
    if not isinstance(trace, Mapping) or not isinstance(trace.get("steps"), list):
        return None
    coverage = trace_coverage(dict(record))
    if coverage.get("status") != "complete":
        return None
    return trace


def _trace_start_signature(trace: Mapping[str, Any]) -> tuple[Any, ...] | None:
    """Build a comparable signature for the serialized initial state."""

    steps = trace.get("steps")
    if not isinstance(steps, list) or not steps or not isinstance(steps[0], Mapping):
        return None
    first = steps[0]
    robot = first.get("robot") if isinstance(first.get("robot"), Mapping) else None
    if robot is None:
        return None
    position = robot.get("position") if isinstance(robot.get("position"), list) else None
    if not isinstance(position, list) or len(position) < 2:
        return None
    actors = []
    for actor in first.get("pedestrians", []) if isinstance(first.get("pedestrians"), list) else []:
        if not isinstance(actor, Mapping):
            return None
        actor_position = actor.get("position") if isinstance(actor.get("position"), list) else None
        if not isinstance(actor_position, list) or len(actor_position) < 2:
            return None
        actors.append(
            (
                str(actor.get("actor_id") or ""),
                float(actor_position[0]),
                float(actor_position[1]),
                _float_or_none(actor.get("radius_m")),
                tuple(_float_or_none(value) for value in (actor.get("velocity") or [])),
            )
        )
    return (
        float(position[0]),
        float(position[1]),
        _float_or_none(robot.get("heading")),
        _float_or_none(robot.get("radius_m")),
        tuple(_float_or_none(value) for value in (robot.get("velocity") or [])),
        tuple(sorted(actors)),
    )


def _close_nested(left: Any, right: Any, *, tolerance: float = 1.0e-9) -> bool:
    """Compare nested numeric signatures without accepting missing values."""

    if isinstance(left, tuple) and isinstance(right, tuple):
        return len(left) == len(right) and all(
            _close_nested(a, b, tolerance=tolerance) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return (
            math.isfinite(float(left))
            and math.isfinite(float(right))
            and abs(float(left) - float(right)) <= tolerance
        )
    return left == right


def _aligned_trace_pairs(
    trace_left: Mapping[str, Any], trace_right: Mapping[str, Any]
) -> list[
    tuple[
        tuple[float, float, float | None, float | None],
        tuple[float, float, float | None, float | None],
    ]
]:
    """Align traces by absolute time/index; never warp or normalize duration."""

    left_steps = trace_left.get("steps")
    right_steps = trace_right.get("steps")
    if not isinstance(left_steps, list) or not isinstance(right_steps, list):
        return []
    pairs = []
    for left_step, right_step in zip(left_steps, right_steps, strict=False):
        if not isinstance(left_step, Mapping) or not isinstance(right_step, Mapping):
            continue
        left_time = _float_or_none(left_step.get("time_s"))
        right_time = _float_or_none(right_step.get("time_s"))
        if left_time is None or right_time is None or abs(left_time - right_time) > 1.0e-9:
            continue
        left_robot = left_step.get("robot") if isinstance(left_step.get("robot"), Mapping) else {}
        right_robot = (
            right_step.get("robot") if isinstance(right_step.get("robot"), Mapping) else {}
        )
        left_position = (
            left_robot.get("position") if isinstance(left_robot.get("position"), list) else []
        )
        right_position = (
            right_robot.get("position") if isinstance(right_robot.get("position"), list) else []
        )
        if len(left_position) < 2 or len(right_position) < 2:
            continue
        left_controls = (
            left_step.get("controls") if isinstance(left_step.get("controls"), Mapping) else {}
        )
        right_controls = (
            right_step.get("controls") if isinstance(right_step.get("controls"), Mapping) else {}
        )
        left_control = (
            left_controls.get("applied")
            if isinstance(left_controls.get("applied"), Mapping)
            else {}
        )
        right_control = (
            right_controls.get("applied")
            if isinstance(right_controls.get("applied"), Mapping)
            else {}
        )
        pairs.append(
            (
                (
                    float(left_position[0]),
                    float(left_position[1]),
                    _float_or_none(left_control.get("linear_m_s")),
                    _float_or_none(left_control.get("turn_rate_rad_s")),
                ),
                (
                    float(right_position[0]),
                    float(right_position[1]),
                    _float_or_none(right_control.get("linear_m_s")),
                    _float_or_none(right_control.get("turn_rate_rad_s")),
                ),
            )
        )
    return pairs


def _trace_comparison_metrics(
    record: Mapping[str, Any], trace: Mapping[str, Any]
) -> dict[str, float | None]:
    """Compute pairwise scalar metrics from one complete trace."""

    features = _campaign_feature_rows(
        record, str(record.get("episode_id") or ""), trace.get("steps"), {"status": "complete"}
    )
    values = {str(row["feature_name"]): row.get("value_number") for row in features}
    return {
        "clearance": _float_or_none(values.get("surface_clearance_min")),
        "progress": _float_or_none(values.get("progress")),
        "event_time": _float_or_none(values.get("event_time")),
    }


def _difference(right: float | None, left: float | None) -> float | None:
    """Subtract two values only when both are observed."""

    return None if right is None or left is None else float(right) - float(left)


def _outcome_number(record: Mapping[str, Any]) -> float:
    """Map terminal outcomes to a descriptive numeric delta."""

    outcome = record.get("outcome") if isinstance(record.get("outcome"), Mapping) else {}
    route_complete, collision = canonical_outcome_flags(outcome)
    if route_complete:
        return 1.0
    if collision:
        return -1.0
    return 0.0


def _load_pyarrow() -> tuple[Any, Any]:
    """Load PyArrow modules only when the export path is used."""
    if pa is None or pq is None:  # pragma: no cover - environment dependent
        msg = (
            "Parquet export requires optional analytics dependencies. "
            "Install them with `uv sync --extra analytics` or `uv sync --all-extras`."
        )
        raise ParquetDependencyError(msg)
    return pa, pq


def _normalize_paths(input_paths: Sequence[str | Path] | str | Path) -> list[Path]:
    """Normalize one or more input paths."""
    if isinstance(input_paths, str | Path):
        return [Path(input_paths)]
    return [Path(path) for path in input_paths]


def _ensure_can_write(paths: Sequence[Path], overwrite: bool) -> None:
    """Fail before partial writes when export files already exist."""
    if overwrite:
        return
    existing = [path for path in paths if path.exists()]
    if existing:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Parquet export output already exists: {names}")


def _schemas(pa: Any) -> dict[str, Any]:
    """Return fixed PyArrow schemas for all exported tables."""
    typed_value_fields = [
        ("value_number", pa.float64()),
        ("value_bool", pa.bool_()),
        ("value_text", pa.string()),
        ("value_json", pa.string()),
    ]
    return {
        "episodes": pa.schema(
            [
                ("episode_id", pa.string()),
                ("scenario_id", pa.string()),
                ("seed", pa.int64()),
                ("started_at_utc", pa.string()),
                ("finished_at_utc", pa.string()),
                ("total_runtime_sec", pa.float64()),
                ("algo", pa.string()),
                ("scenario_family", pa.string()),
                ("termination_reason", pa.string()),
                ("version", pa.string()),
                ("outcome_json", pa.string()),
                ("integrity_json", pa.string()),
                ("record_json_sha256", pa.string()),
            ]
        ),
        "metrics": pa.schema(
            [
                ("episode_id", pa.string()),
                ("metric_path", pa.string()),
                *typed_value_fields,
            ]
        ),
        "scenario_params": pa.schema(
            [
                ("episode_id", pa.string()),
                ("param_path", pa.string()),
                *typed_value_fields,
            ]
        ),
        "algorithm_metadata": pa.schema(
            [
                ("episode_id", pa.string()),
                ("metadata_path", pa.string()),
                *typed_value_fields,
            ]
        ),
    }


def _build_rows(records: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Build normalized table rows from benchmark episode records."""
    episode_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    scenario_param_rows: list[dict[str, Any]] = []
    algorithm_metadata_rows: list[dict[str, Any]] = []

    for record in records:
        episode_id = str(record.get("episode_id", ""))
        episode_rows.append(_episode_row(record, episode_id))
        metric_rows.extend(
            _key_value_rows(
                episode_id=episode_id,
                source=record.get("metrics"),
                path_column="metric_path",
            )
        )
        scenario_param_rows.extend(
            _key_value_rows(
                episode_id=episode_id,
                source=record.get("scenario_params"),
                path_column="param_path",
            )
        )
        algorithm_metadata_rows.extend(
            _key_value_rows(
                episode_id=episode_id,
                source=record.get("algorithm_metadata"),
                path_column="metadata_path",
            )
        )

    return {
        "episodes": episode_rows,
        "metrics": metric_rows,
        "scenario_params": scenario_param_rows,
        "algorithm_metadata": algorithm_metadata_rows,
    }


def _episode_row(record: Mapping[str, Any], episode_id: str) -> dict[str, Any]:
    """Build the fixed top-level episode row."""
    return {
        "episode_id": episode_id,
        "scenario_id": _string_or_none(record.get("scenario_id")),
        "seed": _int_or_none(record.get("seed")),
        "started_at_utc": _resolve_started_at_utc(record),
        "finished_at_utc": _resolve_finished_at_utc(record),
        "total_runtime_sec": _resolve_total_runtime_sec(record),
        "algo": _resolve_algo(record),
        "scenario_family": _resolve_scenario_family(record),
        "termination_reason": _string_or_none(record.get("termination_reason")),
        "version": _string_or_none(record.get("version")),
        "outcome_json": _json_or_none(record.get("outcome")),
        "integrity_json": _json_or_none(record.get("integrity")),
        "record_json_sha256": hashlib.sha256(_json_dumps(record).encode("utf-8")).hexdigest(),
    }


def _key_value_rows(
    *,
    episode_id: str,
    source: Any,
    path_column: str,
) -> list[dict[str, Any]]:
    """Convert a nested mapping into long-form typed key/value rows."""
    if not isinstance(source, Mapping):
        return []

    rows: list[dict[str, Any]] = []
    for path, value in _iter_leaf_values(source):
        value_columns = _typed_value_columns(value)
        rows.append({"episode_id": episode_id, path_column: path, **value_columns})
    return rows


def _iter_leaf_values(source: Mapping[str, Any], prefix: str = "") -> list[tuple[str, Any]]:
    """Flatten nested mappings into dotted leaf paths."""
    rows: list[tuple[str, Any]] = []
    for key in sorted(source):
        value = source[key]
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping) and value:
            rows.extend(_iter_leaf_values(value, path))
        else:
            rows.append((path, value))
    return rows


def _typed_value_columns(value: Any) -> dict[str, Any]:
    """Represent a Python value across stable typed Parquet columns."""
    value_number = None
    value_bool = None
    value_text = None
    value_json = None
    if isinstance(value, bool):
        value_bool = value
    elif isinstance(value, int | float):
        value_number = float(value)
    elif isinstance(value, str):
        value_text = value
    elif value is not None:
        value_json = _json_dumps(value)
    return {
        "value_number": value_number,
        "value_bool": value_bool,
        "value_text": value_text,
        "value_json": value_json,
    }


def _resolve_algo(record: Mapping[str, Any]) -> str | None:
    """Resolve the planner/algorithm identifier with benchmark metadata fallbacks."""
    for value in (
        _nested_value(record, "scenario_params.algo"),
        record.get("algo"),
        _nested_value(record, "algorithm_metadata.algorithm"),
        _nested_value(record, "algorithm_metadata.canonical_algorithm"),
    ):
        text = _string_or_none(value)
        if text:
            return text
    return None


def _resolve_scenario_family(record: Mapping[str, Any]) -> str | None:
    """Resolve a scenario-family key suitable for grouped analytics."""
    for value in (
        _nested_value(record, "scenario_params.scenario_family"),
        _nested_value(record, "scenario_params.family"),
        record.get("scenario_family"),
    ):
        text = _string_or_none(value)
        if text:
            return text
    scenario_id = _string_or_none(record.get("scenario_id"))
    if scenario_id and "_" in scenario_id:
        return scenario_id.split("_", maxsplit=1)[0]
    return scenario_id


def _nested_value(source: Mapping[str, Any], path: str) -> Any:
    """Resolve a dotted path from a nested mapping."""
    current: Any = source
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _int_or_none(value: Any) -> int | None:
    """Coerce an integer-like value when possible."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _float_or_none(value: Any) -> float | None:
    """Coerce a numeric value when possible."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _mapping_number(mapping: Mapping[str, Any], *keys: str) -> float | None:
    """Return the first finite numeric value from a mapping."""

    for key in keys:
        value = _float_or_none(mapping.get(key))
        if value is not None and math.isfinite(value):
            return value
    return None


def _trace_path_length(trace_steps: Any) -> float | None:
    """Compute absolute robot path length from consecutive recorded states."""

    positions: list[tuple[float, float]] = []
    if not isinstance(trace_steps, list):
        return None
    for step in trace_steps:
        robot = (
            step.get("robot")
            if isinstance(step, Mapping) and isinstance(step.get("robot"), Mapping)
            else {}
        )
        position = robot.get("position") if isinstance(robot.get("position"), list) else []
        if len(position) >= 2 and all(isinstance(value, (int, float)) for value in position[:2]):
            positions.append((float(position[0]), float(position[1])))
    if len(positions) < 2:
        return 0.0 if positions else None
    return sum(
        math.hypot(after[0] - before[0], after[1] - before[1])
        for before, after in pairwise(positions)
    )


def _first_trace_event_time(record: Mapping[str, Any], trace_steps: Any) -> float | None:
    """Return the earliest observed safety/event time without inferring one."""

    candidates: list[float] = []
    trace = (
        record.get("algorithm_metadata", {}).get("analysis_trace")
        if isinstance(record.get("algorithm_metadata"), Mapping)
        else None
    )
    events = trace.get("events") if isinstance(trace, Mapping) else None
    if isinstance(events, list):
        for event in events:
            if isinstance(event, Mapping):
                raw_time = event.get("time_s")
                if raw_time is None:
                    raw_time = event.get("collision_time")
                value = _float_or_none(raw_time)
                if value is not None and math.isfinite(value):
                    candidates.append(value)
    if isinstance(trace_steps, list):
        for step in trace_steps:
            if (
                isinstance(step, Mapping)
                and isinstance(step.get("events"), list)
                and step["events"]
            ):
                value = _float_or_none(step.get("time_s"))
                if value is not None and math.isfinite(value):
                    candidates.append(value)
    metric_mapping = record.get("metrics") if isinstance(record.get("metrics"), Mapping) else {}
    metric_time = _mapping_number(metric_mapping, "event_time", "first_collision_time")
    if metric_time is not None:
        candidates.append(metric_time)
    return min(candidates) if candidates else None


def _string_or_none(value: Any) -> str | None:
    """Coerce non-empty string-like values."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _json_or_none(value: Any) -> str | None:
    """Serialize structured values for fixed top-level episode columns."""
    if value is None:
        return None
    return _json_dumps(value)


def _json_dumps(value: Any) -> str:
    """Serialize JSON deterministically for hashes and stored JSON columns."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _resolve_started_at_utc(record: Mapping[str, Any]) -> str | None:
    """Resolve episode start time from canonical or legacy provenance fields."""
    return _string_or_none(record.get("started_at_utc")) or _string_or_none(
        _nested_value(record, "timestamps.start")
    )


def _resolve_finished_at_utc(record: Mapping[str, Any]) -> str | None:
    """Resolve episode finish time from canonical or legacy provenance fields."""
    return _string_or_none(record.get("finished_at_utc")) or _string_or_none(
        _nested_value(record, "timestamps.end")
    )


def _resolve_total_runtime_sec(record: Mapping[str, Any]) -> float | None:
    """Resolve episode runtime from known benchmark provenance fields."""
    for value in (
        record.get("total_runtime_sec"),
        record.get("runtime_sec"),
        record.get("wall_time_sec"),
        record.get("total_runtime"),
    ):
        number = _float_or_none(value)
        if number is not None:
            return number
    return None


def _read_jsonl_files(
    paths: Sequence[Path], *, annotate_source_path: bool = False
) -> list[dict[str, Any]]:
    """Read benchmark episode JSONL files, failing closed on malformed source data.

    Source-path annotations are opt-in because the legacy Parquet export hashes the original
    record surface; adding an internal bookkeeping field there would create unrelated golden
    output drift.
    """
    records: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Benchmark episode JSONL input is not a file: {path}")
        source_sha = _path_sha256(path)
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    record = json.loads(text)
                except json.JSONDecodeError as exc:
                    # Surface the canonical typed input error (a ValueError subclass, so
                    # backward-compatible) so the export-parquet CLI boundary reports it as a
                    # documented non-zero exit instead of a raw traceback. See issue #4988.
                    raise EpisodeRecordInputError(
                        f"{path}:{line_number} is not valid JSON: {exc.msg}"
                    ) from exc
                if not isinstance(record, dict):
                    raise EpisodeRecordInputError(
                        f"{path}:{line_number} must contain a JSON object"
                    )
                if annotate_source_path:
                    record["_source_path"] = str(path)
                    record["_source_sha256"] = source_sha
                records.append(record)
    return records


def _build_metadata(
    *,
    paths: Sequence[Path],
    record_count: int,
    rows: Mapping[str, Sequence[Mapping[str, Any]]],
    table_paths: Mapping[str, Path],
) -> dict[str, Any]:
    """Build export metadata and provenance."""
    return {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "jsonl_is_source_of_truth": True,
        "record_count": record_count,
        "source_files": [
            {
                "path": path.name,
                "sha256": _path_sha256(path) if path.is_file() else None,
            }
            for path in paths
        ],
        "tables": {
            table_name: {
                "file": table_paths[table_name].name,
                "rows": len(table_rows),
            }
            for table_name, table_rows in rows.items()
        },
    }


def _path_sha256(path: Path) -> str:
    """Return a SHA-256 digest for an input file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _duckdb_examples_sql() -> str:
    """Return example DuckDB queries for the exported table layout."""
    return """-- Robot SF benchmark Parquet analytics examples.
-- Run from the export directory, or replace the file paths with absolute paths.

-- Grouped safety metrics by planner and scenario family.
WITH metric_values AS (
    SELECT
        e.algo,
        e.scenario_family,
        m.metric_path,
        m.value_number
    FROM read_parquet('episodes.parquet') AS e
    JOIN read_parquet('metrics.parquet') AS m USING (episode_id)
)
SELECT
    algo,
    scenario_family,
    AVG(CASE WHEN metric_path = 'min_ttc' THEN value_number END) AS avg_min_ttc,
    AVG(CASE WHEN metric_path = 'clearance' THEN value_number END) AS avg_clearance,
    SUM(CASE WHEN metric_path = 'collisions' THEN value_number ELSE 0 END) AS collisions
FROM metric_values
GROUP BY algo, scenario_family
ORDER BY algo, scenario_family;

-- Failure and near-miss mining.
SELECT
    e.episode_id,
    e.algo,
    e.scenario_id,
    e.scenario_family,
    e.termination_reason,
    m.value_number AS min_ttc
FROM read_parquet('episodes.parquet') AS e
LEFT JOIN read_parquet('metrics.parquet') AS m
    ON e.episode_id = m.episode_id AND m.metric_path = 'min_ttc'
WHERE e.termination_reason IN ('collision', 'deadlock', 'timeout')
   OR m.value_number < 0.5
ORDER BY e.algo, min_ttc NULLS LAST;
"""
