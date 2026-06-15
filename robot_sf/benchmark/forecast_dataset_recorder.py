"""Forecast dataset recorder and split manifest helpers."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from robot_sf.analysis_workbench.simulation_trace_export import (
    SimulationTraceExport,
    load_simulation_trace_export,
)
from robot_sf.benchmark.forecast_observation_adapters import (
    ForecastObservationAdapter,
    OracleFullStateForecastAdapter,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

FORECAST_DATASET_SCHEMA_VERSION = "forecast_dataset.v1"
DEFAULT_FORECAST_DATASET_ID = "forecast_dataset_smoke_v1"
SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class ForecastDatasetRecordResult:
    """Paths and manifest payload produced by a forecast dataset recording run."""

    dataset_path: Path
    manifest_path: Path
    manifest: dict[str, Any]


def record_forecast_dataset_from_trace_exports(
    trace_paths: Sequence[Path | str],
    output_dir: Path | str,
    *,
    adapter: ForecastObservationAdapter | None = None,
    feature_schema: dict[str, Any],
    horizons_s: Sequence[float] = (0.5, 1.0),
    dataset_id: str = DEFAULT_FORECAST_DATASET_ID,
) -> ForecastDatasetRecordResult:
    """Record forecast examples and a split manifest from trace exports.

    Returns:
        Paths and manifest payload for the generated dataset.
    """

    paths = [Path(path) for path in trace_paths]
    if not paths:
        raise ValueError("trace_paths must be non-empty")
    schema = _require_feature_schema(feature_schema)
    horizons = _validate_horizons(horizons_s)
    observation_adapter = adapter or OracleFullStateForecastAdapter()
    traces = [load_simulation_trace_export(path) for path in paths]
    splits = _assign_trace_splits(traces)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dataset_path = output / f"{dataset_id}.jsonl"
    rows = _build_dataset_rows(
        traces,
        splits=splits,
        adapter=observation_adapter,
        feature_schema=schema,
        horizons_s=horizons,
    )
    if not rows:
        raise ValueError("no forecast dataset rows were produced")
    _write_jsonl(dataset_path, rows)

    manifest = _build_manifest(
        traces=traces,
        rows=rows,
        splits=splits,
        dataset_id=dataset_id,
        dataset_path=dataset_path,
        output_dir=output,
        adapter=observation_adapter,
        feature_schema=schema,
        horizons_s=horizons,
    )
    validate_forecast_dataset_manifest(manifest)
    manifest_path = output / f"{dataset_id}.manifest.json"
    _atomic_write_json(manifest_path, manifest)
    return ForecastDatasetRecordResult(
        dataset_path=dataset_path,
        manifest_path=manifest_path,
        manifest=manifest,
    )


def validate_forecast_dataset_manifest(payload: dict[str, Any]) -> None:
    """Validate manifest structure and split leakage constraints."""

    if payload.get("schema_version") != FORECAST_DATASET_SCHEMA_VERSION:
        raise ValueError("schema_version must be forecast_dataset.v1")
    if not str(payload.get("dataset_id", "")).strip():
        raise ValueError("dataset_id is required")
    if int(payload.get("example_count", 0)) <= 0:
        raise ValueError("example_count must be positive")
    splits = payload.get("splits")
    if not isinstance(splits, dict):
        raise ValueError("splits must be a mapping")
    missing_splits = set(SPLIT_NAMES) - set(splits)
    if missing_splits:
        raise ValueError(f"splits missing required keys: {sorted(missing_splits)}")
    _validate_disjoint_split_values(splits, "scenario_ids")
    _validate_disjoint_split_values(splits, "scenario_seed_keys")
    examples_path = payload.get("examples_path")
    if not isinstance(examples_path, str) or not examples_path.strip():
        raise ValueError("examples_path is required")
    feature_schema = payload.get("feature_schema")
    if not isinstance(feature_schema, dict) or not feature_schema:
        raise ValueError("feature_schema is required")


def _build_dataset_rows(
    traces: Sequence[SimulationTraceExport],
    *,
    splits: dict[str, str],
    adapter: ForecastObservationAdapter,
    feature_schema: dict[str, Any],
    horizons_s: list[float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trace in traces:
        trace_dict = _trace_dict_for_adapter(trace)
        dt_s = _trace_dt_s(trace)
        for frame_index, frame in enumerate(trace.frames):
            try:
                observed = adapter.adapt_trace(
                    trace_dict,
                    feature_schema=feature_schema,
                    horizons_s=horizons_s,
                    dt_s=dt_s,
                    step_index=frame_index,
                )
            except ValueError as exc:
                if "actor_ids must be non-empty" in str(exc):
                    continue
                raise
            for actor in observed.actors:
                future_positions = _future_positions(
                    trace,
                    actor_id=actor.actor_id,
                    start_index=frame_index,
                    horizons_s=horizons_s,
                    dt_s=dt_s,
                )
                if future_positions is None:
                    continue
                rows.append(
                    {
                        "schema_version": "forecast_dataset_row.v1",
                        "dataset_role": "supervised_forecast_example",
                        "trace_id": trace.trace_id,
                        "episode_id": trace.source.episode_id,
                        "scenario_id": trace.source.scenario_id,
                        "map_id": trace.source.scenario_id,
                        "map_id_source": "simulation_trace_export.source.scenario_id",
                        "seed": trace.source.seed,
                        "planner_id": trace.source.planner_id,
                        "split": splits[trace.trace_id],
                        "frame_step": frame.step,
                        "frame_time_s": frame.time_s,
                        "actor_id": actor.actor_id,
                        "actor_type": actor.state.actor_type,
                        "observation_tier": observed.provenance.observation_tier,
                        "oracle_state": observed.provenance.oracle_state,
                        "feature_schema": feature_schema,
                        "dt_s": observed.provenance.dt_s,
                        "horizons_s": list(observed.provenance.horizons_s),
                        "input": {
                            "position_m": actor.state.position.astype(float).tolist(),
                            "velocity_mps": actor.state.velocity.astype(float).tolist(),
                        },
                        "label": {
                            "future_positions_m": future_positions,
                            "source": "simulation_trace_export.pedestrians",
                        },
                    }
                )
    return rows


def _build_manifest(  # noqa: PLR0913
    *,
    traces: Sequence[SimulationTraceExport],
    rows: Sequence[dict[str, Any]],
    splits: dict[str, str],
    dataset_id: str,
    dataset_path: Path,
    output_dir: Path,
    adapter: ForecastObservationAdapter,
    feature_schema: dict[str, Any],
    horizons_s: list[float],
) -> dict[str, Any]:
    rows_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    traces_by_split: dict[str, list[SimulationTraceExport]] = defaultdict(list)
    for row in rows:
        rows_by_split[str(row["split"])].append(row)
    for trace in traces:
        traces_by_split[splits[trace.trace_id]].append(trace)

    return {
        "schema_version": FORECAST_DATASET_SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "examples_path": str(dataset_path.relative_to(output_dir)),
        "example_count": len(rows),
        "observation_tiers": [adapter.observation_tier],
        "oracle_state": adapter.oracle_state,
        "feature_schema": feature_schema,
        "horizons_s": horizons_s,
        "split_policy": {
            "strategy": "deterministic_trace_order",
            "leakage_prevention": ["scenario_ids", "scenario_seed_keys"],
            "split_names": list(SPLIT_NAMES),
        },
        "source_traces": [_trace_manifest_entry(trace, splits[trace.trace_id]) for trace in traces],
        "splits": {
            split: _split_manifest_entry(rows_by_split[split], traces_by_split[split])
            for split in SPLIT_NAMES
        },
    }


def _assign_trace_splits(traces: Sequence[SimulationTraceExport]) -> dict[str, str]:
    """Assign traces to deterministic train/validation/test splits.

    Returns:
        Mapping from trace id to split name.
    """

    ordered = sorted(traces, key=lambda trace: trace.trace_id)
    if len({trace.trace_id for trace in ordered}) != len(ordered):
        raise ValueError("trace_id values must be unique")
    if len(ordered) == 1:
        split_sequence = ["train"]
    elif len(ordered) == 2:
        split_sequence = ["train", "validation"]
    else:
        split_sequence = ["train"] * (len(ordered) - 2) + ["validation", "test"]
    return {trace.trace_id: split for trace, split in zip(ordered, split_sequence, strict=True)}


def _split_manifest_entry(
    rows: Sequence[dict[str, Any]],
    traces: Sequence[SimulationTraceExport],
) -> dict[str, Any]:
    scenario_ids = sorted({trace.source.scenario_id for trace in traces})
    seeds = sorted({trace.source.seed for trace in traces})
    scenario_seed_keys = sorted(
        f"{trace.source.scenario_id}:{trace.source.seed}" for trace in traces
    )
    return {
        "example_count": len(rows),
        "trace_count": len(traces),
        "trace_ids": sorted(trace.trace_id for trace in traces),
        "scenario_ids": scenario_ids,
        "seeds": seeds,
        "scenario_seed_keys": scenario_seed_keys,
    }


def _trace_manifest_entry(trace: SimulationTraceExport, split: str) -> dict[str, Any]:
    return {
        "trace_id": trace.trace_id,
        "split": split,
        "scenario_id": trace.source.scenario_id,
        "map_id": trace.source.scenario_id,
        "map_id_source": "simulation_trace_export.source.scenario_id",
        "seed": trace.source.seed,
        "planner_id": trace.source.planner_id,
        "episode_id": trace.source.episode_id,
        "frame_count": len(trace.frames),
    }


def _validate_disjoint_split_values(splits: dict[str, Any], field: str) -> None:
    seen: dict[str, str] = {}
    for split in SPLIT_NAMES:
        values = splits.get(split, {}).get(field, [])
        if not isinstance(values, list):
            raise ValueError(f"splits.{split}.{field} must be a list")
        for value in values:
            key = str(value)
            previous = seen.get(key)
            if previous is not None and previous != split:
                raise ValueError(f"{field} leakage across splits: {key}")
            seen[key] = split


def _trace_dict_for_adapter(trace: SimulationTraceExport) -> dict[str, Any]:
    return {
        "scenario_id": trace.source.scenario_id,
        "seed": trace.source.seed,
        "frames": [asdict(frame) for frame in trace.frames],
    }


def _future_positions(
    trace: SimulationTraceExport,
    *,
    actor_id: str,
    start_index: int,
    horizons_s: Sequence[float],
    dt_s: float,
) -> list[list[float]] | None:
    positions: list[list[float]] = []
    for horizon_s in horizons_s:
        target_index = start_index + round(float(horizon_s) / dt_s)
        if target_index >= len(trace.frames):
            return None
        target = _pedestrian_by_id(trace.frames[target_index].pedestrians, actor_id)
        if target is None:
            return None
        positions.append(np.asarray(target["position"], dtype=float).tolist())
    return positions


def _pedestrian_by_id(
    pedestrians: Iterable[dict[str, Any]], actor_id: str
) -> dict[str, Any] | None:
    for pedestrian in pedestrians:
        if str(pedestrian.get("id")) == actor_id or str(pedestrian.get("actor_id")) == actor_id:
            return pedestrian
    return None


def _trace_dt_s(trace: SimulationTraceExport) -> float:
    if len(trace.frames) < 2:
        return 0.1
    dt_s = trace.frames[1].time_s - trace.frames[0].time_s
    return float(dt_s) if dt_s > 0.0 else 0.1


def _require_feature_schema(feature_schema: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(feature_schema, dict) or not feature_schema:
        raise ValueError("feature_schema is required")
    return {str(key): value for key, value in feature_schema.items()}


def _validate_horizons(horizons_s: Sequence[float]) -> list[float]:
    horizons = [float(value) for value in horizons_s]
    if not horizons:
        raise ValueError("horizons_s must be non-empty")
    if any(value <= 0.0 for value in horizons):
        raise ValueError("horizons_s values must be positive")
    if any(later <= earlier for earlier, later in pairwise(horizons)):
        raise ValueError("horizons_s values must be strictly increasing")
    return horizons


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


__all__ = [
    "DEFAULT_FORECAST_DATASET_ID",
    "FORECAST_DATASET_SCHEMA_VERSION",
    "ForecastDatasetRecordResult",
    "record_forecast_dataset_from_trace_exports",
    "validate_forecast_dataset_manifest",
]
