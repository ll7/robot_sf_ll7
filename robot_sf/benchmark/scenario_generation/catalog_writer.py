"""Aggregate and deterministically deduplicate generated scenario hypotheses."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from robot_sf.benchmark.scenario_generation.catalog_schema import validate_catalog_entry

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

CATALOG_SCHEMA_VERSION = "generated-scenario-catalog.v1"
DEDUP_SCHEMA_VERSION = "generated-scenario-dedup.v1"


def deduplicate_catalog_entries(
    entries: Sequence[Mapping[str, Any]],
    *,
    distance_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep the most critical exemplar in each nearby deterministic feature group.

    Grouping uses source map, criticality signal, and actor count.  Within a
    group, the feature distance combines closest-approach location and robot
    route orientation.  For minimum clearance, a smaller value is more critical.

    Returns:
        Kept entries and dropped-duplicate provenance records.
    """

    if not isinstance(distance_threshold, int | float) or isinstance(distance_threshold, bool):
        raise ValueError("distance_threshold must be a finite number")
    distance_threshold = float(distance_threshold)
    if not math.isfinite(distance_threshold) or distance_threshold < 0.0:
        raise ValueError("distance_threshold must be finite and >= 0")

    ordered = sorted((deepcopy(dict(entry)) for entry in entries), key=_stable_order_key)
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for candidate in ordered:
        validate_catalog_entry(candidate)
        duplicate_index = next(
            (
                index
                for index, exemplar in enumerate(kept)
                if _dedup_group(candidate) == _dedup_group(exemplar)
                and _feature_distance(candidate, exemplar) <= distance_threshold
            ),
            None,
        )
        if duplicate_index is None:
            kept.append(candidate)
            continue

        exemplar = kept[duplicate_index]
        if _criticality_rank(candidate) < _criticality_rank(exemplar):
            kept[duplicate_index] = candidate
            dropped.append(_dropped_record(exemplar, candidate, distance_threshold))
        else:
            dropped.append(_dropped_record(candidate, exemplar, distance_threshold))

    kept.sort(key=lambda entry: entry["scenario_id"])
    dropped.sort(key=lambda record: record["dropped_scenario_id"])
    return kept, dropped


def write_generated_catalog(
    output_root: Path,
    entries: Sequence[Mapping[str, Any]],
    *,
    dropped_duplicates: Sequence[Mapping[str, Any]],
    run_manifest_path: Path,
) -> tuple[Path, Path]:
    """Write a generated-only catalog and one-entry-per-row provenance sidecar.

    Returns:
        Paths to the YAML catalog and JSON provenance sidecar.
    """

    output_root.mkdir(parents=True, exist_ok=True)
    normalized = [deepcopy(dict(entry)) for entry in entries]
    for entry in normalized:
        validate_catalog_entry(entry)
    catalog_path = output_root / "generated_catalog.yaml"
    provenance_path = output_root / "generated_catalog.provenance.json"
    catalog = {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "metadata": {
            "source": "auto_generated",
            "required_manual_review": True,
            "benchmark_evidence": False,
            "claim_boundary": "generated scenario hypotheses only",
            "release_matrix_inclusion": False,
        },
        "entries": normalized,
        "deduplication": {
            "schema_version": DEDUP_SCHEMA_VERSION,
            "dropped_count": len(dropped_duplicates),
            "dropped": [dict(record) for record in dropped_duplicates],
        },
    }
    catalog_path.write_text(yaml.safe_dump(catalog, sort_keys=True), encoding="utf-8")
    provenance = {
        "schema_version": "generated-scenario-catalog-provenance.v1",
        "claim_boundary": "generated scenario hypotheses only",
        "run_manifest": run_manifest_path.as_posix(),
        "entries": [
            {
                "scenario_id": entry["scenario_id"],
                "source_episode_id": entry["source_episode"]["episode_id"],
                "source_seed": entry["source_episode"]["source_seed"],
                "source_map": entry["source_episode"]["source_map"],
                "replay_status": entry["replay"]["status"],
            }
            for entry in normalized
        ],
    }
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return catalog_path, provenance_path


def _stable_order_key(entry: Mapping[str, Any]) -> tuple[float, str]:
    return (_criticality_rank(entry), str(entry.get("scenario_id", "")))


def _criticality_rank(entry: Mapping[str, Any]) -> float:
    return float(entry["criticality"]["source_metrics"]["min_clearance_m"])


def _critical_frame(entry: Mapping[str, Any]) -> Mapping[str, Any]:
    observed_at = float(entry["criticality"]["observed_at_s"])
    return min(
        entry["segment"]["trace_frames"],
        key=lambda frame: abs(float(frame["time_s"]) - observed_at),
    )


def _dedup_group(entry: Mapping[str, Any]) -> tuple[str, str, int]:
    frame = _critical_frame(entry)
    return (
        Path(entry["source_episode"]["source_map"]).stem,
        str(entry["criticality"]["signal"]),
        len(frame["pedestrians"]),
    )


def _feature_vector(entry: Mapping[str, Any]) -> tuple[float, float, float, float]:
    frames = entry["segment"]["trace_frames"]
    critical = _critical_frame(entry)
    x, y = (float(value) for value in critical["robot"]["position"])
    start = frames[0]["robot"]["position"]
    end = frames[-1]["robot"]["position"]
    dx = float(end[0]) - float(start[0])
    dy = float(end[1]) - float(start[1])
    norm = math.hypot(dx, dy)
    if norm == 0.0:
        return x, y, 0.0, 0.0
    return x, y, dx / norm, dy / norm


def _feature_distance(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    left_feature = _feature_vector(left)
    right_feature = _feature_vector(right)
    return math.dist(left_feature, right_feature)


def _dropped_record(
    dropped: Mapping[str, Any],
    kept: Mapping[str, Any],
    threshold: float,
) -> dict[str, Any]:
    return {
        "dropped_scenario_id": dropped["scenario_id"],
        "kept_scenario_id": kept["scenario_id"],
        "reason": "near_duplicate_lower_criticality",
        "feature_distance": _feature_distance(dropped, kept),
        "distance_threshold": threshold,
        "group": {
            "source_map_family": _dedup_group(dropped)[0],
            "criticality_signal": _dedup_group(dropped)[1],
            "actor_count": _dedup_group(dropped)[2],
        },
    }


__all__ = [
    "CATALOG_SCHEMA_VERSION",
    "DEDUP_SCHEMA_VERSION",
    "deduplicate_catalog_entries",
    "write_generated_catalog",
]
