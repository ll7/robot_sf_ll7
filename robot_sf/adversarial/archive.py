"""Curate compact adversarial failure archives from search manifests."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ARCHIVE_SCHEMA_VERSION = "adversarial_failure_archive.v1"
SEARCH_MANIFEST_SCHEMA_VERSION = "adversarial-search-manifest.v1"

_SCALAR_FIELDS = (
    ("start_x", ("start", "x")),
    ("start_y", ("start", "y")),
    ("goal_x", ("goal", "x")),
    ("goal_y", ("goal", "y")),
    ("spawn_time_s", ("spawn_time_s",)),
    ("pedestrian_speed_mps", ("pedestrian_speed_mps",)),
    ("pedestrian_delay_s", ("pedestrian_delay_s",)),
)

_EXCLUDED_PRIMARY_FAILURES = frozenset({"", "success", "invalid_candidate", "simulation_error"})


def _build_selection_description() -> str:
    """Return the selection metadata string derived from the filtering constants."""
    excluded = _EXCLUDED_PRIMARY_FAILURES - {""}
    excluded_list = ", ".join(sorted(excluded))
    return (
        "not null, not empty, status is not not_evaluated, "
        f"and primary_failure is not one of {excluded_list}"
    )


def curate_failure_archive(
    manifest_paths: list[str | Path],
    *,
    output_path: str | Path,
) -> dict[str, Any]:
    """Curate a compact deterministic failure archive from search manifests.

    Returns:
        Archive payload written to ``output_path``.
    """
    manifests = sorted(Path(path) for path in manifest_paths)
    source_payloads = [_load_search_manifest(path) for path in manifests]
    entries: list[dict[str, Any]] = []
    source_candidate_count = 0

    for manifest_path, manifest in zip(manifests, source_payloads, strict=True):
        candidates = manifest.get("candidates") or []
        if not isinstance(candidates, list):
            continue
        source_candidate_count += len(candidates)
        for candidate_index, candidate_payload in enumerate(candidates):
            if not isinstance(candidate_payload, dict):
                continue
            if not _is_archivable_failure(candidate_payload):
                continue
            entries.append(
                _archive_entry(
                    manifest_path=manifest_path,
                    manifest=manifest,
                    candidate_index=candidate_index,
                    candidate_payload=candidate_payload,
                    archive_index=len(entries),
                )
            )

    clusters = _cluster_entries(entries)
    payload = {
        "schema_version": ARCHIVE_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "config": {
            "source_manifests": [path.as_posix() for path in manifests],
            "selection": {
                "failure_attribution.primary_failure": _build_selection_description(),
            },
            "grouping": [
                "config.policy",
                "config.scenario_template",
                "failure_attribution.primary_failure",
                "failure_attribution.details.termination_reason",
            ],
            "representative": "smallest normalized perturbation, then highest objective",
        },
        "summary": {
            "source_manifest_count": len(manifests),
            "source_candidate_count": source_candidate_count,
            "archived_failure_count": len(entries),
            "cluster_count": len(clusters),
        },
        "entries": entries,
        "clusters": clusters,
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def failure_archive_feature_rows(archive: dict[str, Any]) -> list[dict[str, Any]]:
    """Return deterministic tabular metadata rows from a failure archive.

    The rows intentionally contain compact archive metadata and scalar candidate
    fields only. They are suitable for fixture-based proposal-model plumbing and
    index/export checks, not for training claims on raw traces.
    """
    rows: list[dict[str, Any]] = []
    for entry in _archive_entries(archive):
        candidate = entry.get("candidate") if isinstance(entry.get("candidate"), dict) else {}
        start = candidate.get("start") if isinstance(candidate.get("start"), dict) else {}
        goal = candidate.get("goal") if isinstance(candidate.get("goal"), dict) else {}
        attribution = (
            entry.get("failure_attribution")
            if isinstance(entry.get("failure_attribution"), dict)
            else {}
        )
        details = attribution.get("details") if isinstance(attribution.get("details"), dict) else {}
        cluster_key = entry.get("cluster_key") if isinstance(entry.get("cluster_key"), dict) else {}

        rows.append(
            {
                "archive_id": str(entry.get("archive_id", "")),
                "cluster_key": _stable_json(cluster_key),
                "source_manifest": _as_optional_str(entry.get("source_manifest")),
                "source_candidate_index": entry.get("source_candidate_index"),
                "bundle_path": _as_optional_str(entry.get("bundle_path")),
                "scenario_yaml_path": _as_optional_str(entry.get("scenario_yaml_path")),
                "start_x": _finite_float(start.get("x")),
                "start_y": _finite_float(start.get("y")),
                "goal_x": _finite_float(goal.get("x")),
                "goal_y": _finite_float(goal.get("y")),
                "spawn_time_s": _finite_float(candidate.get("spawn_time_s")),
                "pedestrian_speed_mps": _finite_float(candidate.get("pedestrian_speed_mps")),
                "pedestrian_delay_s": _finite_float(candidate.get("pedestrian_delay_s")),
                "scenario_seed": _finite_float(candidate.get("scenario_seed")),
                "objective_value": _finite_float(entry.get("objective_value")),
                "primary_failure": _as_optional_str(attribution.get("primary_failure")),
                "termination_reason": _as_optional_str(details.get("termination_reason")),
                "normalized_perturbation": _finite_float(entry.get("normalized_perturbation")),
                "replay_command": _as_optional_str(entry.get("replay_command")),
            }
        )
    return sorted(rows, key=lambda row: row["archive_id"])


def failure_archive_index(archive: dict[str, Any]) -> dict[str, Any]:
    """Build stable lookup indexes over archive feature rows."""
    rows = failure_archive_feature_rows(archive)
    by_archive_id: dict[str, dict[str, Any]] = {}
    by_cluster_key: dict[str, list[str]] = defaultdict(list)
    by_primary_failure: dict[str, list[str]] = defaultdict(list)
    by_scenario_seed: dict[str, list[str]] = defaultdict(list)

    for row in rows:
        archive_id = row["archive_id"]
        by_archive_id[archive_id] = row
        by_cluster_key[str(row["cluster_key"])].append(archive_id)
        by_primary_failure[str(row["primary_failure"])].append(archive_id)
        seed = row.get("scenario_seed")
        if seed is not None:
            by_scenario_seed[_stable_number_key(seed)].append(archive_id)

    return {
        "schema_version": "adversarial_failure_archive_index.v1",
        "row_count": len(rows),
        "by_archive_id": by_archive_id,
        "by_cluster_key": _sorted_list_index(by_cluster_key),
        "by_primary_failure": _sorted_list_index(by_primary_failure),
        "by_scenario_seed": _sorted_list_index(by_scenario_seed),
    }


def _load_search_manifest(path: Path) -> dict[str, Any]:
    """Load one adversarial search manifest and validate the schema tag."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Search manifest must be a JSON object: {path}")
    schema = payload.get("schema_version")
    if schema != SEARCH_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported search manifest schema for {path}: {schema!r}; "
            f"expected {SEARCH_MANIFEST_SCHEMA_VERSION!r}"
        )
    return payload


def _is_archivable_failure(candidate_payload: dict[str, Any]) -> bool:
    """Return whether a candidate should enter the compact failure archive."""
    attribution = candidate_payload.get("failure_attribution")
    if not isinstance(attribution, dict):
        return False
    primary = attribution.get("primary_failure")
    if primary is None:
        return False
    if str(attribution.get("status", "")).strip().lower() == "not_evaluated":
        return False
    return str(primary).strip().lower() not in _EXCLUDED_PRIMARY_FAILURES


def _archive_entry(
    *,
    manifest_path: Path,
    manifest: dict[str, Any],
    candidate_index: int,
    candidate_payload: dict[str, Any],
    archive_index: int,
) -> dict[str, Any]:
    """Build one compact archive entry."""
    config = manifest.get("config") if isinstance(manifest.get("config"), dict) else {}
    attribution = candidate_payload["failure_attribution"]
    bundle_path = _as_optional_str(candidate_payload.get("bundle_path"))
    scenario_yaml_path = _as_optional_str(candidate_payload.get("scenario_yaml_path"))
    entry = {
        "archive_id": f"failure_{archive_index:04d}",
        "source_manifest": manifest_path.as_posix(),
        "source_candidate_index": int(candidate_index),
        "bundle_path": bundle_path,
        "scenario_yaml_path": scenario_yaml_path,
        "trajectory_csv_path": _as_optional_str(candidate_payload.get("trajectory_csv_path")),
        "episode_record_path": _as_optional_str(candidate_payload.get("episode_record_path")),
        "objective_value": candidate_payload.get("objective_value"),
        "candidate": candidate_payload.get("candidate") or {},
        "failure_attribution": attribution,
        "cluster_key": _cluster_key(manifest, candidate_payload),
        "normalized_perturbation": _normalized_perturbation(
            candidate_payload.get("candidate") or {},
            config.get("search_space") if isinstance(config, dict) else {},
        ),
        "replay_command": _replay_command(config, scenario_yaml_path, bundle_path),
    }
    return entry


def _cluster_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group entries and choose conservative representatives."""
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        grouped[tuple(entry["cluster_key"].values())].append(entry)

    clusters: list[dict[str, Any]] = []
    for cluster_index, (_key_tuple, members) in enumerate(
        sorted(grouped.items(), key=lambda item: item[0])
    ):
        members_sorted = sorted(members, key=lambda item: item["archive_id"])
        representative = min(
            members_sorted,
            key=lambda item: (
                item["normalized_perturbation"],
                -_objective_sort_value(item.get("objective_value")),
                item["archive_id"],
            ),
        )
        key = dict(members_sorted[0]["cluster_key"])
        clusters.append(
            {
                "cluster_id": f"cluster_{cluster_index:04d}",
                "mechanism": {
                    "primary_failure": key["primary_failure"],
                    "termination_reason": key["termination_reason"],
                },
                "cluster_key": key,
                "member_count": len(members_sorted),
                "representative_archive_id": representative["archive_id"],
                "member_archive_ids": [item["archive_id"] for item in members_sorted],
            }
        )
    return clusters


def _cluster_key(manifest: dict[str, Any], candidate_payload: dict[str, Any]) -> dict[str, str]:
    """Return a deterministic cluster key for one candidate."""
    config = manifest.get("config") if isinstance(manifest.get("config"), dict) else {}
    attribution = candidate_payload["failure_attribution"]
    details = attribution.get("details") if isinstance(attribution.get("details"), dict) else {}
    termination = details.get("termination_reason") or "unknown"
    return {
        "policy": str(config.get("policy", "")),
        "scenario_template": str(config.get("scenario_template", "")),
        "primary_failure": str(attribution.get("primary_failure", "unknown")),
        "termination_reason": str(termination),
    }


def _normalized_perturbation(candidate: dict[str, Any], search_space: Any) -> float:
    """Compute normalized distance from each configured range midpoint."""
    variables = search_space.get("variables") if isinstance(search_space, dict) else {}
    if not isinstance(variables, dict):
        return 0.0
    total = 0.0
    for variable_name, candidate_path in _SCALAR_FIELDS:
        bounds = variables.get(variable_name)
        if not isinstance(bounds, dict):
            continue
        value = _nested_float(candidate, candidate_path)
        if value is None:
            continue
        low = _finite_float(bounds.get("min"))
        high = _finite_float(bounds.get("max"))
        if low is None or high is None:
            continue
        span = high - low
        if span <= 0.0:
            continue
        midpoint = (low + high) / 2.0
        total += abs(value - midpoint) / span
    return round(total, 6)


def _replay_command(
    config: dict[str, Any],
    scenario_yaml_path: str | None,
    bundle_path: str | None,
) -> str:
    """Build a replay command pointer without copying raw artifacts."""
    scenario = scenario_yaml_path or "<scenario.yaml>"
    out_path = (
        f"{bundle_path.rstrip('/')}/episode_records_replay.jsonl"
        if bundle_path
        else "<bundle>/episode_records_replay.jsonl"
    )
    policy = str(config.get("policy", "goal"))
    return (
        f"uv run robot_sf_bench run --matrix {scenario} --out {out_path} --algo {policy} --no-video"
    )


def _nested_float(payload: dict[str, Any], path: tuple[str, ...]) -> float | None:
    """Read a nested float from a dictionary."""
    current: Any = payload
    for part in path:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return _finite_float(current)


def _finite_float(value: Any) -> float | None:
    """Return a finite float or None."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _objective_sort_value(value: Any) -> float:
    """Return objective value for ordering, treating missing scores as worst."""
    parsed = _finite_float(value)
    if parsed is None:
        return float("-inf")
    return parsed


def _as_optional_str(value: Any) -> str | None:
    """Return value as string, preserving None."""
    if value is None:
        return None
    return str(value)


def _archive_entries(archive: dict[str, Any]) -> list[dict[str, Any]]:
    """Return archive entries, raising for malformed archive payloads."""
    if not isinstance(archive, dict):
        raise ValueError("Failure archive must be a JSON object.")
    schema = archive.get("schema_version")
    if schema != ARCHIVE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported failure archive schema {schema!r}; expected {ARCHIVE_SCHEMA_VERSION!r}"
        )
    entries = archive.get("entries")
    if not isinstance(entries, list):
        raise ValueError("Failure archive entries must be a list.")
    if not all(isinstance(entry, dict) for entry in entries):
        raise ValueError("Failure archive entries must be JSON objects.")
    return entries


def _stable_json(value: Any) -> str:
    """Return deterministic compact JSON string for index keys."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _stable_number_key(value: float) -> str:
    """Return stable key for numeric index values."""
    if float(value).is_integer():
        return str(int(value))
    return str(value)


def _sorted_list_index(index: dict[str, list[str]]) -> dict[str, list[str]]:
    """Return deterministic index lists sorted by key and archive id."""
    return {key: sorted(values) for key, values in sorted(index.items())}


__all__ = [
    "ARCHIVE_SCHEMA_VERSION",
    "curate_failure_archive",
    "failure_archive_feature_rows",
    "failure_archive_index",
]
