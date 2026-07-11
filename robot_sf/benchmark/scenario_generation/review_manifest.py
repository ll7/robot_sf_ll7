"""Validate manual-review manifests for materialized generated replay hypotheses."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import yaml

from robot_sf.benchmark.scenario_generation.catalog_schema import validate_catalog_entry
from robot_sf.benchmark.scenario_generation.catalog_writer import deduplicate_catalog_entries

if TYPE_CHECKING:
    from pathlib import Path

REVIEW_MANIFEST_SCHEMA_VERSION = "generated-scenario-review-manifest.v1"
_CLAIM_BOUNDARY = "generated scenario hypotheses only"
_CHECKLIST_FIELDS = (
    "geometry_valid",
    "route_feasible",
    "pedestrian_density_plausible",
    "critical_window_present",
    "dedup_correct",
)


def validate_review_manifest(catalog_path: Path, review_path: Path) -> dict[str, Any]:
    """Fail closed unless a review manifest covers its whole materialized catalog.

    The checker validates provenance consistency and static review evidence.  A
    ``certified`` verdict is deliberately local to the generated replay packet:
    it never changes the generated entry's benchmark-evidence boundary.

    Returns:
        A compact count and path summary when every catalog entry is review-covered.
    """

    catalog = _load_mapping(catalog_path)
    manifest = _load_mapping(review_path)
    _validate_catalog_header(catalog)
    _validate_manifest_header(manifest)
    catalog_by_id = _entries_by_id(catalog.get("entries"), "catalog")
    review_by_id = _entries_by_id(manifest.get("entries"), "review manifest")
    if set(catalog_by_id) != set(review_by_id):
        raise ValueError("review manifest must cover exactly every catalog entry")

    _validate_deduplication(catalog_by_id, manifest.get("deduplication_distance_threshold"))

    for scenario_id, entry in catalog_by_id.items():
        _validate_review_entry(entry, review_by_id[scenario_id], catalog_path.parent)
    return {"reviewed_count": len(catalog_by_id), "catalog": catalog_path.as_posix()}


def _entries_by_id(raw_entries: object, label: str) -> dict[str, Mapping[str, Any]]:
    if not isinstance(raw_entries, list):
        raise ValueError(f"{label} entries must be a list")
    entries_by_id: dict[str, Mapping[str, Any]] = {}
    for entry in raw_entries:
        if not isinstance(entry, Mapping):
            raise ValueError(f"{label} entries must be mappings")
        if label == "catalog":
            validate_catalog_entry(entry)
        scenario_id = _scenario_id(entry)
        if scenario_id in entries_by_id:
            raise ValueError(f"{label} contains duplicate scenario_id: {scenario_id}")
        entries_by_id[scenario_id] = entry
    return entries_by_id


def _validate_deduplication(
    catalog_by_id: Mapping[str, Mapping[str, Any]], threshold_value: object
) -> None:
    threshold = _finite_nonnegative(threshold_value)
    kept, dropped = deduplicate_catalog_entries(
        list(catalog_by_id.values()), distance_threshold=threshold
    )
    if [entry["scenario_id"] for entry in kept] != sorted(catalog_by_id):
        raise ValueError("catalog entries do not match the declared deduplication threshold")
    if dropped:
        raise ValueError(
            "review catalog must not retain entries that its declared deduplication drops"
        )


def _validate_catalog_header(catalog: Mapping[str, Any]) -> None:
    if catalog.get("schema_version") != "generated-scenario-catalog.v1":
        raise ValueError("catalog.schema_version must be generated-scenario-catalog.v1")
    metadata = catalog.get("metadata")
    if not isinstance(metadata, Mapping) or metadata.get("benchmark_evidence") is not False:
        raise ValueError("catalog must retain benchmark_evidence: false")


def _validate_manifest_header(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema_version") != REVIEW_MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"review manifest schema_version must be {REVIEW_MANIFEST_SCHEMA_VERSION}")
    if manifest.get("claim_boundary") != _CLAIM_BOUNDARY:
        raise ValueError("review manifest must retain the generated-hypothesis claim boundary")


def _validate_review_entry(
    entry: Mapping[str, Any], review: Mapping[str, Any], catalog_dir: Path
) -> None:
    verdict, reason = _review_verdict_and_reason(entry, review)
    _validate_provenance_review(entry, verdict, reason)
    _validate_checklist(entry, review, verdict)
    _validate_materialized_scenario(entry, review, catalog_dir)
    _validate_criticality(entry)


def _review_verdict_and_reason(
    entry: Mapping[str, Any], review: Mapping[str, Any]
) -> tuple[str, str]:
    verdict = review.get("verdict")
    reason = review.get("reason")
    if verdict not in {"certified", "rejected", "needs-fix"}:
        raise ValueError(f"review verdict is invalid for {_scenario_id(entry)}")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError(f"review reason is required for {_scenario_id(entry)}")
    return verdict, reason


def _validate_provenance_review(entry: Mapping[str, Any], verdict: str, reason: str) -> None:
    provenance = entry["provenance"]
    if provenance.get("reviewed") is not True:
        raise ValueError(f"catalog entry is not marked reviewed: {_scenario_id(entry)}")
    if provenance.get("review") != {"verdict": verdict, "reason": reason}:
        raise ValueError(f"catalog provenance review disagrees for {_scenario_id(entry)}")


def _validate_checklist(entry: Mapping[str, Any], review: Mapping[str, Any], verdict: str) -> None:
    checklist = review.get("checklist")
    if not isinstance(checklist, Mapping) or set(checklist) != set(_CHECKLIST_FIELDS):
        raise ValueError(f"review checklist is incomplete for {_scenario_id(entry)}")
    if verdict == "certified" and not all(checklist.values()):
        raise ValueError(f"certified review has an unchecked item: {_scenario_id(entry)}")


def _validate_materialized_scenario(
    entry: Mapping[str, Any], review: Mapping[str, Any], catalog_dir: Path
) -> None:
    candidate_path = review.get("materialized_scenario")
    if not isinstance(candidate_path, str) or not candidate_path.strip():
        raise ValueError(f"materialized scenario path is required for {_scenario_id(entry)}")
    scenario = _load_mapping(catalog_dir / candidate_path)
    scenarios = scenario.get("scenarios")
    if (
        not isinstance(scenarios, list)
        or len(scenarios) != 1
        or not isinstance(scenarios[0], Mapping)
    ):
        raise ValueError(f"materialized scenario payload is invalid for {_scenario_id(entry)}")
    candidate = scenarios[0]
    if candidate.get("name") != _scenario_id(entry):
        raise ValueError(f"materialized scenario id disagrees for {_scenario_id(entry)}")
    metadata = candidate.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError(f"materialized scenario metadata is missing for {_scenario_id(entry)}")
    if (
        metadata.get("required_manual_review") is not True
        or metadata.get("benchmark_evidence") is not False
    ):
        raise ValueError(
            f"materialized scenario lost generated-only governance: {_scenario_id(entry)}"
        )
    replay = metadata.get("generated_replay")
    if not isinstance(replay, Mapping) or replay.get("replay_status") != "replay_validated":
        raise ValueError(f"materialized scenario lacks replay validation: {_scenario_id(entry)}")
    if entry["replay"]["status"] != "replay_validated":
        raise ValueError(f"catalog replay status is not validated: {_scenario_id(entry)}")


def _validate_criticality(entry: Mapping[str, Any]) -> None:
    observed_at = float(entry["criticality"]["observed_at_s"])
    frame = next(
        (
            frame
            for frame in entry["segment"]["trace_frames"]
            if float(frame["time_s"]) == observed_at
        ),
        None,
    )
    if frame is None:
        raise ValueError(f"no trace frame matches observed_at_s: {_scenario_id(entry)}")
    observed = min(
        math.dist(frame["robot"]["position"], pedestrian["position"])
        for pedestrian in frame["pedestrians"]
    )
    recorded = float(entry["criticality"]["source_metrics"]["min_clearance_m"])
    if not math.isclose(observed, recorded, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"criticality metric does not match critical frame: {_scenario_id(entry)}")


def _load_mapping(path: Path) -> Mapping[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a mapping")
    return payload


def _scenario_id(payload: Mapping[str, Any]) -> str:
    value = payload.get("scenario_id")
    if not isinstance(value, str) or not value:
        raise ValueError("scenario_id must be a non-empty string")
    return value


def _finite_nonnegative(value: object) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError("deduplication_distance_threshold must be a finite number")
    if value < 0.0:
        raise ValueError("deduplication_distance_threshold must be >= 0")
    return float(value)


__all__ = ["REVIEW_MANIFEST_SCHEMA_VERSION", "validate_review_manifest"]
