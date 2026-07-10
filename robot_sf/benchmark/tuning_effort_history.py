"""Validate the historical per-arm tuning-effort backfill registry.

The registry is a sidecar for immutable, tracked campaign manifests created before tuning metadata
was emitted directly. It matches each historical arm by its stable ``(key, algo, algo_config_path)``
signature and fails closed on missing, duplicate, malformed, or unused records.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_REGISTRY = Path("configs/benchmarks/tuning_effort_history_v1.yaml")
_SOURCES = {"backfilled", "unknown"}
_TUNING_FIELDS = {
    "parameters_touched",
    "tuning_scenario_ids",
    "eval_set_disjoint",
    "budget_runs",
    "budget_hours",
    "tuned_by",
    "tuned_at_utc",
    "source",
}
logger = logging.getLogger(__name__)


@dataclass(frozen=True, order=True)
class ArmSignature:
    """Stable identity available in both legacy manifests and current configs."""

    key: str
    algo: str
    algo_config_path: str | None


def _mapping(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a mapping")
    return value


def _signature(value: Any, *, context: str) -> ArmSignature:
    raw = _mapping(value, context=context)
    key = raw.get("key")
    algo = raw.get("algo")
    algo_config_path = raw.get("algo_config_path")
    if not isinstance(key, str) or not key.strip():
        raise ValueError(f"{context}.key must be a non-empty string")
    if not isinstance(algo, str) or not algo.strip():
        raise ValueError(f"{context}.algo must be a non-empty string")
    if algo_config_path is not None and (
        not isinstance(algo_config_path, str) or not algo_config_path.strip()
    ):
        raise ValueError(f"{context}.algo_config_path must be null or a non-empty string")
    return ArmSignature(key.strip(), algo.strip(), algo_config_path)


def _validate_tuning_shape(tuning: dict[str, Any], *, context: str) -> str:
    """Validate the manifest-compatible tuning fields.

    Returns:
        The validated tuning source.
    """
    missing_fields = sorted(_TUNING_FIELDS - tuning.keys())
    if missing_fields:
        raise ValueError(f"{context}.tuning is missing explicit fields: {missing_fields}")
    unknown_fields = sorted(tuning.keys() - _TUNING_FIELDS)
    if unknown_fields:
        raise ValueError(f"{context}.tuning has unsupported fields: {unknown_fields}")

    source = tuning.get("source")
    if source not in _SOURCES:
        raise ValueError(f"{context}.tuning.source must be one of {sorted(_SOURCES)}")
    for field in ("parameters_touched", "tuning_scenario_ids"):
        values = tuning.get(field)
        if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
            raise ValueError(f"{context}.tuning.{field} must be a list of strings")
    return source


def _validate_record_evidence(
    record: dict[str, Any],
    tuning: dict[str, Any],
    *,
    source: str,
    context: str,
    repo_root: Path,
) -> None:
    """Require evidence for backfills and explicit emptiness for unknown records."""

    evidence_refs = record.get("evidence_refs")
    if not isinstance(evidence_refs, list) or not all(
        isinstance(item, str) and item.strip() for item in evidence_refs
    ):
        raise ValueError(f"{context}.evidence_refs must be a non-empty list of paths")
    if not evidence_refs:
        raise ValueError(f"{context}.evidence_refs must not be empty")
    missing_refs = [ref for ref in evidence_refs if not (repo_root / ref).is_file()]
    if missing_refs:
        raise ValueError(f"{context}.evidence_refs do not exist: {missing_refs}")

    if source == "backfilled":
        if not tuning["parameters_touched"]:
            raise ValueError(f"{context} backfilled record needs parameters_touched evidence")
        if record.get("unknown_reason") is not None:
            raise ValueError(f"{context} backfilled record must not set unknown_reason")
    else:
        reason = record.get("unknown_reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"{context} unknown record needs a non-empty unknown_reason")
        populated = {
            field: tuning[field]
            for field in _TUNING_FIELDS - {"source"}
            if tuning[field] not in (None, [], "")
        }
        if populated:
            raise ValueError(f"{context} unknown record fabricates populated fields: {populated}")


def _validate_tuning_record(record: dict[str, Any], *, context: str, repo_root: Path) -> None:
    tuning = _mapping(record.get("tuning"), context=f"{context}.tuning")
    source = _validate_tuning_shape(tuning, context=context)
    _validate_record_evidence(record, tuning, source=source, context=context, repo_root=repo_root)


def _load_records(raw_records: Any, *, repo_root: Path) -> dict[ArmSignature, dict[str, Any]]:
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("registry.records must be a non-empty list")
    records: dict[ArmSignature, dict[str, Any]] = {}
    for index, value in enumerate(raw_records):
        record = _mapping(value, context=f"records[{index}]")
        signature = _signature(record.get("planner"), context=f"records[{index}].planner")
        if signature in records:
            raise ValueError(f"duplicate tuning record for {signature}")
        _validate_tuning_record(record, context=f"records[{index}]", repo_root=repo_root)
        records[signature] = record
    return records


def _validate_manifest_coverage(
    manifest_paths: list[Any],
    records: dict[ArmSignature, dict[str, Any]],
    *,
    repo_root: Path,
) -> tuple[set[ArmSignature], dict[str, int], int]:
    used: set[ArmSignature] = set()
    source_counts = dict.fromkeys(sorted(_SOURCES), 0)
    arm_occurrences = 0
    for manifest_rel in manifest_paths:
        if not isinstance(manifest_rel, str) or not manifest_rel.strip():
            raise ValueError("campaign manifest paths must be non-empty strings")
        manifest_path = repo_root / manifest_rel
        if not manifest_path.is_file():
            raise ValueError(f"campaign manifest does not exist: {manifest_rel}")
        manifest = _mapping(
            json.loads(manifest_path.read_text(encoding="utf-8")), context=manifest_rel
        )
        planners = manifest.get("planners")
        if not isinstance(planners, list) or not planners:
            raise ValueError(f"{manifest_rel}.planners must be a non-empty list")
        for index, planner in enumerate(planners):
            signature = _signature(planner, context=f"{manifest_rel}.planners[{index}]")
            record = records.get(signature)
            if record is None:
                raise ValueError(
                    f"missing historical tuning record for {signature} in {manifest_rel}"
                )
            used.add(signature)
            source_counts[record["tuning"]["source"]] += 1
            arm_occurrences += 1
    return used, source_counts, arm_occurrences


def validate_registry(registry_path: Path, *, repo_root: Path) -> dict[str, Any]:
    """Validate registry coverage against every declared historical campaign manifest.

    Returns:
        A deterministic coverage summary suitable for CI logs.
    """
    payload = _mapping(
        yaml.safe_load(registry_path.read_text(encoding="utf-8")), context="registry"
    )
    if payload.get("version") != 1:
        raise ValueError("registry.version must equal 1")

    manifest_paths = payload.get("campaign_manifests")
    if not isinstance(manifest_paths, list) or not manifest_paths:
        raise ValueError("registry.campaign_manifests must be a non-empty list")
    if len(manifest_paths) != len(set(manifest_paths)):
        raise ValueError("registry.campaign_manifests contains duplicate paths")

    records = _load_records(payload.get("records"), repo_root=repo_root)
    used, source_counts, arm_occurrences = _validate_manifest_coverage(
        manifest_paths, records, repo_root=repo_root
    )

    unused = sorted(set(records) - used)
    if unused:
        raise ValueError(f"registry contains unused tuning records: {unused}")
    return {
        "status": "ok",
        "campaign_manifest_count": len(manifest_paths),
        "arm_occurrence_count": arm_occurrences,
        "unique_arm_record_count": len(records),
        "arm_occurrences_by_source": source_counts,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the fail-closed historical tuning registry check.

    Returns:
        Zero when the registry is complete and valid; one otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    try:
        summary = validate_registry(args.registry, repo_root=args.repo_root.resolve())
    except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError) as exc:
        logger.log(
            logging.ERROR,
            json.dumps({"status": "failed", "error": str(exc)}, sort_keys=True),
        )
        return 1
    logger.info(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
