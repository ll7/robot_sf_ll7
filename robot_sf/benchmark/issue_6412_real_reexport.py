"""Assemble and verify the real issue #6412 visualization-only package.

The package is a bounded bridge between the real #6411 binding receipt and the
existing #5756 resolver/figure path.  It keeps normalized traces in a local
disposable package and emits only compact, relative receipts that can be
stored with the dissertation evidence context.  It does not promote release
statistics, alter episode outcomes, or replace the frozen release bundle.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from robot_sf.benchmark.candidate_trace_resolution import (
    ISSUE_5756_MAPPING_SCHEMA_VERSION,
    ISSUE_5756_PINNED_PROVENANCE,
    WORKED_EXAMPLE_OUTCOMES,
    load_episode_mapping,
)
from robot_sf.benchmark.trace_reexport_packaging import (
    REAL_REEXPORT_ARMS,
    REAL_REEXPORT_BINDING_SCHEMA,
    REAL_REEXPORT_EXCEPTION_SEEDS,
)

PACKAGE_SCHEMA_VERSION = "issue_6412_real_reexport_package.v1"
EXPECTED_OUTCOMES_SCHEMA_VERSION = "issue_6412_expected_outcomes.v1"
EXCLUSION_SCHEMA_VERSION = "issue_6412_exclusion.v1"
SOURCE_POINTER_SCHEMA_VERSION = "issue_6412_source_pointer.v1"
PACKAGE_REPORT_SCHEMA_VERSION = "issue_6412_package_report.v1"
FIGURE_QA_SCHEMA_VERSION = "issue_6412_figure_qa.v1"
RESOLUTION_SUMMARY_SCHEMA_VERSION = "issue_6412_resolution_summary.v1"
PACKAGE_COMPLETE_SCHEMA_VERSION = "issue_6412_package_complete.v1"

_CHECKSUMS_NAME = "SHA256SUMS"
_COMPLETE_NAME = "package_complete.json"
_EXCLUDED_TUPLES = frozenset(
    ("ppo", "classic_doorway_medium", seed) for seed in REAL_REEXPORT_EXCEPTION_SEEDS
)


class RealReexportPackageError(ValueError):
    """Raised when the #6412 package cannot be proven complete."""


def _canonical_bytes(payload: Any, *, newline: bool = False) -> bytes:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )
    return data + (b"\n" if newline else b"")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    return _sha256_bytes(_canonical_bytes(payload))


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RealReexportPackageError(f"{label} is not readable JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise RealReexportPackageError(f"{label} must be a JSON object")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(payload, newline=True))


def _canonical_outcome(value: Any) -> str | None:
    if isinstance(value, Mapping):
        for key in ("collision_event", "timeout_event", "route_complete", "success"):
            if value.get(key) is True:
                return key
        return None
    text = str(value).strip() if value is not None else ""
    return text if text in WORKED_EXAMPLE_OUTCOMES else None


def _tuple_key(row: Mapping[str, Any]) -> tuple[str, str, int]:
    try:
        planner = str(row.get("planner", row.get("algo"))).strip()
        scenario_id = str(row["scenario_id"]).strip()
        seed = int(row["seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RealReexportPackageError(
            f"row lacks planner/scenario/seed identity: {row!r}"
        ) from exc
    if not planner or not scenario_id:
        raise RealReexportPackageError(f"row has blank planner/scenario identity: {row!r}")
    return planner, scenario_id, seed


def _load_request_tuples(path: Path) -> tuple[set[tuple[str, str, int]], str]:
    payload = _read_json(path, "request manifest")
    if payload.get("schema_version") != "issue_5446_trace_reexport_list.v1":
        raise RealReexportPackageError("request manifest schema mismatch")
    rows = payload.get("tuples")
    if not isinstance(rows, list) or payload.get("n_tuples") != 90 or len(rows) != 90:
        raise RealReexportPackageError("request manifest must contain exactly 90 tuples")
    tuples = {_tuple_key(row) for row in rows if isinstance(row, Mapping)}
    if len(tuples) != 90:
        raise RealReexportPackageError("request manifest contains invalid or duplicate tuples")
    return tuples, _sha256_file(path)


def _load_expected_outcomes(path: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    payload = _read_json(path, "expected outcomes")
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != 90:
        raise RealReexportPackageError("expected outcomes must contain exactly 90 rows")
    indexed: dict[tuple[str, str, int], dict[str, Any]] = {}
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise RealReexportPackageError("expected outcome row must be an object")
        key = _tuple_key(raw)
        outcome = _canonical_outcome(raw.get("outcome"))
        episode_id = str(raw.get("episode_id", "")).strip()
        if outcome is None or not episode_id:
            raise RealReexportPackageError(
                f"expected outcome row lacks canonical outcome/id: {key}"
            )
        if key in indexed:
            raise RealReexportPackageError(f"duplicate expected outcome tuple: {key}")
        indexed[key] = {
            "planner": key[0],
            "scenario_id": key[1],
            "seed": key[2],
            "release_episode_id": episode_id,
            "expected_release_outcome": outcome,
        }
    return indexed


def _evidence_by_kind(row: Mapping[str, Any]) -> dict[str, str]:
    config = row.get("config")
    paths = config.get("evidence_paths") if isinstance(config, Mapping) else None
    result: dict[str, str] = {}
    if not isinstance(paths, list):
        return result
    for item in paths:
        if not isinstance(item, Mapping):
            continue
        kind = str(item.get("kind", "")).strip()
        sha256 = str(item.get("sha256", "")).strip().lower()
        if kind and len(sha256) == 64:
            result[kind] = sha256
    return result


def _transform_receipt_sha256(row: Mapping[str, Any]) -> str:
    """Hash transform facts while excluding machine-private paths.

    Returns:
        A deterministic SHA-256 digest for the compact transformation receipt.
    """
    source = row.get("source")
    source_compact = {}
    if isinstance(source, Mapping):
        for key in ("episode_id", "episodes_sha256", "row_index"):
            if key in source:
                source_compact[key] = source[key]
    payload = {
        key: row.get(key)
        for key in (
            "schema_version",
            "trace_schema_version",
            "normalization_policy",
            "execution_commit",
            "raw_trace_sha256",
            "normalized_trace_sha256",
            "removed_field_count",
            "removed_field_counts",
            "removed_field_paths_sha256",
            "semantic_payload_unchanged",
        )
        if key in row
    }
    payload["source"] = source_compact
    return _canonical_sha256(payload)


def _source_pointer(
    binding_rows: list[Mapping[str, Any]],
    arm_receipts: list[Mapping[str, Any]],
    *,
    external_root: str,
    host_alias: str,
    retrieval_key: str,
) -> dict[str, Any]:
    arms: list[dict[str, Any]] = []
    rows_by_arm = {str(row.get("arm")): row for row in binding_rows if isinstance(row, Mapping)}
    for arm_receipt in sorted(arm_receipts, key=lambda item: str(item.get("arm", ""))):
        arm = str(arm_receipt.get("arm", ""))
        sample = rows_by_arm.get(arm)
        if sample is None:
            raise RealReexportPackageError(f"binding receipt has no row for arm {arm!r}")
        evidence = _evidence_by_kind(sample)
        job_id = str(arm_receipt.get("job_id", "")).strip()
        if not job_id:
            raise RealReexportPackageError(f"binding receipt has no job id for arm {arm!r}")
        arms.append(
            {
                "arm": arm,
                "job_id": job_id,
                "planner": sample.get("planner"),
                "scenario_id": sample.get("scenario_id"),
                "campaign": sample.get("campaign"),
                "retrieval_key": f"{retrieval_key.rstrip('/')}/job-{job_id}",
                "manifest_sha256": str(arm_receipt.get("manifest_sha256", "")),
                "episodes_sha256": str(arm_receipt.get("episodes_sha256", "")),
                "run_summary_sha256": evidence.get("run_summary.yaml"),
                "preflight_sha256": evidence.get("validate_config.json"),
                "raw_trace_artifacts": "external_data_hub; keep-out-of-git",
            }
        )
    return {
        "schema_version": SOURCE_POINTER_SCHEMA_VERSION,
        "backend": "external_data_hub",
        "host_alias": host_alias,
        "root": external_root,
        "retrieval_key": retrieval_key,
        "raw_artifacts_keep_out_of_git": True,
        "normalized_artifacts_keep_out_of_git": True,
        "arms": arms,
    }


def _validate_binding_receipt(  # noqa: C901, PLR0912
    path: Path,
    request_tuples: set[tuple[str, str, int]],
    expected: Mapping[tuple[str, str, int], Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[tuple[str, str, int], dict[str, Any]]]:
    payload = _read_json(path, "binding receipt")
    if payload.get("schema_version") != REAL_REEXPORT_BINDING_SCHEMA:
        raise RealReexportPackageError("binding receipt schema mismatch")
    if payload.get("status") != "complete" or payload.get("package_status") != (
        "not_created; package assembly belongs to issue #6412"
    ):
        raise RealReexportPackageError("binding receipt is not the expected complete #6411 receipt")
    rows = payload.get("rows")
    summary = payload.get("summary")
    if not isinstance(rows, list) or not isinstance(summary, Mapping):
        raise RealReexportPackageError("binding receipt lacks rows or summary")
    if summary != {"n_admitted": 88, "n_not_admitted": 2, "n_rows": 90} or len(rows) != 90:
        raise RealReexportPackageError("binding receipt is not exactly the approved 88/90 boundary")
    by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise RealReexportPackageError("binding row must be an object")
        row = dict(raw)
        key = _tuple_key(row)
        if key in by_key:
            raise RealReexportPackageError(f"duplicate binding tuple: {key}")
        if key not in request_tuples or key not in expected:
            raise RealReexportPackageError(
                f"binding tuple is outside the requested contract: {key}"
            )
        trace_path = Path(str(row.get("normalized_trace_path", "")))
        trace_sha256 = str(row.get("normalized_trace_sha256", "")).lower()
        if not trace_path.is_file() or len(trace_sha256) != 64:
            raise RealReexportPackageError(f"normalized trace is unavailable for {key}")
        if _sha256_file(trace_path) != trace_sha256:
            raise RealReexportPackageError(f"normalized trace digest mismatch for {key}")
        if row.get("trace_schema_version") != "simulation_trace_export.v1":
            raise RealReexportPackageError(f"trace schema mismatch in binding row {key}")
        expected_outcome = str(expected[key]["expected_release_outcome"])
        release_outcome = str(row.get("release_outcome", ""))
        rerun_outcome = str(row.get("rerun_outcome", ""))
        if release_outcome != expected_outcome or rerun_outcome not in WORKED_EXAMPLE_OUTCOMES:
            raise RealReexportPackageError(
                f"binding outcome mismatch with expected release for {key}"
            )
        admission = str(row.get("admission_status", ""))
        expected_excluded = key in _EXCLUDED_TUPLES
        if admission != ("not_admitted" if expected_excluded else "admitted"):
            raise RealReexportPackageError(f"binding admission mismatch for {key}")
        if (rerun_outcome != release_outcome) != expected_excluded:
            raise RealReexportPackageError(f"binding outcome boundary mismatch for {key}")
        by_key[key] = row
    if set(by_key) != request_tuples or set(by_key) != set(expected):
        raise RealReexportPackageError("binding, request, and expected-outcome tuple sets differ")
    arm_receipts = payload.get("arms")
    if not isinstance(arm_receipts, list) or len(arm_receipts) != len(REAL_REEXPORT_ARMS):
        raise RealReexportPackageError("binding receipt does not contain the three arm receipts")
    return payload, [dict(item) for item in arm_receipts if isinstance(item, Mapping)], by_key


def _copy_trace(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if _sha256_file(source) != _sha256_file(destination):
        raise RealReexportPackageError(f"trace copy digest mismatch: {source}")


def _mapping_row(
    row: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    relative_uri: str,
    arm_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    source = row.get("source")
    source = source if isinstance(source, Mapping) else {}
    evidence = _evidence_by_kind(row)
    source_provenance = {
        "arm": row.get("arm"),
        "campaign": row.get("campaign"),
        "job_id": row.get("job_id"),
        "execution_commit": row.get("execution_commit"),
        "source_manifest_sha256": arm_receipt.get("manifest_sha256"),
        "source_episodes_sha256": arm_receipt.get("episodes_sha256"),
        "source_episode_id": source.get("episode_id"),
        "source_row_index": source.get("row_index"),
        "algorithm_config_hash": row.get("algorithm_config_hash"),
        "row_config_hash": row.get("row_config_hash"),
        "run_summary_sha256": evidence.get("run_summary.yaml"),
        "preflight_sha256": evidence.get("validate_config.json"),
    }
    return {
        "scenario_id": str(row["scenario_id"]),
        "planner": str(row["planner"]),
        "seed": int(row["seed"]),
        "episode_id": str(source.get("episode_id", "")),
        "release_episode_id": str(expected["release_episode_id"]),
        "expected_release_outcome": str(expected["expected_release_outcome"]),
        "rerun_outcome": str(row["rerun_outcome"]),
        "release_outcome": str(row["release_outcome"]),
        "admission_status": str(row["admission_status"]),
        "exclusion_reason": "outcome_mismatch"
        if str(row["admission_status"]) == "not_admitted"
        else None,
        "trace_artifact_uri": relative_uri,
        "trace_sha256": str(row["normalized_trace_sha256"]).lower(),
        "raw_trace_sha256": str(row["raw_trace_sha256"]).lower(),
        "normalized_trace_sha256": str(row["normalized_trace_sha256"]).lower(),
        "transformation_receipt_sha256": _transform_receipt_sha256(row),
        "transformation_schema_version": str(row["schema_version"]),
        "source_provenance": source_provenance,
    }


def _write_checksums(root: Path) -> str:
    entries: list[str] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in {_CHECKSUMS_NAME, _COMPLETE_NAME}:
            continue
        entries.append(f"{_sha256_file(path)}  {relative}")
    content = ("\n".join(entries) + "\n").encode("utf-8") if entries else b""
    (root / _CHECKSUMS_NAME).write_bytes(content)
    return _sha256_bytes(content)


def _compact_resolution_summary(resolution: Mapping[str, Any]) -> dict[str, Any]:
    rows = resolution.get("rows")
    summary = resolution.get("summary")
    if not isinstance(rows, list) or not isinstance(summary, Mapping):
        raise RealReexportPackageError("resolution manifest lacks rows or summary")
    compact_rows = []
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise RealReexportPackageError("resolution row must be an object")
        compact_rows.append(
            {
                "candidate_id": raw.get("candidate_id"),
                "scenario_id": raw.get("scenario_id"),
                "planner_id": raw.get("planner_id"),
                "seed": raw.get("seed"),
                "resolution_status": raw.get("resolution_status"),
                "admission_status": raw.get("admission_status"),
                "exclusion_reason": raw.get("exclusion_reason"),
                "reason_code": raw.get("reason_code"),
                "trace_content_hash": raw.get("trace_content_hash"),
                "raw_trace_sha256": raw.get("raw_trace_sha256"),
                "normalized_trace_sha256": raw.get("normalized_trace_sha256"),
                "transformation_receipt_sha256": raw.get("transformation_receipt_sha256"),
                "source_provenance": raw.get("source_provenance"),
            }
        )
    return {
        "schema_version": RESOLUTION_SUMMARY_SCHEMA_VERSION,
        "visualization_only": True,
        "summary": dict(summary),
        "rows": sorted(compact_rows, key=lambda row: str(row.get("candidate_id"))),
    }


def assemble_real_reexport_package(
    *,
    binding_receipt: Path,
    request_manifest: Path,
    expected_outcomes: Path,
    output_dir: Path,
    external_root: str = "benchmark-results/robot_sf_ll7/issue5756",
    host_alias: str = "imech156-u",
    retrieval_key: str = "issue5756",
) -> dict[str, Any]:
    """Atomically assemble a local 88/2 package from a real binding receipt.

    Returns:
        The assembled package manifest.
    """
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise RealReexportPackageError(f"refusing to overwrite existing package: {output_dir}")
    request_tuples, request_sha256 = _load_request_tuples(request_manifest.resolve())
    expected = _load_expected_outcomes(expected_outcomes.resolve())
    binding, arm_receipts, binding_rows = _validate_binding_receipt(
        binding_receipt.resolve(), request_tuples, expected
    )
    if binding.get("request_contract", {}).get("sha256") != request_sha256:
        raise RealReexportPackageError("binding receipt request-manifest digest mismatch")
    source_pointer = _source_pointer(
        list(binding_rows.values()),
        arm_receipts,
        external_root=external_root,
        host_alias=host_alias,
        retrieval_key=retrieval_key,
    )
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    try:
        arm_receipt_by_key = {
            str(item.get("arm")): item for item in arm_receipts if item.get("arm")
        }
        mapping_rows: list[dict[str, Any]] = []
        compact_expected_rows: list[dict[str, Any]] = []
        exclusions: list[dict[str, Any]] = []
        for key in sorted(binding_rows):
            row = binding_rows[key]
            expected_row = expected[key]
            admission = str(row["admission_status"])
            directory = "traces" if admission == "admitted" else "excluded_traces"
            relative_uri = f"{directory}/{key[0]}/{key[1]}/seed-{key[2]}.json"
            _copy_trace(Path(str(row["normalized_trace_path"])), staging / relative_uri)
            mapping_row = _mapping_row(
                row,
                expected_row,
                relative_uri=relative_uri,
                arm_receipt=arm_receipt_by_key[str(row["arm"])],
            )
            mapping_rows.append(mapping_row)
            compact_expected_rows.append(
                {
                    "planner": key[0],
                    "scenario_id": key[1],
                    "seed": key[2],
                    "release_episode_id": expected_row["release_episode_id"],
                    "expected_release_outcome": expected_row["expected_release_outcome"],
                    "rerun_outcome": row["rerun_outcome"],
                    "admission_status": admission,
                }
            )
            if admission == "not_admitted":
                exclusions.append(
                    {
                        "schema_version": EXCLUSION_SCHEMA_VERSION,
                        "planner": key[0],
                        "scenario_id": key[1],
                        "seed": key[2],
                        "release_episode_id": expected_row["release_episode_id"],
                        "expected_release_outcome": expected_row["expected_release_outcome"],
                        "rerun_outcome": row["rerun_outcome"],
                        "admission_status": "not_admitted",
                        "exclusion_reason": "outcome_mismatch",
                        "raw_trace_sha256": mapping_row["raw_trace_sha256"],
                        "normalized_trace_sha256": mapping_row["normalized_trace_sha256"],
                        "transformation_receipt_sha256": mapping_row[
                            "transformation_receipt_sha256"
                        ],
                        "source_provenance": mapping_row["source_provenance"],
                        "normalized_trace_uri": relative_uri,
                    }
                )
        if len(mapping_rows) != 90 or len(exclusions) != 2:
            raise RealReexportPackageError(
                "assembled package is not exactly 88 admitted plus 2 excluded"
            )
        if {
            (str(item["planner"]), str(item["scenario_id"]), int(item["seed"]))
            for item in exclusions
        } != _EXCLUDED_TUPLES:
            raise RealReexportPackageError("assembled exclusions do not match the named boundary")
        mapping_payload = {
            "schema_version": ISSUE_5756_MAPPING_SCHEMA_VERSION,
            "n_rows": len(mapping_rows),
            "provenance": {
                **ISSUE_5756_PINNED_PROVENANCE,
                "request_manifest_sha256": request_sha256,
            },
            "rows": mapping_rows,
        }
        expected_payload = {
            "schema_version": EXPECTED_OUTCOMES_SCHEMA_VERSION,
            "visualization_only": True,
            "rows": compact_expected_rows,
        }
        _write_json(staging / "source_pointer.json", source_pointer)
        _write_json(staging / "mapping_receipt.json", mapping_payload)
        _write_json(staging / "expected_outcomes.json", expected_payload)
        for exclusion in exclusions:
            name = (
                f"{exclusion['planner']}_{exclusion['scenario_id']}_seed-{exclusion['seed']}.json"
            )
            _write_json(staging / "exclusions" / name, exclusion)
        manifest = {
            "schema_version": PACKAGE_SCHEMA_VERSION,
            "status": "assembled",
            "visualization_only": True,
            "claim_boundary": (
                "Trace figures are a visualization-only re-export of the pinned #5756 inputs; "
                "this package is not release-result evidence."
            ),
            "n_requested": 90,
            "n_admitted": 88,
            "n_excluded": 2,
            "excluded_tuples": [list(key) for key in sorted(_EXCLUDED_TUPLES)],
            "execution_commit": str(binding["execution_commit"]),
            "request_manifest_sha256": request_sha256,
            "binding_receipt_sha256": _sha256_file(binding_receipt.resolve()),
            "expected_outcomes_sha256": _sha256_file(expected_outcomes.resolve()),
            "source_pointer_sha256": _sha256_file(staging / "source_pointer.json"),
            "mapping_receipt_sha256": _sha256_file(staging / "mapping_receipt.json"),
            "expected_outcomes_compact_sha256": _sha256_file(staging / "expected_outcomes.json"),
            "human_evidence_owner_review": "pending",
        }
        _write_json(staging / "package_manifest.json", manifest)
        report = {
            "schema_version": PACKAGE_REPORT_SCHEMA_VERSION,
            "status": "assembled; figure QA pending",
            "visualization_only": True,
            "n_requested": 90,
            "n_admitted": 88,
            "n_excluded": 2,
            "exclusions": exclusions,
            "raw_artifacts": "external_data_hub; keep-out-of-git",
            "normalized_artifacts": "local package only; keep-out-of-git",
            "human_evidence_owner_review": "pending",
        }
        _write_json(staging / "package_report.json", report)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, output_dir)
        return manifest
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def materialize_resolver_mapping(package_dir: Path, output_path: Path) -> dict[str, Any]:
    """Write a resolver-valid mapping with absolute local trace URIs.

    Returns:
        The materialized mapping payload.
    """
    package_dir = package_dir.resolve()
    output_path = output_path.resolve()
    if package_dir in output_path.parents or output_path == package_dir:
        raise RealReexportPackageError("resolver mapping output must be outside the package")
    payload = _read_json(package_dir / "mapping_receipt.json", "package mapping receipt")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise RealReexportPackageError("package mapping receipt has no rows")
    materialized = json.loads(json.dumps(payload))
    for row in materialized["rows"]:
        uri = Path(str(row["trace_artifact_uri"]))
        local_path = (package_dir / uri).resolve()
        if not local_path.is_file():
            raise RealReexportPackageError(f"package trace is missing: {uri}")
        row["trace_artifact_uri"] = str(local_path)
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".json", delete=False) as handle:
        temp_path = Path(handle.name)
        handle.write(_canonical_bytes(materialized, newline=True))
    try:
        load_episode_mapping(
            temp_path,
            expected_count=90,
            expected_provenance=materialized["provenance"],
        )
    finally:
        temp_path.unlink(missing_ok=True)
    _write_json(output_path, materialized)
    return materialized


def finalize_real_reexport_package(
    package_dir: Path,
    *,
    resolution: Mapping[str, Any],
    figure_qa: Mapping[str, Any],
) -> dict[str, Any]:
    """Finalize a package only after resolver and zero-error figure QA pass.

    Returns:
        The package completion marker.
    """
    package_dir = package_dir.resolve()
    manifest = _read_json(package_dir / "package_manifest.json", "package manifest")
    if manifest.get("status") != "assembled":
        raise RealReexportPackageError("package is not in the assembled state")
    if figure_qa.get("schema_version") != FIGURE_QA_SCHEMA_VERSION:
        raise RealReexportPackageError("figure QA schema mismatch")
    if figure_qa.get("status") != "passed" or figure_qa.get("n_error_defects") != 0:
        raise RealReexportPackageError("figure QA did not pass with zero error defects")
    if figure_qa.get("visualization_only") is not True:
        raise RealReexportPackageError("figure QA lacks visualization-only boundary")
    compact_resolution = _compact_resolution_summary(resolution)
    summary = compact_resolution["summary"]
    if summary != {
        "n_candidates": 90,
        "n_resolved": 88,
        "n_trace_missing": 0,
        "n_schema_mismatch": 0,
        "n_provenance_incomplete": 2,
    }:
        raise RealReexportPackageError(f"resolver summary is not 88/2: {summary}")
    _write_json(package_dir / "resolver_summary.json", compact_resolution)
    _write_json(package_dir / "figure_qa.json", dict(figure_qa))
    report = _read_json(package_dir / "package_report.json", "package report")
    report.update(
        {
            "status": "complete",
            "figure_qa": {
                "status": "passed",
                "sha256": _sha256_file(package_dir / "figure_qa.json"),
            },
            "resolver_summary": {
                "status": "88_resolved_plus_2_explicit_exclusions",
                "sha256": _sha256_file(package_dir / "resolver_summary.json"),
            },
        }
    )
    _write_json(package_dir / "package_report.json", report)
    manifest.update(
        {
            "status": "complete",
            "figure_qa_sha256": _sha256_file(package_dir / "figure_qa.json"),
            "resolver_summary_sha256": _sha256_file(package_dir / "resolver_summary.json"),
            "package_report_sha256": _sha256_file(package_dir / "package_report.json"),
        }
    )
    _write_json(package_dir / "package_manifest.json", manifest)
    sums_sha256 = _write_checksums(package_dir)
    complete = {
        "schema_version": PACKAGE_COMPLETE_SCHEMA_VERSION,
        "status": "complete",
        "visualization_only": True,
        "n_requested": 90,
        "n_admitted": 88,
        "n_excluded": 2,
        "sha256sums_sha256": sums_sha256,
        "human_evidence_owner_review": "pending",
    }
    _write_json(package_dir / _COMPLETE_NAME, complete)
    return complete


def verify_complete_package(package_dir: Path) -> dict[str, Any]:
    """Verify the completion marker and every listed package checksum.

    Returns:
        The verified package completion marker.
    """
    package_dir = package_dir.resolve()
    complete = _read_json(package_dir / _COMPLETE_NAME, "package completion marker")
    if complete.get("schema_version") != PACKAGE_COMPLETE_SCHEMA_VERSION:
        raise RealReexportPackageError("package completion schema mismatch")
    sums_path = package_dir / _CHECKSUMS_NAME
    if _sha256_file(sums_path) != complete.get("sha256sums_sha256"):
        raise RealReexportPackageError("package checksum-list digest mismatch")
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        if not separator or _sha256_file(package_dir / relative) != digest:
            raise RealReexportPackageError(f"package checksum mismatch: {relative}")
    manifest = _read_json(package_dir / "package_manifest.json", "package manifest")
    if manifest.get("status") != "complete" or manifest.get("n_admitted") != 88:
        raise RealReexportPackageError("package manifest is not complete 88/2 evidence")
    return complete


__all__ = [
    "EXPECTED_OUTCOMES_SCHEMA_VERSION",
    "FIGURE_QA_SCHEMA_VERSION",
    "PACKAGE_SCHEMA_VERSION",
    "RealReexportPackageError",
    "assemble_real_reexport_package",
    "finalize_real_reexport_package",
    "materialize_resolver_mapping",
    "verify_complete_package",
]
