"""Reconcile compact multi-run search evidence when raw replay inputs are absent."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from robot_sf.adversarial.replay_gallery import (
    GALLERY_SCHEMA_VERSION,
    _display_path,
    _repository_root,
    _validated_output_directory,
    _write_json,
)

_PACKET_INPUTS = (
    "candidate_evaluations.csv",
    "row_status.json",
    "convergence_report.json",
    "summary.json",
    "run_metadata.json",
)
_REPLAY_INPUT_SCOPE = (
    "candidate scenario YAML and episode record only; availability does not establish that "
    "all replay configuration and map dependencies are present"
)
_ROW_STATUS_VALUES = {
    "successful_evidence",
    "accepted_unavailable",
    "unexpected_failure",
    "fallback",
    "degraded",
    "blocked",
}


def build_search_evidence_packet_gallery(
    packet_dir: str | Path,
    output_dir: str | Path,
    *,
    top_k: int,
    tolerance: float,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    """Build a zero-case gallery from a compact, cross-checked search evidence packet.

    This directory input is intended for completed packet bundles that retain their compact
    candidate ledger but not candidate-specific scenario and episode files. A packet with a
    critical candidate must instead be passed as its original search manifest to the regular
    replay path; this function never infers or reconstructs missing replay inputs.
    """
    raw_source = Path(packet_dir).expanduser()
    if raw_source.is_symlink():
        raise ValueError("search evidence packet directory must not be a symlink")
    source = raw_source.resolve()
    root = (repository_root or _repository_root()).resolve()
    destination = _validated_output_directory(output_dir, root=root)
    if not source.is_dir():
        raise ValueError(f"search evidence packet must be a directory: {source}")

    payload, source_hashes = _read_packet(source)
    _check_source_metadata(payload)
    accounting = _reconcile_packet(payload, source_root=_source_root(source, root))
    critical_count = sum(
        row["case_criticality"] == "critical_planner_failure" for row in accounting
    )
    unknown_count = sum(row["case_criticality"] == "unknown" for row in accounting)
    if critical_count:
        raise ValueError(
            f"packet contains {critical_count} critical candidate(s); pass the original "
            "adversarial-search-manifest.v1 file to materialize and replay them"
        )

    pilot_budget = payload["summary"]["pilot_budget"]
    planned_count = _nonnegative_integer(pilot_budget["planned"], "pilot planned")
    missing_planned_count = planned_count - len(accounting)
    scoreless_count = sum(row["objective_value"] is None for row in accounting)
    zero_critical_verified, selection_status, replay_reason = _packet_result_disposition(
        bool(accounting), unknown_count, scoreless_count, missing_planned_count
    )
    _verify_packet_zero_critical_report(zero_critical_verified, payload["convergence_report"])

    missing_input_count = sum(row["replay_input_status"] == "missing" for row in accounting)
    available_input_count = sum(row["replay_input_status"] == "available" for row in accounting)
    summary = {
        "source_candidate_count": len(accounting),
        "planned_candidate_count": planned_count,
        "missing_planned_candidate_count": missing_planned_count,
        "scoreless_candidate_count": scoreless_count,
        "replay_input_scope": _REPLAY_INPUT_SCOPE,
        "selected_case_count": 0,
        "replay_match_count": 0,
        "replay_mismatch_count": 0,
        "replay_unavailable_count": 0,
        "critical_case_count": critical_count,
        "criticality_unknown_count": unknown_count,
        "zero_critical_result_verified": zero_critical_verified,
        "replay_input_status_counts": {
            "available": available_input_count,
            "missing": missing_input_count,
            "partial": sum(row["replay_input_status"] == "partial" for row in accounting),
            "unknown": sum(row["replay_input_status"] == "unknown" for row in accounting),
            "digest_mismatch": sum(
                row["replay_input_status"] == "digest_mismatch" for row in accounting
            ),
        },
        "execution_outcome_counts": dict(
            sorted(Counter(row["execution_outcome"] for row in accounting).items())
        ),
        "scenario_eligibility_counts": dict(
            sorted(Counter(row["scenario_eligibility"] for row in accounting).items())
        ),
        "case_criticality_counts": dict(
            sorted(Counter(row["case_criticality"] for row in accounting).items())
        ),
    }
    source_revision = payload["summary"].get("source_revision")
    manifest = {
        "schema_version": GALLERY_SCHEMA_VERSION,
        "source": {
            "kind": "compact_multi_run_search_evidence_packet",
            "packet_path": _display_path(source, _source_root(source, root)),
            "source_revision": source_revision if isinstance(source_revision, str) else None,
            "revision_status": "known" if isinstance(source_revision, str) else "unknown",
            "input_sha256": source_hashes,
            "source_manifests": [_manifest_summary(item) for item in payload["source_manifests"]],
        },
        "selection": {
            "top_k": top_k,
            "objective": payload["summary"].get("pilot_metrics", {}).get("objective"),
            "ordering": "objective_value_desc_then_source_candidate_index_asc",
            "duplicate_policy": "source_packet_candidate_hashes_preserved; no cases selected",
            "replay_objective_absolute_tolerance": tolerance,
            "status": selection_status,
        },
        "replay": {
            "attempted": False,
            "reason": replay_reason,
            "historical_compatibility_case_is_separate": True,
        },
        "summary": summary,
        "candidates": accounting,
        "cases": [],
    }
    destination.mkdir(parents=True, exist_ok=False)
    _write_json(destination / "gallery_manifest.json", manifest)
    _write_packet_readme(destination, summary, missing_input_count, selection_status)
    return manifest


def _packet_result_disposition(
    has_rows: bool,
    unknown_count: int,
    scoreless_count: int,
    missing_planned_count: int,
) -> tuple[bool, str, str]:
    """Classify a packet result without claiming zero criticality for incomplete evidence."""
    verified = has_rows and not unknown_count and not scoreless_count and not missing_planned_count
    if verified:
        return (
            True,
            "no_critical_candidate_in_reconciled_packet",
            "The complete reconciled budget contains no critical candidate.",
        )
    if unknown_count:
        return (
            False,
            "criticality_unknown_no_candidate_selected",
            "Criticality is incomplete; no candidate was selected or replayed.",
        )
    if scoreless_count:
        return (
            False,
            "scoreless_candidate_no_candidate_selected",
            "At least one candidate is scoreless; no candidate was selected or replayed.",
        )
    return (
        False,
        "incomplete_budget_no_candidate_selected",
        "The declared search budget is incomplete; no candidate was selected or replayed.",
    )


def _verify_packet_zero_critical_report(verified: bool, report: dict[str, Any]) -> None:
    """Require a zero critical aggregate after its full row-level reconciliation."""
    if verified and _reported_critical_count(report) != 0:
        raise ValueError(
            "source convergence report conflicts with the reconciled zero-critical rows"
        )


def _read_packet(source: Path) -> tuple[dict[str, Any], dict[str, str]]:
    """Read one packet snapshot and capture every consumed file digest."""
    files, manifest_paths = _snapshot_packet_files(source)
    bundle_receipts = _validate_bundle_receipts(source, files)
    parsed = _parse_packet_json_inputs(files)
    parsed["candidate_rows"] = _read_candidate_csv(files["candidate_evaluations.csv"])
    parsed["source_manifests"] = _load_source_manifests(source, files, manifest_paths)
    hashes = {
        relative: hashlib.sha256(content).hexdigest() for relative, content in sorted(files.items())
    }
    hashes.update(bundle_receipts)
    return parsed, hashes


def _validate_bundle_receipts(source: Path, files: dict[str, bytes]) -> dict[str, str]:
    """Verify consumed packet bytes against the enclosing evidence bundle receipts."""
    manifest_path = source.parent / "evidence_bundle_manifest.json"
    checksums_path = source.parent / "checksums.sha256"
    manifest_bytes, checksums_bytes = _read_bundle_receipt_files(manifest_path, checksums_path)
    manifest_by_path = _parse_bundle_manifest(manifest_bytes)
    checksum_by_path = _parse_bundle_checksums(checksums_bytes)
    _validate_bundle_artifacts(source, manifest_by_path, checksum_by_path)
    if not set(files) <= set(manifest_by_path):
        missing = sorted(set(files) - set(manifest_by_path))
        raise ValueError(f"consumed packet files are absent from bundle manifest: {missing}")
    for relative, content in files.items():
        entry = manifest_by_path[relative]
        if (
            len(content) != entry["size_bytes"]
            or hashlib.sha256(content).hexdigest() != str(entry["sha256"]).lower()
        ):
            raise ValueError(f"consumed packet file conflicts with bundle manifest: {relative}")
    return {
        "../evidence_bundle_manifest.json": hashlib.sha256(manifest_bytes).hexdigest(),
        "../checksums.sha256": hashlib.sha256(checksums_bytes).hexdigest(),
    }


def _read_bundle_receipt_files(manifest_path: Path, checksums_path: Path) -> tuple[bytes, bytes]:
    """Read required receipt files without following symlinks."""
    if manifest_path.is_symlink() or checksums_path.is_symlink():
        raise ValueError("search evidence packet bundle receipts must be regular files")
    if not manifest_path.is_file() or not checksums_path.is_file():
        raise ValueError(
            "search evidence packet requires sibling evidence_bundle_manifest.json and checksums.sha256"
        )
    return manifest_path.read_bytes(), checksums_path.read_bytes()


def _parse_bundle_manifest(manifest_bytes: bytes) -> dict[str, dict[str, Any]]:
    """Parse and validate unique path, checksum, and size records."""
    try:
        manifest = json.loads(manifest_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("search evidence packet bundle manifest is malformed") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != "evidence_bundle.v1":
        raise ValueError("unsupported search evidence packet bundle manifest schema")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("search evidence packet bundle manifest has no file entries")

    manifest_by_path: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("search evidence packet bundle manifest contains a malformed entry")
        relative = _safe_bundle_relative_path(entry.get("path"))
        digest = entry.get("sha256")
        size = entry.get("size_bytes")
        if (
            relative in manifest_by_path
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in digest)
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
        ):
            raise ValueError(
                "search evidence packet bundle manifest has duplicate or invalid entries"
            )
        manifest_by_path[relative] = entry
    return manifest_by_path


def _parse_bundle_checksums(checksums_bytes: bytes) -> dict[str, str]:
    """Parse a sha256sum-style receipt and reject duplicate or malformed paths."""
    try:
        checksums_text = checksums_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("search evidence packet checksum sidecar is malformed") from exc
    checksum_by_path: dict[str, str] = {}
    for line_number, line in enumerate(checksums_text.splitlines(), start=1):
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            raise ValueError(f"malformed bundle checksum line {line_number}")
        digest, relative_raw = fields
        relative = _safe_bundle_relative_path(relative_raw)
        if (
            len(digest) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in digest)
            or relative in checksum_by_path
        ):
            raise ValueError(f"duplicate or invalid bundle checksum line {line_number}")
        checksum_by_path[relative] = digest.lower()
    return checksum_by_path


def _validate_bundle_artifacts(
    source: Path,
    manifest_by_path: dict[str, dict[str, Any]],
    checksum_by_path: dict[str, str],
) -> None:
    """Check all bundle payload files against both independent recorded digests."""
    expected_checksum_paths: set[str] = set()
    for relative, entry in manifest_by_path.items():
        checksum_relative = f"{source.name}/{relative}"
        expected_checksum_paths.add(checksum_relative)
        digest = str(entry["sha256"]).lower()
        if checksum_by_path.get(checksum_relative) != digest:
            raise ValueError(f"bundle checksum sidecar conflicts with manifest entry: {relative}")
        artifact = _bundle_artifact_path(source, relative)
        content = artifact.read_bytes()
        if len(content) != entry["size_bytes"] or hashlib.sha256(content).hexdigest() != digest:
            raise ValueError(f"bundle artifact digest or size conflicts with manifest: {relative}")
    if set(checksum_by_path) != expected_checksum_paths:
        raise ValueError("bundle checksum sidecar and manifest list different file sets")


def _bundle_artifact_path(source: Path, relative: str) -> Path:
    """Resolve one bundle path while rejecting symlinked path components."""
    artifact = source
    for component in Path(relative).parts:
        artifact = artifact / component
        if artifact.is_symlink():
            raise ValueError(f"bundle manifest artifact is missing or unsafe: {relative}")
    try:
        artifact.resolve().relative_to(source.resolve())
    except ValueError as exc:
        raise ValueError(f"bundle manifest artifact is missing or unsafe: {relative}") from exc
    if not artifact.is_file():
        raise ValueError(f"bundle manifest artifact is missing or unsafe: {relative}")
    return artifact


def _safe_bundle_relative_path(value: Any) -> str:
    """Return a canonical in-bundle path or reject absolute/traversal paths."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("bundle paths must be non-empty strings")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value or path.as_posix() != value:
        raise ValueError(f"unsafe bundle-relative path: {value!r}")
    return path.as_posix()


def _snapshot_packet_files(source: Path) -> tuple[dict[str, bytes], list[Path]]:
    """Capture required packet files and source manifests as one read-only snapshot."""
    files: dict[str, bytes] = {}
    for relative in _PACKET_INPUTS:
        path = source / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"search evidence packet is missing required input: {relative}")
        files[relative] = path.read_bytes()
    manifests_dir = source / "source_manifests"
    if manifests_dir.is_symlink():
        raise ValueError("search evidence packet source_manifests directory must not be a symlink")
    manifest_paths = sorted(manifests_dir.glob("*.json")) if manifests_dir.is_dir() else []
    if not manifest_paths:
        raise ValueError("search evidence packet has no source_manifests/*.json files")
    for path in manifest_paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError(
                f"search evidence packet source manifest must be a regular file: {path.name}"
            )
        files[path.relative_to(source).as_posix()] = path.read_bytes()
    return files, manifest_paths


def _parse_packet_json_inputs(files: dict[str, bytes]) -> dict[str, Any]:
    """Parse packet JSON inputs, leaving CSV and individual source manifests to their readers."""
    parsed: dict[str, Any] = {}
    for relative in _PACKET_INPUTS:
        if relative.endswith(".csv"):
            continue
        try:
            parsed[relative.removesuffix(".json")] = json.loads(files[relative])
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"invalid JSON packet input: {relative}") from exc
    if not all(isinstance(value, dict) for value in parsed.values()):
        raise ValueError("search evidence packet JSON inputs must be objects")
    return parsed


def _load_source_manifests(
    source: Path, files: dict[str, bytes], manifest_paths: list[Path]
) -> list[dict[str, Any]]:
    """Parse source manifests and retain only their content-bound payloads."""
    manifests = []
    for path in manifest_paths:
        relative = path.relative_to(source).as_posix()
        try:
            value = json.loads(files[relative])
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"invalid source search manifest: {relative}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"source search manifest must be an object: {relative}")
        manifests.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(files[relative]).hexdigest(),
                "payload": value,
            }
        )
    return manifests


def _read_candidate_csv(content: bytes) -> list[dict[str, str]]:
    """Parse the candidate ledger without silently accepting malformed rows."""
    try:
        text = content.decode("utf-8")
        reader = csv.DictReader(text.splitlines())
        rows = list(reader)
    except (UnicodeDecodeError, csv.Error) as exc:
        raise ValueError("candidate_evaluations.csv is not valid UTF-8 CSV") from exc
    required = {
        "run_id",
        "sampler",
        "sampler_seed",
        "evaluation_index",
        "scenario_id",
        "candidate_bundle",
        "candidate_parameters_json",
        "candidate_sha256",
        "effective_scenario_sha256",
        "certification_status",
        "certification_classification",
        "benchmark_eligibility",
        "analysis_eligibility",
        "row_status",
        "counts_as_execution_success_evidence",
        "execution_mode",
        "readiness_status",
        "availability_status",
        "episode_status",
        "termination_reason",
        "collision_event",
        "route_complete",
        "timeout_event",
        "failure_attribution",
        "objective_value",
        "min_clearance_m",
        "distance_to_human_min_m",
        "near_miss_count",
        "total_collision_count",
        "ped_collision_count",
        "steps",
        "source_manifest_sha256",
        "scenario_yaml_sha256",
        "episode_records_sha256",
        "error",
    }
    if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
        raise ValueError("candidate_evaluations.csv has missing or duplicate column names")
    if not required.issubset(reader.fieldnames):
        missing = sorted(required - set(reader.fieldnames or ()))
        raise ValueError(f"candidate_evaluations.csv is missing required columns: {missing}")
    if not rows:
        raise ValueError("candidate_evaluations.csv contains no candidate rows")
    if any(None in row for row in rows):
        raise ValueError("candidate_evaluations.csv contains a row with extra columns")
    return rows


def _reconcile_packet(payload: dict[str, Any], *, source_root: Path) -> list[dict[str, Any]]:
    """Cross-check every candidate against its manifest, row-status record and reports."""
    by_sha = _index_source_manifests(payload["source_manifests"])
    row_status = payload["row_status"]
    status_rows = row_status.get("rows")
    if not isinstance(status_rows, list):
        raise ValueError("row_status.json has no rows list")
    status_by_id = {str(row.get("row_id")): row for row in status_rows if isinstance(row, dict)}
    if len(status_by_id) != len(status_rows):
        raise ValueError("row_status.json has missing or duplicate row identities")
    accounting, by_run, seen_row_ids = _reconcile_candidate_rows(
        payload["candidate_rows"], by_sha, status_by_id, source_root=source_root
    )
    if seen_row_ids != set(status_by_id):
        raise ValueError("candidate ledger and row_status.json contain different candidate sets")
    _check_row_status_counts(row_status, status_by_id)
    _check_summary(
        payload["summary"],
        by_run,
        accounting,
        source_manifests=payload["source_manifests"],
    )
    _check_convergence(payload["convergence_report"], by_run)
    for item in accounting:
        item.pop("source_manifest_config", None)
    return accounting


def _check_source_metadata(payload: dict[str, Any]) -> None:
    """Verify the packet identity and source revision agree across metadata files."""
    summary = payload["summary"]
    run_metadata = payload["run_metadata"]
    row_status = payload["row_status"]
    convergence = payload["convergence_report"]
    if summary.get("schema_version") != "issue_9645_bounded_pilot_summary.v1":
        raise ValueError("unsupported #9645 pilot summary schema")
    if summary.get("issue") != 9645:
        raise ValueError("directory packet mode accepts the #9645 compact evidence packet only")
    if (
        row_status.get("schema_version") != "benchmark_row_status.v1"
        or row_status.get("issue") != 9645
    ):
        raise ValueError("unsupported #9645 row-status schema or issue identity")
    if convergence.get("schema_version") != "adversarial-search-convergence-report.v1":
        raise ValueError("unsupported falsification convergence report schema")
    source_revision = summary.get("source_revision")
    provenance_revision = run_metadata.get("experiment_source_commit")
    if source_revision != provenance_revision:
        raise ValueError("summary source revision conflicts with run metadata")
    if source_revision is not None and not _is_git_revision(source_revision):
        raise ValueError("summary source revision is not a full Git commit hash")
    if run_metadata.get("schema_version") != "issue_9645_execution_provenance.v1":
        raise ValueError("unsupported #9645 execution provenance schema")


def _manifest_summary(entry: dict[str, Any]) -> dict[str, Any]:
    """Project a source manifest into a compact, human-auditable index row."""
    payload = entry["payload"]
    config = payload.get("config") if isinstance(payload.get("config"), dict) else {}
    return {
        "path": entry["path"],
        "sha256": entry["sha256"],
        "schema_version": payload.get("schema_version"),
        "candidate_count": len(payload.get("candidates", [])),
        "seed": config.get("seed"),
        "budget": config.get("budget"),
        "objective": config.get("objective"),
        "output_dir": config.get("output_dir"),
    }


def _index_source_manifests(manifests: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Index valid source manifests by their exact content digest."""
    by_sha: dict[str, dict[str, Any]] = {}
    for entry in manifests:
        manifest = entry["payload"]
        config = manifest.get("config")
        candidates = manifest.get("candidates")
        if manifest.get("schema_version") != "adversarial-search-manifest.v1":
            raise ValueError(f"unsupported source manifest schema: {entry['path']}")
        if not isinstance(config, dict) or not isinstance(candidates, list):
            raise ValueError(f"source manifest lacks config or candidates: {entry['path']}")
        budget = _nonnegative_integer(config.get("budget"), "source manifest budget")
        if len(candidates) > budget:
            raise ValueError(f"source manifest exceeds its declared budget: {entry['path']}")
        if entry["sha256"] in by_sha:
            raise ValueError("duplicate source manifest content in packet")
        by_sha[entry["sha256"]] = entry
    return by_sha


def _reconcile_candidate_rows(
    candidate_rows: list[dict[str, str]],
    by_sha: dict[str, dict[str, Any]],
    status_by_id: dict[str, dict[str, Any]],
    *,
    source_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], set[str]]:
    """Bind all compact rows and status records to their source manifest candidate."""
    summaries: dict[str, list[dict[str, Any]]] = defaultdict(list)
    accounting: list[dict[str, Any]] = []
    seen_row_ids: set[str] = set()
    seen_candidate_indexes: dict[str, set[int]] = defaultdict(set)
    for row in candidate_rows:
        digest = row["source_manifest_sha256"].strip().lower()
        entry = by_sha.get(digest)
        if entry is None:
            raise ValueError("candidate ledger references an unknown source manifest digest")
        source_manifest = entry["payload"]
        config = source_manifest["config"]
        candidates = source_manifest["candidates"]
        run_id = row["run_id"].strip()
        sampler = row["sampler"].strip()
        seed = _integer(row["sampler_seed"], "sampler_seed")
        evaluation_index = _integer(row["evaluation_index"], "evaluation_index")
        if evaluation_index < 1 or evaluation_index > len(candidates):
            raise ValueError(f"candidate row index is outside its source manifest: {run_id}")
        output_dir = config.get("output_dir")
        if (
            seed != config.get("seed")
            or run_id != f"{sampler}_{seed}"
            or not isinstance(output_dir, str)
            or Path(output_dir).name != sampler
        ):
            raise ValueError(f"candidate row run identity conflicts with its manifest: {run_id}")
        if row["candidate_bundle"] != candidates[evaluation_index - 1].get("bundle_path"):
            raise ValueError(f"candidate bundle path conflicts with its manifest: {run_id}")

        candidate_payload = candidates[evaluation_index - 1]
        _check_candidate_row(row, candidate_payload, run_id=run_id)
        row_id = f"{run_id}:{evaluation_index:02d}"
        if row_id in seen_row_ids:
            raise ValueError(f"duplicate candidate row identity: {row_id}")
        seen_row_ids.add(row_id)
        seen_candidate_indexes[digest].add(evaluation_index)
        status = status_by_id.get(row_id)
        if status is None:
            raise ValueError(f"candidate row is absent from row_status.json: {row_id}")
        _check_row_status(row, candidate_payload, status, row_id=row_id)
        item = _candidate_accounting(
            row,
            candidate_payload,
            status=status,
            row_id=row_id,
            manifest_path=entry["path"],
            manifest_sha256=digest,
            source_root=source_root,
        )
        item["source_manifest_config"] = dict(config)
        accounting.append(item)
        summaries[run_id].append(item)
    for digest, entry in by_sha.items():
        candidate_count = len(entry["payload"]["candidates"])
        expected_indexes = set(range(1, candidate_count + 1))
        if seen_candidate_indexes.get(digest, set()) != expected_indexes:
            raise ValueError(
                f"candidate ledger does not account for every candidate in {entry['path']}"
            )
    return accounting, summaries, seen_row_ids


def _check_candidate_row(row: dict[str, str], candidate: dict[str, Any], *, run_id: str) -> None:
    """Require the compact candidate ledger to agree with its serialized search manifest."""
    _check_candidate_identity(row, candidate, run_id=run_id)
    _check_candidate_certificate(row, candidate, run_id=run_id)
    _check_candidate_attribution(row, candidate, run_id=run_id)


def _check_candidate_identity(
    row: dict[str, str], candidate: dict[str, Any], *, run_id: str
) -> None:
    """Check candidate parameters, objective and effective scenario identity."""
    try:
        parameters = json.loads(row["candidate_parameters_json"])
        eligibility = json.loads(row["analysis_eligibility"])
    except json.JSONDecodeError as exc:
        raise ValueError(f"candidate ledger has invalid embedded JSON: {run_id}") from exc
    if parameters != candidate.get("candidate"):
        raise ValueError(f"candidate parameters conflict with source manifest: {run_id}")
    parameter_digest = hashlib.sha256(_stable_json(parameters).encode("utf-8")).hexdigest()
    if parameter_digest != row["candidate_sha256"].strip().lower():
        raise ValueError(f"candidate parameter digest mismatch: {run_id}")
    if eligibility != candidate.get("analysis_eligibility"):
        raise ValueError(f"candidate analysis eligibility conflicts with source manifest: {run_id}")
    if row["effective_scenario_sha256"] != candidate.get("effective_scenario_hash"):
        raise ValueError(
            f"candidate effective-scenario digest conflicts with source manifest: {run_id}"
        )
    if not _same_finite_number(row["objective_value"], candidate.get("objective_value")):
        raise ValueError(f"candidate objective conflicts with source manifest: {run_id}")


def _check_candidate_certificate(
    row: dict[str, str], candidate: dict[str, Any], *, run_id: str
) -> None:
    """Compare the compact certificate summary with the nested source certificate."""
    cert_status = candidate.get("certification_status")
    details = cert_status.get("details") if isinstance(cert_status, dict) else None
    certificates = details.get("certificates") if isinstance(details, dict) else None
    if (
        not isinstance(certificates, list)
        or not certificates
        or not isinstance(certificates[0], dict)
    ):
        raise ValueError(f"candidate has no scenario certificate: {run_id}")
    certificate = certificates[0]
    expected = {
        "certification_status": cert_status.get("status"),
        "certification_classification": certificate.get("classification"),
        "benchmark_eligibility": certificate.get("benchmark_eligibility"),
        "scenario_id": certificate.get("scenario_id"),
    }
    for name, value in expected.items():
        if value is not None and row[name] != str(value):
            raise ValueError(f"candidate {name} conflicts with source manifest: {run_id}")


def _check_candidate_attribution(
    row: dict[str, str], candidate: dict[str, Any], *, run_id: str
) -> None:
    """Compare primary failure and canonical episode outcome fields."""
    attribution = candidate.get("failure_attribution")
    if not isinstance(attribution, dict) or row["failure_attribution"] != str(
        attribution.get("primary_failure")
    ):
        raise ValueError(f"candidate failure attribution conflicts with source manifest: {run_id}")
    attr_details = attribution.get("details")
    outcome = attr_details.get("outcome") if isinstance(attr_details, dict) else None
    if isinstance(outcome, dict):
        for column in ("collision_event", "route_complete", "timeout_event"):
            if _boolean(row[column], column) is not outcome.get(column):
                raise ValueError(
                    f"candidate outcome field {column} conflicts with manifest: {run_id}"
                )
    for column, key in (
        ("episode_status", "status"),
        ("termination_reason", "termination_reason"),
        ("execution_mode", "execution_mode"),
        ("readiness_status", "readiness_status"),
        ("availability_status", "availability_status"),
    ):
        if (
            isinstance(attr_details, dict)
            and key in attr_details
            and row[column] != str(attr_details[key])
        ):
            raise ValueError(f"candidate {column} conflicts with manifest: {run_id}")


def _check_row_status(
    row: dict[str, str], candidate: dict[str, Any], status: dict[str, Any], *, row_id: str
) -> None:
    """Check the compact status ledger against both CSV and manifest evidence."""
    _check_row_status_fields(row, status, row_id=row_id)
    _check_row_execution_evidence(row, candidate, status, row_id=row_id)


def _check_row_status_fields(row: dict[str, str], status: dict[str, Any], *, row_id: str) -> None:
    """Check status labels and planner availability against the candidate ledger."""
    if row["row_status"] not in _ROW_STATUS_VALUES:
        raise ValueError(f"candidate row has unsupported benchmark status: {row_id}")
    if status.get("status") != row["row_status"]:
        raise ValueError(f"row_status status conflicts with candidate ledger: {row_id}")
    _check_row_status_metadata(row, status, row_id=row_id)
    _check_row_status_evidence_flags(status, row_id=row_id)


def _check_row_status_metadata(row: dict[str, str], status: dict[str, Any], *, row_id: str) -> None:
    """Compare optional row-status metadata fields with the compact candidate ledger."""
    fields = (
        ("candidate_certification_status", "certification_status"),
        ("candidate_classification", "certification_classification"),
        ("execution_mode", "execution_mode"),
        ("readiness_status", "readiness_status"),
        ("availability_status", "availability_status"),
        ("error", "error"),
    )
    for status_field, csv_field in fields:
        expected = row[csv_field] or None
        if status.get(status_field) not in (None, expected):
            raise ValueError(f"row_status {status_field} conflicts: {row_id}")


def _check_row_status_evidence_flags(status: dict[str, Any], *, row_id: str) -> None:
    """Require typed success and degraded-execution flags before attribution."""
    if not isinstance(status.get("fallback_or_degraded"), bool):
        raise ValueError(f"row_status fallback/degraded status is missing or invalid: {row_id}")
    if not isinstance(status.get("counts_as_success_evidence"), bool):
        raise ValueError(f"row_status success attribution is missing or invalid: {row_id}")
    fallback_status = status["status"] in {"fallback", "degraded"}
    if status["fallback_or_degraded"] is not fallback_status:
        raise ValueError(f"row_status fallback/degraded label conflicts with its flag: {row_id}")
    if status.get("fallback_or_degraded") is True:
        if status.get("counts_as_success_evidence") is True:
            raise ValueError(f"fallback/degraded row cannot count as success evidence: {row_id}")


def _check_row_execution_evidence(
    row: dict[str, str],
    candidate: dict[str, Any],
    status: dict[str, Any],
    *,
    row_id: str,
) -> None:
    """Ensure execution-evidence claims agree across all three packet sources."""
    expected_execution_evidence = row["row_status"] == "successful_evidence"
    csv_execution_evidence = _boolean(
        row["counts_as_execution_success_evidence"], "counts_as_execution_success_evidence"
    )
    if csv_execution_evidence != expected_execution_evidence:
        raise ValueError(f"execution success attribution conflicts with row status: {row_id}")
    if status.get("counts_as_success_evidence") is not csv_execution_evidence:
        raise ValueError(
            f"row_status success attribution conflicts with candidate ledger: {row_id}"
        )
    analysis = candidate.get("analysis_eligibility")
    if isinstance(analysis, dict) and status.get("counts_as_success_evidence") is True:
        if analysis.get("eligible") is not True or row["availability_status"] != "available":
            raise ValueError(f"success evidence contradicts candidate eligibility: {row_id}")


def _candidate_accounting(
    row: dict[str, str],
    candidate: dict[str, Any],
    status: dict[str, Any],
    *,
    row_id: str,
    manifest_path: str,
    manifest_sha256: str,
    source_root: Path,
) -> dict[str, Any]:
    """Separate execution result, certificate eligibility, criticality and replay inputs."""
    attribution = candidate.get("failure_attribution")
    attribution = attribution if isinstance(attribution, dict) else {}
    details = attribution.get("details", {})
    details = details if isinstance(details, dict) else {}
    outcome = details.get("outcome", {}) if isinstance(details, dict) else {}
    primary_failure = attribution.get("primary_failure")
    analysis = candidate.get("analysis_eligibility")
    eligible = (
        row["benchmark_eligibility"] == "eligible"
        and isinstance(analysis, dict)
        and analysis.get("eligible") is True
    )
    execution_evidence = _boolean(
        row["counts_as_execution_success_evidence"], "counts_as_execution_success_evidence"
    )
    no_failure = (
        attribution.get("status") == "attributed"
        and primary_failure == "success"
        and details.get("status") == "success"
        and outcome.get("collision_event") is False
        and outcome.get("timeout_event") is False
        and outcome.get("route_complete") is True
        and row["row_status"] == "successful_evidence"
        and execution_evidence
        and status.get("counts_as_success_evidence") is True
        and status.get("fallback_or_degraded") is False
        and eligible
        and not row["error"].strip()
    )
    explicit_failure = (
        attribution.get("status") == "attributed"
        and isinstance(primary_failure, str)
        and primary_failure != "success"
        and row["availability_status"] == "available"
        and row["execution_mode"] in {"native", "adapter", "mixed"}
        and row["readiness_status"] in {"native", "adapter"}
        and row["row_status"] == "successful_evidence"
        and execution_evidence
        and status.get("counts_as_success_evidence") is True
        and status.get("fallback_or_degraded") is False
        and eligible
        and (
            outcome.get("collision_event") is True
            or outcome.get("timeout_event") is True
            or outcome.get("route_complete") is False
        )
    )
    if status.get("fallback_or_degraded") is True:
        criticality = "unknown"
        execution_outcome = "fallback_or_degraded"
    elif no_failure:
        criticality = "noncritical_success"
        execution_outcome = "successful_execution"
    elif explicit_failure:
        criticality = "critical_planner_failure"
        execution_outcome = "completed_with_attributed_failure"
    else:
        criticality = "unknown"
        execution_outcome = "evaluation_failed" if row["error"].strip() else "outcome_unresolved"

    declared_inputs = {
        "scenario_yaml": (candidate.get("scenario_yaml_path"), row["scenario_yaml_sha256"]),
        "episode_record": (candidate.get("episode_record_path"), row["episode_records_sha256"]),
    }
    missing_inputs: list[str] = []
    digest_errors: list[str] = []
    present_count = 0
    unsafe_input = False
    unknown_input_digest = False
    for name, (raw_path, expected_digest) in declared_inputs.items():
        resolved = _resolve_packet_input(raw_path, source_root)
        if resolved is None:
            unsafe_input = True
            missing_inputs.append(name)
            continue
        if not resolved.is_file():
            missing_inputs.append(name)
            continue
        present_count += 1
        actual = _sha256_file(resolved)
        if not expected_digest:
            unknown_input_digest = True
        elif actual != expected_digest.lower():
            digest_errors.append(name)
    replay_status = (
        "digest_mismatch"
        if digest_errors
        else "unknown"
        if unsafe_input or unknown_input_digest
        else (
            "available"
            if present_count == len(declared_inputs)
            else "missing"
            if present_count == 0
            else "partial"
        )
    )
    return {
        "source_candidate_id": row_id,
        "sampler": row["sampler"],
        "sampler_seed": _integer(row["sampler_seed"], "sampler_seed"),
        "evaluation_index": _integer(row["evaluation_index"], "evaluation_index"),
        "scenario_id": row["scenario_id"],
        "candidate_parameters": candidate.get("candidate"),
        "source_manifest": manifest_path,
        "source_manifest_sha256": manifest_sha256,
        "candidate_sha256": row["candidate_sha256"],
        "effective_scenario_sha256": row["effective_scenario_sha256"],
        "objective_value": _finite_number(row["objective_value"]),
        "failure_attribution": primary_failure,
        "certification_status": row["certification_status"],
        "certification_classification": row["certification_classification"],
        "execution_outcome": execution_outcome,
        "scenario_eligibility": "eligible" if eligible else "ineligible_or_unknown",
        "case_criticality": criticality,
        "outcome_metrics": {
            "collision_event": _boolean(row["collision_event"], "collision_event"),
            "route_complete": _boolean(row["route_complete"], "route_complete"),
            "timeout_event": _boolean(row["timeout_event"], "timeout_event"),
            "min_clearance_m": _csv_number(row["min_clearance_m"], "min_clearance_m"),
            "distance_to_human_min_m": _csv_number(
                row["distance_to_human_min_m"], "distance_to_human_min_m"
            ),
            "near_miss_count": _integer(row["near_miss_count"], "near_miss_count"),
            "total_collision_count": _integer(
                row["total_collision_count"], "total_collision_count"
            ),
            "ped_collision_count": _integer(row["ped_collision_count"], "ped_collision_count"),
            "steps": _integer(row["steps"], "steps"),
            "source": "candidate_evaluations.csv; raw episode inputs are not implied",
        },
        "replay_input_status": replay_status,
        "replay_input_scope": _REPLAY_INPUT_SCOPE,
        "missing_replay_inputs": missing_inputs,
        "input_digest_errors": digest_errors,
        "input_digest_status": (
            "mismatch"
            if digest_errors
            else "verified"
            if replay_status == "available"
            else "unknown"
            if unknown_input_digest or unsafe_input
            else "declared_only"
        ),
        "selection_status": "not_selected_no_critical_failure"
        if criticality == "noncritical_success"
        else "no_case_selected",
    }


def _check_row_status_counts(
    row_status: dict[str, Any], status_by_id: dict[str, dict[str, Any]]
) -> None:
    """Reconcile the status histogram in row_status.json."""
    counts = row_status.get("counts")
    if not isinstance(counts, dict):
        raise ValueError("row_status.json has no counts object")
    observed = dict(Counter(str(item.get("status")) for item in status_by_id.values()))
    if counts != observed:
        raise ValueError("row_status.json counts do not match its candidate rows")


def _check_summary(
    summary: dict[str, Any],
    by_run: dict[str, list[dict[str, Any]]],
    accounting: list[dict[str, Any]],
    *,
    source_manifests: list[dict[str, Any]],
) -> None:
    """Cross-check experiment totals and run ledger against candidate rows."""
    if summary.get("issue") != 9645:
        raise ValueError("directory packet mode accepts the #9645 compact evidence packet only")
    _check_budget_summary(summary, accounting, source_manifests)
    _check_run_summaries(summary, by_run)
    _check_reported_outcomes(summary, accounting)


def _check_budget_summary(
    summary: dict[str, Any],
    accounting: list[dict[str, Any]],
    source_manifests: list[dict[str, Any]],
) -> None:
    """Compare the declared pilot budget to the reconciled candidate outcomes."""
    budget = summary.get("pilot_budget")
    if not isinstance(budget, dict):
        raise ValueError("summary pilot budget is missing")
    attempted = _nonnegative_integer(budget.get("attempted"), "pilot attempted")
    planned = _nonnegative_integer(budget.get("planned"), "pilot planned")
    planned_from_manifests = sum(
        _nonnegative_integer(entry["payload"]["config"].get("budget"), "source manifest budget")
        for entry in source_manifests
    )
    if attempted != len(accounting) or planned != planned_from_manifests:
        raise ValueError("summary pilot budget does not match candidate ledger")
    completed_count = sum(item["execution_outcome"] != "evaluation_failed" for item in accounting)
    if _nonnegative_integer(budget.get("completed"), "pilot completed") != completed_count:
        raise ValueError("summary completed count does not match candidate outcomes")
    failed = sum(item["execution_outcome"] == "evaluation_failed" for item in accounting)
    invalid = sum(item["scenario_eligibility"] != "eligible" for item in accounting)
    unknown_or_scoreless = sum(
        item["case_criticality"] == "unknown" or item["objective_value"] is None
        for item in accounting
    )
    if (
        _nonnegative_integer(budget.get("failed"), "pilot failed") != failed
        or _nonnegative_integer(budget.get("invalid"), "pilot invalid") != invalid
        or _nonnegative_integer(budget.get("unknown_or_scoreless"), "pilot unknown_or_scoreless")
        != unknown_or_scoreless
    ):
        raise ValueError("summary failed/invalid counts do not match candidate rows")


def _check_run_summaries(summary: dict[str, Any], by_run: dict[str, list[dict[str, Any]]]) -> None:
    """Compare per-run candidate totals and best objective with the candidate ledger."""
    expected_runs = summary.get("runs")
    if not isinstance(expected_runs, list):
        raise ValueError("summary has no per-run accounting")
    run_map = {str(item.get("run_id")): item for item in expected_runs if isinstance(item, dict)}
    if len(run_map) != len(expected_runs) or set(run_map) != set(by_run):
        raise ValueError("summary run identities do not match the candidate ledger")
    for run_id, rows in by_run.items():
        run_summary = run_map[run_id]
        sampler = rows[0]["sampler"]
        seed = rows[0]["sampler_seed"]
        planned = _nonnegative_integer(
            rows[0]["source_manifest_config"]["budget"], f"source manifest budget for {run_id}"
        )
        if any(
            item["sampler"] != sampler
            or item["sampler_seed"] != seed
            or item["source_manifest_config"]["budget"] != planned
            for item in rows
        ):
            raise ValueError(f"candidate source manifest identities conflict for run {run_id}")
        expected_failed = sum(item["execution_outcome"] == "evaluation_failed" for item in rows)
        expected_invalid = sum(item["scenario_eligibility"] != "eligible" for item in rows)
        if _nonnegative_integer(
            run_summary.get("candidate_rows"), f"candidate rows for {run_id}"
        ) != len(rows):
            raise ValueError(f"summary candidate count conflicts for run {run_id}")
        if _nonnegative_integer(
            run_summary.get("attempted_evaluations"), f"attempted evaluations for {run_id}"
        ) != len(rows):
            raise ValueError(f"summary attempted count conflicts for run {run_id}")
        if (
            run_summary.get("sampler") != sampler
            or run_summary.get("sampler_seed") != seed
            or _nonnegative_integer(
                run_summary.get("planned_evaluations"), f"planned evaluations for {run_id}"
            )
            != planned
            or _nonnegative_integer(
                run_summary.get("failed_evaluations"), f"failed evaluations for {run_id}"
            )
            != expected_failed
            or _nonnegative_integer(
                run_summary.get("invalid_candidates"), f"invalid candidates for {run_id}"
            )
            != expected_invalid
        ):
            raise ValueError(f"summary run budget or outcome counts conflict for {run_id}")
        if run_summary.get("best_objective") != max(
            (item["objective_value"] for item in rows if item["objective_value"] is not None),
            default=None,
        ):
            raise ValueError(f"summary best objective conflicts for run {run_id}")


def _check_reported_outcomes(summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    """Compare summary metric and criticality totals with the reconciled candidate rows."""
    metrics = summary.get("pilot_metrics")
    candidate_outcomes = summary.get("candidate_outcomes")
    if not isinstance(metrics, dict) or not isinstance(candidate_outcomes, dict):
        raise ValueError("summary is missing candidate outcome or metric totals")
    _check_pilot_metrics(metrics, rows)
    _check_candidate_totals(summary, candidate_outcomes, rows)


def _check_pilot_metrics(metrics: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    """Compare route, collision, near-miss, distance and objective totals."""
    row_metrics = [row["outcome_metrics"] for row in rows]
    totals = {
        "collision_events": sum(item["collision_event"] for item in row_metrics),
        "timeouts": sum(item["timeout_event"] for item in row_metrics),
        "route_completions": sum(item["route_complete"] for item in row_metrics),
        "near_miss_count_total": sum(item["near_miss_count"] for item in row_metrics),
    }
    for key, value in totals.items():
        if metrics.get(key) != value:
            raise ValueError(f"summary {key} conflicts with candidate rows")
    for key in ("min_clearance_m_range", "distance_to_human_min_m_range"):
        observed = [item[key.removesuffix("_range")] for item in row_metrics]
        finite = [value for value in observed if value is not None]
        expected = [min(finite), max(finite)] if finite else None
        if metrics.get(key) != expected:
            raise ValueError(f"summary {key} conflicts with candidate rows")
    score_set = sorted(
        {row["objective_value"] for row in rows if row["objective_value"] is not None}
    )
    if metrics.get("score_set") != score_set:
        raise ValueError("summary score set conflicts with candidate rows")


def _check_candidate_totals(
    summary: dict[str, Any], outcomes: dict[str, Any], rows: list[dict[str, Any]]
) -> None:
    """Compare distinct-candidate, criticality, feasibility and certificate totals."""
    critical_count = sum(row["case_criticality"] == "critical_planner_failure" for row in rows)
    distinct_scenarios = len({row["effective_scenario_sha256"] for row in rows})
    distinct_candidates = len({row["candidate_sha256"] for row in rows})
    expected_outcomes = {
        "new_counterexamples_discovered": critical_count,
        "distinct_effective_scenarios": distinct_scenarios,
        "distinct_candidate_specs": distinct_candidates,
        "duplicate_candidate_specs": len(rows) - distinct_candidates,
    }
    for key, value in expected_outcomes.items():
        if outcomes.get(key) != value:
            raise ValueError(f"summary candidate outcome {key} conflicts with candidate rows")
    consumed = summary.get("evaluation_budget_consumed")
    if not isinstance(consumed, dict) or consumed.get("pilot_search_candidates") != len(rows):
        raise ValueError("summary search evaluation budget conflicts with candidate rows")
    feasibility = summary.get("feasibility")
    if not isinstance(feasibility, dict):
        raise ValueError("summary is missing pilot feasibility counts")
    successful = sum(row["execution_outcome"] == "successful_execution" for row in rows)
    if feasibility.get("pilot_empirical_goal_successes") != successful:
        raise ValueError("summary empirical success count conflicts with candidate rows")
    expected_certificates = dict(
        Counter(
            f"{row['certification_status']}:{row['certification_classification']}" for row in rows
        )
    )
    if feasibility.get("pilot_scenario_certification") != expected_certificates:
        raise ValueError("summary scenario certificate counts conflict with candidate rows")


def _check_convergence(report: dict[str, Any], by_run: dict[str, list[dict[str, Any]]]) -> None:
    """Cross-check each sampler aggregate and outcome count with source manifests."""
    aggregate_by_sampler = _index_convergence_aggregates(report)
    expected_by_sampler = _group_runs_by_sampler(by_run)
    if set(aggregate_by_sampler) != set(expected_by_sampler):
        raise ValueError("convergence aggregate sampler identities conflict with candidate ledger")
    for sampler, sampler_rows in expected_by_sampler.items():
        _check_sampler_aggregate(sampler, sampler_rows, aggregate_by_sampler[sampler], by_run)
    if report.get("claim_scope") is None:
        raise ValueError("convergence report has no explicit claim scope")


def _index_convergence_aggregates(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index one well-formed candidate-accounting aggregate per sampler."""
    aggregates = report.get("aggregates")
    if (
        not isinstance(aggregates, list)
        or not aggregates
        or any(not isinstance(aggregate, dict) for aggregate in aggregates)
    ):
        raise ValueError("convergence report has no aggregate rows")
    result: dict[str, dict[str, Any]] = {}
    for aggregate in aggregates:
        sampler = aggregate.get("sampler")
        if not isinstance(sampler, str) or not sampler or sampler in result:
            raise ValueError("convergence report has missing or duplicated sampler identity")
        candidate_accounting = aggregate.get("candidate_accounting")
        if not isinstance(candidate_accounting, dict):
            raise ValueError("convergence aggregate lacks candidate accounting")
        result[sampler] = candidate_accounting
    return result


def _group_runs_by_sampler(
    by_run: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    """Collect candidate rows across seeds for sampler-level convergence checks."""
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run_rows in by_run.values():
        result[run_rows[0]["sampler"]].extend(run_rows)
    return result


def _check_sampler_aggregate(
    sampler: str,
    rows: list[dict[str, Any]],
    observed: dict[str, Any],
    by_run: dict[str, list[dict[str, Any]]],
) -> None:
    """Compare sampler accounting counts and rates to candidate rows and declared budgets."""
    planned = sum(
        _nonnegative_integer(
            run_rows[0]["source_manifest_config"]["budget"],
            f"source manifest budget for {run_rows[0]['sampler']}_{run_rows[0]['sampler_seed']}",
        )
        for run_rows in by_run.values()
        if run_rows[0]["sampler"] == sampler
    )
    attempted = len(rows)
    critical = sum(item["case_criticality"] == "critical_planner_failure" for item in rows)
    failed = sum(item["execution_outcome"] == "evaluation_failed" for item in rows)
    invalid = sum(item["scenario_eligibility"] != "eligible" for item in rows)
    candidate_hashes = [item["candidate_sha256"] for item in rows]
    duplicate = len(candidate_hashes) - len(set(candidate_hashes))
    expected = {
        "attempted": attempted,
        "critical": critical,
        "failed": failed,
        "invalid": invalid,
        "missing": planned - attempted,
        "duplicate": duplicate,
        "valid_total_minus_invalid_minus_failed": attempted - invalid - failed,
    }
    for key, value in expected.items():
        if _nonnegative_integer(observed.get(key), f"{sampler} aggregate {key}") != value:
            raise ValueError(
                f"convergence aggregate {sampler} {key} conflicts with candidate ledger"
            )
    for key, value in (
        ("duplicate_rate_observed", duplicate / attempted if attempted else 0.0),
        ("invalid_rate_observed", invalid / attempted if attempted else 0.0),
    ):
        if not _same_finite_number(observed.get(key), value):
            raise ValueError(
                f"convergence aggregate {sampler} {key} conflicts with candidate ledger"
            )


def _reported_critical_count(report: dict[str, Any]) -> int:
    """Sum the stored critical count after packet aggregate validation."""
    return sum(
        _integer(item["candidate_accounting"]["critical"], "aggregate critical")
        for item in report["aggregates"]
    )


def _resolve_packet_input(raw_path: Any, source_root: Path) -> Path | None:
    """Resolve candidate files only inside the source checkout; never follow escaping paths."""
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    path = Path(raw_path)
    if path.is_absolute() or ".." in path.parts:
        return None
    resolved = (source_root / path).resolve()
    try:
        resolved.relative_to(source_root.resolve())
    except ValueError:
        return None
    return resolved


def _source_root(packet_dir: Path, fallback: Path) -> Path:
    """Find the worktree whose ignored output paths are referenced by the packet."""
    for parent in packet_dir.parents:
        if (parent / "AGENTS.md").is_file() and (parent / ".git").exists():
            return parent.resolve()
    return fallback


def _same_finite_number(left: Any, right: Any) -> bool:
    """Compare parsed finite values without treating booleans as numbers."""
    if isinstance(left, bool) or isinstance(right, bool):
        return False
    left_missing = left is None or (isinstance(left, str) and not left.strip())
    right_missing = right is None or (isinstance(right, str) and not right.strip())
    if left_missing or right_missing:
        return left_missing and right_missing
    try:
        left_value = float(left)
        right_value = float(right)
    except (TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(left_value) and math.isfinite(right_value) and left_value == right_value


def _is_git_revision(value: Any) -> bool:
    """Return whether a claimed source revision is a full hexadecimal Git object ID."""
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def _finite_number(value: Any) -> float | None:
    """Parse a finite numeric CSV value."""
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) else None


def _csv_number(value: Any, name: str) -> float | None:
    """Parse one finite numeric CSV metric, preserving unavailable values as null."""
    parsed = _finite_number(value)
    if parsed is None and str(value).strip():
        raise ValueError(f"{name} must be a finite numeric value or empty")
    return parsed


def _integer(value: Any, name: str) -> int:
    """Parse an integer count without accepting booleans or fractional numbers."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if str(value).strip() not in {str(parsed), f"{parsed}.0"}:
        raise ValueError(f"{name} must be an integer")
    return parsed


def _nonnegative_integer(value: Any, name: str) -> int:
    """Parse a nonnegative integer count without accepting booleans or fractions."""
    parsed = _integer(value, name)
    if parsed < 0:
        raise ValueError(f"{name} must be nonnegative")
    return parsed


def _boolean(value: Any, name: str) -> bool:
    """Parse a CSV boolean strictly."""
    if value is True or value == "True":
        return True
    if value is False or value == "False":
        return False
    raise ValueError(f"{name} must be the literal True or False")


def _sha256_file(path: Path) -> str:
    """Hash a source file for replay-input availability verification."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_json(value: Any) -> str:
    """Encode canonical candidate JSON exactly as the search ledger does."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _write_packet_readme(
    output_dir: Path,
    summary: dict[str, Any],
    missing_input_count: int,
    selection_status: str,
) -> None:
    """Write a short explanation of the reconciled no-case result."""
    lines = [
        "# Adversarial replay gallery",
        "",
        "Generated by reconciling a compact multi-run search evidence packet.",
        "",
        f"Candidates accounted: {summary['source_candidate_count']}",
        f"Candidates planned: {summary['planned_candidate_count']}",
        f"Planned candidates missing from evidence: {summary['missing_planned_candidate_count']}",
        f"Attributed critical planner failures: {summary['critical_case_count']}",
        f"Criticality-unknown rows: {summary['criticality_unknown_count']}",
        f"Scoreless candidate rows: {summary['scoreless_candidate_count']}",
        f"Candidates selected for replay: {summary['selected_case_count']}",
        f"Candidate rows with missing raw replay inputs: {missing_input_count}",
        f"Selection status: `{selection_status}`",
        "",
        "Scenario eligibility and case criticality are separate fields. Missing replay inputs do not",
        "mean a case is infeasible. This packet result is bounded to its recorded search budget; it",
        "does not establish that no counterexample exists.",
        "",
        "Historical compatibility replays are separate from these search candidates.",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


__all__ = ["build_search_evidence_packet_gallery"]
