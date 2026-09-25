"""Build deterministic convergence reports from persisted adversarial-search runs.

The report consumes the existing ``adversarial-sampler-comparison.v3`` index and
``adversarial-search-manifest.v1`` run manifests. It never launches a search or a
simulator. All aggregations are descriptive; a finite search budget cannot establish
that no counterexample exists.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

COMPARISON_SCHEMA = "adversarial-sampler-comparison.v3"
SEARCH_SCHEMA = "adversarial-search-manifest.v1"
REPORT_SCHEMA = "adversarial-search-convergence-report.v2"
CRITICAL_FAILURES = frozenset(
    {
        "collision",
        "severe_intrusion",
        "timeout",
        "near_miss",
        "comfort_violation",
        "incomplete",
    }
)
_FAILURE_STATUSES = frozenset({"invalid_candidate", "evaluation_error"})
_EXECUTION_CONTEXT_SCHEMA = "adversarial_execution_context.v1"
_FULL_COMMIT_SHA = re.compile(r"[0-9a-fA-F]{40}\Z")
_FULL_SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")
_MANIFEST_PATH_MAP_SCHEMA = "falsification_manifest_path_map.v1"


def _is_full_commit_sha(value: Any) -> bool:
    return isinstance(value, str) and _FULL_COMMIT_SHA.fullmatch(value.strip()) is not None


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bytes_sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _is_full_sha256(value: Any) -> bool:
    return isinstance(value, str) and _FULL_SHA256.fullmatch(value) is not None


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON input {path}: {exc}") from exc


def _validate_manifest_path_map_header(payload: Any, comparison_sha256: str) -> list[Any]:
    if not isinstance(payload, dict):
        raise ValueError("manifest path map must be a JSON object")
    required_fields = {"schema_version", "source_comparison_sha256", "bindings"}
    if set(payload) != required_fields:
        raise ValueError(
            "manifest path map fields must be exactly schema_version, "
            "source_comparison_sha256, and bindings"
        )
    if payload.get("schema_version") != _MANIFEST_PATH_MAP_SCHEMA:
        raise ValueError(f"manifest path map schema must be {_MANIFEST_PATH_MAP_SCHEMA}")
    source_digest = payload.get("source_comparison_sha256")
    if not _is_full_sha256(source_digest):
        raise ValueError("manifest path map source_comparison_sha256 must be a full SHA-256")
    if source_digest.lower() != comparison_sha256:
        raise ValueError("manifest path map is not bound to the exact comparison input bytes")
    bindings = payload.get("bindings")
    if not isinstance(bindings, list):
        raise ValueError("manifest path map bindings must be an array")
    return bindings


def _manifest_path_map_binding_fields(
    binding: Any, index: int, comparison_paths: set[str]
) -> tuple[str, str, str]:
    if not isinstance(binding, dict):
        raise ValueError(f"manifest path map binding {index} must be an object")
    required = {"declared_manifest_path", "archived_manifest_path", "archived_manifest_sha256"}
    if not required.issubset(binding) or not set(binding).issubset(required | {"path"}):
        raise ValueError(
            f"manifest path map binding {index} fields must include declared_manifest_path, "
            "archived_manifest_path, and archived_manifest_sha256; optional path must be "
            "identical to archived_manifest_path"
        )
    declared_path = binding.get("declared_manifest_path")
    if not isinstance(declared_path, str) or not declared_path.strip():
        raise ValueError(
            f"manifest path map binding {index} declared_manifest_path must be a nonempty string"
        )
    if declared_path not in comparison_paths:
        raise ValueError(
            f"manifest path map binding {index} does not match a declared comparison path"
        )
    archived_path = binding.get("archived_manifest_path")
    if not isinstance(archived_path, str) or not archived_path.strip():
        raise ValueError(
            f"manifest path map binding {index} archived_manifest_path must be nonempty"
        )
    if "path" in binding and binding["path"] != archived_path:
        raise ValueError(
            f"manifest path map binding {index} path must equal archived_manifest_path"
        )
    expected_sha256 = binding.get("archived_manifest_sha256")
    if not _is_full_sha256(expected_sha256):
        raise ValueError(
            f"manifest path map binding {index} archived_manifest_sha256 must be a full SHA-256"
        )
    return declared_path, archived_path, expected_sha256.lower()


def _resolve_archived_manifest_path(archived_path: str, index: int, repo_root: Path) -> Path:
    posix_path = PurePosixPath(archived_path)
    windows_path = PureWindowsPath(archived_path)
    unsafe = (
        "\\" in archived_path
        or posix_path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or posix_path.as_posix() != archived_path
        or any(part in {"", ".", ".."} for part in posix_path.parts)
    )
    if unsafe:
        raise ValueError(
            f"manifest path map binding {index} archived_manifest_path must be a normalized "
            "repository-relative path without traversal"
        )
    root = repo_root.resolve()
    try:
        archived_file = root.joinpath(*posix_path.parts).resolve(strict=True)
    except OSError as exc:
        raise ValueError(
            f"manifest path map binding {index} archived file is unavailable: {archived_path}"
        ) from exc
    except (RuntimeError, ValueError) as exc:
        raise ValueError(
            f"manifest path map binding {index} archived path cannot be resolved safely"
        ) from exc
    try:
        archived_file.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"manifest path map binding {index} archived file resolves outside repo_root"
        ) from exc
    if not archived_file.is_file():
        raise ValueError(f"manifest path map binding {index} archived target is not a regular file")
    return archived_file


def _load_manifest_path_map(
    path: Path,
    *,
    comparison_sha256: str,
    rows: list[dict[str, Any]],
    repo_root: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Load and verify archived-manifest bindings without changing comparison rows."""
    try:
        content = path.read_bytes()
        payload = json.loads(content.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read manifest path map {path}: {exc}") from exc
    raw_bindings = _validate_manifest_path_map_header(payload, comparison_sha256)
    comparison_paths = {row["manifest_path"] for row in rows}
    bindings: dict[str, dict[str, Any]] = {}
    seen_archives: set[Path] = set()
    for index, raw_binding in enumerate(raw_bindings, start=1):
        declared_path, archived_path, expected_sha256 = _manifest_path_map_binding_fields(
            raw_binding, index, comparison_paths
        )
        if declared_path in bindings:
            raise ValueError(f"manifest path map has duplicate declared path key: {declared_path}")
        archived_file = _resolve_archived_manifest_path(archived_path, index, repo_root)
        if archived_file in seen_archives:
            raise ValueError(
                f"manifest path map has duplicate or ambiguous archived target: {archived_path}"
            )
        seen_archives.add(archived_file)
        observed_sha256 = _file_sha256(archived_file)
        if observed_sha256 != expected_sha256:
            raise ValueError(
                f"manifest path map binding {index} archived file SHA-256 does not match"
            )
        bindings[declared_path] = {
            "declared_manifest_path": declared_path,
            "archived_manifest_path": archived_path,
            "archived_manifest_sha256": observed_sha256,
            "path": archived_file,
        }

    provenance = {
        "schema_version": _MANIFEST_PATH_MAP_SCHEMA,
        "path": _portable_path(path.resolve(), repo_root=repo_root),
        "sha256": _bytes_sha256(content),
        "source_comparison_sha256": comparison_sha256,
        "binding_count": len(bindings),
    }
    return bindings, provenance


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return None
    return value


def _integer(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _resolve_path(value: Any, *, anchors: tuple[Path, ...]) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    raw = Path(value)
    candidates = (raw,) if raw.is_absolute() else tuple(anchor / raw for anchor in anchors)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0].resolve(strict=False) if candidates else None


def _portable_path(path: Path, *, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _sampler_label(value: str) -> str:
    normalized = value.strip().lower()
    if normalized == "random":
        return "Random"
    if normalized in {"optuna", "tpe"}:
        return "TPE"
    return value.strip() or "unknown"


def _candidate_failure(item: dict[str, Any]) -> str | None:
    attribution = item.get("failure_attribution")
    if isinstance(attribution, dict):
        value = attribution.get("primary_failure")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _candidate_status(item: Any) -> tuple[str, str | None, float | None, bool]:
    """Return (status, failure, score, critical) using candidate-level evidence."""
    if not isinstance(item, dict):
        return "invalid", "candidate_row_malformed", None, False
    certification = item.get("certification_status")
    cert_status = (
        str(certification.get("status", "")).strip().lower()
        if isinstance(certification, dict)
        else ""
    )
    failure = _candidate_failure(item)
    error = item.get("error")
    has_error = error is not None and error != ""
    score = _finite_number(item.get("objective_value"))
    if cert_status != "passed" or failure == "invalid_candidate":
        return "invalid", failure or "certification_not_passed", score, False
    if not isinstance(item.get("candidate"), dict):
        return "invalid", "candidate_spec_missing", score, False
    if has_error or failure == "evaluation_error":
        return "failed", failure or "evaluation_error", score, False
    critical = failure in CRITICAL_FAILURES
    if score is None:
        return "scoreless", failure, None, critical
    return "scored", failure, score, critical


def _analysis_eligibility(item: dict[str, Any]) -> bool | None:
    receipt = item.get("analysis_eligibility")
    if isinstance(receipt, dict) and isinstance(receipt.get("eligible"), bool):
        return receipt["eligible"]
    return None


def _status_reason_token(value: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    return token or "unknown"


def _analysis_evidence_ineligibility_reasons(
    item: dict[str, Any], *, status: str, episode_evidence: dict[str, Any] | None
) -> list[str]:
    """Explain why an observed row does not meet artifact-verified eligibility."""
    reasons: list[str] = []
    if status != "scored":
        reasons.append(f"candidate_status_{_status_reason_token(status)}")
    eligibility = _analysis_eligibility(item)
    if eligibility is not True:
        reasons.append(
            "analysis_eligibility_unknown" if eligibility is None else "analysis_eligibility_false"
        )

    execution_requirements = (
        ("execution_mode", _execution_mode(item), "native"),
        ("readiness_status", _execution_status(item, "readiness_status"), "native"),
        ("availability_status", _execution_status(item, "availability_status"), "available"),
    )
    for field, observed, required in execution_requirements:
        if observed == required:
            continue
        if observed is None:
            reasons.append(f"{field}_missing")
        else:
            reasons.append(f"{field}_{_status_reason_token(observed)}")

    effective_hash = item.get("effective_scenario_hash")
    if not isinstance(effective_hash, str) or not effective_hash.strip():
        reasons.append("effective_scenario_hash_missing")

    episode_status = (
        episode_evidence.get("artifact_status") if isinstance(episode_evidence, dict) else None
    )
    if episode_status != "available":
        token = (
            _status_reason_token(episode_status) if isinstance(episode_status, str) else "unknown"
        )
        reasons.append(f"episode_record_{token}")
    return sorted(set(reasons))


def _execution_mode(item: dict[str, Any]) -> str | None:
    attribution = item.get("failure_attribution")
    details = attribution.get("details") if isinstance(attribution, dict) else None
    if not isinstance(details, dict):
        return None
    value = details.get("execution_mode")
    return value.strip().lower() if isinstance(value, str) and value.strip() else None


def _execution_status(item: dict[str, Any], key: str) -> str | None:
    attribution = item.get("failure_attribution")
    details = attribution.get("details") if isinstance(attribution, dict) else None
    value = details.get(key) if isinstance(details, dict) else None
    return value.strip().lower() if isinstance(value, str) and value.strip() else None


def _execution_risk_mode(item: dict[str, Any]) -> str | None:
    for key in ("execution_mode", "readiness_status", "availability_status"):
        value = _execution_status(item, key)
        if value in {"fallback", "degraded"}:
            return value
    return None


def _candidate_identity(item: dict[str, Any]) -> tuple[str | None, str | None]:
    candidate = item.get("candidate")
    candidate_hash = _canonical_sha256(candidate) if isinstance(candidate, dict) else None
    effective_hash = item.get("effective_scenario_hash")
    effective_hash = (
        effective_hash.strip()
        if isinstance(effective_hash, str) and effective_hash.strip()
        else None
    )
    return candidate_hash, effective_hash


def _missing_evaluation(index: int, *, within_budget: bool) -> dict[str, Any]:
    return {
        "evaluation_index": index,
        "status": "missing",
        "failure_type": None,
        "objective_value": None,
        "critical": False,
        "observed_critical": False,
        "candidate": None,
        "candidate_sha256": None,
        "effective_scenario_hash": None,
        "duplicate_of_evaluation": None,
        "duplicate_basis": [],
        "analysis_eligible": None,
        "execution_mode": None,
        "readiness_status": None,
        "availability_status": None,
        "episode_record_status": "not_evaluated",
        "episode_record_sha256": None,
        "analysis_evidence_eligible": False,
        "analysis_evidence_ineligibility_reason_codes": ["candidate_evaluation_missing"],
        "within_budget": within_budget,
        "error": "candidate evaluation was expected by the recorded budget but is absent",
        "best_so_far_objective": None,
        "best_so_far_observed_objective": None,
        "best_so_far_analysis_eligible_objective": None,
    }


def _read_execution_context(manifest_path: Path) -> dict[str, Any] | None:
    for filename in ("execution_context.txt", "execution_context.json"):
        path = manifest_path.parent / filename
        if not path.is_file():
            continue
        try:
            payload = _load_json(path)
        except ValueError:
            return {"status": "malformed", "path": filename}
        if not isinstance(payload, dict):
            return {"status": "malformed", "path": filename}
        return {"status": "available", "path": filename, "payload": payload}
    return None


def _commit_value(record: dict[str, Any]) -> str | None:
    candidates: list[Any] = [record.get("git_hash")]
    provenance = record.get("provenance")
    if isinstance(provenance, dict):
        candidates.extend((provenance.get("commit_hash"), provenance.get("commit_sha")))
    for raw in candidates:
        if isinstance(raw, str) and raw.strip().lower() not in {
            "",
            "unknown",
            "unavailable",
            "none",
        }:
            return raw.strip()
    return None


def _episode_record_evidence(
    item: Any,
    evaluation_index: int,
    *,
    manifest_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    raw_path = item.get("episode_record_path") if isinstance(item, dict) else None
    if not isinstance(raw_path, str) or not raw_path.strip():
        return {
            "evaluation_index": evaluation_index,
            "path": None,
            "sha256": None,
            "commit_sha": None,
            "commit_identifier": None,
            "status": "missing_path",
            "artifact_status": "missing_path",
        }
    resolved = _resolve_path(raw_path, anchors=(repo_root, manifest_path.parent, Path.cwd()))
    if resolved is None or not resolved.is_file():
        return {
            "evaluation_index": evaluation_index,
            "path": raw_path,
            "sha256": None,
            "commit_sha": None,
            "commit_identifier": None,
            "status": "missing",
            "artifact_status": "missing",
        }
    digest = hashlib.sha256()
    try:
        with resolved.open("rb") as handle:
            first_record_line = b""
            for line in handle:
                digest.update(line)
                if line.strip():
                    first_record_line = line
                    break
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return {
            "evaluation_index": evaluation_index,
            "path": _portable_path(resolved, repo_root=repo_root),
            "sha256": None,
            "commit_sha": None,
            "commit_identifier": None,
            "status": "commit_unknown",
            "artifact_status": "unreadable",
        }
    try:
        parsed = json.loads(first_record_line.decode("utf-8")) if first_record_line else None
    except (UnicodeError, json.JSONDecodeError):
        parsed = None
    if not isinstance(parsed, dict):
        return {
            "evaluation_index": evaluation_index,
            "path": _portable_path(resolved, repo_root=repo_root),
            "sha256": digest.hexdigest(),
            "commit_sha": None,
            "commit_identifier": None,
            "status": "commit_unknown",
            "artifact_status": "malformed",
        }
    identifier = _commit_value(parsed)
    commit = identifier.lower() if _is_full_commit_sha(identifier) else None
    provenance_status = (
        "available" if commit else "commit_unverified" if identifier else "commit_unknown"
    )
    return {
        "evaluation_index": evaluation_index,
        "path": _portable_path(resolved, repo_root=repo_root),
        "sha256": digest.hexdigest(),
        "commit_sha": commit,
        "commit_identifier": identifier,
        "status": provenance_status,
        "artifact_status": "available",
    }


def _execution_context_revision(
    context: dict[str, Any] | None,
) -> tuple[str | None, str | None]:
    payload = context.get("payload") if isinstance(context, dict) else None
    if not isinstance(payload, dict):
        return None, None
    identifier = payload.get("commit_sha")
    identifier = identifier.strip() if isinstance(identifier, str) and identifier.strip() else None
    exact = (
        identifier.lower()
        if payload.get("schema_version") == _EXECUTION_CONTEXT_SCHEMA
        and _is_full_commit_sha(identifier)
        else None
    )
    return exact, identifier if identifier and exact is None else None


def _episode_provenance(
    candidate_items: list[Any], *, manifest_path: Path, repo_root: Path
) -> dict[str, Any]:
    evidence_by_evaluation: dict[int, dict[str, Any]] = {}
    evidence_rows: list[dict[str, Any]] = []
    commits: set[str] = set()
    unverified: set[str] = set()
    unresolved = 0
    for index, item in enumerate(candidate_items, start=1):
        evidence = _episode_record_evidence(
            item, index, manifest_path=manifest_path, repo_root=repo_root
        )
        evidence_by_evaluation[index] = evidence
        if evidence.get("path") is None:
            continue
        evidence_rows.append(evidence)
        commit = evidence.get("commit_sha")
        if commit:
            commits.add(commit)
        identifier = evidence.get("commit_identifier")
        if identifier and not _is_full_commit_sha(identifier):
            unverified.add(identifier)
        unresolved += int(commit is None)
    return {
        "by_evaluation": evidence_by_evaluation,
        "records": evidence_rows,
        "commit_shas": commits,
        "unresolved_count": unresolved,
        "unverified_commit_identifiers": unverified,
    }


def _source_evidence(
    manifest_path: Path,
    episode_provenance: dict[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    context = _read_execution_context(manifest_path)
    context_payload = context.get("payload") if isinstance(context, dict) else None
    context_sha, unverified_context_identifier = _execution_context_revision(context)
    environment = None
    if isinstance(context_payload, dict):
        environment = {
            key: context_payload.get(key)
            for key in ("hostname", "cpu_model", "python_version", "platform", "thread_env")
            if context_payload.get(key) is not None
        }
    if isinstance(context, dict) and isinstance(context_payload, dict):
        context = {
            **context,
            "revision_status": "exact"
            if context_sha
            else "unverified"
            if unverified_context_identifier
            else "unknown",
        }

    episode_evidence = episode_provenance["records"]
    commits = episode_provenance["commit_shas"]
    unresolved_refs = episode_provenance["unresolved_count"]
    unverified_commit_identifiers = set(episode_provenance["unverified_commit_identifiers"])
    if unverified_context_identifier:
        unverified_commit_identifiers.add(unverified_context_identifier)

    all_commits = set(commits)
    if context_sha:
        all_commits.add(context_sha)
    if len(all_commits) > 1:
        status, exact_revision = "conflicting", None
    elif context_sha:
        status, exact_revision = "execution_context", context_sha
    elif len(commits) == 1 and unresolved_refs == 0 and episode_evidence:
        status, exact_revision = "consistent_episode_records", next(iter(commits))
    elif len(commits) == 1:
        status, exact_revision = "partial_episode_records", None
    else:
        status, exact_revision = "unknown", None

    return {
        "status": status,
        "exact_source_revision": exact_revision,
        "observed_commit_shas": sorted(all_commits),
        "unverified_commit_identifiers": sorted(unverified_commit_identifiers),
        "unresolved_episode_provenance_count": unresolved_refs,
        "execution_context": context,
        "environment": environment,
        "episode_records": episode_evidence,
    }


def _config_provenance(config: Any, *, manifest_path: Path, repo_root: Path) -> dict[str, Any]:
    if not isinstance(config, dict):
        return {"status": "missing", "config_sha256": None, "files": []}
    file_keys = (
        "scenario_template",
        "search_space_path",
        "algo_config_path",
        "snqi_weights_path",
        "snqi_baseline_path",
    )
    files: list[dict[str, Any]] = []
    for key in file_keys:
        value = config.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        path = _resolve_path(value, anchors=(repo_root, manifest_path.parent, Path.cwd()))
        files.append(
            {
                "config_key": key,
                "declared_path": value,
                "resolved_path": _portable_path(path, repo_root=repo_root) if path else None,
                "sha256": _file_sha256(path) if path and path.is_file() else None,
                "status": "available" if path and path.is_file() else "missing",
            }
        )
    result = {
        "status": "available",
        "config_sha256": _canonical_sha256(config),
        "search_space_sha256": _canonical_sha256(config["search_space"])
        if isinstance(config.get("search_space"), dict)
        else None,
        "files": files,
    }
    result["comparison_identity_sha256"] = _comparison_config_identity(config, files)
    return result


def _comparison_config_identity(config: dict[str, Any], files: list[dict[str, Any]]) -> str | None:
    """Hash shared scenario/planner inputs, excluding only per-run output, seed, and budget."""
    required = {
        "policy",
        "scenario_template",
        "search_space_path",
        "search_space",
        "objective",
        "benchmark_profile",
    }
    if not required.issubset(config) or not isinstance(config.get("search_space"), dict):
        return None

    file_digests = {
        item["config_key"]: item.get("sha256")
        for item in files
        if isinstance(item, dict) and isinstance(item.get("config_key"), str)
    }
    path_keys = {
        "scenario_template",
        "search_space_path",
        "algo_config_path",
        "snqi_weights_path",
        "snqi_baseline_path",
    }
    normalized: dict[str, Any] = {}
    for key, value in config.items():
        if key in {"output_dir", "seed", "budget"}:
            continue
        if key in path_keys:
            if value is None or value == "":
                normalized[key] = None
                continue
            digest = file_digests.get(key)
            if not digest:
                return None
            normalized[key] = {"sha256": digest}
        else:
            normalized[key] = value
    return _canonical_sha256(normalized)


def _runtime_seconds(
    row: dict[str, Any], manifest: dict[str, Any] | None
) -> tuple[float | None, str]:
    candidates: tuple[tuple[str, Any], ...] = (
        (
            "search_manifest.runtime_seconds",
            manifest.get("runtime_seconds") if isinstance(manifest, dict) else None,
        ),
        (
            "search_manifest.summary.runtime_seconds",
            manifest.get("summary", {}).get("runtime_seconds")
            if isinstance(manifest, dict) and isinstance(manifest.get("summary"), dict)
            else None,
        ),
    )
    for source, value in candidates:
        parsed = _finite_number(value)
        if parsed is not None and parsed >= 0:
            return parsed, source
    return None, "not_recorded"


def _load_manifest_artifact(
    manifest_ref: str,
    *,
    comparison_path: Path,
    repo_root: Path,
    path_binding: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if path_binding is not None:
        path = path_binding["path"]
        root = repo_root.resolve()
        try:
            current_path = path.resolve(strict=True)
            current_path.relative_to(root)
            content = current_path.read_bytes()
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"mapped archived manifest is unavailable or outside repo_root: "
                f"{path_binding['archived_manifest_path']}"
            ) from exc
        observed_sha256 = _bytes_sha256(content)
        if observed_sha256 != path_binding["archived_manifest_sha256"]:
            raise ValueError(
                "mapped archived manifest SHA-256 changed after path-map validation: "
                f"{path_binding['archived_manifest_path']}"
            )
        try:
            payload = json.loads(content.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"mapped archived manifest is not valid JSON: "
                f"{path_binding['archived_manifest_path']}"
            ) from exc
        if not isinstance(payload, dict) or payload.get("schema_version") != SEARCH_SCHEMA:
            raise ValueError(
                f"mapped archived manifest schema must be {SEARCH_SCHEMA}: "
                f"{path_binding['archived_manifest_path']}"
            )
        return {
            "path": current_path,
            "payload": payload,
            "sha256": observed_sha256,
            "status": "available",
            "warnings": [],
            "path_binding": path_binding,
        }

    path = _resolve_path(
        manifest_ref,
        anchors=(repo_root, comparison_path.parent, Path.cwd()),
    )
    warnings: list[str] = []
    if path is None or not path.is_file():
        warnings.append(f"search manifest is missing: {manifest_ref}")
        return {
            "path": path,
            "payload": None,
            "sha256": None,
            "status": "missing",
            "warnings": warnings,
            "path_binding": None,
        }
    try:
        payload = _load_json(path)
    except ValueError as exc:
        warnings.append(str(exc))
        return {
            "path": path,
            "payload": None,
            "sha256": None,
            "status": "malformed",
            "warnings": warnings,
            "path_binding": None,
        }
    if not isinstance(payload, dict) or payload.get("schema_version") != SEARCH_SCHEMA:
        warnings.append(f"search manifest schema must be {SEARCH_SCHEMA}")
        return {
            "path": path,
            "payload": None,
            "sha256": None,
            "status": "malformed",
            "warnings": warnings,
            "path_binding": None,
        }
    return {
        "path": path,
        "payload": payload,
        "sha256": _file_sha256(path),
        "status": "available",
        "warnings": warnings,
        "path_binding": None,
    }


def _derive_evaluations(
    candidates: list[Any],
    *,
    expected_slots: int,
    summary_budget: int,
    episode_evidence_by_index: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    evaluations: list[dict[str, Any]] = []
    best_budgeted_observed: float | None = None
    best_all_observed: float | None = None
    best_budgeted_eligible: float | None = None
    best_all_eligible: float | None = None
    seen_candidates: dict[str, int] = {}
    seen_effective: dict[str, int] = {}
    for index, item in enumerate(candidates, start=1):
        status, failure, score, critical = _candidate_status(item)
        is_mapping = isinstance(item, dict)
        candidate = item.get("candidate") if is_mapping else None
        candidate_hash, effective_hash = _candidate_identity(item) if is_mapping else (None, None)
        duplicate_basis: list[str] = []
        prior_indices: list[int] = []
        if candidate_hash and candidate_hash in seen_candidates:
            duplicate_basis.append("candidate_spec")
            prior_indices.append(seen_candidates[candidate_hash])
        if effective_hash and effective_hash in seen_effective:
            duplicate_basis.append("effective_scenario_hash")
            prior_indices.append(seen_effective[effective_hash])
        if candidate_hash:
            seen_candidates.setdefault(candidate_hash, index)
        if effective_hash:
            seen_effective.setdefault(effective_hash, index)

        eligible = _analysis_eligibility(item) if is_mapping else None
        episode_evidence = episode_evidence_by_index.get(index)
        eligibility_reasons = (
            _analysis_evidence_ineligibility_reasons(
                item, status=status, episode_evidence=episode_evidence
            )
            if is_mapping
            else ["candidate_record_malformed"]
        )
        analysis_evidence_eligible = not eligibility_reasons
        within_budget = index <= summary_budget
        if status == "scored" and score is not None:
            best_all_observed = (
                score if best_all_observed is None else max(best_all_observed, score)
            )
            if within_budget:
                best_budgeted_observed = (
                    score if best_budgeted_observed is None else max(best_budgeted_observed, score)
                )
            if analysis_evidence_eligible:
                best_all_eligible = (
                    score if best_all_eligible is None else max(best_all_eligible, score)
                )
                if within_budget:
                    best_budgeted_eligible = (
                        score
                        if best_budgeted_eligible is None
                        else max(best_budgeted_eligible, score)
                    )
        error = item.get("error") if is_mapping else None
        evaluation = {
            "evaluation_index": index,
            "status": status,
            "failure_type": failure,
            "objective_value": score,
            "critical": critical and analysis_evidence_eligible and within_budget,
            "observed_critical": critical,
            "candidate": candidate,
            "candidate_sha256": candidate_hash,
            "effective_scenario_hash": effective_hash,
            "duplicate_of_evaluation": min(prior_indices) if prior_indices else None,
            "duplicate_basis": duplicate_basis,
            "analysis_eligible": eligible,
            "execution_mode": _execution_mode(item) if is_mapping else None,
            "readiness_status": _execution_status(item, "readiness_status") if is_mapping else None,
            "availability_status": _execution_status(item, "availability_status")
            if is_mapping
            else None,
            "episode_record_status": (
                episode_evidence.get("artifact_status", "unknown")
                if isinstance(episode_evidence, dict)
                else "unknown"
            ),
            "episode_record_sha256": (
                episode_evidence.get("sha256") if isinstance(episode_evidence, dict) else None
            ),
            "analysis_evidence_eligible": analysis_evidence_eligible,
            "analysis_evidence_ineligibility_reason_codes": eligibility_reasons,
            "within_budget": within_budget,
            "error": error if isinstance(error, str) else str(error) if error is not None else None,
            "best_so_far_objective": best_budgeted_observed,
            "best_so_far_observed_objective": best_all_observed,
            "best_so_far_analysis_eligible_objective": best_budgeted_eligible,
            "_raw": item,
        }
        evaluations.append(evaluation)
    actual_count = len(candidates)
    evaluations.extend(
        _missing_evaluation(index, within_budget=index <= summary_budget)
        for index in range(actual_count + 1, expected_slots + 1)
    )
    observed = evaluations[:actual_count]
    budgeted = [item for item in observed if item["within_budget"]]
    over_budget = [item for item in observed if not item["within_budget"]]
    invalid = sum(item["status"] == "invalid" for item in observed)
    failed = sum(item["status"] == "failed" for item in observed)
    budgeted_invalid = sum(item["status"] == "invalid" for item in budgeted)
    budgeted_failed = sum(item["status"] == "failed" for item in budgeted)
    critical_types = Counter(
        str(item["failure_type"]) for item in budgeted if item["critical"] and item["failure_type"]
    )
    observed_critical_types = Counter(
        str(item["failure_type"])
        for item in observed
        if item["observed_critical"] and item["failure_type"]
    )
    budgeted_observed_critical_types = Counter(
        str(item["failure_type"])
        for item in budgeted
        if item["observed_critical"] and item["failure_type"]
    )
    return {
        "evaluations": evaluations,
        "num_candidates": actual_count,
        "num_budgeted_candidates": len(budgeted),
        "num_over_budget_candidates": len(over_budget),
        "num_over_budget_critical_candidates": sum(
            item["observed_critical"] for item in over_budget
        ),
        "num_missing_evaluations": sum(item["status"] == "missing" for item in evaluations),
        "num_missing_budgeted_evaluations": sum(
            item["status"] == "missing" and item["within_budget"] for item in evaluations
        ),
        "num_valid_candidates": len(budgeted) - budgeted_invalid - budgeted_failed,
        "num_invalid_candidates": budgeted_invalid,
        "num_failed_evaluations": budgeted_failed,
        "num_observed_valid_candidates": actual_count - invalid - failed,
        "num_observed_invalid_candidates": invalid,
        "num_observed_failed_evaluations": failed,
        "num_scored_valid_candidates": sum(item["status"] == "scored" for item in budgeted),
        "num_scoreless_valid_candidates": sum(item["status"] == "scoreless" for item in budgeted),
        "num_critical_candidates": sum(item["critical"] for item in budgeted),
        "num_observed_critical_candidates": sum(item["observed_critical"] for item in observed),
        "num_budgeted_observed_critical_candidates": sum(
            item["observed_critical"] for item in budgeted
        ),
        "first_critical_evaluation": next(
            (item["evaluation_index"] for item in budgeted if item["critical"]), None
        ),
        "first_observed_critical_evaluation": next(
            (item["evaluation_index"] for item in observed if item["observed_critical"]), None
        ),
        "critical_failure_counts": dict(sorted(critical_types.items())),
        "observed_critical_failure_counts": dict(sorted(observed_critical_types.items())),
        "budgeted_observed_critical_failure_counts": dict(
            sorted(budgeted_observed_critical_types.items())
        ),
        "num_duplicate_candidates": sum(bool(item["duplicate_basis"]) for item in budgeted),
        "num_candidate_spec_duplicates": sum(
            "candidate_spec" in item["duplicate_basis"] for item in budgeted
        ),
        "num_effective_scenario_duplicates": sum(
            "effective_scenario_hash" in item["duplicate_basis"] for item in budgeted
        ),
        "duplicate_rate_observed": sum(bool(item["duplicate_basis"]) for item in budgeted)
        / len(budgeted)
        if budgeted
        else None,
        "invalid_rate_observed": budgeted_invalid / len(budgeted) if budgeted else None,
        "analysis_eligible_count": sum(item["analysis_evidence_eligible"] for item in budgeted),
        "analysis_ineligible_count": sum(item["analysis_eligible"] is False for item in budgeted),
        "analysis_eligibility_unknown_count": sum(
            item["analysis_eligible"] is None for item in budgeted
        ),
        "execution_mode_counts": _observed_status_counts(budgeted, "execution_mode"),
        "readiness_status_counts": _observed_status_counts(budgeted, "readiness_status"),
        "availability_status_counts": _observed_status_counts(budgeted, "availability_status"),
        "fallback_candidate_count": sum(
            _execution_risk_mode(item["_raw"]) == "fallback" for item in budgeted
        ),
        "degraded_candidate_count": sum(
            _execution_risk_mode(item["_raw"]) == "degraded" for item in budgeted
        ),
        "best_objective_value": best_budgeted_observed,
        "best_observed_objective_value": best_all_observed,
        "best_analysis_eligible_objective_value": best_budgeted_eligible,
        "best_observed_analysis_eligible_objective_value": best_all_eligible,
    }


def _observed_status_counts(evaluations: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts = Counter(item[key] or "unknown" for item in evaluations)
    return dict(sorted(counts.items()))


def _legacy_summary(
    manifest: dict[str, Any] | None,
    derived: dict[str, Any],
    warnings: list[str],
) -> dict[str, int | None]:
    summary = manifest.get("summary", {}) if isinstance(manifest, dict) else {}
    summary = summary if isinstance(summary, dict) else {}
    keys = (
        "num_candidates",
        "num_valid_candidates",
        "num_invalid_candidates",
        "num_failed_evaluations",
    )
    legacy = {key: _integer(summary.get(key)) for key in keys}
    checks = {
        "num_candidates": derived["num_candidates"],
        "num_invalid_candidates": derived["num_observed_invalid_candidates"],
        "num_failed_evaluations": derived["num_observed_failed_evaluations"],
        "num_valid_candidates": derived["num_observed_valid_candidates"],
    }
    for key, actual in checks.items():
        if legacy[key] is not None and legacy[key] != actual:
            detail = (
                "legacy manifest summary num_valid_candidates disagrees with candidate-level "
                "total-minus-invalid-minus-failed accounting"
                if key == "num_valid_candidates"
                else f"legacy manifest summary {key} disagrees with candidate rows"
            )
            warnings.append(detail)
    return legacy


def _manifest_index_identity(
    config: dict[str, Any], row: dict[str, Any], warnings: list[str]
) -> tuple[int | None, list[str]]:
    manifest_budget = _positive_int(config.get("budget"))
    manifest_seed = _integer(config.get("seed"))
    manifest_objective = config.get("objective")
    reasons: list[str] = []
    identity_fields = (
        (
            manifest_budget is None,
            "search manifest config budget is missing or invalid",
            "manifest_budget_missing_or_invalid",
        ),
        (
            manifest_seed is None,
            "search manifest config seed is missing or invalid",
            "manifest_seed_missing_or_invalid",
        ),
        (
            not isinstance(manifest_objective, str) or not manifest_objective.strip(),
            "search manifest config objective is missing or invalid",
            "manifest_objective_missing_or_invalid",
        ),
    )
    for missing, warning, reason in identity_fields:
        if missing:
            warnings.append(warning)
            reasons.append(reason)
    mismatches = (
        (
            manifest_budget is not None and manifest_budget != row["budget"],
            "comparison budget disagrees with search manifest config budget",
            "manifest_budget_mismatch",
        ),
        (
            manifest_seed is not None and manifest_seed != row["seed"],
            "comparison seed disagrees with search manifest config seed",
            "manifest_seed_mismatch",
        ),
        (
            isinstance(manifest_objective, str) and manifest_objective.strip() != row["objective"],
            "comparison objective disagrees with search manifest config objective",
            "manifest_objective_mismatch",
        ),
    )
    for mismatched, warning, reason in mismatches:
        if mismatched:
            warnings.append(warning)
            reasons.append(reason)
    return manifest_budget, reasons


def _validate_mapped_manifest_identity(
    manifest: dict[str, Any],
    row: dict[str, Any],
    *,
    declared_manifest_path: str,
    archived_manifest_path: str,
) -> None:
    """Require mapped bytes to identify the original indexed run exactly."""
    config = manifest.get("config")
    if not isinstance(config, dict):
        raise ValueError(
            f"mapped manifest config is missing or malformed: {archived_manifest_path}"
        )
    manifest_budget = _positive_int(config.get("budget"))
    manifest_seed = _integer(config.get("seed"))
    manifest_objective = config.get("objective")
    objective_matches = (
        isinstance(manifest_objective, str)
        and manifest_objective.strip() == row["objective"].strip()
    )
    if manifest_budget != row["budget"] or manifest_seed != row["seed"] or not objective_matches:
        raise ValueError(
            "mapped manifest config identity does not match the indexed row "
            f"(sampler={row['sampler']!r}, seed={row['seed']!r}, budget={row['budget']!r}, "
            f"objective={row['objective']!r}): {archived_manifest_path}"
        )

    output_dir = config.get("output_dir")
    if not isinstance(output_dir, str) or not output_dir.strip():
        raise ValueError(
            f"mapped manifest config output_dir is missing or invalid: {archived_manifest_path}"
        )
    declared_parent = Path(declared_manifest_path).parent.resolve(strict=False)
    recorded_output_dir = Path(output_dir).resolve(strict=False)
    if recorded_output_dir != declared_parent:
        raise ValueError(
            "mapped manifest config output_dir does not match the original declared "
            f"manifest_path parent: {archived_manifest_path}"
        )


def _manifest_candidates(
    manifest: dict[str, Any] | None,
    artifact: dict[str, Any],
    *,
    indexed_budget: int,
    manifest_budget: int | None,
    warnings: list[str],
) -> tuple[list[Any], int]:
    if not isinstance(manifest, dict):
        expected_budget = manifest_budget or indexed_budget
        return [], max(expected_budget, indexed_budget)
    raw_candidates = manifest.get("candidates")
    if not isinstance(raw_candidates, list):
        warnings.append("search manifest candidates must be an array")
        raw_candidates = []
        artifact["status"] = "malformed"
    expected_budget = manifest_budget or indexed_budget
    if len(raw_candidates) > expected_budget:
        warnings.append("candidate count exceeds the recorded run budget")
    if len(raw_candidates) > indexed_budget:
        warnings.append("candidate count exceeds the comparison row budget")
    return raw_candidates, max(expected_budget, indexed_budget)


def _comparison_ineligibility_reasons(
    index_reasons: list[str],
    *,
    manifest: dict[str, Any] | None,
    artifact_status: str,
    derived: dict[str, Any],
    config_provenance: dict[str, Any],
) -> list[str]:
    reasons = list(index_reasons)
    if not manifest or artifact_status != "available":
        reasons.append("manifest_unavailable")
    if derived["num_over_budget_candidates"]:
        reasons.append("over_budget_candidates_present")
    if config_provenance.get("comparison_identity_sha256") is None:
        reasons.append("shared_configuration_identity_unknown")
    return sorted(set(reasons))


def _build_run(
    row: dict[str, Any],
    row_index: int,
    *,
    comparison_path: Path,
    repo_root: Path,
    path_binding: dict[str, Any] | None = None,
    path_map_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sampler = row["sampler"].strip()
    objective = row["objective"].strip()
    seed = row["seed"]
    indexed_budget = row["budget"]
    manifest_ref = row["manifest_path"]
    artifact = _load_manifest_artifact(
        manifest_ref,
        comparison_path=comparison_path,
        repo_root=repo_root,
        path_binding=path_binding,
    )
    manifest_path = artifact["path"]
    manifest = artifact["payload"]
    warnings = list(artifact["warnings"])
    if path_binding is not None and isinstance(manifest, dict):
        _validate_mapped_manifest_identity(
            manifest,
            row,
            declared_manifest_path=manifest_ref,
            archived_manifest_path=path_binding["archived_manifest_path"],
        )
    config = manifest.get("config", {}) if isinstance(manifest, dict) else {}
    config = config if isinstance(config, dict) else {}
    manifest_budget, index_identity_reasons = _manifest_index_identity(config, row, warnings)
    raw_candidates, expected_slots = _manifest_candidates(
        manifest,
        artifact,
        indexed_budget=indexed_budget,
        manifest_budget=manifest_budget,
        warnings=warnings,
    )
    episode_provenance = (
        _episode_provenance(raw_candidates, manifest_path=manifest_path, repo_root=repo_root)
        if manifest_path is not None and artifact["status"] == "available"
        else {
            "by_evaluation": {},
            "records": [],
            "commit_shas": set(),
            "unresolved_count": 0,
            "unverified_commit_identifiers": set(),
        }
    )
    derived = _derive_evaluations(
        raw_candidates,
        expected_slots=expected_slots,
        summary_budget=indexed_budget,
        episode_evidence_by_index=episode_provenance["by_evaluation"],
    )
    legacy_summary = _legacy_summary(manifest, derived, warnings)
    runtime_seconds, runtime_source = _runtime_seconds(row, manifest)
    source = (
        _source_evidence(manifest_path, episode_provenance, repo_root=repo_root)
        if manifest_path is not None and artifact["status"] == "available"
        else {
            "status": "unknown",
            "exact_source_revision": None,
            "observed_commit_shas": [],
            "unverified_commit_identifiers": [],
            "unresolved_episode_provenance_count": 0,
            "execution_context": None,
            "environment": None,
            "episode_records": [],
        }
    )
    config_provenance = (
        _config_provenance(config, manifest_path=manifest_path, repo_root=repo_root)
        if manifest_path is not None and isinstance(manifest, dict)
        else {
            "status": "unknown",
            "config_sha256": None,
            "search_space_sha256": None,
            "comparison_identity_sha256": None,
            "files": [],
        }
    )
    comparison_ineligibility_reasons = _comparison_ineligibility_reasons(
        index_identity_reasons,
        manifest=manifest,
        artifact_status=artifact["status"],
        derived=derived,
        config_provenance=config_provenance,
    )
    legacy_count = _positive_int(legacy_summary["num_candidates"])
    if legacy_count is not None and legacy_count != derived["num_candidates"]:
        warnings.append("manifest summary num_candidates disagrees with candidate list length")
    public_evaluations = [
        {key: value for key, value in item.items() if key != "_raw"}
        for item in derived["evaluations"]
    ]
    manifest_mapping = (
        {
            "method": "archived_manifest_path_map",
            "declared_manifest_path": manifest_ref,
            "archived_manifest_path": path_binding["archived_manifest_path"],
            "archived_manifest_sha256": path_binding["archived_manifest_sha256"],
            "path_map": path_map_provenance,
        }
        if path_binding is not None
        else None
    )
    return {
        "run_id": f"{sampler}:{seed}:{indexed_budget}:{objective}:row{row_index}",
        "comparison_row_index": row_index,
        "sampler": sampler,
        "method": _sampler_label(sampler),
        "objective": objective,
        "objective_direction": "maximize",
        "seed": seed,
        "budget": indexed_budget,
        "manifest_budget": manifest_budget,
        "expected_evaluations": expected_slots,
        "manifest_path": manifest_ref,
        "resolved_manifest_path": _portable_path(manifest_path, repo_root=repo_root)
        if manifest_path is not None
        else None,
        "manifest_mapping": manifest_mapping,
        "manifest_sha256": artifact["sha256"],
        "artifact_status": artifact["status"],
        "index_identity_valid": not index_identity_reasons and artifact["status"] == "available",
        "index_identity_reason_codes": sorted(set(index_identity_reasons)),
        "comparison_eligible": not comparison_ineligibility_reasons,
        "comparison_ineligibility_reason_codes": sorted(set(comparison_ineligibility_reasons)),
        "runtime_seconds": runtime_seconds,
        "runtime_source": runtime_source,
        "legacy_summary": legacy_summary,
        "config_provenance": config_provenance,
        "source_revision": source,
        "warnings": sorted(set(warnings)),
        **{key: value for key, value in derived.items() if key != "evaluations"},
        "evaluations": public_evaluations,
    }


def _validate_comparison_row(row: Any, index: int) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise ValueError(f"comparison row {index} must be an object")
    sampler = row.get("sampler")
    objective = row.get("objective")
    manifest_path = row.get("manifest_path")
    budget = _positive_int(row.get("budget"))
    seed = _integer(row.get("seed"))
    missing: list[str] = []
    if not isinstance(sampler, str) or not sampler.strip():
        missing.append("sampler")
    if not isinstance(objective, str) or not objective.strip():
        missing.append("objective")
    if not isinstance(manifest_path, str) or not manifest_path.strip():
        missing.append("manifest_path")
    if budget is None:
        missing.append("positive integer budget")
    if seed is None:
        missing.append("integer seed")
    if missing:
        raise ValueError(f"comparison row {index} missing/invalid fields: {', '.join(missing)}")
    return {
        **row,
        "sampler": sampler,
        "objective": objective,
        "manifest_path": manifest_path,
        "budget": budget,
        "seed": seed,
    }


def _aggregate_group(
    objective: str,
    sampler: str,
    budget: int,
    configuration_identity_sha256: str | None,
    group_runs: list[dict[str, Any]],
) -> dict[str, Any]:
    observed = [
        run
        for run in group_runs
        if run["artifact_status"] == "available"
        and run["index_identity_valid"]
        and run["config_provenance"].get("comparison_identity_sha256") is not None
    ]
    score_values = [
        run["best_objective_value"] for run in observed if run["best_objective_value"] is not None
    ]
    eligible_score_values = [
        run["best_analysis_eligible_objective_value"]
        for run in observed
        if run["best_analysis_eligible_objective_value"] is not None
    ]
    runtime_values = [
        run["runtime_seconds"] for run in observed if run["runtime_seconds"] is not None
    ]
    invalid = sum(run["num_invalid_candidates"] for run in group_runs)
    failed = sum(run["num_failed_evaluations"] for run in group_runs)
    valid = sum(run["num_valid_candidates"] for run in group_runs)
    critical = sum(run["num_critical_candidates"] for run in group_runs)
    observed_critical = sum(run["num_observed_critical_candidates"] for run in group_runs)
    duplicate = sum(run["num_duplicate_candidates"] for run in group_runs)
    attempted = sum(run["num_budgeted_candidates"] for run in group_runs)
    recorded_attempted = sum(run["num_candidates"] for run in group_runs)
    curve: list[dict[str, Any]] = []
    for index in range(1, budget + 1):
        observed_values = [
            point["best_so_far_objective"]
            for run in observed
            for point in run["evaluations"][: run["num_candidates"]]
            if point["evaluation_index"] == index and point["best_so_far_objective"] is not None
        ]
        eligible_values = [
            point["best_so_far_analysis_eligible_objective"]
            for run in observed
            for point in run["evaluations"][: run["num_candidates"]]
            if point["evaluation_index"] == index
            and point["best_so_far_analysis_eligible_objective"] is not None
        ]
        curve.append(
            {
                "evaluation_index": index,
                "n_available_runs": len(eligible_values),
                "median_best_so_far": statistics.median(eligible_values)
                if eligible_values
                else None,
                "min_best_so_far": min(eligible_values) if eligible_values else None,
                "max_best_so_far": max(eligible_values) if eligible_values else None,
                "n_observed_runs": len(observed_values),
                "median_observed_best_so_far": statistics.median(observed_values)
                if observed_values
                else None,
                "min_observed_best_so_far": min(observed_values) if observed_values else None,
                "max_observed_best_so_far": max(observed_values) if observed_values else None,
                "aggregation": "descriptive_run_range",
            }
        )
    seeds = sorted({run["seed"] for run in group_runs})
    return {
        "objective": objective,
        "sampler": sampler,
        "method": _sampler_label(sampler),
        "budget": budget,
        "configuration_identity_sha256": configuration_identity_sha256,
        "run_count": len(group_runs),
        "available_manifest_count": len(observed),
        "seed_count": len(seeds),
        "seeds": seeds,
        "final_best_objective": {
            "median": statistics.median(eligible_score_values) if eligible_score_values else None,
            "min": min(eligible_score_values) if eligible_score_values else None,
            "max": max(eligible_score_values) if eligible_score_values else None,
            "n": len(eligible_score_values),
            "interpretation": "analysis-eligible evidence only; descriptive, no inference",
        },
        "final_best_observed_objective": {
            "median": statistics.median(score_values) if score_values else None,
            "min": min(score_values) if score_values else None,
            "max": max(score_values) if score_values else None,
            "n": len(score_values),
            "interpretation": "budget-limited raw observations; may include ineligible/degraded attempts",
        },
        "runtime_seconds": {
            "median": statistics.median(runtime_values) if runtime_values else None,
            "min": min(runtime_values) if runtime_values else None,
            "max": max(runtime_values) if runtime_values else None,
            "n": len(runtime_values),
            "status": "recorded" if runtime_values else "not_recorded",
        },
        "candidate_accounting": {
            "attempted": attempted,
            "recorded_attempted_including_over_budget": recorded_attempted,
            "over_budget": sum(run["num_over_budget_candidates"] for run in group_runs),
            "valid_total_minus_invalid_minus_failed": valid,
            "invalid": invalid,
            "failed": failed,
            "critical": critical,
            "observed_critical": observed_critical,
            "duplicate": duplicate,
            "missing_within_budget": sum(
                run["num_missing_budgeted_evaluations"] for run in group_runs
            ),
            "missing_all_expected_slots": sum(run["num_missing_evaluations"] for run in group_runs),
            "duplicate_rate_observed": duplicate / attempted if attempted else None,
            "invalid_rate_observed": invalid / attempted if attempted else None,
        },
        "curve": curve,
        "aggregation_note": "Seed/run medians and observed ranges are descriptive summaries, not confidence intervals.",
    }


def _paired_random_tpe(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    paired: list[dict[str, Any]] = []
    objective_budgets = sorted({(run["objective"], run["budget"]) for run in runs})
    for objective, budget in objective_budgets:
        group = [run for run in runs if run["objective"] == objective and run["budget"] == budget]
        random_by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
        tpe_by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for run in group:
            method_key = run["sampler"].strip().lower()
            if method_key == "random":
                random_by_seed[run["seed"]].append(run)
            elif method_key in {"optuna", "tpe"}:
                tpe_by_seed[run["seed"]].append(run)
        pairs, unavailable, ambiguous_random, ambiguous_tpe, ineligible = _pair_seed_runs(
            random_by_seed,
            tpe_by_seed,
        )
        deltas = [
            pair["tpe_minus_random"] for pair in pairs if pair["tpe_minus_random"] is not None
        ]
        random_seeds = sorted(random_by_seed)
        tpe_seeds = sorted(tpe_by_seed)
        paired.append(
            {
                "objective": objective,
                "budget": budget,
                "matched_seed_count": len(pairs),
                "unmatched_random_seeds": sorted(set(random_seeds) - set(tpe_seeds)),
                "unmatched_tpe_seeds": sorted(set(tpe_seeds) - set(random_seeds)),
                "unavailable_matched_seeds": unavailable,
                "ineligible_matched_seeds": ineligible,
                "ambiguous_random_seeds": ambiguous_random,
                "ambiguous_tpe_seeds": ambiguous_tpe,
                "pairs": pairs,
                "tpe_minus_random_median": statistics.median(deltas) if deltas else None,
                "tpe_minus_random_min": min(deltas) if deltas else None,
                "tpe_minus_random_max": max(deltas) if deltas else None,
                "inference_status": "not_performed",
                "interpretation": "paired descriptive differences only; this report does not perform inferential tests",
            }
        )
    return paired


def _pair_seed_runs(
    random_by_seed: dict[int, list[dict[str, Any]]],
    tpe_by_seed: dict[int, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[int], list[int], list[int], list[dict[str, Any]]]:
    pairs: list[dict[str, Any]] = []
    unavailable: list[int] = []
    ambiguous_random: list[int] = []
    ambiguous_tpe: list[int] = []
    ineligible: list[dict[str, Any]] = []
    for seed in sorted(set(random_by_seed) & set(tpe_by_seed)):
        random_runs = random_by_seed[seed]
        tpe_runs = tpe_by_seed[seed]
        if len(random_runs) != 1:
            ambiguous_random.append(seed)
        if len(tpe_runs) != 1:
            ambiguous_tpe.append(seed)
        if len(random_runs) != 1 or len(tpe_runs) != 1:
            continue
        random_run, tpe_run = random_runs[0], tpe_runs[0]
        if (
            random_run["artifact_status"] != "available"
            or tpe_run["artifact_status"] != "available"
        ):
            unavailable.append(seed)
            continue
        random_reasons = random_run["comparison_ineligibility_reason_codes"]
        tpe_reasons = tpe_run["comparison_ineligibility_reason_codes"]
        if random_reasons or tpe_reasons:
            ineligible.append(
                {
                    "seed": seed,
                    "reason_codes": sorted(set(random_reasons) | set(tpe_reasons)),
                    "random_reason_codes": random_reasons,
                    "tpe_reason_codes": tpe_reasons,
                }
            )
            continue
        random_identity = random_run["config_provenance"].get("comparison_identity_sha256")
        tpe_identity = tpe_run["config_provenance"].get("comparison_identity_sha256")
        if not random_identity or not tpe_identity or random_identity != tpe_identity:
            ineligible.append(
                {
                    "seed": seed,
                    "reason_codes": ["shared_configuration_mismatch"],
                    "random_reason_codes": [],
                    "tpe_reason_codes": [],
                    "random_config_identity_sha256": random_identity,
                    "tpe_config_identity_sha256": tpe_identity,
                }
            )
            continue
        random_missing = random_run["num_missing_budgeted_evaluations"]
        tpe_missing = tpe_run["num_missing_budgeted_evaluations"]
        if random_missing or tpe_missing:
            ineligible.append(
                {
                    "seed": seed,
                    "reason_codes": ["incomplete_budgeted_evaluations"],
                    "random_reason_codes": (
                        ["incomplete_budgeted_evaluations"] if random_missing else []
                    ),
                    "tpe_reason_codes": (
                        ["incomplete_budgeted_evaluations"] if tpe_missing else []
                    ),
                    "random_missing_budgeted_evaluations": random_missing,
                    "tpe_missing_budgeted_evaluations": tpe_missing,
                }
            )
            continue
        random_score = random_run["best_analysis_eligible_objective_value"]
        tpe_score = tpe_run["best_analysis_eligible_objective_value"]
        if random_score is None or tpe_score is None:
            ineligible.append(
                {
                    "seed": seed,
                    "reason_codes": ["analysis_eligible_score_unavailable"],
                    "random_reason_codes": [],
                    "tpe_reason_codes": [],
                }
            )
            continue
        pairs.append(
            {
                "seed": seed,
                "random_final_best": random_score,
                "tpe_final_best": tpe_score,
                "tpe_minus_random": tpe_score - random_score,
                "configuration_identity_sha256": random_identity,
            }
        )
    return pairs, unavailable, ambiguous_random, ambiguous_tpe, ineligible


def _aggregate_runs(
    runs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[tuple[str, str, int, str | None], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        identity = run["config_provenance"].get("comparison_identity_sha256")
        groups[(run["objective"], run["sampler"], run["budget"], identity)].append(run)
    aggregates = [
        _aggregate_group(objective, sampler, budget, identity, group_runs)
        for (objective, sampler, budget, identity), group_runs in sorted(
            groups.items(), key=lambda item: (item[0][0], item[0][1], item[0][2], item[0][3] or "")
        )
    ]
    return aggregates, _paired_random_tpe(runs)


def build_convergence_report(
    comparison_path: Path,
    *,
    repo_root: Path | None = None,
    manifest_path_map_path: Path | None = None,
) -> dict[str, Any]:
    """Build the machine-readable report from a v3 comparison index and its search manifests."""
    input_path = comparison_path.resolve()
    root = (repo_root or Path.cwd()).resolve()
    try:
        comparison_bytes = input_path.read_bytes()
        comparison = json.loads(comparison_bytes.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read comparison input {input_path}: {exc}") from exc
    if not isinstance(comparison, dict) or comparison.get("schema_version") != COMPARISON_SCHEMA:
        raise ValueError(f"comparison input schema must be {COMPARISON_SCHEMA}")
    raw_rows = comparison.get("rows")
    if not isinstance(raw_rows, list):
        raise ValueError("comparison input rows must be an array")
    comparison_digest = _bytes_sha256(comparison_bytes)
    rows = [_validate_comparison_row(row, index) for index, row in enumerate(raw_rows, start=1)]
    path_bindings: dict[str, dict[str, Any]] = {}
    path_map_provenance: dict[str, Any] | None = None
    if manifest_path_map_path is not None:
        path_bindings, path_map_provenance = _load_manifest_path_map(
            manifest_path_map_path.resolve(),
            comparison_sha256=comparison_digest,
            rows=rows,
            repo_root=root,
        )
    runs = [
        _build_run(
            row,
            index,
            comparison_path=input_path,
            repo_root=root,
            path_binding=path_bindings.get(row["manifest_path"]),
            path_map_provenance=path_map_provenance,
        )
        for index, row in enumerate(rows, start=1)
    ]
    aggregates, paired = _aggregate_runs(runs)
    observed_revisions = sorted(
        {
            run["source_revision"]["exact_source_revision"]
            for run in runs
            if run["source_revision"].get("exact_source_revision")
        }
    )
    conflicting_runs = [
        run["run_id"] for run in runs if run["source_revision"]["status"] == "conflicting"
    ]
    unknown_revision_runs = [
        run["run_id"] for run in runs if run["source_revision"].get("exact_source_revision") is None
    ]
    unverified_commit_identifiers = sorted(
        {
            identifier
            for run in runs
            for identifier in run["source_revision"].get("unverified_commit_identifiers", [])
        }
    )
    if conflicting_runs or len(observed_revisions) > 1:
        global_revision_status = "conflicting"
        exact_revision = None
    elif len(observed_revisions) == 1 and not unknown_revision_runs:
        global_revision_status = "consistent"
        exact_revision = observed_revisions[0]
    elif len(observed_revisions) == 1:
        global_revision_status = "partial"
        exact_revision = None
    else:
        global_revision_status = "unknown"
        exact_revision = None
    objectives = sorted({run["objective"] for run in runs})
    return {
        "schema_version": REPORT_SCHEMA,
        "claim_scope": "diagnostic_only_finite_search_budget",
        "comparison": {
            "path": _portable_path(input_path, repo_root=root),
            "schema_version": COMPARISON_SCHEMA,
            "sha256": comparison_digest,
        },
        "provenance": {
            "search_manifest_count": len(runs),
            "search_manifests": [
                {
                    "run_id": run["run_id"],
                    "path": run["resolved_manifest_path"] or run["manifest_path"],
                    "declared_path": run["manifest_path"],
                    "mapping": run["manifest_mapping"],
                    "sha256": run["manifest_sha256"],
                    "status": run["artifact_status"],
                    "config_sha256": run["config_provenance"].get("config_sha256"),
                }
                for run in runs
            ],
            "manifest_path_map": path_map_provenance,
            "source_revision": {
                "status": global_revision_status,
                "exact_source_revision": exact_revision,
                "observed_commit_shas": observed_revisions,
                "unverified_commit_identifiers": unverified_commit_identifiers,
                "runs_without_exact_revision": unknown_revision_runs,
                "conflicting_runs": conflicting_runs,
            },
        },
        "objectives": objectives,
        "runs": runs,
        "aggregates": aggregates,
        "random_vs_tpe": paired,
        "figures": [f"convergence_{_slug(objective)}.png" for objective in objectives],
        "limitations": [
            "A finite search budget that finds no counterexample does not establish that none exists.",
            "Seed/run aggregates are descriptive; observed min/max ranges are not confidence intervals.",
            "No inferential test is performed, and small pilot seed counts do not support broad claims.",
            "Search-level runtime is reported only when a finite nonnegative runtime_seconds field is recorded in the search manifest or its summary; comparison-row fields are ignored.",
            "The current runner's legacy num_valid_candidates summary omits evaluator failures; this report derives valid as candidate rows minus invalid minus failed and retains the legacy field for audit.",
            "Fixture tests verify report accounting; finite-budget reports do not establish planner safety, search-space coverage, or absence of counterexamples.",
        ],
    }


def _slug(value: str) -> str:
    normalized = "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized or "objective"


def _display_number(value: Any) -> str:
    if value is None:
        return "Not recorded"
    if isinstance(value, int):
        return str(value)
    parsed = _finite_number(value)
    return f"{parsed:.4g}" if parsed is not None else "Not recorded"


def _render_status_counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{status}: {count}" for status, count in counts.items()) or "Not recorded"


def render_markdown(report: dict[str, Any]) -> str:
    """Render a deterministic Markdown report with run-level and paired summaries."""
    path_map = report["provenance"].get("manifest_path_map")
    lines = [
        "# Falsification search convergence report",
        "",
        "This diagnostic report summarizes persisted search attempts and their best-so-far objective by evaluation budget.",
        "",
        f"- Claim scope: `{report['claim_scope']}`.",
        f"- Comparison input SHA-256: `{report['comparison']['sha256']}`.",
        (
            f"- Archived manifest path map: `{path_map['path']}`; SHA-256 `{path_map['sha256']}`; "
            f"bound comparison SHA-256 `{path_map['source_comparison_sha256']}`."
            if path_map is not None
            else "- Archived manifest path map: `not supplied`."
        ),
        f"- Source revision status: `{report['provenance']['source_revision']['status']}`.",
        f"- Exact source revision: `{report['provenance']['source_revision']['exact_source_revision'] or 'unknown'}`.",
        "",
        "## Per-run accounting",
        "",
        "| Objective | Method | Seed | Budget | Search runtime (s) | Best observed ≤B | Best eligible ≤B | Observed critical (all rows) | Eligible critical ≤B | First eligible critical eval | Over-budget rows | Valid / invalid / failed / scoreless / missing ≤B / all expected | Duplicates ≤B | Execution modes | Availability | Run input status | Artifact |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|",
    ]
    for run in report["runs"]:
        accounting = (
            f"{run['num_valid_candidates']} / {run['num_invalid_candidates']} / "
            f"{run['num_failed_evaluations']} / {run['num_scoreless_valid_candidates']} / "
            f"{run['num_missing_budgeted_evaluations']} / {run['num_missing_evaluations']}"
        )
        duplicate = (
            f"{run['num_duplicate_candidates']}/{run['num_budgeted_candidates']}"
            if run["num_budgeted_candidates"]
            else "Not recorded"
        )
        run_input_status = (
            "input checks passed"
            if run["comparison_eligible"]
            else "; ".join(run["comparison_ineligibility_reason_codes"])
        )
        execution_modes = _render_status_counts(run["execution_mode_counts"])
        availability = _render_status_counts(run["availability_status_counts"])
        lines.append(
            f"| `{run['objective']}` | {run['method']} | {run['seed']} | {run['budget']} | "
            f"{_display_number(run['runtime_seconds'])} | "
            f"{_display_number(run['best_objective_value'])} | "
            f"{_display_number(run['best_analysis_eligible_objective_value'])} | "
            f"{run['num_observed_critical_candidates']} | {run['num_critical_candidates']} | "
            f"{run['first_critical_evaluation'] if run['first_critical_evaluation'] is not None else 'None recorded'} | "
            f"{run['num_over_budget_candidates']} | {accounting} | {duplicate} | "
            f"{execution_modes} | {availability} | {run_input_status} | {run['artifact_status']} |"
        )
    if not report["runs"]:
        lines.append(
            "| — | — | — | — | Not recorded | Not recorded | Not recorded | — | — | — | — | — | — | — | — | — | no runs |"
        )
    lines.extend(
        [
            "",
            "Budget-limited summaries use only the first B comparison-indexed candidate rows. Extra rows remain in JSON audit history and cannot alter best-so-far values or paired deltas.",
            "",
            "Observed best and critical counts retain raw candidate evidence. Eligible best/critical summaries require a scored objective, `execution_mode=native`, `readiness_status=native`, `availability_status=available`, a parseable episode-record artifact, an effective-scenario hash, and an explicit `analysis_eligibility.eligible=true` receipt; contradictory or incomplete evidence stays ineligible. Per-evaluation reason codes identify failed checks.",
            "",
            "Valid candidates within B are derived as `candidate rows within B - invalid - failed`; scoreless valid evaluations remain in that count. Missing and over-budget attempts remain explicit.",
            "",
            "## Matched Random vs TPE",
            "",
        ]
    )
    lines.extend(
        [
            "| Objective | Budget | Matched eligible seeds | Excluded matched seeds | TPE − Random median | Observed delta range | Inference |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for comparison in report["random_vs_tpe"]:
        low = _display_number(comparison["tpe_minus_random_min"])
        high = _display_number(comparison["tpe_minus_random_max"])
        range_text = f"{low} to {high}" if comparison["matched_seed_count"] else "Not available"
        lines.append(
            f"| `{comparison['objective']}` | {comparison['budget']} | "
            f"{comparison['matched_seed_count']} | "
            f"{len(comparison['ineligible_matched_seeds']) + len(comparison['unavailable_matched_seeds'])} | "
            f"{_display_number(comparison['tpe_minus_random_median'])} | {range_text} | "
            "Not performed; descriptive only |"
        )
    if not report["random_vs_tpe"]:
        lines.append("| — | — | 0 | 0 | Not available | Not available | Not performed |")
    lines.extend(
        [
            "",
            "Run input status reports per-run index/manifest/config checks only; it does not assert that a Random/TPE pair exists or qualifies. Pairs are decided separately and require index/manifest agreement, matching normalized scenario/search/planner configuration, no over-budget rows, complete budgeted evaluations, and an analysis-eligible score from both methods. Missing, unmatched, ambiguous, and counterpart-ineligible cases remain in the JSON with reason codes.",
        ]
    )
    lines.extend(["", "## Figures", ""])
    if report["figures"]:
        for figure in report["figures"]:
            lines.append(f"![Best-so-far convergence by evaluation budget]({figure})")
            lines.append("")
    else:
        lines.append(
            "No figure was produced because the comparison input contains no objective runs."
        )
        lines.append("")
    lines.extend(["## Interpretation limits", ""])
    lines.extend(f"- {limitation}" for limitation in report["limitations"])
    lines.extend(
        ["", "## Provenance", "", f"Comparison artifact: `{report['comparison']['path']}`.", ""]
    )
    for entry in report["provenance"]["search_manifests"]:
        mapping = entry["mapping"]
        resolved_path = entry["path"]
        path_description = (
            f"declared `{entry['declared_path']}`; archived `{mapping['archived_manifest_path']}`"
            if mapping is not None
            else f"declared/resolved `{resolved_path}`"
        )
        lines.append(
            f"- `{entry['run_id']}`: {path_description}; "
            f"SHA-256 `{entry['sha256'] or 'unavailable'}`; "
            f"config SHA-256 `{entry['config_sha256'] or 'unavailable'}`; status `{entry['status']}`."
        )
        if mapping is not None:
            lines.append(
                f"  Mapping provenance: `{mapping['path_map']['path']}`; path-map SHA-256 "
                f"`{mapping['path_map']['sha256']}`; archived SHA-256 "
                f"`{mapping['archived_manifest_sha256']}`."
            )
    lines.append("")
    return "\n".join(lines)


def _aggregate_point(runs: list[dict[str, Any]], index: int, *, field: str) -> list[float]:
    values: list[float] = []
    for run in runs:
        for point in run["evaluations"][: run["num_candidates"]]:
            value = point.get(field)
            if point["evaluation_index"] == index and value is not None:
                values.append(float(value))
                break
    return values


def _plot_configuration_curves(
    axis: Any,
    objective_runs: list[dict[str, Any]],
    *,
    budget: int,
    sampler: str,
    identity: str,
    configuration_count: int,
) -> None:
    method_runs = [
        run
        for run in objective_runs
        if run["budget"] == budget
        and run["sampler"] == sampler
        and run["artifact_status"] == "available"
        and run["index_identity_valid"]
        and run["config_provenance"].get("comparison_identity_sha256") == identity
    ]
    if not method_runs:
        return
    x_values = list(range(1, budget + 1))
    eligible_values = [
        _aggregate_point(method_runs, index, field="best_so_far_analysis_eligible_objective")
        for index in x_values
    ]
    observed_values = [
        _aggregate_point(method_runs, index, field="best_so_far_observed_objective")
        for index in x_values
    ]
    eligible_medians = [
        statistics.median(values) if values else math.nan for values in eligible_values
    ]
    eligible_mins = [min(values) if values else math.nan for values in eligible_values]
    eligible_maxs = [max(values) if values else math.nan for values in eligible_values]
    observed_medians = [
        statistics.median(values) if values else math.nan for values in observed_values
    ]
    eligible_seed_count = len(
        {
            run["seed"]
            for run in method_runs
            if run["best_analysis_eligible_objective_value"] is not None
        }
    )
    observed_seed_count = len(
        {run["seed"] for run in method_runs if run["best_objective_value"] is not None}
    )
    config_label = f" cfg={identity[:8]}" if configuration_count > 1 else ""
    eligible_line = axis.plot(
        x_values,
        eligible_medians,
        marker=".",
        linewidth=1.8,
        label=f"{_sampler_label(sampler)} eligible{config_label} (n={eligible_seed_count})",
    )[0]
    axis.plot(
        x_values,
        observed_medians,
        linestyle="--",
        linewidth=1.1,
        alpha=0.75,
        color=eligible_line.get_color(),
        label=f"{_sampler_label(sampler)} observed{config_label} (n={observed_seed_count})",
    )
    if eligible_seed_count > 1:
        axis.fill_between(x_values, eligible_mins, eligible_maxs, alpha=0.15)


def render_figures(report: dict[str, Any], output_dir: Path) -> list[Path]:
    """Render deterministic static budget curves using the repository Matplotlib dependency."""
    if not report["objectives"]:
        return []
    import matplotlib  # noqa: PLC0415 - defer plotting dependency until a figure is requested

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415 - defer plotting dependency until a figure is requested

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for objective in report["objectives"]:
        objective_runs = [run for run in report["runs"] if run["objective"] == objective]
        budgets = sorted({run["budget"] for run in objective_runs})
        columns = min(3, max(1, len(budgets)))
        rows = math.ceil(len(budgets) / columns)
        fig, axes = plt.subplots(
            rows,
            columns,
            figsize=(6.2 * columns, 4.4 * rows),
            squeeze=False,
        )
        flat_axes = [axis for line in axes for axis in line]
        method_order = sorted(
            {run["sampler"] for run in objective_runs},
            key=lambda value: (value.strip().lower() not in {"random", "optuna", "tpe"}, value),
        )
        for axis, budget in zip(flat_axes, budgets, strict=False):
            configuration_ids = sorted(
                {
                    identity
                    for run in objective_runs
                    if run["budget"] == budget
                    if (identity := run["config_provenance"].get("comparison_identity_sha256"))
                }
            )
            for sampler in method_order:
                for identity in configuration_ids:
                    _plot_configuration_curves(
                        axis,
                        objective_runs,
                        budget=budget,
                        sampler=sampler,
                        identity=identity,
                        configuration_count=len(configuration_ids),
                    )
            axis.set_title(f"Budget {budget}")
            axis.set_xlabel("Evaluation index")
            axis.set_ylabel("Best-so-far objective (maximize)")
            axis.set_xlim(1, max(1, budget))
            axis.grid(True, alpha=0.25)
            has_scored_observations = any(
                math.isfinite(float(value))
                for line in axis.get_lines()
                for value in line.get_ydata()
            )
            if has_scored_observations:
                axis.legend(loc="best")
            else:
                axis.text(
                    0.5,
                    0.5,
                    "No scored observations\nwithin this budget",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )
        for axis in flat_axes[len(budgets) :]:
            axis.set_visible(False)
        fig.suptitle(
            f"{objective}: recorded best-so-far by evaluation budget\n"
            "Solid=analysis-eligible; dashed=all observed scores; curves are capped at the row budget",
            fontsize=10,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        path = output_dir / f"convergence_{_slug(objective)}.png"
        fig.savefig(path, dpi=140, metadata={"Software": "Robot SF falsification report"})
        plt.close(fig)
        paths.append(path)
    return paths


def write_convergence_report(report: dict[str, Any], output_dir: Path) -> dict[str, str]:
    """Write stable JSON, Markdown and static figures to ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "falsification_report.json"
    markdown_path = output_dir / "falsification_report.md"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    figure_paths = render_figures(report, output_dir)
    return {
        "json": json_path.as_posix(),
        "markdown": markdown_path.as_posix(),
        "figures": [path.as_posix() for path in figure_paths],
    }


__all__ = [
    "COMPARISON_SCHEMA",
    "REPORT_SCHEMA",
    "SEARCH_SCHEMA",
    "build_convergence_report",
    "render_figures",
    "render_markdown",
    "write_convergence_report",
]
