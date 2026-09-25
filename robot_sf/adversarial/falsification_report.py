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
from pathlib import Path
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


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON input {path}: {exc}") from exc


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


def _analysis_evidence_eligible(item: dict[str, Any], *, status: str) -> bool:
    """Require explicit analysis eligibility and non-degraded scored evidence."""
    return (
        status in {"scored", "scoreless"}
        and _analysis_eligibility(item) is True
        and _execution_risk_mode(item) is None
    )


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
        "analysis_evidence_eligible": False,
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
    row: dict[str, Any],
    *,
    manifest_path: Path,
    repo_root: Path,
) -> tuple[dict[str, Any] | None, str | None, bool]:
    raw_item = row.get("_raw")
    raw_path = raw_item.get("episode_record_path") if isinstance(raw_item, dict) else None
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None, None, False
    resolved = _resolve_path(raw_path, anchors=(repo_root, manifest_path.parent, Path.cwd()))
    if resolved is None or not resolved.is_file():
        return (
            {"evaluation_index": row["evaluation_index"], "path": raw_path, "status": "missing"},
            None,
            True,
        )
    commit = None
    parsed = None
    try:
        with resolved.open(encoding="utf-8") as handle:
            first_line = handle.readline()
        parsed = json.loads(first_line) if first_line else None
        if isinstance(parsed, dict):
            identifier = _commit_value(parsed)
            commit = identifier.lower() if _is_full_commit_sha(identifier) else None
    except (OSError, UnicodeError, json.JSONDecodeError):
        commit = None
    evidence = {
        "evaluation_index": row["evaluation_index"],
        "path": _portable_path(resolved, repo_root=repo_root),
        "sha256": _file_sha256(resolved),
        "commit_sha": commit,
        "commit_identifier": _commit_value(parsed) if isinstance(parsed, dict) else None,
        "status": "available"
        if commit
        else "commit_unverified"
        if isinstance(parsed, dict) and _commit_value(parsed)
        else "commit_unknown",
    }
    return evidence, commit, commit is None


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
    candidate_rows: list[dict[str, Any]], *, manifest_path: Path, repo_root: Path
) -> tuple[list[dict[str, Any]], set[str], int, set[str]]:
    evidence_rows: list[dict[str, Any]] = []
    commits: set[str] = set()
    unverified: set[str] = set()
    unresolved = 0
    for row in candidate_rows:
        evidence, commit, has_unresolved = _episode_record_evidence(
            row, manifest_path=manifest_path, repo_root=repo_root
        )
        if evidence is None:
            continue
        evidence_rows.append(evidence)
        if commit:
            commits.add(commit)
        identifier = evidence.get("commit_identifier")
        if identifier and not _is_full_commit_sha(identifier):
            unverified.add(identifier)
        unresolved += int(has_unresolved)
    return evidence_rows, commits, unresolved, unverified


def _source_evidence(
    manifest_path: Path,
    candidate_rows: list[dict[str, Any]],
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

    episode_evidence, commits, unresolved_refs, unverified_commit_identifiers = _episode_provenance(
        candidate_rows, manifest_path=manifest_path, repo_root=repo_root
    )
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
        ("comparison_row.runtime_seconds", row.get("runtime_seconds")),
        ("comparison_row.duration_seconds", row.get("duration_seconds")),
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
) -> dict[str, Any]:
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
        }
    if not isinstance(payload, dict) or payload.get("schema_version") != SEARCH_SCHEMA:
        warnings.append(f"search manifest schema must be {SEARCH_SCHEMA}")
        return {
            "path": path,
            "payload": None,
            "sha256": None,
            "status": "malformed",
            "warnings": warnings,
        }
    return {
        "path": path,
        "payload": payload,
        "sha256": _file_sha256(path),
        "status": "available",
        "warnings": warnings,
    }


def _derive_evaluations(
    candidates: list[Any],
    *,
    expected_slots: int,
    summary_budget: int,
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
        analysis_evidence_eligible = (
            _analysis_evidence_eligible(item, status=status) if is_mapping else False
        )
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
            "analysis_evidence_eligible": analysis_evidence_eligible,
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
        "num_missing_evaluations": max(0, expected_slots - actual_count),
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
) -> dict[str, Any]:
    sampler = row["sampler"].strip()
    objective = row["objective"].strip()
    seed = row["seed"]
    indexed_budget = row["budget"]
    manifest_ref = row["manifest_path"].strip()
    artifact = _load_manifest_artifact(
        manifest_ref,
        comparison_path=comparison_path,
        repo_root=repo_root,
    )
    manifest_path = artifact["path"]
    manifest = artifact["payload"]
    warnings = list(artifact["warnings"])
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
    derived = _derive_evaluations(
        raw_candidates,
        expected_slots=expected_slots,
        summary_budget=indexed_budget,
    )
    legacy_summary = _legacy_summary(manifest, derived, warnings)
    runtime_seconds, runtime_source = _runtime_seconds(row, manifest)
    source = (
        _source_evidence(
            manifest_path, derived["evaluations"][: derived["num_candidates"]], repo_root=repo_root
        )
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
            "missing": sum(run["num_missing_evaluations"] for run in group_runs),
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
    comparison_path: Path, *, repo_root: Path | None = None
) -> dict[str, Any]:
    """Build the machine-readable report from a v3 comparison index and its search manifests."""
    input_path = comparison_path.resolve()
    root = (repo_root or Path.cwd()).resolve()
    comparison = _load_json(input_path)
    if not isinstance(comparison, dict) or comparison.get("schema_version") != COMPARISON_SCHEMA:
        raise ValueError(f"comparison input schema must be {COMPARISON_SCHEMA}")
    raw_rows = comparison.get("rows")
    if not isinstance(raw_rows, list):
        raise ValueError("comparison input rows must be an array")
    comparison_digest = _file_sha256(input_path)
    runs = [
        _build_run(
            _validate_comparison_row(row, index),
            index,
            comparison_path=input_path,
            repo_root=root,
        )
        for index, row in enumerate(raw_rows, start=1)
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
                    "sha256": run["manifest_sha256"],
                    "status": run["artifact_status"],
                    "config_sha256": run["config_provenance"].get("config_sha256"),
                }
                for run in runs
            ],
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
            "Search-level runtime remains unknown unless explicitly recorded in the comparison row or search manifest.",
            "The current runner's legacy num_valid_candidates summary omits evaluator failures; this report derives valid as candidate rows minus invalid minus failed and retains the legacy field for audit.",
            "A report built from fixtures demonstrates report behavior, not planner safety or empirical search performance.",
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
    lines = [
        "# Falsification search convergence report",
        "",
        "This diagnostic report summarizes persisted search attempts and their best-so-far objective by evaluation budget.",
        "",
        f"- Claim scope: `{report['claim_scope']}`.",
        f"- Comparison input SHA-256: `{report['comparison']['sha256']}`.",
        f"- Source revision status: `{report['provenance']['source_revision']['status']}`.",
        f"- Exact source revision: `{report['provenance']['source_revision']['exact_source_revision'] or 'unknown'}`.",
        "",
        "## Per-run accounting",
        "",
        "| Objective | Method | Seed | Budget | Best observed ≤B | Best eligible ≤B | Observed critical (all rows) | Eligible critical ≤B | First eligible critical eval | Over-budget rows | Valid / invalid / failed / scoreless / missing (≤B) | Duplicates ≤B | Execution modes | Availability | Pair status | Artifact |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|",
    ]
    for run in report["runs"]:
        accounting = (
            f"{run['num_valid_candidates']} / {run['num_invalid_candidates']} / "
            f"{run['num_failed_evaluations']} / {run['num_scoreless_valid_candidates']} / "
            f"{run['num_missing_evaluations']}"
        )
        duplicate = (
            f"{run['num_duplicate_candidates']}/{run['num_budgeted_candidates']}"
            if run["num_budgeted_candidates"]
            else "Not recorded"
        )
        pair_status = (
            "eligible"
            if run["comparison_eligible"]
            else "; ".join(run["comparison_ineligibility_reason_codes"])
        )
        execution_modes = _render_status_counts(run["execution_mode_counts"])
        availability = _render_status_counts(run["availability_status_counts"])
        lines.append(
            f"| `{run['objective']}` | {run['method']} | {run['seed']} | {run['budget']} | "
            f"{_display_number(run['best_objective_value'])} | "
            f"{_display_number(run['best_analysis_eligible_objective_value'])} | "
            f"{run['num_observed_critical_candidates']} | {run['num_critical_candidates']} | "
            f"{run['first_critical_evaluation'] if run['first_critical_evaluation'] is not None else 'None recorded'} | "
            f"{run['num_over_budget_candidates']} | {accounting} | {duplicate} | "
            f"{execution_modes} | {availability} | {pair_status} | {run['artifact_status']} |"
        )
    if not report["runs"]:
        lines.append(
            "| — | — | — | — | Not recorded | Not recorded | — | — | — | — | — | — | — | — | — | no runs |"
        )
    lines.extend(
        [
            "",
            "Budget-limited summaries use only the first B comparison-indexed candidate rows. Extra rows remain in JSON audit history and cannot alter best-so-far values or paired deltas.",
            "",
            "Observed best and critical counts retain raw candidate evidence. Eligible best/critical summaries require an explicit `analysis_eligibility.eligible=true` and exclude fallback/degraded execution; unknown eligibility is not promoted.",
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
            "Pairs require index/manifest agreement, matching normalized scenario/search/planner configuration, no over-budget rows, and an analysis-eligible score from both methods. Exclusion reason codes are retained in JSON.",
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
        lines.append(
            f"- `{entry['run_id']}`: `{entry['path']}`; SHA-256 `{entry['sha256'] or 'unavailable'}`; "
            f"config SHA-256 `{entry['config_sha256'] or 'unavailable'}`; status `{entry['status']}`."
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
            if axis.get_legend_handles_labels()[0]:
                axis.legend(loc="best")
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
