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
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

COMPARISON_SCHEMA = "adversarial-sampler-comparison.v3"
SEARCH_SCHEMA = "adversarial-search-manifest.v1"
REPORT_SCHEMA = "adversarial-search-convergence-report.v1"
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


def _missing_evaluation(index: int) -> dict[str, Any]:
    return {
        "evaluation_index": index,
        "status": "missing",
        "failure_type": None,
        "objective_value": None,
        "critical": False,
        "candidate": None,
        "candidate_sha256": None,
        "effective_scenario_hash": None,
        "duplicate_of_evaluation": None,
        "duplicate_basis": [],
        "analysis_eligible": None,
        "execution_mode": None,
        "readiness_status": None,
        "availability_status": None,
        "error": "candidate evaluation was expected by the recorded budget but is absent",
        "best_so_far_objective": None,
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
    try:
        with resolved.open(encoding="utf-8") as handle:
            first_line = handle.readline()
        parsed = json.loads(first_line) if first_line else None
        if isinstance(parsed, dict):
            commit = _commit_value(parsed)
    except (OSError, UnicodeError, json.JSONDecodeError):
        commit = None
    evidence = {
        "evaluation_index": row["evaluation_index"],
        "path": _portable_path(resolved, repo_root=repo_root),
        "sha256": _file_sha256(resolved),
        "commit_sha": commit,
        "status": "available" if commit else "commit_unknown",
    }
    return evidence, commit, commit is None


def _source_evidence(
    manifest_path: Path,
    candidate_rows: list[dict[str, Any]],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    context = _read_execution_context(manifest_path)
    context_payload = context.get("payload") if isinstance(context, dict) else None
    context_sha = context_payload.get("commit_sha") if isinstance(context_payload, dict) else None
    context_sha = (
        context_sha.strip() if isinstance(context_sha, str) and context_sha.strip() else None
    )
    environment = None
    if isinstance(context_payload, dict):
        environment = {
            key: context_payload.get(key)
            for key in ("hostname", "cpu_model", "python_version", "platform", "thread_env")
            if context_payload.get(key) is not None
        }

    episode_evidence: list[dict[str, Any]] = []
    commits: set[str] = set()
    unresolved_refs = 0
    for row in candidate_rows:
        evidence, commit, unresolved = _episode_record_evidence(
            row,
            manifest_path=manifest_path,
            repo_root=repo_root,
        )
        if evidence is None:
            continue
        episode_evidence.append(evidence)
        if commit:
            commits.add(commit)
        unresolved_refs += int(unresolved)

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
    return {
        "status": "available",
        "config_sha256": _canonical_sha256(config),
        "search_space_sha256": _canonical_sha256(config["search_space"])
        if isinstance(config.get("search_space"), dict)
        else None,
        "files": files,
    }


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
) -> dict[str, Any]:
    evaluations: list[dict[str, Any]] = []
    best: float | None = None
    best_eligible: float | None = None
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
        if status == "scored" and score is not None:
            best = score if best is None else max(best, score)
            if eligible is True:
                best_eligible = score if best_eligible is None else max(best_eligible, score)
        error = item.get("error") if is_mapping else None
        evaluation = {
            "evaluation_index": index,
            "status": status,
            "failure_type": failure,
            "objective_value": score,
            "critical": critical,
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
            "error": error if isinstance(error, str) else str(error) if error is not None else None,
            "best_so_far_objective": best,
            "best_so_far_analysis_eligible_objective": best_eligible,
            "_raw": item,
        }
        evaluations.append(evaluation)
    actual_count = len(candidates)
    evaluations.extend(
        _missing_evaluation(index) for index in range(actual_count + 1, expected_slots + 1)
    )
    observed = evaluations[:actual_count]
    invalid = sum(item["status"] == "invalid" for item in observed)
    failed = sum(item["status"] == "failed" for item in observed)
    critical_types = Counter(
        str(item["failure_type"]) for item in observed if item["critical"] and item["failure_type"]
    )
    return {
        "evaluations": evaluations,
        "num_candidates": actual_count,
        "num_missing_evaluations": max(0, expected_slots - actual_count),
        "num_valid_candidates": actual_count - invalid - failed,
        "num_invalid_candidates": invalid,
        "num_failed_evaluations": failed,
        "num_scored_valid_candidates": sum(item["status"] == "scored" for item in observed),
        "num_scoreless_valid_candidates": sum(item["status"] == "scoreless" for item in observed),
        "num_critical_candidates": sum(item["critical"] for item in observed),
        "first_critical_evaluation": next(
            (item["evaluation_index"] for item in observed if item["critical"]), None
        ),
        "critical_failure_counts": dict(sorted(critical_types.items())),
        "num_duplicate_candidates": sum(bool(item["duplicate_basis"]) for item in observed),
        "num_candidate_spec_duplicates": sum(
            "candidate_spec" in item["duplicate_basis"] for item in observed
        ),
        "num_effective_scenario_duplicates": sum(
            "effective_scenario_hash" in item["duplicate_basis"] for item in observed
        ),
        "duplicate_rate_observed": sum(bool(item["duplicate_basis"]) for item in observed)
        / actual_count
        if actual_count
        else None,
        "invalid_rate_observed": invalid / actual_count if actual_count else None,
        "analysis_eligible_count": sum(item["analysis_eligible"] is True for item in observed),
        "analysis_ineligible_count": sum(item["analysis_eligible"] is False for item in observed),
        "analysis_eligibility_unknown_count": sum(
            item["analysis_eligible"] is None for item in observed
        ),
        "execution_mode_counts": _observed_status_counts(observed, "execution_mode"),
        "readiness_status_counts": _observed_status_counts(observed, "readiness_status"),
        "availability_status_counts": _observed_status_counts(observed, "availability_status"),
        "fallback_candidate_count": sum(
            _execution_risk_mode(item["_raw"]) == "fallback" for item in observed
        ),
        "degraded_candidate_count": sum(
            _execution_risk_mode(item["_raw"]) == "degraded" for item in observed
        ),
        "best_objective_value": best,
        "best_analysis_eligible_objective_value": best_eligible,
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
        "num_invalid_candidates": derived["num_invalid_candidates"],
        "num_failed_evaluations": derived["num_failed_evaluations"],
        "num_valid_candidates": derived["num_valid_candidates"],
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
    manifest_budget = _positive_int(config.get("budget"))
    manifest_seed = _integer(config.get("seed"))
    manifest_objective = config.get("objective")
    if manifest_budget is not None and manifest_budget != indexed_budget:
        warnings.append("comparison budget disagrees with search manifest config budget")
    if manifest_seed is not None and manifest_seed != seed:
        warnings.append("comparison seed disagrees with search manifest config seed")
    if isinstance(manifest_objective, str) and manifest_objective != objective:
        warnings.append("comparison objective disagrees with search manifest config objective")

    if isinstance(manifest, dict):
        raw_candidates = manifest.get("candidates")
        if not isinstance(raw_candidates, list):
            warnings.append("search manifest candidates must be an array")
            raw_candidates = []
            artifact["status"] = "malformed"
    else:
        raw_candidates = []
    expected_budget = manifest_budget or indexed_budget
    if len(raw_candidates) > expected_budget:
        warnings.append("candidate count exceeds the recorded run budget")
    expected_slots = max(expected_budget, indexed_budget)
    derived = _derive_evaluations(raw_candidates, expected_slots=expected_slots)
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
            "unresolved_episode_provenance_count": 0,
            "execution_context": None,
            "environment": None,
            "episode_records": [],
        }
    )
    config_provenance = (
        _config_provenance(config, manifest_path=manifest_path, repo_root=repo_root)
        if manifest_path is not None and isinstance(manifest, dict)
        else {"status": "unknown", "config_sha256": None, "search_space_sha256": None, "files": []}
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
    group_runs: list[dict[str, Any]],
) -> dict[str, Any]:
    observed = [run for run in group_runs if run["artifact_status"] == "available"]
    score_values = [
        run["best_objective_value"] for run in observed if run["best_objective_value"] is not None
    ]
    runtime_values = [
        run["runtime_seconds"] for run in observed if run["runtime_seconds"] is not None
    ]
    invalid = sum(run["num_invalid_candidates"] for run in group_runs)
    failed = sum(run["num_failed_evaluations"] for run in group_runs)
    valid = sum(run["num_valid_candidates"] for run in group_runs)
    critical = sum(run["num_critical_candidates"] for run in group_runs)
    duplicate = sum(run["num_duplicate_candidates"] for run in group_runs)
    attempted = sum(run["num_candidates"] for run in group_runs)
    curve: list[dict[str, Any]] = []
    for index in range(1, budget + 1):
        values = [
            point["best_so_far_objective"]
            for run in observed
            for point in run["evaluations"][: run["num_candidates"]]
            if point["evaluation_index"] == index and point["best_so_far_objective"] is not None
        ]
        curve.append(
            {
                "evaluation_index": index,
                "n_available_runs": len(values),
                "median_best_so_far": statistics.median(values) if values else None,
                "min_best_so_far": min(values) if values else None,
                "max_best_so_far": max(values) if values else None,
                "aggregation": "descriptive_run_range",
            }
        )
    seeds = sorted({run["seed"] for run in group_runs})
    return {
        "objective": objective,
        "sampler": sampler,
        "method": _sampler_label(sampler),
        "budget": budget,
        "run_count": len(group_runs),
        "available_manifest_count": len(observed),
        "seed_count": len(seeds),
        "seeds": seeds,
        "final_best_objective": {
            "median": statistics.median(score_values) if score_values else None,
            "min": min(score_values) if score_values else None,
            "max": max(score_values) if score_values else None,
            "n": len(score_values),
            "interpretation": "descriptive only; no inferential test performed",
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
            "valid_total_minus_invalid_minus_failed": valid,
            "invalid": invalid,
            "failed": failed,
            "critical": critical,
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
        pairs, unavailable, ambiguous_random, ambiguous_tpe = _pair_seed_runs(
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
) -> tuple[list[dict[str, Any]], list[int], list[int], list[int]]:
    pairs: list[dict[str, Any]] = []
    unavailable: list[int] = []
    ambiguous_random: list[int] = []
    ambiguous_tpe: list[int] = []
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
        random_score = random_run["best_objective_value"]
        tpe_score = tpe_run["best_objective_value"]
        pairs.append(
            {
                "seed": seed,
                "random_final_best": random_score,
                "tpe_final_best": tpe_score,
                "tpe_minus_random": tpe_score - random_score
                if random_score is not None and tpe_score is not None
                else None,
            }
        )
    return pairs, unavailable, ambiguous_random, ambiguous_tpe


def _aggregate_runs(
    runs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        groups[(run["objective"], run["sampler"], run["budget"])].append(run)
    aggregates = [
        _aggregate_group(objective, sampler, budget, group_runs)
        for (objective, sampler, budget), group_runs in sorted(groups.items())
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
        "| Objective | Method | Seed | Budget | Best objective | First critical eval | Critical | Valid / invalid / failed / scoreless / missing | Duplicates | Execution modes | Availability | Runtime (s) | Artifact |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|",
    ]
    for run in report["runs"]:
        accounting = (
            f"{run['num_valid_candidates']} / {run['num_invalid_candidates']} / "
            f"{run['num_failed_evaluations']} / {run['num_scoreless_valid_candidates']} / "
            f"{run['num_missing_evaluations']}"
        )
        duplicate = (
            f"{run['num_duplicate_candidates']}/{run['num_candidates']}"
            if run["num_candidates"]
            else "Not recorded"
        )
        execution_modes = _render_status_counts(run["execution_mode_counts"])
        availability = _render_status_counts(run["availability_status_counts"])
        lines.append(
            f"| `{run['objective']}` | {run['method']} | {run['seed']} | {run['budget']} | "
            f"{_display_number(run['best_objective_value'])} | "
            f"{run['first_critical_evaluation'] if run['first_critical_evaluation'] is not None else 'None found'} | "
            f"{run['num_critical_candidates']} | {accounting} | {duplicate} | "
            f"{execution_modes} | {availability} | "
            f"{_display_number(run['runtime_seconds'])} | {run['artifact_status']} |"
        )
    if not report["runs"]:
        lines.append(
            "| — | — | — | — | Not recorded | — | — | — | — | — | — | Not recorded | no runs |"
        )
    lines.extend(
        [
            "",
            "Valid candidates are derived as `candidate rows - invalid - failed`; scoreless valid candidates remain in the valid count.",
            "",
            "Invalid, failed, scoreless and missing evaluation attempts remain explicit and are not dropped from candidate accounting.",
            "",
            "## Matched Random vs TPE",
            "",
        ]
    )
    lines.extend(
        [
            "| Objective | Budget | Matched seeds | TPE − Random median | Observed delta range | Inference |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for comparison in report["random_vs_tpe"]:
        low = _display_number(comparison["tpe_minus_random_min"])
        high = _display_number(comparison["tpe_minus_random_max"])
        range_text = f"{low} to {high}" if comparison["matched_seed_count"] else "Not available"
        lines.append(
            f"| `{comparison['objective']}` | {comparison['budget']} | "
            f"{comparison['matched_seed_count']} | "
            f"{_display_number(comparison['tpe_minus_random_median'])} | {range_text} | "
            "Not performed; descriptive only |"
        )
    if not report["random_vs_tpe"]:
        lines.append("| — | — | 0 | Not available | Not available | Not performed |")
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


def _aggregate_point(runs: list[dict[str, Any]], index: int) -> list[float]:
    values: list[float] = []
    for run in runs:
        for point in run["evaluations"][: run["num_candidates"]]:
            if point["evaluation_index"] == index and point["best_so_far_objective"] is not None:
                values.append(float(point["best_so_far_objective"]))
                break
    return values


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
            for sampler in method_order:
                method_runs = [
                    run
                    for run in objective_runs
                    if run["budget"] == budget
                    and run["sampler"] == sampler
                    and run["artifact_status"] == "available"
                ]
                x_values = list(range(1, budget + 1))
                medians: list[float] = []
                minima: list[float] = []
                maxima: list[float] = []
                counts: list[int] = []
                for index in x_values:
                    values = _aggregate_point(method_runs, index)
                    counts.append(len(values))
                    medians.append(statistics.median(values) if values else math.nan)
                    minima.append(min(values) if values else math.nan)
                    maxima.append(max(values) if values else math.nan)
                seed_count = len({run["seed"] for run in method_runs})
                axis.plot(
                    x_values,
                    medians,
                    marker=".",
                    linewidth=1.8,
                    label=f"{_sampler_label(sampler)} (n={seed_count} seed runs)",
                )
                if seed_count > 1:
                    axis.fill_between(x_values, minima, maxima, alpha=0.15)
            axis.set_title(f"Budget {budget}")
            axis.set_xlabel("Evaluation index")
            axis.set_ylabel("Best-so-far objective (maximize)")
            axis.set_xlim(1, max(1, budget))
            axis.grid(True, alpha=0.25)
            axis.legend(loc="best")
        for axis in flat_axes[len(budgets) :]:
            axis.set_visible(False)
        fig.suptitle(
            f"{objective}: recorded best-so-far by evaluation budget\n"
            "Invalid, failed, scoreless and missing attempts remain in the companion report",
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
