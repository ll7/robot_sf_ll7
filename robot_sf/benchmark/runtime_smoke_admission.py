"""Fail-closed admission of the canonical 14-arm release runtime smoke."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from robot_sf.benchmark.identity.hash_utils import sha256_file

RUNTIME_SMOKE_RELEASE_ID = "paper_experiment_matrix_v2_h600_s30_runtime_smoke_v0_2"
RUNTIME_SMOKE_MANIFEST = Path(
    "configs/benchmarks/releases/paper_experiment_matrix_v2_h600_s30_runtime_smoke_v0_2.yaml"
)
RUNTIME_SMOKE_CONFIG = Path(
    "configs/benchmarks/paper_experiment_matrix_v2_h600_s30_runtime_smoke.yaml"
)


class RuntimeSmokeAdmissionError(ValueError):
    """Raised when runtime-smoke evidence cannot admit a full release run."""


def _read_object(path: Path, label: str) -> dict[str, Any]:
    """Read one JSON mapping with a bounded public error.

    Returns:
        Parsed JSON object.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeSmokeAdmissionError(f"{label} is missing or invalid") from exc
    if not isinstance(payload, dict):
        raise RuntimeSmokeAdmissionError(f"{label} is not a JSON object")
    return payload


def _require_equal(problems: list[str], actual: Any, expected: Any, label: str) -> None:
    """Append a stable mismatch description."""
    if actual != expected:
        problems.append(f"{label} mismatch")


def _validate_age(run_meta: dict[str, Any], *, max_age_hours: float) -> str:
    """Return the normalized smoke completion timestamp or reject stale evidence."""
    raw = str(run_meta.get("finished_at_utc", "")).strip()
    try:
        finished = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RuntimeSmokeAdmissionError("runtime smoke completion timestamp is invalid") from exc
    if finished.tzinfo is None:
        finished = finished.replace(tzinfo=UTC)
    age_hours = (datetime.now(UTC) - finished.astimezone(UTC)).total_seconds() / 3600
    if age_hours < 0 or age_hours > max_age_hours:
        raise RuntimeSmokeAdmissionError("runtime smoke result is stale or future-dated")
    return finished.astimezone(UTC).isoformat().replace("+00:00", "Z")


def validate_runtime_smoke_result(
    result_path: Path,
    *,
    repo_root: Path,
    expected_source_commit: str,
    expected_planner_keys: tuple[str, ...],
    max_age_hours: float = 24.0,
) -> dict[str, Any]:
    """Validate a byte-addressable smoke result before a full v0.2 campaign.

    Returns:
        Sanitized admission metadata suitable for release provenance and launch packets.
    """
    resolved_repo = repo_root.resolve()
    resolved_result = result_path.resolve()
    if not resolved_result.is_relative_to(resolved_repo):
        raise RuntimeSmokeAdmissionError("runtime smoke result must be inside the release worktree")
    result = _read_object(resolved_result, "runtime smoke result")
    campaign_root = resolved_result.parent.parent
    run_meta = _read_object(campaign_root / "run_meta.json", "runtime smoke run metadata")
    summary = _read_object(
        campaign_root / "reports" / "campaign_summary.json", "runtime smoke campaign summary"
    )

    problems: list[str] = []
    release = result.get("benchmark_release")
    release = release if isinstance(release, dict) else {}
    _require_equal(problems, release.get("release_id"), RUNTIME_SMOKE_RELEASE_ID, "release_id")
    _require_equal(
        problems,
        release.get("manifest_path"),
        RUNTIME_SMOKE_MANIFEST.as_posix(),
        "runtime smoke manifest path",
    )
    _require_equal(
        problems,
        release.get("canonical_campaign_config"),
        RUNTIME_SMOKE_CONFIG.as_posix(),
        "runtime smoke config path",
    )
    manifest_path = resolved_repo / RUNTIME_SMOKE_MANIFEST
    config_path = resolved_repo / RUNTIME_SMOKE_CONFIG
    if not manifest_path.is_file() or not config_path.is_file():
        problems.append("canonical runtime smoke inputs are missing")
    else:
        _require_equal(
            problems,
            release.get("manifest_sha256"),
            sha256_file(manifest_path),
            "runtime smoke manifest hash",
        )
        _require_equal(
            problems,
            release.get("canonical_campaign_config_sha256"),
            sha256_file(config_path),
            "runtime smoke config hash",
        )

    repo = run_meta.get("repo")
    repo = repo if isinstance(repo, dict) else {}
    _require_equal(problems, repo.get("commit"), expected_source_commit, "source commit")
    _require_equal(problems, run_meta.get("campaign_id"), result.get("campaign_id"), "campaign id")
    resolved = result.get("resolved_manifest")
    resolved = resolved if isinstance(resolved, dict) else {}
    planners = resolved.get("planners")
    planners = planners if isinstance(planners, dict) else {}
    _require_equal(
        problems, tuple(planners.get("keys") or ()), expected_planner_keys, "planner roster"
    )
    expected_rows = len(expected_planner_keys)
    for field in ("total_runs", "successful_runs", "total_episodes"):
        _require_equal(problems, result.get(field), expected_rows, field)
    for field in ("non_success_runs", "accepted_unavailable_runs", "unexpected_failed_runs"):
        _require_equal(problems, result.get(field), 0, field)
    row_status = result.get("row_status_summary")
    row_status = row_status if isinstance(row_status, dict) else {}
    _require_equal(
        problems,
        row_status.get("successful_evidence_rows"),
        expected_rows,
        "successful evidence rows",
    )
    for field in (
        "accepted_unavailable_rows",
        "unexpected_failed_rows",
        "fallback_or_degraded_rows",
    ):
        _require_equal(problems, row_status.get(field), 0, field)
    integrity = result.get("campaign_integrity")
    integrity = integrity if isinstance(integrity, dict) else {}
    _require_equal(problems, integrity.get("status"), "valid", "campaign integrity")
    _require_equal(problems, integrity.get("checked_arm_count"), expected_rows, "checked arm count")
    checkpoint = result.get("checkpoint_staging_receipt")
    checkpoint = checkpoint if isinstance(checkpoint, dict) else {}
    _require_equal(problems, checkpoint.get("submit_safe"), True, "checkpoint submit_safe")
    campaign = summary.get("campaign")
    campaign = campaign if isinstance(campaign, dict) else {}
    _require_equal(problems, campaign.get("benchmark_success"), True, "campaign benchmark success")
    _require_equal(problems, result.get("release_benchmark_success"), True, "release success")
    _require_equal(problems, result.get("release_status"), "ok", "release status")
    _require_equal(problems, result.get("release_exit_code"), 0, "release exit code")
    if problems:
        raise RuntimeSmokeAdmissionError("runtime smoke admission failed: " + "; ".join(problems))
    finished_at_utc = _validate_age(run_meta, max_age_hours=max_age_hours)
    return {
        "schema_version": "benchmark-runtime-smoke-admission.v1",
        "status": "admitted",
        "result_sha256": sha256_file(resolved_result),
        "source_commit": expected_source_commit,
        "campaign_id": result["campaign_id"],
        "finished_at_utc": finished_at_utc,
        "planner_arms": expected_rows,
        "episode_cells": expected_rows,
        "fallback_or_degraded_rows": 0,
    }


__all__ = ["RuntimeSmokeAdmissionError", "validate_runtime_smoke_result"]
