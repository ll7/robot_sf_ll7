"""Fail-closed admission of the canonical 14-arm release runtime smoke."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.camera_ready._run_state import _resolve_integrity_artifact_path
from robot_sf.benchmark.camera_ready_campaign import load_campaign_config
from robot_sf.benchmark.checkpoint_staging_receipt import (
    CheckpointStagingReceiptError,
    validate_checkpoint_staging_receipt,
)
from robot_sf.benchmark.fallback_policy import runtime_fallback_or_degraded_marker
from robot_sf.benchmark.identity.hash_utils import sha256_file

RUNTIME_SMOKE_RELEASE_ID = "paper_experiment_matrix_v2_h600_s30_runtime_smoke_v0_2"
RUNTIME_SMOKE_MANIFEST = Path(
    "configs/benchmarks/releases/paper_experiment_matrix_v2_h600_s30_runtime_smoke_v0_2.yaml"
)
RUNTIME_SMOKE_CONFIG = Path(
    "configs/benchmarks/paper_experiment_matrix_v2_h600_s30_runtime_smoke.yaml"
)
RUNTIME_SMOKE_HORIZON = 600
RUNTIME_SMOKE_KINEMATICS = "differential_drive"


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


def _read_yaml_object(path: Path, label: str) -> dict[str, Any]:
    """Read one YAML mapping with a bounded public error.

    Returns:
        Parsed YAML object.
    """
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeSmokeAdmissionError(f"{label} is missing or invalid") from exc
    if not isinstance(payload, dict):
        raise RuntimeSmokeAdmissionError(f"{label} is not a YAML object")
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


def _strict_int(value: Any) -> int | None:
    """Return an integer without accepting booleans or fractional values."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value)
    return None


def _source_commit(row: dict[str, Any]) -> str:
    """Resolve the exact source commit recorded by an episode row.

    Returns:
        Normalized commit token, or an empty string when absent.
    """
    provenance = row.get("result_provenance")
    if isinstance(provenance, dict) and provenance.get("repo_commit"):
        return str(provenance["repo_commit"]).strip().lower()
    return str(row.get("git_hash", "")).strip().lower()


def _episode_horizon(row: dict[str, Any]) -> int | None:
    """Resolve the authoritative episode horizon.

    Returns:
        Parsed horizon, or ``None`` when absent or malformed.
    """
    if row.get("horizon") is not None:
        return _strict_int(row["horizon"])
    provenance = row.get("result_provenance")
    if isinstance(provenance, dict):
        settings = provenance.get("simulator_settings")
        if isinstance(settings, dict):
            return _strict_int(settings.get("horizon"))
    return None


def _read_episode_rows(path: Path) -> list[dict[str, Any]]:
    """Read a smoke episode JSONL artifact without accepting malformed rows.

    Returns:
        Parsed episode row objects.
    """
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, 1):
                if not raw_line.strip():
                    continue
                payload = json.loads(raw_line)
                if not isinstance(payload, dict):
                    raise RuntimeSmokeAdmissionError(
                        f"runtime smoke episode row {path}:{line_number} is not an object"
                    )
                rows.append(payload)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeSmokeAdmissionError(
            "runtime smoke episode artifact is missing or invalid"
        ) from exc
    return rows


def _canonical_smoke_contract(  # noqa: C901
    *, repo_root: Path, expected_planner_keys: tuple[str, ...]
) -> tuple[Path, Path, str, int, dict[str, str]]:
    """Resolve the tracked smoke axes and planner algorithms from canonical inputs.

    Returns:
        Manifest path, config path, scenario ID, seed, and planner-to-algorithm mapping.
    """
    manifest_path = repo_root / RUNTIME_SMOKE_MANIFEST
    config_path = repo_root / RUNTIME_SMOKE_CONFIG
    manifest = _read_yaml_object(manifest_path, "canonical runtime smoke manifest")
    config = _read_yaml_object(config_path, "canonical runtime smoke config")
    if manifest.get("release_id") != RUNTIME_SMOKE_RELEASE_ID:
        raise RuntimeSmokeAdmissionError("canonical runtime smoke release identity mismatch")
    if manifest.get("campaign_config_sha256") != sha256_file(config_path):
        raise RuntimeSmokeAdmissionError("canonical runtime smoke config pin mismatch")
    manifest_planners = manifest.get("planners")
    manifest_planners = manifest_planners if isinstance(manifest_planners, dict) else {}
    config_planners = config.get("planners")
    config_planners = config_planners if isinstance(config_planners, list) else []
    enabled = [
        entry for entry in config_planners if isinstance(entry, dict) and entry.get("enabled", True)
    ]
    config_keys = tuple(str(entry.get("key", "")).strip() for entry in enabled)
    manifest_keys = tuple(str(key).strip() for key in manifest_planners.get("keys", []))
    if manifest_keys != expected_planner_keys or config_keys != expected_planner_keys:
        raise RuntimeSmokeAdmissionError("canonical runtime smoke planner roster mismatch")
    if _strict_int(config.get("horizon")) != RUNTIME_SMOKE_HORIZON:
        raise RuntimeSmokeAdmissionError("canonical runtime smoke horizon mismatch")
    if tuple(config.get("kinematics_matrix") or ()) != (RUNTIME_SMOKE_KINEMATICS,):
        raise RuntimeSmokeAdmissionError("canonical runtime smoke kinematics mismatch")
    manifest_kinematics = manifest.get("kinematics")
    manifest_kinematics = manifest_kinematics if isinstance(manifest_kinematics, dict) else {}
    if tuple(manifest_kinematics.get("matrix") or ()) != (RUNTIME_SMOKE_KINEMATICS,):
        raise RuntimeSmokeAdmissionError("runtime smoke manifest kinematics mismatch")
    seed_policy = config.get("seed_policy")
    seed_policy = seed_policy if isinstance(seed_policy, dict) else {}
    seeds = seed_policy.get("seeds")
    if not isinstance(seeds, list) or len(seeds) != 1 or _strict_int(seeds[0]) is None:
        raise RuntimeSmokeAdmissionError("canonical runtime smoke seed contract mismatch")
    scenario = manifest.get("scenario")
    scenario = scenario if isinstance(scenario, dict) else {}
    scenario_rel = scenario.get("matrix_path")
    if not isinstance(scenario_rel, str) or not scenario_rel.strip():
        raise RuntimeSmokeAdmissionError("canonical runtime smoke scenario path is missing")
    scenario_path = (manifest_path.parent / scenario_rel).resolve()
    if not scenario_path.is_file():
        raise RuntimeSmokeAdmissionError("canonical runtime smoke scenario is missing")
    if scenario.get("matrix_sha256") != sha256_file(scenario_path):
        raise RuntimeSmokeAdmissionError("canonical runtime smoke scenario pin mismatch")
    scenario_payload = _read_yaml_object(scenario_path, "canonical runtime smoke scenario")
    scenarios = scenario_payload.get("scenarios")
    if not isinstance(scenarios, list) or len(scenarios) != 1 or not isinstance(scenarios[0], dict):
        raise RuntimeSmokeAdmissionError(
            "canonical runtime smoke must resolve exactly one scenario"
        )
    scenario_id = str(scenarios[0].get("name") or scenarios[0].get("id") or "").strip()
    if not scenario_id:
        raise RuntimeSmokeAdmissionError("canonical runtime smoke scenario identifier is missing")
    algorithms = {str(entry["key"]): str(entry.get("algo", "")) for entry in enabled}
    return manifest_path, config_path, scenario_id, int(seeds[0]), algorithms


def validate_runtime_smoke_result(  # noqa: C901, PLR0912, PLR0915
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
    manifest_path, config_path, scenario_id, seed, algorithms = _canonical_smoke_contract(
        repo_root=resolved_repo,
        expected_planner_keys=expected_planner_keys,
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

    checkpoint = result.get("checkpoint_staging_receipt")
    checkpoint = checkpoint if isinstance(checkpoint, dict) else {}
    checkpoint_rel = checkpoint.get("path")
    if not isinstance(checkpoint_rel, str) or not checkpoint_rel.strip():
        problems.append("checkpoint staging receipt path is missing")
    else:
        checkpoint_path = (resolved_repo / checkpoint_rel).resolve()
        if not checkpoint_path.is_relative_to(resolved_repo) or not checkpoint_path.is_file():
            problems.append("checkpoint staging receipt path is invalid")
        else:
            _require_equal(
                problems,
                checkpoint.get("sha256"),
                sha256_file(checkpoint_path),
                "checkpoint staging receipt hash",
            )
            try:
                cfg = load_campaign_config(config_path)
                admitted_checkpoint = validate_checkpoint_staging_receipt(
                    cfg,
                    checkpoint_path,
                    campaign_config_path=config_path,
                    max_age_hours=max_age_hours,
                )
            except (OSError, ValueError, CheckpointStagingReceiptError) as exc:
                problems.append(f"checkpoint staging receipt rejected: {exc}")
            else:
                _require_equal(
                    problems,
                    admitted_checkpoint.get("submit_safe"),
                    True,
                    "checkpoint submit_safe",
                )
                _require_equal(
                    problems,
                    checkpoint.get("submit_safe"),
                    True,
                    "release result checkpoint submit_safe",
                )

    campaign = summary.get("campaign")
    campaign = campaign if isinstance(campaign, dict) else {}
    expected_rows = len(expected_planner_keys)
    for field, expected in (
        ("campaign_id", result.get("campaign_id")),
        ("git_hash", expected_source_commit),
        ("total_runs", expected_rows),
        ("successful_runs", expected_rows),
        ("total_episodes", expected_rows),
        ("non_success_runs", 0),
        ("accepted_unavailable_runs", 0),
        ("unexpected_failed_runs", 0),
        ("benchmark_success", True),
        ("campaign_execution_status", "completed"),
        ("evidence_status", "valid"),
    ):
        _require_equal(problems, campaign.get(field), expected, f"campaign {field}")

    runs = summary.get("runs")
    planner_rows = summary.get("planner_rows")
    if not isinstance(runs, list):
        runs = []
        problems.append("runtime smoke summary runs are missing")
    if not isinstance(planner_rows, list):
        planner_rows = []
        problems.append("runtime smoke summary planner_rows are missing")
    observed_arms: list[str] = []
    observed_episode_identities: list[tuple[str, str, int]] = []
    fallback_markers: list[str] = []
    for index, entry in enumerate(runs):
        if not isinstance(entry, dict):
            problems.append(f"runtime smoke run {index} is not an object")
            continue
        planner = entry.get("planner")
        planner = planner if isinstance(planner, dict) else {}
        planner_key = str(planner.get("key", "")).strip()
        observed_arms.append(planner_key)
        _require_equal(problems, entry.get("status"), "ok", f"run {index} status")
        _require_equal(
            problems,
            planner.get("kinematics"),
            RUNTIME_SMOKE_KINEMATICS,
            f"run {index} kinematics",
        )
        _require_equal(
            problems,
            _strict_int(planner.get("horizon")),
            RUNTIME_SMOKE_HORIZON,
            f"run {index} horizon",
        )
        _require_equal(
            problems, planner.get("algo"), algorithms.get(planner_key), f"run {index} algorithm"
        )
        run_marker = runtime_fallback_or_degraded_marker(entry)
        if run_marker is not None:
            fallback_markers.append(f"runs[{index}].{run_marker[0]}={run_marker[1]}")
        raw_episodes_path = entry.get("episodes_path")
        if not isinstance(raw_episodes_path, str) or not raw_episodes_path.strip():
            problems.append(f"run {index} episodes_path is missing")
            continue
        try:
            episodes_path = _resolve_integrity_artifact_path(campaign_root, raw_episodes_path)
            rows = _read_episode_rows(episodes_path)
        except (OSError, ValueError, RuntimeSmokeAdmissionError) as exc:
            problems.append(f"run {index} episode artifact rejected: {exc}")
            continue
        if len(rows) != 1:
            problems.append(f"run {index} must contain exactly one episode row")
        declared = entry.get("summary")
        declared = declared if isinstance(declared, dict) else {}
        declared_count = declared.get("episodes_total", declared.get("written"))
        _require_equal(problems, _strict_int(declared_count), len(rows), f"run {index} row count")
        _require_equal(
            problems, _strict_int(declared.get("failed_jobs")), 0, f"run {index} failures"
        )
        _require_equal(problems, declared.get("failures"), [], f"run {index} failure list")
        for row_index, row in enumerate(rows):
            marker = runtime_fallback_or_degraded_marker(row)
            if marker is not None:
                fallback_markers.append(f"runs[{index}].rows[{row_index}].{marker[0]}={marker[1]}")
            _require_equal(
                problems,
                _source_commit(row),
                expected_source_commit,
                f"run {index} episode source commit",
            )
            _require_equal(
                problems,
                _episode_horizon(row),
                RUNTIME_SMOKE_HORIZON,
                f"run {index} episode horizon",
            )
            _require_equal(problems, row.get("scenario_id"), scenario_id, f"run {index} scenario")
            _require_equal(problems, _strict_int(row.get("seed")), seed, f"run {index} seed")
            metadata = row.get("algorithm_metadata")
            metadata = metadata if isinstance(metadata, dict) else {}
            _require_equal(
                problems,
                metadata.get("algorithm"),
                algorithms.get(planner_key),
                f"run {index} algorithm",
            )
            observed_episode_identities.append((planner_key, scenario_id, seed))

    if tuple(observed_arms) != expected_planner_keys or len(set(observed_arms)) != expected_rows:
        problems.append("runtime smoke raw planner arms do not match the exact roster")
    expected_identities = {(key, scenario_id, seed) for key in expected_planner_keys}
    if (
        len(observed_episode_identities) != expected_rows
        or set(observed_episode_identities) != expected_identities
    ):
        problems.append("runtime smoke raw episode identities do not match the exact matrix")
    if fallback_markers:
        problems.append(
            "runtime smoke contains fallback or degraded markers: " + fallback_markers[0]
        )

    campaign_row_status = campaign.get("row_status_summary")
    campaign_row_status = campaign_row_status if isinstance(campaign_row_status, dict) else {}
    for field, expected in (
        ("successful_evidence_rows", expected_rows),
        ("accepted_unavailable_rows", 0),
        ("unexpected_failed_rows", 0),
        ("fallback_or_degraded_rows", 0),
    ):
        _require_equal(
            problems,
            _strict_int(campaign_row_status.get(field)),
            expected,
            f"campaign row status {field}",
        )

    planner_row_arms: list[str] = []
    for index, row in enumerate(planner_rows):
        if not isinstance(row, dict):
            problems.append(f"runtime smoke planner row {index} is not an object")
            continue
        planner_row_arms.append(str(row.get("planner_key", "")).strip())
        _require_equal(problems, row.get("status"), "ok", f"planner row {index} status")
        _require_equal(
            problems, _strict_int(row.get("episodes")), 1, f"planner row {index} episodes"
        )
        marker = runtime_fallback_or_degraded_marker(row)
        if marker is not None:
            problems.append(f"planner row {index} contains fallback or degraded marker")
    if (
        tuple(planner_row_arms) != expected_planner_keys
        or len(set(planner_row_arms)) != expected_rows
    ):
        problems.append("runtime smoke planner aggregates do not match the exact roster")

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
        "checkpoint_receipt_sha256": checkpoint["sha256"],
        "source_commit": expected_source_commit,
        "campaign_id": result["campaign_id"],
        "finished_at_utc": finished_at_utc,
        "planner_arms": expected_rows,
        "episode_cells": len(observed_episode_identities),
        "fallback_or_degraded_rows": 0,
    }


__all__ = ["RuntimeSmokeAdmissionError", "validate_runtime_smoke_result"]
