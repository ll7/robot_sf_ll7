#!/usr/bin/env python3
"""Materialize a replayable slice from a provenance-checked benchmark showcase.

Case ranking and diversity selection are owned by ``benchmark_showcase.py``. This
consumer verifies its selected rows against the durable campaign payload, writes
per-case source snapshots and canonical single-scenario commands, and can run at
most five representative replays when explicitly requested.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

SCHEMA_VERSION = "benchmark-hard-case-slice.v1"
SHOWCASE_SCHEMA = "benchmark-showcase.v1"
CASE_ID_RE = re.compile(r"case-[0-9a-f]{16}\Z")
MAX_CASES = 50
MAX_REPLAYS = 5
EVENT_FIELDS = ("collision_event", "timeout_event", "route_complete")
METRIC_FIELDS = (
    "collisions",
    "total_collision_count",
    "near_misses",
    "force_exceed_events",
    "comfort_exposure",
    "clearing_distance_min",
    "time_to_goal_norm",
    "path_efficiency",
)
REPO_ROOT = Path(__file__).resolve().parents[2]


class MaterializationError(RuntimeError):
    """An input or provenance check failed before materialization."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MaterializationError(f"could not read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MaterializationError(f"JSON root must be an object: {path}")
    return value


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _safe_relative(root: Path, value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise MaterializationError(f"{label} must be a non-empty relative path")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise MaterializationError(f"unsafe {label}: {value!r}")
    result = (root / Path(*relative.parts)).resolve()
    try:
        result.relative_to(root.resolve())
    except ValueError as exc:
        raise MaterializationError(f"{label} escapes its root: {value!r}") from exc
    return result


def _sanitize(value: Any, path: str = "$") -> tuple[Any, list[dict[str, str]]]:
    """Convert non-finite JSON extensions to null while recording each source field."""
    missing: list[dict[str, str]] = []
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, child in value.items():
            cleaned_child, child_missing = _sanitize(child, f"{path}.{key}")
            cleaned[key] = cleaned_child
            missing.extend(child_missing)
        return cleaned, missing
    if isinstance(value, list):
        cleaned_list = []
        for index, child in enumerate(value):
            cleaned_child, child_missing = _sanitize(child, f"{path}[{index}]")
            cleaned_list.append(cleaned_child)
            missing.extend(child_missing)
        return cleaned_list, missing
    if isinstance(value, float) and not math.isfinite(value):
        missing.append({"field": path, "source_value": repr(value)})
        return None, missing
    return value, missing


def _load_source_row(
    campaign_root: Path, case: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = case.get("source")
    if not isinstance(source, dict):
        raise MaterializationError(f"{case.get('case_id')}: selected case has no source reference")
    episode_path = _safe_relative(campaign_root, source.get("episode_file"), label="episode file")
    if not episode_path.is_file():
        raise MaterializationError(f"{case.get('case_id')}: source episode file is missing")
    source_file_hash = source.get("episode_file_sha256")
    if source_file_hash != _sha256(episode_path):
        raise MaterializationError(f"{case.get('case_id')}: source episode file hash mismatch")
    line_number = source.get("line_number")
    if not isinstance(line_number, int) or isinstance(line_number, bool) or line_number < 1:
        raise MaterializationError(f"{case.get('case_id')}: invalid source line number")
    lines = episode_path.read_bytes().splitlines(keepends=True)
    if line_number > len(lines):
        raise MaterializationError(f"{case.get('case_id')}: source line is missing")
    raw_line = lines[line_number - 1]
    record_hash = hashlib.sha256(raw_line).hexdigest()
    if record_hash != source.get("record_sha256"):
        raise MaterializationError(f"{case.get('case_id')}: source record hash mismatch")
    try:
        row = json.loads(raw_line)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise MaterializationError(
            f"{case.get('case_id')}: source row is malformed: {exc}"
        ) from exc
    if not isinstance(row, dict):
        raise MaterializationError(f"{case.get('case_id')}: source row is not an object")
    identities = {
        "episode_id": (case.get("episode_id"), row.get("episode_id")),
        "scenario_id": (case.get("scenario_id"), row.get("scenario_id")),
        "seed": (case.get("seed"), row.get("seed")),
        "planner_key": (case.get("planner_key"), row.get("algo")),
    }
    mismatches = [key for key, pair in identities.items() if pair[0] != pair[1]]
    if mismatches:
        raise MaterializationError(
            f"{case.get('case_id')}: showcase/source identity mismatch: {', '.join(mismatches)}"
        )
    return row, {
        "episode_file": str(source.get("episode_file")),
        "episode_file_sha256": source_file_hash,
        "line_number": line_number,
        "record_sha256": record_hash,
    }


def _verify_bundle(source: dict[str, Any], bundle: Path | None) -> str | None:
    expected_bundle_hash = source.get("bundle_sha256")
    if bundle is not None:
        actual_bundle_hash = _sha256(bundle)
        if expected_bundle_hash and actual_bundle_hash != expected_bundle_hash:
            raise MaterializationError(
                "supplied bundle does not match showcase source bundle SHA-256"
            )
    elif expected_bundle_hash:
        raise MaterializationError(
            "a source bundle was declared; pass --bundle to verify its SHA-256"
        )
    return actual_bundle_hash if bundle is not None else None


def _verify_source_files(source: dict[str, Any], campaign_root: Path) -> list[str]:
    checksums = source.get("source_files_sha256")
    if not isinstance(checksums, dict) or not checksums:
        raise MaterializationError("showcase summary has no source file checksum inventory")
    verified_files = []
    for relative, expected in sorted(checksums.items()):
        file_path = _safe_relative(campaign_root, relative, label="source artifact")
        if (
            not file_path.is_file()
            or not isinstance(expected, str)
            or _sha256(file_path) != expected
        ):
            raise MaterializationError(f"source artifact missing or checksum mismatch: {relative}")
        verified_files.append(relative)
    return verified_files


def _verify_campaign_identity(source: dict[str, Any], campaign_root: Path) -> str:
    campaign = _read_object(campaign_root / "campaign_manifest.json")
    if campaign.get("campaign_id") != source.get("campaign_id"):
        raise MaterializationError("campaign ID does not match showcase summary")
    campaign_git = campaign.get("git") if isinstance(campaign.get("git"), dict) else {}
    source_revision = campaign_git.get("commit") or campaign.get("git_hash")
    if source_revision != source.get("campaign_source_revision"):
        raise MaterializationError("campaign source revision does not match showcase summary")
    return source_revision


def _verify_matrix(source: dict[str, Any], campaign_root: Path, matrix: Path) -> tuple[str, str]:
    release_path = campaign_root / "release" / "release_manifest.resolved.json"
    release_manifest = _read_object(release_path)
    scenario_info = release_manifest.get("scenario")
    if not isinstance(scenario_info, dict):
        raise MaterializationError("resolved release manifest has no scenario identity")
    scenario_path = scenario_info.get("matrix_path")
    scenario_hash = scenario_info.get("matrix_sha256")
    if scenario_path != source.get("scenario_matrix"):
        raise MaterializationError("scenario matrix path differs from showcase summary")
    if not isinstance(scenario_hash, str) or _sha256(matrix) != scenario_hash:
        raise MaterializationError(
            "current scenario matrix does not match the pinned release matrix"
        )
    if matrix.resolve() != _safe_relative(REPO_ROOT, scenario_path, label="scenario matrix"):
        raise MaterializationError("--matrix must be the canonical in-repository matrix path")
    embedded = source.get("embedded_checksums")
    if not isinstance(embedded, dict) or embedded.get("status") != "passed":
        raise MaterializationError(
            "showcase summary did not pass embedded bundle checksum validation"
        )
    return scenario_path, scenario_hash


def _verify_inputs(
    summary: dict[str, Any], campaign_root: Path, bundle: Path | None, matrix: Path
) -> dict[str, Any]:
    source = summary.get("source")
    if summary.get("schema_version") != SHOWCASE_SCHEMA or not isinstance(source, dict):
        raise MaterializationError(f"input must be a {SHOWCASE_SCHEMA} summary")
    actual_bundle_hash = _verify_bundle(source, bundle)
    verified_files = _verify_source_files(source, campaign_root)
    source_revision = _verify_campaign_identity(source, campaign_root)
    scenario_path, scenario_hash = _verify_matrix(source, campaign_root, matrix)
    return {
        "campaign_id": source.get("campaign_id"),
        "bundle_sha256": actual_bundle_hash or source.get("bundle_sha256"),
        "source_revision": source_revision,
        "matrix_path": scenario_path,
        "matrix_sha256": scenario_hash,
        "source_files_verified": verified_files,
        "embedded_checksums_status": source.get("embedded_checksums", {}).get("status"),
    }


def _compact_source_showcase_diagnostics(summary: dict[str, Any]) -> dict[str, Any]:
    """Retain upstream accounting and contradiction checks without local path details."""
    accounting = summary.get("accounting")
    if not isinstance(accounting, dict):
        return {}
    run_statuses = accounting.get("run_statuses")
    analyzer = summary.get("camera_ready_analyzer")
    consistency = summary.get("collision_event_metric_consistency")
    return {
        "accounting": {
            key: accounting.get(key)
            for key in (
                "denominator_status",
                "expected_identity_count",
                "present_episode_rows",
                "present_unique_identities",
                "missing_identity_count",
                "duplicate_identity_count",
                "malformed_line_count",
                "unexpected_identity_count",
                "expected_run_count",
            )
        },
        "run_statuses": [
            {
                key: row.get(key)
                for key in (
                    "planner_key",
                    "kinematics",
                    "expected_episodes",
                    "present_episode_rows",
                    "benchmark_eligible",
                    "benchmark_eligible_episode_rows",
                    "excluded_episode_rows",
                    "run_status",
                    "availability_status",
                    "fallback_or_degraded",
                    "unavailable",
                    "failed",
                    "episode_outcome_status_counts",
                )
            }
            for row in run_statuses
            if isinstance(row, dict)
        ]
        if isinstance(run_statuses, list)
        else [],
        "camera_ready_analyzer": {
            key: analyzer.get(key)
            for key in ("status", "credibility_status", "finding_count", "findings")
        }
        if isinstance(analyzer, dict)
        else None,
        "collision_event_metric_consistency": consistency
        if isinstance(consistency, dict)
        else None,
    }


def _planner_entry(campaign: dict[str, Any], planner_key: str) -> dict[str, Any]:
    planners = campaign.get("planners")
    if not isinstance(planners, list):
        return {}
    return next(
        (item for item in planners if isinstance(item, dict) and item.get("key") == planner_key), {}
    )


def _case_anomalies(case: dict[str, Any], row: dict[str, Any]) -> list[str]:
    anomalies: list[str] = []
    outcome = row.get("outcome") if isinstance(row.get("outcome"), dict) else {}
    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
    if outcome.get("collision_event") is True:
        counts = [metrics.get("collisions"), metrics.get("total_collision_count")]
        finite = [_finite_number(value) for value in counts]
        if any(value is not None for value in finite) and all(
            value is None or value <= 0 for value in finite
        ):
            anomalies.append("collision_event_without_positive_collision_metric")
    if case.get("benchmark_eligible") is not True:
        anomalies.append("source_run_not_benchmark_eligible")
    return anomalies


def _config_snapshot(row: dict[str, Any]) -> dict[str, Any] | None:
    metadata = row.get("algorithm_metadata")
    if not isinstance(metadata, dict):
        return None
    config = metadata.get("config")
    return config if isinstance(config, dict) else None


def _model_artifact_missing(config: dict[str, Any]) -> bool:
    for key in ("model_path", "checkpoint_path", "predictive_foresight_checkpoint_path"):
        value = config.get(key)
        if isinstance(value, str) and value:
            model_path = Path(value)
            if not model_path.is_absolute():
                model_path = REPO_ROOT / model_path
            if not model_path.is_file():
                return True
    return False


def _replay_ineligibility(row: dict[str, Any], matrix: Path, matrix_match: bool) -> str | None:
    if not matrix_match:
        return "unavailable_matrix_mismatch"
    if not isinstance(row.get("scenario_params"), dict):
        return "unavailable_scenario_parameters"
    params = row["scenario_params"]
    horizon = row.get("horizon") or params.get("run_horizon")
    if not isinstance(horizon, int | float) or isinstance(horizon, bool) or horizon <= 0:
        return "unavailable_replay_horizon"
    if _finite_number(params.get("run_dt")) is None:
        return "unavailable_replay_timestep"
    if _scenario_map_path(params, matrix) is None:
        return "unavailable_scenario_map"
    if row.get("seed") is None or not row.get("scenario_id") or not row.get("algo"):
        return "unavailable_case_identity"
    config = _config_snapshot(row)
    if config is None:
        return "unavailable_planner_configuration"
    if _model_artifact_missing(config):
        return "unavailable_model_artifact"
    return None


def _scenario_map_path(params: dict[str, Any], matrix: Path) -> Path | None:
    value = params.get("map_file")
    if not isinstance(value, str) or not value:
        return None
    source_path = Path(value)
    candidates = [source_path] if source_path.is_absolute() else [REPO_ROOT / source_path]
    if not source_path.is_absolute():
        candidates.append(matrix.parent / source_path)
    return next((path.resolve() for path in candidates if path.is_file()), None)


def _materialize_replay_matrix(row: dict[str, Any], matrix: Path, input_dir: Path) -> Path:
    params = row.get("scenario_params")
    if not isinstance(params, dict):
        raise MaterializationError("source row has no scenario parameter mapping")
    map_path = _scenario_map_path(params, matrix)
    if map_path is None:
        raise MaterializationError("source scenario map is unavailable in this checkout")
    scenario = dict(params)
    scenario["map_file"] = os.path.relpath(map_path, input_dir)
    scenario["seeds"] = [int(row["seed"])]
    search_paths = [os.path.relpath(REPO_ROOT / "maps/svg_maps", input_dir)]
    replay_matrix = input_dir / "replay_matrix.yaml"
    payload = {"map_search_paths": search_paths, "scenarios": [scenario]}
    replay_matrix.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
    return replay_matrix


def _materialize_case_replay_inputs(
    row: dict[str, Any],
    matrix: Path,
    case_dir: Path,
    *,
    replay_ineligibility: str | None,
) -> dict[str, Any]:
    """Write reusable single-seed scenario/config inputs without starting a simulation."""
    input_dir = case_dir / "replay_input"
    input_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "status": "unavailable",
        "replay_eligible": replay_ineligibility is None,
        "replay_ineligibility": replay_ineligibility,
    }
    params = row.get("scenario_params")
    if isinstance(params, dict):
        try:
            replay_matrix = _materialize_replay_matrix(row, matrix, input_dir)
        except (MaterializationError, TypeError, ValueError) as exc:
            result["scenario_matrix_error"] = str(exc)
        else:
            result["scenario_matrix_path"] = "replay_input/replay_matrix.yaml"
            result["scenario_matrix_sha256"] = _sha256(replay_matrix)
    else:
        result["scenario_matrix_error"] = "source row has no scenario parameter mapping"
    config = _config_snapshot(row)
    if config is not None:
        config_path = input_dir / "planner_config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
        result["planner_config_path"] = "replay_input/planner_config.yaml"
        result["planner_config_sha256"] = _sha256(config_path)
    else:
        result["planner_config_error"] = "source row has no planner configuration snapshot"
    if result.get("scenario_matrix_path") and result.get("planner_config_path"):
        result["status"] = "materialized"
    return result


def _replay_command(
    case: dict[str, Any],
    row: dict[str, Any],
    matrix: Path,
    out_path: Path,
    config_path: Path,
    profile: str,
) -> list[str]:
    params = row.get("scenario_params") if isinstance(row.get("scenario_params"), dict) else {}
    command = [
        "uv",
        "run",
        "robot_sf_bench",
        "run",
        "--matrix",
        os.path.relpath(matrix, REPO_ROOT),
        "--out",
        os.path.relpath(out_path, REPO_ROOT),
        "--base-seed",
        str(case["seed"]),
        "--repeats",
        "1",
        "--horizon",
        str(row.get("horizon") or params.get("run_horizon") or 0),
        "--dt",
        str(params.get("run_dt") or 0.1),
        "--algo",
        str(case["planner_key"]),
        "--algo-config",
        os.path.relpath(config_path, REPO_ROOT),
        "--benchmark-profile",
        profile,
        "--workers",
        "1",
        "--no-resume",
        "--no-video",
        "--video-renderer",
        "none",
        "--fail-fast",
    ]
    if params.get("record_forces") is True:
        command.append("--record-forces")
    return command


def _compare(expected: dict[str, Any], observed: dict[str, Any]) -> dict[str, Any]:
    expected_outcome = expected.get("outcome") if isinstance(expected.get("outcome"), dict) else {}
    observed_outcome = observed.get("outcome") if isinstance(observed.get("outcome"), dict) else {}
    event_checks = {
        field: {
            "expected": expected_outcome.get(field),
            "observed": observed_outcome.get(field),
            "status": (
                "match"
                if isinstance(expected_outcome.get(field), bool)
                and expected_outcome.get(field) == observed_outcome.get(field)
                else "mismatch"
                if isinstance(expected_outcome.get(field), bool)
                and isinstance(observed_outcome.get(field), bool)
                else "unavailable"
            ),
        }
        for field in EVENT_FIELDS
    }
    expected_metrics = expected.get("metrics") if isinstance(expected.get("metrics"), dict) else {}
    observed_metrics = observed.get("metrics") if isinstance(observed.get("metrics"), dict) else {}
    metric_checks: dict[str, dict[str, Any]] = {}
    for field in METRIC_FIELDS:
        left = _finite_number(expected_metrics.get(field))
        right = _finite_number(observed_metrics.get(field))
        if left is None or right is None:
            status = "unavailable"
        else:
            status = (
                "match" if math.isclose(left, right, rel_tol=1e-6, abs_tol=1e-6) else "mismatch"
            )
        metric_checks[field] = {"expected": left, "observed": right, "status": status}
    statuses = [item["status"] for item in [*event_checks.values(), *metric_checks.values()]]
    if "mismatch" in statuses:
        overall = "mismatch"
    elif all(status == "match" for status in statuses):
        overall = "match"
    else:
        overall = "incomplete"
    return {"overall": overall, "outcomes": event_checks, "metrics": metric_checks}


def _run_replay(
    case: dict[str, Any],
    row: dict[str, Any],
    matrix: Path,
    case_dir: Path,
    campaign: dict[str, Any],
    replay_revision: str,
) -> dict[str, Any]:
    planner = _planner_entry(campaign, str(case["planner_key"]))
    profile = planner.get("benchmark_profile") or "baseline-safe"
    config = _config_snapshot(row)
    if config is None:
        return {"status": "unavailable_planner_configuration", "attempted": False}
    replay_dir = case_dir / "replay"
    replay_dir.mkdir(parents=True, exist_ok=True)
    input_dir = case_dir / "replay_input"
    input_dir.mkdir(parents=True, exist_ok=True)
    config_path = input_dir / "planner_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    replay_matrix = _materialize_replay_matrix(row, matrix, input_dir)
    episode_path = replay_dir / "episodes.jsonl"
    command = _replay_command(case, row, replay_matrix, episode_path, config_path, str(profile))
    stdout_path = replay_dir / "stdout.txt"
    stderr_path = replay_dir / "stderr.txt"
    try:
        with (
            stdout_path.open("w", encoding="utf-8") as stdout,
            stderr_path.open("w", encoding="utf-8") as stderr,
        ):
            completed = subprocess.run(
                command, cwd=REPO_ROOT, stdout=stdout, stderr=stderr, check=False
            )
    except OSError as exc:
        return {
            "attempted": True,
            "status": "runner_unavailable",
            "command": command,
            "command_shell": shlex.join(command),
            "error": str(exc),
            "source_revision": row.get("git_hash"),
            "replay_revision": replay_revision,
        }
    result: dict[str, Any] = {
        "attempted": True,
        "command": command,
        "command_shell": shlex.join(command),
        "returncode": completed.returncode,
        "replay_revision": replay_revision,
        "source_revision": row.get("git_hash"),
        "same_repository_revision": row.get("git_hash") == replay_revision,
        "source_matrix_sha256": _sha256(matrix),
        "replay_matrix_sha256": _sha256(replay_matrix),
        "replay_matrix_path": os.path.relpath(replay_matrix, REPO_ROOT),
        "episode_output": str(episode_path.relative_to(case_dir)),
        "stdout_path": str(stdout_path.relative_to(case_dir)),
        "stderr_path": str(stderr_path.relative_to(case_dir)),
    }
    replay_rows, malformed_rows = _read_replay_rows(episode_path)
    if episode_path.is_file():
        result["episode_output_sha256"] = _sha256(episode_path)
        result["episode_output_checksum_status"] = "captured_at_run"
        result["episode_output_checksum_origin"] = "captured_at_run"
    result["replay_row_count"] = len(replay_rows)
    result["replay_malformed_line_count"] = malformed_rows
    if completed.returncode != 0 or len(replay_rows) != 1 or malformed_rows:
        result["status"] = (
            "runner_failed"
            if completed.returncode != 0
            else "invalid_output"
            if malformed_rows
            else "unavailable_output"
        )
        return result
    actual = replay_rows[0]
    identity = _replay_identity(case, actual)
    result["identity"] = identity
    if identity["status"] != "match":
        result["status"] = "replay_identity_mismatch"
        return result
    metadata = (
        actual.get("algorithm_metadata")
        if isinstance(actual.get("algorithm_metadata"), dict)
        else {}
    )
    execution_mode = _execution_mode_identity(row, actual)
    result["execution_mode"] = execution_mode["replay"]
    result["execution_mode_source"] = execution_mode["source"]
    result["episode_status"] = actual.get("status")
    result["config_hash_source"] = (
        (row.get("algorithm_metadata") or {}).get("config_hash")
        if isinstance(row.get("algorithm_metadata"), dict)
        else None
    )
    result["config_hash_replay"] = metadata.get("config_hash")
    if execution_mode["status"] != "match":
        result["status"] = execution_mode["status"]
        return result
    if result["config_hash_source"] != result["config_hash_replay"]:
        result["status"] = "planner_config_identity_mismatch"
        return result
    comparison = _compare(row, actual)
    result["comparison"] = comparison
    if comparison["overall"] == "mismatch":
        result["status"] = (
            "mismatch_same_revision"
            if result["same_repository_revision"]
            else "mismatch_different_revision"
        )
    elif not result["same_repository_revision"]:
        result["status"] = (
            "matched_different_revision"
            if comparison["overall"] == "match"
            else "incomplete_different_revision"
        )
    else:
        result["status"] = (
            "exact_match" if comparison["overall"] == "match" else "incomplete_same_revision"
        )
    return result


def _read_replay_rows(episode_path: Path) -> tuple[list[dict[str, Any]], int]:
    if not episode_path.is_file():
        return [], 0
    rows = []
    malformed = 0
    for raw_line in episode_path.read_bytes().splitlines():
        if not raw_line.strip():
            continue
        try:
            parsed = json.loads(raw_line)
        except (UnicodeError, json.JSONDecodeError):
            malformed += 1
            continue
        if isinstance(parsed, dict):
            rows.append(parsed)
        else:
            malformed += 1
    return rows, malformed


def _replay_identity(case: dict[str, Any], observed: dict[str, Any]) -> dict[str, Any]:
    expected = {
        "scenario_id": case.get("scenario_id"),
        "seed": case.get("seed"),
        "algo": case.get("planner_key"),
    }
    actual = {key: observed.get(key) for key in expected}
    return {
        "status": "match" if actual == expected else "mismatch",
        "expected": expected,
        "observed": actual,
    }


def _execution_mode_identity(source: dict[str, Any], replay: dict[str, Any]) -> dict[str, Any]:
    """Require exact source/replay execution-mode evidence before metric comparison."""
    source_metadata = (
        source.get("algorithm_metadata")
        if isinstance(source.get("algorithm_metadata"), dict)
        else {}
    )
    replay_metadata = (
        replay.get("algorithm_metadata")
        if isinstance(replay.get("algorithm_metadata"), dict)
        else {}
    )
    source_kinematics = source_metadata.get("planner_kinematics")
    replay_kinematics = replay_metadata.get("planner_kinematics")
    source_kinematics = source_kinematics if isinstance(source_kinematics, dict) else {}
    replay_kinematics = replay_kinematics if isinstance(replay_kinematics, dict) else {}
    source_mode = source_kinematics.get("execution_mode")
    replay_mode = replay_kinematics.get("execution_mode")
    if replay_mode in {"fallback", "degraded"} or replay_metadata.get("status") in {
        "fallback",
        "degraded",
    }:
        status = "fallback_or_degraded_not_evidence"
    elif not isinstance(source_mode, str) or not isinstance(replay_mode, str):
        status = "unavailable_execution_mode_identity"
    else:
        status = "match" if source_mode == replay_mode else "execution_mode_identity_mismatch"
    return {"source": source_mode, "replay": replay_mode, "status": status}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def _selected_cases(summary: dict[str, Any]) -> list[dict[str, Any]]:
    cases = summary.get("cases")
    if not isinstance(cases, list) or not cases:
        raise MaterializationError("showcase summary has no selected cases")
    if len(cases) > MAX_CASES:
        raise MaterializationError(f"showcase selected {len(cases)} cases; maximum is {MAX_CASES}")
    case_ids = [case.get("case_id") for case in cases if isinstance(case, dict)]
    if len(case_ids) != len(cases) or any(
        not isinstance(case_id, str) or not CASE_ID_RE.fullmatch(case_id) for case_id in case_ids
    ):
        raise MaterializationError(
            "all selected cases must have stable case-<16 lowercase hex> IDs"
        )
    if len(set(case_ids)) != len(case_ids):
        raise MaterializationError("showcase summary has duplicate selected case IDs")
    return cases


def _load_resume_records(
    resume_from: Path | None, *, source: dict[str, Any]
) -> tuple[Path | None, dict[str, dict[str, Any]]]:
    if resume_from is None:
        return None, {}
    manifest_path = resume_from / "manifest.json" if resume_from.is_dir() else resume_from
    manifest = _read_object(manifest_path)
    previous_source = manifest.get("source") if isinstance(manifest.get("source"), dict) else {}
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or previous_source.get("campaign_id") != source.get("campaign_id")
        or previous_source.get("bundle_sha256") != source.get("bundle_sha256")
        or previous_source.get("matrix_sha256") != source.get("matrix_sha256")
        or previous_source.get("source_revision") != source.get("source_revision")
    ):
        raise MaterializationError(
            "resume slice does not match the exact summary and source identity"
        )
    rows = manifest.get("cases")
    if not isinstance(rows, list):
        raise MaterializationError("resume manifest has no case inventory")
    records = {
        row["case_id"]: row
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("case_id"), str)
    }
    return manifest_path.parent.resolve(), records


def _annotate_reused_episode_checksum(
    replay: dict[str, Any], previous_case_dir: Path, current_case_dir: Path
) -> None:
    """Hash a resumed episode artifact and compare any hash in its prior receipt."""
    episode_output = replay.get("episode_output")
    if not isinstance(episode_output, str):
        return
    previous_episode = _safe_relative(
        previous_case_dir, episode_output, label="resumed replay episode output"
    )
    current_episode = _safe_relative(
        current_case_dir, episode_output, label="copied replay episode output"
    )
    if not previous_episode.is_file() or not current_episode.is_file():
        replay["episode_output_checksum_status"] = "missing"
        return
    copied_hash = _sha256(current_episode)
    previous_hash = replay.get("episode_output_sha256")
    checksum_origin = replay.get("episode_output_checksum_origin")
    if not isinstance(checksum_origin, str):
        checksum_origin = replay.get("episode_output_checksum_status")
    if not isinstance(checksum_origin, str):
        checksum_origin = "calculated_on_resume_prior_receipt_unavailable"
    replay["episode_output_checksum_origin"] = checksum_origin
    if isinstance(previous_hash, str) and previous_hash != copied_hash:
        replay["status"] = "replay_artifact_checksum_mismatch"
        replay["episode_output_checksum_status"] = "mismatch"
    elif isinstance(previous_hash, str):
        replay["episode_output_checksum_status"] = "verified"
    else:
        replay["episode_output_checksum_status"] = "calculated_on_resume_prior_receipt_unavailable"
    replay["episode_output_sha256"] = copied_hash


def _annotate_reused_execution_mode(
    replay: dict[str, Any], row: dict[str, Any], case_dir: Path
) -> None:
    """Recheck source/replay execution mode from the copied replay episode row."""
    episode_output = replay.get("episode_output")
    if not isinstance(episode_output, str):
        return
    episode_path = _safe_relative(case_dir, episode_output, label="reused replay episode output")
    replay_rows, malformed_rows = _read_replay_rows(episode_path)
    replay["replay_row_count"] = len(replay_rows)
    replay["replay_malformed_line_count"] = malformed_rows
    if len(replay_rows) != 1 or malformed_rows:
        if replay.get("status") != "replay_artifact_checksum_mismatch":
            replay["status"] = "invalid_output" if malformed_rows else "unavailable_output"
        return
    identity = _execution_mode_identity(row, replay_rows[0])
    replay["execution_mode_source"] = identity["source"]
    replay["execution_mode"] = identity["replay"]
    if (
        identity["status"] != "match"
        and replay.get("status") != "replay_artifact_checksum_mismatch"
    ):
        replay["status"] = identity["status"]


def materialize(  # noqa: C901, PLR0915 - provenance, reuse, and budget gates share one output transaction
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Verify selected source rows, materialize case records, and run bounded replays."""
    summary_path = args.summary.resolve()
    campaign_root = args.campaign_root.resolve()
    matrix = args.matrix.resolve()
    out_dir = args.out_dir.resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise MaterializationError(f"refusing to overwrite a non-empty output directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = _read_object(summary_path)
    source_provenance = _verify_inputs(summary, campaign_root, args.bundle, matrix)
    summary_sha256 = _sha256(summary_path)
    resume_root, resume_records = _load_resume_records(args.resume_from, source=source_provenance)
    resume_manifest_sha256 = None
    resume_summary_sha256 = None
    if resume_root is not None:
        resume_manifest_path = (
            resume_root / "manifest.json"
            if (resume_root / "manifest.json").is_file()
            else args.resume_from
        )
        resume_manifest = _read_object(resume_manifest_path)
        resume_manifest_sha256 = _sha256(resume_manifest_path)
        resume_manifest_source = resume_manifest.get("source")
        if isinstance(resume_manifest_source, dict):
            resume_summary_sha256 = resume_manifest_source.get("summary_sha256")
    campaign = _read_object(campaign_root / "campaign_manifest.json")
    cases = _selected_cases(summary)
    try:
        replay_revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise MaterializationError(f"could not identify replay source revision: {exc}") from exc
    case_records = []
    replay_candidates = []
    reused_count = 0
    for case in cases:
        row, source_ref = _load_source_row(campaign_root, case)
        if case.get("benchmark_eligible") is True:
            ineligible = _replay_ineligibility(row, matrix, True)
        else:
            ineligible = "unavailable_source_row_not_benchmark_eligible"
        cleaned_row, nonfinite_fields = _sanitize(row)
        replay_status = {"status": ineligible or "not_attempted", "attempted": False}
        record = {
            "schema_version": "benchmark-hard-case.v1",
            "case_id": case["case_id"],
            "selection": {
                "selected_groups": case.get("selected_groups", []),
                "selector_schema": summary.get("schema_version"),
                "selection_contract": summary.get("selection", {}),
            },
            "criticality": {
                "outcome": case.get("outcome"),
                "metrics": case.get("metrics"),
                "anomalies": _case_anomalies(case, row),
                "evidence_tier": "diagnostic_only",
            },
            "source": {
                **source_ref,
                "campaign_id": source_provenance["campaign_id"],
                "campaign_source_revision": source_provenance["source_revision"],
                "scenario_matrix": source_provenance["matrix_path"],
                "scenario_matrix_sha256": source_provenance["matrix_sha256"],
                "bundle_sha256": source_provenance["bundle_sha256"],
                "episode_id": row.get("episode_id"),
                "planner_key": row.get("algo"),
                "planner_config_hash": (row.get("algorithm_metadata") or {}).get("config_hash")
                if isinstance(row.get("algorithm_metadata"), dict)
                else None,
                "row_git_hash": row.get("git_hash"),
            },
            "scenario": {
                "scenario_id": row.get("scenario_id"),
                "scenario_family": case.get("scenario_family"),
                "seed": row.get("seed"),
                "kinematics": (row.get("scenario_params") or {}).get("robot_config")
                if isinstance(row.get("scenario_params"), dict)
                else None,
            },
            "planner": {
                "key": row.get("algo"),
                "algorithm_metadata": row.get("algorithm_metadata"),
                "configuration_snapshot": _config_snapshot(row),
            },
            "source_showcase_renderer": case.get("replay"),
            "source_record": cleaned_row,
            "source_row_benchmark_eligible": case.get("benchmark_eligible"),
            "source_nonfinite_fields": nonfinite_fields,
            "replay": replay_status,
        }
        case_dir = out_dir / "cases" / case["case_id"]
        case_dir.mkdir(parents=True, exist_ok=False)
        record["replay_input"] = _materialize_case_replay_inputs(
            row, matrix, case_dir, replay_ineligibility=ineligible
        )
        record["case_file"] = f"cases/{case['case_id']}/case.json"
        previous = resume_records.get(case["case_id"])
        previous_file = None
        if previous and resume_root:
            previous_file = _safe_relative(
                resume_root, previous.get("case_file"), label="resume case"
            )
        if previous and previous_file and previous_file.is_file():
            previous_case = _read_object(previous_file)
            previous_replay = previous_case.get("replay")
            same_row = (
                previous_case.get("source", {}).get("record_sha256") == source_ref["record_sha256"]
            )
            if (
                same_row
                and isinstance(previous_replay, dict)
                and previous_replay.get("attempted") is True
            ):
                previous_replay_dir = previous_file.parent / "replay"
                if previous_replay_dir.is_dir():
                    shutil.copytree(previous_replay_dir, case_dir / "replay")
                    previous_comparison = previous_replay.get("comparison")
                    if (
                        isinstance(previous_comparison, dict)
                        and previous_comparison.get("overall") == "mismatch"
                    ):
                        previous_replay["status"] = (
                            "mismatch_same_revision"
                            if previous_replay.get("same_repository_revision") is True
                            else "mismatch_different_revision"
                        )
                    _annotate_reused_episode_checksum(
                        previous_replay, previous_file.parent, case_dir
                    )
                    _annotate_reused_execution_mode(previous_replay, row, case_dir)
                    record["replay"] = {
                        **previous_replay,
                        "reused": True,
                        "reused_from_summary_sha256": resume_summary_sha256,
                        "reused_from_manifest_sha256": resume_manifest_sha256,
                    }
                    reused_count += 1
        _write_json(case_dir / "case.json", record)
        case_records.append(record)
        if not ineligible and not record["replay"].get("attempted"):
            replay_candidates.append((case, row, record, case_dir))
    attempted_cases = replay_candidates[: args.replay_limit]
    for case, row, record, case_dir in attempted_cases:
        record["replay"] = _run_replay(case, row, matrix, case_dir, campaign, replay_revision)
        _write_json(case_dir / "case.json", record)
    for case_record in case_records:
        on_disk = _read_object(out_dir / case_record["case_file"])
        case_record["replay"] = on_disk.get("replay", case_record["replay"])
    replay_counts = Counter(record["replay"].get("status", "unknown") for record in case_records)
    renderer_counts = Counter(
        (record.get("source_showcase_renderer") or {}).get("status", "unknown")
        for record in case_records
        if isinstance(record.get("source_showcase_renderer"), dict)
    )
    selected_groups = Counter(
        group
        for case in cases
        for group in case.get("selected_groups", [])
        if isinstance(group, str)
    )
    planner_counts = Counter(record["planner"]["key"] for record in case_records)
    scenario_family_counts = Counter(
        record["scenario"]["scenario_family"] for record in case_records
    )
    anomaly_counts = Counter(
        anomaly for record in case_records for anomaly in record["criticality"]["anomalies"]
    )
    replay_revisions = sorted(
        {
            replay["replay_revision"]
            for record in case_records
            if (replay := record.get("replay", {})).get("attempted") is True
            and isinstance(replay.get("replay_revision"), str)
        }
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "materialized",
        "claim_boundary": "Historical simulator records and bounded same-scenario reruns only; no real-world safety claim.",
        "evidence_tier": "diagnostic_only",
        "source": {
            **source_provenance,
            "summary_sha256": summary_sha256,
            "showcase_tool_revision": (summary.get("tool_provenance") or {}).get("git_revision")
            if isinstance(summary.get("tool_provenance"), dict)
            else None,
        },
        "source_showcase_diagnostics": _compact_source_showcase_diagnostics(summary),
        "execution_environment": {
            "source_environment": "not_recorded_in_release_bundle",
            "replay_environment": {
                "python_version": sys.version.split()[0],
                "python_implementation": platform.python_implementation(),
                "platform_system": platform.system(),
                "platform_release": platform.release(),
                "machine": platform.machine(),
                "uv_lock_sha256": _sha256(REPO_ROOT / "uv.lock")
                if (REPO_ROOT / "uv.lock").is_file()
                else None,
            },
        },
        "selection": {
            "selector_schema": summary.get("schema_version"),
            "selector_contract": summary.get("selection", {}),
            "case_count": len(case_records),
            "selected_group_counts": dict(sorted(selected_groups.items())),
            "planner_counts": dict(sorted(planner_counts.items())),
            "scenario_family_counts": dict(sorted(scenario_family_counts.items())),
            "scenario_id_count": len(
                {record["scenario"]["scenario_id"] for record in case_records}
            ),
            "case_ids": [record["case_id"] for record in case_records],
        },
        "criticality_anomaly_counts": dict(sorted(anomaly_counts.items())),
        "source_showcase_renderer_status_counts": dict(sorted(renderer_counts.items())),
        "replay": {
            "requested_limit": args.replay_limit,
            "maximum_allowed": MAX_REPLAYS,
            "attempted": sum(record["replay"].get("attempted") is True for record in case_records),
            "new_attempted": len(attempted_cases),
            "reused_attempts": reused_count,
            "status_counts": dict(sorted(replay_counts.items())),
            "replay_revision": replay_revisions[0] if len(replay_revisions) == 1 else None,
            "replay_revisions": replay_revisions,
            "materializer_revision": replay_revision,
            "source_revision": source_provenance["source_revision"],
        },
        "cases": [
            {
                "case_id": record["case_id"],
                "case_file": record["case_file"],
                "source_record": {
                    "episode_file": record["source"]["episode_file"],
                    "episode_file_sha256": record["source"]["episode_file_sha256"],
                    "line_number": record["source"]["line_number"],
                    "record_sha256": record["source"]["record_sha256"],
                },
                "scenario_id": record["scenario"]["scenario_id"],
                "scenario_family": record["scenario"]["scenario_family"],
                "seed": record["scenario"]["seed"],
                "planner_key": record["planner"]["key"],
                "selected_groups": record["selection"]["selected_groups"],
                "criticality": record["criticality"],
                "benchmark_eligible": record["source_row_benchmark_eligible"],
                "source_showcase_renderer": record["source_showcase_renderer"],
                "replay_input": record["replay_input"],
                "replay": record["replay"],
            }
            for record in case_records
        ],
    }
    _write_json(out_dir / "manifest.json", manifest)
    _write_report(out_dir / "report.md", manifest)
    return manifest


def _write_report(path: Path, manifest: dict[str, Any]) -> None:
    counts = manifest["replay"]["status_counts"]
    diagnostics = manifest.get("source_showcase_diagnostics", {})
    accounting = diagnostics.get("accounting", {}) if isinstance(diagnostics, dict) else {}
    consistency = (
        diagnostics.get("collision_event_metric_consistency", {})
        if isinstance(diagnostics, dict)
        else {}
    )
    analyzer = diagnostics.get("camera_ready_analyzer", {}) if isinstance(diagnostics, dict) else {}
    environment = manifest.get("execution_environment", {})
    replay_environment = environment.get("replay_environment", {})
    diagnostic_lines = []
    if accounting:
        diagnostic_lines.append(
            f"- Source episode accounting: {accounting.get('present_episode_rows')}/"
            f"{accounting.get('expected_identity_count')} present; "
            f"missing={accounting.get('missing_identity_count')}, "
            f"duplicates={accounting.get('duplicate_identity_count')}, "
            f"malformed={accounting.get('malformed_line_count')}"
        )
    if consistency:
        diagnostic_lines.append(
            f"- Full-source collision events: {consistency.get('canonical_collision_event_count')}; "
            f"both count metrics non-positive: {consistency.get('both_count_metrics_nonpositive_count')}"
        )
    if analyzer:
        diagnostic_lines.append(
            f"- Camera-ready analyzer: `{analyzer.get('status')}` with "
            f"{analyzer.get('finding_count')} retained finding(s)"
        )
    lines = [
        "# Historical benchmark hard-case slice",
        "",
        f"- Evidence tier: `{manifest['evidence_tier']}`",
        f"- Campaign: `{manifest['source']['campaign_id']}`",
        f"- Source revision: `{manifest['source']['source_revision']}`",
        f"- Source bundle SHA-256: `{manifest['source']['bundle_sha256']}`",
        f"- Scenario matrix SHA-256: `{manifest['source']['matrix_sha256']}`",
        f"- Source execution environment: `{environment.get('source_environment', 'not_recorded')}`",
        f"- Replay execution environment: Python `{replay_environment.get('python_version')}`, "
        f"{replay_environment.get('platform_system')} `{replay_environment.get('platform_release')}`; "
        f"lock SHA-256 `{replay_environment.get('uv_lock_sha256')}`",
        f"- Selected cases materialized: {manifest['selection']['case_count']}",
        f"- Planner counts: `{manifest['selection']['planner_counts']}`",
        f"- Scenario-family counts: `{manifest['selection']['scenario_family_counts']}`",
        f"- Distinct scenario IDs: {manifest['selection']['scenario_id_count']}",
        f"- Source anomaly counts: `{manifest['criticality_anomaly_counts']}`",
        f"- Scenario/config snapshots: `{dict(sorted(Counter(case['replay_input']['status'] for case in manifest['cases']).items()))}`",
        f"- Bounded single-scenario replays attempted: {manifest['replay']['attempted']}/{manifest['replay']['maximum_allowed']}",
        f"- Source showcase renderer statuses: `{manifest['source_showcase_renderer_status_counts']}`",
        *diagnostic_lines,
        "",
        "Selection uses the named event/metric groups and deterministic diversity order emitted by `benchmark-showcase.v1`; this tool does not rank cases again.",
        "",
        "Replay statuses:",
        "",
    ]
    lines.extend(f"- `{status}`: {count}" for status, count in sorted(counts.items()))
    lines.extend(
        [
            "",
            "## Cases",
            "",
            "| Case | Planner | Family / scenario / seed | Selected groups | Input | Source warnings | Replay |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for case in manifest["cases"]:
        lines.append(
            f"| `{case['case_id']}` | {case['planner_key']} | {case['scenario_family']} / {case['scenario_id']} / {case['seed']} | {', '.join(case['selected_groups'])} | {case['replay_input'].get('status', 'unknown')} | {', '.join(case['criticality']['anomalies']) or 'none'} | {case['replay'].get('status', 'unknown')} |"
        )
    lines.extend(
        [
            "",
            "## Limits",
            "",
            "A source collision event with a non-positive collision-count metric is retained and flagged; neither field is rewritten. Replays on a different repository revision are labeled revision-divergent and do not establish exact historical reproduction. Missing or unavailable metrics remain unavailable. This artifact mines historical simulator evidence and makes no real-world safety claim.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary", type=Path, required=True, help="benchmark-showcase.v1 JSON summary"
    )
    parser.add_argument(
        "--campaign-root", type=Path, required=True, help="verified bundle payload directory"
    )
    parser.add_argument(
        "--bundle", type=Path, help="source archive; required when summary declares a bundle SHA"
    )
    parser.add_argument(
        "--matrix", type=Path, required=True, help="canonical scenario matrix in this repository"
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="new or empty output directory")
    parser.add_argument(
        "--replay-limit",
        type=int,
        default=0,
        help="execute 0–5 single-scenario replays (default: 0)",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="reuse same-source case replay receipts from an earlier slice directory or manifest",
    )
    args = parser.parse_args(argv)
    if not 0 <= args.replay_limit <= MAX_REPLAYS:
        parser.error(f"--replay-limit must be between 0 and {MAX_REPLAYS}")
    return args


def main(argv: list[str] | None = None) -> int:
    """Run the materialization CLI and return its process exit code."""
    try:
        args = _parse_args(argv)
        manifest = materialize(args)
    except MaterializationError as exc:
        print(json.dumps({"schema_version": SCHEMA_VERSION, "status": "error", "error": str(exc)}))
        return 2
    print(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "status": manifest["status"],
                "case_count": manifest["selection"]["case_count"],
                "replay_attempted": manifest["replay"]["attempted"],
                "out_dir": str(args.out_dir.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
