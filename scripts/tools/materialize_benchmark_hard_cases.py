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

from robot_sf.benchmark.event_ledger import build_event_ledger

SCHEMA_VERSION = "benchmark-hard-case-slice.v1"
SHOWCASE_SCHEMA = "benchmark-showcase.v1"
REPLAY_CHECKOUT_SNAPSHOT_SCHEMA = "replay-checkout-snapshot.v1"
CASE_ID_RE = re.compile(r"case-[0-9a-f]{16}\Z")
MAX_CASES = 50
MAX_REPLAYS = 5
GIT_REVISION_RE = re.compile(r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})\Z")
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
# `benchmark-showcase.v1` stores selected measurements under these normalized
# output names. Keep this mapping aligned with `scripts/tools/benchmark_showcase.py`;
# materialization verifies those values against the checksum-pinned raw episode row.
SHOWCASE_METRIC_FIELDS = (
    ("clearing_distance_min", "minimum_clearance_m"),
    ("near_misses", "near_misses"),
    ("force_exceed_events", "force_exceed_events"),
    ("comfort_exposure", "comfort_exposure"),
    ("time_to_goal_norm", "time_to_goal_norm"),
    ("path_efficiency", "path_efficiency"),
    ("snqi", "snqi"),
    ("success", "success_metric"),
    ("collisions", "collisions_metric"),
    ("total_collision_count", "total_collision_count"),
)
SHOWCASE_EVENT_GROUPS = {
    "collision_event": ("collision_event", True),
    "timeout_event": ("timeout_event", True),
    "route_not_complete": ("route_complete", False),
}
SHOWCASE_METRIC_GROUPS = {
    "minimum_clearance": "minimum_clearance_m",
    "near_miss_extreme": "near_misses",
    "force_exceed_extreme": "force_exceed_events",
    "comfort_exposure_extreme": "comfort_exposure",
    "slow_normalized_time": "time_to_goal_norm",
    "low_path_efficiency": "path_efficiency",
}
SUPPORTED_EXECUTION_MODES = frozenset({"native", "adapter", "mixed"})
SUCCESSFUL_EXECUTION_STATUSES = frozenset({"ok", "success", "passed", "complete", "completed"})
AVAILABLE_EXECUTION_STATUSES = SUCCESSFUL_EXECUTION_STATUSES | {"available"}
SOURCE_RUNTIME_INPUT_PROVENANCE_SCHEMA = "benchmark-runtime-input-provenance.v1"
SOURCE_ENVIRONMENT_PROVENANCE_SCHEMA = "benchmark-execution-environment.v1"
RUNTIME_MODEL_PATH_KEYS = frozenset(
    {
        "model_path",
        "checkpoint_path",
        "predictive_foresight_checkpoint_path",
        "predictive_checkpoint_path",
        "sacadrl_checkpoint_path",
        "learned_gmm_checkpoint_path",
        "learned_policy_checkpoint",
    }
)
RUNTIME_MODEL_ID_KEYS = frozenset(
    {
        "model_id",
        "checkpoint_id",
        "sacadrl_model_id",
        "predictive_model_id",
        "predictive_foresight_model_id",
        "learned_gmm_model_id",
        "learned_policy_model_id",
    }
)
RUNTIME_MODEL_ID_PATH_KEYS = {
    "model_id": "model_path",
    "checkpoint_id": "checkpoint_path",
    "sacadrl_model_id": "sacadrl_checkpoint_path",
    "predictive_model_id": "predictive_checkpoint_path",
    "predictive_foresight_model_id": "predictive_foresight_checkpoint_path",
    "learned_gmm_model_id": "learned_gmm_checkpoint_path",
    "learned_policy_model_id": "learned_policy_checkpoint",
}
RUNTIME_MODEL_ID_ACTIVE_FLAGS = {"predictive_foresight_model_id": "predictive_foresight_enabled"}
RUNTIME_DEFAULT_MODEL_REFERENCES = {
    "ppo": frozenset({"model_id", "model_path"}),
    "prediction_planner": frozenset({"predictive_model_id", "predictive_checkpoint_path"}),
    "prediction": frozenset({"predictive_model_id", "predictive_checkpoint_path"}),
    "sacadrl": frozenset({"sacadrl_model_id", "sacadrl_checkpoint_path"}),
}
CROWDNAV_HEIGHT_DEFAULT_REPO_ROOT = "output/repos/CrowdNav_HEIGHT"
CROWDNAV_HEIGHT_DEFAULT_MODEL_DIR = (
    "output/external_checkpoints/crowdnav_height_extracted/HEIGHT/HEIGHT"
)
CROWDNAV_HEIGHT_DEFAULT_CHECKPOINT_NAME = "237800.pt"
CROWDNAV_HEIGHT_REPO_ASSET_KIND = "crowdnav_height_upstream_repo"
CROWDNAV_HEIGHT_CONFIG_ASSET_KIND = "crowdnav_height_config"
CROWDNAV_HEIGHT_CHECKPOINT_ASSET_KIND = "crowdnav_height_checkpoint"
CROWDNAV_HEIGHT_ASSET_KINDS = frozenset(
    {
        CROWDNAV_HEIGHT_REPO_ASSET_KIND,
        CROWDNAV_HEIGHT_CONFIG_ASSET_KIND,
        CROWDNAV_HEIGHT_CHECKPOINT_ASSET_KIND,
    }
)
FAILED_EXECUTION_STATUSES = frozenset(
    {
        "blocked",
        "error",
        "failed",
        "failure",
        "missing",
        "not_available",
        "not_run",
        "partial",
        "partial_failure",
        "placeholder",
        "skipped",
        "unknown",
        "unavailable",
    }
)
FALLBACK_MARKER_KEYS = frozenset(
    {
        "degraded",
        "fallback",
        "fallback_active",
        "fallback_or_degraded",
        "fallback_triggered",
        "fallback_used",
    }
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


def _showcase_bool(value: Any) -> bool | None:
    """Normalize source event values the same way as the showcase selector."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float) and not isinstance(value, bool) and value in (0, 1):
        return bool(value)
    return None


def _showcase_measurements(
    row: dict[str, Any],
) -> tuple[dict[str, bool | None], dict[str, float | None]]:
    """Derive the v1 selector's event and metric snapshot from one raw episode row."""
    outcome = row.get("outcome") if isinstance(row.get("outcome"), dict) else {}
    raw_metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
    events = {
        "route_complete": _showcase_bool(outcome.get("route_complete", outcome.get("success"))),
        "collision_event": _showcase_bool(outcome.get("collision_event", outcome.get("collision"))),
        "timeout_event": _showcase_bool(outcome.get("timeout_event", outcome.get("timeout"))),
    }
    metrics: dict[str, float | None] = {}
    for source_key, output_key in SHOWCASE_METRIC_FIELDS:
        value = raw_metrics.get(source_key)
        if source_key == "success" and isinstance(value, bool):
            metrics[output_key] = float(value)
        else:
            metrics[output_key] = _finite_number(value)
    return events, metrics


def _verify_selector_events(case: dict[str, Any], expected: dict[str, bool | None]) -> None:
    """Require the selector's normalized event snapshot to match the source row."""
    case_id = case.get("case_id", "unknown case")
    supplied = case.get("outcome")
    if not isinstance(supplied, dict) or set(supplied) != set(expected):
        raise MaterializationError(
            f"{case_id}: showcase outcome fields do not match the canonical source-row fields"
        )
    for field, expected_value in expected.items():
        observed = supplied.get(field)
        if (
            observed is not None and not isinstance(observed, bool)
        ) or observed is not expected_value:
            raise MaterializationError(
                f"{case_id}: showcase outcome {field} differs from the checksum-pinned source row"
            )


def _verify_selector_metrics(case: dict[str, Any], expected: dict[str, float | None]) -> None:
    """Require the selector's normalized metrics snapshot to match the source row."""
    case_id = case.get("case_id", "unknown case")
    supplied = case.get("metrics")
    if not isinstance(supplied, dict) or set(supplied) != set(expected):
        raise MaterializationError(
            f"{case_id}: showcase metric fields do not match the canonical source-row fields"
        )
    for field, expected_value in expected.items():
        observed = supplied.get(field)
        if expected_value is None:
            matches = observed is None
        else:
            matches = _finite_number(observed) == expected_value
        if not matches:
            raise MaterializationError(
                f"{case_id}: showcase metric {field} differs from the checksum-pinned source row"
            )


def _verify_selector_groups(
    case: dict[str, Any],
    events: dict[str, bool | None],
    metrics: dict[str, float | None],
) -> None:
    """Reject selected event/metric groups unsupported by the checksum-pinned row."""
    case_id = case.get("case_id", "unknown case")
    groups = case.get("selected_groups")
    if (
        not isinstance(groups, list)
        or not groups
        or any(not isinstance(group, str) or not group for group in groups)
        or len(set(groups)) != len(groups)
    ):
        raise MaterializationError(
            f"{case_id}: selected groups are missing, malformed, or repeated"
        )
    for group in groups:
        if group in SHOWCASE_EVENT_GROUPS:
            event, selected_value = SHOWCASE_EVENT_GROUPS[group]
            if events.get(event) is not selected_value:
                raise MaterializationError(
                    f"{case_id}: selected event group {group} is not supported by the source row"
                )
        elif group in SHOWCASE_METRIC_GROUPS:
            metric = SHOWCASE_METRIC_GROUPS[group]
            if metrics.get(metric) is None:
                raise MaterializationError(
                    f"{case_id}: selected metric group {group} has no source-row value"
                )
        else:
            raise MaterializationError(f"{case_id}: unsupported showcase selection group {group!r}")


def _selector_case_measurements(
    case: dict[str, Any], row: dict[str, Any]
) -> tuple[dict[str, bool | None], dict[str, float | None]]:
    """Bind selector-reported outcomes, metrics, and event groups to the pinned row."""
    events, metrics = _showcase_measurements(row)
    _verify_selector_events(case, events)
    _verify_selector_metrics(case, metrics)
    _verify_selector_groups(case, events, metrics)
    return events, metrics


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


def _showcase_tool_snapshot(
    summary: dict[str, Any],
    snapshot_revision: str,
    *,
    fallback_snapshot_revision: str | None = None,
) -> dict[str, Any]:
    """Preserve the executed showcase revision and verify an equivalent reachable snapshot.

    Showcase PR heads can become unreachable when a PR is squash-merged. The
    original revision remains the execution provenance; a separate snapshot
    revision is recorded only when every source-file checksum matches that Git
    tree. This avoids rewriting the historical execution identity.
    """
    provenance = summary.get("tool_provenance")
    if not isinstance(provenance, dict):
        return {
            "showcase_tool_revision": None,
            "showcase_tool_source_files_sha256": {},
            "showcase_tool_source_snapshot_revision": None,
            "showcase_tool_source_snapshot_status": "unavailable_tool_provenance",
        }

    raw_files = provenance.get("files")
    if not isinstance(raw_files, dict) or not raw_files:
        files: dict[str, str] = {}
        status = "unavailable_file_inventory"
        verified_revision = None
    else:
        files = {
            path: checksum
            for path, checksum in raw_files.items()
            if isinstance(path, str) and isinstance(checksum, str)
        }
        candidate_revisions = [snapshot_revision]
        if (
            isinstance(fallback_snapshot_revision, str)
            and GIT_REVISION_RE.fullmatch(fallback_snapshot_revision)
            and fallback_snapshot_revision not in candidate_revisions
        ):
            candidate_revisions.append(fallback_snapshot_revision)
        history_head = _git_head_revision()
        if history_head is not None:
            candidate_revisions.extend(
                revision
                for revision in _showcase_source_history_revisions(files, history_head)
                if revision not in candidate_revisions
            )
        verified_revision = next(
            (
                candidate_revision
                for candidate_revision in candidate_revisions
                if _showcase_source_files_match(files, raw_files, candidate_revision)
            ),
            None,
        )
        status = "verified_file_hash_match" if verified_revision else "source_files_not_matched"

    revision = provenance.get("git_revision")
    return {
        "showcase_tool_revision": revision if isinstance(revision, str) else None,
        "showcase_tool_source_files_sha256": dict(sorted(files.items())),
        "showcase_tool_source_snapshot_revision": verified_revision,
        "showcase_tool_source_snapshot_status": status,
    }


def _git_head_revision() -> str | None:
    """Return the current commit only when Git can resolve an exact HEAD."""
    result = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD^{commit}"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
        text=True,
    )
    revision = result.stdout.strip()
    return revision if result.returncode == 0 and GIT_REVISION_RE.fullmatch(revision) else None


def _showcase_source_history_revisions(files: dict[str, str], history_head: str) -> list[str]:
    """Return commits on current HEAD history that changed one of the selected source files."""
    if not files or not GIT_REVISION_RE.fullmatch(history_head):
        return []
    result = subprocess.run(
        ["git", "rev-list", history_head, "--", *sorted(files)],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode != 0:
        return []
    return [line for line in result.stdout.splitlines() if GIT_REVISION_RE.fullmatch(line)]


def _showcase_source_files_match(
    files: dict[str, str], raw_files: dict[str, Any], revision: str
) -> bool:
    """Verify every source hash against one reachable Git tree."""
    if not GIT_REVISION_RE.fullmatch(revision) or len(files) != len(raw_files):
        return False
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", revision, "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    if ancestry.returncode != 0:
        return False
    for relative, expected in sorted(files.items()):
        relative_path = PurePosixPath(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            return False
        result = subprocess.run(
            ["git", "show", f"{revision}:{relative_path.as_posix()}"],
            cwd=REPO_ROOT,
            capture_output=True,
            check=False,
        )
        if (
            result.returncode != 0
            or len(expected) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in expected)
            or hashlib.sha256(result.stdout).hexdigest() != expected.lower()
        ):
            return False
    return True


def _resume_showcase_snapshot_revision(resume_manifest: dict[str, Any]) -> str | None:
    """Return a prior verified snapshot revision, never a revision asserted by a case row."""
    source = resume_manifest.get("source")
    if not isinstance(source, dict):
        return None
    revision = source.get("showcase_tool_source_snapshot_revision")
    status = source.get("showcase_tool_source_snapshot_status")
    if status != "verified_file_hash_match" or not isinstance(revision, str):
        return None
    return revision if GIT_REVISION_RE.fullmatch(revision) else None


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


def _git_tree_file_identity(revision: str, path: Path, *, kind: str) -> dict[str, Any]:
    """Bind a runtime file to a regular file in a recorded Git tree."""
    try:
        relative_path = path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except (OSError, RuntimeError, ValueError):
        return {"kind": kind, "status": "unavailable", "reason": "outside_repository"}
    if not relative_path or relative_path == ".":
        return {"kind": kind, "status": "unavailable", "reason": "invalid_repository_path"}
    try:
        entry = subprocess.run(
            ["git", "ls-tree", "-z", revision, "--", relative_path],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        ).stdout
        records = [record for record in entry.split(b"\0") if record]
        if len(records) != 1:
            return {"kind": kind, "status": "unavailable", "reason": "not_in_source_tree"}
        metadata, recorded_path = records[0].split(b"\t", 1)
        mode, object_type, object_id = metadata.decode("ascii").split()
        if recorded_path.decode("utf-8", errors="strict") != relative_path:
            return {"kind": kind, "status": "unavailable", "reason": "tree_path_mismatch"}
        if mode not in {"100644", "100755"} or object_type != "blob":
            return {"kind": kind, "status": "unavailable", "reason": "not_regular_git_file"}
        blob = subprocess.run(
            ["git", "cat-file", "blob", object_id],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        ).stdout
        runtime_bytes = path.read_bytes()
    except (OSError, subprocess.CalledProcessError, UnicodeError, ValueError):
        return {"kind": kind, "status": "unavailable", "reason": "git_tree_lookup_failed"}
    if runtime_bytes != blob:
        return {
            "kind": kind,
            "status": "unavailable",
            "reason": "runtime_bytes_differ_from_source_tree",
            "path": relative_path,
        }
    return {
        "kind": kind,
        "status": "verified",
        "path": relative_path,
        "git_blob_oid": object_id,
        "sha256": hashlib.sha256(runtime_bytes).hexdigest(),
    }


def _runtime_model_paths(config: dict[str, Any]) -> list[tuple[str, str]]:
    """Collect recognized model/checkpoint paths wherever they occur in config."""
    found: list[tuple[str, str]] = []
    stack: list[Any] = [config]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            for key, value in current.items():
                if key in RUNTIME_MODEL_PATH_KEYS and isinstance(value, str) and value:
                    found.append((key, value))
                if isinstance(value, dict | list):
                    stack.append(value)
        elif isinstance(current, list):
            stack.extend(current)
    return sorted(set(found))


def _invalid_runtime_model_path_values(config: dict[str, Any]) -> list[dict[str, Any]]:
    invalid = []
    stack: list[Any] = [config]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            for key, value in current.items():
                if (
                    key in RUNTIME_MODEL_PATH_KEYS
                    and value is not None
                    and not isinstance(value, str)
                ):
                    invalid.append(
                        {
                            "kind": key,
                            "reason": "runtime_model_path_malformed",
                            "value_type": type(value).__name__,
                        }
                    )
                if isinstance(value, dict | list):
                    stack.append(value)
        elif isinstance(current, list):
            stack.extend(current)
    return invalid


def _runtime_model_path_components(key: str, value: str) -> list[Path] | None:
    """Resolve files consumed by a configured model path without guessing defaults."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    try:
        path = path.resolve()
        if key != "sacadrl_checkpoint_path":
            return [path] if path.is_file() else None
        prefix = path.with_suffix("") if path.suffix == ".meta" else path
        meta_path = prefix.with_name(f"{prefix.name}.meta")
        index_path = prefix.with_name(f"{prefix.name}.index")
        data_paths = sorted(prefix.parent.glob(f"{prefix.name}.data*"))
        components = [meta_path, index_path, *data_paths]
        if not meta_path.is_file() or not index_path.is_file() or not data_paths:
            return None
        return components
    except (OSError, RuntimeError):
        return None


def _runtime_input_descriptors(
    row: dict[str, Any], *, matrix: Path | None
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    """Describe effective map/model inputs, preserving unresolved model references."""
    params = row.get("scenario_params")
    config = _config_snapshot(row)
    if not isinstance(params, dict) or config is None:
        return [], [
            {"kind": "runtime_inputs", "reason": "scenario_or_planner_configuration_unavailable"}
        ]

    descriptors: list[dict[str, str]] = []
    unresolved: list[dict[str, Any]] = _invalid_runtime_model_path_values(config)
    map_value = params.get("map_file")
    map_path = _scenario_map_runtime_path(params, matrix)
    if not isinstance(map_value, str) or not map_value or map_path is None:
        unresolved.append({"kind": "scenario_map", "reason": "scenario_map_file_unavailable"})
    else:
        descriptors.append(
            {
                "kind": "scenario_map",
                "name": map_path.name,
                "reference": map_value,
            }
        )

    model_descriptors, model_unresolved = _runtime_model_input_descriptors(config, row)
    descriptors.extend(model_descriptors)
    unresolved.extend(model_unresolved)
    if row.get("algo") == "crowdnav_height":
        descriptors.extend(_crowdnav_height_input_specs(config))
    return sorted(
        descriptors, key=lambda item: (item["kind"], item["name"], item["reference"])
    ), unresolved


def _crowdnav_height_input_specs(config: dict[str, Any]) -> list[dict[str, str]]:
    """Describe the effective HEIGHT repo, checkpoint config, and default/model checkpoint."""
    repo_root = str(config.get("repo_root", CROWDNAV_HEIGHT_DEFAULT_REPO_ROOT))
    model_dir = Path(str(config.get("model_dir", CROWDNAV_HEIGHT_DEFAULT_MODEL_DIR)))
    checkpoint_name = str(config.get("checkpoint_name", CROWDNAV_HEIGHT_DEFAULT_CHECKPOINT_NAME))
    config_path = model_dir / "configs" / "config.py"
    checkpoint_path = model_dir / "checkpoints" / checkpoint_name
    return [
        {
            "kind": CROWDNAV_HEIGHT_REPO_ASSET_KIND,
            "name": "git",
            "reference": repo_root,
        },
        {
            "kind": CROWDNAV_HEIGHT_CONFIG_ASSET_KIND,
            "name": config_path.name,
            "reference": str(config_path),
        },
        {
            "kind": CROWDNAV_HEIGHT_CHECKPOINT_ASSET_KIND,
            "name": checkpoint_path.name,
            "reference": str(checkpoint_path),
        },
    ]


def _resolved_external_path(reference: str) -> Path:
    path = Path(reference)
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def _crowdnav_height_repo_identity(reference: str) -> dict[str, Any]:
    repo_path = _resolved_external_path(reference)
    try:
        revision = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(repo_path), "status", "--porcelain", "--untracked-files=all"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError, UnicodeError):
        return {"status": "unavailable", "reason": "crowdnav_height_repo_identity_unavailable"}
    if re.fullmatch(r"[0-9a-f]{40,64}", revision) is None:
        return {"status": "unavailable", "reason": "crowdnav_height_repo_revision_malformed"}
    if status:
        return {"status": "unavailable", "reason": "crowdnav_height_repo_dirty"}
    return {
        "status": "verified",
        "name": f"commit-{revision}",
        "sha256": hashlib.sha256(revision.encode("ascii")).hexdigest(),
    }


def _crowdnav_height_runtime_asset_identity(descriptor: dict[str, str]) -> dict[str, Any]:
    kind = descriptor["kind"]
    if kind == CROWDNAV_HEIGHT_REPO_ASSET_KIND:
        return _crowdnav_height_repo_identity(descriptor["reference"])
    try:
        path = _resolved_external_path(descriptor["reference"])
        if not path.is_file():
            return {"status": "unavailable", "reason": "crowdnav_height_runtime_file_missing"}
        return {"status": "verified", "name": descriptor["name"], "sha256": _sha256(path)}
    except (OSError, RuntimeError, ValueError):
        return {"status": "unavailable", "reason": "crowdnav_height_runtime_file_unavailable"}


def _runtime_model_input_descriptors(
    config: dict[str, Any], row: dict[str, Any]
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    descriptors: list[dict[str, str]] = []
    unresolved: list[dict[str, Any]] = []
    paths = _runtime_model_paths(config)
    path_keys = {key for key, _value in paths}
    for key, value in paths:
        components = _runtime_model_path_components(key, value)
        if components is None:
            unresolved.append(
                {"kind": key, "reason": "runtime_model_files_unavailable", "reference": value}
            )
            continue
        descriptors.extend(
            {"kind": key, "name": component.name, "reference": value} for component in components
        )

    unresolved.extend(_unresolved_runtime_model_references(config, row, path_keys))
    return descriptors, unresolved


def _unresolved_runtime_model_references(
    config: dict[str, Any], row: dict[str, Any], path_keys: set[str]
) -> list[dict[str, Any]]:
    unresolved = []
    for id_key in sorted(RUNTIME_MODEL_ID_KEYS):
        active_flag = RUNTIME_MODEL_ID_ACTIVE_FLAGS.get(id_key)
        if id_key not in config or (active_flag is not None and config.get(active_flag) is False):
            continue
        if RUNTIME_MODEL_ID_PATH_KEYS[id_key] not in path_keys:
            unresolved.append(
                {
                    "kind": id_key,
                    "reason": "registry_model_bytes_not_recorded",
                    "model_id": config.get(id_key),
                }
            )
    algorithm = row.get("algo")
    if isinstance(algorithm, str):
        expected = RUNTIME_DEFAULT_MODEL_REFERENCES.get(algorithm.strip().lower(), frozenset())
        if expected and not (path_keys & expected):
            unresolved.append(
                {
                    "kind": "default_model_reference",
                    "reason": "implicit_default_model_bytes_not_recorded",
                    "algorithm": algorithm,
                }
            )
    return unresolved


def _source_runtime_input_references(
    row: dict[str, Any], raw_assets: Any
) -> tuple[list[tuple[str, str, str]], list[dict[str, Any]]]:
    """Validate source asset references from the row and recorded names only."""
    params = row.get("scenario_params")
    config = _config_snapshot(row)
    if not isinstance(params, dict) or config is None:
        return [], [
            {"kind": "runtime_inputs", "reason": "scenario_or_planner_configuration_unavailable"}
        ]
    map_value = params.get("map_file")
    if not isinstance(map_value, str) or not map_value:
        return [], [{"kind": "scenario_map", "reason": "scenario_map_reference_unavailable"}]

    expected = [("scenario_map", Path(map_value).name, map_value)]
    model_expected, unresolved = _source_model_input_references(config, row, raw_assets)
    if row.get("algo") == "crowdnav_height":
        crowdnav_expected, crowdnav_unresolved = _source_crowdnav_height_references(
            config, raw_assets
        )
        model_expected.extend(crowdnav_expected)
        unresolved.extend(crowdnav_unresolved)
    return sorted(expected + model_expected), unresolved


def _source_crowdnav_height_references(
    config: dict[str, Any], raw_assets: Any
) -> tuple[list[tuple[str, str, str]], list[dict[str, Any]]]:
    specs = _crowdnav_height_input_specs(config)
    expected: list[tuple[str, str, str]] = []
    unresolved: list[dict[str, Any]] = []
    assets = raw_assets if isinstance(raw_assets, list) else []
    for spec in specs:
        if spec["kind"] == CROWDNAV_HEIGHT_REPO_ASSET_KIND:
            matching = [
                asset
                for asset in assets
                if isinstance(asset, dict)
                and asset.get("kind") == spec["kind"]
                and asset.get("reference") == spec["reference"]
            ]
            if len(matching) != 1:
                unresolved.append(
                    {
                        "kind": spec["kind"],
                        "reason": "crowdnav_height_source_repo_identity_missing",
                        "reference": spec["reference"],
                    }
                )
                continue
            name = matching[0].get("name")
            digest = matching[0].get("sha256")
            commit_match = (
                re.fullmatch(r"commit-([0-9a-f]{40,64})", name) if isinstance(name, str) else None
            )
            if (
                commit_match is None
                or digest != hashlib.sha256(commit_match[1].encode("ascii")).hexdigest()
            ):
                unresolved.append(
                    {
                        "kind": spec["kind"],
                        "reason": "crowdnav_height_source_repo_identity_invalid",
                        "reference": spec["reference"],
                    }
                )
                continue
            expected.append((spec["kind"], name, spec["reference"]))
        else:
            expected.append((spec["kind"], spec["name"], spec["reference"]))
    return expected, unresolved


def _source_model_input_references(
    config: dict[str, Any], row: dict[str, Any], raw_assets: Any
) -> tuple[list[tuple[str, str, str]], list[dict[str, Any]]]:
    expected: list[tuple[str, str, str]] = []
    unresolved: list[dict[str, Any]] = _invalid_runtime_model_path_values(config)
    paths = _runtime_model_paths(config)
    path_keys = {key for key, _value in paths}
    for key, value in paths:
        if key == "sacadrl_checkpoint_path":
            names, complete = _source_sacadrl_component_names(raw_assets, key, value)
            if not complete:
                unresolved.append(
                    {
                        "kind": key,
                        "reason": "source_checkpoint_bundle_incomplete",
                        "reference": value,
                    }
                )
            expected.extend((key, name, value) for name in names)
        else:
            expected.append((key, Path(value).expanduser().name, value))
    unresolved.extend(_unresolved_runtime_model_references(config, row, path_keys))
    return expected, unresolved


def _source_sacadrl_component_names(
    raw_assets: Any, key: str, value: str
) -> tuple[list[str], bool]:
    checkpoint = Path(value).expanduser()
    prefix = checkpoint.with_suffix("") if checkpoint.suffix == ".meta" else checkpoint
    matching = []
    if isinstance(raw_assets, list):
        matching = [
            asset
            for asset in raw_assets
            if isinstance(asset, dict)
            and asset.get("kind") == key
            and asset.get("reference") == value
        ]
    names = [asset.get("name") for asset in matching]
    complete = (
        f"{prefix.name}.meta" in names
        and f"{prefix.name}.index" in names
        and any(isinstance(name, str) and name.startswith(f"{prefix.name}.data") for name in names)
    )
    return [name for name in names if isinstance(name, str)], complete


def _valid_source_environment_identity(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    environment_fields = (
        "python_version",
        "python_implementation",
        "platform_system",
        "platform_release",
        "machine",
        "uv_lock_sha256",
    )
    if (
        value.get("schema_version") != SOURCE_ENVIRONMENT_PROVENANCE_SCHEMA
        or value.get("complete") is not True
        or not all(isinstance(value.get(key), str) and value[key] for key in environment_fields)
        or re.fullmatch(r"[0-9a-f]{64}", value["uv_lock_sha256"]) is None
    ):
        return False
    identity = {key: value[key] for key in environment_fields}
    expected = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return value.get("identity_sha256") == expected


def _execution_environment_identity() -> dict[str, Any]:
    """Capture the Python/platform/lock identity used by one materializer process."""
    lock_path = REPO_ROOT / "uv.lock"
    try:
        lock_sha256 = _sha256(lock_path) if lock_path.is_file() else None
    except OSError:
        lock_sha256 = None
    identity = {
        "python_version": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "machine": platform.machine(),
        "uv_lock_sha256": lock_sha256,
    }
    complete = all(isinstance(value, str) and value for value in identity.values()) and (
        re.fullmatch(r"[0-9a-f]{64}", identity["uv_lock_sha256"]) is not None
    )
    digest = (
        hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if complete
        else None
    )
    return {
        "schema_version": SOURCE_ENVIRONMENT_PROVENANCE_SCHEMA,
        "complete": complete,
        **identity,
        "identity_sha256": digest,
    }


def _source_runtime_input_identity(row: dict[str, Any]) -> dict[str, Any]:
    """Read source-side runtime hashes only when the source row recorded them."""
    provenance = row.get("runtime_input_provenance")
    revision = row.get("git_hash")
    if not isinstance(provenance, dict):
        return {
            "status": "unavailable",
            "reason": "historical_source_environment_or_runtime_inputs_not_recorded",
            "assets": [],
        }
    environment = provenance.get("source_environment")
    if (
        provenance.get("schema_version") != SOURCE_RUNTIME_INPUT_PROVENANCE_SCHEMA
        or provenance.get("source_revision") != revision
        or provenance.get("source_checkout_clean") is not True
        or provenance.get("complete") is not True
    ):
        return {
            "status": "unavailable",
            "reason": "source_runtime_input_provenance_invalid",
            "assets": [],
        }
    if not _valid_source_environment_identity(environment):
        return {
            "status": "unavailable",
            "reason": "historical_source_environment_not_recorded",
            "assets": [],
        }
    raw_assets = provenance.get("assets")
    expected_refs, unresolved = _source_runtime_input_references(row, raw_assets)
    if unresolved:
        return {
            "status": "unavailable",
            "reason": "source_runtime_input_reference_unresolved",
            "assets": [],
            "unresolved": unresolved,
        }
    if not isinstance(raw_assets, list) or not raw_assets:
        return {
            "status": "unavailable",
            "reason": "source_runtime_input_assets_missing",
            "assets": [],
        }
    actual_refs: list[tuple[str, str, str]] = []
    assets: list[dict[str, str]] = []
    for asset in raw_assets:
        if not isinstance(asset, dict) or set(asset) != {"kind", "name", "reference", "sha256"}:
            return {
                "status": "unavailable",
                "reason": "source_runtime_input_asset_malformed",
                "assets": [],
            }
        kind, name, reference, digest = (
            asset.get("kind"),
            asset.get("name"),
            asset.get("reference"),
            asset.get("sha256"),
        )
        if (
            not all(isinstance(value, str) and value for value in (kind, name, reference))
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            return {
                "status": "unavailable",
                "reason": "source_runtime_input_asset_malformed",
                "assets": [],
            }
        actual_refs.append((kind, name, reference))
        assets.append({"kind": kind, "name": name, "reference": reference, "sha256": digest})
    if sorted(actual_refs) != expected_refs:
        return {
            "status": "unavailable",
            "reason": "source_runtime_input_asset_set_incomplete",
            "assets": [],
        }
    assets.sort(key=lambda item: (item["kind"], item["name"], item["reference"]))
    return {
        "status": "verified",
        "revision": revision,
        "source_environment": environment,
        "source_environment_identity_sha256": environment["identity_sha256"],
        "assets": assets,
    }


def _scenario_map_runtime_path(params: dict[str, Any], matrix: Path | None) -> Path | None:
    map_value = params.get("map_file")
    if not isinstance(map_value, str) or not map_value:
        return None
    try:
        if Path(map_value).is_absolute():
            map_path = Path(map_value).resolve()
        elif matrix is not None:
            map_path = _scenario_map_path(params, matrix)
        else:
            candidate = (REPO_ROOT / map_value).resolve()
            map_path = candidate if candidate.is_file() else None
        return map_path if map_path is not None and map_path.is_file() else None
    except (OSError, RuntimeError):
        return None


def _runtime_input_identity(
    row: dict[str, Any], revision: Any, *, matrix: Path | None
) -> dict[str, Any]:
    """Verify replay runtime input bytes against the recorded replay Git tree."""
    if not isinstance(revision, str) or not revision:
        return {"status": "unavailable", "reason": "revision_unavailable", "assets": []}
    descriptors, unresolved = _runtime_input_descriptors(row, matrix=matrix)
    if unresolved:
        return {
            "status": "unavailable",
            "reason": "runtime_input_reference_unresolved",
            "assets": [],
            "unresolved": unresolved,
        }
    assets = []
    for descriptor in descriptors:
        if descriptor["kind"] in CROWDNAV_HEIGHT_ASSET_KINDS:
            external_identity = _crowdnav_height_runtime_asset_identity(descriptor)
            asset = {
                "kind": descriptor["kind"],
                "name": external_identity.get("name", descriptor["name"]),
                "reference": descriptor["reference"],
                "status": external_identity.get("status"),
                "sha256": external_identity.get("sha256"),
            }
            if external_identity.get("reason"):
                asset["reason"] = external_identity["reason"]
            assets.append(asset)
            continue
        path_value = descriptor["reference"]
        if descriptor["kind"] == "scenario_map":
            params = row.get("scenario_params")
            path = _scenario_map_runtime_path(params, matrix) if isinstance(params, dict) else None
        else:
            path = _runtime_model_path_components(descriptor["kind"], path_value)
            # Component descriptors name individual TensorFlow files. The path
            # reference is the configured prefix; resolve the matching component.
            if isinstance(path, list):
                path = next((item for item in path if item.name == descriptor["name"]), None)
        if not isinstance(path, Path) or not path.is_file():
            assets.append(
                {**descriptor, "status": "unavailable", "reason": "runtime_file_unavailable"}
            )
            continue
        identity = _git_tree_file_identity(revision, path, kind=descriptor["kind"])
        assets.append(
            {
                "kind": descriptor["kind"],
                "name": descriptor["name"],
                "reference": descriptor["reference"],
                "status": identity.get("status"),
                "sha256": identity.get("sha256"),
                **({"reason": identity["reason"]} if identity.get("reason") else {}),
            }
        )
    assets.sort(
        key=lambda item: (str(item.get("kind")), str(item.get("name")), str(item.get("reference")))
    )
    if any(asset.get("status") != "verified" for asset in assets):
        return {
            "status": "unavailable",
            "reason": "runtime_asset_not_source_bound",
            "assets": assets,
        }
    return {"status": "verified", "revision": revision, "assets": assets}


def _matching_runtime_input_identity(
    source_row: dict[str, Any],
    replay_row: dict[str, Any],
    source_revision: Any,
    replay_revision: Any,
    *,
    source_matrix: Path | None,
    replay_matrix: Path | None,
    replay_environment_identity: Any,
) -> dict[str, Any]:
    source_identity = _source_runtime_input_identity(source_row)
    replay_identity = _runtime_input_identity(replay_row, replay_revision, matrix=replay_matrix)
    result: dict[str, Any] = {
        "status": "unavailable",
        "source": source_identity,
        "replay": replay_identity,
    }
    if source_identity.get("status") != "verified" or replay_identity.get("status") != "verified":
        return result
    if not _valid_source_environment_identity(replay_environment_identity):
        result["status"] = "replay_environment_unavailable"
        result["replay_environment_identity"] = replay_environment_identity
        return result
    result["replay_environment_identity"] = replay_environment_identity
    if source_identity.get("source_environment_identity_sha256") != replay_environment_identity.get(
        "identity_sha256"
    ):
        result["status"] = "replay_environment_mismatch"
        return result
    source_assets = [
        (asset["kind"], asset["name"], asset["sha256"]) for asset in source_identity["assets"]
    ]
    replay_assets = [
        (asset["kind"], asset["name"], asset["sha256"]) for asset in replay_identity["assets"]
    ]
    if sorted(source_assets) != sorted(replay_assets):
        result["status"] = "mismatch"
        return result
    result["status"] = "verified_git_tree_match"
    return result


def _exact_match_runtime_input_status(
    source_row: dict[str, Any],
    replay_row: dict[str, Any],
    source_revision: Any,
    replay_revision: Any,
    *,
    source_matrix: Path | None,
    replay_matrix: Path | None,
    replay_environment_identity: Any,
) -> tuple[str, dict[str, Any]]:
    identity = _matching_runtime_input_identity(
        source_row,
        replay_row,
        source_revision,
        replay_revision,
        source_matrix=source_matrix,
        replay_matrix=replay_matrix,
        replay_environment_identity=replay_environment_identity,
    )
    status = {
        "verified_git_tree_match": "exact_match",
        "mismatch": "runtime_input_identity_mismatch",
        "replay_environment_mismatch": "replay_environment_mismatch",
        "replay_environment_unavailable": "unavailable_replay_environment_identity",
    }.get(identity["status"], "unavailable_runtime_input_identity")
    return status, identity


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
    replay_environment_identity = _execution_environment_identity()
    replay_checkout_before: dict[str, Any] = {"capture_status": "unavailable"}
    replay_revision: Any = None
    try:
        with (
            stdout_path.open("w", encoding="utf-8") as stdout,
            stderr_path.open("w", encoding="utf-8") as stderr,
        ):
            replay_checkout_before = _capture_replay_checkout_provenance()
            replay_revision = replay_checkout_before.get("revision")
            completed = subprocess.run(
                command, cwd=REPO_ROOT, stdout=stdout, stderr=stderr, check=False
            )
    except OSError as exc:
        replay_checkout_after = _capture_replay_checkout_provenance()
        checkout_fields = _replay_checkout_run_fields(
            replay_checkout_before, replay_checkout_after, replay_revision
        )
        return {
            "attempted": True,
            "status": "runner_unavailable",
            "command": command,
            "command_shell": shlex.join(command),
            "error": str(exc),
            "source_revision": row.get("git_hash"),
            "replay_revision": replay_revision,
            "replay_environment_identity": replay_environment_identity,
            **checkout_fields,
        }
    replay_checkout_after = _capture_replay_checkout_provenance()
    checkout_fields = _replay_checkout_run_fields(
        replay_checkout_before, replay_checkout_after, replay_revision
    )
    result: dict[str, Any] = {
        "attempted": True,
        "command": command,
        "command_shell": shlex.join(command),
        "returncode": completed.returncode,
        "replay_revision": replay_revision,
        "source_revision": row.get("git_hash"),
        "same_repository_revision": row.get("git_hash") == replay_revision,
        "replay_environment_identity": replay_environment_identity,
        **checkout_fields,
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
    result.update(
        _classify_replay_row(
            case,
            row,
            actual,
            replay_revision,
            source_matrix=matrix,
            replay_matrix=replay_matrix,
            replay_checkout={
                "replay_checkout_clean": checkout_fields["replay_checkout_clean"],
                "replay_checkout_stability_status": checkout_fields[
                    "replay_checkout_stability_status"
                ],
            },
            replay_environment_identity=replay_environment_identity,
        )
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
    statuses = (
        source_mode,
        replay_mode,
        source_metadata.get("status"),
        replay_metadata.get("status"),
    )
    if any(isinstance(value, str) and value in {"fallback", "degraded"} for value in statuses):
        status = "fallback_or_degraded_not_evidence"
    elif (
        not isinstance(source_mode, str)
        or source_mode not in SUPPORTED_EXECUTION_MODES
        or not isinstance(replay_mode, str)
        or replay_mode not in SUPPORTED_EXECUTION_MODES
    ):
        status = "unavailable_execution_mode_identity"
    else:
        status = "match" if source_mode == replay_mode else "execution_mode_identity_mismatch"
    return {"source": source_mode, "replay": replay_mode, "status": status}


def _availability_marker_issue(key: str, value: Any) -> str | None:
    """Classify an explicit availability marker, leaving absent fields unknown."""
    if key not in {"availability", "availability_status", "available", "unavailable"}:
        return None
    if key == "unavailable":
        if not isinstance(value, bool):
            return "unavailable"
        return "unavailable" if value else None
    if key == "available":
        return None if value is True else "unavailable"
    if key == "availability" and isinstance(value, dict):
        status = value.get("availability_status", value.get("status"))
        if not isinstance(status, str):
            return "unavailable"
        normalized = status.strip().lower().replace("-", "_")
        return None if normalized in AVAILABLE_EXECUTION_STATUSES else "unavailable"
    if key == "availability" and isinstance(value, bool):
        return None if value else "unavailable"
    normalized = value.strip().lower().replace("-", "_") if isinstance(value, str) else ""
    return None if normalized in AVAILABLE_EXECUTION_STATUSES else "unavailable"


def _runtime_status_marker_issue(key: str, value: Any) -> str | None:
    """Classify execution status values from nested operational metadata."""
    if key == "benchmark_success":
        return None if value is True else "failed"
    if key != "status" and not key.endswith("_status"):
        return None
    normalized = value.strip().lower().replace("-", "_") if isinstance(value, str) else ""
    if normalized in {"fallback", "degraded"}:
        return "fallback"
    if normalized in {"unknown", "unavailable", "not_available", "missing", "not_run"}:
        return "unavailable"
    if normalized in FAILED_EXECUTION_STATUSES:
        return "failed"
    return None if normalized in AVAILABLE_EXECUTION_STATUSES else "unavailable"


def _runtime_marker_issue(key: str, value: Any) -> str | None:
    """Classify one runtime status, availability, or fallback marker."""
    if key in FALLBACK_MARKER_KEYS:
        if not isinstance(value, bool):
            return "unavailable"
        return "fallback" if value else None
    if key == "fallback_count" or key.endswith("_fallback_count"):
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or (isinstance(value, float) and not math.isfinite(value))
            or value < 0
        ):
            return "unavailable"
        return "fallback" if value > 0 else None
    if key == "fallback_reason":
        if value is None or value == "":
            return None
        return "fallback" if isinstance(value, str) else "unavailable"
    if key == "benchmark_eligible" and value is not True:
        return "unavailable"
    return _availability_marker_issue(key, value) or _runtime_status_marker_issue(key, value)


def _runtime_marker_issues(value: Any, *, root: bool = False) -> set[str]:
    """Collect runtime marker issues while excluding outcome and planner config payloads."""
    issues: set[str] = set()
    if isinstance(value, dict):
        for raw_key, child in value.items():
            key = str(raw_key).strip().lower()
            if key in {"config", "planner_contract", "safety_shield_contract"}:
                continue
            if root and key in {"outcome", "metrics", "scenario_params", "status"}:
                continue
            issue = _runtime_marker_issue(key, child)
            if issue is not None:
                issues.add(issue)
            issues.update(_runtime_marker_issues(child))
    elif isinstance(value, list):
        for child in value:
            issues.update(_runtime_marker_issues(child))
    return issues


def _row_execution_evidence_status(row: dict[str, Any]) -> str:
    """Require successful, available execution metadata without reading outcome status.

    Top-level episode ``status`` describes the scenario outcome (for example, collision
    or failure), so it is deliberately excluded. Runtime status and fallback markers
    are read from the row's operational metadata instead.
    """
    metadata = row.get("algorithm_metadata")
    if not isinstance(metadata, dict):
        return "unavailable_execution_status"
    metadata_status = metadata.get("status")
    normalized_metadata_status = (
        metadata_status.strip().lower().replace("-", "_")
        if isinstance(metadata_status, str)
        else ""
    )
    if normalized_metadata_status in {"fallback", "degraded"}:
        return "fallback_or_degraded_not_evidence"
    if normalized_metadata_status not in SUCCESSFUL_EXECUTION_STATUSES:
        return "unsuccessful_execution_status"
    if "benchmark_eligible" in row and row.get("benchmark_eligible") is not True:
        return "unavailable_benchmark_eligibility"
    issues = _runtime_marker_issues(row, root=True)
    if "fallback" in issues:
        return "fallback_or_degraded_not_evidence"
    if "failed" in issues:
        return "unsuccessful_execution_status"
    if "unavailable" in issues:
        return "unavailable_execution_availability"
    return "available"


def _row_invalid_run_status(row: dict[str, Any]) -> str:
    """Classify canonical or explicit invalid-run evidence without treating outcomes as failures."""
    explicit_status = row.get("invalid_run")
    if "invalid_run" in row:
        if explicit_status is True:
            return "invalid"
        if explicit_status is not False:
            return "unavailable"

    stored_ledger = row.get("event_ledger")
    if isinstance(stored_ledger, dict):
        exact_events = stored_ledger.get("exact_events")
        if isinstance(exact_events, dict) and "invalid_run" in exact_events:
            stored_invalid_run = exact_events["invalid_run"]
            if stored_invalid_run is True:
                return "invalid"
            if stored_invalid_run is not False:
                return "unavailable"

    try:
        canonical_ledger = build_event_ledger(row)
    except (AttributeError, KeyError, OverflowError, TypeError, ValueError):
        return "unavailable"
    canonical_exact_events = canonical_ledger.get("exact_events")
    if not isinstance(canonical_exact_events, dict):
        return "unavailable"
    return "invalid" if canonical_exact_events.get("invalid_run") is True else "available"


def _replay_execution_evidence_status(
    case: dict[str, Any], source_row: dict[str, Any], replay_row: dict[str, Any]
) -> tuple[str, str, str, str, str | None]:
    """Summarize source and replay runtime evidence before metric comparison."""
    source_status = _row_execution_evidence_status(source_row)
    replay_status = _row_execution_evidence_status(replay_row)
    source_invalid_run = _row_invalid_run_status(source_row)
    replay_invalid_run = _row_invalid_run_status(replay_row)
    if "invalid" in {source_invalid_run, replay_invalid_run}:
        rejection = "invalid_run_not_evidence"
    elif "unavailable" in {source_invalid_run, replay_invalid_run}:
        rejection = "unavailable_invalid_run_evidence"
    elif case.get("benchmark_eligible") is not True:
        rejection = "unavailable_source_row_not_benchmark_eligible"
    elif "fallback_or_degraded_not_evidence" in {source_status, replay_status}:
        rejection = "fallback_or_degraded_not_evidence"
    elif source_status != "available" or replay_status != "available":
        rejection = "unavailable_execution_evidence"
    else:
        rejection = None
    return source_status, replay_status, source_invalid_run, replay_invalid_run, rejection


def _exact_match_checkout_status(
    replay_checkout_clean: bool | None, stability_status: str | None
) -> str:
    """Gate a metric match on checkout evidence captured around the replay."""
    if stability_status is None:
        return (
            "replay_checkout_dirty"
            if replay_checkout_clean is False
            else "replay_checkout_cleanliness_unavailable"
        )
    if stability_status == "clean_stable" and replay_checkout_clean is True:
        return "exact_match"
    return {
        "dirty": "replay_checkout_dirty",
        "head_changed": "replay_checkout_head_changed_during_run",
        "working_tree_changed": "replay_checkout_changed_during_run",
    }.get(stability_status, "replay_checkout_cleanliness_unavailable")


def _classify_replay_row(
    case: dict[str, Any],
    source_row: dict[str, Any],
    replay_row: dict[str, Any],
    replay_revision: Any,
    *,
    source_matrix: Path | None = None,
    replay_matrix: Path | None = None,
    replay_checkout: dict[str, Any] | None,
    replay_environment_identity: Any = None,
) -> dict[str, Any]:
    """Derive replay classification from the checksum-pinned source and replay rows."""
    source_revision = source_row.get("git_hash")
    replay_checkout = replay_checkout if isinstance(replay_checkout, dict) else {}
    same_revision = (
        isinstance(source_revision, str)
        and bool(source_revision)
        and isinstance(replay_revision, str)
        and bool(replay_revision)
        and source_revision == replay_revision
    )
    result: dict[str, Any] = {
        "source_revision": source_revision,
        "replay_revision": replay_revision,
        "same_repository_revision": same_revision,
        "replay_checkout_clean": replay_checkout.get("replay_checkout_clean"),
        "replay_checkout_stability_status": replay_checkout.get("replay_checkout_stability_status"),
        "replay_environment_identity": replay_environment_identity,
        "identity": _replay_identity(case, replay_row),
        "episode_status": replay_row.get("status"),
        "config_hash_source": (
            (source_row.get("algorithm_metadata") or {}).get("config_hash")
            if isinstance(source_row.get("algorithm_metadata"), dict)
            else None
        ),
        "config_hash_replay": (
            (replay_row.get("algorithm_metadata") or {}).get("config_hash")
            if isinstance(replay_row.get("algorithm_metadata"), dict)
            else None
        ),
    }
    if result["identity"]["status"] != "match":
        result["status"] = "replay_identity_mismatch"
        return result

    (
        source_execution_status,
        replay_execution_status,
        source_invalid_run_status,
        replay_invalid_run_status,
        execution_rejection,
    ) = _replay_execution_evidence_status(case, source_row, replay_row)
    result["execution_evidence_source"] = source_execution_status
    result["execution_evidence_replay"] = replay_execution_status
    result["invalid_run_evidence_source"] = source_invalid_run_status
    result["invalid_run_evidence_replay"] = replay_invalid_run_status
    if execution_rejection is not None:
        result["status"] = execution_rejection
        return result

    mode_identity = _execution_mode_identity(source_row, replay_row)
    result["execution_mode"] = mode_identity["replay"]
    result["execution_mode_source"] = mode_identity["source"]
    if mode_identity["status"] != "match":
        result["status"] = mode_identity["status"]
        return result
    config_hash_source = result["config_hash_source"]
    config_hash_replay = result["config_hash_replay"]
    if not all(
        isinstance(value, str) and bool(value.strip())
        for value in (config_hash_source, config_hash_replay)
    ):
        result["status"] = "unavailable_planner_config_identity"
        return result
    if config_hash_source != config_hash_replay:
        result["status"] = "planner_config_identity_mismatch"
        return result

    comparison = _compare(source_row, replay_row)
    result["comparison"] = comparison
    if comparison["overall"] == "mismatch":
        result["status"] = (
            "mismatch_same_revision" if same_revision else "mismatch_different_revision"
        )
    elif not same_revision:
        result["status"] = (
            "matched_different_revision"
            if comparison["overall"] == "match"
            else "incomplete_different_revision"
        )
    else:
        result["status"] = "incomplete_same_revision"
        if comparison["overall"] == "match":
            runtime_status, runtime_input_identity = _exact_match_runtime_input_status(
                source_row,
                replay_row,
                source_revision,
                replay_revision,
                source_matrix=source_matrix,
                replay_matrix=replay_matrix,
                replay_environment_identity=replay_environment_identity,
            )
            result["runtime_input_identity"] = runtime_input_identity
            result["status"] = runtime_status
        if result["status"] == "exact_match":
            result["status"] = _exact_match_checkout_status(
                replay_checkout.get("replay_checkout_clean"),
                replay_checkout.get("replay_checkout_stability_status"),
            )
    return result


def _replay_checkout_provenance() -> dict[str, Any]:
    """Capture the replay HEAD and whether its source checkout had any changes."""
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout.strip()
        status_output = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise MaterializationError(f"could not identify replay source checkout: {exc}") from exc
    if not revision:
        raise MaterializationError("could not identify replay source revision")
    status_entries = [line for line in status_output.splitlines() if line]
    return {
        "schema": REPLAY_CHECKOUT_SNAPSHOT_SCHEMA,
        "revision": revision,
        "clean": not status_entries,
        "status_sha256": hashlib.sha256(status_output.encode("utf-8")).hexdigest(),
        "status_entries": status_entries,
    }


def _capture_replay_checkout_provenance() -> dict[str, Any]:
    """Capture checkout state without losing an otherwise useful replay attempt."""
    try:
        return _replay_checkout_provenance()
    except MaterializationError as exc:
        return {"capture_status": "unavailable", "error": str(exc)}


def _replay_checkout_stability_status(before: Any, after: Any, replay_revision: Any) -> str:
    """Validate the replay's before/after checkout binding, failing closed on gaps."""

    def valid_snapshot(snapshot: Any) -> bool:
        if not isinstance(snapshot, dict):
            return False
        revision = snapshot.get("revision")
        clean = snapshot.get("clean")
        entries = snapshot.get("status_entries")
        status_sha256 = snapshot.get("status_sha256")
        if (
            snapshot.get("schema") != REPLAY_CHECKOUT_SNAPSHOT_SCHEMA
            or not isinstance(revision, str)
            or not revision.strip()
            or not isinstance(clean, bool)
            or not isinstance(entries, list)
            or any(
                not isinstance(entry, str) or not entry or "\n" in entry or "\r" in entry
                for entry in entries
            )
            or not isinstance(status_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", status_sha256) is None
            or clean is not (len(entries) == 0)
        ):
            return False
        status_output = "\n".join(entries) + ("\n" if entries else "")
        return hashlib.sha256(status_output.encode("utf-8")).hexdigest() == status_sha256

    if not valid_snapshot(before) or not valid_snapshot(after):
        return "unavailable"
    if not isinstance(replay_revision, str) or replay_revision != before["revision"]:
        return "unavailable"
    if before["revision"] != after["revision"]:
        return "head_changed"
    if before["clean"] is not True or after["clean"] is not True:
        return "dirty"
    if before["status_sha256"] != after["status_sha256"]:
        return "working_tree_changed"
    return "clean_stable"


def _replay_checkout_run_fields(
    before: dict[str, Any], after: dict[str, Any], replay_revision: Any
) -> dict[str, Any]:
    """Build JSON-safe replay provenance and its fail-closed exact-match gate."""
    stability_status = _replay_checkout_stability_status(before, after, replay_revision)
    if stability_status == "clean_stable":
        checkout_clean: bool | None = True
    elif stability_status in {"dirty", "head_changed", "working_tree_changed"}:
        checkout_clean = False
    else:
        checkout_clean = None
    return {
        "replay_checkout_before": before,
        "replay_checkout_after": after,
        "replay_checkout_stability_status": stability_status,
        "replay_checkout_clean": checkout_clean,
        "replay_checkout_status_sha256": before.get("status_sha256"),
        "replay_checkout_post_status_sha256": after.get("status_sha256"),
    }


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
        or previous_source.get("source_campaign_id", previous_source.get("campaign_id"))
        != source.get("campaign_id")
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
        replay["episode_output_sha256"] = previous_hash
        replay["episode_output_observed_sha256"] = copied_hash
    elif isinstance(previous_hash, str):
        replay["episode_output_checksum_status"] = "verified"
        replay["episode_output_sha256"] = previous_hash
        replay["episode_output_observed_sha256"] = copied_hash
    else:
        replay["episode_output_checksum_status"] = "calculated_on_resume_prior_receipt_unavailable"
        replay["episode_output_sha256"] = copied_hash


def _annotate_reused_replay_classification(
    replay: dict[str, Any],
    case: dict[str, Any],
    row: dict[str, Any],
    case_dir: Path,
    *,
    matrix: Path,
    prior_checksum_verified: bool,
    manifest_receipt_matches: bool,
) -> None:
    """Recompute a reused result from its copied row and checksum-pinned source row."""
    replay["resume_receipt_consistency_status"] = (
        "match" if manifest_receipt_matches else "manifest_case_replay_mismatch"
    )
    checksum_mismatch = replay.get("episode_output_checksum_status") == "mismatch"
    episode_output = replay.get("episode_output")
    if not isinstance(episode_output, str):
        replay["status"] = (
            "replay_artifact_checksum_mismatch" if checksum_mismatch else "unavailable_output"
        )
        return
    episode_path = _safe_relative(case_dir, episode_output, label="reused replay episode output")
    replay_rows, malformed_rows = _read_replay_rows(episode_path)
    replay["replay_row_count"] = len(replay_rows)
    replay["replay_malformed_line_count"] = malformed_rows
    if len(replay_rows) != 1 or malformed_rows:
        replay["status"] = (
            "replay_artifact_checksum_mismatch"
            if checksum_mismatch
            else "invalid_output"
            if malformed_rows
            else "unavailable_output"
        )
        return
    returncode = replay.get("returncode")
    if not isinstance(returncode, int) or isinstance(returncode, bool):
        replay["status"] = (
            "replay_artifact_checksum_mismatch"
            if checksum_mismatch
            else "resume_runner_returncode_unavailable"
        )
        return
    if returncode != 0:
        replay["status"] = (
            "replay_artifact_checksum_mismatch" if checksum_mismatch else "runner_failed"
        )
        return

    checkout_before = replay.get("replay_checkout_before")
    checkout_after = replay.get("replay_checkout_after")
    checkout_stability = _replay_checkout_stability_status(
        checkout_before, checkout_after, replay.get("replay_revision")
    )
    replay_checkout_clean = (
        True
        if checkout_stability == "clean_stable"
        else False
        if checkout_stability in {"dirty", "head_changed", "working_tree_changed"}
        else None
    )
    # Recompute these fields from the preserved snapshots. In particular, a legacy
    # pre-run-only cleanliness flag is not evidence about the checkout after replay.
    replay["replay_checkout_stability_status"] = checkout_stability
    replay["replay_checkout_clean"] = replay_checkout_clean
    classification = _classify_replay_row(
        case,
        row,
        replay_rows[0],
        replay.get("replay_revision"),
        source_matrix=matrix,
        replay_matrix=(
            case_dir / "replay_input" / "replay_matrix.yaml"
            if (case_dir / "replay_input" / "replay_matrix.yaml").is_file()
            else matrix
        ),
        replay_checkout={
            "replay_checkout_clean": replay_checkout_clean,
            "replay_checkout_stability_status": checkout_stability,
        },
        # Reuse the environment captured by the original replay attempt. The
        # current materializer environment cannot stand in for missing history.
        replay_environment_identity=replay.get("replay_environment_identity"),
    )
    replay.update(classification)
    if checksum_mismatch:
        replay["status"] = "replay_artifact_checksum_mismatch"
    elif not manifest_receipt_matches and classification.get("status") == "exact_match":
        replay["resume_receipt_consistency_status"] = "manifest_case_replay_mismatch"
        replay["status"] = "replay_receipt_manifest_mismatch"
    elif not prior_checksum_verified and classification.get("status") == "exact_match":
        replay["checksum_unverified_derived_status"] = "exact_match"
        replay["status"] = "replay_artifact_checksum_unverified"


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
    replay_checkout = _replay_checkout_provenance()
    materializer_environment_identity = _execution_environment_identity()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = _read_object(summary_path)
    source_provenance = _verify_inputs(summary, campaign_root, args.bundle, matrix)
    summary_sha256 = _sha256(summary_path)
    resume_root, resume_records = _load_resume_records(args.resume_from, source=source_provenance)
    resume_manifest_sha256 = None
    resume_summary_sha256 = None
    resume_source_snapshot_revision = None
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
        resume_source_snapshot_revision = _resume_showcase_snapshot_revision(resume_manifest)
    campaign = _read_object(campaign_root / "campaign_manifest.json")
    cases = _selected_cases(summary)
    replay_revision = replay_checkout["revision"]
    case_records = []
    replay_candidates = []
    reused_count = 0
    for case in cases:
        row, source_ref = _load_source_row(campaign_root, case)
        selector_outcome, selector_metrics = _selector_case_measurements(case, row)
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
                "outcome": selector_outcome,
                "metrics": selector_metrics,
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
            manifest_receipt_matches = previous.get("replay") == previous_replay
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
                    _annotate_reused_replay_classification(
                        previous_replay,
                        case,
                        row,
                        case_dir,
                        matrix=matrix,
                        prior_checksum_verified=(
                            previous_replay.get("episode_output_checksum_status") == "verified"
                            and previous_replay.get("episode_output_checksum_origin")
                            == "captured_at_run"
                        ),
                        manifest_receipt_matches=manifest_receipt_matches,
                    )
                    record["replay"] = {
                        **previous_replay,
                        "reused": True,
                        "reused_from_summary_sha256": resume_summary_sha256,
                        "reused_from_manifest_sha256": resume_manifest_sha256,
                    }
                    reused_count += 1
                else:
                    prior_status = previous_replay.get("status")
                    record["replay"] = {
                        **previous_replay,
                        "status": "replay_artifact_missing_on_resume",
                        "resume_prior_status": prior_status,
                        "resume_artifact_status": "missing",
                        "resume_artifact_reason": "prior attempted replay directory is missing",
                        "episode_output_checksum_status": "missing",
                        "reused": False,
                    }
        _write_json(case_dir / "case.json", record)
        case_records.append(record)
        if not ineligible and not record["replay"].get("attempted"):
            replay_candidates.append((case, row, record, case_dir))
    attempted_cases = replay_candidates[: args.replay_limit]
    for case, row, record, case_dir in attempted_cases:
        record["replay"] = _run_replay(
            case,
            row,
            matrix,
            case_dir,
            campaign,
        )
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
            **{key: value for key, value in source_provenance.items() if key != "campaign_id"},
            "source_campaign_id": source_provenance["campaign_id"],
            "summary_sha256": summary_sha256,
            **_showcase_tool_snapshot(
                summary,
                replay_revision,
                fallback_snapshot_revision=resume_source_snapshot_revision,
            ),
        },
        "source_showcase_diagnostics": _compact_source_showcase_diagnostics(summary),
        "execution_environment": {
            "source_environment": "not_recorded_in_release_bundle",
            "materializer_environment_identity": materializer_environment_identity,
            "replay_attempt_environments": {
                record["case_id"]: record["replay"].get("replay_environment_identity")
                for record in case_records
                if record["replay"].get("attempted") is True
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
            "preserved_attempts_missing_artifacts": sum(
                record["replay"].get("status") == "replay_artifact_missing_on_resume"
                for record in case_records
            ),
            "status_counts": dict(sorted(replay_counts.items())),
            "replay_revision": replay_revisions[0] if len(replay_revisions) == 1 else None,
            "replay_revisions": replay_revisions,
            "materializer_revision": replay_revision,
            "materializer_checkout_clean": replay_checkout["clean"],
            "materializer_checkout_status_sha256": replay_checkout["status_sha256"],
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
    materializer_environment = environment.get("materializer_environment_identity", {})
    replay_attempt_environments = environment.get("replay_attempt_environments", {})
    replay_environment_hashes = {
        case_id: (
            identity.get("identity_sha256")
            if isinstance(identity, dict) and _valid_source_environment_identity(identity)
            else "unavailable"
        )
        for case_id, identity in replay_attempt_environments.items()
    }
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
        f"- Source campaign: `{manifest['source']['source_campaign_id']}`",
        f"- Source revision: `{manifest['source']['source_revision']}`",
        f"- Showcase execution revision: `{manifest['source']['showcase_tool_revision']}`",
        f"- Showcase source snapshot: `{manifest['source']['showcase_tool_source_snapshot_status']}`"
        + (
            f" at `{manifest['source']['showcase_tool_source_snapshot_revision']}`"
            if manifest["source"].get("showcase_tool_source_snapshot_revision")
            else ""
        ),
        f"- Source bundle SHA-256: `{manifest['source']['bundle_sha256']}`",
        f"- Scenario matrix SHA-256: `{manifest['source']['matrix_sha256']}`",
        f"- Source execution environment: `{environment.get('source_environment', 'not_recorded')}`",
        f"- Materializer environment: Python `{materializer_environment.get('python_version')}`, "
        f"{materializer_environment.get('platform_system')} "
        f"`{materializer_environment.get('platform_release')}`; lock SHA-256 "
        f"`{materializer_environment.get('uv_lock_sha256')}`",
        f"- Replay-attempt environment identity SHA-256 values: `{replay_environment_hashes}`",
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
