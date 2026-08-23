"""Fail-closed acceptance checks for a full benchmark-data release.

The camera-ready campaign runner intentionally supports partial and
``core``-only success semantics for exploratory work.  A publication release
has a stricter contract: every declared arm must complete every declared
scenario/seed cell, and no fallback or degraded row may be promoted.  This
module keeps that publication gate separate from the bounded runtime smoke.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from robot_sf.benchmark.camera_ready._config import _load_campaign_scenarios
from robot_sf.benchmark.camera_ready._preflight import _resolved_seed_inventory
from robot_sf.benchmark.camera_ready._run_state import _resolve_integrity_artifact_path
from robot_sf.benchmark.camera_ready_campaign import load_campaign_config
from robot_sf.benchmark.fallback_policy import runtime_fallback_or_degraded_marker

if TYPE_CHECKING:
    from pathlib import Path

FULL_RELEASE_SCHEMA_VERSION = "benchmark-release-manifest.v0.2"
FULL_RELEASE_EXPECTED_PLANNER_ARMS = 14
FULL_RELEASE_EXPECTED_EPISODE_CELLS = 20_160
FULL_RELEASE_EXPECTED_HORIZON_STEPS = 600
FULL_RELEASE_KINEMATICS = "differential_drive"
FULL_RELEASE_ACCEPTANCE_SCHEMA_VERSION = "benchmark-full-release-acceptance.v1"

_FORBIDDEN_STATUSES = frozenset(
    {
        "degraded",
        "error",
        "excluded",
        "failed",
        "fallback",
        "not-available",
        "not_available",
        "partial-failure",
        "partial_failure",
        "placeholder",
        "unavailable",
    }
)
_FORBIDDEN_STATUS_PREFIXES = ("predictive_foresight_model_fallback",)
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_LEGACY_EMERGENCY_MODES = frozenset({"emergency_stop", "reorient"})
_LEGACY_EMERGENCY_SOURCES = frozenset({"all_candidates_rejected", "static_reorient"})


def _normalized_status(value: Any) -> str:
    """Normalize a status marker without treating absent fields as failures.

    Returns:
        Lowercase status token, or an empty string for an absent value.
    """
    return str(value).strip().lower().replace(" ", "_") if value is not None else ""


def _strict_int(value: Any) -> int | None:
    """Parse only integral JSON values or their decimal-string representation.

    Returns:
        Parsed integer, or ``None`` for non-integral values.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value)
    return None


def _append_blocker(blockers: list[str], message: str) -> None:
    """Add a deterministic blocker while bounding pathological row-level output."""
    if message not in blockers and len(blockers) < 100:
        blockers.append(message)


def _emergency_stop_marker(payload: Any) -> tuple[str, str] | None:
    """Reject legacy emergency-stop paths without changing fallback counters.

    Any positive ``emergency_stop_count`` is forbidden for this release.  The
    generic fallback marker remains responsible for ``fallback_count`` so its
    semantics do not change as this stricter release gate evolves.

    Returns:
        Path/value pair for the first legacy or insufficient emergency marker.
    """
    if not isinstance(payload, Mapping):
        return None

    def _normalized(value: Any) -> str:
        return str(value).strip().lower().replace("-", "_")

    fields = [
        ("selected_source", payload.get("selected_source")),
        ("planner_mode", payload.get("planner_mode")),
    ]
    decision = payload.get("last_decision")
    if isinstance(decision, Mapping):
        fields.extend(
            [
                ("last_decision.selected_source", decision.get("selected_source")),
                ("last_decision.planner_mode", decision.get("planner_mode")),
            ]
        )
    for path, value in fields:
        normalized_source = _normalized(value)
        if path.endswith("selected_source") and normalized_source in _LEGACY_EMERGENCY_SOURCES:
            return path, normalized_source
        if path.endswith("planner_mode") and normalized_source in _LEGACY_EMERGENCY_MODES:
            return path, normalized_source

    counter = payload.get("emergency_stop_count")
    if counter is None:
        return None
    parsed_counter = _strict_int(counter)
    if parsed_counter is None or parsed_counter < 0:
        return "emergency_stop_count", "invalid"
    if parsed_counter > 0:
        return "emergency_stop_count", str(counter)
    return None


def _status_markers(  # noqa: C901, PLR0912
    payload: Mapping[str, Any], prefix: str
) -> list[tuple[str, str]]:
    """Extract only execution/evidence status markers from one structured row.

    Returns:
        Path/value pairs for forbidden execution or evidence markers.
    """
    markers: list[tuple[str, str]] = []

    def _add(path: str, value: Any) -> None:
        status = _normalized_status(value)
        if status in _FORBIDDEN_STATUSES or any(
            status.startswith(prefix) for prefix in _FORBIDDEN_STATUS_PREFIXES
        ):
            markers.append((f"{prefix}.{path}", status))

    for field in ("status", "row_status", "readiness_status", "availability_status"):
        if field in payload:
            _add(field, payload[field])
    for field in ("evidence_status", "execution_status"):
        if field in payload:
            _add(field, payload[field])
    benchmark_success = payload.get("benchmark_success")
    if benchmark_success is False or (
        isinstance(benchmark_success, str)
        and benchmark_success.strip().lower() in {"false", "0", "no"}
    ):
        markers.append((f"{prefix}.benchmark_success", "false"))

    for field in ("fallback_triggered", "degraded", "fallback_or_degraded"):
        if payload.get(field) is True:
            markers.append((f"{prefix}.{field}", "true"))
    runtime_fields = {"selected_source", "planner_mode", "emergency_stop_count", "last_decision"}
    if runtime_fields.intersection(payload):
        emergency_marker = _emergency_stop_marker(payload)
        if emergency_marker is not None:
            marker_path, marker_value = emergency_marker
            markers.append((f"{prefix}.{marker_path}", marker_value))
    for field in ("algorithm_metadata", "algorithm_metadata_contract"):
        metadata = payload.get(field)
        if not isinstance(metadata, Mapping):
            continue
        _add(f"{field}.status", metadata.get("status"))
        if metadata.get("fallback_or_degraded") is True:
            markers.append((f"{prefix}.{field}.fallback_or_degraded", "true"))
        planner_kinematics = metadata.get("planner_kinematics")
        if isinstance(planner_kinematics, Mapping):
            _add(
                f"{field}.planner_kinematics.execution_mode",
                planner_kinematics.get("execution_mode"),
            )
        adapter_impact = metadata.get("adapter_impact")
        if isinstance(adapter_impact, Mapping):
            _add(f"{field}.adapter_impact.execution_mode", adapter_impact.get("execution_mode"))
        runtime_marker = runtime_fallback_or_degraded_marker(metadata.get("planner_runtime"))
        if runtime_marker is not None:
            marker_path, marker_value = runtime_marker
            markers.append((f"{prefix}.{field}.planner_runtime.{marker_path}", marker_value))
        emergency_marker = _emergency_stop_marker(metadata.get("planner_runtime"))
        if emergency_marker is not None:
            marker_path, marker_value = emergency_marker
            markers.append((f"{prefix}.{field}.planner_runtime.{marker_path}", marker_value))
        foresight_marker = runtime_fallback_or_degraded_marker(metadata.get("foresight_prediction"))
        if foresight_marker is not None:
            marker_path, marker_value = foresight_marker
            markers.append((f"{prefix}.{field}.foresight_prediction.{marker_path}", marker_value))
    availability = payload.get("benchmark_availability")
    if isinstance(availability, Mapping):
        for field in ("status", "readiness_status", "availability_status", "execution_mode"):
            if field in availability:
                _add(f"benchmark_availability.{field}", availability[field])
    return markers


def _read_campaign_summary(campaign_root: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Load the authoritative campaign summary, returning a shaped error.

    Returns:
        The parsed summary and an optional human-readable read error.
    """
    path = campaign_root / "reports" / "campaign_summary.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"campaign summary cannot be read: {exc}"
    if not isinstance(payload, dict):
        return None, "campaign summary must be a JSON object"
    return payload, None


def _read_episode_rows(path: Path) -> tuple[list[dict[str, Any]], str | None]:
    """Read one JSONL arm artifact without accepting malformed rows.

    Returns:
        Parsed row objects and an optional malformed-artifact error.
    """
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, 1):
                if not raw_line.strip():
                    continue
                try:
                    payload = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    return [], f"{path}:{line_number}: invalid JSON: {exc}"
                if not isinstance(payload, dict):
                    return [], f"{path}:{line_number}: episode row must be an object"
                rows.append(payload)
    except OSError as exc:
        return [], f"{path}: cannot read episode artifact: {exc}"
    return rows, None


def _source_commit(row: Mapping[str, Any]) -> str:
    """Resolve the preferred source commit from a row's provenance fields.

    Returns:
        Lowercase source commit token, or an empty string when absent.
    """
    provenance = row.get("result_provenance")
    if isinstance(provenance, Mapping) and provenance.get("repo_commit"):
        return str(provenance["repo_commit"]).strip().lower()
    return str(row.get("git_hash", "")).strip().lower()


def _episode_horizon(row: Mapping[str, Any]) -> tuple[int | None, bool]:
    """Resolve an episode horizon and whether an authoritative value was present.

    Returns:
        Tuple of parsed horizon and presence flag.
    """
    if row.get("horizon") is not None:
        return _strict_int(row["horizon"]), True
    provenance = row.get("result_provenance")
    if isinstance(provenance, Mapping):
        settings = provenance.get("simulator_settings")
        if isinstance(settings, Mapping) and settings.get("horizon") is not None:
            return _strict_int(settings["horizon"]), True
    return None, False


def _scenario_id(scenario: Mapping[str, Any]) -> str:
    """Resolve the stable scenario identifier used by campaign episode identity.

    Returns:
        Stable scenario identifier, or an empty string when absent.
    """
    return str(
        scenario.get("id") or scenario.get("scenario_id") or scenario.get("name") or ""
    ).strip()


def _resolve_expected_matrix_axes(
    manifest: Any,
    campaign_config: Any | None,
) -> tuple[tuple[str, ...], tuple[int, ...], list[str]]:
    """Resolve the exact scenario/seed axes pinned by the manifest and campaign config.

    Returns:
        Scenario IDs, resolved seeds, and any axis-resolution blockers.
    """
    blockers: list[str] = []
    if campaign_config is None:
        config_path = getattr(manifest, "canonical_campaign_config_path", None)
        if config_path is not None:
            try:
                campaign_config = load_campaign_config(config_path)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                blockers.append(f"canonical campaign config cannot be resolved: {exc}")
    if campaign_config is not None:
        try:
            scenarios = _load_campaign_scenarios(campaign_config)
            scenario_ids = tuple(_scenario_id(scenario) for scenario in scenarios)
            seeds = tuple(_resolved_seed_inventory(scenarios))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            blockers.append(f"resolved campaign matrix cannot be loaded: {exc}")
            return (), (), blockers
        if any(not scenario_id for scenario_id in scenario_ids):
            blockers.append("resolved campaign matrix contains an empty scenario identifier")
        if len(set(scenario_ids)) != len(scenario_ids):
            blockers.append("resolved campaign matrix contains duplicate scenario identifiers")
        manifest_seeds = tuple(
            seed
            for raw_seed in getattr(manifest, "resolved_seeds", ())
            if (seed := _strict_int(raw_seed)) is not None
        )
        if manifest_seeds != seeds:
            blockers.append("manifest resolved seeds do not match campaign config")
        return scenario_ids, seeds, blockers

    # Lightweight fixtures may provide the already-resolved axes directly. Real v0.2
    # runs always pass the loaded canonical campaign config from the release runner.
    scenario_ids = tuple(
        str(value).strip() for value in getattr(manifest, "resolved_scenario_ids", ())
    )
    seeds = tuple(
        seed
        for raw_seed in getattr(manifest, "resolved_seeds", ())
        if (seed := _strict_int(raw_seed)) is not None
    )
    if not scenario_ids or not seeds:
        blockers.append("resolved campaign scenario/seed axes are unavailable")
    return scenario_ids, seeds, blockers


def validate_full_benchmark_release_acceptance(  # noqa: C901, PLR0912, PLR0915
    campaign_root: Path,
    *,
    manifest: Any,
    campaign_config: Any | None = None,
) -> dict[str, Any]:
    """Validate the publication-grade S30/H600 campaign contract.

    v0.1 manifests (including the one-scenario runtime smoke) return
    ``not_applicable``.  They remain useful diagnostic execution checks but
    cannot satisfy or accidentally inherit the full-release gate.

    Returns:
        JSON-safe acceptance report.  ``status=valid`` is the only status that
        permits publication of a v0.2 benchmark-data release.
    """
    if getattr(manifest, "schema_version", None) != FULL_RELEASE_SCHEMA_VERSION:
        return {
            "schema_version": FULL_RELEASE_ACCEPTANCE_SCHEMA_VERSION,
            "status": "not_applicable",
            "benchmark_success": False,
            "claim_boundary": "bounded runtime smoke or legacy release manifest; no full-release claim",
            "blockers": [],
        }

    blockers: list[str] = []
    expected_cells = getattr(manifest, "expected_episode_cells", None)
    expected_horizon = getattr(manifest, "expected_horizon_steps", None)
    planner_keys = tuple(str(key) for key in getattr(manifest, "planner_keys", ()))
    kinematics = tuple(str(value) for value in getattr(manifest, "expected_kinematics_matrix", ()))
    if len(planner_keys) != FULL_RELEASE_EXPECTED_PLANNER_ARMS:
        _append_blocker(
            blockers,
            f"manifest must declare exactly {FULL_RELEASE_EXPECTED_PLANNER_ARMS} planner arms",
        )
    if expected_cells != FULL_RELEASE_EXPECTED_EPISODE_CELLS:
        _append_blocker(
            blockers,
            f"manifest expected_episode_cells must be {FULL_RELEASE_EXPECTED_EPISODE_CELLS}",
        )
    if expected_horizon != FULL_RELEASE_EXPECTED_HORIZON_STEPS:
        _append_blocker(
            blockers,
            f"manifest expected_horizon_steps must be {FULL_RELEASE_EXPECTED_HORIZON_STEPS}",
        )
    if kinematics != (FULL_RELEASE_KINEMATICS,):
        _append_blocker(
            blockers,
            f"manifest kinematics must be [{FULL_RELEASE_KINEMATICS!r}]",
        )
    scenario_ids, resolved_seeds, axis_blockers = _resolve_expected_matrix_axes(
        manifest, campaign_config
    )
    for blocker in axis_blockers:
        _append_blocker(blockers, blocker)
    if len(scenario_ids) != 48:
        _append_blocker(blockers, "manifest-resolved campaign must contain exactly 48 scenarios")
    if len(resolved_seeds) != 30:
        _append_blocker(blockers, "manifest-resolved campaign must contain exactly 30 seeds")
    if len(planner_keys) * len(scenario_ids) * len(resolved_seeds) != expected_cells:
        _append_blocker(
            blockers, "manifest-resolved planner/scenario/seed product mismatches 20,160"
        )

    summary, summary_error = _read_campaign_summary(campaign_root.resolve())
    if summary_error:
        _append_blocker(blockers, summary_error)
        summary = {}
    campaign = summary.get("campaign") if isinstance(summary, dict) else None
    campaign = campaign if isinstance(campaign, Mapping) else {}
    for field, expected in (
        ("status", "benchmark_success"),
        ("evidence_status", "valid"),
        ("campaign_execution_status", "completed"),
    ):
        actual = campaign.get(field)
        if field == "status":
            if actual not in {"benchmark_success", "ok"}:
                _append_blocker(blockers, f"campaign.{field} must report benchmark success")
        elif actual != expected:
            _append_blocker(blockers, f"campaign.{field} must be {expected!r}")
    if campaign.get("benchmark_success") is not True:
        _append_blocker(blockers, "campaign.benchmark_success must be true")

    integrity = summary.get("campaign_integrity") if isinstance(summary, dict) else None
    if not isinstance(integrity, Mapping) or integrity.get("status") != "valid":
        _append_blocker(blockers, "campaign_integrity.status must be valid")

    row_status_summary = campaign.get("row_status_summary")
    if not isinstance(row_status_summary, Mapping):
        _append_blocker(blockers, "campaign.row_status_summary is missing")
    else:
        expected_summary = {
            "successful_evidence_rows": FULL_RELEASE_EXPECTED_PLANNER_ARMS,
            "accepted_unavailable_rows": 0,
            "unexpected_failed_rows": 0,
            "fallback_or_degraded_rows": 0,
        }
        for field, expected in expected_summary.items():
            actual = _strict_int(row_status_summary.get(field, -1))
            if actual != expected:
                _append_blocker(
                    blockers,
                    f"campaign.row_status_summary.{field} must be {expected}",
                )

    expected_arms = {(key, FULL_RELEASE_KINEMATICS) for key in planner_keys}
    expected_identities = {
        (planner_key, FULL_RELEASE_KINEMATICS, scenario_id, seed)
        for planner_key in planner_keys
        for scenario_id in scenario_ids
        for seed in resolved_seeds
    }
    runs = summary.get("runs") if isinstance(summary, dict) else None
    planner_rows = summary.get("planner_rows") if isinstance(summary, dict) else None
    if not isinstance(runs, list):
        runs = []
        _append_blocker(blockers, "campaign summary must contain a runs list")
    if not isinstance(planner_rows, list):
        planner_rows = []
        _append_blocker(blockers, "campaign summary must contain a planner_rows list")

    observed_arms: set[tuple[str, str]] = set()
    duplicate_arms: set[tuple[str, str]] = set()
    planner_row_arms: set[tuple[str, str]] = set()
    duplicate_planner_row_arms: set[tuple[str, str]] = set()
    forbidden_status_counts: Counter[str] = Counter()
    source_commits: set[str] = set()
    identities: set[tuple[str, str, str, int]] = set()
    observed_episode_rows = 0
    expected_per_arm = (
        FULL_RELEASE_EXPECTED_EPISODE_CELLS // len(expected_arms) if expected_arms else 0
    )

    for index, entry in enumerate(runs):
        if not isinstance(entry, Mapping):
            _append_blocker(blockers, f"runs[{index}] must be an object")
            continue
        planner = entry.get("planner")
        planner = planner if isinstance(planner, Mapping) else {}
        arm = (str(planner.get("key", "")).strip(), str(planner.get("kinematics", "")).strip())
        if arm in observed_arms:
            duplicate_arms.add(arm)
        observed_arms.add(arm)
        if str(entry.get("status", "")).strip().lower() != "ok":
            _append_blocker(blockers, f"runs[{index}] status is not ok")
        for marker_path, marker in _status_markers(entry, f"runs[{index}]"):
            forbidden_status_counts[marker] += 1
            _append_blocker(blockers, f"forbidden {marker_path}={marker}")
        entry_summary = entry.get("summary")
        if isinstance(entry_summary, Mapping):
            for marker_path, marker in _status_markers(entry_summary, f"runs[{index}].summary"):
                forbidden_status_counts[marker] += 1
                _append_blocker(blockers, f"forbidden {marker_path}={marker}")
            failed_jobs = _strict_int(entry_summary.get("failed_jobs"))
            if failed_jobs is not None and failed_jobs > 0:
                _append_blocker(blockers, f"runs[{index}] reports {failed_jobs} failed jobs")
            failures = entry_summary.get("failures")
            if isinstance(failures, list) and failures:
                _append_blocker(blockers, f"runs[{index}] reports non-empty failures")
        entry_horizon = planner.get("horizon")
        entry_horizon_value = _strict_int(entry_horizon)
        if entry_horizon_value != FULL_RELEASE_EXPECTED_HORIZON_STEPS:
            _append_blocker(blockers, f"runs[{index}] planner horizon is not 600")
        raw_path = str(entry.get("episodes_path", "")).strip()
        if not raw_path:
            _append_blocker(blockers, f"runs[{index}] is missing episodes_path")
            continue
        try:
            episodes_path = _resolve_integrity_artifact_path(campaign_root.resolve(), raw_path)
        except (OSError, ValueError) as exc:
            _append_blocker(blockers, f"runs[{index}] episodes_path rejected: {exc}")
            continue
        rows, error = _read_episode_rows(episodes_path)
        if error:
            _append_blocker(blockers, error)
            continue
        observed_episode_rows += len(rows)
        declared = entry.get("summary")
        declared = declared if isinstance(declared, Mapping) else {}
        declared_count = declared.get("episodes_total", declared.get("written"))
        if len(rows) != expected_per_arm:
            _append_blocker(
                blockers,
                f"runs[{index}] contains {len(rows)} rows; expected {expected_per_arm}",
            )
        if declared_count is None or _strict_int(declared_count) != len(rows):
            _append_blocker(blockers, f"runs[{index}] declared episode count mismatches artifact")
        arm_identities: set[tuple[str, str, str, int]] = set()
        for row_index, row in enumerate(rows):
            for marker_path, marker in _status_markers(row, f"runs[{index}].rows[{row_index}]"):
                forbidden_status_counts[marker] += 1
                _append_blocker(blockers, f"forbidden {marker_path}={marker}")
            scenario_id = str(row.get("scenario_id", "")).strip()
            if not scenario_id:
                _append_blocker(blockers, f"runs[{index}].rows[{row_index}] missing scenario_id")
            seed = _strict_int(row.get("seed"))
            if seed is None:
                _append_blocker(blockers, f"runs[{index}].rows[{row_index}] missing integer seed")
                continue
            identity = (arm[0], arm[1], scenario_id, seed)
            if identity in arm_identities or identity in identities:
                _append_blocker(blockers, f"duplicate episode identity {identity!r}")
            arm_identities.add(identity)
            identities.add(identity)
            commit = _source_commit(row)
            if not _GIT_SHA_RE.fullmatch(commit):
                _append_blocker(
                    blockers,
                    f"runs[{index}].rows[{row_index}] source commit is not a 40-character SHA",
                )
            elif commit:
                source_commits.add(commit)
            horizon, present = _episode_horizon(row)
            if not present or horizon != FULL_RELEASE_EXPECTED_HORIZON_STEPS:
                _append_blocker(blockers, f"runs[{index}].rows[{row_index}] horizon is not 600")

    if duplicate_arms:
        _append_blocker(blockers, f"duplicate planner arms: {sorted(duplicate_arms)!r}")
    if observed_arms != expected_arms:
        _append_blocker(blockers, "successful planner arms do not match the manifest roster")
    if len(runs) != FULL_RELEASE_EXPECTED_PLANNER_ARMS:
        _append_blocker(blockers, "campaign must contain exactly 14 planner arm rows")

    for index, row in enumerate(planner_rows):
        if not isinstance(row, Mapping):
            _append_blocker(blockers, f"planner_rows[{index}] must be an object")
            continue
        arm = (str(row.get("planner_key", "")).strip(), str(row.get("kinematics", "")).strip())
        if arm in planner_row_arms:
            duplicate_planner_row_arms.add(arm)
        planner_row_arms.add(arm)
        for marker_path, marker in _status_markers(row, f"planner_rows[{index}]"):
            forbidden_status_counts[marker] += 1
            _append_blocker(blockers, f"forbidden {marker_path}={marker}")
        if str(row.get("status", "")).strip().lower() != "ok":
            _append_blocker(blockers, f"planner_rows[{index}] status is not ok")
        if _strict_int(row.get("episodes", -1)) != expected_per_arm:
            _append_blocker(
                blockers, f"planner_rows[{index}] episode count is not {expected_per_arm}"
            )
        if arm not in expected_arms:
            _append_blocker(blockers, f"planner_rows[{index}] is outside the manifest roster")
    if len(planner_rows) != FULL_RELEASE_EXPECTED_PLANNER_ARMS:
        _append_blocker(blockers, "campaign must contain exactly 14 planner aggregate rows")
    if duplicate_planner_row_arms:
        _append_blocker(
            blockers,
            f"duplicate planner aggregate rows: {sorted(duplicate_planner_row_arms)!r}",
        )
    if planner_row_arms != expected_arms:
        _append_blocker(blockers, "planner aggregate rows do not match the manifest roster")

    expected_source = str(campaign.get("git_hash", "")).strip().lower()
    if not _GIT_SHA_RE.fullmatch(expected_source):
        _append_blocker(blockers, "campaign.git_hash must be an exact 40-character SHA")
    if expected_source and source_commits and source_commits != {expected_source}:
        _append_blocker(blockers, "episode source commits do not match campaign.git_hash")
    if len(source_commits) != 1:
        _append_blocker(blockers, "episode provenance must name exactly one source commit")
    if observed_episode_rows != FULL_RELEASE_EXPECTED_EPISODE_CELLS:
        _append_blocker(
            blockers,
            f"observed episode rows must be {FULL_RELEASE_EXPECTED_EPISODE_CELLS}",
        )
    if len(identities) != FULL_RELEASE_EXPECTED_EPISODE_CELLS:
        _append_blocker(
            blockers,
            f"unique episode identities must be {FULL_RELEASE_EXPECTED_EPISODE_CELLS}",
        )
    missing_identities = expected_identities - identities
    unexpected_identities = identities - expected_identities
    if missing_identities or unexpected_identities:
        _append_blocker(
            blockers,
            "episode identities do not match the exact manifest-resolved planner/scenario/seed product",
        )

    status = "valid" if not blockers else "invalid"
    return {
        "schema_version": FULL_RELEASE_ACCEPTANCE_SCHEMA_VERSION,
        "status": status,
        "benchmark_success": status == "valid",
        "expected_planner_arms": FULL_RELEASE_EXPECTED_PLANNER_ARMS,
        "successful_planner_arms": sum(
            1
            for entry in runs
            if isinstance(entry, Mapping) and str(entry.get("status", "")).strip().lower() == "ok"
        ),
        "expected_episode_cells": FULL_RELEASE_EXPECTED_EPISODE_CELLS,
        "expected_scenario_count": len(scenario_ids),
        "expected_seed_count": len(resolved_seeds),
        "observed_episode_rows": observed_episode_rows,
        "unique_episode_identities": len(identities),
        "missing_episode_identities": len(missing_identities),
        "unexpected_episode_identities": len(unexpected_identities),
        "source_commits": sorted(source_commits),
        "forbidden_status_counts": dict(sorted(forbidden_status_counts.items())),
        "blockers": blockers,
        "claim_boundary": (
            "Publication-grade benchmark evidence requires all 14 arms, all 20,160 unique "
            "manifest-resolved planner/scenario/seed identities, one source commit, and zero "
            "fallback/degraded/failed/unavailable rows."
        ),
    }


__all__ = [
    "FULL_RELEASE_ACCEPTANCE_SCHEMA_VERSION",
    "FULL_RELEASE_EXPECTED_EPISODE_CELLS",
    "FULL_RELEASE_EXPECTED_HORIZON_STEPS",
    "FULL_RELEASE_EXPECTED_PLANNER_ARMS",
    "validate_full_benchmark_release_acceptance",
]
