"""Produce the diagnostic ``report_crosswalk.v1`` sidecar for camera-ready campaigns.

The producer reads the canonical campaign episode JSONL files and delegates all
diagnostic interpretation to :mod:`robot_sf.benchmark.report_crosswalk`.  It
retains source-artifact hashes and episode identity fields so a sidecar cannot
be mistaken for an unproven aggregate of unrelated rows.

The sidecar is diagnostic-only.  It does not create metrics, deserialize an
execution monitor from arbitrary JSON, or establish causality, safety,
intervention effectiveness, planner ranking, generalization, or benchmark
validity.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from loguru import logger

from robot_sf.benchmark.camera_ready._util import _sha256_file
from robot_sf.benchmark.report_crosswalk import (
    REPORT_CROSSWALK_SCHEMA_VERSION,
    REPORT_CROSSWALK_SOURCE,
    EpisodeDiagnosticSummary,
    build_campaign_diagnostic_summary,
    build_episode_diagnostic_summary,
    validate_campaign_diagnostic_summary,
    validate_episode_diagnostic_summary,
)

CROSSWALK_SIDECAR_FILENAME = "report_crosswalk.v1.json"

_CLAIM_BOUNDARY = (
    "This sidecar is a diagnostic-only reporting artifact. It must not be treated "
    "as benchmark evidence, causality, safety, planner-ranking, or "
    "intervention-effectiveness proof."
)


def _path_label(path: Path, *, repo_root: Path) -> str:
    """Return a stable source path, relative to the supplied repository when possible."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _string_or_none(value: Any) -> str | None:
    """Return a non-empty string or ``None`` without inventing metadata."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _finite_number_or_none(value: Any) -> float | None:
    """Return a finite JSON number or ``None``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _extract_core_metrics(record: dict[str, Any]) -> dict[str, Any]:
    """Extract the canonical success, collision, and comfort fields.

    Success and collision come from the episode outcome contract.  The
    canonical comfort field is ``metrics.comfort_exposure``; ``metrics.comfort``
    is retained only as a compatibility input for older rows that already use
    that name.

    Returns:
        Mapping with nullable success, collision, and comfort values.
    """
    outcome = record.get("outcome")
    metrics = record.get("metrics")

    success: bool | None = None
    collision: bool | None = None
    if isinstance(outcome, Mapping):
        route_complete = outcome.get("route_complete")
        collision_event = outcome.get("collision_event")
        if isinstance(route_complete, bool):
            success = route_complete
        if isinstance(collision_event, bool):
            collision = collision_event

    comfort: float | None = None
    if isinstance(metrics, Mapping):
        raw_comfort = metrics.get("comfort_exposure")
        if raw_comfort is None:
            raw_comfort = metrics.get("comfort")
        comfort = _finite_number_or_none(raw_comfort)

    return {"success": success, "collision": collision, "comfort": comfort}


def _diagnosis_input(record: Mapping[str, Any]) -> tuple[Any, bool]:
    """Return the optional serialized diagnosis payload and whether it was present."""
    for key in ("diagnosis_payload", "failure_diagnosis"):
        if key in record:
            return record[key], True
    return None, False


def _resolve_episode_path(raw_path: str, *, repo_root: Path) -> tuple[Path, bool]:
    """Resolve an episode path and report whether the original path was a symlink.

    Returns:
        Resolved path and a flag indicating that the supplied path was a symlink.
    """
    candidate = Path(raw_path)
    was_symlink = candidate.is_symlink()
    resolved = (candidate if candidate.is_absolute() else repo_root / candidate).resolve()
    return resolved, was_symlink


def _episode_provenance(
    *,
    campaign_id: str,
    episode: Mapping[str, Any],
    source_artifact: Mapping[str, Any],
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Build exact episode identity and source-artifact provenance.

    Returns:
        JSON-safe episode provenance mapping.
    """
    episode_id = episode["episode_id"]
    scenario_id = episode.get("scenario_id")
    seed = episode.get("seed")
    planner_key = episode["planner_key"]
    record_provenance = record.get("result_provenance")
    record_provenance = dict(record_provenance) if isinstance(record_provenance, Mapping) else None
    provenance_fields = record_provenance or {}

    provenance_complete = all(
        (
            episode_id != "unknown",
            isinstance(scenario_id, str) and bool(scenario_id.strip()),
            isinstance(seed, int) and not isinstance(seed, bool),
            planner_key != "unknown",
            source_artifact.get("status") == "available",
            isinstance(source_artifact.get("episodes_sha256"), str),
            isinstance(record_provenance, dict),
            provenance_fields.get("scenario_id") == scenario_id,
            provenance_fields.get("seed") == seed,
            _string_or_none(provenance_fields.get("config_hash")) is not None,
            _string_or_none(provenance_fields.get("repo_commit")) is not None,
        )
    )
    return {
        "status": "complete" if provenance_complete else "incomplete",
        "campaign_id": campaign_id,
        "run_index": source_artifact.get("run_index"),
        "run_status": source_artifact.get("run_status"),
        "episodes_path": source_artifact.get("episodes_path"),
        "summary_path": source_artifact.get("summary_path"),
        "episodes_sha256": source_artifact.get("episodes_sha256"),
        "episode_id": episode_id,
        "scenario_id": scenario_id,
        "seed": seed,
        "planner_key": planner_key,
        "planner_algo": episode.get("planner_algo"),
        "kinematics": episode.get("kinematics"),
        "record_provenance": record_provenance,
    }


def _collect_episodes_for_crosswalk(  # noqa: C901
    run_entries: list[dict[str, Any]],
    *,
    campaign_id: str,
    repo_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Read campaign episode JSONL and return episodes, source receipts, and input errors.

    Returns:
        Tuple of episode inputs, source-artifact receipts, and invalid input records.
    """
    episodes: list[dict[str, Any]] = []
    source_artifacts: list[dict[str, Any]] = []
    invalid_source_records: list[dict[str, Any]] = []

    for run_index, entry in enumerate(run_entries):
        planner = entry.get("planner") if isinstance(entry.get("planner"), Mapping) else {}
        planner_key = _string_or_none(planner.get("key")) or "unknown"
        planner_algo = _string_or_none(planner.get("algo"))
        kinematics = _string_or_none(planner.get("kinematics"))
        raw_path = entry.get("episodes_path")
        source_artifact: dict[str, Any] = {
            "run_index": run_index,
            "run_status": _string_or_none(entry.get("status")) or "unknown",
            "episodes_path": None,
            "summary_path": (
                _string_or_none(entry.get("summary_path"))
                if isinstance(entry.get("summary_path"), str)
                else None
            ),
            "episodes_sha256": None,
            "status": "unavailable",
            "reason": "episodes_path_not_provided",
            "record_count": 0,
            "invalid_record_count": 0,
            "planner": {
                "key": planner_key,
                "algo": planner_algo,
                "kinematics": kinematics,
            },
        }
        source_artifacts.append(source_artifact)
        if not isinstance(raw_path, str) or not raw_path.strip():
            continue

        path, was_symlink = _resolve_episode_path(raw_path, repo_root=repo_root)
        source_artifact["episodes_path"] = _path_label(path, repo_root=repo_root)
        if was_symlink:
            source_artifact.update(status="invalid", reason="episodes_path_is_symlink")
            continue
        if not path.is_file():
            source_artifact.update(status="unavailable", reason="episodes_file_not_found")
            logger.warning("Crosswalk sidecar: episodes file not found, skipping: {}", raw_path)
            continue

        try:
            source_artifact["episodes_sha256"] = _sha256_file(path)
        except RuntimeError as exc:
            source_artifact.update(status="invalid", reason="episodes_file_hash_failed")
            invalid_source_records.append(
                {
                    "run_index": run_index,
                    "episodes_path": source_artifact["episodes_path"],
                    "line": None,
                    "reason": str(exc),
                }
            )
            continue

        source_artifact.update(status="available", reason=None)
        try:
            handle = path.open("r", encoding="utf-8")
        except OSError as exc:
            source_artifact.update(status="invalid", reason="episodes_file_open_failed")
            invalid_source_records.append(
                {
                    "run_index": run_index,
                    "episodes_path": source_artifact["episodes_path"],
                    "line": None,
                    "reason": str(exc),
                }
            )
            continue

        with handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except (TypeError, ValueError) as exc:
                    source_artifact["invalid_record_count"] += 1
                    invalid_source_records.append(
                        {
                            "run_index": run_index,
                            "episodes_path": source_artifact["episodes_path"],
                            "line": line_number,
                            "reason": f"invalid_json:{exc}",
                        }
                    )
                    continue
                if not isinstance(payload, dict):
                    source_artifact["invalid_record_count"] += 1
                    invalid_source_records.append(
                        {
                            "run_index": run_index,
                            "episodes_path": source_artifact["episodes_path"],
                            "line": line_number,
                            "reason": "episode_record_not_mapping",
                        }
                    )
                    continue

                episode_id = _string_or_none(payload.get("episode_id")) or "unknown"
                diagnosis_payload, diagnosis_present = _diagnosis_input(payload)
                has_execution_deviation = any(
                    key in payload for key in ("execution_deviation", "execution_deviation_result")
                )
                episode = {
                    "episode_id": episode_id,
                    "scenario_id": payload.get("scenario_id"),
                    "seed": payload.get("seed"),
                    "planner_key": planner_key,
                    "planner_algo": planner_algo,
                    "kinematics": kinematics,
                    "diagnosis_payload": diagnosis_payload,
                    "diagnosis_present": diagnosis_present,
                    "execution_deviation_present": has_execution_deviation,
                    "core_metrics": _extract_core_metrics(payload),
                }
                episode["provenance"] = _episode_provenance(
                    campaign_id=campaign_id,
                    episode=episode,
                    source_artifact=source_artifact,
                    record=payload,
                )
                episodes.append(episode)
                source_artifact["record_count"] += 1

    return episodes, source_artifacts, invalid_source_records


def _episode_summary_payload(
    episode: Mapping[str, Any],
) -> tuple[EpisodeDiagnosticSummary, dict[str, Any]]:
    """Build and validate one episode summary, preserving unsupported inputs as invalid.

    Returns:
        Tuple of the typed crosswalk summary and its validated JSON payload.
    """
    summary = build_episode_diagnostic_summary(
        episode_id=str(episode["episode_id"]),
        planner_id=str(episode["planner_key"]),
        diagnosis_payload=episode.get("diagnosis_payload"),
        execution_deviation_result=None,
        success=episode["core_metrics"].get("success"),
        collision=episode["core_metrics"].get("collision"),
        comfort=episode["core_metrics"].get("comfort"),
    )
    summary_payload = summary.to_dict()
    if episode.get("execution_deviation_present"):
        # JSON rows do not contain a validated ExecutionDeviationResult object.  Keep the
        # fact that an input was supplied visible, but never turn arbitrary JSON into a
        # monitor result or an available execution metric.
        summary_payload["execution_deviation"].update(
            {
                "available": False,
                "validity_state": "invalid",
                "provenance": "incomplete",
                "validity_reason": (
                    "serialized_execution_deviation_requires_validated_result_object"
                ),
                "claim_boundary": None,
            }
        )
    validate_episode_diagnostic_summary(summary_payload)
    summary_payload["provenance"] = episode["provenance"]
    summary_payload["input_quality"] = {
        "diagnosis_payload_present": bool(episode.get("diagnosis_present")),
        "execution_deviation_input_present": bool(episode.get("execution_deviation_present")),
    }
    return summary, summary_payload


def _overall_provenance_status(
    source_artifacts: list[Mapping[str, Any]],
    episodes: list[Mapping[str, Any]],
    invalid_source_records: list[Mapping[str, Any]],
) -> str:
    """Aggregate source receipts into a conservative provenance status.

    Returns:
        ``complete``, ``incomplete``, or ``unknown``.
    """
    if not source_artifacts:
        return "unknown"
    if invalid_source_records or any(
        item.get("status") != "available" for item in source_artifacts
    ):
        return "incomplete"
    if any(item.get("provenance", {}).get("status") != "complete" for item in episodes):
        return "incomplete"
    return "complete"


def build_crosswalk_sidecar_payload(
    *,
    campaign_id: str,
    run_entries: list[dict[str, Any]],
    repo_root: Path,
) -> dict[str, Any]:
    """Build a validated ``report_crosswalk.v1`` campaign sidecar payload.

    Returns:
        JSON-safe sidecar payload with source receipts and diagnostic summaries.
    """
    episodes, source_artifacts, invalid_source_records = _collect_episodes_for_crosswalk(
        run_entries,
        campaign_id=campaign_id,
        repo_root=repo_root,
    )

    episode_summaries: list[EpisodeDiagnosticSummary] = []
    episode_payloads: list[dict[str, Any]] = []
    for episode in episodes:
        summary, summary_payload = _episode_summary_payload(episode)
        episode_summaries.append(summary)
        episode_payloads.append(summary_payload)

    campaign_summary = build_campaign_diagnostic_summary(
        campaign_id=campaign_id,
        episode_summaries=episode_summaries,
    )
    campaign_payload = campaign_summary.to_dict()
    validate_campaign_diagnostic_summary(campaign_payload)

    diagnosis_state_counts: dict[str, int] = {}
    execution_state_counts: dict[str, int] = {}
    for episode_payload in episode_payloads:
        diagnosis_state = episode_payload["diagnosis"]["validity_state"]
        execution_state = episode_payload["execution_deviation"]["validity_state"]
        diagnosis_state_counts[diagnosis_state] = diagnosis_state_counts.get(diagnosis_state, 0) + 1
        execution_state_counts[execution_state] = execution_state_counts.get(execution_state, 0) + 1

    return {
        "schema_version": REPORT_CROSSWALK_SCHEMA_VERSION,
        "report_source": REPORT_CROSSWALK_SOURCE,
        "campaign_id": campaign_id,
        "episode_count": campaign_summary.episode_count,
        "episodes": episode_payloads,
        "campaign": campaign_payload,
        "provenance": {
            "status": _overall_provenance_status(
                source_artifacts,
                episodes,
                invalid_source_records,
            ),
            "campaign_id": campaign_id,
            "source_artifacts": source_artifacts,
            "invalid_source_records": invalid_source_records,
        },
        "input_quality": {
            "status": (
                "invalid"
                if invalid_source_records
                else (
                    "incomplete"
                    if any(item.get("status") != "available" for item in source_artifacts)
                    else "valid"
                )
            ),
            "source_record_count": sum(
                int(item["record_count"]) + int(item["invalid_record_count"])
                for item in source_artifacts
            ),
            "episode_count": len(episodes),
            "invalid_source_record_count": len(invalid_source_records),
            "diagnosis_validity_state_counts": diagnosis_state_counts,
            "execution_deviation_validity_state_counts": execution_state_counts,
        },
        "caveats": list(campaign_summary.caveats),
        "diagnostic_only": True,
        "claim_boundary": _CLAIM_BOUNDARY,
    }


def write_crosswalk_sidecar(
    reports_dir: Path,
    *,
    campaign_id: str,
    run_entries: list[dict[str, Any]],
    repo_root: Path,
) -> Path:
    """Build and write the report crosswalk sidecar, including empty campaigns.

    Returns:
        Path to the written sidecar.
    """
    payload = build_crosswalk_sidecar_payload(
        campaign_id=campaign_id,
        run_entries=run_entries,
        repo_root=repo_root,
    )
    sidecar_path = reports_dir / CROSSWALK_SIDECAR_FILENAME
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "Crosswalk sidecar written: {} ({} episodes)",
        sidecar_path,
        payload["episode_count"],
    )
    return sidecar_path


__all__ = [
    "CROSSWALK_SIDECAR_FILENAME",
    "build_crosswalk_sidecar_payload",
    "write_crosswalk_sidecar",
]
