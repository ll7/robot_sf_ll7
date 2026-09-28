"""Project retained native traces into the SREV-16/SREV-17 model boundary.

This adapter is intentionally read-only.  It copies explicit state and time
from a digest-bound ``analysis-trace.v1`` and keeps absent video, metrics,
geometry, and media presentation visible as missingness.  A retained trace is
not a historical recording and a derived render is never attached as one.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

TRACE_SCHEMA_VERSION = "analysis-trace.v1"
PANEL_SCHEMA_VERSION = "review-panels.v1"
EDITOR_SCHEMA_VERSION = "review-editor.v1"
TRACE_ARTIFACT_ID = "retained-native-analysis-trace"
TRACE_URI = "retained://analysis-trace.v1"
MAX_TRACE_STEPS = 100_000


def _clone(value: Any) -> Any:
    """Detach projected JSON-compatible data from the retained source.

    Returns:
        Detached value.
    """

    return copy.deepcopy(value)


def _finite(value: Any) -> float | None:
    """Return a finite numeric value, or ``None`` for explicit missingness."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _canonical(value: Any) -> str:
    """Return deterministic JSON for internal source-identity tokens.

    Returns:
        Canonical JSON text.
    """

    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _selected_identity(selected: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten a service row while preserving authoritative top-level fields.

    Returns:
        Flattened selected identity.
    """

    row = selected.get("row")
    result = dict(row) if isinstance(row, Mapping) else {}
    result.update({key: value for key, value in selected.items() if key != "row"})
    return result


def _trace_from(
    selected: Mapping[str, Any], trace: Mapping[str, Any] | None
) -> Mapping[str, Any] | None:
    """Return an inline trace; undeclared paths are never followed."""

    if isinstance(trace, Mapping):
        return trace
    if selected.get("schema_version") == TRACE_SCHEMA_VERSION:
        return selected
    metadata = selected.get("algorithm_metadata")
    if isinstance(metadata, Mapping) and isinstance(metadata.get("analysis_trace"), Mapping):
        return metadata["analysis_trace"]
    for key in ("retained_trace", "analysis_trace", "simulation_trace", "trace"):
        if isinstance(selected.get(key), Mapping):
            return selected[key]
    retained = selected.get("retained_state")
    if isinstance(retained, Mapping):
        for key in ("trace", "simulation_trace"):
            if isinstance(retained.get(key), Mapping):
                return retained[key]
    return None


def _identity_error(
    selected: Mapping[str, Any],
    trace: Mapping[str, Any],
    *,
    trusted_scan_identity_defaults: Mapping[str, Any] | None,
) -> str | None:
    """Reuse BA-05's scanner-aware trace admission, without relaxing row claims.

    BA-01 may add campaign/source/execution wrapper identities to a readable
    row that a standalone native runner trace did not itself declare.  Those
    additions are recorded in a service-owned scan-default map.  A campaign
    source digest is never interchangeable with the trace artifact digest.

    Returns:
        An admission error reason, or ``None`` for a verified binding.
    """

    from robot_sf.analysis_workbench.audit_materialize import (  # noqa: PLC0415
        MaterializationValidationError,
        _verify_native_trace_identity,
    )

    identity = _selected_identity(selected)
    identity.pop("_audit_scan_identity_defaults", None)
    try:
        _verify_native_trace_identity(
            identity,
            trace,
            trusted_scan_identity_defaults=trusted_scan_identity_defaults,
        )
    except (MaterializationValidationError, KeyError, TypeError, ValueError, OverflowError) as exc:
        return str(exc) or "native retained trace identity is unavailable"
    return None


def _point(value: Any) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 2
        and all(_finite(item) is not None for item in value)
    )


def _missing_fields(step: Mapping[str, Any]) -> list[str]:  # noqa: C901
    """List absent/malformed state fields without filling them.

    Returns:
        Dotted field paths that remain unavailable.
    """

    missing: list[str] = []
    robot = step.get("robot")
    if not isinstance(robot, Mapping):
        return ["robot"]
    if not _point(robot.get("position")):
        missing.append("robot.position")
    if not isinstance(robot.get("actor_id"), str) or not robot.get("actor_id"):
        missing.append("robot.actor_id")
    if _finite(robot.get("heading")) is None:
        missing.append("robot.heading")
    if not _point(robot.get("velocity")):
        missing.append("robot.velocity")
    radius = robot.get("radius_m")
    if _finite(radius) is None or float(radius) <= 0.0:
        missing.append("robot.radius_m")
    pedestrians = step.get("pedestrians")
    if not isinstance(pedestrians, list):
        return [*missing, "pedestrians"]
    for index, actor in enumerate(pedestrians):
        prefix = f"pedestrians[{index}]"
        if not isinstance(actor, Mapping):
            missing.append(prefix)
            continue
        if not _point(actor.get("position")):
            missing.append(f"{prefix}.position")
        if not isinstance(actor.get("actor_id"), str) or not actor.get("actor_id"):
            missing.append(f"{prefix}.actor_id")
        if actor.get("velocity") is not None and not _point(actor["velocity"]):
            missing.append(f"{prefix}.velocity")
        radius = actor.get("radius_m")
        if radius is not None and (_finite(radius) is None or float(radius) <= 0.0):
            missing.append(f"{prefix}.radius_m")
    return missing


def _scene_samples(  # noqa: C901
    trace: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Copy explicit trace timestamps/state and report missingness.

    Returns:
        Samples, missingness, and diagnostics.
    """

    missingness: dict[str, Any] = {
        "status": "available",
        "sample_count": 0,
        "missing_count": 0,
        "timestamp_missing_count": 0,
        "fields": [],
    }
    diagnostics: list[dict[str, Any]] = []
    raw_steps = trace.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        missingness.update(status="unavailable", reason="native trace steps are unavailable")
        return [], missingness, [{"reason_code": "scene_trace_unavailable"}]
    if len(raw_steps) > MAX_TRACE_STEPS:
        missingness.update(status="unavailable", reason="native trace steps exceed resource limit")
        return (
            [],
            missingness,
            [{"reason_code": "scene_trace_resource_limit", "limit": MAX_TRACE_STEPS}],
        )

    samples: list[dict[str, Any]] = []
    previous_time: float | None = None
    order_invalid = False
    for source_index, raw in enumerate(raw_steps):
        if not isinstance(raw, Mapping):
            missingness["fields"].append({"source_index": source_index, "fields": ["step"]})
            diagnostics.append(
                {"reason_code": "scene_step_not_mapping", "source_index": source_index}
            )
            continue
        time_s = _finite(raw.get("time_s"))
        if time_s is None:
            missingness["timestamp_missing_count"] += 1
            missingness["fields"].append({"source_index": source_index, "fields": ["time_s"]})
            diagnostics.append(
                {"reason_code": "scene_timestamp_missing", "source_index": source_index}
            )
            continue
        if previous_time is not None and time_s <= previous_time:
            order_invalid = True
            diagnostics.append(
                {"reason_code": "scene_timestamp_not_increasing", "source_index": source_index}
            )
        previous_time = time_s
        fields = _missing_fields(raw)
        if fields:
            missingness["fields"].append({"source_index": source_index, "fields": fields})
        required_missing = any(
            field in {"robot", "robot.position", "pedestrians"} or field.endswith("].position")
            for field in fields
        )
        samples.append(
            {
                "time_s": time_s,
                "value": _clone({key: value for key, value in raw.items() if key != "time_s"}),
                "source_index": source_index,
                "missing": required_missing,
            }
        )
        missingness["sample_count"] += 1
        missingness["missing_count"] += int(required_missing)
    if missingness["fields"]:
        missingness["status"] = "partial"
    if not samples:
        missingness.update(status="unavailable", reason="scene timestamps are unavailable")
    if order_invalid:
        missingness.update(
            status="unavailable", reason="scene timestamps are not strictly increasing"
        )
    return samples, missingness, diagnostics


def _events(
    trace: Mapping[str, Any], cursor: float | None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Copy only events with their own explicit timestamps.

    Returns:
        Event rows and event missingness.
    """

    raw_events = trace.get("events")
    missingness: dict[str, Any] = {"status": "unavailable", "count": 0, "missing_count": 0}
    if not isinstance(raw_events, list) or not raw_events:
        missingness["reason"] = "native trace events are unavailable"
        return [], missingness
    rows: list[dict[str, Any]] = []
    for raw in raw_events:
        if not isinstance(raw, Mapping):
            missingness["missing_count"] += 1
            continue
        row = _clone(dict(raw))
        time_s = _finite(row.get("time_s"))
        if time_s is None:
            row.update(status="unavailable", missingness={"time_s": "missing"})
            missingness["missing_count"] += 1
        else:
            row.update(time_s=time_s, seek_time_s=time_s, selected=time_s == cursor)
            missingness["count"] += 1
        rows.append(row)
    missingness["status"] = "available" if missingness["count"] else "unavailable"
    if missingness["missing_count"] and missingness["count"]:
        missingness["status"] = "partial"
    return rows, missingness


def _source_identity(
    trace: Mapping[str, Any], selected: Mapping[str, Any], digest: str
) -> dict[str, Any]:
    """Build one source identity shared by panels and editor.

    Returns:
        Shared source identity.
    """

    selected_values = _selected_identity(selected)
    fields = {
        "campaign_id": trace.get("campaign_id", selected_values.get("campaign_id")),
        "execution_id": trace.get("execution_id", selected_values.get("execution_id")),
        "episode_id": trace.get("episode_id"),
        "scenario_id": trace.get("scenario_id", selected_values.get("scenario_id")),
        "planner_id": trace.get("planner", selected_values.get("planner_id")),
        "seed": trace.get("seed"),
        "source_commit": trace.get("git_hash", selected_values.get("source_commit")),
        "config_identity": selected_values.get("config_identity"),
        "config_digest": trace.get("config_digest"),
        "coordinate_frame": trace.get("coordinate_frame"),
        "units": _clone(trace.get("units")),
    }
    identity = {key: value for key, value in fields.items() if value is not None}
    identity["sources"] = {
        TRACE_ARTIFACT_ID: {
            "artifact_id": TRACE_ARTIFACT_ID,
            "uri": TRACE_URI,
            "format": TRACE_SCHEMA_VERSION,
            "schema": TRACE_SCHEMA_VERSION,
            "sha256": digest,
            "declared_sha256": digest,
            "computed_sha256": digest,
            "integrity": "verified",
            "availability": "retained",
            "admission": "not_evaluated",
            "source_commit": trace.get("git_hash"),
            "config_identity": selected_values.get("config_identity"),
            "config_digest": trace.get("config_digest"),
            "coordinate_frame": trace.get("coordinate_frame"),
            "units": _clone(trace.get("units")),
        }
    }
    return identity


def _unavailable_stream(reason: str) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "reason": reason,
        "resolution_s": 0.0,
        "samples": [],
        "missingness": {"status": "unavailable", "reason": reason, "sample_count": 0},
    }


def _snapshot(stream: Mapping[str, Any], cursor: float | None) -> dict[str, Any]:
    """Return an exact source sample; never interpolate or hold a stale tail."""

    if cursor is None or stream.get("status") == "unavailable":
        return {"status": "unavailable", "reason": stream.get("reason", "stream_unavailable")}
    for sample in stream.get("samples", ()):
        if isinstance(sample, Mapping) and sample.get("time_s") == cursor:
            if sample.get("missing"):
                return {
                    "status": "unavailable",
                    "reason": "sample_missing",
                    "sample_time_s": cursor,
                }
            return {
                "status": "available",
                "value": _clone(sample.get("value")),
                "sample_time_s": cursor,
                "temporal_error_s": 0.0,
                "source_index": sample.get("source_index"),
            }
    return {"status": "unavailable", "reason": "exact_source_sample_unavailable"}


def _empty_models(reason: str, request_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build stable unavailable envelopes for missing/invalid trace input.

    Returns:
        Unavailable panel and editor models.
    """

    panel = {
        "schema_version": PANEL_SCHEMA_VERSION,
        "component_id": "srev16-review-panels",
        "component_version": "1.0.0",
        "request_id": request_id,
        "status": "unavailable",
        "diagnostic_only": True,
        "time": {
            "authority": "simulation_time",
            "origin_s": None,
            "terminal_s": None,
            "cursor": {"time_s": None},
            "rule": "explicit_trace_time_only; no_interpolation",
        },
        "context": {"schema_version": "review-context.v1", "context_revision": 0},
        "source_identity": {},
        "source_identity_status": {},
        "streams": {"scene": _unavailable_stream(reason), "video": _unavailable_stream(reason)},
        "panels": {
            "scene": {"status": "unavailable", "reason": reason},
            "video": {"status": "unavailable", "reason": reason},
            "metrics": {},
            "events": [],
        },
        "metrics": {},
        "events": [],
        "intervals": [],
        "panel_status": {
            "scene": "unavailable",
            "video": "unavailable",
            "metrics": "unavailable",
            "events": "unavailable",
        },
        "scene_surface": {"status": "unavailable", "reason": reason},
        "goal_geometry": {"status": "unavailable", "reason": reason},
        "missingness": {"trace": {"status": "unavailable", "reason": reason}},
        "provenance": {
            "evidence_status": "diagnostic_only",
            "source_mutation": "none",
            "admission": "not_evaluated",
            "original_recording_presented": False,
        },
        "diagnostics": [{"reason_code": "native_trace_unavailable", "detail": reason}],
    }
    editor = {
        "schema_version": EDITOR_SCHEMA_VERSION,
        "component_id": "srev17-review-editor",
        "component_version": "1.0.0",
        "status": "unavailable",
        "request_id": request_id,
        "context": _clone(panel["context"]),
        "panel_model": _clone(panel),
        "streams": _clone(panel["streams"]),
        "scene_surface": _clone(panel["scene_surface"]),
        "goal_geometry": _clone(panel["goal_geometry"]),
        "metrics": {},
        "events": [],
        "time": _clone(panel["time"]),
        "annotations": [],
        "diagnostics": _clone(panel["diagnostics"]),
        "provenance": _clone(panel["provenance"]),
        "missingness": _clone(panel["missingness"]),
    }
    return panel, editor


def _editor_model(panel: Mapping[str, Any], request_id: str) -> dict[str, Any]:
    """Build the thin SREV-17 envelope from one panel model.

    Returns:
        Editor model mapping.
    """

    identity = _clone(panel["source_identity"])
    identity_token = _sha256(identity)
    source_revision = _canonical(
        {"sources": identity.get("sources", {}), "episode_id": identity.get("episode_id", "")}
    )
    context = _clone(panel["context"])
    return {
        "schema_version": EDITOR_SCHEMA_VERSION,
        "component_id": "srev17-review-editor",
        "component_version": "1.0.0",
        "status": panel["status"],
        "request_id": request_id,
        "context": context,
        "source_identity": identity,
        "source_identity_status": _clone(panel["source_identity_status"]),
        "storyboard_record_id": f"storyboard-{identity_token}",
        "panel_model": _clone(panel),
        "streams": _clone(panel["streams"]),
        "scene_surface": _clone(panel["scene_surface"]),
        "goal_geometry": _clone(panel["goal_geometry"]),
        "metrics": {},
        "events": _clone(panel["events"]),
        "time": _clone(panel["time"]),
        "annotation_hints": {
            "selected_actor": context.get("actor_id"),
            "active_goal": None,
            "final_goal": None,
            "completion_boundaries": [],
        },
        "annotation_modes": ["one_click", "quick", "full"],
        "classifications": [
            "normal",
            "interesting_valid",
            "planner_defect",
            "benchmark_defect",
            "scenario_defect",
            "instrumentation_defect",
            "unclear",
        ],
        "triage_labels": {
            "normal": "normal",
            "suspicious": "interesting_valid",
            "bug": "planner_defect",
            "unsure": "unclear",
        },
        "annotations": [],
        "overlay_state": {
            "numbered": True,
            "highlights": True,
            "arrows": False,
            "rings": False,
            "distances": False,
        },
        "storyboard": {
            "schema_version": "review-storyboard-edit.v1",
            "source_identity": identity_token,
            "source_revision": source_revision,
            "intervals": [],
            "order": [],
            "captions": {},
        },
        "controls": {
            "typing_suppresses_shortcuts": True,
            "source_time_authority": "simulation_time",
        },
        "persistence": {
            "adapter": "ba03-audit-store",
            "service_boundary": "ba05-service-compatible",
            "canonical": "AuditStore/audit.ndjson",
            "browser_local_storage": False,
            "operation_id_required": True,
            "expected_revision_required": True,
            "acknowledge_after_durable_commit": True,
        },
        "coverage": {"status": "diagnostic_only", "reviewed": 0},
        "diagnostics": _clone(panel["diagnostics"]),
        "missingness": _clone(panel["missingness"]),
        "provenance": _clone(panel["provenance"]),
    }


def project_retained_native_trace(
    selected: Mapping[str, Any] | None = None,
    *,
    trace: Mapping[str, Any] | None = None,
    request_id: str | None = None,
    materialization: Mapping[str, Any] | Any | None = None,
    trusted_scan_identity_defaults: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Project one retained native trace into SREV-16 and SREV-17.

    ``selected`` normally contains an inline ``retained_trace`` and the
    selected episode identity.  ``trace=`` is useful for offline callers.  A
    path declaration is not read here; source-root and digest admission stay
    in the materialization owner. Scanner defaults may only be supplied by a
    trusted service caller, never inferred from fields inside ``selected``.

    Returns:
        Projection status, panel/editor models, diagnostics, and provenance.
    """

    selected_value = selected if isinstance(selected, Mapping) else {}
    request = str(request_id or selected_value.get("episode_id") or "audit-trace-projection")
    retained = _trace_from(selected_value, trace)
    if not isinstance(retained, Mapping):
        panel, editor = _empty_models("retained native trace payload is unavailable", request)
        return {
            "status": "unavailable",
            "reason": "retained_native_trace_unavailable",
            "diagnostics": panel["diagnostics"],
            "panel_model": panel,
            "editor_model": editor,
            "provenance": panel["provenance"],
        }
    reason = _identity_error(
        selected_value,
        retained,
        trusted_scan_identity_defaults=trusted_scan_identity_defaults,
    )
    if reason:
        panel, editor = _empty_models(reason, request)
        return {
            "status": "unavailable",
            "reason": "retained_native_trace_unavailable",
            "diagnostics": panel["diagnostics"],
            "panel_model": panel,
            "editor_model": editor,
            "provenance": panel["provenance"],
        }

    digest = str(retained["artifact_sha256"])
    samples, scene_missingness, scene_diagnostics = _scene_samples(retained)
    cursor = samples[0]["time_s"] if samples else None
    origin = cursor
    terminal = samples[-1]["time_s"] if samples else None
    events, event_missingness = _events(retained, cursor)
    identity = _source_identity(retained, selected_value, digest)
    source = identity["sources"][TRACE_ARTIFACT_ID]
    context_values = {
        "campaign_id": identity.get("campaign_id"),
        "execution_id": identity.get("execution_id"),
        "episode_id": identity.get("episode_id"),
        "scenario_id": identity.get("scenario_id"),
        "planner_id": identity.get("planner_id"),
        "seed": identity.get("seed"),
        "actor_id": "robot",
    }
    context = {key: value for key, value in context_values.items() if value is not None}
    context.update(
        {
            "context_revision": 0,
            "cursor": {
                "time_s": cursor,
                "context_revision": 0,
                "source": "retained_native_trace",
                "interval_id": None,
            },
        }
    )
    scene_status = "unavailable" if not samples else scene_missingness["status"]
    scene = {
        "status": scene_status,
        "resolution_s": 0.0,
        "range": {"start_s": origin, "end_s": terminal} if samples else None,
        "samples": samples,
        "source_ids": [TRACE_ARTIFACT_ID],
        "source_identity": source,
        "missingness": scene_missingness,
    }
    if scene_status != "available":
        scene["reason"] = scene_missingness.get("reason", "retained scene is partial")
    video_reason = "retained native analysis trace does not contain video"
    metrics_reason = "retained native analysis trace does not contain metric series"
    scene_surface_reason = "retained native analysis trace does not contain scene surface geometry"
    goal_reason = "retained native analysis trace does not contain goal geometry"
    status = (
        "unavailable"
        if scene_status == "unavailable"
        else (
            "partial"
            if scene_status != "available" or event_missingness["status"] == "partial"
            else "complete"
        )
    )
    scene_surface = {
        "schema_version": "threejs-viewer.v1",
        "status": "unavailable",
        "reason": scene_surface_reason,
        "map": None,
        "trajectory": [],
    }
    if isinstance(retained.get("map"), Mapping):
        scene_surface.update(status="available", map=_clone(retained["map"]))
        scene_surface.pop("reason")
    goal = {
        "status": "unavailable",
        "reason": goal_reason,
        "goal_point": None,
        "completion_boundary": None,
    }
    if retained.get("goal_point") is not None or retained.get("goal") is not None:
        goal.update(
            status="available", goal_point=_clone(retained.get("goal_point", retained.get("goal")))
        )
        goal.pop("reason")
    provenance: dict[str, Any] = {
        "evidence_status": "diagnostic_only",
        "evidence_boundary": "analysis_workbench_only",
        "source_mutation": "none",
        "admission": "not_evaluated",
        "source_kind": "retained_native_analysis_trace",
        "source_schema": TRACE_SCHEMA_VERSION,
        "source_digest": digest,
        "retained_trace_digest": digest,
        "simulation_advanced": False,
        "original_recording_presented": False,
        "derived_media_presented_as_original": False,
        "media_presentation": {
            "video": "unavailable",
            "original_recording": "not_attached",
            "derived_render": "not_attached",
        },
        "claim_boundary": "retained native state projection; not historical media or benchmark evidence",
        "source_identity": _clone(identity),
    }
    if materialization is not None:
        value = (
            materialization.to_dict() if hasattr(materialization, "to_dict") else materialization
        )
        if isinstance(value, Mapping):
            provenance["materialization"] = {
                "status": value.get("status"),
                "kind": value.get("materialization_kind"),
                "fidelity": value.get("fidelity"),
                "artifacts": _clone(value.get("artifacts", [])),
                "presented_as": "derived_only",
            }
    video = _unavailable_stream(video_reason)
    panel = {
        "schema_version": PANEL_SCHEMA_VERSION,
        "component_id": "srev16-review-panels",
        "component_version": "1.0.0",
        "request_id": request,
        "status": status,
        "evidence_boundary": "analysis_workbench_only",
        "diagnostic_only": True,
        "time": {
            "authority": "simulation_time",
            "origin_s": origin,
            "terminal_s": terminal,
            "interval": {"start_s": origin, "end_s": terminal, "interval_id": None},
            "cursor": context["cursor"],
            "resolution_s": {"scene": 0.0, "video": 0.0},
            "rule": "explicit_trace_time_only; no_interpolation",
            "video_alignment": "explicit_source_time_mapping_only",
        },
        "context": context,
        "source_identity": identity,
        "source_identity_status": {
            TRACE_ARTIFACT_ID: {"status": "verified", "sha256": digest, "reason": ""}
        },
        "goal_geometry": goal,
        "goal": goal,
        "scene_surface": scene_surface,
        "surfaces": {
            "scene": {
                "renderer": "offline-canvas",
                "mount": "mountSceneViewer",
                "contract": "threejs-viewer.v1",
                "status": scene_surface["status"],
            },
            "video": {
                "renderer": "HTMLVideoElement",
                "mount": "review-video",
                "mapping": "explicit_source_time_only",
                "status": "unavailable",
            },
        },
        "streams": {"scene": scene, "video": video},
        "panels": {
            "scene": {**_snapshot(scene, cursor), "goal_geometry": goal},
            "video": _snapshot(video, cursor),
            "metrics": {},
            "events": events,
        },
        "metrics": {},
        "events": events,
        "intervals": [],
        "panel_status": {
            "scene": scene_status,
            "video": "unavailable",
            "metrics": "unavailable",
            "events": event_missingness["status"],
        },
        "controls": {
            "keyboard": {
                "play_pause": "Space",
                "previous_step": "ArrowLeft",
                "next_step": "ArrowRight",
                "speed_down": "-",
                "speed_up": "+",
            },
            "speeds": [0.25, 0.5, 1.0, 2.0, 4.0],
            "default_speed": 1.0,
            "read_only": True,
            "feedback_loop_policy": "one_shared_cursor_dispatch",
        },
        "missingness": {
            "trace": {"status": "available"},
            "scene": scene_missingness,
            "video": {"status": "unavailable", "reason": video_reason},
            "metrics": {"status": "unavailable", "reason": metrics_reason, "derived": False},
            "events": event_missingness,
            "scene_surface": {
                "status": scene_surface["status"],
                "reason": scene_surface.get("reason", ""),
            },
            "goal_geometry": {"status": goal["status"], "reason": goal.get("reason", "")},
        },
        "provenance": provenance,
        "diagnostics": [
            {
                "reason_code": "retained_native_trace_projected",
                "detail": "scene/time copied from retained analysis-trace.v1",
            },
            *scene_diagnostics,
            {"reason_code": "video_unavailable", "detail": video_reason},
            {"reason_code": "metrics_unavailable", "detail": metrics_reason},
            {"reason_code": "scene_surface_unavailable", "detail": scene_surface_reason},
            {"reason_code": "goal_geometry_unavailable", "detail": goal_reason},
        ],
    }
    if event_missingness["status"] != "available":
        panel["diagnostics"].append(
            {
                "reason_code": "events_unavailable",
                "detail": event_missingness.get("reason", "event timestamps are missing"),
            }
        )
    editor = _editor_model(panel, request)
    return {
        "status": status,
        "reason": "retained_native_trace_projected"
        if status != "unavailable"
        else "retained_native_trace_unavailable",
        "diagnostics": _clone(panel["diagnostics"]),
        "panel_model": panel,
        "editor_model": editor,
        "provenance": _clone(provenance),
    }


def project_native_trace(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Alias for integrations naming the source as a native trace.

    Returns:
        Retained-native projection result.
    """

    return project_retained_native_trace(*args, **kwargs)


def project_native_analysis_trace(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Alias for integrations naming the full analysis-trace schema.

    Returns:
        Retained-native projection result.
    """

    return project_retained_native_trace(*args, **kwargs)


__all__ = [
    "EDITOR_SCHEMA_VERSION",
    "PANEL_SCHEMA_VERSION",
    "TRACE_ARTIFACT_ID",
    "TRACE_SCHEMA_VERSION",
    "project_native_analysis_trace",
    "project_native_trace",
    "project_retained_native_trace",
]
