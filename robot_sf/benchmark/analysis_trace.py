"""Provenance-first telemetry profiles and trace adapters.

The benchmark runner historically exposed two independent trace booleans.  This
module adds a small, planner-agnostic profile on top of those booleans without
changing the action loop.  The profile is deliberately data-only: it describes
what was recorded and provides fail-closed coverage checks for downstream case
discovery.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ANALYSIS_TRACE_SCHEMA_VERSION = "analysis-telemetry-profile.v1"
ANALYSIS_TRACE_RECORD_SCHEMA_VERSION = "analysis-trace.v1"
TRACE_COVERAGE_VERSION = "analysis-trace-coverage.v1"
_SHA256_RE = r"[0-9a-f]{64}"


@dataclass(frozen=True, slots=True)
class TelemetryProfile:
    """Normalized telemetry selection for a benchmark run."""

    analysis_trace: str = "off"
    planner_debug_trace: str = "none"

    def __post_init__(self) -> None:
        """Validate the two stable profile selectors."""

        if self.analysis_trace not in {"off", "all"}:
            raise ValueError("telemetry.analysis_trace must be 'off' or 'all'")
        if self.planner_debug_trace not in {"none", "all"}:
            raise ValueError("telemetry.planner_debug_trace must be 'none' or 'all'")

    @property
    def analysis_enabled(self) -> bool:
        """Whether every episode should carry the analysis-ready trace."""

        return self.analysis_trace == "all"

    def to_mapping(self) -> dict[str, str]:
        """Return the stable configuration mapping."""

        return {
            "schema_version": ANALYSIS_TRACE_SCHEMA_VERSION,
            "analysis_trace": self.analysis_trace,
            "planner_debug_trace": self.planner_debug_trace,
        }


def normalize_telemetry_profile(value: Any) -> TelemetryProfile:
    """Normalize a profile mapping while preserving the legacy default.

    ``None`` and an absent mapping intentionally mean ``off``.  Existing trace
    booleans therefore remain readable and keep their historical behavior.
    Returns:
        Normalized immutable telemetry profile.
    """

    if value is None:
        return TelemetryProfile()
    if isinstance(value, TelemetryProfile):
        return value
    if not isinstance(value, dict):
        raise TypeError("telemetry profile must be a mapping")
    return TelemetryProfile(
        analysis_trace=str(value.get("analysis_trace", "off")),
        planner_debug_trace=str(value.get("planner_debug_trace", "none")),
    )


def telemetry_from_scenario(scenario: dict[str, Any]) -> TelemetryProfile:
    """Resolve telemetry from the explicit field or scenario metadata.

    Returns:
        Normalized profile, defaulting to legacy recording behavior.
    """

    value = scenario.get("telemetry")
    if value is None:
        metadata = scenario.get("metadata")
        if isinstance(metadata, dict):
            value = metadata.get("telemetry")
    return normalize_telemetry_profile(value)


def canonical_json(value: Any) -> str:
    """Serialize a value deterministically for artifact identity.

    Returns:
        Canonical JSON text.
    """

    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def sha256_json(value: Any) -> str:
    """Return a SHA-256 digest of canonical JSON.

    Returns:
        Lower-case SHA-256 digest.
    """

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def map_digest(scenario: dict[str, Any]) -> str | None:
    """Return the content digest of the scenario's referenced map when resolvable."""

    explicit = scenario.get("map_digest") or scenario.get("map_sha256")
    raw_path = scenario.get("map_file") or scenario.get("map")
    if isinstance(explicit, str) and explicit:
        if not re.fullmatch(_SHA256_RE, explicit):
            return None
        # An explicit digest is only authoritative when there is no local map
        # to verify.  When a path is available, reject a stale or forged digest
        # instead of allowing the trace to be compared against another map.
        if not isinstance(raw_path, str) or not raw_path:
            return explicit
    else:
        explicit = None
    if not isinstance(raw_path, str) or not raw_path:
        return None
    path = Path(raw_path)
    candidates = [path]
    if not path.is_absolute():
        candidates.extend([Path.cwd() / path, Path(__file__).resolve().parents[2] / path])
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            actual = hashlib.sha256(candidate.read_bytes()).hexdigest()
            if explicit is not None and actual != explicit:
                return None
            return actual
        except OSError:
            continue
    return None


def _infer_initial_pedestrian_ids(steps: list[dict[str, Any]], count: int) -> list[Any] | None:
    """Use explicit simulator actor labels from the first frame when available.

    Returns:
        Stable source labels, or ``None`` when the first frame cannot provide a
        complete unique registry.
    """

    if count == 0 or not steps or not isinstance(steps[0], Mapping):
        return [] if count == 0 else None
    pedestrians = steps[0].get("pedestrians")
    if not isinstance(pedestrians, list) or len(pedestrians) != count:
        return None
    ids: list[Any] = []
    for actor in pedestrians:
        if not isinstance(actor, Mapping):
            return None
        # ``id`` is the legacy positional slot emitted by the map runner.  It
        # is not a stable simulator identity, so it cannot establish an
        # analysis-ready actor registry on its own.
        actor_id = actor.get("actor_id")
        if actor_id is None:
            actor_id = actor.get("pedestrian_id")
        if actor_id is None:
            return None
        ids.append(actor_id)
    if len({str(value) for value in ids}) != len(ids):
        return None
    return ids


def _canonical_actor_id(value: Any) -> str:
    """Return the canonical trace identifier for one pedestrian label."""

    text = str(value)
    return text if text.startswith("pedestrian-") else f"pedestrian-{text}"


def build_analysis_trace(  # noqa: C901, PLR0912, PLR0913, PLR0915
    *,
    steps: list[dict[str, Any]],
    initial_robot_position: Any,
    initial_robot_heading: float,
    initial_pedestrians: Any,
    dt: float,
    horizon: int,
    robot_radius_m: float,
    pedestrian_radius_m: float,
    scenario: dict[str, Any],
    planner: str,
    planner_commit: str | None,
    config_hash: str,
    git_hash: str | None,
    termination_reason: str,
    safety_events: list[dict[str, Any]],
    initial_robot_velocity: Any = None,
    initial_pedestrian_velocities: Any = None,
    initial_pedestrian_ids: list[Any] | None = None,
    initial_pedestrian_id_source: str | None = None,
    coordinate_frame: str = "world",
    units: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build an analysis-ready trace envelope from the legacy step trace.

    The legacy entries represent post-action states at ``t > 0``.  The new
    envelope prepends an explicit ``t=0`` state and annotates all actors with
    stable IDs and radii.  No planner call or simulator state is touched.
    Returns:
        Analysis trace envelope with a deterministic artifact digest.
    """

    initial_robot = np.asarray(initial_robot_position, dtype=float).reshape(-1)[:2]
    initial_peds = np.asarray(initial_pedestrians, dtype=float).reshape(-1, 2)
    robot_velocity = _vector2_or_none(initial_robot_velocity)
    if initial_pedestrian_velocities is None:
        ped_velocities: list[list[float] | None] = [None] * len(initial_peds)
    else:
        ped_velocity_array = np.asarray(initial_pedestrian_velocities, dtype=float).reshape(-1, 2)
        ped_velocities = [
            _vector2_or_none(value) for value in ped_velocity_array[: len(initial_peds)]
        ]
        ped_velocities.extend([None] * (len(initial_peds) - len(ped_velocities)))
    inferred_ids = _infer_initial_pedestrian_ids(steps, len(initial_peds))
    resolved_initial_ids = (
        list(initial_pedestrian_ids) if initial_pedestrian_ids is not None else inferred_ids
    )
    if resolved_initial_ids is not None and (
        len(resolved_initial_ids) != len(initial_peds)
        or len({str(value) for value in resolved_initial_ids}) != len(resolved_initial_ids)
        or any(value is None or isinstance(value, bool) for value in resolved_initial_ids)
    ):
        resolved_initial_ids = None
    pedestrian_ids = list(
        resolved_initial_ids if resolved_initial_ids is not None else range(len(initial_peds))
    )
    pedestrian_ids.extend(range(len(pedestrian_ids), len(initial_peds)))
    initial_pedestrian_radii = [float(pedestrian_radius_m)] * len(initial_peds)
    if steps and isinstance(steps[0], Mapping) and isinstance(steps[0].get("pedestrians"), list):
        for index, actor in enumerate(steps[0]["pedestrians"][: len(initial_peds)]):
            if isinstance(actor, Mapping) and _finite_positive(actor.get("radius_m")):
                initial_pedestrian_radii[index] = float(actor["radius_m"])
    resolved_units = units or {
        "position": "m",
        "velocity": "m/s",
        "heading": "rad",
        "time": "s",
    }
    initial_step = {
        "step": 0,
        "time_s": 0.0,
        "robot": {
            "actor_id": "robot",
            "position": [float(initial_robot[0]), float(initial_robot[1])],
            "heading": float(initial_robot_heading),
            "velocity": robot_velocity,
            "radius_m": float(robot_radius_m),
        },
        "pedestrians": [
            {
                "actor_id": (
                    _canonical_actor_id(pedestrian_ids[index])
                    if resolved_initial_ids is not None
                    else None
                ),
                "id": pedestrian_ids[index],
                "position": [float(pos[0]), float(pos[1])],
                "velocity": ped_velocities[index],
                "radius_m": initial_pedestrian_radii[index],
            }
            for index, pos in enumerate(initial_peds)
        ],
        "controls": {"requested": None, "applied": None},
        "events": [],
    }
    normalized_steps: list[dict[str, Any]] = [initial_step]
    for normalized_index, raw in enumerate(steps, start=1):
        item = json.loads(canonical_json(raw))
        # The legacy trace starts its first post-action frame at step zero.
        # The analysis envelope reserves step zero for the explicit t=0 frame.
        item["step"] = normalized_index
        robot = item.get("robot") if isinstance(item.get("robot"), dict) else {}
        robot["actor_id"] = "robot"
        if not _finite_positive(robot.get("radius_m")):
            robot["radius_m"] = float(robot_radius_m)
        item["robot"] = robot
        pedestrians = item.get("pedestrians")
        if isinstance(pedestrians, list):
            for index, actor in enumerate(pedestrians):
                if not isinstance(actor, dict):
                    continue
                actor_id = actor.get("actor_id")
                if actor_id is None:
                    actor_id = actor.get("pedestrian_id")
                if isinstance(actor_id, str) and actor_id:
                    actor["actor_id"] = _canonical_actor_id(actor_id)
                    actor.setdefault("id", actor_id)
                else:
                    actor.setdefault("id", index)
                    if resolved_initial_ids is not None and index < len(resolved_initial_ids):
                        actor["id"] = resolved_initial_ids[index]
                        actor["actor_id"] = _canonical_actor_id(resolved_initial_ids[index])
                    else:
                        # Keep positional legacy identity visible for review,
                        # but leave the canonical actor_id unavailable so the
                        # coverage gate cannot promote it as stable.
                        actor["actor_id"] = None
                if not _finite_positive(actor.get("radius_m")):
                    actor["radius_m"] = float(pedestrian_radius_m)
        planner_payload = item.get("planner")
        amv = planner_payload.get("amv") if isinstance(planner_payload, dict) else None
        if isinstance(amv, dict):
            item["controls"] = {
                "requested": {
                    "linear_m_s": amv.get("requested_linear_m_s"),
                    "turn_rate_rad_s": amv.get("requested_angular_rad_s"),
                },
                "applied": {
                    "linear_m_s": amv.get("applied_linear_m_s"),
                    "turn_rate_rad_s": amv.get("applied_angular_rad_s"),
                },
            }
        else:
            selected_action = (
                planner_payload.get("selected_action")
                if isinstance(planner_payload, dict)
                else None
            )
            applied_action = (
                planner_payload.get("applied_environment_action")
                if isinstance(planner_payload, Mapping)
                else None
            )
            if isinstance(selected_action, Mapping):
                requested_linear, requested_turn = _action_control_values(selected_action)
                applied_linear, applied_turn = _action_control_values(applied_action)
                if _finite_number(requested_linear) or _finite_number(requested_turn):
                    item["controls"] = {
                        "requested": {
                            "linear_m_s": requested_linear,
                            "turn_rate_rad_s": requested_turn,
                        },
                        "applied": {
                            "linear_m_s": applied_linear,
                            "turn_rate_rad_s": applied_turn,
                        },
                        "source": (
                            "planner.selected_action+environment_action"
                            if isinstance(applied_action, Mapping)
                            else "planner.selected_action"
                        ),
                    }
                elif not isinstance(item.get("controls"), Mapping):
                    item["controls"] = {"requested": None, "applied": None}
            else:
                item.setdefault("controls", {"requested": None, "applied": None})
        item["events"] = _normalize_events(
            item.get("events"),
            index_offset=normalized_index,
            actor_ids=resolved_initial_ids,
        )
        normalized_steps.append(item)

    scenario_identity = {
        key: value for key, value in scenario.items() if key not in {"seed", "repeats"}
    }
    payload: dict[str, Any] = {
        "schema_version": ANALYSIS_TRACE_RECORD_SCHEMA_VERSION,
        "dt": float(dt),
        "horizon": int(horizon),
        "coordinate_frame": coordinate_frame,
        "units": resolved_units,
        "actor_geometry": {
            "robot_radius_m": float(robot_radius_m),
            "pedestrian_radius_m": float(pedestrian_radius_m),
        },
        "actor_id_source": (
            initial_pedestrian_id_source
            if initial_pedestrian_id_source is not None
            else (
                "explicit"
                if resolved_initial_ids is not None and initial_pedestrian_ids is not None
                else ("simulator" if resolved_initial_ids is not None else "positional_index")
            )
        ),
        "planner": str(planner),
        "planner_commit": planner_commit,
        "scenario_id": scenario.get("id") or scenario.get("name") or scenario.get("scenario_id"),
        "map_file": scenario.get("map_file") or scenario.get("map"),
        "scenario_digest": sha256_json(scenario_identity),
        # Keep the legacy short ``config_hash`` for v1 readers, but bind pair
        # comparisons to a full digest that also includes the effective trace
        # timing.  This prevents the short legacy fingerprint from becoming an
        # exact provenance claim.
        "config_digest": sha256_json(
            {**scenario_identity, "dt": float(dt), "horizon": int(horizon)}
        ),
        "map_digest": map_digest(scenario),
        "config_hash": str(config_hash),
        "git_hash": git_hash,
        "termination_reason": str(termination_reason),
        "events": _normalize_events(safety_events, actor_ids=resolved_initial_ids),
        "steps": normalized_steps,
    }
    _deduplicate_trace_event_ids(payload)
    payload["artifact_sha256"] = sha256_json(payload)
    return payload


def trace_coverage(record: dict[str, Any]) -> dict[str, Any]:  # noqa: C901, PLR0912, PLR0915
    """Return explicit, fail-closed coverage for a benchmark record.

    Returns:
        Coverage mapping with a complete/unavailable status and reasons.
    """

    metadata = record.get("algorithm_metadata")
    trace = metadata.get("analysis_trace") if isinstance(metadata, dict) else None
    if not isinstance(trace, dict):
        legacy = metadata.get("simulation_step_trace") if isinstance(metadata, dict) else None
        reason = (
            "legacy_trace_without_analysis_profile" if isinstance(legacy, dict) else "trace_absent"
        )
        return {
            "schema_version": TRACE_COVERAGE_VERSION,
            "status": "unavailable",
            "reason": reason,
            "steps": 0,
            "has_initial_state": False,
            "stable_actor_ids": False,
            "radii": False,
            "controls": False,
            "finite_states": False,
            "monotonic_time": False,
            "timing": False,
            "provenance": False,
            "map_digest": False,
        }
    steps = trace.get("steps")
    has_steps = isinstance(steps, list) and bool(steps)
    has_initial = bool(
        has_steps
        and isinstance(steps[0], dict)
        and _finite_number(steps[0].get("time_s"))
        and float(steps[0]["time_s"]) == 0.0
    )
    actor_ids = bool(has_steps)
    radii = bool(has_steps)
    controls = bool(has_steps)
    control_observed = False
    finite_states = bool(has_steps)
    monotonic_time = bool(has_steps)
    prior_time: float | None = None
    actor_radii: dict[str, float] = {}
    for index, step in enumerate(steps if has_steps else []):
        if not isinstance(step, dict) or not isinstance(step.get("robot"), dict):
            actor_ids = radii = finite_states = False
            break
        current_time = step.get("time_s")
        if not _finite_number(current_time) or (
            prior_time is not None and current_time <= prior_time
        ):
            monotonic_time = False
        prior_time = float(current_time) if _finite_number(current_time) else prior_time
        robot = step["robot"]
        if robot.get("actor_id") != "robot":
            actor_ids = False
        if not _finite_positive(robot.get("radius_m")):
            radii = False
        if not _finite_vector(robot.get("position"), 2) or not _finite_number(robot.get("heading")):
            finite_states = False
        if not _finite_vector(robot.get("velocity"), 2):
            finite_states = False
        seen_ids = {"robot"}
        pedestrians = step.get("pedestrians", [])
        if not isinstance(pedestrians, list):
            actor_ids = False
            pedestrians = []
        for actor in pedestrians:
            if not isinstance(actor, dict):
                actor_ids = radii = finite_states = False
                continue
            actor_id = actor.get("actor_id")
            if not isinstance(actor_id, str) or not actor_id or actor_id in seen_ids:
                actor_ids = False
            else:
                seen_ids.add(actor_id)
            if not _finite_positive(actor.get("radius_m")):
                radii = False
            elif isinstance(actor.get("actor_id"), str):
                actor_id = str(actor["actor_id"])
                radius = float(actor["radius_m"])
                previous_radius = actor_radii.setdefault(actor_id, radius)
                if not math.isclose(radius, previous_radius, rel_tol=0.0, abs_tol=1.0e-9):
                    radii = False
            if not _finite_vector(actor.get("position"), 2) or not _finite_vector(
                actor.get("velocity"), 2
            ):
                finite_states = False
        controls_payload = step.get("controls")
        if not isinstance(controls_payload, dict):
            controls = False
        elif index > 0 and not _control_payload_complete(controls_payload):
            controls = False
        elif _control_payload_complete(controls_payload):
            control_observed = True
    controls = controls and control_observed
    actor_ids = actor_ids and trace.get("actor_id_source") in {
        "explicit",
        "simulator",
        "simulator_slot",
    }
    coordinate_frame = trace.get("coordinate_frame") == "world"
    expected_units = {"position": "m", "velocity": "m/s", "heading": "rad", "time": "s"}
    units = trace.get("units") == expected_units
    timing = (
        _finite_number(trace.get("dt"))
        and float(trace["dt"]) > 0
        and _positive_int(trace.get("horizon"))
    )
    provenance = all(
        isinstance(trace.get(key), str)
        and bool(str(trace.get(key)).strip())
        and str(trace.get(key)).strip().lower() not in {"unknown", "none", "null", "unavailable"}
        for key in (
            "config_hash",
            "config_digest",
            "scenario_digest",
            "map_digest",
            "git_hash",
            "planner_commit",
            "planner",
        )
    )
    provenance = provenance and all(
        isinstance(trace.get(key), str) and bool(re.fullmatch(_SHA256_RE, str(trace.get(key))))
        for key in ("scenario_digest", "map_digest", "config_digest")
    )
    provenance = provenance and all(
        isinstance(trace.get(key), str)
        and bool(re.fullmatch(r"[0-9a-f]{7,64}", str(trace.get(key)).lower()))
        for key in ("git_hash", "planner_commit")
    )
    record_scenario = record.get("scenario_id")
    record_planner = record.get("algo") or record.get("planner")
    trace_identity = (
        record_scenario is None or str(record_scenario) == str(trace.get("scenario_id"))
    ) and (record_planner is None or str(record_planner) == str(trace.get("planner")))
    artifact_hash = (
        isinstance(trace.get("artifact_sha256"), str)
        and bool(re.fullmatch(_SHA256_RE, str(trace.get("artifact_sha256"))))
        and str(trace.get("artifact_sha256")) == trace_artifact_sha256(trace)
    )
    complete = (
        trace.get("schema_version") == ANALYSIS_TRACE_RECORD_SCHEMA_VERSION
        and has_initial
        and actor_ids
        and radii
        and controls
        and control_observed
        and finite_states
        and monotonic_time
        and coordinate_frame
        and units
        and timing
        and provenance
        and trace_identity
        and artifact_hash
    )
    reasons = []
    for name, valid in (
        ("initial_state", has_initial),
        ("stable_actor_ids", actor_ids),
        ("radii", radii),
        ("controls", controls),
        ("finite_states", finite_states),
        ("monotonic_time", monotonic_time),
        ("coordinate_frame", coordinate_frame),
        ("units", units),
        ("timing", timing),
        ("provenance", provenance),
        ("trace_identity", trace_identity),
        ("artifact_hash", artifact_hash),
    ):
        if not valid:
            reasons.append(name)
    return {
        "schema_version": TRACE_COVERAGE_VERSION,
        "status": "complete" if complete else "unavailable",
        "reason": None if complete else "analysis_trace_fields_incomplete",
        "reasons": reasons,
        "steps": len(steps) if has_steps else 0,
        "has_initial_state": has_initial,
        "stable_actor_ids": actor_ids,
        "radii": radii,
        "controls": controls,
        "finite_states": finite_states,
        "monotonic_time": monotonic_time,
        "timing": timing,
        "coordinate_frame": coordinate_frame,
        "units": units,
        "provenance": provenance,
        "trace_identity": trace_identity,
        "artifact_hash": artifact_hash,
        "map_digest": trace.get("map_digest") not in (None, ""),
    }


def _finite_number(value: Any) -> bool:
    """Return whether a value is a finite real number."""

    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _finite_positive(value: Any) -> bool:
    """Return whether a value is finite and strictly positive."""

    return _finite_number(value) and float(value) > 0.0


def _finite_vector(value: Any, size: int) -> bool:
    """Return whether a sequence contains exactly ``size`` finite numbers."""

    return (
        isinstance(value, (list, tuple))
        and len(value) == size
        and all(_finite_number(item) for item in value)
    )


def _vector2_or_none(value: Any) -> list[float] | None:
    """Normalize an optional two-dimensional velocity without fabricating zeros.

    Returns:
        A finite two-dimensional vector, or ``None`` when unavailable.
    """

    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if len(array) < 2 or not np.isfinite(array[:2]).all():
        return None
    return [float(array[0]), float(array[1])]


def _positive_int(value: Any) -> bool:
    """Return whether a value is a positive integer-like scalar."""

    return isinstance(value, (int, np.integer)) and not isinstance(value, bool) and int(value) > 0


def _control_payload_complete(payload: Mapping[str, Any]) -> bool:
    """Return whether both requested and applied control dimensions are finite."""

    for section in ("requested", "applied"):
        values = payload.get(section)
        if not isinstance(values, Mapping) or not all(
            _finite_number(values.get(key)) for key in ("linear_m_s", "turn_rate_rad_s")
        ):
            return False
    return True


def _action_control_values(action: Any) -> tuple[float | None, float | None]:
    """Extract canonical linear and turn controls from a runtime action payload.

    Returns:
        The linear and turn controls, or ``None`` for unavailable dimensions.
    """

    if not isinstance(action, Mapping):
        return None, None
    linear = action.get("linear_m_s")
    if linear is None:
        linear = action.get("linear_velocity")
    turn = action.get("turn_rate_rad_s")
    if turn is None:
        turn = action.get("angular_velocity")
    if linear is None and turn is None:
        vx = action.get("vx")
        vy = action.get("vy")
        if _finite_number(vx) and _finite_number(vy):
            linear = math.hypot(float(vx), float(vy))
    return linear, turn


def _normalize_events(
    events: Any,
    *,
    index_offset: int = 0,
    actor_ids: list[Any] | None = None,
) -> list[dict[str, Any]]:
    """Normalize runtime and legacy safety events to one canonical shape.

    Returns:
        Canonical event mappings with raw details retained for auditability.
    """

    if not isinstance(events, list):
        return []
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, event in enumerate(events):
        item = _normalize_event(
            event,
            event_id=f"event-{index_offset:04d}-{index:04d}",
            actor_ids=actor_ids,
        )
        if item["event_id"] in seen_ids:
            item["event_id"] = f"duplicate-{index_offset:04d}-{index:04d}"
            item["status"] = "unavailable"
            item["reason"] = "duplicate_event_id"
        seen_ids.add(item["event_id"])
        normalized.append(item)
    return normalized


def _normalize_event(event: Any, *, event_id: str, actor_ids: list[Any] | None) -> dict[str, Any]:
    """Normalize one event while retaining the unmodified source details.

    Returns:
        A canonical event mapping with typed unavailable fields when the input
        is not an event object.
    """

    if not isinstance(event, Mapping):
        return {
            "event_id": f"unavailable-{event_id}",
            "event_type": "unavailable",
            "time_s": None,
            "status": "unavailable",
            "reason": "event_not_mapping",
            "details": event,
        }
    raw = dict(event)
    event_type = raw.get("event_type") or raw.get("type")
    if event_type is None and (
        raw.get("collision_time") is not None
        or raw.get("collision_partner_id") is not None
        or raw.get("collision") is True
    ):
        event_type = "collision"
    event_type = str(event_type or "unknown")
    time_value = raw.get("time_s")
    if time_value is None:
        time_value = raw.get("collision_time")
    partner = raw.get("actor_id")
    if partner is None:
        partner = raw.get("partner_id")
    if partner is None:
        partner = raw.get("collision_partner_id")
    if actor_ids is not None and partner is not None:
        try:
            partner_index = int(partner)
        except (TypeError, ValueError):
            partner_index = -1
        if 0 <= partner_index < len(actor_ids):
            partner = actor_ids[partner_index]
    if (
        event_type == "collision"
        and partner is not None
        and not str(partner).startswith("pedestrian-")
    ):
        partner = f"pedestrian-{partner}"
    return {
        "event_id": str(raw.get("event_id") or event_id),
        "event_type": event_type,
        "time_s": float(time_value) if _finite_number(time_value) else None,
        "status": str(raw.get("status") or "observed"),
        "reason": raw.get("reason"),
        "actor_id": raw.get("actor_id"),
        "partner_id": partner,
        "details": raw,
    }


def _deduplicate_trace_event_ids(trace: dict[str, Any]) -> None:
    """Namespace duplicate event IDs across top-level and step event ledgers."""

    seen: set[str] = set()
    ledgers = [trace.get("events", [])]
    ledgers.extend(
        step.get("events", []) for step in trace.get("steps", []) if isinstance(step, Mapping)
    )
    for ledger_index, ledger in enumerate(ledgers):
        if not isinstance(ledger, list):
            continue
        for event_index, event in enumerate(ledger):
            if not isinstance(event, dict):
                continue
            event_id = str(event.get("event_id") or "")
            if event_id in seen:
                event["event_id"] = f"duplicate-ledger-{ledger_index:04d}-{event_index:04d}"
                event["status"] = "unavailable"
                event["reason"] = "duplicate_event_id"
            seen.add(str(event.get("event_id")))


def trace_artifact_sha256(trace: Mapping[str, Any]) -> str:
    """Compute the canonical trace digest excluding its stored digest field.

    Returns:
        Lower-case SHA-256 digest for the canonical trace payload.
    """

    payload = dict(trace)
    payload.pop("artifact_sha256", None)
    return sha256_json(payload)


__all__ = [
    "ANALYSIS_TRACE_RECORD_SCHEMA_VERSION",
    "ANALYSIS_TRACE_SCHEMA_VERSION",
    "TelemetryProfile",
    "build_analysis_trace",
    "map_digest",
    "normalize_telemetry_profile",
    "sha256_json",
    "telemetry_from_scenario",
    "trace_artifact_sha256",
    "trace_coverage",
]
