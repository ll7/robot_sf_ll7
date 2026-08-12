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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ANALYSIS_TRACE_SCHEMA_VERSION = "analysis-telemetry-profile.v1"
ANALYSIS_TRACE_RECORD_SCHEMA_VERSION = "analysis-trace.v1"
TRACE_COVERAGE_VERSION = "analysis-trace-coverage.v1"


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
    if isinstance(explicit, str) and explicit:
        return explicit
    raw_path = scenario.get("map_file") or scenario.get("map")
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
            return hashlib.sha256(candidate.read_bytes()).hexdigest()
        except OSError:
            continue
    return None


def build_analysis_trace(  # noqa: PLR0913
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
    initial_step = {
        "step": 0,
        "time_s": 0.0,
        "robot": {
            "actor_id": "robot",
            "position": [float(initial_robot[0]), float(initial_robot[1])],
            "heading": float(initial_robot_heading),
            "velocity": [0.0, 0.0],
            "radius_m": float(robot_radius_m),
        },
        "pedestrians": [
            {
                "actor_id": f"pedestrian-{index}",
                "id": int(index),
                "position": [float(pos[0]), float(pos[1])],
                "velocity": [0.0, 0.0],
                "radius_m": float(pedestrian_radius_m),
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
        robot["radius_m"] = float(robot_radius_m)
        item["robot"] = robot
        pedestrians = item.get("pedestrians")
        if isinstance(pedestrians, list):
            for index, actor in enumerate(pedestrians):
                if not isinstance(actor, dict):
                    continue
                actor.setdefault("id", index)
                actor["actor_id"] = f"pedestrian-{actor['id']}"
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
            item.setdefault("controls", {"requested": None, "applied": None})
        item.setdefault("events", [])
        normalized_steps.append(item)

    payload: dict[str, Any] = {
        "schema_version": ANALYSIS_TRACE_RECORD_SCHEMA_VERSION,
        "dt": float(dt),
        "horizon": int(horizon),
        "coordinate_frame": "world",
        "units": {"position": "m", "velocity": "m/s", "heading": "rad", "time": "s"},
        "actor_geometry": {
            "robot_radius_m": float(robot_radius_m),
            "pedestrian_radius_m": float(pedestrian_radius_m),
        },
        "planner": str(planner),
        "planner_commit": planner_commit,
        "scenario_id": scenario.get("id") or scenario.get("name") or scenario.get("scenario_id"),
        "scenario_digest": sha256_json(scenario),
        "map_digest": map_digest(scenario),
        "config_hash": str(config_hash),
        "git_hash": git_hash,
        "termination_reason": str(termination_reason),
        "events": json.loads(canonical_json(safety_events)),
        "steps": normalized_steps,
    }
    payload["artifact_sha256"] = sha256_json(payload)
    return payload


def trace_coverage(record: dict[str, Any]) -> dict[str, Any]:  # noqa: C901
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
            "provenance": False,
            "map_digest": False,
        }
    steps = trace.get("steps")
    has_steps = isinstance(steps, list) and bool(steps)
    has_initial = bool(has_steps and isinstance(steps[0], dict) and steps[0].get("time_s") == 0.0)
    actor_ids = bool(has_steps)
    radii = bool(has_steps)
    controls = bool(has_steps)
    for step in steps if has_steps else []:
        if not isinstance(step, dict) or not isinstance(step.get("robot"), dict):
            actor_ids = False
            radii = False
            break
        robot = step["robot"]
        if not isinstance(step.get("controls"), dict):
            controls = False
        if robot.get("actor_id") != "robot" or not isinstance(robot.get("radius_m"), (int, float)):
            actor_ids = False
            radii = False
            break
        seen_ids = {"robot"}
        for actor in (
            step.get("pedestrians", []) if isinstance(step.get("pedestrians"), list) else []
        ):
            if not isinstance(actor, dict):
                actor_ids = False
                radii = False
                break
            actor_id = actor.get("actor_id")
            if not isinstance(actor_id, str) or not actor_id or actor_id in seen_ids:
                actor_ids = False
            else:
                seen_ids.add(actor_id)
            if not isinstance(actor.get("radius_m"), (int, float)):
                radii = False
        if not actor_ids or not radii:
            break
    coordinate_frame = trace.get("coordinate_frame") == "world"
    units = isinstance(trace.get("units"), dict) and all(
        trace["units"].get(key) not in (None, "")
        for key in ("position", "velocity", "heading", "time")
    )
    provenance = all(trace.get(key) not in (None, "") for key in ("config_hash", "scenario_digest"))
    complete = (
        has_initial
        and actor_ids
        and radii
        and controls
        and coordinate_frame
        and units
        and provenance
    )
    return {
        "schema_version": TRACE_COVERAGE_VERSION,
        "status": "complete" if complete else "unavailable",
        "reason": None if complete else "analysis_trace_fields_incomplete",
        "steps": len(steps) if has_steps else 0,
        "has_initial_state": has_initial,
        "stable_actor_ids": actor_ids,
        "radii": radii,
        "controls": controls,
        "coordinate_frame": coordinate_frame,
        "units": units,
        "provenance": provenance,
        "map_digest": trace.get("map_digest") not in (None, ""),
    }


__all__ = [
    "ANALYSIS_TRACE_RECORD_SCHEMA_VERSION",
    "ANALYSIS_TRACE_SCHEMA_VERSION",
    "TelemetryProfile",
    "build_analysis_trace",
    "map_digest",
    "normalize_telemetry_profile",
    "sha256_json",
    "telemetry_from_scenario",
    "trace_coverage",
]
