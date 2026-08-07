"""Typed loader for ``simulation_trace_export.v1`` analysis-workbench traces."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from robot_sf.common.json_pointer import json_pointer
from robot_sf.errors import RobotSfError

SIMULATION_TRACE_EXPORT_SCHEMA_VERSION = "simulation_trace_export.v1"
SIMULATION_TRACE_EXPORT_SCHEMA_FILE = (
    Path(__file__).with_name("schemas") / "simulation_trace_export.v1.json"
)


@dataclass(frozen=True, slots=True)
class SimulationTraceSource:
    """Source metadata for an exported simulation trace."""

    scenario_id: str
    seed: int
    planner_id: str
    episode_id: str
    generated_by: str


@dataclass(frozen=True, slots=True)
class SimulationTraceFrame:
    """One playback frame in an analysis-workbench trace."""

    step: int
    time_s: float
    robot: dict[str, Any]
    pedestrians: list[dict[str, Any]]
    planner: dict[str, Any]

    def __post_init__(self) -> None:
        """Reject containers that only become equivalent after JSON coercion."""

        if type(self.pedestrians) is not list:
            raise SimulationTraceExportValidationError(
                ["/pedestrians: expected exact JSON array list"]
            )


@dataclass(frozen=True, slots=True)
class SimulationTraceExport:
    """Typed ``simulation_trace_export.v1`` payload."""

    schema_version: str
    trace_id: str
    source: SimulationTraceSource
    evidence_boundary: str
    coordinate_frame: str
    units: dict[str, str]
    frames: list[SimulationTraceFrame]

    def __post_init__(self) -> None:
        """Reject containers that only become equivalent after JSON coercion."""

        if type(self.frames) is not list:
            raise SimulationTraceExportValidationError(["/frames: expected exact JSON array list"])

    def to_dict(self) -> dict[str, Any]:
        """Convert the export to JSON-safe primitives.

        Returns:
            Dictionary representation suitable for JSON Schema validation.
        """

        return asdict(self)


class SimulationTraceExportValidationError(RobotSfError, ValueError):
    """Raised when a simulation trace export fails validation."""

    def __init__(self, errors: list[str], *, source: str | Path | None = None):
        """Build an actionable validation error."""

        self.errors = tuple(errors)
        self.source = str(source) if source is not None else None
        prefix = f"{self.source}: " if self.source else ""
        super().__init__(prefix + "; ".join(errors))


class SimulationTraceNormalizationError(RobotSfError, ValueError):
    """Raised when a strict provenance metadata projection is invalid."""


_STRICT_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_STRICT_FALLBACK_IDENTITIES = {
    "",
    "0",
    "none",
    "null",
    "unknown",
    "unknown_planner",
    "unknown_scenario",
}


def _strict_trace_identity_errors(payload: Mapping[str, Any]) -> list[str]:  # noqa: C901
    """Return strict-identity violations for a provenance-bound trace."""

    errors: list[str] = []
    source = payload.get("source")
    if not isinstance(source, Mapping):
        return ["/source: source identity must be an object"]

    for field in ("scenario_id", "planner_id", "episode_id"):
        value = source.get(field)
        if not isinstance(value, str) or value.strip().lower() in _STRICT_FALLBACK_IDENTITIES:
            errors.append(f"/source/{field}: fallback or unavailable identity")
    if type(source.get("seed")) is not int:
        errors.append("/source/seed: strict source seed must be an integer")

    frames = payload.get("frames")
    if not isinstance(frames, list):
        return errors
    for frame_index, frame in enumerate(frames):
        if not isinstance(frame, Mapping):
            continue
        pedestrians = frame.get("pedestrians")
        if not isinstance(pedestrians, list):
            continue
        for pedestrian_index, pedestrian in enumerate(pedestrians):
            if not isinstance(pedestrian, Mapping):
                continue
            actor_id = pedestrian.get("id")
            if not isinstance(actor_id, str) or not actor_id.strip():
                errors.append(f"/frames/{frame_index}/pedestrians/{pedestrian_index}/id: missing")
                continue
            if actor_id == str(pedestrian_index) or re.fullmatch(r"ped[-_]\d+", actor_id):
                errors.append(
                    f"/frames/{frame_index}/pedestrians/{pedestrian_index}/id: generated actor id"
                )
    return errors


def _strict_projection_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a trace with strict metadata removed for semantic comparison."""

    projection = copy.deepcopy(dict(payload))
    frames = projection.get("frames")
    if isinstance(frames, list):
        for frame in frames:
            if not isinstance(frame, dict):
                continue
            planner = frame.get("planner")
            if isinstance(planner, dict):
                planner.pop("run_config", None)
                planner.pop("outcome", None)
    return projection


def _strict_canonical_sha256(payload: Any) -> str:
    """Hash one strict metadata projection with its newline policy.

    Returns:
        Lowercase SHA-256 digest of the canonical projection.
    """

    serialized = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(serialized + b"\n").hexdigest()


def apply_strict_metadata_projection(  # noqa: C901, PLR0912
    payload: Mapping[str, Any],
    *,
    run_config: Mapping[str, Any],
    terminal_outcome: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Add verified run metadata without changing canonical trace state.

    This public analysis-workbench primitive is deliberately opt-in. Legacy
    callers retain their byte-compatible output, while provenance-bound callers
    receive a fail-closed identity check and a receipt proving that only planner
    metadata was added.

    Returns:
        The enriched trace and its strict metadata-delta receipt.
    """

    if not isinstance(payload, Mapping):
        raise SimulationTraceNormalizationError("strict trace payload must be an object")
    identity_errors = _strict_trace_identity_errors(payload)
    if identity_errors:
        raise SimulationTraceNormalizationError("; ".join(identity_errors))

    required_config = {"map_id", "horizon", "time_step_s", "config_digest"}
    if set(run_config) != required_config:
        missing = sorted(required_config - set(run_config))
        extra = sorted(set(run_config) - required_config)
        details: list[str] = []
        if missing:
            details.append(f"missing {missing}")
        if extra:
            details.append(f"unexpected {extra}")
        raise SimulationTraceNormalizationError(
            f"strict run_config fields invalid: {', '.join(details)}"
        )
    map_id = run_config["map_id"]
    horizon = run_config["horizon"]
    time_step_s = run_config["time_step_s"]
    config_digest = run_config["config_digest"]
    if not isinstance(map_id, str) or not map_id.strip():
        raise SimulationTraceNormalizationError("strict run_config map_id must be non-empty")
    if type(horizon) is not int or horizon <= 0:
        raise SimulationTraceNormalizationError(
            "strict run_config horizon must be positive integer"
        )
    if not isinstance(time_step_s, (int, float)) or isinstance(time_step_s, bool):
        raise SimulationTraceNormalizationError("strict run_config time_step_s must be numeric")
    if not math.isfinite(float(time_step_s)) or float(time_step_s) <= 0:
        raise SimulationTraceNormalizationError("strict run_config time_step_s must be positive")
    if not isinstance(config_digest, str) or not _STRICT_SHA256_RE.fullmatch(config_digest):
        raise SimulationTraceNormalizationError(
            "strict run_config config_digest must be lowercase SHA-256"
        )

    outcome: dict[str, bool] | None = None
    if terminal_outcome is not None:
        outcome_fields = {"collision_event", "timeout_event", "route_complete"}
        if set(terminal_outcome) != outcome_fields:
            raise SimulationTraceNormalizationError(
                "strict terminal_outcome must contain exactly the typed event fields"
            )
        if any(type(terminal_outcome[field]) is not bool for field in outcome_fields):
            raise SimulationTraceNormalizationError(
                "strict terminal_outcome fields must be booleans"
            )
        outcome = {field: bool(terminal_outcome[field]) for field in sorted(outcome_fields)}

    enriched = copy.deepcopy(dict(payload))
    frames = enriched.get("frames")
    if not isinstance(frames, list) or not frames:
        raise SimulationTraceNormalizationError("strict trace must contain at least one frame")
    for frame_index, frame in enumerate(frames):
        if not isinstance(frame, dict) or not isinstance(frame.get("planner"), dict):
            raise SimulationTraceNormalizationError(
                f"/frames/{frame_index}/planner: expected object"
            )
        planner = frame["planner"]
        if "run_config" in planner or "outcome" in planner:
            raise SimulationTraceNormalizationError(
                "strict metadata projection is non-additive: run_config/outcome already present"
            )
        planner["run_config"] = copy.deepcopy(dict(run_config))
    if outcome is not None:
        enriched["frames"][-1]["planner"]["outcome"] = outcome

    if _strict_projection_payload(enriched) != _strict_projection_payload(payload):
        raise SimulationTraceNormalizationError(
            "strict metadata projection changed canonical trace state"
        )
    simulation_trace_export_from_dict(enriched)
    # The equality check above fails closed, so both digests below describe the
    # same unchanged canonical state on every successfully returned receipt.
    receipt = {
        "schema_version": "issue_6814_metadata_delta.v1",
        "status": "complete",
        "before_projection_sha256": _strict_canonical_sha256(_strict_projection_payload(payload)),
        "after_projection_sha256": _strict_canonical_sha256(_strict_projection_payload(enriched)),
        "semantic_payload_unchanged": True,
        "added_paths": [f"/frames/{index}/planner/run_config" for index in range(len(frames))],
        "terminal_outcome_path": (
            f"/frames/{len(frames) - 1}/planner/outcome" if outcome is not None else None
        ),
        "run_config": copy.deepcopy(dict(run_config)),
        "terminal_outcome": outcome,
    }
    return enriched, receipt


@lru_cache(maxsize=1)
def load_simulation_trace_export_schema() -> dict[str, Any]:
    """Load the public ``simulation_trace_export.v1`` JSON schema.

    Returns:
        Parsed JSON Schema dictionary.
    """

    return json.loads(SIMULATION_TRACE_EXPORT_SCHEMA_FILE.read_text(encoding="utf-8"))


def load_simulation_trace_export(path: Path) -> SimulationTraceExport:
    """Load one simulation trace export from JSON.

    Returns:
        Typed simulation trace export.
    """

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise SimulationTraceExportValidationError(["expected a mapping payload"], source=path)
    return simulation_trace_export_from_dict(raw, source=path)


def simulation_trace_export_from_dict(
    payload: Mapping[str, Any],
    *,
    source: str | Path | None = None,
) -> SimulationTraceExport:
    """Validate and convert a mapping into a typed simulation trace export.

    Returns:
        Typed simulation trace export.
    """

    errors = _schema_validation_errors(payload)
    errors.extend(_semantic_validation_errors(payload))
    if errors:
        raise SimulationTraceExportValidationError(errors, source=source)
    return _export_from_payload(payload)


def _schema_validation_errors(payload: Mapping[str, Any]) -> list[str]:
    """Return sorted JSON Schema validation errors."""

    validator = Draft202012Validator(load_simulation_trace_export_schema())
    return [
        f"{json_pointer(error.absolute_path)}: {error.message}"
        for error in sorted(
            validator.iter_errors(payload),
            key=lambda err: list(err.absolute_path),
        )
    ]


def _semantic_validation_errors(payload: Mapping[str, Any]) -> list[str]:
    """Return trace-order validation errors not expressible in the JSON Schema."""

    frames = payload.get("frames")
    if not isinstance(frames, list):
        return []
    errors: list[str] = []
    previous_step: int | None = None
    previous_time: float | None = None
    for index, frame in enumerate(frames):
        if not isinstance(frame, Mapping):
            continue
        step = frame.get("step")
        time_s = frame.get("time_s")
        if isinstance(step, int):
            if previous_step is not None and step <= previous_step:
                errors.append(f"/frames/{index}/step: expected strictly increasing step")
            previous_step = step
        if isinstance(time_s, int | float):
            time_value = float(time_s)
            if previous_time is not None and time_value <= previous_time:
                errors.append(f"/frames/{index}/time_s: expected strictly increasing time_s")
            previous_time = time_value
    return errors


def _export_from_payload(payload: Mapping[str, Any]) -> SimulationTraceExport:
    """Build a typed export from a schema-valid payload.

    Returns:
        Typed simulation trace export.
    """

    source = payload["source"]
    return SimulationTraceExport(
        schema_version=str(payload["schema_version"]),
        trace_id=str(payload["trace_id"]),
        source=SimulationTraceSource(
            scenario_id=str(source["scenario_id"]),
            seed=int(source["seed"]),
            planner_id=str(source["planner_id"]),
            episode_id=str(source["episode_id"]),
            generated_by=str(source["generated_by"]),
        ),
        evidence_boundary=str(payload["evidence_boundary"]),
        coordinate_frame=str(payload["coordinate_frame"]),
        units=dict(payload["units"]),
        frames=[
            SimulationTraceFrame(
                step=int(frame["step"]),
                time_s=float(frame["time_s"]),
                robot=dict(frame["robot"]),
                pedestrians=[dict(pedestrian) for pedestrian in frame["pedestrians"]],
                planner=dict(frame["planner"]),
            )
            for frame in payload["frames"]
        ],
    )
