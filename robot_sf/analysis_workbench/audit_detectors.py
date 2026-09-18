"""Deterministic benchmark-audit detector registry and detector engine.

The detector layer intentionally consumes already-recorded episode mappings.  It
does not know how to create an environment, replay an episode, or fetch an
artifact.  Missing optional streams therefore produce an explicit
``unavailable`` signal, while a source contract that promised a value but
contains an invalid value produces a candidate integrity signal or an
``error`` signal.  Every returned signal is a BA-03 :class:`Signal` record.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from statistics import median
from typing import Any

from robot_sf.analysis_workbench.audit_contracts import (
    SIGNAL_STATUSES,
    Signal,
    TimeInterval,
    canonical_json,
)

DETECTOR_REGISTRY_SCHEMA_VERSION = "audit-detector-registry.v1"
DETECTOR_ENGINE_VERSION = "audit-detectors.v1"
DETECTOR_REGISTRY_VERSION = DETECTOR_REGISTRY_SCHEMA_VERSION
GOAL_ADJACENT_TIMEOUT_VERSION = "goal_adjacent_timeout.v1"

# These names are stable family identifiers.  The method version is carried
# independently in ``DetectorSpec.version`` and in every Signal.
DETERMINISTIC_DETECTOR_IDS = (
    "goal_adjacent_timeout",
    "initial_reset_anomaly",
    "stuck_no_progress",
    "oscillation_limit_cycle",
    "outcome_metric_contradiction",
    "extreme_measurements",
    "actuator_mismatch",
    "telemetry_integrity",
    "provenance_consistency",
    "common_mode_anomaly",
    "planner_disagreement",
    "seed_outlier",
)
ADVISORY_DETECTOR_IDS = (
    "cohort_multivariate_outlier",
    "trajectory_shape_outlier",
)
ALL_DETECTOR_IDS = DETERMINISTIC_DETECTOR_IDS + ADVISORY_DETECTOR_IDS

# Friendly aliases make the API tolerant of the family labels used in issue
# discussions without registering a second detector or producing duplicate
# signals.
DETECTOR_ALIASES = {
    "completion_goal_geometry": "goal_adjacent_timeout",
    "completion_and_goal_geometry": "goal_adjacent_timeout",
    "goal_geometry": "goal_adjacent_timeout",
    "initial_reset": "initial_reset_anomaly",
    "initial_reset_anomalies": "initial_reset_anomaly",
    "reset_anomaly": "initial_reset_anomaly",
    "no_progress": "stuck_no_progress",
    "stuck": "stuck_no_progress",
    "oscillation": "oscillation_limit_cycle",
    "oscillation_loops": "oscillation_limit_cycle",
    "outcome_metric": "outcome_metric_contradiction",
    "outcome_metric_inconsistency": "outcome_metric_contradiction",
    "extreme_measurement": "extreme_measurements",
    "command_execution_mismatch": "actuator_mismatch",
    "actuator_saturation": "actuator_mismatch",
    "common_mode": "common_mode_anomaly",
    "common_mode_anomalies": "common_mode_anomaly",
    "planner_peer_disagreement": "planner_disagreement",
    "configuration_provenance": "provenance_consistency",
    "seed_cohort_outlier": "seed_outlier",
    "cohort_outlier": "cohort_multivariate_outlier",
    "multivariate_outlier": "cohort_multivariate_outlier",
    "trajectory_outlier": "trajectory_shape_outlier",
}
_NON_ADMISSIBLE_EXECUTION_STATUSES = frozenset(
    {
        "fallback",
        "degraded",
        "failed",
        "failure",
        "partial",
        "partial_failure",
        "not_available",
        "unavailable",
        "unsupported",
        "error",
        "truncated",
        "diagnostic_only",
        "diagnostic_stub",
        "skipped",
        "cancelled",
    }
)
_EXECUTION_STATUS_FIELDS = (
    "row_status",
    "status",
    "availability_status",
    "readiness_status",
    "campaign_execution_status",
    "execution_status",
    "execution_mode",
    "preflight_status",
)
_FALLBACK_COUNTER_FIELDS = (
    "fallback_count",
    "fallback_steps",
    "fallback_actions",
    "fallback_events",
    "fallback_invocations",
    "fallback_used",
    "fallback_counter",
    "fallback_counters",
)
_NATIVE_EXECUTION_STATUSES = frozenset(
    {
        "native",
        "adapter",
        "available",
        "ready",
        "complete",
        "success",
        "collision",
        "ok",
        "passed",
        "pass",
        "admitted",
        "included",
        "running",
    }
)
_DESCRIPTIVE_STATUS_FIELDS = frozenset({"status", "execution_status"})
_DESCRIPTIVE_STATUS_VALUES = frozenset(
    {
        "success",
        "completed",
        "collision",
        "timeout",
        "timed_out",
        "horizon",
        "horizon_reached",
        "time_limit",
        "max_steps",
    }
)


def _canonical_detector_id(value: str) -> str:
    """Normalize documented family spellings to one stable detector ID.

    Returns:
        Canonical lower-case detector identifier.
    """

    token = value.strip().lower().replace("-", "_").replace(" ", "_")
    return DETECTOR_ALIASES.get(token, token)


class DetectorError(ValueError):
    """Raised when a detector specification or invocation is malformed."""


def _finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DetectorError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise DetectorError(f"{name} must be a finite number")
    return result


def _finite_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _configured_number(
    config: Mapping[str, Any],
    spec: DetectorSpec,
    name: str,
    *,
    minimum: float | None = None,
    strictly_greater: bool = False,
) -> float | None:
    """Read one finite detector parameter and enforce its physical domain.

    Returns:
        A finite configured value, or ``None`` when the value is malformed or
        outside the declared domain.
    """

    value = config.get(name, spec.parameters.get(name))
    result = _finite_or_none(value)
    if result is None:
        return None
    if minimum is not None and (result <= minimum if strictly_greater else result < minimum):
        return None
    return result


def _configured_integer(
    config: Mapping[str, Any],
    spec: DetectorSpec,
    name: str,
    *,
    minimum: int | None = None,
) -> int | None:
    """Read one bounded integer parameter without silently truncating values.

    Detector configuration is part of the reproducibility boundary.  A value
    such as ``3.9`` must therefore be rejected rather than quietly changing
    the requested threshold to ``3``.

    Returns:
        A configured integer, or ``None`` when its type/domain is invalid.
    """

    value = config.get(name, spec.parameters.get(name))
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if minimum is not None and value < minimum:
        return None
    return value


def _contains_nonfinite(value: Any) -> bool:
    """Return whether a nested promised telemetry value contains NaN/Inf."""

    pending = [value]
    visited: set[int] = set()
    while pending:
        current = pending.pop()
        if isinstance(current, float) and not math.isfinite(current):
            return True
        if isinstance(current, Mapping):
            marker = id(current)
            if marker in visited:
                continue
            visited.add(marker)
            pending.extend(current.values())
        elif isinstance(current, (list, tuple)):
            marker = id(current)
            if marker in visited:
                continue
            visited.add(marker)
            pending.extend(current)
    return False


def _promised_missing(value: Any) -> bool:
    """Return whether a promised telemetry value is absent, empty, or null."""

    if value is None:
        return True
    if isinstance(value, (str, bytes)):
        return not value.strip()
    if isinstance(value, (Mapping, list, tuple)):
        return not value
    return False


def _closed(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise DetectorError(f"{name} must be a mapping")
    try:
        # canonical_json both rejects non-finite values and catches unsupported
        # nested values before they can enter a Signal or registry digest.
        canonical_json(value)
    except (TypeError, ValueError, RecursionError) as exc:
        raise DetectorError(f"{name} is not strict JSON") from exc
    if any(not isinstance(key, str) for key in value):
        raise DetectorError(f"{name} keys must be strings")
    return json.loads(canonical_json(value))


def _strings(value: Any, *, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DetectorError(f"{name} must be a sequence of strings")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise DetectorError(f"{name} must contain non-empty strings")
    return tuple(item.strip() for item in value)


@dataclass(frozen=True, slots=True)
class DetectorSpec:
    """Versioned declaration for one detector family.

    ``cohort_definition``, ``parameters``, ``units`` and ``provenance`` are
    deliberately retained with the registry rather than hidden in executable
    code.  This lets a report explain exactly what a signal means and makes a
    changed configuration produce a new cache identity.
    """

    detector_id: str
    family: str
    description: str
    version: str = "1.0.0"
    required_capabilities: tuple[str, ...] = ()
    optional_capabilities: tuple[str, ...] = ()
    cohort_definition: Mapping[str, Any] = field(default_factory=dict)
    parameters: Mapping[str, Any] = field(default_factory=dict)
    units: Mapping[str, str] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    advisory: bool = False

    def __post_init__(self) -> None:
        """Validate a closed, reproducible detector declaration."""

        for name, value in (
            ("detector_id", self.detector_id),
            ("family", self.family),
            ("description", self.description),
            ("version", self.version),
        ):
            if not isinstance(value, str) or not value.strip():
                raise DetectorError(f"{name} must be a non-empty string")
        if not isinstance(self.advisory, bool):
            raise DetectorError("advisory must be a boolean")
        object.__setattr__(
            self,
            "required_capabilities",
            _strings(self.required_capabilities, name="required_capabilities"),
        )
        object.__setattr__(
            self,
            "optional_capabilities",
            _strings(self.optional_capabilities, name="optional_capabilities"),
        )
        for name in ("cohort_definition", "parameters", "units", "provenance"):
            object.__setattr__(self, name, _closed(getattr(self, name), name=name))
        if any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in self.units.items()
        ):
            raise DetectorError("units must map string names to string units")

    @property
    def detector_version(self) -> str:
        """Return the method version under the BA-03 field name."""

        return self.version

    def to_dict(self) -> dict[str, Any]:
        """Return the strict registry representation."""

        return {
            "detector_id": self.detector_id,
            "family": self.family,
            "description": self.description,
            "version": self.version,
            "required_capabilities": list(self.required_capabilities),
            "optional_capabilities": list(self.optional_capabilities),
            "cohort_definition": dict(self.cohort_definition),
            "parameters": dict(self.parameters),
            "units": dict(self.units),
            "provenance": dict(self.provenance),
            "advisory": self.advisory,
        }


@dataclass(frozen=True, slots=True)
class DetectorRegistry:
    """Immutable, deterministic collection of :class:`DetectorSpec` values."""

    version: str = DETECTOR_REGISTRY_SCHEMA_VERSION
    detectors: tuple[DetectorSpec, ...] = ()

    def __post_init__(self) -> None:
        """Reject duplicate IDs and unstable detector ordering."""

        if not isinstance(self.version, str) or not self.version.strip():
            raise DetectorError("registry version must be a non-empty string")
        specs = tuple(self.detectors)
        if any(not isinstance(item, DetectorSpec) for item in specs):
            raise DetectorError("registry detectors must be DetectorSpec values")
        ids = [item.detector_id for item in specs]
        if len(ids) != len(set(ids)):
            raise DetectorError("registry detector IDs must be unique")
        if ids != sorted(ids):
            raise DetectorError("registry detector IDs must be in deterministic order")
        object.__setattr__(self, "detectors", specs)

    @property
    def ids(self) -> tuple[str, ...]:
        """Return detector IDs in canonical execution order."""

        return tuple(item.detector_id for item in self.detectors)

    def get(self, detector_id: str) -> DetectorSpec:
        """Return one detector, resolving documented aliases."""

        canonical = _canonical_detector_id(detector_id)
        for detector in self.detectors:
            if detector.detector_id == canonical:
                return detector
        raise KeyError(detector_id)

    def to_dict(self) -> dict[str, Any]:
        """Return a machine-readable registry document."""

        return {
            "schema_version": self.version,
            "engine_version": DETECTOR_ENGINE_VERSION,
            "detectors": [item.to_dict() for item in self.detectors],
        }

    @property
    def digest(self) -> str:
        """Return the content digest used for cache invalidation."""

        return hashlib.sha256(canonical_json(self.to_dict()).encode("utf-8")).hexdigest()

    def __iter__(self):
        """Iterate over detector declarations in canonical order.

        Returns:
            An iterator over detector specifications.
        """

        return iter(self.detectors)

    def __len__(self) -> int:
        """Return the number of detector declarations."""

        return len(self.detectors)


def _spec(  # noqa: PLR0913
    detector_id: str,
    family: str,
    description: str,
    *,
    required: Iterable[str] = ("episode",),
    optional: Iterable[str] = (),
    cohort: Mapping[str, Any] | None = None,
    parameters: Mapping[str, Any] | None = None,
    units: Mapping[str, str] | None = None,
    version: str = "1.0.0",
    advisory: bool = False,
) -> DetectorSpec:
    return DetectorSpec(
        detector_id=detector_id,
        family=family,
        description=description,
        version=version,
        required_capabilities=tuple(required),
        optional_capabilities=tuple(optional),
        cohort_definition=cohort or {"kind": "episode", "key": ["campaign", "episode"]},
        parameters=parameters or {},
        units=units or {},
        provenance={
            "owner": "robot_sf.analysis_workbench.audit_detectors",
            "engine": DETECTOR_ENGINE_VERSION,
            "evidence_boundary": "diagnostic_only",
            "source": "recorded_campaign_only",
        },
        advisory=advisory,
    )


def default_registry(*, include_advisory: bool = True) -> DetectorRegistry:
    """Build the versioned default registry.

    The declarations are rebuilt as plain immutable values on each call, so a
    caller cannot mutate global state and accidentally affect later scans.

    Returns:
        A deterministic detector registry.
    """

    specs = [
        _spec(
            "actuator_mismatch",
            "command versus execution discrepancy",
            "Detect saturation, clipping, discontinuity, or desired/applied mismatch.",
            optional=("commands", "trace"),
            parameters={"absolute_tolerance": 1e-6, "relative_tolerance": 0.1},
            units={"linear_velocity": "m/s", "angular_velocity": "rad/s"},
        ),
        _spec(
            "cohort_multivariate_outlier",
            "cohort multivariate outlier",
            "Use disclosed robust feature distances for advisory within-cohort discovery.",
            optional=("cohort", "metrics", "trace", "outcome"),
            cohort={
                "kind": "same_configuration",
                "key": ["planner_id", "scenario_id", "config_id"],
            },
            parameters={"minimum_cohort": 4, "mad_z_threshold": 3.5},
            units={"robust_distance": "dimensionless"},
            advisory=True,
        ),
        _spec(
            "common_mode_anomaly",
            "cross-planner common-mode anomaly",
            "Identify unusually shared outcomes within a compatible planner cohort.",
            optional=("cohort", "outcome", "metrics"),
            cohort={
                "kind": "compatible_peer",
                "key": ["scenario_id", "config_id", "initial_state"],
                "require_present": ["scenario_id", "config_id", "initial_state"],
            },
            parameters={"failure_fraction": 0.8, "minimum_planners": 2},
        ),
        _spec(
            "extreme_measurements",
            "extreme clearance/collision/force/TTC measurements",
            "Check finite, physically bounded scalar measurements while preserving undefined TTC/PET.",
            optional=("metrics", "forces"),
            parameters={
                "clearance_min_m": 0.0,
                "force_max_N": 100.0,
                "ttc_min_s": 0.0,
                "collision_max": 1.0,
            },
            units={"clearance": "m", "force": "N", "ttc": "s", "collision": "count"},
        ),
        _spec(
            "goal_adjacent_timeout",
            "completion and goal geometry",
            "Apply goal_adjacent_timeout.v1 to recorded non-collision timeouts.",
            optional=("trace", "geometry", "forces", "events"),
            cohort={"kind": "episode", "key": ["campaign_digest", "episode_id"]},
            parameters={"tail_steps": 100, "goal_adjacent_radius_m": 4.0, "wall_margin_m": 0.5},
            units={"distance": "m", "time": "s", "force": "N"},
            version=GOAL_ADJACENT_TIMEOUT_VERSION,
        ),
        _spec(
            "initial_reset_anomaly",
            "initial/reset anomaly",
            "Detect initial overlap, immediate collision, or a reset mismatch under declared geometry.",
            optional=("initial_state", "trace", "events", "geometry"),
            parameters={"minimum_separation_m": 0.0, "initial_steps": 1},
            units={"distance": "m", "time": "s"},
        ),
        _spec(
            "oscillation_limit_cycle",
            "oscillation, repeated turning, and loops",
            "Detect repeated reversals or trajectory loops without interpreting them as infeasibility.",
            optional=("trace", "commands"),
            parameters={"minimum_reversals": 4, "loop_radius_m": 1.0, "heading_reversal_rad": 0.2},
            units={"angle": "rad", "distance": "m", "time": "s"},
        ),
        _spec(
            "outcome_metric_contradiction",
            "outcome versus metric contradiction",
            "Check outcome/event combinations only under the declared termination contract.",
            optional=("metrics", "events", "outcome"),
            parameters={"collision_count_max_for_success": 0.0},
        ),
        _spec(
            "planner_disagreement",
            "unusual planner relative to compatible peers",
            "Compare one planner with compatible peers without treating disagreement as proof.",
            optional=("cohort", "metrics", "outcome"),
            cohort={
                "kind": "compatible_peer",
                "key": ["scenario_id", "config_id", "seed", "initial_state"],
                "require_present": ["scenario_id", "config_id", "seed", "initial_state"],
            },
            parameters={"minimum_peers": 1, "metric_z_threshold": 3.0},
        ),
        _spec(
            "provenance_consistency",
            "planner/config/checkpoint provenance",
            "Check identity fields and declared campaign/config/checkpoint consistency.",
            optional=("provenance", "config"),
            parameters={
                "required_identity_fields": ["campaign_id", "source_commit", "config_identity"]
            },
        ),
        _spec(
            "seed_outlier",
            "seed/cohort outlier",
            "Report robust within-configuration seed outliers; seed numbering is never a distance.",
            optional=("cohort", "metrics"),
            cohort={
                "kind": "same_configuration",
                "key": ["planner_id", "scenario_id", "config_id"],
            },
            parameters={"minimum_cohort": 4, "mad_z_threshold": 3.5},
        ),
        _spec(
            "stuck_no_progress",
            "stuck/no progress",
            "Detect stalled route or goal progress while preserving expected waiting/yielding controls.",
            optional=("trace", "metrics", "geometry"),
            parameters={
                "minimum_duration_s": 1.0,
                "progress_tolerance_m": 0.05,
                "speed_tolerance_m_s": 0.02,
            },
            units={"distance": "m", "time": "s", "speed": "m/s"},
        ),
        _spec(
            "telemetry_integrity",
            "missing/nonfinite/incomplete telemetry",
            "Report missing promised telemetry separately from unrecorded optional streams.",
            optional=("trace", "events", "commands", "metrics"),
            parameters={"required_fields": []},
        ),
        _spec(
            "trajectory_shape_outlier",
            "trajectory-shape outlier",
            "Use disclosed robust trajectory features for advisory within-cohort discovery.",
            optional=("trace", "cohort"),
            cohort={
                "kind": "same_configuration",
                "key": ["planner_id", "scenario_id", "config_id"],
            },
            parameters={"minimum_cohort": 4, "mad_z_threshold": 3.5},
            units={"path_length": "m", "displacement": "m", "turns": "count"},
            advisory=True,
        ),
    ]
    if include_advisory:
        # The registry itself is sorted by ID.  This means adding a new family
        # cannot make scan order depend on source insertion order.
        return DetectorRegistry(detectors=tuple(sorted(specs, key=lambda item: item.detector_id)))
    return DetectorRegistry(
        detectors=tuple(
            sorted(
                (item for item in specs if not item.advisory),
                key=lambda item: item.detector_id,
            )
        )
    )


DEFAULT_REGISTRY = default_registry()


def detector_registry(*, include_advisory: bool = True) -> DetectorRegistry:
    """Compatibility factory for the default registry.

    Returns:
        A deterministic detector registry.
    """

    return default_registry(include_advisory=include_advisory)


def registry_document(*, include_advisory: bool = True) -> dict[str, Any]:
    """Return a JSON-safe registry document."""

    return default_registry(include_advisory=include_advisory).to_dict()


def normalize_detector_ids(
    detector_ids: Sequence[str] | None,
    *,
    include_advisory: bool = True,
    registry: DetectorRegistry | None = None,
) -> tuple[str, ...]:
    """Normalize requested detector IDs and reject unknown names.

    Returns:
        Sorted canonical detector IDs.
    """

    active_registry = registry or default_registry(include_advisory=include_advisory)
    if detector_ids is None:
        return active_registry.ids
    result: list[str] = []
    for item in detector_ids:
        if not isinstance(item, str) or not item.strip():
            raise DetectorError("detector IDs must be non-empty strings")
        canonical = _canonical_detector_id(item)
        try:
            active_registry.get(canonical)
        except KeyError as exc:
            raise DetectorError(f"unknown detector: {item!r}") from exc
        if canonical not in result:
            result.append(canonical)
    return tuple(sorted(result))


def _mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _first_mapping(row: Mapping[str, Any], *keys: str) -> Mapping[str, Any] | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, Mapping):
            return value
    return None


def _metrics(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("metrics")
    metrics = dict(value) if isinstance(value, Mapping) else {}
    # Operational-metric exports are a canonical sidecar in retained campaign
    # records.  Treat it as a read-only compatibility projection when the
    # compact ``metrics`` block does not already provide a field; a detector
    # must not silently prefer a derived alias over an explicit metric.
    operational = row.get("operational_metrics")
    if isinstance(operational, Mapping):
        for key, item in operational.items():
            if isinstance(key, str):
                metrics.setdefault(key, item)
    # Some retained campaign exports flatten operational metrics at the row
    # level.  Preserve the nested block as authoritative while accepting the
    # documented scalar aliases as a read-only compatibility projection.
    for key in (
        "clearance_m",
        "min_clearance_m",
        "wall_clearance_m",
        "force_N",
        "force_max_N",
        "ttc_s",
        "pet_s",
        "collision_count",
        "collisions",
    ):
        if key in row and key not in metrics:
            metrics[key] = row[key]
    return metrics


def _counter_failure(value: Any, *, path: str) -> tuple[str, str] | None:
    """Return a fail-closed disposition for a nested fallback counter."""

    if isinstance(value, bool):
        return ("unavailable", f"{path}_nonzero") if value else None
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)) or value < 0:
            return "error", f"{path}_malformed"
        return ("unavailable", f"{path}_nonzero") if value > 0 else None
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                return "error", f"{path}_malformed"
            failure = _counter_failure(item, path=f"{path}.{key}")
            if failure is not None:
                return failure
        return None
    if isinstance(value, list):
        return ("unavailable", f"{path}_nonzero") if value else None
    return "error", f"{path}_malformed"


def _admission_containers(row: Mapping[str, Any]) -> tuple[tuple[str, Mapping[str, Any]], ...]:
    """Collect canonical status/counter containers for direct detector calls.

    Returns:
        Named row mappings whose status and fallback fields are authoritative.
    """

    containers: list[tuple[str, Mapping[str, Any]]] = [("row", row)]
    for name in ("metrics", "operational_metrics"):
        value = row.get(name)
        if isinstance(value, Mapping):
            containers.append((name, value))
    metadata = row.get("algorithm_metadata")
    if isinstance(metadata, Mapping):
        containers.append(("algorithm_metadata", metadata))
        trace = metadata.get("analysis_trace")
        if isinstance(trace, Mapping):
            containers.append(("analysis_trace", trace))
    top_level_trace = row.get("analysis_trace")
    if isinstance(top_level_trace, Mapping):
        containers.append(("analysis_trace", top_level_trace))
    return tuple(containers)


def _admission_failure(  # noqa: C901, PLR0912
    row: Mapping[str, Any], *, check_nonfinite: bool = True
) -> tuple[str, str] | None:
    """Return a fail-closed execution disposition for direct detector calls.

    Returns:
        ``(status, reason)`` when a row is unavailable or malformed, otherwise
        ``None`` for a native, finite row.
    """

    metadata = row.get("algorithm_metadata")
    if metadata is not None and not isinstance(metadata, Mapping):
        return "error", "algorithm_metadata_malformed"
    trace = metadata.get("analysis_trace") if isinstance(metadata, Mapping) else None
    if trace is not None and not isinstance(trace, Mapping):
        return "error", "analysis_trace_malformed"
    top_level_trace = row.get("analysis_trace")
    if top_level_trace is not None and not isinstance(top_level_trace, Mapping):
        return "error", "analysis_trace_malformed"
    for name in ("metrics", "operational_metrics"):
        value = row.get(name)
        if value is not None and not isinstance(value, Mapping):
            return "error", f"{name}_malformed"
    for path, container in _admission_containers(row):
        for key in _EXECUTION_STATUS_FIELDS:
            if key not in container:
                continue
            value = container[key]
            if not isinstance(value, str) or not value.strip():
                return "error", f"{path}.{key}_malformed"
            token = value.strip().lower().replace("-", "_").replace(" ", "_")
            if token in _NON_ADMISSIBLE_EXECUTION_STATUSES:
                return "unavailable", f"non_admissible_execution_status_{token}"
            if token not in _NATIVE_EXECUTION_STATUSES:
                # Canonical v2 reserves row_status for execution mode while
                # status/execution_status may describe the terminal outcome.
                if key in _DESCRIPTIVE_STATUS_FIELDS and token in _DESCRIPTIVE_STATUS_VALUES:
                    continue
                return "unavailable", f"unknown_execution_status_{token}"
        for key in _FALLBACK_COUNTER_FIELDS:
            if key not in container:
                continue
            failure = _counter_failure(container[key], path=f"{path}.{key}")
            if failure is not None:
                return failure
    if check_nonfinite and _contains_nonfinite(row):
        return "error", "row_contains_nonfinite_value"
    return None


def _outcome(row: Mapping[str, Any]) -> Mapping[str, Any]:  # noqa: C901, PLR0912
    value = row.get("outcome")
    if isinstance(value, Mapping):
        outcome = dict(value)
    elif isinstance(value, str):
        outcome = {"label": value}
    else:
        outcome = {}
    metrics = _metrics(row)
    if isinstance(metrics, Mapping):
        for key in ("success", "route_complete", "reached_goal", "completed"):
            if key in metrics:
                binary = _binary(metrics[key])
                if binary is not None:
                    outcome.setdefault(key, binary)
        for key in ("collision", "collision_event"):
            if key in metrics:
                binary = _binary(metrics[key])
                if binary is not None:
                    outcome.setdefault(key, binary)
        for key in ("timeout", "timed_out", "timeout_event", "horizon_reached"):
            if key in metrics:
                binary = _binary(metrics[key])
                if binary is not None:
                    outcome.setdefault(key, binary)
        if "collisions" in metrics:
            collisions = _finite_or_none(metrics["collisions"])
            if collisions is not None:
                outcome.setdefault("collision_count", collisions)
                outcome.setdefault("collision", collisions > 0.0)
        if "collision_count" in metrics:
            collisions = _finite_or_none(metrics["collision_count"])
            if collisions is not None:
                outcome.setdefault("collision_count", collisions)
                outcome.setdefault("collision", collisions > 0.0)
    return outcome


def _trace(  # noqa: C901, PLR0912
    row: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]] | None, str | None]:
    """Find a trace without interpreting an arbitrary object as telemetry.

    Returns:
        Trace frames and an unavailable reason when no trace is present.
    """

    metadata = row.get("algorithm_metadata")
    # ``analysis_trace`` is the canonical campaign-result-store.v2 trace.  It
    # must win over compatibility projections: a flattened ``trace`` field
    # may be stale or empty even when the admitted canonical trace is present.
    candidates: list[Any] = []
    if isinstance(metadata, Mapping):
        candidates.extend(
            [
                metadata.get("analysis_trace"),
                metadata.get("simulation_step_trace"),
                metadata.get("trace"),
                metadata.get("simulation_trace"),
            ]
        )
    candidates.extend([row.get("analysis_trace"), row.get("trace"), row.get("simulation_trace")])
    coverage_values: list[Any] = [row.get("trace_coverage")]
    for candidate in candidates:
        if isinstance(candidate, Mapping):
            coverage_values.append(candidate.get("trace_coverage"))
    for coverage in coverage_values:
        if not isinstance(coverage, Mapping):
            continue
        status = coverage.get("status")
        if status is not None and not isinstance(status, str):
            return None, "trace_coverage_malformed"
        normalized = status.strip().lower().replace("-", "_") if isinstance(status, str) else ""
        if normalized in {
            "unavailable",
            "not_available",
            "partial",
            "partial_failure",
            "incomplete",
            "failed",
            "error",
            "truncated",
        }:
            return None, "trace_coverage_unavailable"
    for candidate in candidates:
        if isinstance(candidate, list):
            frames = candidate
        elif isinstance(candidate, Mapping):
            frames = candidate.get("frames")
            if frames is None:
                frames = candidate.get("steps")
            if frames is None:
                frames = candidate.get("trajectory")
        else:
            continue
        if isinstance(frames, list):
            if not all(isinstance(frame, Mapping) for frame in frames):
                return None, "trace_frame_malformed"
            if not frames:
                return [], "trace_empty"
            return [frame for frame in frames if isinstance(frame, Mapping)], None
    return None, "trace_not_recorded"


def _vector(value: Any, *, name: str) -> tuple[float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
        raise DetectorError(f"{name} must be a two-coordinate vector")
    return (_finite(value[0], name=f"{name}[0]"), _finite(value[1], name=f"{name}[1]"))


def _optional_vector(value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    return _vector(value, name="vector")


def _wrapped_angle_delta(current: float, previous: float) -> float:
    """Return the shortest signed heading delta in radians."""

    return (current - previous + math.pi) % (2.0 * math.pi) - math.pi


def _position(frame: Mapping[str, Any]) -> tuple[float, float] | None:
    robot = frame.get("robot")
    if not isinstance(robot, Mapping):
        return None
    if "position" not in robot:
        return None
    return _optional_vector(robot.get("position"))


def _time(frame: Mapping[str, Any], fallback: float) -> float:
    value = frame.get("time_s", fallback)
    return _finite(value, name="trace.time_s")


def _trace_positions(
    row: Mapping[str, Any],
) -> tuple[list[tuple[float, float]], list[float], list[Mapping[str, Any]]]:
    frames, reason = _trace(row)
    if frames is None:
        raise DetectorError(reason or "trace_not_recorded")
    positions: list[tuple[float, float]] = []
    times: list[float] = []
    previous_time: float | None = None
    for index, frame in enumerate(frames):
        position = _position(frame)
        if position is None:
            raise DetectorError(f"robot_position_missing_at_step_{index}")
        if "time_s" not in frame:
            raise DetectorError(f"trace_time_missing_at_step_{index}")
        current_time = _time(frame, float(index))
        if previous_time is not None and current_time <= previous_time:
            raise DetectorError(f"trace_time_nonmonotonic_at_step_{index}")
        positions.append(position)
        times.append(current_time)
        previous_time = current_time
    if not positions:
        raise DetectorError("trace_empty")
    return positions, times, frames


def _trace_interval(frames: Sequence[Mapping[str, Any]]) -> TimeInterval | None:
    if not frames:
        return None
    try:
        start = _time(frames[0], 0.0)
        end = _time(frames[-1], start)
    except DetectorError:
        return None
    return TimeInterval(start, end)


def _lookup(row: Mapping[str, Any], *keys: str) -> Any:
    """Look up explicit aliases while avoiding fuzzy/unsafe recursive search.

    Returns:
        The first explicitly named value, or ``None``.
    """

    for key in keys:
        if key in row:
            return row[key]
    for container_name in ("metrics", "operational_metrics", "scenario_params", "geometry"):
        container = row.get(container_name)
        if isinstance(container, Mapping):
            for key in keys:
                if key in container:
                    return container[key]
    return None


def _goal(row: Mapping[str, Any]) -> tuple[tuple[float, float] | None, float | None]:
    value = _lookup(row, "final_waypoint", "goal_position", "goal", "target_position", "waypoint")
    if isinstance(value, Mapping):
        nested = value
        value = nested.get("position") or nested.get("point") or nested.get("xy")
        if value is None and "x" in nested and "y" in nested:
            value = [nested["x"], nested["y"]]
    if isinstance(value, str):
        value = None
    goal = _optional_vector(value) if value is not None else None
    radius_value = _lookup(
        row,
        "completion_radius_m",
        "goal_proximity_threshold",
        "completion_threshold_m",
        "goal_radius_m",
    )
    radius = _finite_or_none(radius_value)
    return goal, radius


def _goal_context(row: Mapping[str, Any]) -> dict[str, tuple[float, float]]:
    """Read optional waypoint/visible-goal points for explanatory evidence.

    These points do not alter ``goal_adjacent_timeout.v1``.  They only make a
    signal useful when a campaign records an active waypoint or a displayed
    goal distinct from the sampled final completion point.

    Returns:
        Validated optional point values keyed by their recorded role.
    """

    context: dict[str, tuple[float, float]] = {}
    for role, keys in (
        ("active_waypoint", ("active_waypoint", "current_waypoint")),
        ("visible_goal", ("visible_goal", "visible_goal_position")),
    ):
        value = _lookup(row, *keys)
        if isinstance(value, Mapping):
            nested = value
            value = nested.get("position") or nested.get("point") or nested.get("xy")
            if value is None and "x" in nested and "y" in nested:
                value = [nested["x"], nested["y"]]
        if isinstance(value, str):
            # An identifier-only waypoint/goal reference is useful context,
            # but it cannot be treated as a coordinate measurement.
            continue
        if value is not None:
            point = _optional_vector(value)
            if point is not None:
                context[role] = point
    return context


def _bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _binary(value: Any) -> bool | None:
    """Interpret explicitly boolean or 0/1 compatibility metrics only.

    Returns:
        A boolean for an explicit binary value, otherwise ``None``.
    """

    if isinstance(value, bool):
        return value
    numeric = _finite_or_none(value)
    if numeric in (0.0, 1.0):
        return bool(numeric)
    return None


def _timeout(row: Mapping[str, Any]) -> bool | None:
    outcome = _outcome(row)
    for key in ("timeout_event", "timeout", "timed_out", "horizon_reached"):
        if key in outcome:
            return _bool(outcome[key])
        if key in row:
            return _bool(row[key])
    label = outcome.get("label") or outcome.get("status") or row.get("termination_reason")
    if isinstance(label, str):
        normalized = label.strip().lower().replace("-", "_")
        if normalized in {
            "timeout",
            "timed_out",
            "horizon",
            "horizon_reached",
            "time_limit",
            "max_steps",
        }:
            return True
        if normalized in {"success", "completed", "collision", "failed"}:
            return False
    return None


def _collision(row: Mapping[str, Any]) -> bool | None:
    outcome = _outcome(row)
    for key in ("collision_event", "collision", "collided"):
        if key in outcome:
            return _bool(outcome[key])
        if key in row:
            return _bool(row[key])
    label = outcome.get("label") or outcome.get("status") or row.get("termination_reason")
    return label.strip().lower() in {"collision", "collided"} if isinstance(label, str) else None


def _success(row: Mapping[str, Any]) -> bool | None:
    outcome = _outcome(row)
    for key in ("success", "route_complete", "reached_goal", "completed"):
        if key in outcome:
            return _bool(outcome[key])
        if key in row:
            return _bool(row[key])
    label = outcome.get("label") or outcome.get("status") or row.get("termination_reason")
    return (
        label.strip().lower()
        in {"success", "completed", "complete", "goal_reached", "goal_reached_success"}
        if isinstance(label, str)
        else None
    )


def _outcome_label(row: Mapping[str, Any]) -> str | None:
    """Return a comparable outcome label from labels or explicit booleans.

    Retained campaigns use both a human-readable ``label`` and compact
    success/collision/timeout booleans.  Peer detectors compare this one
    normalized state so a schema projection does not hide disagreement.

    Returns:
        A stable outcome label, or ``None`` when no outcome is recorded.
    """

    outcome = _outcome(row)
    for key in ("label", "status"):
        value = outcome.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower().replace("-", "_")
    if _collision(row) is True:
        return "collision"
    if _timeout(row) is True:
        return "timeout"
    if _success(row) is True:
        return "success"
    if _success(row) is False:
        return "failure"
    return None


def _capabilities(row: Mapping[str, Any]) -> set[str]:
    declared = row.get("capabilities")
    result = {"episode"}
    if isinstance(declared, Mapping):
        result.update(
            key for key, present in declared.items() if isinstance(key, str) and present is True
        )
    elif isinstance(declared, Sequence) and not isinstance(declared, (str, bytes)):
        result.update(item for item in declared if isinstance(item, str))
    if isinstance(row.get("metrics"), Mapping) or isinstance(
        row.get("operational_metrics"), Mapping
    ):
        result.add("metrics")
    if _trace(row)[0] is not None:
        result.add("trace")
    for key, capability in (
        ("events", "events"),
        ("provenance", "provenance"),
        ("config", "config"),
        ("initial_state", "initial_state"),
        ("geometry", "geometry"),
        ("commands", "commands"),
        ("forces", "forces"),
    ):
        if isinstance(row.get(key), Mapping) or isinstance(row.get(key), list):
            result.add(capability)
    if "commands" in row or "desired_action" in row or "executed_action" in row:
        result.add("commands")
    if "cohort" in row:
        result.add("cohort")
    return result


def _identity(row: Mapping[str, Any], key: str, default: str = "unknown") -> str:
    value = row.get(key)
    if value is None and key == "planner_id":
        value = row.get("planner") or row.get("algo") or row.get("algorithm")
        params = row.get("scenario_params")
        if value is None and isinstance(params, Mapping):
            value = params.get("planner") or params.get("algo") or params.get("algorithm")
    if value is None and key == "scenario_id":
        value = row.get("scenario")
    if value is None and key == "config_id":
        value = row.get("config_identity") or row.get("config_hash")
        config = row.get("config")
        if isinstance(config, Mapping):
            value = (
                config.get("config_id")
                or config.get("config_identity")
                or config.get("config_hash")
                or value
            )
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default


def _cohort_key(row: Mapping[str, Any], keys: Sequence[str]) -> tuple[Any, ...]:
    values: list[Any] = []
    for key in keys:
        if key in {"initial_state", "initial_state_digest"}:
            value = row.get("initial_state_digest") or row.get("initial_state")
            if isinstance(value, Mapping):
                value = canonical_json(value)
        elif key == "campaign_digest":
            value = row.get("campaign_digest") or row.get("campaign_id")
        else:
            value = row.get(key)
            if key == "planner_id" and value is None:
                value = _identity(row, "planner_id")
            if key == "scenario_id" and value is None:
                value = _identity(row, "scenario_id")
            if key == "config_id" and value is None:
                value = _identity(row, "config_id")
        try:
            canonical = canonical_json(value)
        except (TypeError, ValueError):
            canonical = repr(value)
        values.append(canonical)
    return tuple(values)


def _cohort_field_present(row: Mapping[str, Any], key: str) -> bool:
    """Return whether one declared compatibility field is actually recorded."""

    if key in {"initial_state", "initial_state_digest"}:
        return row.get("initial_state_digest") is not None or row.get("initial_state") is not None
    if key == "planner_id":
        return bool(_identity(row, "planner_id", default=""))
    if key == "scenario_id":
        return bool(_identity(row, "scenario_id", default=""))
    if key == "config_id":
        return bool(_identity(row, "config_id", default=""))
    if key == "campaign_digest":
        return bool(row.get("campaign_digest") or row.get("campaign_id"))
    return row.get(key) is not None


def _group_cohort(
    row: Mapping[str, Any], cohort: Sequence[Mapping[str, Any]], spec: DetectorSpec
) -> list[Mapping[str, Any]]:
    definition = spec.cohort_definition
    keys = definition.get("key", []) if isinstance(definition, Mapping) else []
    if not isinstance(keys, Sequence) or isinstance(keys, (str, bytes)):
        return []
    string_keys = [item for item in keys if isinstance(item, str)]
    required = (
        definition.get("require_present", string_keys)
        if isinstance(definition, Mapping)
        else string_keys
    )
    if not isinstance(required, Sequence) or isinstance(required, (str, bytes)):
        return []
    required_keys = [item for item in required if isinstance(item, str)]
    if any(not _cohort_field_present(row, key) for key in required_keys):
        return []
    target = _cohort_key(row, string_keys)
    return [
        item
        for item in cohort
        if all(_cohort_field_present(item, key) for key in required_keys)
        and _cohort_key(item, string_keys) == target
    ]


def _signal_id(
    spec: DetectorSpec,
    episode_id: str,
    *,
    config: Mapping[str, Any] | None = None,
    identity: Mapping[str, Any] | None = None,
) -> str:
    payload = {
        "detector_id": spec.detector_id,
        "detector_version": spec.version,
        "episode_id": episode_id,
        "config": dict(config or {}),
        "identity": dict(identity or {}),
    }
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return f"signal-{digest}"


def _base_evidence(
    spec: DetectorSpec,
    *,
    cohort: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "kind": "detector_method",
        "detector_id": spec.detector_id,
        "method_version": spec.version,
        "engine_version": DETECTOR_ENGINE_VERSION,
        "evidence_boundary": "diagnostic_only",
        "units": dict(spec.units),
        "parameters": dict(spec.parameters),
        "cohort_definition": dict(spec.cohort_definition),
        "provenance": dict(spec.provenance),
    }
    if cohort is not None:
        value["cohort"] = dict(cohort)
    if config:
        applied = {key: config[key] for key in sorted(config) if key in spec.parameters}
        if applied:
            value["parameters_applied"] = applied
    if spec.advisory:
        value["score_semantics"] = "uncalibrated review priority, not probability"
    return value


def _recorded_context_references(row: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    """Summarize canonical sidecars without copying untrusted trace payloads.

    BA-01 consumes sidecars produced by the review-context, event-alignment,
    timeline/phase, failure-predicate/diagnosis, operational-metric, and case
    portfolio owners.  Their contracts remain owned by those modules; a
    signal only carries bounded identity/status references so a detector cannot
    accidentally reinterpret or rewrite a sidecar.

    Returns:
        Strict context-reference evidence entries.
    """

    references: list[Mapping[str, Any]] = []
    context_names = (
        "review_context",
        "event_alignment",
        "simulation_timeline",
        "timeline",
        "episode_phases",
        "trace_failure_predicates",
        "failure_diagnosis",
        "operational_metrics",
        "case_portfolio",
        "events",
    )
    identity_fields = (
        "schema_version",
        "version",
        "digest",
        "source_digest",
        "artifact_id",
        "status",
        "validity_status",
        "evidence_boundary",
    )
    for name in context_names:
        value = row.get(name)
        if value is None:
            continue
        if isinstance(value, Mapping):
            summary = {
                key: value[key]
                for key in identity_fields
                if key in value
                and isinstance(value[key], (str, int, float, bool))
                and (not isinstance(value[key], float) or math.isfinite(value[key]))
            }
            summary["present"] = True
        elif isinstance(value, list):
            summary = {"count": len(value)}
        else:
            summary = {"value_type": type(value).__name__}
        references.append({"kind": "recorded_context", "context": name, "summary": summary})
    return tuple(references)


def _make_signal(  # noqa: PLR0913
    spec: DetectorSpec,
    row: Mapping[str, Any],
    status: str,
    *,
    reason: str,
    message: str = "",
    measured: Mapping[str, Any] | None = None,
    threshold: Mapping[str, Any] | None = None,
    evidence: Sequence[Mapping[str, Any]] = (),
    missingness: Sequence[str] = (),
    interval: TimeInterval | None = None,
    config: Mapping[str, Any] | None = None,
) -> Signal:
    if status not in SIGNAL_STATUSES:
        raise DetectorError(f"unknown signal status: {status}")
    episode_id = row.get("episode_id", "")
    if not isinstance(episode_id, str):
        episode_id = str(episode_id)
    identity = {
        key: row.get(key)
        for key in (
            "campaign_digest",
            "source_digest",
            "execution_id",
            "config_digest",
            "checkpoint_digest",
            "environment_digest",
            "campaign_id",
            "source_commit",
            "config_identity",
        )
        if row.get(key) not in (None, "")
    }
    if "execution_id" not in identity:
        # EpisodeRef uses the episode key as its deterministic fallback when
        # a legacy row omits a separate execution identity.  Carry the same
        # fallback into signal identity so reruns cannot accidentally share a
        # signal merely because the visible episode label is reused.
        identity["execution_id"] = episode_id
    provenance = row.get("provenance")
    if isinstance(provenance, Mapping):
        for key in ("campaign_id", "source_commit", "config_identity"):
            if provenance.get(key) not in (None, ""):
                identity[key] = provenance[key]
    full_evidence: tuple[Mapping[str, Any], ...] = (_base_evidence(spec, config=config),)
    full_evidence += _recorded_context_references(row)
    source_reference = {
        key: row.get(key)
        for key in ("source_artifact_id", "source_uri", "source_digest")
        if row.get(key) not in (None, "")
    }
    if source_reference:
        full_evidence += (
            {
                "kind": "source_reference",
                "reference": source_reference,
            },
        )
    if identity:
        full_evidence += (
            {
                "kind": "source_identity",
                "identity": identity,
            },
        )
    full_evidence += tuple(dict(item) for item in evidence)
    if status in {"unavailable", "error"} and not message:
        message = reason or "detector input is unavailable"
    return Signal(
        signal_id=_signal_id(spec, episode_id, config=config, identity=identity),
        detector_id=spec.detector_id,
        detector_version=spec.version,
        status=status,
        reason_code=reason,
        episode_id=episode_id,
        evidence=full_evidence,
        measured=measured or {},
        interval=interval,
        threshold=threshold,
        missingness=tuple(missingness),
        message=message,
    )


def _unavailable(
    spec: DetectorSpec,
    row: Mapping[str, Any],
    reason: str,
    *,
    missing: Sequence[str] = (),
    config: Mapping[str, Any] | None = None,
) -> Signal:
    evidence: tuple[Mapping[str, Any], ...] = ()
    if spec.advisory:
        # Keep the statistical interpretation explicit even when no score can
        # be computed (for example, a too-small cohort or absent trace).
        evidence = (
            {
                "statistical_method": "robust_cohort_outlier_discovery",
                "scaling": "median_and_MAD_robust_z",
                "cohort": dict(spec.cohort_definition),
                "missingness": list(missing) or [reason],
                "score_semantics": "uncalibrated review priority, not probability",
            },
        )
    return _make_signal(
        spec,
        row,
        "unavailable",
        reason=reason,
        message=reason,
        missingness=tuple(missing) or (reason,),
        evidence=evidence,
        config=config,
    )


def _detector_error(
    spec: DetectorSpec,
    row: Mapping[str, Any],
    reason: str,
    *,
    missing: Sequence[str] = (),
    config: Mapping[str, Any] | None = None,
) -> Signal:
    evidence: tuple[Mapping[str, Any], ...] = ()
    if spec.advisory:
        evidence = (
            {
                "statistical_method": "robust_cohort_outlier_discovery",
                "scaling": "median_and_MAD_robust_z",
                "cohort": dict(spec.cohort_definition),
                "missingness": list(missing) or [reason],
                "score_semantics": "uncalibrated review priority, not probability",
            },
        )
    return _make_signal(
        spec,
        row,
        "error",
        reason=reason,
        message=reason,
        missingness=tuple(missing),
        evidence=evidence,
        config=config,
    )


def _trace_failure(
    spec: DetectorSpec,
    row: Mapping[str, Any],
    error: DetectorError,
    *,
    config: Mapping[str, Any] | None = None,
) -> Signal:
    reason = str(error)
    if reason in {"trace_not_recorded", "trace_empty", "trace_coverage_unavailable"}:
        return _unavailable(spec, row, reason, missing=("trace",), config=config)
    if reason.startswith("robot_position_missing_at_step_") or reason.startswith(
        "trace_time_missing_at_step_"
    ):
        return _unavailable(spec, row, reason, missing=("trace",), config=config)
    return _detector_error(spec, row, reason, config=config)


def _goal_adjacent_timeout(  # noqa: C901, PLR0912
    spec: DetectorSpec,
    row: Mapping[str, Any],
    cohort: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> Signal:
    timeout = _timeout(row)
    collision = _collision(row)
    if timeout is False or collision is True:
        return _make_signal(spec, row, "clear", reason="not_noncollision_timeout", config=config)
    if timeout is None:
        return _unavailable(
            spec,
            row,
            "timeout_status_unavailable",
            missing=("outcome.timeout_event",),
            config=config,
        )
    if collision is None:
        return _unavailable(
            spec,
            row,
            "collision_status_unavailable",
            missing=("outcome.collision_event",),
            config=config,
        )
    try:
        positions, _times, frames = _trace_positions(row)
    except DetectorError as error:
        return _trace_failure(spec, row, error, config=config)
    tail_steps = _configured_integer(config, spec, "tail_steps", minimum=1)
    adjacent_radius = _configured_number(
        config, spec, "goal_adjacent_radius_m", minimum=0.0, strictly_greater=True
    )
    if tail_steps is None or adjacent_radius is None:
        return _detector_error(spec, row, "invalid_goal_adjacent_parameters", config=config)
    if len(positions) < tail_steps:
        return _unavailable(
            spec,
            row,
            f"trace_shorter_than_{tail_steps}_steps",
            missing=("trace.tail",),
            config=config,
        )
    goal, completion_radius = _goal(row)
    if goal is None:
        return _unavailable(
            spec,
            row,
            "final_waypoint_unavailable",
            missing=("geometry.final_waypoint",),
            config=config,
        )
    completion_radius_value = _lookup(
        row,
        "completion_radius_m",
        "goal_proximity_threshold",
        "completion_threshold_m",
        "goal_radius_m",
    )
    if completion_radius is None and completion_radius_value is not None:
        return _detector_error(spec, row, "completion_radius_m_malformed", config=config)
    if completion_radius is not None and completion_radius <= 0:
        return _detector_error(spec, row, "completion_radius_m_out_of_range", config=config)
    if completion_radius is None:
        return _unavailable(
            spec,
            row,
            "completion_radius_unavailable",
            missing=("geometry.completion_radius_m",),
            config=config,
        )
    distances = [math.hypot(x - goal[0], y - goal[1]) for x, y in positions]
    min_tail = min(distances[-tail_steps:])
    min_episode = min(distances)
    flagged = min_tail < adjacent_radius and min_episode > completion_radius
    evidence = [
        {
            "predicate": "goal_adjacent_timeout.v1",
            "tail_steps": tail_steps,
            "interpretation": "candidate goal-approach pathology; not a geometrical impossibility claim",
            "completion_rule": "distance_to_final_sampled_waypoint <= completion_radius_m",
        }
    ]
    measured = {
        "min_tail_distance_m": min_tail,
        "min_episode_distance_m": min_episode,
        "completion_radius_m": completion_radius,
        "goal_adjacent_radius_m": adjacent_radius,
        "timeout_event": True,
        "collision_event": bool(collision),
        "infeasibility_claimed": False,
    }
    for role, point in _goal_context(row).items():
        measured[f"{role}_xy_m"] = list(point)
    forces = row.get("forces")
    if forces is not None:
        if isinstance(forces, list):
            if _contains_nonfinite(forces):
                return _detector_error(spec, row, "force_context_nonfinite", config=config)
            measured["force_response_records"] = len(forces)
        elif not isinstance(forces, Mapping):
            return _detector_error(spec, row, "force_context_malformed", config=config)
        elif _contains_nonfinite(forces):
            return _detector_error(spec, row, "force_context_nonfinite", config=config)
        else:
            force_values: dict[str, float] = {}
            for name, value in forces.items():
                if not isinstance(name, str) or "force" not in name.lower():
                    continue
                numeric = _finite_or_none(value)
                if value is not None and numeric is None:
                    return _detector_error(spec, row, f"forces.{name}_malformed", config=config)
                if numeric is not None:
                    force_values[name] = numeric
            if force_values:
                measured["force_response"] = force_values
    wall_distance = _finite_or_none(
        _lookup(
            row,
            "final_waypoint_wall_distance_m",
            "wall_clearance_m",
            "goal_wall_distance_m",
        )
    )
    wall_distance_value = _lookup(
        row,
        "final_waypoint_wall_distance_m",
        "wall_clearance_m",
        "goal_wall_distance_m",
    )
    if wall_distance is None and wall_distance_value is not None:
        return _detector_error(spec, row, "wall_distance_m_malformed", config=config)
    if wall_distance is not None and wall_distance < 0.0:
        return _detector_error(spec, row, "wall_distance_m_out_of_range", config=config)
    wall_margin = _configured_number(config, spec, "wall_margin_m", minimum=0.0)
    if wall_margin is None:
        return _detector_error(spec, row, "invalid_goal_adjacent_parameters", config=config)
    if wall_distance is not None:
        measured["final_waypoint_wall_distance_m"] = wall_distance
        measured["wall_adjacent_threshold_m"] = completion_radius + (wall_margin or 0.0)
    threshold = {
        "min_tail_distance_m": {"operator": "<", "value": adjacent_radius, "unit": "m"},
        "min_episode_distance_m": {"operator": ">", "value": completion_radius, "unit": "m"},
    }
    if wall_distance is not None:
        threshold["final_waypoint_wall_distance_m"] = {
            "operator": "<=",
            "value": completion_radius + (wall_margin or 0.0),
            "unit": "m",
            "interpretation": "context only; does not establish geometric impossibility",
        }
    return _make_signal(
        spec,
        row,
        "flagged" if flagged else "clear",
        reason="goal_adjacent_timeout" if flagged else "goal_adjacent_predicate_false",
        measured=measured,
        threshold=threshold,
        evidence=evidence,
        interval=_trace_interval(frames),
        config=config,
    )


def _initial_reset_anomaly(  # noqa: C901, PLR0912
    spec: DetectorSpec,
    row: Mapping[str, Any],
    cohort: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> Signal:
    explicit = [
        row.get("initial_collision"),
        row.get("spawn_overlap"),
        row.get("reset_mismatch"),
    ]
    if any(value is not None and not isinstance(value, bool) for value in explicit):
        return _detector_error(spec, row, "initial_reset_flags_malformed", config=config)
    if any(value is True for value in explicit):
        return _make_signal(
            spec,
            row,
            "flagged",
            reason="initial_reset_anomaly",
            measured={"explicit": True},
            config=config,
        )
    initial_steps = _configured_integer(config, spec, "initial_steps", minimum=1)
    if initial_steps is None:
        return _detector_error(spec, row, "invalid_initial_reset_parameters", config=config)
    trace_frames, trace_reason = _trace(row)
    if trace_reason == "trace_frame_malformed":
        return _detector_error(spec, row, trace_reason, config=config)
    if trace_frames:
        for index, frame in enumerate(trace_frames[:initial_steps]):
            collision_value = frame.get("collision")
            if collision_value is None:
                robot = frame.get("robot")
                collision_value = robot.get("collision") if isinstance(robot, Mapping) else None
            if collision_value is not None and not isinstance(collision_value, bool):
                return _detector_error(spec, row, "initial_collision_flag_malformed", config=config)
            if collision_value is True:
                return _make_signal(
                    spec,
                    row,
                    "flagged",
                    reason="initial_collision",
                    measured={"trace_step": index},
                    config=config,
                )
    initial = row.get("initial_state")
    if initial is None:
        # campaign-result-store.v2 carries reset geometry on the first
        # canonical analysis-trace step when a separate initial_state block
        # was not retained.  This is a direct projection, not an inferred
        # radius or position.
        canonical_frames, canonical_reason = _trace(row)
        if canonical_frames:
            first = canonical_frames[0]
            initial = {
                "robot": first.get("robot"),
                "pedestrians": first.get("pedestrians", first.get("agents")),
            }
        elif canonical_reason not in {"trace_not_recorded", "trace_empty"}:
            return _unavailable(
                spec,
                row,
                canonical_reason or "initial_state_not_recorded",
                missing=("initial_state",),
                config=config,
            )
    if isinstance(initial, Mapping):
        for key in ("collision", "collision_event", "spawn_overlap", "reset_mismatch"):
            if initial.get(key) is True:
                return _make_signal(
                    spec,
                    row,
                    "flagged",
                    reason="initial_reset_anomaly",
                    measured={key: True},
                    config=config,
                )
        robot = initial.get("robot")
        if "pedestrians" in initial:
            actors = initial["pedestrians"]
        else:
            actors = initial.get("agents")
        minimum_separation = _configured_number(config, spec, "minimum_separation_m", minimum=0.0)
        if minimum_separation is None:
            return _detector_error(spec, row, "invalid_initial_reset_parameters", config=config)
        if robot is not None and not isinstance(robot, Mapping):
            return _detector_error(spec, row, "initial_robot_state_malformed", config=config)
        if robot is None:
            return _unavailable(
                spec,
                row,
                "initial_robot_state_unavailable",
                missing=("initial_state.robot",),
                config=config,
            )
        if isinstance(robot, Mapping) and (
            "position" not in robot or robot.get("position") is None
        ):
            return _unavailable(
                spec,
                row,
                "initial_robot_position_unavailable",
                missing=("initial_state.robot.position",),
                config=config,
            )
        if actors is None and not any(
            initial.get(name) == 0 for name in ("actor_count", "pedestrian_count", "agent_count")
        ):
            return _unavailable(
                spec,
                row,
                "initial_actor_state_unavailable",
                missing=("initial_state.pedestrians",),
                config=config,
            )
        if actors is not None and not isinstance(actors, list):
            return _detector_error(spec, row, "initial_actor_state_malformed", config=config)
        if isinstance(robot, Mapping) and isinstance(actors, list):
            robot_pos = _optional_vector(robot.get("position"))
            if robot_pos is None:
                return _unavailable(
                    spec,
                    row,
                    "initial_robot_position_unavailable",
                    missing=("initial_state.robot.position",),
                    config=config,
                )
            robot_radius_value = robot.get("radius_m", robot.get("radius"))
            robot_radius = _finite_or_none(robot_radius_value)
            if robot_radius is None:
                if robot_radius_value is not None:
                    return _detector_error(
                        spec, row, "initial_robot_radius_malformed", config=config
                    )
                return _unavailable(
                    spec,
                    row,
                    "initial_robot_radius_unavailable",
                    missing=("initial_state.robot.radius_m",),
                    config=config,
                )
            if robot_radius < 0.0:
                return _detector_error(
                    spec, row, "initial_robot_radius_out_of_range", config=config
                )
            for index, actor in enumerate(actors):
                if not isinstance(actor, Mapping):
                    return _detector_error(
                        spec, row, "initial_actor_state_malformed", config=config
                    )
                actor_pos = _optional_vector(actor.get("position"))
                if actor_pos is None:
                    return _unavailable(
                        spec,
                        row,
                        f"initial_actor_position_unavailable_{index}",
                        missing=(f"initial_state.pedestrians[{index}].position",),
                        config=config,
                    )
                actor_radius_value = actor.get("radius_m", actor.get("radius"))
                actor_radius = _finite_or_none(actor_radius_value)
                if actor_radius is None:
                    if actor_radius_value is not None:
                        return _detector_error(
                            spec, row, "initial_actor_radius_malformed", config=config
                        )
                    return _unavailable(
                        spec,
                        row,
                        f"initial_actor_radius_unavailable_{index}",
                        missing=(f"initial_state.pedestrians[{index}].radius_m",),
                        config=config,
                    )
                if actor_radius < 0.0:
                    return _detector_error(
                        spec, row, "initial_actor_radius_out_of_range", config=config
                    )
                distance = math.hypot(robot_pos[0] - actor_pos[0], robot_pos[1] - actor_pos[1])
                threshold_value = robot_radius + actor_radius + minimum_separation
                if distance <= threshold_value:
                    return _make_signal(
                        spec,
                        row,
                        "flagged",
                        reason="spawn_overlap",
                        measured={
                            "actor_index": index,
                            "separation_m": distance,
                            "minimum_separation_m": minimum_separation,
                        },
                        threshold={
                            "operator": "<=",
                            "value": threshold_value,
                            "minimum_separation_m": minimum_separation,
                            "unit": "m",
                        },
                        config=config,
                    )
        return _make_signal(spec, row, "clear", reason="initial_state_valid", config=config)
    return _unavailable(
        spec,
        row,
        "initial_state_not_recorded",
        missing=("initial_state",),
        config=config,
    )


def _stuck_no_progress(  # noqa: C901
    spec: DetectorSpec,
    row: Mapping[str, Any],
    cohort: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> Signal:
    expected_wait = row.get("expected_waiting") is True or row.get("expected_yielding") is True
    if expected_wait:
        return _make_signal(
            spec, row, "clear", reason="expected_waiting_or_yielding", config=config
        )
    progress_value = _lookup(
        row,
        "progress_m",
        "route_progress_m",
        "route_progress",
        "goal_progress_m",
        "goal_progress",
    )
    duration_value = _lookup(
        row,
        "duration_s",
        "episode_duration_s",
        "time_to_goal_s",
        "time_to_goal",
    )
    speed_value = _lookup(
        row,
        "mean_speed_m_s",
        "speed_m_s",
        "average_speed_m_s",
        "avg_speed_m_s",
        "avg_speed",
        "average_speed",
    )
    progress = _finite_or_none(progress_value)
    duration = _finite_or_none(duration_value)
    speed = _finite_or_none(speed_value)
    if (
        (progress_value is not None and progress is None)
        or (duration_value is not None and duration is None)
        or (speed_value is not None and speed is None)
    ):
        return _detector_error(spec, row, "progress_telemetry_malformed", config=config)
    frames: list[Mapping[str, Any]] | None = None
    interval = None
    if duration is None:
        try:
            _positions, times, frames = _trace_positions(row)
            duration = times[-1] - times[0]
            interval = _trace_interval(frames)
            if speed is None and len(times) > 1 and duration > 0:
                # This is a speed estimate from recorded motion, not route or
                # goal progress.  Never substitute endpoint displacement for a
                # promised progress metric.
                positions = [_position(frame) for frame in frames]
                if all(position is not None for position in positions):
                    speed = (
                        sum(
                            math.hypot(right[0] - left[0], right[1] - left[1])
                            for left, right in pairwise(positions)
                        )
                        / duration
                    )
        except DetectorError as error:
            return _trace_failure(spec, row, error, config=config)
    if progress is None or duration is None:
        missing: list[str] = []
        if progress is None:
            missing.append("progress_m")
        if duration is None:
            missing.append("duration_s")
        return _unavailable(
            spec,
            row,
            "progress_telemetry_unavailable",
            missing=tuple(missing),
            config=config,
        )
    if duration < 0.0 or (speed is not None and speed < 0.0):
        return _detector_error(spec, row, "progress_telemetry_out_of_range", config=config)
    min_duration = _configured_number(config, spec, "minimum_duration_s", minimum=0.0)
    tolerance = _configured_number(config, spec, "progress_tolerance_m", minimum=0.0)
    speed_tolerance = _configured_number(config, spec, "speed_tolerance_m_s", minimum=0.0)
    if min_duration is None or tolerance is None or speed_tolerance is None:
        return _detector_error(spec, row, "invalid_stuck_parameters", config=config)
    flagged = (
        duration >= min_duration
        and progress <= tolerance
        and (speed is None or speed <= speed_tolerance)
    )
    return _make_signal(
        spec,
        row,
        "flagged" if flagged else "clear",
        reason="no_progress" if flagged else "progress_observed",
        measured={"progress_m": progress, "duration_s": duration, "mean_speed_m_s": speed},
        threshold={
            "progress_m": {"operator": "<=", "value": tolerance, "unit": "m"},
            "duration_s": {"operator": ">=", "value": min_duration, "unit": "s"},
        },
        interval=interval,
        config=config,
    )


def _oscillation(  # noqa: C901
    spec: DetectorSpec,
    row: Mapping[str, Any],
    cohort: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> Signal:
    try:
        positions, _times, frames = _trace_positions(row)
    except DetectorError as error:
        return _trace_failure(spec, row, error, config=config)
    angular: list[float] = []
    headings: list[float] = []
    for frame in frames:
        controls = frame.get("controls")
        canonical_turn: Any = None
        if isinstance(controls, Mapping):
            for control_key in ("requested", "applied"):
                control = controls.get(control_key)
                if isinstance(control, Mapping):
                    canonical_turn = control.get("turn_rate_rad_s")
                    if canonical_turn is None:
                        canonical_turn = control.get("angular_velocity")
                    if canonical_turn is not None:
                        break
        if canonical_turn is not None:
            angular.append(_finite(canonical_turn, name="controls.turn_rate_rad_s"))
            continue
        planner = frame.get("planner")
        action = planner.get("selected_action") if isinstance(planner, Mapping) else None
        if not isinstance(action, Mapping):
            action = frame.get("action")
        if isinstance(action, Mapping) and "angular_velocity" in action:
            angular.append(_finite(action["angular_velocity"], name="angular_velocity"))
        robot = frame.get("robot")
        if isinstance(robot, Mapping):
            heading = robot.get("heading", robot.get("heading_rad"))
            if heading is not None:
                headings.append(_finite(heading, name="heading"))
    values = (
        angular
        if len(angular) >= 2
        else [
            _wrapped_angle_delta(headings[index + 1], headings[index])
            for index in range(len(headings) - 1)
        ]
    )
    if not values:
        return _unavailable(
            spec,
            row,
            "turning_telemetry_unavailable",
            missing=("trace.heading", "commands.angular_velocity"),
            config=config,
        )
    reversal_threshold = _configured_number(
        config, spec, "heading_reversal_rad", minimum=0.0, strictly_greater=True
    )
    if reversal_threshold is None:
        return _detector_error(spec, row, "invalid_oscillation_parameters", config=config)
    nonzero = [value for value in values if abs(value) >= reversal_threshold]
    reversals = sum(1 for left, right in pairwise(nonzero) if left * right < 0)
    displacement = math.hypot(
        positions[-1][0] - positions[0][0], positions[-1][1] - positions[0][1]
    )
    minimum_reversals = _configured_integer(config, spec, "minimum_reversals", minimum=1)
    loop_radius = _configured_number(config, spec, "loop_radius_m", minimum=0.0)
    if loop_radius is None or minimum_reversals is None:
        return _detector_error(spec, row, "invalid_oscillation_parameters", config=config)
    flagged = reversals >= minimum_reversals and displacement <= loop_radius
    return _make_signal(
        spec,
        row,
        "flagged" if flagged else "clear",
        reason="repeated_turning_limit_cycle" if flagged else "no_repeated_turning_limit_cycle",
        measured={"sign_reversals": reversals, "endpoint_displacement_m": displacement},
        threshold={
            "sign_reversals": {"operator": ">=", "value": minimum_reversals, "unit": "count"},
            "endpoint_displacement_m": {"operator": "<=", "value": loop_radius, "unit": "m"},
        },
        evidence=({"interpretation": "trajectory pattern only; no infeasibility or causal claim"},),
        interval=_trace_interval(frames),
        config=config,
    )


def _termination_exclusivity(  # noqa: C901, PLR0912
    contract: Mapping[str, Any],
) -> tuple[bool | None, bool | None, bool | None] | None:
    """Read pairwise exclusivity from a typed campaign termination contract.

    A row-level ``termination_contract`` is not authoritative merely because
    it is an object.  The contract must identify its schema/version and carry
    explicit boolean exclusivity semantics.  ``None`` means that the source
    cannot establish the predicate and callers must return ``unavailable``.

    Returns:
        Pairwise ``(success/timeout, success/collision, collision/timeout)``
        exclusivity, with ``None`` for a pair the contract does not define.
    """

    versions = [contract[name] for name in ("schema_version", "version") if name in contract]
    if not versions:
        return None
    if any(not isinstance(value, str) or not value.strip() for value in versions):
        raise DetectorError("termination_contract.schema_version_malformed")
    if len({value.strip() for value in versions}) != 1:
        raise DetectorError("termination_contract.schema_version_conflict")
    exclusivity_marker = next(
        (
            name
            for name in (
                "mutually_exclusive",
                "outcomes_mutually_exclusive",
                "outcome_flags_mutually_exclusive",
                "exclusive",
            )
            if name in contract
        ),
        None,
    )
    if exclusivity_marker is not None:
        value = contract[exclusivity_marker]
        if not isinstance(value, bool):
            raise DetectorError(f"termination_contract.{exclusivity_marker}_malformed")
        for marker in (
            "mutually_exclusive",
            "outcomes_mutually_exclusive",
            "outcome_flags_mutually_exclusive",
            "exclusive",
        ):
            if marker in contract and contract[marker] != value:
                raise DetectorError("termination_contract.exclusivity_conflict")
        if value:
            return True, True, True
        return False, False, False
    exclusivity = contract.get("exclusivity")
    if exclusivity is not None:
        if not isinstance(exclusivity, Mapping):
            raise DetectorError("termination_contract.exclusivity_malformed")
        aliases = {
            "success_timeout": (
                "success_timeout",
                "success_and_timeout",
                "success_and_timeout_valid",
            ),
            "success_collision": (
                "success_collision",
                "success_and_collision",
                "success_and_collision_valid",
            ),
            "collision_timeout": (
                "collision_timeout",
                "collision_and_timeout",
                "collision_and_timeout_valid",
            ),
        }
        values: list[bool | None] = []
        for names in aliases.values():
            present = [name for name in names if name in exclusivity]
            if not present:
                values.append(None)
                continue
            normalized_values: list[bool] = []
            for name in present:
                value = exclusivity[name]
                if not isinstance(value, bool):
                    raise DetectorError(f"termination_contract.exclusivity.{name}_malformed")
                normalized_values.append(not value if name.endswith("_valid") else value)
            if len(set(normalized_values)) > 1:
                raise DetectorError("termination_contract.exclusivity_conflict")
            values.append(normalized_values[0])
        if not any(name in exclusivity for names in aliases.values() for name in names):
            return None
        return values[0], values[1], values[2]
    explicit_names = (
        "success_and_timeout_valid",
        "success_and_collision_valid",
        "collision_and_timeout_valid",
    )
    present = [name for name in explicit_names if name in contract]
    if not present:
        return None
    values: list[bool | None] = []
    for name in explicit_names:
        if name not in contract:
            values.append(None)
            continue
        value = contract[name]
        if not isinstance(value, bool):
            raise DetectorError(f"termination_contract.{name}_malformed")
        values.append(not value)
    return values[0], values[1], values[2]


def _contradiction(  # noqa: C901, PLR0912
    spec: DetectorSpec,
    row: Mapping[str, Any],
    cohort: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> Signal:
    success = _success(row)
    collision = _collision(row)
    timeout = _timeout(row)
    contract = row.get("termination_contract")
    if contract is None:
        return _unavailable(
            spec,
            row,
            "termination_contract_unavailable",
            missing=("termination_contract",),
            config=config,
        )
    if not isinstance(contract, Mapping):
        return _detector_error(spec, row, "termination_contract_malformed", config=config)
    try:
        exclusivity = _termination_exclusivity(contract)
    except DetectorError as error:
        return _detector_error(spec, row, str(error), config=config)
    if exclusivity is None:
        return _unavailable(
            spec,
            row,
            "termination_contract_exclusivity_unavailable",
            missing=("termination_contract.exclusivity",),
            config=config,
        )
    outcome = _outcome(row)
    for key in (
        "success",
        "route_complete",
        "reached_goal",
        "completed",
        "collision_event",
        "collision",
        "collided",
        "timeout_event",
        "timeout",
        "timed_out",
        "horizon_reached",
    ):
        for container in (row, outcome):
            if key in container and not isinstance(container[key], bool):
                return _detector_error(spec, row, f"outcome_{key}_malformed", config=config)
    if success is None and collision is None and timeout is None:
        return _unavailable(
            spec, row, "outcome_telemetry_unavailable", missing=("outcome",), config=config
        )
    contradictions: list[str] = []
    success_timeout_exclusive, success_collision_exclusive, collision_timeout_exclusive = (
        exclusivity
    )
    unknown_pairs: list[str] = []
    if success is True and timeout is True and success_timeout_exclusive is None:
        unknown_pairs.append("success_timeout")
    if success is True and collision is True and success_collision_exclusive is None:
        unknown_pairs.append("success_collision")
    if collision is True and timeout is True and collision_timeout_exclusive is None:
        unknown_pairs.append("collision_timeout")
    if unknown_pairs:
        return _unavailable(
            spec,
            row,
            "termination_contract_pair_exclusivity_unavailable",
            missing=tuple(f"termination_contract.exclusivity.{item}" for item in unknown_pairs),
            config=config,
        )
    if success is True and timeout is True and success_timeout_exclusive:
        contradictions.append("success_and_timeout")
    if success is True and collision is True and success_collision_exclusive:
        contradictions.append("success_and_collision")
    if collision is True and timeout is True and collision_timeout_exclusive:
        contradictions.append("collision_and_timeout")
    collision_count_value = _lookup(row, "collision_count", "collisions")
    if collision_count_value is None:
        for key in ("collision_count", "collisions"):
            if key in outcome:
                collision_count_value = outcome[key]
                break
    collision_count = _finite_or_none(collision_count_value)
    if collision_count_value is not None and collision_count is None:
        return _detector_error(spec, row, "collision_count_malformed", config=config)
    duration_value = _lookup(row, "duration_s", "episode_duration_s")
    if duration_value is None:
        for key in ("duration_s", "episode_duration_s", "duration"):
            if key in outcome:
                duration_value = outcome[key]
                break
    duration = _finite_or_none(duration_value)
    if duration_value is not None and duration is None:
        return _detector_error(spec, row, "duration_malformed", config=config)
    if collision_count is not None and collision_count < 0.0:
        return _detector_error(spec, row, "collision_count_out_of_range", config=config)
    if duration is not None and duration < 0.0:
        return _detector_error(spec, row, "duration_out_of_range", config=config)
    collision_count_max = _configured_number(
        config, spec, "collision_count_max_for_success", minimum=0.0
    )
    if collision_count_max is None:
        return _detector_error(spec, row, "invalid_outcome_parameters", config=config)
    if success is True and collision_count is not None and collision_count > collision_count_max:
        contradictions.append("success_with_collision_count")
    flagged = bool(contradictions)
    return _make_signal(
        spec,
        row,
        "flagged" if flagged else "clear",
        reason="outcome_metric_contradiction" if flagged else "outcome_metric_consistent",
        measured={
            "success": success,
            "collision": collision,
            "timeout": timeout,
            "collision_count": collision_count,
            "duration_s": duration,
            "contradictions": contradictions,
        },
        threshold={
            "termination_contract": "typed_explicit_exclusivity",
            "exclusivity": {
                "success_timeout": success_timeout_exclusive,
                "success_collision": success_collision_exclusive,
                "collision_timeout": collision_timeout_exclusive,
            },
            "collision_count_max_for_success": collision_count_max,
        },
        evidence=(
            {
                "termination_contract": dict(contract),
                "exclusivity": {
                    "success_timeout": success_timeout_exclusive,
                    "success_collision": success_collision_exclusive,
                    "collision_timeout": collision_timeout_exclusive,
                },
            },
        ),
        config=config,
    )


def _extreme(  # noqa: C901, PLR0912
    spec: DetectorSpec,
    row: Mapping[str, Any],
    cohort: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> Signal:
    metrics = _metrics(row)
    values: dict[str, float] = {}
    invalid: list[str] = []
    for name, value in metrics.items():
        if not isinstance(name, str):
            continue
        if value is None:
            continue
        if isinstance(value, bool):
            # Outcome booleans are often retained in the metrics block for
            # compatibility.  They are not scalar physical measurements.
            continue
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            invalid.append(name)
            continue
        values[name] = float(value)
    if not values and not invalid:
        return _unavailable(
            spec, row, "measurements_not_recorded", missing=("metrics",), config=config
        )
    if invalid:
        return _detector_error(
            spec,
            row,
            "nonfinite_or_malformed_measurement",
            missing=tuple(f"metrics.{name}" for name in invalid),
            config=config,
        )
    extreme: dict[str, float] = {}
    clearance_min = _configured_number(config, spec, "clearance_min_m", minimum=0.0)
    force_max = _configured_number(config, spec, "force_max_N", minimum=0.0, strictly_greater=True)
    ttc_min = _configured_number(config, spec, "ttc_min_s", minimum=0.0)
    collision_max = _configured_number(config, spec, "collision_max", minimum=0.0)
    if any(value is None for value in (clearance_min, force_max, ttc_min, collision_max)):
        return _detector_error(spec, row, "invalid_extreme_measurement_parameters", config=config)
    for name, value in values.items():
        lowered = name.lower()
        if "clearance" in lowered or "distance" in lowered:
            if value < clearance_min:
                extreme[name] = value
        elif "force" in lowered:
            if abs(value) > force_max:
                extreme[name] = value
        elif lowered in {"ttc", "ttc_s", "time_to_collision_s", "pet_s"}:
            # Explicitly undefined TTC/PET is represented by a missing key or
            # null, and therefore never becomes a corrupted measurement.
            if value < ttc_min:
                extreme[name] = value
        elif "collision" in lowered and (value < 0.0 or value > collision_max):
            extreme[name] = value
    return _make_signal(
        spec,
        row,
        "flagged" if extreme else "clear",
        reason="extreme_measurement" if extreme else "measurements_within_declared_bounds",
        measured={
            "values": values,
            "extreme": extreme,
            "missing_undefined_values": [
                name for name in ("ttc_s", "pet_s") if name not in metrics
            ],
        },
        threshold={
            "clearance_min_m": clearance_min,
            "force_max_N": force_max,
            "ttc_min_s": ttc_min,
            "collision_max": collision_max,
        },
        config=config,
    )


def _actions(  # noqa: C901
    row: Mapping[str, Any],
) -> list[tuple[Mapping[str, Any], Mapping[str, Any] | None]]:
    frames, _reason = _trace(row)
    pairs: list[tuple[Mapping[str, Any], Mapping[str, Any] | None]] = []
    if frames is not None:
        for frame in frames:
            controls = frame.get("controls")
            if isinstance(controls, Mapping):
                desired = controls.get("requested")
                applied = controls.get("applied")
                if isinstance(desired, Mapping):
                    pairs.append((desired, applied if isinstance(applied, Mapping) else None))
            planner = frame.get("planner")
            desired = planner.get("desired_action") if isinstance(planner, Mapping) else None
            if desired is None and isinstance(planner, Mapping):
                desired = planner.get("commanded_action")
            applied = planner.get("executed_action") if isinstance(planner, Mapping) else None
            if applied is None and isinstance(planner, Mapping):
                applied = planner.get("applied_action")
            selected = planner.get("selected_action") if isinstance(planner, Mapping) else None
            if desired is None:
                desired = selected
            if isinstance(desired, Mapping):
                pairs.append((desired, applied if isinstance(applied, Mapping) else None))
            elif isinstance(frame.get("desired_action"), Mapping):
                desired = frame["desired_action"]
                applied = frame.get("executed_action") or frame.get("applied_action")
                pairs.append((desired, applied if isinstance(applied, Mapping) else None))
    for value in (row.get("commands"), row.get("control")):
        if isinstance(value, list):
            for item in value:
                if isinstance(item, Mapping):
                    desired = (
                        item.get("desired") or item.get("commanded") or item.get("desired_action")
                    )
                    applied = (
                        item.get("executed") or item.get("applied") or item.get("executed_action")
                    )
                    if isinstance(desired, Mapping):
                        pairs.append((desired, applied if isinstance(applied, Mapping) else None))
    desired = row.get("desired_action") or row.get("commanded_action")
    applied = row.get("executed_action") or row.get("applied_action")
    if isinstance(desired, Mapping):
        pairs.append((desired, applied if isinstance(applied, Mapping) else None))
    return pairs


def _action_shape_error(row: Mapping[str, Any]) -> str | None:  # noqa: C901, PLR0912
    """Return a stable reason when a promised command stream is malformed."""

    for key in ("commands", "control"):
        value = row.get(key)
        if value is not None and not isinstance(value, list):
            return f"{key}_malformed"
        if isinstance(value, list):
            for item in value:
                if not isinstance(item, Mapping):
                    return f"{key}_item_malformed"
                for action_key in (
                    "desired",
                    "commanded",
                    "desired_action",
                    "executed",
                    "applied",
                    "executed_action",
                ):
                    if (
                        action_key in item
                        and item[action_key] is not None
                        and not isinstance(item[action_key], Mapping)
                    ):
                        return f"{key}.{action_key}_malformed"
    for key in ("desired_action", "commanded_action", "executed_action", "applied_action"):
        if key in row and row[key] is not None and not isinstance(row[key], Mapping):
            return f"{key}_malformed"
    frames, reason = _trace(row)
    if reason == "trace_frame_malformed":
        return reason
    for frame in frames or ():
        controls = frame.get("controls")
        if controls is not None and not isinstance(controls, Mapping):
            return "trace.controls_malformed"
        if isinstance(controls, Mapping):
            for control_key in ("requested", "applied"):
                value = controls.get(control_key)
                if value is not None and not isinstance(value, Mapping):
                    return f"trace.controls.{control_key}_malformed"
        planner = frame.get("planner")
        if planner is not None and not isinstance(planner, Mapping):
            return "trace.planner_malformed"
        if isinstance(planner, Mapping):
            for action_key in (
                "desired_action",
                "commanded_action",
                "executed_action",
                "applied_action",
                "selected_action",
            ):
                if (
                    action_key in planner
                    and planner[action_key] is not None
                    and not isinstance(planner[action_key], Mapping)
                ):
                    return f"trace.planner.{action_key}_malformed"
    return None


def _actuator(  # noqa: C901, PLR0912
    spec: DetectorSpec,
    row: Mapping[str, Any],
    cohort: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> Signal:
    shape_error = _action_shape_error(row)
    if shape_error is not None:
        return _detector_error(spec, row, shape_error, config=config)
    _frames, trace_reason = _trace(row)
    if trace_reason == "trace_frame_malformed":
        return _detector_error(spec, row, trace_reason, config=config)
    if _frames:
        try:
            _trace_positions(row)
        except DetectorError as error:
            return _trace_failure(spec, row, error, config=config)
    pairs = _actions(row)
    if not pairs:
        if row.get("requires_commands") is True:
            return _unavailable(
                spec, row, "command_telemetry_unavailable", missing=("commands",), config=config
            )
        return _unavailable(
            spec, row, "command_telemetry_not_recorded", missing=("commands",), config=config
        )
    mismatches: list[dict[str, Any]] = []
    compared_samples = 0
    max_linear = _finite_or_none(
        _lookup(row, "max_linear_velocity_m_s", "linear_velocity_limit_m_s")
    )
    max_angular = _finite_or_none(
        _lookup(row, "max_angular_velocity_rad_s", "angular_velocity_limit_rad_s")
    )
    for name, raw, parsed in (
        (
            "max_linear_velocity_m_s",
            _lookup(row, "max_linear_velocity_m_s", "linear_velocity_limit_m_s"),
            max_linear,
        ),
        (
            "max_angular_velocity_rad_s",
            _lookup(row, "max_angular_velocity_rad_s", "angular_velocity_limit_rad_s"),
            max_angular,
        ),
    ):
        if raw is not None and parsed is None:
            return _detector_error(spec, row, f"{name}_malformed", config=config)
        if parsed is not None and parsed <= 0.0:
            return _detector_error(spec, row, f"{name}_out_of_range", config=config)
    tolerance = _configured_number(config, spec, "absolute_tolerance", minimum=0.0)
    relative = _configured_number(config, spec, "relative_tolerance", minimum=0.0)
    if tolerance is None or relative is None:
        return _detector_error(spec, row, "invalid_actuator_parameters", config=config)
    action_aliases = {
        "linear_velocity": ("linear_velocity", "linear_m_s"),
        "angular_velocity": ("angular_velocity", "turn_rate_rad_s"),
    }
    for index, (desired, applied) in enumerate(pairs):
        for key, aliases in action_aliases.items():
            desired_key = next((alias for alias in aliases if alias in desired), None)
            if desired_key is None:
                continue
            desired_value = _finite(desired[desired_key], name=f"desired.{desired_key}")
            applied_key = (
                next((alias for alias in aliases if alias in applied), None)
                if applied is not None
                else None
            )
            if applied_key is not None and applied is not None:
                compared_samples += 1
                applied_value = _finite(applied[applied_key], name=f"applied.{applied_key}")
                delta = abs(desired_value - applied_value)
                bound = tolerance + relative * abs(desired_value)
                if delta > bound:
                    mismatches.append(
                        {
                            "index": index,
                            "field": key,
                            "desired": desired_value,
                            "applied": applied_value,
                            "delta": delta,
                        }
                    )
            limit = max_linear if key == "linear_velocity" else max_angular
            if limit is not None and abs(desired_value) >= limit - tolerance:
                mismatches.append(
                    {"index": index, "field": key, "desired": desired_value, "saturated_at": limit}
                )
    if compared_samples == 0 and not mismatches:
        return _unavailable(
            spec,
            row,
            "executed_command_telemetry_unavailable",
            missing=("commands.executed",),
            config=config,
        )
    return _make_signal(
        spec,
        row,
        "flagged" if mismatches else "clear",
        reason="actuator_saturation_or_mismatch" if mismatches else "command_execution_consistent",
        measured={"samples": len(pairs), "mismatches": mismatches},
        threshold={"absolute_tolerance": tolerance, "relative_tolerance": relative},
        interval=_trace_interval(_frames) if _frames else None,
        config=config,
    )


def _telemetry(  # noqa: C901, PLR0912
    spec: DetectorSpec,
    row: Mapping[str, Any],
    cohort: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> Signal:
    trace_frames, trace_reason = _trace(row)
    if trace_reason in {"trace_frame_malformed", "trace_coverage_malformed"}:
        return _detector_error(spec, row, trace_reason, config=config)
    if trace_frames:
        try:
            _trace_positions(row)
        except DetectorError as error:
            return _trace_failure(spec, row, error, config=config)
    trace_interval = _trace_interval(trace_frames) if trace_frames else None
    contract = row.get("telemetry_contract")
    required: list[str] = []
    if contract is not None and not isinstance(contract, Mapping):
        return _detector_error(spec, row, "telemetry_contract_malformed", config=config)
    if isinstance(contract, Mapping):
        value = contract.get("required_fields", [])
        if not isinstance(value, list) or any(
            not isinstance(item, str) or not item.strip() for item in value
        ):
            return _detector_error(
                spec, row, "telemetry_contract_required_fields_malformed", config=config
            )
        required.extend(item.strip() for item in value)
    value = row.get("required_telemetry")
    if value is not None and (
        not isinstance(value, list)
        or any(not isinstance(item, str) or not item.strip() for item in value)
    ):
        return _detector_error(spec, row, "required_telemetry_malformed", config=config)
    if isinstance(value, list):
        required.extend(item.strip() for item in value)
    configured = config.get("required_fields", spec.parameters["required_fields"])
    if not isinstance(configured, list) or any(
        not isinstance(item, str) or not item.strip() for item in configured
    ):
        return _detector_error(spec, row, "required_fields_malformed", config=config)
    required.extend(item.strip() for item in configured)
    required = sorted(set(required))
    missing: list[str] = []
    nonfinite: list[str] = []
    for field_name in required:
        current: Any = row
        for part in field_name.split("."):
            if not isinstance(current, Mapping) or part not in current:
                current = None
                break
            current = current[part]
        if _promised_missing(current):
            missing.append(field_name)
        elif _contains_nonfinite(current):
            nonfinite.append(field_name)
    if nonfinite:
        return _detector_error(
            spec, row, "promised_telemetry_nonfinite", missing=tuple(nonfinite), config=config
        )
    if _contains_nonfinite(row):
        return _detector_error(spec, row, "nonfinite_telemetry_payload", config=config)
    if missing:
        return _make_signal(
            spec,
            row,
            "flagged",
            reason="promised_telemetry_missing",
            measured={"missing_fields": missing},
            threshold={"required_fields": required},
            missingness=tuple(missing),
            interval=trace_interval,
            config=config,
        )
    if trace_reason in {"trace_empty", "trace_coverage_unavailable"}:
        return _unavailable(
            spec,
            row,
            trace_reason,
            missing=("trace",),
            config=config,
        )
    if not required and not _metrics(row) and not trace_frames:
        return _unavailable(
            spec, row, "optional_telemetry_not_recorded", missing=("telemetry",), config=config
        )
    return _make_signal(
        spec,
        row,
        "clear",
        reason="telemetry_integrity_verified",
        measured={"required_fields": required},
        interval=trace_interval,
        config=config,
    )


def _provenance(  # noqa: C901
    spec: DetectorSpec,
    row: Mapping[str, Any],
    cohort: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> Signal:
    if any(key not in row for key in ("campaign_id", "source_commit", "config_identity")):
        row = dict(row)
        row.setdefault("campaign_id", row.get("study_id") or row.get("campaign"))
        row.setdefault("source_commit", row.get("git_hash"))
        row.setdefault("config_identity", row.get("config_hash"))
    provenance = row.get("provenance")
    if not isinstance(provenance, Mapping):
        provenance = {}
    required = config.get("required_identity_fields", spec.parameters["required_identity_fields"])
    if not isinstance(required, list) or any(
        not isinstance(item, str) or not item.strip() for item in required
    ):
        return _detector_error(spec, row, "required_identity_fields_malformed", config=config)
    required = [item.strip() for item in required]
    if not provenance and not any(row.get(field_name) for field_name in required):
        if row.get("requires_provenance") is not True:
            return _unavailable(
                spec, row, "provenance_not_recorded", missing=("provenance",), config=config
            )
    missing = [
        field_name
        for field_name in required
        if not isinstance(provenance.get(field_name) or row.get(field_name), str)
        or not str(provenance.get(field_name) or row.get(field_name)).strip()
    ]
    mismatches: list[str] = []
    expected = row.get("expected_provenance")
    if expected is not None and not isinstance(expected, Mapping):
        return _detector_error(spec, row, "expected_provenance_malformed", config=config)
    if isinstance(expected, Mapping):
        for key, expected_value in expected.items():
            actual = provenance.get(key, row.get(key))
            if actual != expected_value:
                mismatches.append(str(key))
    if missing:
        return _make_signal(
            spec,
            row,
            "flagged",
            reason="provenance_identity_missing",
            measured={"missing_fields": missing},
            threshold={"required_fields": required},
            missingness=tuple(missing),
            config=config,
        )
    if mismatches:
        return _make_signal(
            spec,
            row,
            "flagged",
            reason="provenance_identity_mismatch",
            measured={"mismatched_fields": mismatches},
            config=config,
        )
    return _make_signal(
        spec,
        row,
        "clear",
        reason="provenance_consistent",
        measured={"identity_fields": required},
        config=config,
    )


def _common_mode(
    spec: DetectorSpec,
    row: Mapping[str, Any],
    cohort: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> Signal:
    peers = _group_cohort(row, cohort, spec)
    planners = {_identity(item, "planner_id") for item in peers}
    minimum_planners = _configured_integer(config, spec, "minimum_planners", minimum=1)
    threshold = _configured_number(config, spec, "failure_fraction", minimum=0.0)
    if minimum_planners is None or threshold is None or threshold > 1.0:
        return _detector_error(spec, row, "invalid_common_mode_parameters", config=config)
    if len(planners) < minimum_planners:
        return _unavailable(
            spec,
            row,
            "compatible_planner_cohort_too_small",
            missing=("cohort.planners",),
            config=config,
        )
    failures = [
        item
        for item in peers
        if (_success(item) is False or _collision(item) is True or _timeout(item) is True)
    ]
    fraction = len(failures) / len(peers) if peers else 0.0
    flagged = fraction >= threshold and len(planners) >= minimum_planners
    return _make_signal(
        spec,
        row,
        "flagged" if flagged else "clear",
        reason="common_mode_failure" if flagged else "no_common_mode_anomaly",
        measured={
            "cohort_size": len(peers),
            "planner_count": len(planners),
            "failure_fraction": fraction,
        },
        threshold={"failure_fraction": threshold},
        evidence=(
            {
                "cohort_definition": dict(spec.cohort_definition),
                "cohort_episode_ids": sorted(str(item.get("episode_id", "")) for item in peers),
            },
        ),
        config=config,
    )


def _planner_disagreement(
    spec: DetectorSpec,
    row: Mapping[str, Any],
    cohort: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> Signal:
    peers = _group_cohort(row, cohort, spec)
    planner = _identity(row, "planner_id")
    others = [item for item in peers if _identity(item, "planner_id") != planner]
    if not others:
        return _unavailable(
            spec, row, "compatible_peer_unavailable", missing=("cohort.peers",), config=config
        )
    current_outcome = _outcome_label(row)
    peer_outcomes = [_outcome_label(item) for item in others]
    outcome_disagreement = current_outcome is not None and any(
        value is not None and value != current_outcome for value in peer_outcomes
    )
    metric_diffs: dict[str, float] = {}
    metric_z_scores: dict[str, float] = {}
    metric_z_threshold = _finite_or_none(
        config.get("metric_z_threshold", spec.parameters["metric_z_threshold"])
    )
    if metric_z_threshold is None or metric_z_threshold < 0:
        return _detector_error(spec, row, "invalid_planner_disagreement_parameters", config=config)
    metrics = _metrics(row)
    peer_metrics = [_metrics(item) for item in others]
    for name, value in metrics.items():
        numeric = _finite_or_none(value)
        peers_numeric = [_finite_or_none(item.get(name)) for item in peer_metrics]
        peers_numeric = [item for item in peers_numeric if item is not None]
        if numeric is not None and peers_numeric:
            diff = abs(numeric - median(peers_numeric))
            z_score = _robust_z(numeric, peers_numeric)
            if z_score >= metric_z_threshold:
                metric_diffs[name] = diff
                metric_z_scores[name] = z_score
    flagged = outcome_disagreement or bool(metric_diffs)
    return _make_signal(
        spec,
        row,
        "flagged" if flagged else "clear",
        reason="planner_disagreement" if flagged else "planner_agrees_with_peers",
        measured={
            "planner_id": planner,
            "peer_count": len(others),
            "outcome": current_outcome,
            "peer_outcomes": peer_outcomes,
            "metric_absolute_differences": metric_diffs,
            "metric_robust_z_scores": metric_z_scores,
        },
        evidence=(
            {
                "interpretation": "peer disagreement is a candidate signal, not proof of a planner defect"
            },
        ),
        config=config,
    )


def _robust_z(value: float, peers: Sequence[float]) -> float:
    center = median(peers)
    deviations = [abs(item - center) for item in peers]
    mad = median(deviations)
    if mad > 0:
        return abs(value - center) / (1.4826 * mad)
    # A zero-spread cohort makes any distinct value maximally unusual.  Keep
    # that representation finite because BA-03 records reject non-finite
    # measured values.
    return 1_000_000_000.0 if value != center else 0.0


def _seed_outlier(
    spec: DetectorSpec,
    row: Mapping[str, Any],
    cohort: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> Signal:
    peers = _group_cohort(row, cohort, spec)
    minimum = _configured_integer(config, spec, "minimum_cohort", minimum=2)
    threshold = _configured_number(config, spec, "mad_z_threshold", minimum=0.0)
    if minimum is None or threshold is None:
        return _detector_error(spec, row, "invalid_seed_outlier_parameters", config=config)
    if len(peers) < minimum:
        return _unavailable(
            spec, row, "too_small_cohort", missing=("cohort.minimum",), config=config
        )
    metrics = _metrics(row)
    candidates: list[tuple[str, float, list[float]]] = []
    for name, value in metrics.items():
        numeric = _finite_or_none(value)
        if numeric is None:
            continue
        peer_values = [
            _finite_or_none(_metrics(item).get(name))
            for item in peers
            if item.get("episode_id") != row.get("episode_id")
        ]
        peer_values = [item for item in peer_values if item is not None]
        if len(peer_values) >= minimum - 1:
            candidates.append((name, numeric, peer_values))
    if not candidates:
        return _unavailable(spec, row, "cohort_metric_missing", missing=("metrics",), config=config)
    scores = {name: _robust_z(value, peers) for name, value, peers in candidates}
    flagged = any(score >= threshold for score in scores.values())
    return _make_signal(
        spec,
        row,
        "flagged" if flagged else "clear",
        reason="seed_cohort_outlier" if flagged else "within_seed_cohort",
        measured={
            "seed": row.get("seed"),
            "feature_z_scores": scores,
            "cohort_size": len(peers),
            "seed_values_are_keys_only": True,
            "priority_score_uncalibrated": max(scores.values()) if scores else 0.0,
        },
        threshold={"mad_z_threshold": threshold},
        evidence=(
            {
                "features": sorted(scores),
                "scaling": "median_and_MAD_robust_z",
                "cohort": dict(spec.cohort_definition),
                "missingness": "features absent from an episode are excluded; small cohorts are unavailable",
            },
        ),
        config=config,
    )


def _multivariate(
    spec: DetectorSpec,
    row: Mapping[str, Any],
    cohort: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> Signal:
    peers = _group_cohort(row, cohort, spec)
    minimum = _configured_integer(config, spec, "minimum_cohort", minimum=2)
    threshold = _configured_number(config, spec, "mad_z_threshold", minimum=0.0)
    if minimum is None or threshold is None:
        return _detector_error(spec, row, "invalid_multivariate_parameters", config=config)
    if len(peers) < minimum:
        return _unavailable(
            spec, row, "too_small_cohort", missing=("cohort.minimum",), config=config
        )
    feature_names = config.get("features")
    if feature_names is not None and (
        not isinstance(feature_names, list)
        or any(not isinstance(name, str) or not name.strip() for name in feature_names)
    ):
        return _detector_error(spec, row, "features_malformed", config=config)
    if not feature_names:
        feature_names = sorted(
            {name for item in peers for name in _metrics(item) if isinstance(name, str)}
        )
    features: dict[str, float] = {}
    z_scores: dict[str, float] = {}
    missing: list[str] = []
    for name in feature_names:
        value = _finite_or_none(_metrics(row).get(name))
        peer_values = [_finite_or_none(_metrics(item).get(name)) for item in peers]
        peer_values = [item for item in peer_values if item is not None]
        if value is None or len(peer_values) < minimum - 1:
            missing.append(name)
            continue
        features[name] = value
        z_scores[name] = _robust_z(value, peer_values)
    if not z_scores:
        return _unavailable(
            spec,
            row,
            "cohort_features_unavailable",
            missing=tuple(missing) or ("metrics",),
            config=config,
        )
    score = math.sqrt(sum(value * value for value in z_scores.values()))
    flagged = any(value >= threshold for value in z_scores.values())
    return _make_signal(
        spec,
        row,
        "flagged" if flagged else "clear",
        reason="cohort_multivariate_outlier" if flagged else "within_cohort_feature_range",
        measured={
            "features": features,
            "robust_z_scores": z_scores,
            "priority_score_uncalibrated": score,
            "feature_missingness": missing,
            "cohort_size": len(peers),
        },
        threshold={"mad_z_threshold": threshold},
        evidence=(
            {
                "features": sorted(features),
                "scaling": "per-feature median_and_MAD_robust_z",
                "cohort": dict(spec.cohort_definition),
                "missingness": missing,
                "score_semantics": "uncalibrated review priority, not probability",
            },
        ),
        config=config,
    )


def _trajectory_features(row: Mapping[str, Any]) -> tuple[dict[str, float], TimeInterval | None]:
    positions, _times, frames = _trace_positions(row)
    path_length = sum(
        math.hypot(right[0] - left[0], right[1] - left[1]) for left, right in pairwise(positions)
    )
    displacement = math.hypot(
        positions[-1][0] - positions[0][0], positions[-1][1] - positions[0][1]
    )
    headings: list[float] = []
    for frame in frames:
        robot = frame.get("robot")
        if isinstance(robot, Mapping):
            heading = robot.get("heading", robot.get("heading_rad"))
            if heading is not None:
                headings.append(_finite(heading, name="heading"))
    turns = sum(
        1 for left, right in pairwise(headings) if abs(_wrapped_angle_delta(right, left)) >= 0.2
    )
    return {
        "path_length_m": path_length,
        "displacement_m": displacement,
        "turns": float(turns),
    }, _trace_interval(frames)


def _trajectory(
    spec: DetectorSpec,
    row: Mapping[str, Any],
    cohort: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> Signal:
    try:
        features, interval = _trajectory_features(row)
    except DetectorError as error:
        return _trace_failure(spec, row, error, config=config)
    peers = _group_cohort(row, cohort, spec)
    minimum = _configured_integer(config, spec, "minimum_cohort", minimum=2)
    threshold = _configured_number(config, spec, "mad_z_threshold", minimum=0.0)
    if minimum is None or threshold is None:
        return _detector_error(spec, row, "invalid_trajectory_parameters", config=config)
    if len(peers) < minimum:
        return _unavailable(
            spec, row, "too_small_cohort", missing=("cohort.minimum",), config=config
        )
    peer_features: dict[str, list[float]] = {name: [] for name in features}
    for item in peers:
        try:
            other, _ = _trajectory_features(item)
        except DetectorError:
            continue
        for name in peer_features:
            peer_features[name].append(other[name])
    z_scores = {
        name: _robust_z(value, peer_features[name])
        for name, value in features.items()
        if len(peer_features[name]) >= minimum - 1
    }
    if not z_scores:
        return _unavailable(
            spec,
            row,
            "cohort_trajectory_features_unavailable",
            missing=("cohort.trace",),
            config=config,
        )
    flagged = any(value >= threshold for value in z_scores.values())
    return _make_signal(
        spec,
        row,
        "flagged" if flagged else "clear",
        reason="trajectory_shape_outlier" if flagged else "within_cohort_trajectory_shape",
        measured={
            "features": features,
            "robust_z_scores": z_scores,
            "priority_score_uncalibrated": math.sqrt(
                sum(value * value for value in z_scores.values())
            ),
            "cohort_size": len(peers),
        },
        threshold={"mad_z_threshold": threshold},
        evidence=(
            {
                "features": sorted(features),
                "scaling": "median_and_MAD_robust_z",
                "cohort": dict(spec.cohort_definition),
                "missingness": "episodes without a valid trace are excluded",
                "score_semantics": "uncalibrated review priority, not probability",
            },
        ),
        interval=interval,
        config=config,
    )


_DETECTORS: dict[
    str,
    Callable[
        [DetectorSpec, Mapping[str, Any], Sequence[Mapping[str, Any]], Mapping[str, Any]], Signal
    ],
] = {
    "goal_adjacent_timeout": _goal_adjacent_timeout,
    "initial_reset_anomaly": _initial_reset_anomaly,
    "stuck_no_progress": _stuck_no_progress,
    "oscillation_limit_cycle": _oscillation,
    "outcome_metric_contradiction": _contradiction,
    "extreme_measurements": _extreme,
    "actuator_mismatch": _actuator,
    "telemetry_integrity": _telemetry,
    "provenance_consistency": _provenance,
    "common_mode_anomaly": _common_mode,
    "planner_disagreement": _planner_disagreement,
    "seed_outlier": _seed_outlier,
    "cohort_multivariate_outlier": _multivariate,
    "trajectory_shape_outlier": _trajectory,
}


def detect(  # noqa: C901
    detector: DetectorSpec | str,
    row: Mapping[str, Any],
    *,
    cohort: Sequence[Mapping[str, Any]] = (),
    config: Mapping[str, Any] | None = None,
    registry: DetectorRegistry | None = None,
) -> Signal:
    """Evaluate one detector against one recorded episode.

    The function is fail-closed: malformed detector input becomes an explicit
    ``error`` signal rather than an uncaught exception or an inferred clear.

    Returns:
        A strict BA-03 detector signal.
    """

    if not isinstance(row, Mapping):
        raise DetectorError("detector row must be a mapping")
    if registry is not None and not isinstance(registry, DetectorRegistry):
        raise DetectorError("registry must be a DetectorRegistry")
    if config is not None and not isinstance(config, Mapping):
        raise DetectorError("detector config must be a mapping")
    if not isinstance(cohort, Sequence) or isinstance(cohort, (str, bytes)):
        raise DetectorError("detector cohort must be a sequence of mappings")
    if any(not isinstance(item, Mapping) for item in cohort):
        raise DetectorError("detector cohort must contain mappings")
    active_registry = registry or DEFAULT_REGISTRY
    try:
        spec = detector if isinstance(detector, DetectorSpec) else active_registry.get(detector)
    except (KeyError, TypeError) as exc:
        raise DetectorError(f"unknown detector: {detector!r}") from exc
    config_mapping = _closed(config or {}, name="detector config")
    episode_id = row.get("episode_id")
    if not isinstance(episode_id, str) or not episode_id.strip():
        # Keep the error a BA-03 record with a stable synthetic identity.
        row = {**dict(row), "episode_id": "invalid-episode"}
    admission_failure = _admission_failure(
        row, check_nonfinite=spec.detector_id != "telemetry_integrity"
    )
    if admission_failure is not None:
        failure_status, failure_reason = admission_failure
        if failure_status == "unavailable":
            return _unavailable(
                spec,
                row,
                failure_reason,
                missing=("execution_status",),
                config=config_mapping,
            )
        return _detector_error(spec, row, failure_reason, config=config_mapping)
    capabilities = _capabilities(row)
    missing = sorted(set(spec.required_capabilities) - capabilities)
    if missing:
        return _unavailable(
            spec,
            row,
            "required_capability_unavailable",
            missing=tuple(missing),
            config=config_mapping,
        )
    try:
        return _DETECTORS[spec.detector_id](spec, row, tuple(cohort), config_mapping)
    except DetectorError as error:
        return _detector_error(spec, row, str(error), config=config_mapping)
    except (TypeError, ValueError, OverflowError, KeyError) as error:
        return _detector_error(
            spec, row, f"detector_input_error:{type(error).__name__}", config=config_mapping
        )


def run_detector(
    detector: DetectorSpec | str,
    row: Mapping[str, Any],
    *,
    cohort: Sequence[Mapping[str, Any]] = (),
    config: Mapping[str, Any] | None = None,
    registry: DetectorRegistry | None = None,
) -> Signal:
    """Alias for :func:`detect` used by callers that prefer imperative naming.

    Returns:
        A strict BA-03 detector signal.
    """

    return detect(detector, row, cohort=cohort, config=config, registry=registry)


def unavailable_signal(
    detector: DetectorSpec,
    row: Mapping[str, Any],
    reason: str,
    *,
    missing: Sequence[str] = (),
    config: Mapping[str, Any] | None = None,
) -> Signal:
    """Build an explicit unavailable signal for a non-readable input row.

    Returns:
        A strict BA-03 ``Signal`` record.
    """

    return _unavailable(detector, row, reason, missing=missing, config=config)


def run_detectors(
    rows: Sequence[Mapping[str, Any]],
    *,
    registry: DetectorRegistry | None = None,
    detector_ids: Sequence[str] | None = None,
    config: Mapping[str, Any] | None = None,
) -> tuple[Signal, ...]:
    """Run selected detectors in deterministic episode/registry order.

    Returns:
        Signals ordered by episode ID, detector ID, and signal ID.
    """

    active = registry or DEFAULT_REGISTRY
    selected = (
        active.ids
        if detector_ids is None
        else normalize_detector_ids(
            detector_ids,
            include_advisory=any(item.advisory for item in active),
            registry=active,
        )
    )
    specs = [active.get(item) for item in selected]
    ordered_rows = sorted(rows, key=lambda row: str(row.get("episode_id", "")))
    signals: list[Signal] = []
    for row in ordered_rows:
        for spec in specs:
            signals.append(detect(spec, row, cohort=ordered_rows, config=config, registry=active))
    return tuple(signals)


# Explicit aliases make the small API discoverable without exposing internal
# function names.  They also preserve the distinction between candidate
# signals and confirmed findings: this module never creates a Finding.
evaluate_detector = detect
build_detector_registry = default_registry


def signal_status_counts(signals: Iterable[Signal]) -> dict[str, int]:
    """Count statuses without collapsing unavailable/error into unevaluable.

    Returns:
        Counts for every BA-03 signal status.
    """

    counts = Counter(signal.status for signal in signals)
    return {status: counts.get(status, 0) for status in SIGNAL_STATUSES}


__all__ = [
    "ADVISORY_DETECTOR_IDS",
    "ALL_DETECTOR_IDS",
    "DEFAULT_REGISTRY",
    "DETECTOR_ALIASES",
    "DETECTOR_ENGINE_VERSION",
    "DETECTOR_REGISTRY_SCHEMA_VERSION",
    "DETECTOR_REGISTRY_VERSION",
    "DETERMINISTIC_DETECTOR_IDS",
    "GOAL_ADJACENT_TIMEOUT_VERSION",
    "DetectorError",
    "DetectorRegistry",
    "DetectorSpec",
    "build_detector_registry",
    "default_registry",
    "detect",
    "detector_registry",
    "evaluate_detector",
    "normalize_detector_ids",
    "registry_document",
    "run_detector",
    "run_detectors",
    "signal_status_counts",
    "unavailable_signal",
]
