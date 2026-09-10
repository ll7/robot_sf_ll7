"""Typed, fail-closed snapshots for the existing counterfactual replay seam.

This module is a preparation-only prototype for the ``#7394`` state contract.  It
does not replace :mod:`last_avoidable_replay` or make a simulator checkpoint a
benchmark result.  The durable representation is deliberately boring: JSON
metadata plus a compressed NumPy payload.  Loading never executes a pickled
object, and compatibility is checked before a destination simulator is mutated.

The supported native subset is the production ``SimulatorCounterfactualModel``
with one robot.  Controller, sensor-history, metric-accumulator, and recurrent
planner state are recorded in the inventory as unsupported until an owner supplies
an explicit adapter.  A snapshot that omits one of those fields must therefore be
treated as a conditional continuation, not a full environment restart.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
import tempfile
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from robot_sf.benchmark.simulator_counterfactual_adapter import _SimulatorSnapshot

SNAPSHOT_SCHEMA = "simulator_typed_snapshot.v2"
SNAPSHOT_BOUNDARY = "pre_step"
_DIGEST_FIELDS = ("map_sha256", "config_sha256", "code_revision")
_COMPATIBILITY_FIELDS = frozenset(
    {
        "map_sha256",
        "config_sha256",
        "code_revision",
        "dt_s",
        "planner_id",
        "checkpoint_sha256",
        "platform_tag",
    }
)
_REQUIRED_COMPATIBILITY_FIELDS = _COMPATIBILITY_FIELDS - {"checkpoint_sha256"}
_BOUNDARY_FIELDS = frozenset(
    {
        "step_index",
        "absolute_time_s",
        "remaining_budget_steps",
        "phase",
        "next_observation_ready",
    }
)
_REQUIRED_BOUNDARY_FIELDS = _BOUNDARY_FIELDS - {
    "remaining_budget_steps",
    "next_observation_ready",
}
_REQUIRED_STATE_FIELDS = frozenset(
    {
        "actor_order",
        "robot_poses",
        "robot_states",
        "robot_navigators",
        "single_runtimes",
        "route_navigators",
        "pedestrian_groups",
        "pedestrian_group_by_ped",
        "rng_capture_complete",
        "global_rng",
        "python_random_state",
        "behavior_rng_states",
        "residual_adversary_state",
        "peds_have_obstacle_forces",
    }
)
_MISSING = object()
_RESTORE_FAILURES = (
    AttributeError,
    IndexError,
    KeyError,
    OverflowError,
    RuntimeError,
    TypeError,
    ValueError,
)


class SnapshotContractError(ValueError):
    """Base error for malformed or unsupported typed snapshot artifacts."""


class SnapshotCompatibilityError(SnapshotContractError):
    """Raised when a snapshot cannot be applied to the declared destination."""


class SnapshotPayloadError(SnapshotContractError):
    """Raised when JSON metadata or numeric payload validation fails."""


def _is_digest(value: str) -> bool:
    """Return whether ``value`` is a complete lowercase SHA-256 digest."""
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _finite_float(value: Any, name: str) -> float:
    """Coerce one finite float or raise a typed contract error.

    Returns:
        The finite float value.
    """
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SnapshotContractError(f"{name} must be a finite float") from exc
    if not math.isfinite(result):
        raise SnapshotContractError(f"{name} must be a finite float")
    return result


def _strict_float(value: Any, name: str) -> float:
    """Validate a numeric float without accepting string or boolean coercion.

    Returns:
        The validated finite float.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise SnapshotContractError(f"{name} must be a finite float")
    return _finite_float(value, name)


def _strict_string(value: Any, name: str) -> str:
    """Validate one string-valued contract field without string coercion.

    Returns:
        The validated string.
    """
    if type(value) is not str:
        raise SnapshotContractError(f"{name} must be a string")
    return value


def _payload_int(value: Any, name: str) -> int:
    """Validate one JSON integer without truncating or accepting booleans.

    Returns:
        The validated integer.
    """
    if type(value) is not int:
        raise SnapshotPayloadError(f"{name} must be an integer")
    return value


def _payload_float(value: Any, name: str) -> float:
    """Validate one JSON numeric scalar without accepting string coercions.

    Returns:
        The validated finite float.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SnapshotPayloadError(f"{name} must be a finite number")
    return _finite_float(value, name)


def _payload_bool(value: Any, name: str) -> bool:
    """Validate one JSON boolean without accepting truthiness coercion.

    Returns:
        The validated boolean.
    """
    if type(value) is not bool:
        raise SnapshotPayloadError(f"{name} must be a boolean")
    return value


@dataclass(frozen=True, slots=True)
class SnapshotCompatibility:
    """Immutable inputs that must match before a snapshot can be restored."""

    map_sha256: str
    config_sha256: str
    code_revision: str
    dt_s: float
    planner_id: str = "unsupported"
    checkpoint_sha256: str | None = None
    platform_tag: str = field(default_factory=platform.platform)

    def __post_init__(self) -> None:
        """Validate required identity and timestep fields."""
        for name in _DIGEST_FIELDS:
            value = _strict_string(getattr(self, name), name).strip().lower()
            if not _is_digest(value):
                raise SnapshotContractError(f"{name} must be a complete lowercase SHA-256 digest")
            object.__setattr__(self, name, value)
        if self.checkpoint_sha256 is not None:
            checkpoint = _strict_string(self.checkpoint_sha256, "checkpoint_sha256").strip().lower()
            if not _is_digest(checkpoint):
                raise SnapshotContractError(
                    "checkpoint_sha256 must be a complete lowercase SHA-256 digest when set"
                )
            object.__setattr__(self, "checkpoint_sha256", checkpoint)
        planner_id = _strict_string(self.planner_id, "planner_id")
        if not planner_id.strip():
            raise SnapshotContractError("planner_id must be non-empty")
        object.__setattr__(self, "planner_id", planner_id)
        dt_s = _strict_float(self.dt_s, "dt_s")
        object.__setattr__(self, "dt_s", dt_s)
        if dt_s <= 0.0:
            raise SnapshotContractError("dt_s must be > 0")
        platform_tag = _strict_string(self.platform_tag, "platform_tag")
        if not platform_tag.strip():
            raise SnapshotContractError("platform_tag must be non-empty")
        object.__setattr__(self, "platform_tag", platform_tag)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe compatibility metadata."""
        return {
            "map_sha256": self.map_sha256,
            "config_sha256": self.config_sha256,
            "code_revision": self.code_revision,
            "dt_s": self.dt_s,
            "planner_id": self.planner_id,
            "checkpoint_sha256": self.checkpoint_sha256,
            "platform_tag": self.platform_tag,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SnapshotCompatibility:
        """Parse and validate compatibility metadata.

        Returns:
            A validated compatibility object.
        """
        if not isinstance(payload, Mapping):
            raise SnapshotPayloadError("compatibility must be an object")
        missing = sorted(_REQUIRED_COMPATIBILITY_FIELDS - set(payload))
        if missing:
            raise SnapshotPayloadError(f"compatibility is missing fields: {missing}")
        unknown = sorted(set(payload) - _COMPATIBILITY_FIELDS)
        if unknown:
            raise SnapshotPayloadError(f"compatibility contains unknown fields: {unknown}")
        try:
            return cls(
                map_sha256=_strict_string(payload["map_sha256"], "map_sha256"),
                config_sha256=_strict_string(payload["config_sha256"], "config_sha256"),
                code_revision=_strict_string(payload["code_revision"], "code_revision"),
                dt_s=_payload_float(payload["dt_s"], "dt_s"),
                planner_id=_strict_string(payload["planner_id"], "planner_id"),
                checkpoint_sha256=payload.get("checkpoint_sha256"),
                platform_tag=_strict_string(payload["platform_tag"], "platform_tag"),
            )
        except (TypeError, ValueError, SnapshotContractError) as exc:
            raise SnapshotPayloadError(f"invalid compatibility metadata: {exc}") from exc

    def assert_compatible(self, expected: SnapshotCompatibility) -> None:
        """Raise before mutation when any declared identity differs."""
        mismatches: list[str] = []
        for name in (
            "map_sha256",
            "config_sha256",
            "code_revision",
            "planner_id",
            "checkpoint_sha256",
            "platform_tag",
        ):
            actual = getattr(self, name)
            wanted = getattr(expected, name)
            if actual != wanted:
                mismatches.append(f"{name}: snapshot={actual!r}, destination={wanted!r}")
        if not math.isclose(self.dt_s, expected.dt_s, rel_tol=0.0, abs_tol=0.0):
            mismatches.append(f"dt_s: snapshot={self.dt_s!r}, destination={expected.dt_s!r}")
        if mismatches:
            raise SnapshotCompatibilityError(
                "snapshot compatibility mismatch: " + "; ".join(mismatches)
            )


@dataclass(frozen=True, slots=True)
class SnapshotBoundary:
    """Explicit pre-decision clock and budget convention."""

    step_index: int
    absolute_time_s: float
    remaining_budget_steps: int | None
    phase: str = SNAPSHOT_BOUNDARY
    next_observation_ready: bool = False

    def __post_init__(self) -> None:
        """Validate clock, budget, and phase fields."""
        if not isinstance(self.step_index, int) or isinstance(self.step_index, bool):
            raise SnapshotContractError("boundary.step_index must be an integer")
        if self.step_index < 0:
            raise SnapshotContractError("boundary.step_index must be >= 0")
        object.__setattr__(
            self,
            "absolute_time_s",
            _strict_float(self.absolute_time_s, "boundary.absolute_time_s"),
        )
        if self.absolute_time_s < 0.0:
            raise SnapshotContractError("boundary.absolute_time_s must be >= 0")
        if self.remaining_budget_steps is not None:
            if not isinstance(self.remaining_budget_steps, int) or isinstance(
                self.remaining_budget_steps, bool
            ):
                raise SnapshotContractError("boundary.remaining_budget_steps must be an integer")
            if self.remaining_budget_steps < 0:
                raise SnapshotContractError("boundary.remaining_budget_steps must be >= 0")
        if type(self.phase) is not str:
            raise SnapshotContractError("boundary.phase must be a string")
        if self.phase != SNAPSHOT_BOUNDARY:
            raise SnapshotContractError(
                f"unsupported snapshot phase {self.phase!r}; expected {SNAPSHOT_BOUNDARY!r}"
            )
        if not isinstance(self.next_observation_ready, bool):
            raise SnapshotContractError("boundary.next_observation_ready must be a boolean")

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe boundary metadata."""
        return {
            "step_index": self.step_index,
            "absolute_time_s": self.absolute_time_s,
            "remaining_budget_steps": self.remaining_budget_steps,
            "phase": self.phase,
            "next_observation_ready": self.next_observation_ready,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SnapshotBoundary:
        """Parse and validate a boundary mapping.

        Returns:
            A validated snapshot boundary.
        """
        if not isinstance(payload, Mapping):
            raise SnapshotPayloadError("snapshot boundary must be an object")
        missing = sorted(_REQUIRED_BOUNDARY_FIELDS - set(payload))
        if missing:
            raise SnapshotPayloadError(f"invalid snapshot boundary: missing fields {missing}")
        unknown = sorted(set(payload) - _BOUNDARY_FIELDS)
        if unknown:
            raise SnapshotPayloadError(f"invalid snapshot boundary: unknown fields {unknown}")
        raw_ready = payload.get("next_observation_ready", False)
        if not isinstance(raw_ready, bool):
            raise SnapshotPayloadError("boundary.next_observation_ready must be a boolean")
        try:
            return cls(
                step_index=_payload_int(payload["step_index"], "boundary.step_index"),
                absolute_time_s=_payload_float(
                    payload["absolute_time_s"], "boundary.absolute_time_s"
                ),
                remaining_budget_steps=(
                    None
                    if payload.get("remaining_budget_steps") is None
                    else _payload_int(
                        payload["remaining_budget_steps"], "boundary.remaining_budget_steps"
                    )
                ),
                phase=_strict_string(payload["phase"], "boundary.phase"),
                next_observation_ready=raw_ready,
            )
        except (KeyError, TypeError, ValueError, SnapshotContractError) as exc:
            raise SnapshotPayloadError(f"invalid snapshot boundary: {exc}") from exc


def _array_copy(value: Any, name: str) -> np.ndarray:
    """Validate one numeric array and return an owned copy.

    Returns:
        An owned numeric array copy.
    """
    array = np.asarray(value)
    if array.dtype == object:
        raise SnapshotContractError(f"{name} has unsafe object dtype")
    if not np.issubdtype(array.dtype, np.number):
        raise SnapshotContractError(f"{name} must have a numeric dtype")
    if not np.all(np.isfinite(array)):
        raise SnapshotContractError(f"{name} contains non-finite values")
    return np.array(array, copy=True)


def _array_reference(value: np.ndarray, arrays: dict[str, np.ndarray], path: str) -> dict[str, str]:
    """Store one numeric array and return its metadata reference.

    Returns:
        A JSON-safe reference to the stored array.
    """
    name = path.replace(".", "_").replace("[", "_").replace("]", "") or "array"
    base = name
    suffix = 1
    while name in arrays:
        suffix += 1
        name = f"{base}_{suffix}"
    arrays[name] = _array_copy(value, path)
    return {"$array": name}


def _encode_mapping(
    value: Mapping[Any, Any], arrays: dict[str, np.ndarray], path: str
) -> dict[str, Any]:
    """Encode mapping values in deterministic key order.

    Returns:
        An encoded mapping.
    """
    return {
        str(key): _encode_value(item, arrays, f"{path}.{key}")
        for key, item in sorted(value.items(), key=lambda item: str(item[0]))
    }


def _encode_sequence(value: Sequence[Any], arrays: dict[str, np.ndarray], path: str) -> list[Any]:
    """Encode sequence members while preserving their order.

    Returns:
        An encoded list of sequence members.
    """
    return [_encode_value(item, arrays, f"{path}[]") for item in value]


def _encode_value(value: Any, arrays: dict[str, np.ndarray], path: str) -> Any:
    """Encode nested state with explicit references for numeric arrays.

    Returns:
        JSON-compatible value with numeric arrays replaced by references.
    """
    if isinstance(value, np.ndarray):
        return _array_reference(value, arrays, path)
    if isinstance(value, np.generic):
        return _encode_value(value.item(), arrays, path)
    if isinstance(value, bool | int | str) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SnapshotContractError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        return _encode_mapping(value, arrays, path)
    if isinstance(value, tuple):
        return {"$tuple": _encode_sequence(value, arrays, path)}
    if isinstance(value, list):
        return _encode_sequence(value, arrays, path)
    raise SnapshotContractError(f"{path} contains unsupported value type {type(value).__name__}")


def _decode_value(value: Any, arrays: Mapping[str, np.ndarray], path: str) -> Any:
    """Decode explicit array/tuple references from validated metadata.

    Returns:
        Rehydrated state value containing owned numeric arrays and tuples.
    """
    if isinstance(value, list):
        return [_decode_value(item, arrays, f"{path}[]") for item in value]
    if not isinstance(value, dict):
        return value
    if "$array" in value and set(value) != {"$array"}:
        raise SnapshotPayloadError(f"{path} has unknown array-reference fields")
    if "$tuple" in value and set(value) != {"$tuple"}:
        raise SnapshotPayloadError(f"{path} has unknown tuple-reference fields")
    if set(value) == {"$array"}:
        name = value["$array"]
        if not isinstance(name, str) or name not in arrays:
            raise SnapshotPayloadError(f"{path} references missing array {name!r}")
        return arrays[name].copy()
    if set(value) == {"$tuple"}:
        items = value["$tuple"]
        if not isinstance(items, list):
            raise SnapshotPayloadError(f"{path} has malformed tuple reference")
        return tuple(_decode_value(item, arrays, f"{path}[]") for item in items)
    return {key: _decode_value(item, arrays, f"{path}.{key}") for key, item in value.items()}


def _qualified_type(value: Any) -> str:
    """Return a stable type name for a known dataclass value."""
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _serialize_dataclass(value: Any) -> dict[str, Any] | None:
    """Serialize a known robot state dataclass without executable code.

    Returns:
        Typed field mapping, or ``None`` when the source state is absent.
    """
    if value is None:
        return None
    if not is_dataclass(value):
        raise SnapshotContractError(f"unsupported robot state type {type(value).__name__}")
    return {
        "type": _qualified_type(value),
        "fields": {item.name: deepcopy(getattr(value, item.name)) for item in fields(value)},
    }


def _deserialize_dataclass(current: Any, payload: Mapping[str, Any]) -> Any:
    """Rehydrate fields into a destination-owned dataclass instance.

    Returns:
        A destination-owned state copy with restored fields.
    """
    if not isinstance(payload, Mapping):
        raise SnapshotPayloadError("robot state payload must be an object")
    if not is_dataclass(current):
        raise SnapshotCompatibilityError("destination robot state is not a dataclass")
    if set(payload) != {"type", "fields"}:
        raise SnapshotPayloadError("robot state contains incomplete or unknown fields")
    if type(payload["type"]) is not str:
        raise SnapshotPayloadError("robot state type must be a string")
    if payload.get("type") != _qualified_type(current):
        raise SnapshotCompatibilityError(
            f"robot state type mismatch: snapshot={payload.get('type')!r}, "
            f"destination={_qualified_type(current)!r}"
        )
    raw_fields = payload.get("fields")
    if not isinstance(raw_fields, Mapping):
        raise SnapshotPayloadError("robot state fields must be an object")
    known = {item.name for item in fields(current)}
    if set(raw_fields) != known:
        raise SnapshotCompatibilityError(
            f"robot state field set mismatch: snapshot={sorted(raw_fields)}, destination={sorted(known)}"
        )
    restored = deepcopy(current)
    for name, value in raw_fields.items():
        setattr(restored, name, _coerce_like(value, getattr(current, name)))
    return restored


def _coerce_sequence_like(value: Any, template: tuple[Any, ...] | list[Any]) -> Any:
    """Restore one JSON sequence using the destination field's container shape.

    Returns:
        A destination-shaped sequence with recursively validated members.
    """
    if isinstance(template, tuple) and isinstance(value, (tuple, list)):
        if len(value) != len(template):
            raise SnapshotPayloadError("typed tuple field has an unexpected length")
        return tuple(_coerce_like(item, template[index]) for index, item in enumerate(value))
    if isinstance(template, list) and isinstance(value, (tuple, list)):
        item_template = template[0] if template else None
        return [_coerce_like(item, item_template) for item in value]
    raise SnapshotPayloadError("typed sequence field has an incompatible shape")


def _coerce_scalar_like(value: Any, template: Any) -> Any:
    """Restore one scalar field without accepting scalar coercion.

    Returns:
        A value validated against the destination scalar shape.
    """
    if isinstance(template, bool):
        if not isinstance(value, bool):
            raise SnapshotPayloadError("typed boolean field must be a boolean")
        return value
    if isinstance(template, int) and not isinstance(template, bool):
        return _payload_int(value, "typed integer field")
    if isinstance(template, float):
        return _payload_float(value, "typed float field")
    if isinstance(template, str):
        if type(value) is not str:
            raise SnapshotPayloadError("typed string field must be a string")
        return value
    if template is None:
        if value is not None:
            raise SnapshotPayloadError("typed null field must be null")
        return None
    return deepcopy(value)


def _coerce_like(value: Any, template: Any) -> Any:
    """Restore a typed field without accepting scalar coercion.

    Returns:
        A value validated against the destination field's container/scalar shape.
    """
    if isinstance(template, (tuple, list)):
        if not isinstance(value, (tuple, list)):
            raise SnapshotPayloadError("typed sequence field has an incompatible shape")
        return _coerce_sequence_like(value, template)
    return _coerce_scalar_like(value, template)


def _serialize_navigator(navigator: Any) -> dict[str, Any]:
    """Serialize mutable route-navigator fields with stable scalar types.

    Returns:
        JSON-safe route navigator fields.
    """
    return {
        "waypoints": [
            [float(point[0]), float(point[1])] for point in getattr(navigator, "waypoints", [])
        ],
        "waypoint_id": int(navigator.waypoint_id),
        "proximity_threshold": float(navigator.proximity_threshold),
        "pos": [float(navigator.pos[0]), float(navigator.pos[1])],
        "reached_waypoint": bool(navigator.reached_waypoint),
    }


def _deserialize_navigator(current: Any, payload: Mapping[str, Any]) -> Any:
    """Restore a route navigator into a destination-owned copy.

    Returns:
        A destination-owned navigator copy.
    """
    if not isinstance(payload, Mapping):
        raise SnapshotPayloadError("robot navigator payload must be an object")
    required = {"waypoints", "waypoint_id", "proximity_threshold", "pos", "reached_waypoint"}
    if set(payload) != required:
        raise SnapshotPayloadError("robot navigator field set is incomplete or unknown")
    restored = deepcopy(current)
    raw_waypoints = payload["waypoints"]
    if not isinstance(raw_waypoints, list):
        raise SnapshotPayloadError("robot navigator waypoints must be a list")
    waypoints: list[tuple[float, float]] = []
    for index, point in enumerate(raw_waypoints):
        if not isinstance(point, list) or len(point) != 2:
            raise SnapshotPayloadError(f"robot navigator waypoint {index} must be a pair")
        waypoints.append(
            (
                _payload_float(point[0], f"robot navigator waypoints[{index}][0]"),
                _payload_float(point[1], f"robot navigator waypoints[{index}][1]"),
            )
        )
    restored.waypoints = waypoints
    restored.waypoint_id = _payload_int(payload["waypoint_id"], "robot navigator waypoint_id")
    restored.proximity_threshold = _payload_float(
        payload["proximity_threshold"], "robot navigator proximity_threshold"
    )
    raw_pos = payload["pos"]
    if not isinstance(raw_pos, list) or len(raw_pos) != 2:
        raise SnapshotPayloadError("robot navigator pos must be a pair")
    restored.pos = (
        _payload_float(raw_pos[0], "robot navigator pos[0]"),
        _payload_float(raw_pos[1], "robot navigator pos[1]"),
    )
    if not isinstance(payload["reached_waypoint"], bool):
        raise SnapshotPayloadError("robot navigator reached_waypoint must be a boolean")
    restored.reached_waypoint = payload["reached_waypoint"]
    if restored.waypoints and not 0 <= restored.waypoint_id < len(restored.waypoints):
        raise SnapshotCompatibilityError("robot navigator waypoint_id is outside its route")
    return restored


def _serialize_global_rng(state: Any) -> dict[str, Any] | None:
    """Serialize the legacy NumPy RNG tuple as metadata plus an array reference.

    Returns:
        JSON/scalar metadata with the state array retained for payload extraction.
    """
    if state is None:
        return None
    try:
        key, values, pos, has_gauss, cached_gauss = state
    except (TypeError, ValueError) as exc:
        raise SnapshotContractError("global NumPy RNG state has an unsupported shape") from exc
    return {
        "kind": str(key),
        "state": np.asarray(values).copy(),
        "pos": int(pos),
        "has_gauss": int(has_gauss),
        "cached_gauss": float(cached_gauss),
    }


def _deserialize_global_rng(state: Any) -> tuple[Any, ...] | None:
    """Rebuild a NumPy legacy RNG tuple from decoded state.

    Returns:
        A tuple accepted by ``numpy.random.set_state``, or ``None``.
    """
    if state is None:
        return None
    if not isinstance(state, Mapping):
        raise SnapshotPayloadError("global_rng must be an object or null")
    required = {"kind", "state", "pos", "has_gauss", "cached_gauss"}
    if set(state) != required:
        raise SnapshotPayloadError("global_rng metadata is incomplete or unknown")
    if type(state["kind"]) is not str:
        raise SnapshotPayloadError("global_rng.kind must be a string")
    if not isinstance(state["state"], np.ndarray):
        raise SnapshotPayloadError("global_rng.state must be a numeric array")
    return (
        state["kind"],
        _array_copy(state["state"], "global_rng.state"),
        _payload_int(state["pos"], "global_rng.pos"),
        _payload_int(state["has_gauss"], "global_rng.has_gauss"),
        _payload_float(state["cached_gauss"], "global_rng.cached_gauss"),
    )


def _group_integer(value: Any, path: str, error_type: type[SnapshotContractError]) -> int:
    """Return one pedestrian/group identifier with an explicit integer contract."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise error_type(f"{path} must be an integer")
    return int(value)


def _serialize_group_memberships(value: Any, name: str) -> list[list[Any]] | None:
    """Serialize group membership without losing integer mapping keys in JSON.

    Returns:
        Sorted ``[group_id, pedestrian_ids]`` entries, or ``None``.
    """
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise SnapshotContractError(f"{name} must be a mapping or null")
    entries: list[list[Any]] = []
    seen_group_ids: set[int] = set()
    for raw_group_id, raw_pedestrians in value.items():
        group_id = _group_integer(raw_group_id, f"{name}.group_id", SnapshotContractError)
        if group_id in seen_group_ids:
            raise SnapshotContractError(f"{name} contains duplicate group id {group_id}")
        seen_group_ids.add(group_id)
        if not isinstance(raw_pedestrians, (set, frozenset, list, tuple)):
            raise SnapshotContractError(f"{name}[{group_id}] must be a pedestrian-id sequence")
        pedestrian_ids = [
            _group_integer(pedestrian_id, f"{name}[{group_id}][]", SnapshotContractError)
            for pedestrian_id in raw_pedestrians
        ]
        if len(set(pedestrian_ids)) != len(pedestrian_ids):
            raise SnapshotContractError(f"{name}[{group_id}] contains duplicate pedestrian ids")
        entries.append([group_id, sorted(pedestrian_ids)])
    return sorted(entries, key=lambda entry: int(entry[0]))


def _serialize_group_reverse_lookup(value: Any, name: str) -> list[list[int]] | None:
    """Serialize the pedestrian-to-group lookup with integer keys and values preserved.

    Returns:
        Sorted ``[pedestrian_id, group_id]`` entries, or ``None``.
    """
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise SnapshotContractError(f"{name} must be a mapping or null")
    entries: list[list[int]] = []
    seen_pedestrian_ids: set[int] = set()
    for raw_pedestrian_id, raw_group_id in value.items():
        pedestrian_id = _group_integer(
            raw_pedestrian_id, f"{name}.pedestrian_id", SnapshotContractError
        )
        if pedestrian_id in seen_pedestrian_ids:
            raise SnapshotContractError(f"{name} contains duplicate pedestrian id {pedestrian_id}")
        seen_pedestrian_ids.add(pedestrian_id)
        group_id = _group_integer(raw_group_id, f"{name}[{pedestrian_id}]", SnapshotContractError)
        entries.append([pedestrian_id, group_id])
    return sorted(entries, key=lambda entry: int(entry[0]))


def _deserialize_group_memberships(value: Any, name: str) -> dict[int, set[int]] | None:
    """Decode the JSON-safe group membership sequence into native set-valued mappings.

    Returns:
        A native set-valued mapping, or ``None``.
    """
    if value is None:
        return None
    if not isinstance(value, list):
        raise SnapshotPayloadError(f"{name} must be a list or null")
    memberships: dict[int, set[int]] = {}
    for index, entry in enumerate(value):
        path = f"{name}[{index}]"
        if not isinstance(entry, list) or len(entry) != 2:
            raise SnapshotPayloadError(f"{path} must contain [group_id, pedestrian_ids]")
        group_id = _group_integer(entry[0], f"{path}[0]", SnapshotPayloadError)
        if group_id in memberships:
            raise SnapshotPayloadError(f"{name} contains duplicate group id {group_id}")
        raw_pedestrian_ids = entry[1]
        if not isinstance(raw_pedestrian_ids, list):
            raise SnapshotPayloadError(f"{path}[1] must be a list")
        pedestrian_ids = {
            _group_integer(pedestrian_id, f"{path}[1][]", SnapshotPayloadError)
            for pedestrian_id in raw_pedestrian_ids
        }
        if len(pedestrian_ids) != len(raw_pedestrian_ids):
            raise SnapshotPayloadError(f"{path}[1] contains duplicate pedestrian ids")
        memberships[group_id] = pedestrian_ids
    return memberships


def _deserialize_group_reverse_lookup(value: Any, name: str) -> dict[int, int] | None:
    """Decode the JSON-safe reverse lookup into a native integer mapping.

    Returns:
        A native pedestrian-to-group mapping, or ``None``.
    """
    if value is None:
        return None
    if not isinstance(value, list):
        raise SnapshotPayloadError(f"{name} must be a list or null")
    reverse_lookup: dict[int, int] = {}
    for index, entry in enumerate(value):
        path = f"{name}[{index}]"
        if not isinstance(entry, list) or len(entry) != 2:
            raise SnapshotPayloadError(f"{path} must contain [pedestrian_id, group_id]")
        pedestrian_id = _group_integer(entry[0], f"{path}[0]", SnapshotPayloadError)
        if pedestrian_id in reverse_lookup:
            raise SnapshotPayloadError(f"{name} contains duplicate pedestrian id {pedestrian_id}")
        reverse_lookup[pedestrian_id] = _group_integer(entry[1], f"{path}[1]", SnapshotPayloadError)
    return reverse_lookup


def _validate_group_memberships(
    memberships: dict[int, set[int]], reverse_lookup: dict[int, int], ped_count: int
) -> None:
    """Validate membership IDs and agreement with the reverse lookup."""
    seen_pedestrians: set[int] = set()
    for group_id, pedestrian_ids in memberships.items():
        for pedestrian_id in pedestrian_ids:
            if not 0 <= pedestrian_id < ped_count:
                raise SnapshotCompatibilityError(
                    f"pedestrian group id {pedestrian_id} is outside destination actor count"
                )
            if pedestrian_id in seen_pedestrians:
                raise SnapshotPayloadError(
                    f"pedestrian {pedestrian_id} appears in multiple pedestrian groups"
                )
            seen_pedestrians.add(pedestrian_id)
            if reverse_lookup.get(pedestrian_id) != group_id:
                raise SnapshotPayloadError(
                    f"pedestrian group reverse lookup disagrees for pedestrian {pedestrian_id}"
                )


def _validate_group_reverse_lookup(
    memberships: dict[int, set[int]], reverse_lookup: dict[int, int], ped_count: int
) -> None:
    """Validate that every reverse lookup entry has a matching forward membership."""
    for pedestrian_id, group_id in reverse_lookup.items():
        if not 0 <= pedestrian_id < ped_count:
            raise SnapshotCompatibilityError(
                f"reverse lookup pedestrian id {pedestrian_id} is outside destination actor count"
            )
        if group_id not in memberships or pedestrian_id not in memberships[group_id]:
            raise SnapshotPayloadError(
                f"reverse lookup points to missing membership for pedestrian {pedestrian_id}"
            )


def _validate_group_state(
    memberships: dict[int, set[int]] | None,
    reverse_lookup: dict[int, int] | None,
    ped_count: int,
    destination_groups: Any,
) -> None:
    """Validate group completeness and destination compatibility before restore."""
    if (memberships is None) != (reverse_lookup is None):
        raise SnapshotPayloadError("pedestrian group membership and reverse lookup must agree")
    if memberships is None:
        if destination_groups is not None:
            raise SnapshotCompatibilityError("snapshot omits group state for grouped destination")
        return
    if destination_groups is None:
        raise SnapshotCompatibilityError("destination does not expose pedestrian group state")
    if reverse_lookup is None:
        raise SnapshotPayloadError("pedestrian group reverse lookup is missing")
    _validate_group_memberships(memberships, reverse_lookup, ped_count)
    _validate_group_reverse_lookup(memberships, reverse_lookup, ped_count)


def _restore_group_payload(
    state: Mapping[str, Any], sim: Any, ped_count: int
) -> tuple[dict[int, set[int]] | None, dict[int, int] | None]:
    """Decode and validate group state before constructing a runtime snapshot.

    Returns:
        Native forward and reverse group mappings, or matching ``None`` values.
    """
    raw_memberships = state.get("pedestrian_groups", _MISSING)
    raw_reverse_lookup = state.get("pedestrian_group_by_ped", _MISSING)
    if raw_memberships is _MISSING or raw_reverse_lookup is _MISSING:
        raise SnapshotPayloadError("snapshot pedestrian group state is incomplete")
    memberships = _deserialize_group_memberships(raw_memberships, "pedestrian_groups")
    reverse_lookup = _deserialize_group_reverse_lookup(
        raw_reverse_lookup, "pedestrian_group_by_ped"
    )
    _validate_group_state(memberships, reverse_lookup, ped_count, getattr(sim, "groups", None))
    return memberships, reverse_lookup


def _destination_pedestrian_arrays(sim: Any) -> dict[str, np.ndarray]:
    """Read the destination pedestrian arrays required by the typed adapter.

    Returns:
        Destination-owned arrays keyed by their typed snapshot field names.
    """
    try:
        return {
            "pysf_state": np.asarray(sim.pysf_state.pysf_states()),
            "ped_headings": np.asarray(sim.ped_headings),
            "ped_angular_velocities": np.asarray(sim.ped_angular_velocities),
        }
    except (AttributeError, TypeError, ValueError) as exc:
        raise SnapshotCompatibilityError(
            "destination does not expose the required pedestrian state arrays"
        ) from exc


def _validate_pedestrian_array_shapes(
    arrays: Mapping[str, np.ndarray], destination_arrays: Mapping[str, np.ndarray]
) -> None:
    """Reject any typed pedestrian array whose shape differs from the destination."""
    for name, destination in destination_arrays.items():
        snapshot_shape = tuple(arrays[name].shape)
        destination_shape = tuple(destination.shape)
        if snapshot_shape != destination_shape:
            raise SnapshotCompatibilityError(
                f"{name} shape mismatch: snapshot={snapshot_shape}, destination={destination_shape}"
            )


def _validate_pedestrian_array_dtypes(
    arrays: Mapping[str, np.ndarray], destination_arrays: Mapping[str, np.ndarray]
) -> None:
    """Reject numeric dtype drift that could silently round restored values."""
    for name, destination in destination_arrays.items():
        snapshot_dtype = np.dtype(arrays[name].dtype)
        destination_dtype = np.dtype(destination.dtype)
        if snapshot_dtype != destination_dtype:
            raise SnapshotCompatibilityError(
                f"{name} dtype mismatch: snapshot={snapshot_dtype}, destination={destination_dtype}"
            )


def _validate_max_speed_shape(arrays: Mapping[str, np.ndarray], sim: Any) -> None:
    """Reject speed-cap shape drift, including destinations without speed caps."""
    destination_peds = getattr(getattr(sim, "pysf_sim", None), "peds", None)
    destination_max_speeds = getattr(destination_peds, "max_speeds", None)
    if destination_max_speeds is None:
        if tuple(arrays["ped_max_speeds"].shape) != (0,):
            raise SnapshotCompatibilityError(
                "ped_max_speeds shape mismatch: destination has no speed-cap array"
            )
        return
    destination_shape = tuple(np.asarray(destination_max_speeds).shape)
    if tuple(arrays["ped_max_speeds"].shape) != destination_shape:
        raise SnapshotCompatibilityError(
            f"ped_max_speeds shape mismatch: snapshot={arrays['ped_max_speeds'].shape}, "
            f"destination={destination_shape}"
        )


def _validate_max_speed_dtype(arrays: Mapping[str, np.ndarray], sim: Any) -> None:
    """Reject speed-cap dtype drift when the destination exposes speed caps."""
    destination_peds = getattr(getattr(sim, "pysf_sim", None), "peds", None)
    destination_max_speeds = getattr(destination_peds, "max_speeds", None)
    if destination_max_speeds is None:
        return
    snapshot_dtype = np.dtype(arrays["ped_max_speeds"].dtype)
    destination_dtype = np.dtype(np.asarray(destination_max_speeds).dtype)
    if snapshot_dtype != destination_dtype:
        raise SnapshotCompatibilityError(
            f"ped_max_speeds dtype mismatch: snapshot={snapshot_dtype}, "
            f"destination={destination_dtype}"
        )


def _validate_destination_shape(
    state: Mapping[str, Any], arrays: Mapping[str, np.ndarray], sim: Any
) -> None:
    """Validate destination array shapes and stable actor order before restoration."""
    required_arrays = {
        "pysf_state",
        "ped_headings",
        "ped_angular_velocities",
        "ped_max_speeds",
    }
    missing_arrays = sorted(required_arrays - set(arrays))
    if missing_arrays:
        raise SnapshotPayloadError(f"snapshot numeric payload is missing arrays: {missing_arrays}")
    destination_arrays = _destination_pedestrian_arrays(sim)
    _validate_pedestrian_array_shapes(arrays, destination_arrays)
    _validate_pedestrian_array_dtypes(arrays, destination_arrays)
    _validate_max_speed_shape(arrays, sim)
    _validate_max_speed_dtype(arrays, sim)
    actor_order = state.get("actor_order")
    if not isinstance(actor_order, Mapping):
        raise SnapshotPayloadError("snapshot actor order is missing")
    if set(actor_order) != {"robots", "pedestrians"}:
        raise SnapshotPayloadError("snapshot actor order contains incomplete or unknown fields")
    robot_order = actor_order.get("robots")
    ped_order = actor_order.get("pedestrians")
    if not isinstance(robot_order, list) or not isinstance(ped_order, list):
        raise SnapshotPayloadError("snapshot actor order must contain robot/pedestrian lists")
    if any(type(value) is not str for value in (*robot_order, *ped_order)):
        raise SnapshotPayloadError("snapshot actor order entries must be strings")
    if robot_order != [f"robot:{index}" for index in range(len(sim.robots))]:
        raise SnapshotCompatibilityError("snapshot robot actor identity/order does not match")
    expected_ped_order = [
        f"ped:{index}" for index in range(int(destination_arrays["ped_headings"].shape[0]))
    ]
    if ped_order != expected_ped_order:
        raise SnapshotCompatibilityError("snapshot pedestrian actor identity/order does not match")


def _restore_robot_payload(
    state: Mapping[str, Any], sim: Any
) -> tuple[list[Any], list[Any], list[Any]]:
    """Rebuild destination-owned robot states, navigators, and poses.

    Returns:
        ``(robot_poses, robot_states, robot_navigators)`` for adapter construction.
    """
    robot_states_payload = state.get("robot_states")
    robot_nav_payload = state.get("robot_navigators")
    if not isinstance(robot_states_payload, list) or not isinstance(robot_nav_payload, list):
        raise SnapshotPayloadError("snapshot robot state is incomplete")
    if len(robot_states_payload) != len(sim.robots) or len(robot_nav_payload) != len(
        sim.robot_navs
    ):
        raise SnapshotCompatibilityError("snapshot robot actor count does not match destination")
    robot_states = [
        _deserialize_dataclass(robot.state, payload) if payload is not None else None
        for robot, payload in zip(sim.robots, robot_states_payload, strict=True)
    ]
    robot_navigators = [
        _deserialize_navigator(navigator, payload)
        for navigator, payload in zip(sim.robot_navs, robot_nav_payload, strict=True)
    ]
    raw_robot_poses = state.get("robot_poses")
    if not isinstance(raw_robot_poses, list):
        raise SnapshotPayloadError("snapshot robot poses must be a list")
    robot_poses = []
    for index, raw_pose in enumerate(raw_robot_poses):
        if not isinstance(raw_pose, (tuple, list)) or len(raw_pose) != 2:
            raise SnapshotPayloadError(f"robot pose {index} must contain position and heading")
        raw_position, raw_heading = raw_pose
        if not isinstance(raw_position, (tuple, list)) or len(raw_position) != 2:
            raise SnapshotPayloadError(f"robot pose {index} position must be a pair")
        robot_poses.append(
            (
                (
                    _payload_float(raw_position[0], f"robot_poses[{index}][0][0]"),
                    _payload_float(raw_position[1], f"robot_poses[{index}][0][1]"),
                ),
                _payload_float(raw_heading, f"robot_poses[{index}][1]"),
            )
        )
    if not isinstance(robot_poses, list) or len(robot_poses) != len(sim.robots):
        raise SnapshotCompatibilityError("snapshot robot pose count does not match destination")
    return robot_poses, robot_states, robot_navigators


def _resolve_global_rng(state: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> Any:
    """Resolve an in-memory or decoded global-RNG array reference.

    Returns:
        A mapping ready for legacy NumPy RNG deserialization, or ``None``.
    """
    global_rng = deepcopy(state.get("global_rng"))
    if not isinstance(global_rng, dict):
        return global_rng
    if "state_ref" in global_rng:
        if set(global_rng) != {"kind", "state_ref", "pos", "has_gauss", "cached_gauss"}:
            raise SnapshotPayloadError("global_rng metadata is incomplete or unknown")
        ref = global_rng.pop("state_ref")
    elif isinstance(global_rng.get("state"), dict):
        if set(global_rng["state"]) != {"$array"}:
            raise SnapshotPayloadError("global_rng state reference has unknown fields")
        ref = global_rng["state"].get("$array")
    else:
        return global_rng
    if not isinstance(ref, str) or ref not in arrays:
        raise SnapshotPayloadError("global_rng state array reference is missing")
    global_rng["state"] = arrays[ref].copy()
    return global_rng


def _validate_rng_capture_state(state: Mapping[str, Any], model: Any) -> None:
    """Require complete RNG state and a rollback-capable destination."""
    if state.get("rng_capture_complete") is not True:
        raise SnapshotCompatibilityError(
            "typed snapshot does not contain complete RNG state; "
            "nondeterministic continuation is not restorable"
        )
    if getattr(model, "capture_rng", True) is not True:
        raise SnapshotCompatibilityError(
            "destination model must capture RNG state for atomic typed restore"
        )
    if state.get("global_rng") is None or state.get("python_random_state") is None:
        raise SnapshotPayloadError("typed snapshot RNG state is incomplete")


def _validate_required_state_fields(state: Mapping[str, Any]) -> None:
    """Reject snapshots that silently omit supported mutable state fields."""
    if not isinstance(state, Mapping):
        raise SnapshotPayloadError("snapshot state must be an object")
    missing = sorted(_REQUIRED_STATE_FIELDS - set(state))
    if missing:
        raise SnapshotPayloadError(f"snapshot state is missing supported fields: {missing}")
    unknown = sorted(set(state) - _REQUIRED_STATE_FIELDS)
    if unknown:
        raise SnapshotPayloadError(f"snapshot state contains unknown fields: {unknown}")


def _metadata_digest(metadata: Mapping[str, Any]) -> str:
    """Hash canonical metadata while excluding its self-referential digest.

    Returns:
        The SHA-256 digest of the canonical metadata payload.
    """
    payload = deepcopy(dict(metadata))
    payload.pop("metadata_sha256", None)
    try:
        encoded = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SnapshotPayloadError(f"snapshot metadata is not canonical JSON: {exc}") from exc
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class TypedSimulatorSnapshot:
    """In-memory typed snapshot, independent of process-local object identity."""

    compatibility: SnapshotCompatibility
    boundary: SnapshotBoundary
    state: Mapping[str, Any]
    arrays: Mapping[str, np.ndarray] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        """Own and validate numeric payload arrays."""
        if not isinstance(self.state, Mapping):
            raise SnapshotContractError("snapshot state must be an object")
        if not isinstance(self.arrays, Mapping):
            raise SnapshotContractError("snapshot arrays must be an object")
        if any(type(name) is not str for name in self.arrays):
            raise SnapshotContractError("snapshot array names must be strings")
        copied = {name: _array_copy(value, f"arrays.{name}") for name, value in self.arrays.items()}
        object.__setattr__(self, "arrays", copied)

    @classmethod
    def from_adapter_snapshot(
        cls,
        snapshot: Any,
        compatibility: SnapshotCompatibility,
        *,
        boundary: SnapshotBoundary | None = None,
    ) -> TypedSimulatorSnapshot:
        """Convert the existing adapter seam into a typed representation.

        Returns:
            A typed snapshot with JSON state and numeric arrays.
        """
        if not isinstance(snapshot, _SimulatorSnapshot):
            raise SnapshotContractError(
                "typed snapshot prototype supports only SimulatorCounterfactualModel snapshots"
            )
        if boundary is None:
            absolute_time = (
                snapshot.absolute_time_s
                if snapshot.absolute_time_s is not None
                else snapshot.step_index * compatibility.dt_s
            )
            boundary = SnapshotBoundary(
                step_index=int(snapshot.step_index),
                absolute_time_s=float(absolute_time),
                remaining_budget_steps=snapshot.remaining_budget_steps,
            )
        state = {
            "actor_order": {
                "robots": [f"robot:{index}" for index in range(len(snapshot.robot_poses))],
                "pedestrians": [
                    f"ped:{index}" for index in range(int(snapshot.ped_headings.shape[0]))
                ],
            },
            "robot_poses": deepcopy(snapshot.robot_poses),
            "robot_states": [_serialize_dataclass(value) for value in snapshot.robot_velocities],
            "robot_navigators": [
                _serialize_navigator(value) for value in snapshot.robot_navigators
            ],
            "single_runtimes": deepcopy(snapshot.single_runtimes),
            "route_navigators": deepcopy(snapshot.route_navigators),
            "pedestrian_groups": _serialize_group_memberships(
                snapshot.pedestrian_groups, "pedestrian_groups"
            ),
            "pedestrian_group_by_ped": _serialize_group_reverse_lookup(
                snapshot.pedestrian_group_by_ped, "pedestrian_group_by_ped"
            ),
            "rng_capture_complete": (
                snapshot.global_rng_state is not None and snapshot.python_random_state is not None
            ),
            "global_rng": _serialize_global_rng(snapshot.global_rng_state),
            "python_random_state": deepcopy(snapshot.python_random_state),
            "behavior_rng_states": deepcopy(snapshot.behavior_rng_states),
            "residual_adversary_state": deepcopy(snapshot.residual_adversary_state),
            "peds_have_obstacle_forces": bool(snapshot.peds_have_obstacle_forces),
        }
        arrays = {
            "pysf_state": _array_copy(snapshot.pysf_state, "pysf_state"),
            "ped_headings": _array_copy(snapshot.ped_headings, "ped_headings"),
            "ped_angular_velocities": _array_copy(
                snapshot.ped_angular_velocities, "ped_angular_velocities"
            ),
            "ped_max_speeds": _array_copy(
                snapshot.ped_max_speeds
                if snapshot.ped_max_speeds is not None
                else np.empty((0,), dtype=float),
                "ped_max_speeds",
            ),
        }
        if isinstance(state["global_rng"], dict) and isinstance(
            state["global_rng"].get("state"), np.ndarray
        ):
            arrays["global_rng_state"] = _array_copy(
                state["global_rng"].pop("state"), "global_rng_state"
            )
            state["global_rng"]["state_ref"] = "global_rng_state"
        return cls(compatibility=compatibility, boundary=boundary, state=state, arrays=arrays)

    def _metadata_and_arrays(self) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        """Build JSON metadata and a complete numeric payload mapping.

        Returns:
            ``(metadata, arrays)`` ready for durable serialization.
        """
        arrays = {name: value.copy() for name, value in self.arrays.items()}
        encoded_state = _encode_value(dict(self.state), arrays, "state")
        if not isinstance(encoded_state, dict):
            raise SnapshotContractError("snapshot state must encode as an object")
        global_rng = encoded_state.get("global_rng")
        if isinstance(global_rng, dict) and "state_ref" in global_rng:
            global_rng["state"] = {"$array": str(global_rng.pop("state_ref"))}
        metadata = {
            "schema_version": SNAPSHOT_SCHEMA,
            "compatibility": self.compatibility.to_dict(),
            "boundary": self.boundary.to_dict(),
            "state": encoded_state,
            "arrays": {
                name: {
                    "dtype": str(value.dtype),
                    "shape": list(value.shape),
                }
                for name, value in sorted(arrays.items())
            },
        }
        return metadata, arrays

    def to_metadata_dict(self) -> dict[str, Any]:
        """Return JSON metadata without a payload digest."""
        metadata, _ = self._metadata_and_arrays()
        return metadata

    def to_adapter_snapshot(self, model: Any) -> Any:
        """Rebuild an adapter snapshot using destination-owned runtime objects.

        Returns:
            A private adapter snapshot whose mutable objects belong to ``model``.
        """
        sim = getattr(model, "sim", None)
        if sim is None or len(sim.robots) != 1:
            raise SnapshotCompatibilityError(
                "destination model is not a supported one-robot adapter"
            )
        state = self.state
        _validate_required_state_fields(state)
        _validate_rng_capture_state(state, model)
        _validate_destination_shape(state, self.arrays, sim)
        robot_poses, robot_states, robot_navigators = _restore_robot_payload(state, sim)
        pedestrian_groups, pedestrian_group_by_ped = _restore_group_payload(
            state, sim, int(self.arrays["pysf_state"].shape[0])
        )
        return _SimulatorSnapshot(
            step_index=self.boundary.step_index,
            pysf_state=self.arrays["pysf_state"].copy(),
            ped_headings=self.arrays["ped_headings"].copy(),
            ped_angular_velocities=self.arrays["ped_angular_velocities"].copy(),
            robot_poses=robot_poses,
            robot_velocities=robot_states,
            robot_navigators=robot_navigators,
            single_runtimes=deepcopy(state.get("single_runtimes")),
            route_navigators=deepcopy(state.get("route_navigators", {})),
            pedestrian_groups=pedestrian_groups,
            pedestrian_group_by_ped=pedestrian_group_by_ped,
            global_rng_state=_deserialize_global_rng(_resolve_global_rng(state, self.arrays)),
            peds_have_obstacle_forces=_payload_bool(
                state.get("peds_have_obstacle_forces"), "peds_have_obstacle_forces"
            ),
            ped_max_speeds=self.arrays.get("ped_max_speeds", np.empty((0,), dtype=float)).copy(),
            python_random_state=deepcopy(state.get("python_random_state")),
            behavior_rng_states=deepcopy(state.get("behavior_rng_states")),
            residual_adversary_state=deepcopy(state.get("residual_adversary_state")),
            absolute_time_s=self.boundary.absolute_time_s,
            remaining_budget_steps=self.boundary.remaining_budget_steps,
        )


def capture_typed_snapshot(
    model: Any,
    compatibility: SnapshotCompatibility,
    *,
    boundary: SnapshotBoundary | None = None,
) -> TypedSimulatorSnapshot:
    """Capture a typed snapshot from the existing native adapter seam.

    Returns:
        A validated in-memory typed snapshot.
    """
    return TypedSimulatorSnapshot.from_adapter_snapshot(
        model.snapshot(), compatibility, boundary=boundary
    )


def restore_typed_snapshot(
    model: Any,
    snapshot: TypedSimulatorSnapshot,
    expected: SnapshotCompatibility,
) -> None:
    """Validate compatibility, then restore the destination model atomically."""
    snapshot.compatibility.assert_compatible(expected)
    runtime_snapshot = snapshot.to_adapter_snapshot(model)
    snapshot_method = getattr(model, "snapshot", None)
    restore_method = getattr(model, "restore", None)
    if not callable(snapshot_method) or not callable(restore_method):
        raise SnapshotCompatibilityError(
            "destination model must expose callable snapshot and restore methods"
        )
    before = snapshot_method()
    try:
        restore_method(runtime_snapshot)
    except _RESTORE_FAILURES as exc:
        try:
            restore_method(before)
        except _RESTORE_FAILURES as rollback_exc:
            raise SnapshotContractError(
                "typed snapshot restore failed and destination rollback failed; "
                "destination state may be partial"
            ) from rollback_exc
        raise SnapshotContractError(
            "typed snapshot restore failed; destination state was rolled back"
        ) from exc


def _sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for one artifact file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class SnapshotArtifact:
    """Paths and measured sizes for one durable snapshot pair."""

    metadata_path: Path
    payload_path: Path
    metadata_bytes: int
    payload_bytes: int

    @property
    def total_bytes(self) -> int:
        """Return metadata plus numeric payload size."""
        return self.metadata_bytes + self.payload_bytes


def write_typed_snapshot(
    snapshot: TypedSimulatorSnapshot, metadata_path: str | Path
) -> SnapshotArtifact:
    """Write each file of the typed snapshot pair with atomic replacement.

    The metadata and payload are separate files, so their pair replacement is not
    one filesystem transaction. The reader validates both digests and sizes and
    rejects a mixed-generation pair.

    Returns:
        Artifact paths and measured byte sizes.
    """
    metadata_target = Path(metadata_path)
    payload_target = metadata_target.with_suffix(metadata_target.suffix + ".npz")
    metadata_target.parent.mkdir(parents=True, exist_ok=True)
    metadata, arrays = snapshot._metadata_and_arrays()
    with tempfile.NamedTemporaryFile(
        dir=metadata_target.parent, suffix=".npz", delete=False
    ) as tmp:
        payload_tmp = Path(tmp.name)
        np.savez_compressed(tmp, **arrays)
    metadata_tmp = metadata_target.with_suffix(metadata_target.suffix + ".tmp")
    try:
        payload_tmp.replace(payload_target)
        metadata["payload_sha256"] = _sha256_file(payload_target)
        metadata["payload_bytes"] = payload_target.stat().st_size
        metadata["metadata_sha256"] = _metadata_digest(metadata)
        metadata_tmp.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        metadata_tmp.replace(metadata_target)
    finally:
        payload_tmp.unlink(missing_ok=True)
        metadata_tmp.unlink(missing_ok=True)
    return SnapshotArtifact(
        metadata_path=metadata_target,
        payload_path=payload_target,
        metadata_bytes=metadata_target.stat().st_size,
        payload_bytes=payload_target.stat().st_size,
    )


def _read_snapshot_metadata(metadata_target: Path) -> Mapping[str, Any]:
    """Read JSON metadata and validate its top-level schema marker.

    Returns:
        Parsed JSON metadata mapping.
    """
    try:
        metadata = json.loads(metadata_target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SnapshotPayloadError(f"could not read snapshot metadata: {exc}") from exc
    if not isinstance(metadata, Mapping):
        raise SnapshotPayloadError("snapshot metadata must be a JSON object")
    required = {
        "schema_version",
        "compatibility",
        "boundary",
        "state",
        "arrays",
        "payload_sha256",
        "payload_bytes",
        "metadata_sha256",
    }
    missing = sorted(required - set(metadata))
    if missing:
        raise SnapshotPayloadError(f"snapshot metadata is missing fields: {missing}")
    unknown = sorted(set(metadata) - required)
    if unknown:
        raise SnapshotPayloadError(f"snapshot metadata contains unknown fields: {unknown}")
    if metadata.get("schema_version") != SNAPSHOT_SCHEMA:
        raise SnapshotPayloadError(
            f"unsupported snapshot schema {metadata.get('schema_version')!r}"
        )
    return metadata


def _load_snapshot_arrays(
    payload_target: Path, descriptors: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    """Load descriptor-matched numeric arrays with pickle disabled.

    Returns:
        Owned numeric arrays keyed by the descriptor names.
    """
    try:
        with np.load(payload_target, allow_pickle=False) as loaded:
            expected_names = set(descriptors)
            if set(loaded.files) != expected_names:
                raise SnapshotPayloadError(
                    "snapshot payload array names do not match metadata descriptors"
                )
            arrays: dict[str, np.ndarray] = {}
            for name, descriptor in descriptors.items():
                if not isinstance(name, str) or not isinstance(descriptor, Mapping):
                    raise SnapshotPayloadError("malformed snapshot array descriptor")
                if set(descriptor) != {"dtype", "shape"}:
                    raise SnapshotPayloadError(
                        f"snapshot array descriptor for {name!r} has unknown fields"
                    )
                if type(descriptor["dtype"]) is not str:
                    raise SnapshotPayloadError(
                        f"snapshot array descriptor dtype for {name!r} must be a string"
                    )
                if type(descriptor["shape"]) is not list or any(
                    type(dimension) is not int or dimension < 0 for dimension in descriptor["shape"]
                ):
                    raise SnapshotPayloadError(
                        f"snapshot array descriptor shape for {name!r} must be integer list"
                    )
                if name not in loaded:
                    raise SnapshotPayloadError(f"snapshot payload is missing array {name!r}")
                value = _array_copy(loaded[name], f"arrays.{name}")
                if str(value.dtype) != descriptor.get("dtype") or list(
                    value.shape
                ) != descriptor.get("shape"):
                    raise SnapshotPayloadError(f"snapshot array descriptor mismatch for {name!r}")
                arrays[name] = value
    except (OSError, ValueError, KeyError, SnapshotContractError) as exc:
        raise SnapshotPayloadError(f"invalid snapshot numeric payload: {exc}") from exc
    return arrays


def read_typed_snapshot(metadata_path: str | Path) -> TypedSimulatorSnapshot:
    """Read and validate a typed snapshot without allowing pickle execution.

    Returns:
        A validated typed snapshot loaded from JSON plus NPZ.
    """
    metadata_target = Path(metadata_path)
    metadata = _read_snapshot_metadata(metadata_target)
    expected_metadata_hash = metadata.get("metadata_sha256")
    if not isinstance(expected_metadata_hash, str) or not _is_digest(expected_metadata_hash):
        raise SnapshotPayloadError("snapshot metadata_sha256 is missing or malformed")
    if _metadata_digest(metadata) != expected_metadata_hash:
        raise SnapshotPayloadError("snapshot metadata digest mismatch")
    payload_target = metadata_target.with_suffix(metadata_target.suffix + ".npz")
    expected_hash = metadata.get("payload_sha256")
    if not isinstance(expected_hash, str) or not _is_digest(expected_hash):
        raise SnapshotPayloadError("snapshot payload_sha256 is missing or malformed")
    if not payload_target.is_file() or _sha256_file(payload_target) != expected_hash:
        raise SnapshotPayloadError("snapshot numeric payload digest mismatch or file is missing")
    expected_bytes = metadata.get("payload_bytes")
    if not isinstance(expected_bytes, int) or isinstance(expected_bytes, bool):
        raise SnapshotPayloadError("snapshot payload_bytes is missing or malformed")
    if payload_target.stat().st_size != expected_bytes:
        raise SnapshotPayloadError("snapshot numeric payload size mismatch")
    compatibility_payload = metadata.get("compatibility")
    boundary_payload = metadata.get("boundary")
    if not isinstance(compatibility_payload, Mapping) or not isinstance(boundary_payload, Mapping):
        raise SnapshotPayloadError("snapshot compatibility and boundary must be objects")
    compatibility = SnapshotCompatibility.from_dict(compatibility_payload)
    boundary = SnapshotBoundary.from_dict(boundary_payload)
    descriptors = metadata.get("arrays")
    if not isinstance(descriptors, Mapping):
        raise SnapshotPayloadError("snapshot arrays metadata must be an object")
    arrays = _load_snapshot_arrays(payload_target, descriptors)
    state = _decode_value(metadata.get("state"), arrays, "state")
    if not isinstance(state, Mapping):
        raise SnapshotPayloadError("snapshot state must be an object")
    return TypedSimulatorSnapshot(
        compatibility=compatibility,
        boundary=boundary,
        state=state,
        arrays=arrays,
    )


@dataclass(frozen=True, slots=True)
class NoOpStep:
    """All declared continuation observations compared at one timestep."""

    step: int
    state: Any
    observation: Any = None
    decision: Any = None
    applied_action: Any = None
    events: Any = None
    terminal: Any = None
    metrics: Any = None
    rng: Any = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe mapping for a no-op trace receipt."""
        return {
            "step": self.step,
            "state": self.state,
            "observation": self.observation,
            "decision": self.decision,
            "applied_action": self.applied_action,
            "events": self.events,
            "terminal": self.terminal,
            "metrics": self.metrics,
            "rng": self.rng,
        }


@dataclass(frozen=True, slots=True)
class NoOpComparison:
    """Result of field-by-field continuation comparison."""

    equivalent: bool
    compared_steps: int
    first_divergence_step: int | None = None
    first_divergence_field: str | None = None
    expected: Any = None
    actual: Any = None


def _first_array_difference(expected: Any, actual: Any, path: str) -> tuple[str, Any, Any] | None:
    """Compare array shape, dtype, and exact values.

    Returns:
        The first mismatch tuple, or ``None`` when arrays are equal.
    """
    if not isinstance(expected, np.ndarray) or not isinstance(actual, np.ndarray):
        return path, expected, actual
    if (
        expected.shape != actual.shape
        or expected.dtype != actual.dtype
        or not np.array_equal(expected, actual)
    ):
        return path, expected, actual
    return None


def _first_mapping_difference(
    expected: Mapping[Any, Any], actual: Mapping[Any, Any], path: str
) -> tuple[str, Any, Any] | None:
    """Compare mapping keys and recursively compare values in stable order.

    Returns:
        The first mismatch tuple, or ``None`` when mappings are equal.
    """
    for key in sorted(set(expected) | set(actual), key=str):
        if key not in expected or key not in actual:
            return f"{path}.{key}", expected.get(key), actual.get(key)
        mismatch = _first_difference(expected[key], actual[key], f"{path}.{key}")
        if mismatch is not None:
            return mismatch
    return None


def _first_sequence_difference(
    expected: Sequence[Any], actual: Sequence[Any], path: str
) -> tuple[str, Any, Any] | None:
    """Compare sequence length and recursively compare members.

    Returns:
        The first mismatch tuple, or ``None`` when sequences are equal.
    """
    if len(expected) != len(actual):
        return path, expected, actual
    for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
        mismatch = _first_difference(left, right, f"{path}[{index}]")
        if mismatch is not None:
            return mismatch
    return None


def _first_difference(expected: Any, actual: Any, path: str) -> tuple[str, Any, Any] | None:
    """Return the first deterministic nested mismatch."""
    if isinstance(expected, np.ndarray) or isinstance(actual, np.ndarray):
        return _first_array_difference(expected, actual, path)
    if isinstance(expected, Mapping) or isinstance(actual, Mapping):
        if not isinstance(expected, Mapping) or not isinstance(actual, Mapping):
            return path, expected, actual
        return _first_mapping_difference(expected, actual, path)
    if isinstance(expected, (tuple, list)) or isinstance(actual, (tuple, list)):
        if not isinstance(expected, (tuple, list)) or not isinstance(actual, (tuple, list)):
            return path, expected, actual
        return _first_sequence_difference(expected, actual, path)
    if expected != actual:
        return path, expected, actual
    return None


def compare_continuation_traces(
    expected: Sequence[NoOpStep | Mapping[str, Any]],
    actual: Sequence[NoOpStep | Mapping[str, Any]],
) -> NoOpComparison:
    """Compare every declared state, observation, action, event, clock, metric, and RNG field.

    Equal collision time or equal terminal status is insufficient: the first
    differing field and timestep are returned for a durable negative-control
    receipt.

    Returns:
        A comparison result with the first divergence, if any.
    """
    compared = min(len(expected), len(actual))
    for index in range(compared):
        left = (
            expected[index].to_dict()
            if isinstance(expected[index], NoOpStep)
            else dict(expected[index])
        )
        right = (
            actual[index].to_dict() if isinstance(actual[index], NoOpStep) else dict(actual[index])
        )
        mismatch = _first_difference(left, right, "step")
        if mismatch is not None:
            field_name, left_value, right_value = mismatch
            step = left.get("step", right.get("step", index))
            return NoOpComparison(False, index, int(step), field_name, left_value, right_value)
    if len(expected) != len(actual):
        step = compared
        return NoOpComparison(False, compared, step, "trace_length", len(expected), len(actual))
    return NoOpComparison(True, compared)


@dataclass(frozen=True, slots=True)
class StateInventoryEntry:
    """One field in the augmented continuation-state inventory."""

    path: str
    owner: str
    type_name: str
    unit: str
    update_phase: str
    classification: str
    serialization: str
    restore_method: str
    test: str
    status: str
    unsupported_modes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a machine-readable inventory row."""
        return {
            "path": self.path,
            "owner": self.owner,
            "type": self.type_name,
            "unit": self.unit,
            "update_phase": self.update_phase,
            "classification": self.classification,
            "serialization": self.serialization,
            "restore_method": self.restore_method,
            "test": self.test,
            "status": self.status,
            "unsupported_modes": list(self.unsupported_modes),
        }


STATE_INVENTORY: tuple[StateInventoryEntry, ...] = (
    StateInventoryEntry(
        "actors.order",
        "Simulator",
        "stable row order",
        "index",
        "all",
        "dynamic",
        "JSON actor count/order",
        "reject count drift",
        "test_snapshot_restore_reproduces_baseline_deterministically",
        "supported",
    ),
    StateInventoryEntry(
        "pysf_state",
        "Simulator.pysf_state",
        "float[N,6]",
        "m/seconds",
        "pedestrian step",
        "dynamic",
        "NPZ numeric array",
        "SimulatorCounterfactualModel.restore",
        "test_snapshot_restore_reproduces_baseline_deterministically",
        "supported",
    ),
    StateInventoryEntry(
        "ped_max_speeds",
        "PySocialForce.peds",
        "float[N]",
        "m/s",
        "force update",
        "dynamic",
        "NPZ numeric array",
        "adapter restore",
        "test_snapshot_restore_reproduces_baseline_deterministically",
        "supported",
    ),
    StateInventoryEntry(
        "pedestrian_groups",
        "Simulator.groups",
        "mapping[int, set[int]]",
        "pedestrian IDs",
        "behavior/force update",
        "dynamic",
        "JSON integer-entry lists",
        "adapter restore plus backend synchronization",
        "test_typed_snapshot_round_trip_preserves_exact_continuation",
        "supported",
    ),
    StateInventoryEntry(
        "pedestrian_group_by_ped",
        "Simulator.groups",
        "mapping[int, int]",
        "group IDs",
        "behavior/force update",
        "dynamic",
        "JSON integer-entry lists",
        "adapter restore plus backend synchronization",
        "test_typed_snapshot_round_trip_preserves_exact_continuation",
        "supported",
    ),
    StateInventoryEntry(
        "robot.state",
        "robot drive model",
        "typed dataclass",
        "native SI",
        "robot step",
        "dynamic",
        "JSON typed fields",
        "destination-owned dataclass",
        "test_adapter_satisfies_counterfactual_model_protocol",
        "supported",
    ),
    StateInventoryEntry(
        "robot_navs",
        "Simulator.robot_navs",
        "RouteNavigator",
        "m",
        "route update",
        "dynamic",
        "JSON stable fields",
        "destination-owned navigator",
        "test_route_group_navigator_progress_restores",
        "supported",
    ),
    StateInventoryEntry(
        "peds_behaviors.single_runtimes",
        "SinglePedestrianBehavior",
        "typed runtime list",
        "seconds/index",
        "behavior step",
        "dynamic",
        "JSON stable actor IDs",
        "adapter restore",
        "test_route_group_navigator_progress_restores",
        "supported",
    ),
    StateInventoryEntry(
        "peds_behaviors.route_navigators",
        "FollowRouteBehavior",
        "mapping[str, state]",
        "index",
        "behavior step",
        "dynamic",
        "JSON stable behavior IDs",
        "adapter restore",
        "test_route_group_navigator_progress_restores",
        "supported",
    ),
    StateInventoryEntry(
        "numpy.random",
        "process RNG",
        "legacy RNG tuple",
        "stream",
        "sampling",
        "dynamic",
        "NPZ state + JSON scalars",
        "numpy.random.set_state",
        "test_rng_capture_seam_prevents_divergence",
        "supported",
    ),
    StateInventoryEntry(
        "random",
        "stdlib RNG",
        "tuple",
        "stream",
        "route sampling",
        "dynamic",
        "JSON tuple encoding",
        "random.setstate",
        "test_rng_capture_seam_prevents_divergence",
        "supported",
    ),
    StateInventoryEntry(
        "behavior.rng",
        "behavior owner",
        "Generator state",
        "stream",
        "behavior step",
        "dynamic",
        "JSON/NPZ typed state",
        "generator.bit_generator.state",
        "owned-generator probe",
        "conditional",
        ("unknown behavior generator implementation",),
    ),
    StateInventoryEntry(
        "residual_adversary",
        "Simulator._residual_adversary",
        "typed mutable fields",
        "native SI",
        "force update",
        "dynamic",
        "JSON/NPZ typed state",
        "adapter restore or reject",
        "residual-adversary state probe",
        "conditional",
        ("inactive/uninstantiated adversary",),
    ),
    StateInventoryEntry(
        "map/config/model",
        "immutable inputs",
        "digest references",
        "n/a",
        "construction",
        "immutable",
        "JSON SHA-256",
        "compatibility check before mutation",
        "test_incompatible_digest_rejected",
        "supported",
    ),
    StateInventoryEntry(
        "controller.planner_memory",
        "planner/controller owner",
        "implementation-specific",
        "native",
        "decision",
        "dynamic",
        "not serialized by native adapter",
        "reject as full restart",
        "test_recurrent_state_is_not_claimed",
        "unsupported",
        ("recurrent planners", "hidden controller state"),
    ),
    StateInventoryEntry(
        "observation.sensor_history",
        "Gym/SensorFusion",
        "history buffers",
        "native",
        "observation",
        "dynamic",
        "not serialized by native adapter",
        "reject as full restart",
        "test_history_omission_not_evaluable",
        "unsupported",
        ("stacked observations", "sensor caches"),
    ),
    StateInventoryEntry(
        "clock.termination_budget",
        "RobotState",
        "step/time/budget",
        "steps/seconds",
        "termination",
        "dynamic",
        "JSON boundary",
        "caller must preserve",
        "test_boundary_preserves_remaining_budget",
        "boundary_only",
    ),
    StateInventoryEntry(
        "metrics.events",
        "Gym/benchmark metrics",
        "accumulators",
        "metric units",
        "post-step",
        "output-only",
        "not serialized by native adapter",
        "recompute only with owner",
        "test_metric_omission_not_evaluable",
        "unsupported",
        ("full-episode metric recomposition",),
    ),
)


def state_inventory_payload() -> dict[str, Any]:
    """Return the complete machine-readable inventory payload."""
    return {
        "schema_version": "simulator_state_inventory.v1",
        "snapshot_schema": SNAPSHOT_SCHEMA,
        "platform": sys.platform,
        "entries": [entry.to_dict() for entry in STATE_INVENTORY],
    }


def write_state_inventory(path: str | Path) -> Path:
    """Write the inventory to a deterministic UTF-8 JSON file.

    Returns:
        The destination path.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(state_inventory_payload(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return target


__all__ = [
    "SNAPSHOT_BOUNDARY",
    "SNAPSHOT_SCHEMA",
    "STATE_INVENTORY",
    "NoOpComparison",
    "NoOpStep",
    "SnapshotArtifact",
    "SnapshotBoundary",
    "SnapshotCompatibility",
    "SnapshotCompatibilityError",
    "SnapshotContractError",
    "SnapshotPayloadError",
    "StateInventoryEntry",
    "TypedSimulatorSnapshot",
    "capture_typed_snapshot",
    "compare_continuation_traces",
    "read_typed_snapshot",
    "restore_typed_snapshot",
    "state_inventory_payload",
    "write_state_inventory",
    "write_typed_snapshot",
]
